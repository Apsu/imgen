import argparse
import asyncio
import base64
import io
import logging
import os
import pickle
import queue
import random
import signal
import sys
import time
import traceback
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib

import torch
import torch.multiprocessing as mp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torch import Generator
from diffusers import AutoPipelineForText2Image
from diffusers.utils import logging as diffusers_logging

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up Triton cache directory for persistent kernel compilation
TRITON_CACHE_DIR = os.environ.get("TRITON_CACHE_DIR", "/home/ubuntu/imgen/.triton_cache")
os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE_DIR
os.makedirs(TRITON_CACHE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set library log levels
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Disable diffusers progress bars
diffusers_logging.disable_progress_bar()

# Set up our logger
logger = logging.getLogger(__name__)

# Global configuration and state variables
server_config: Optional[argparse.Namespace]

# Global pipeline storage for worker processes
pipelines = {}

# Global shutdown event
shutdown_event = mp.Event()

# Usage tracking for optimization
dimension_usage_counter = Counter()
dimension_usage_lock = threading.Lock()


@dataclass
class GenerationConfig:
    """Configuration for a generation request."""
    width: int
    height: int
    steps: int
    
    def __hash__(self):
        return hash((self.width, self.height, self.steps))
    
    def cache_key(self):
        """Generate a cache key for this configuration."""
        return f"{self.width}x{self.height}_{self.steps}steps"


class PersistentCUDAGraphCache:
    """Persistent CUDA graph cache with disk storage."""
    
    def __init__(self, cache_dir: str, max_memory_items: int = 100):
        self.cache_dir = Path(cache_dir) / "cuda_graphs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: OrderedDict[str, Any] = OrderedDict()
        self.max_memory_items = max_memory_items
        self.stats = {"hits": 0, "misses": 0, "disk_loads": 0}
        
        # Load index of available graphs
        self.index_file = self.cache_dir / "index.pkl"
        self.disk_index = self._load_index()
        
    def _load_index(self) -> set:
        """Load the index of cached graphs from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return set()
    
    def _save_index(self):
        """Save the index of cached graphs to disk."""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.disk_index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def get(self, config: GenerationConfig) -> Optional[Any]:
        """Get a cached graph if available."""
        cache_key = config.cache_key()
        
        # Check memory cache
        if cache_key in self.memory_cache:
            self.memory_cache.move_to_end(cache_key)
            self.stats["hits"] += 1
            return self.memory_cache[cache_key]
        
        # Check disk cache
        if cache_key in self.disk_index:
            cache_file = self.cache_dir / f"{cache_key}.pt"
            if cache_file.exists():
                try:
                    graph_data = torch.load(cache_file, map_location='cuda')
                    self._add_to_memory_cache(cache_key, graph_data)
                    self.stats["disk_loads"] += 1
                    self.stats["hits"] += 1
                    return graph_data
                except Exception as e:
                    logger.warning(f"Failed to load cached graph {cache_key}: {e}")
                    self.disk_index.discard(cache_key)
        
        self.stats["misses"] += 1
        return None
    
    def put(self, config: GenerationConfig, graph_data: Any):
        """Store a graph in the cache."""
        cache_key = config.cache_key()
        
        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.pt"
        try:
            torch.save(graph_data, cache_file)
            self.disk_index.add(cache_key)
            self._save_index()
        except Exception as e:
            logger.error(f"Failed to save graph {cache_key}: {e}")
            return
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, graph_data)
    
    def _add_to_memory_cache(self, cache_key: str, graph_data: Any):
        """Add item to memory cache with LRU eviction."""
        self.memory_cache[cache_key] = graph_data
        self.memory_cache.move_to_end(cache_key)
        
        # Evict if necessary
        while len(self.memory_cache) > self.max_memory_items:
            self.memory_cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "memory_items": len(self.memory_cache),
            "disk_items": len(self.disk_index),
            "hit_rate": f"{hit_rate:.2%}"
        }


# Request model definition
class TextToImageInput(BaseModel):
    prompt: str
    width: int = 512
    height: int = 384
    steps: int = 4
    guidance: float = 0.0
    seed: int | None = None


def init_worker(model_path: str, gpu_assignments: Dict[int, int], ready_queue: mp.Queue, error_queue: mp.Queue):
    """Initialize a worker process with its GPU and model."""
    global pipelines
    
    # Ignore SIGINT in worker processes - let the main process handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    worker_id = None
    rank = None
    
    try:
        # Get the worker ID and corresponding GPU
        worker_id = mp.current_process()._identity[0] - 1
        rank = gpu_assignments[worker_id]
        
        # Set the GPU for this process
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        # Initialize random seed for this worker using worker_id and current time
        # This ensures each worker has a different random sequence
        random.seed(worker_id + int(time.time() * 1000000))

        logger.info(f"Worker {rank} starting on GPU {rank}")

        # Set float32 matmul precision to high for better performance on Hopper+
        torch.set_float32_matmul_precision("high")
        
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking for consistent workloads
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction to avoid OOM with multiple workers
        torch.cuda.set_per_process_memory_fraction(0.95 / gpu_assignments.__len__(), device=rank)

        # Load model pipeline
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(device)

        # Optimize model components
        pipeline.transformer.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)

        # Fuse QKV projections for better performance
        pipeline.transformer.fuse_qkv_projections()
        pipeline.vae.fuse_qkv_projections()

        # Disable progress bars for this pipeline
        pipeline.set_progress_bar_config(disable=True)

        generator = Generator(device)
        
        # Create CUDA graph cache for this worker
        cuda_cache_dir = os.path.join(TRITON_CACHE_DIR, f"worker_{worker_id}")
        cuda_graph_cache = PersistentCUDAGraphCache(cuda_cache_dir, max_memory_items=50)
        
        # Compile key components with torch.compile
        logger.info(f"Worker {rank} compiling model components...")
        
        # Text encoder usually has static shapes, compile with fullgraph
        pipeline.text_encoder.forward = torch.compile(
            pipeline.text_encoder.forward,
            mode="reduce-overhead",
            fullgraph=True
        )
        
        # VAE decoder benefits from compilation
        pipeline.vae.decode = torch.compile(
            pipeline.vae.decode,
            mode="reduce-overhead",
            dynamic=False
        )

        logger.info(f"Worker {rank} loaded model successfully")

        # Warmup with common configurations
        warmup_configs = [
            GenerationConfig(512, 512, 4),
            GenerationConfig(1024, 768, 4),
            GenerationConfig(1024, 1024, 4),
        ]
        
        logger.info(f"Worker {rank} warming up...")
        for config in warmup_configs:
            warmup_args = {
                "prompt": "warmup",
                "width": config.width,
                "height": config.height,
                "num_inference_steps": config.steps,
                "guidance_scale": 0.0,
            }
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _ = pipeline(**warmup_args, output_type="pil")
        
        logger.info(f"Worker {rank} warmup complete")
        
        # Store pipeline and components for this worker
        pipelines[worker_id] = {
            'pipeline': pipeline,
            'generator': generator,
            'device': device,
            'rank': rank,
            'cuda_graph_cache': cuda_graph_cache,
        }
        
        # Signal that this worker is ready
        ready_queue.put((worker_id, rank))
        
    except Exception as e:
        logger.error(f"Worker {rank if rank is not None else 'unknown'} failed to initialize: {str(e)}")
        logger.error(traceback.format_exc())
        error_queue.put((worker_id, str(e), traceback.format_exc()))
        raise


def capture_cuda_graph(pipeline, config: GenerationConfig, generator):
    """Capture a CUDA graph for the given configuration."""
    logger.info(f"Capturing CUDA graph for {config.cache_key()}")
    
    # Prepare static inputs
    static_input = {
        "prompt": "static prompt for graph capture",
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.steps,
        "guidance_scale": 0.0,
        "generator": generator,
        "output_type": "pil"
    }
    
    # Warm up
    for _ in range(3):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = pipeline(**static_input)
    
    # Capture the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = pipeline(**static_input)
    
    return {
        'graph': graph,
        'output': output,
        'config': config
    }


def worker_generate_image(args):
    """Generate an image using the initialized pipeline."""
    # Check if we should exit early due to shutdown
    if shutdown_event.is_set():
        return None, 0
        
    gen_args, seed, enable_cuda_graphs = args
    
    # Get worker ID and its components
    worker_id = mp.current_process()._identity[0] - 1
    worker_data = pipelines[worker_id]
    pipeline = worker_data['pipeline']
    generator = worker_data['generator']
    device = worker_data['device']
    rank = worker_data['rank']
    cuda_graph_cache = worker_data['cuda_graph_cache']
    
    # Track dimension usage
    config = GenerationConfig(
        width=gen_args['width'],
        height=gen_args['height'],
        steps=gen_args['num_inference_steps']
    )
    
    logger.info(
        f"Worker {rank} processing: "
        f"prompt='{gen_args['prompt'][:50]}...' "
        f"size={gen_args['width']}x{gen_args['height']} "
        f"steps={gen_args['num_inference_steps']} guidance={gen_args['guidance_scale']} seed={seed}"
    )
    
    if seed is not None:
        generator = generator.manual_seed(seed)
    else:
        # Generate a random seed for this request
        random_seed = random.randint(0, 2**32 - 1)
        generator = generator.manual_seed(random_seed)
    
    start_time = time.time()
    
    # Check if we can use a cached CUDA graph
    use_graph = False
    if enable_cuda_graphs and gen_args['guidance_scale'] == 0.0:  # Graphs work best with CFG disabled
        cached_graph = cuda_graph_cache.get(config)
        if cached_graph:
            use_graph = True
            logger.info(f"Worker {rank} using cached CUDA graph for {config.cache_key()}")
    
    # Generate the image
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Clear cache periodically to avoid memory fragmentation
        if random.random() < 0.05:  # 5% chance
            torch.cuda.empty_cache()
        
        if use_graph:
            # TODO: Implement CUDA graph replay with dynamic prompt
            # For now, fall back to regular generation
            output = pipeline(**gen_args, generator=generator, output_type="pil")
        else:
            # Regular generation
            output = pipeline(**gen_args, generator=generator, output_type="pil")
            
            # Opportunistically cache if this is a common configuration
            if enable_cuda_graphs and gen_args['guidance_scale'] == 0.0:
                with dimension_usage_lock:
                    dimension_usage_counter[config] += 1
                    count = dimension_usage_counter[config]
                
                # Cache after seeing the configuration multiple times
                if count == 5:  # Threshold for caching
                    try:
                        graph_data = capture_cuda_graph(pipeline, config, generator)
                        cuda_graph_cache.put(config, graph_data)
                        logger.info(f"Worker {rank} cached CUDA graph for {config.cache_key()}")
                    except Exception as e:
                        logger.warning(f"Failed to capture CUDA graph: {e}")
    
    elapsed = time.time() - start_time
    
    # Convert the generated image to a base64-encoded PNG
    buffered = io.BytesIO()
    output.images[0].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Log cache stats periodically
    if random.random() < 0.1:  # 10% chance
        stats = cuda_graph_cache.get_stats()
        logger.info(f"Worker {rank} cache stats: {stats}")
    
    logger.info(f"Worker {rank} completed in {elapsed:.2f}s")
    
    return img_str, elapsed


class MultiGPUGenerator:
    """Manages multiple GPU workers for image generation."""

    def __init__(self, model_path: str, num_gpus: int = 8, enable_cuda_graphs: bool = True):
        self.model_path = model_path
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.shutdown_requested = False
        self.enable_cuda_graphs = enable_cuda_graphs
        
        logger.info(f"Starting {self.num_gpus} GPU workers (CUDA graphs={'enabled' if enable_cuda_graphs else 'disabled'})")
        
        # Create GPU assignments for workers
        gpu_assignments = {i: i for i in range(self.num_gpus)}
        
        # Create queues for worker synchronization
        self.ready_queue = mp.Queue()
        self.error_queue = mp.Queue()
        
        # Create pool with initializer
        self.pool = mp.Pool(
            processes=self.num_gpus,
            initializer=init_worker,
            initargs=(model_path, gpu_assignments, self.ready_queue, self.error_queue)
        )
        
        # Wait for all workers to be ready
        ready_workers = []
        errors = []
        
        timeout = 120  # 120 seconds timeout for initialization (compilation takes time)
        start_time = time.time()
        
        while len(ready_workers) < self.num_gpus:
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Timeout waiting for workers to initialize. Only {len(ready_workers)}/{self.num_gpus} ready.")
            
            # Check for errors
            try:
                worker_id, error_msg, traceback_str = self.error_queue.get_nowait()
                errors.append(f"Worker {worker_id}: {error_msg}")
                logger.error(f"Worker {worker_id} initialization failed:\n{traceback_str}")
            except queue.Empty:
                pass
            
            # Check for ready workers
            try:
                worker_id, rank = self.ready_queue.get(timeout=0.1)
                ready_workers.append((worker_id, rank))
                logger.info(f"Worker {rank} ready ({len(ready_workers)}/{self.num_gpus})")
            except queue.Empty:
                pass
            
            # If we have errors, fail fast
            if errors:
                self.shutdown()
                raise RuntimeError(f"Worker initialization failed:\n" + "\n".join(errors))
        
        logger.info("All workers initialized and ready")
        
        # Start background thread to analyze usage patterns
        self.usage_analyzer_thread = threading.Thread(target=self._analyze_usage_patterns, daemon=True)
        self.usage_analyzer_thread.start()

    def _analyze_usage_patterns(self):
        """Background thread to analyze dimension usage patterns."""
        while not self.shutdown_requested:
            time.sleep(300)  # Analyze every 5 minutes
            
            with dimension_usage_lock:
                if dimension_usage_counter:
                    top_configs = dimension_usage_counter.most_common(10)
                    logger.info("Top dimension configurations:")
                    for config, count in top_configs:
                        logger.info(f"  {config.cache_key()}: {count} requests")

    async def generate(
        self, gen_args: Dict[str, Any], seed: int | None
    ) -> Tuple[str, float]:
        """Submit a generation job and wait for the result."""
        if self.shutdown_requested:
            raise RuntimeError("Generator is shutting down")
        
        # Run generation in pool - Pool will automatically assign to next available worker
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.pool.apply,
            worker_generate_image,
            ((gen_args, seed, self.enable_cuda_graphs),)
        )
        
        if result[0] is None:
            raise RuntimeError("Worker shutdown during generation")
            
        return result

    def shutdown(self):
        """Shutdown all worker processes."""
        if self.shutdown_requested:
            return
            
        self.shutdown_requested = True
        logger.info("Shutting down GPU workers")
        
        # Signal all workers to stop
        shutdown_event.set()
        
        if hasattr(self, 'pool'):
            # Terminate pool gracefully
            self.pool.close()
            # Give workers a chance to finish current work
            self.pool.join()
            
        logger.info("GPU workers shutdown complete")


shared_generator: Optional[MultiGPUGenerator] = None
app: Optional[FastAPI] = None
shutdown_in_progress = False
shutdown_lock = threading.Lock()


def handle_shutdown():
    """Common shutdown handler for both signals and lifespan."""
    global shutdown_in_progress, shared_generator
    
    with shutdown_lock:
        if shutdown_in_progress:
            return
        shutdown_in_progress = True
    
    logger.info("Initiating graceful shutdown...")
    
    # Set the shutdown event
    shutdown_event.set()
    
    # Shutdown the generator if it exists
    if shared_generator:
        shared_generator.shutdown()


def get_available_gpus() -> int:
    """Get the number of available GPUs based on CUDA_VISIBLE_DEVICES or total count."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cuda_visible is not None and cuda_visible.strip():
        # Count the number of comma-separated GPU IDs
        gpu_ids = [gpu.strip() for gpu in cuda_visible.split(",") if gpu.strip()]
        return len(gpu_ids)
    else:
        # Return total GPU count if CUDA_VISIBLE_DEVICES not set
        return torch.cuda.device_count()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FastAPI server for diffusion image generation with multi-GPU support and advanced optimizations."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-schnell"),
        help="Model path or identifier",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Hostname or IP to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (defaults to CUDA_VISIBLE_DEVICES count or all available)",
    )
    parser.add_argument(
        "--no-cuda-graphs",
        action="store_true",
        help="Disable CUDA graph optimization",
    )

    args = parser.parse_args()

    # Set default num_gpus based on available GPUs if not specified
    if args.num_gpus is None:
        args.num_gpus = get_available_gpus()

    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown."""
    global shared_generator

    try:
        # Initialize the multi-GPU generator
        logger.info(f"Initializing with {server_config.num_gpus} GPUs")
        logger.info(f"Triton cache directory: {TRITON_CACHE_DIR}")
        if cuda_visible := os.environ.get("CUDA_VISIBLE_DEVICES"):
            logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

        shared_generator = MultiGPUGenerator(
            model_path=server_config.model, 
            num_gpus=server_config.num_gpus,
            enable_cuda_graphs=not server_config.no_cuda_graphs
        )
        yield
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise
    finally:
        # Cleanup resources
        handle_shutdown()
        logger.info("Server shutdown complete")


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/stats", response_class=JSONResponse)
async def get_stats():
    """Get server statistics including cache performance."""
    if not shared_generator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    with dimension_usage_lock:
        top_dims = dimension_usage_counter.most_common(5)
    
    return JSONResponse({
        "top_dimensions": [
            {
                "config": f"{c.width}x{c.height}@{c.steps}",
                "count": count
            } for c, count in top_dims
        ],
        "triton_cache_dir": TRITON_CACHE_DIR
    })


@app.post("/v1/images/generations", response_class=JSONResponse)
async def generate_image(request: Request, image_input: TextToImageInput):
    """Generate an image based on the provided text input."""
    try:
        if not shared_generator:
            raise HTTPException(status_code=503, detail="Service not ready")
            
        # Prepare arguments for the pipeline
        gen_args = {
            "prompt": image_input.prompt,
            "width": image_input.width,
            "height": image_input.height,
            "num_inference_steps": image_input.steps,
            "guidance_scale": image_input.guidance,
        }

        # Generate image using the worker pool
        img_str, gen_time = await shared_generator.generate(gen_args, image_input.seed)

        return JSONResponse({"image": img_str, "gen_time": gen_time})
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Parse command-line arguments and store in global server_config
    server_config = parse_args()

    # Set start method for multiprocessing
    mp.set_start_method("spawn", force=True)

    try:
        # Create a custom uvicorn server with proper signal handling
        config = uvicorn.Config(
            app,
            host=server_config.host,
            port=server_config.port,
            log_level="info",
            access_log=False,
        )
        server = uvicorn.Server(config)
        
        # Override uvicorn's signal handlers to ensure our cleanup runs
        original_signal_handler = server.handle_exit
        
        def custom_signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            handle_shutdown()
            original_signal_handler(sig, frame)
        
        server.handle_exit = custom_signal_handler
        
        # Run the server
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        # Ensure cleanup happens
        handle_shutdown()