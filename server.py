import argparse
import asyncio
import base64
import io
import logging
import os
import queue
import random
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Tuple
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

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

        logger.info(f"Worker {rank} loaded model successfully")

        # Warmup
        warmup_args = {
            "prompt": "warmup",
            "width": 1024,
            "height": 768,
            "num_inference_steps": 1,
            "guidance_scale": 0.0,
        }
        _ = pipeline(**warmup_args, output_type="pil")
        logger.info(f"Worker {rank} warmup complete")
        
        # Store pipeline and generator for this worker
        pipelines[worker_id] = (pipeline, generator, device, rank)
        
        # Signal that this worker is ready
        ready_queue.put((worker_id, rank))
        
    except Exception as e:
        logger.error(f"Worker {rank if rank is not None else 'unknown'} failed to initialize: {str(e)}")
        logger.error(traceback.format_exc())
        error_queue.put((worker_id, str(e), traceback.format_exc()))
        raise


def worker_generate_image(args):
    """Generate an image using the initialized pipeline."""
    # Check if we should exit early due to shutdown
    if shutdown_event.is_set():
        return None, 0
        
    gen_args, seed = args
    
    # Get worker ID and its pipeline
    worker_id = mp.current_process()._identity[0] - 1
    pipeline, generator, device, rank = pipelines[worker_id]
    
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
        # Using random.randint which was seeded differently per worker
        random_seed = random.randint(0, 2**32 - 1)
        generator = generator.manual_seed(random_seed)
    
    start_time = time.time()
    
    # Perform the actual image generation
    output = pipeline(**gen_args, generator=generator, output_type="pil")
    elapsed = time.time() - start_time
    
    # Convert the generated image to a base64-encoded PNG
    buffered = io.BytesIO()
    output.images[0].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    logger.info(f"Worker {rank} completed in {elapsed:.2f}s")
    
    return img_str, elapsed


class MultiGPUGenerator:
    """Manages multiple GPU workers for image generation."""

    def __init__(self, model_path: str, num_gpus: int = 8):
        self.model_path = model_path
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.shutdown_requested = False
        
        logger.info(f"Starting {self.num_gpus} GPU workers")
        
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
        
        timeout = 60  # 60 seconds timeout for initialization
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
            ((gen_args, seed),)
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
            # Note: multiprocessing.Pool.join() doesn't accept timeout parameter
            
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
        description="FastAPI server for diffusion image generation with multi-GPU support."
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
        if cuda_visible := os.environ.get("CUDA_VISIBLE_DEVICES"):
            logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

        shared_generator = MultiGPUGenerator(
            model_path=server_config.model, num_gpus=server_config.num_gpus
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