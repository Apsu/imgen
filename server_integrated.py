import argparse
import asyncio
import base64
import io
import json
import logging
import os
import queue
import random
import signal
import sys
import time
import traceback
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import torch.multiprocessing as mp
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
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

# WebSocket connections
websocket_connections: List[WebSocket] = []
websocket_lock = threading.Lock()

# Generation queue and stats
generation_queue = deque(maxlen=100)
generation_stats = {
    "total_requests": 0,
    "total_completed": 0,
    "average_time": 0.0,
    "dimension_counts": {},
}
stats_lock = threading.Lock()

# Allowed dimensions - must be divisible by 16 for VAE
ALLOWED_DIMENSIONS = [
    (256, 256),   # Tiny
    (512, 384),   # Small landscape  
    (384, 512),   # Small portrait
    (512, 512),   # Square
    (768, 512),   # Wide
    (512, 768),   # Tall
    (768, 768),   # Large square
    (1024, 768),  # Landscape
    (768, 1024),  # Portrait
    (1024, 1024), # Large square
    (1280, 768),  # Wide HD (adjusted from 720)
    (1920, 1088), # Full HD (adjusted from 1080)
    (1536, 1024), # Wide
    (1024, 1536), # Tall
    (2048, 1152), # Ultra wide
    (1152, 2048), # Ultra tall
]

# Max total pixels - we have 80GB VRAM per GPU!
MAX_TOTAL_PIXELS = 8388608  # 4096x2048

# Common steps
ALLOWED_STEPS = [1, 4, 8, 12, 16, 20, 30, 50]


# Request model definition with validation
class TextToImageInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    width: int = Field(512, ge=256, le=2048)
    height: int = Field(512, ge=256, le=2048)
    steps: int = Field(4, ge=1, le=50)
    guidance: float = Field(0.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(None, ge=-2147483648, le=2147483647)
    
    @field_validator('width', 'height')
    def dimensions_divisible_by_8(cls, v):
        # Actually needs to be divisible by 16 for some models
        if v % 16 != 0:
            raise ValueError('Dimensions must be divisible by 16')
        return v
    
    @field_validator('height')
    def total_pixels_limit(cls, v, info):
        if 'width' in info.data:
            total_pixels = info.data['width'] * v
            if total_pixels > MAX_TOTAL_PIXELS:
                raise ValueError(f'Total pixels ({total_pixels}) exceeds limit ({MAX_TOTAL_PIXELS})')
        return v


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
        return None, 0, None
        
    gen_args, seed, request_id = args
    
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
        seed = random_seed  # Return the seed that was used
    
    start_time = time.time()
    
    # Perform the actual image generation
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = pipeline(**gen_args, generator=generator, output_type="pil")
    
    elapsed = time.time() - start_time
    
    # Convert the generated image to a base64-encoded PNG
    buffered = io.BytesIO()
    output.images[0].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    logger.info(f"Worker {rank} completed in {elapsed:.2f}s")
    
    return img_str, elapsed, seed


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
        self, gen_args: Dict[str, Any], seed: int | None, request_id: str
    ) -> Tuple[str, float, int]:
        """Submit a generation job and wait for the result."""
        if self.shutdown_requested:
            raise RuntimeError("Generator is shutting down")
        
        # Run generation in pool - Pool will automatically assign to next available worker
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.pool.apply,
            worker_generate_image,
            ((gen_args, seed, request_id),)
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
        description="FastAPI server for diffusion image generation with integrated web interface."
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
app = FastAPI(lifespan=lifespan, title="Image Generation Server")

# Set up static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

# Create directories if they don't exist
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# Mount static files
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Set up templates
templates = Jinja2Templates(directory=str(templates_dir))


async def broadcast_status(message: dict):
    """Broadcast status updates to all connected WebSocket clients."""
    with websocket_lock:
        disconnected = []
        for websocket in websocket_connections:
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            websocket_connections.remove(ws)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "allowed_dimensions": ALLOWED_DIMENSIONS,
        "allowed_steps": ALLOWED_STEPS,
        "max_pixels": MAX_TOTAL_PIXELS,
    })


@app.get("/api/stats")
async def get_stats():
    """Get server statistics."""
    with stats_lock:
        stats = generation_stats.copy()
    
    return JSONResponse({
        "stats": stats,
        "queue_length": len(generation_queue),
        "num_gpus": server_config.num_gpus if server_config else 0,
    })


@app.get("/api/dimensions")
async def get_dimensions():
    """Get allowed dimensions."""
    return JSONResponse({
        "dimensions": ALLOWED_DIMENSIONS,
        "max_pixels": MAX_TOTAL_PIXELS,
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    with websocket_lock:
        websocket_connections.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "queue_length": len(generation_queue),
        })
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        with websocket_lock:
            if websocket in websocket_connections:
                websocket_connections.remove(websocket)


@app.post("/v1/images/generations", response_class=JSONResponse)
async def generate_image(request: Request, image_input: TextToImageInput):
    """Generate an image based on the provided text input."""
    try:
        if not shared_generator:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Add to queue for tracking
        generation_queue.append({
            "id": request_id,
            "timestamp": datetime.now(),
            "status": "processing",
        })
        
        # Broadcast queue update
        await broadcast_status({
            "type": "queue_update",
            "queue_length": len(generation_queue),
            "request_id": request_id,
        })
        
        # Prepare arguments for the pipeline
        gen_args = {
            "prompt": image_input.prompt,
            "width": image_input.width,
            "height": image_input.height,
            "num_inference_steps": image_input.steps,
            "guidance_scale": image_input.guidance,
        }

        # Generate image using the worker pool
        img_str, gen_time, used_seed = await shared_generator.generate(gen_args, image_input.seed, request_id)
        
        # Update statistics
        with stats_lock:
            generation_stats["total_requests"] += 1
            generation_stats["total_completed"] += 1
            total = generation_stats["total_completed"]
            generation_stats["average_time"] = (
                (generation_stats["average_time"] * (total - 1) + gen_time) / total
            )
            
            # Track dimension usage
            dim_key = f"{image_input.width}x{image_input.height}"
            if dim_key not in generation_stats["dimension_counts"]:
                generation_stats["dimension_counts"][dim_key] = 0
            generation_stats["dimension_counts"][dim_key] += 1
        
        # Broadcast completion
        await broadcast_status({
            "type": "generation_complete",
            "request_id": request_id,
            "gen_time": gen_time,
        })

        return JSONResponse({
            "image": img_str,
            "gen_time": gen_time,
            "seed": used_seed,
            "width": image_input.width,
            "height": image_input.height,
        })
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Broadcast error
        await broadcast_status({
            "type": "generation_error",
            "request_id": request_id if 'request_id' in locals() else None,
            "error": str(e),
        })
        
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