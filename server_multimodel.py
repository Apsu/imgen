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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
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
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline, StableDiffusionPipeline
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

# Model configurations
@dataclass
class ModelConfig:
    name: str
    model_id: str
    pipeline_class: Any
    default_steps: int
    default_guidance: float
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    vram_gb: float  # Estimated VRAM usage
    description: str
    
    def supports_dimensions(self, width: int, height: int) -> bool:
        return (self.min_width <= width <= self.max_width and 
                self.min_height <= height <= self.max_height)


# Available models - configure based on your preferences
AVAILABLE_MODELS = {
    "flux-schnell": ModelConfig(
        name="FLUX.1 Schnell",
        model_id="black-forest-labs/FLUX.1-schnell",
        pipeline_class=AutoPipelineForText2Image,
        default_steps=4,
        default_guidance=0.0,
        min_width=256,
        max_width=2048,
        min_height=256,
        max_height=2048,
        vram_gb=12.0,
        description="Fast high-quality generation, no CFG needed"
    ),
    "flux-dev": ModelConfig(
        name="FLUX.1 Dev",
        model_id="black-forest-labs/FLUX.1-dev",
        pipeline_class=AutoPipelineForText2Image,
        default_steps=20,
        default_guidance=3.5,
        min_width=256,
        max_width=2048,
        min_height=256,
        max_height=2048,
        vram_gb=24.0,
        description="Higher quality, supports guidance"
    ),
    "sdxl": ModelConfig(
        name="Stable Diffusion XL",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_class=StableDiffusionXLPipeline,
        default_steps=30,
        default_guidance=7.5,
        min_width=512,
        max_width=2048,
        min_height=512,
        max_height=2048,
        vram_gb=10.0,
        description="SDXL base model, versatile"
    ),
    "playground": ModelConfig(
        name="Playground v2.5",
        model_id="playgroundai/playground-v2.5-1024px-aesthetic",
        pipeline_class=AutoPipelineForText2Image,
        default_steps=30,
        default_guidance=3.0,
        min_width=512,
        max_width=1536,
        min_height=512,
        max_height=1536,
        vram_gb=12.0,
        description="Aesthetic-focused model"
    ),
    "turbo": ModelConfig(
        name="SDXL Turbo",
        model_id="stabilityai/sdxl-turbo",
        pipeline_class=AutoPipelineForText2Image,
        default_steps=2,  # 1-4 steps work, 2 is more stable
        default_guidance=0.0,
        min_width=512,
        max_width=512,
        min_height=512,
        max_height=512,
        vram_gb=8.0,
        description="Ultra-fast SDXL generation (1-4 steps)"
    ),
    "realvisxl": ModelConfig(
        name="RealVisXL V4",
        model_id="SG161222/RealVisXL_V4.0",
        pipeline_class=StableDiffusionXLPipeline,
        default_steps=25,
        default_guidance=7.0,
        min_width=512,
        max_width=2048,
        min_height=512,
        max_height=2048,
        vram_gb=10.0,
        description="Photorealistic imagery"
    ),
}

# Global configuration and state variables
server_config: Optional[argparse.Namespace]

# Global pipeline storage for worker processes
# Each worker will have a model assigned to it
worker_pipeline = None  # Will be set in each worker process

# Model assignment tracking
model_assignments = {}  # model_key -> [worker_ids]
model_load_status = {}  # model_key -> bool

# Global shutdown event
shutdown_event = mp.Event()

# WebSocket connections
websocket_connections: List[WebSocket] = []
websocket_lock = threading.Lock()

# Generation stats per model
generation_stats = {}
stats_lock = threading.Lock()


# Request model definition with model selection
class TextToImageInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    model: str = Field("flux-schnell", description="Model to use for generation")
    width: int = Field(1024, ge=256, le=4096)
    height: int = Field(768, ge=256, le=4096)
    steps: Optional[int] = Field(None, ge=1, le=100)
    guidance: Optional[float] = Field(None, ge=0.0, le=20.0)
    seed: Optional[int] = Field(None, ge=-2147483648, le=2147483647)
    
    @field_validator('width', 'height')
    def dimensions_divisible_by_16(cls, v):
        if v % 16 != 0:
            raise ValueError('Dimensions must be divisible by 16')
        return v
    
    @field_validator('model')
    def valid_model(cls, v):
        if v not in AVAILABLE_MODELS:
            raise ValueError(f'Model {v} not available')
        return v


def worker_process(worker_id: int, model_key: str, gpu_id: int, pipe: mp.Pipe, ready_queue: mp.Queue, error_queue: mp.Queue):
    """Worker process that handles image generation requests."""
    global worker_pipeline
    
    # Ignore SIGINT in worker processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        # Set the GPU for this process
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        # Initialize random seed
        random.seed(worker_id + int(time.time() * 1000000))

        model_config = AVAILABLE_MODELS[model_key]
        logger.info(f"Worker {worker_id} starting on GPU {gpu_id} with model {model_config.name}")

        # Set float32 matmul precision
        torch.set_float32_matmul_precision("high")
        
        # Enable TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load the appropriate model
        if model_config.pipeline_class == AutoPipelineForText2Image:
            # Special handling for SDXL Turbo
            if model_key == "turbo":
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_config.model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                ).to(device)
            else:
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_config.model_id,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                ).to(device)
        else:
            # For specific pipeline classes
            pipeline = model_config.pipeline_class.from_pretrained(
                model_config.model_id,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="fp16" if "xl" in model_config.model_id.lower() else None,
            ).to(device)

        # Optimize model components if available
        if hasattr(pipeline, 'transformer'):
            pipeline.transformer.to(memory_format=torch.channels_last)
            if hasattr(pipeline.transformer, 'fuse_qkv_projections'):
                pipeline.transformer.fuse_qkv_projections()
        
        if hasattr(pipeline, 'vae'):
            pipeline.vae.to(memory_format=torch.channels_last)
            if hasattr(pipeline.vae, 'fuse_qkv_projections'):
                pipeline.vae.fuse_qkv_projections()

        # Disable progress bars
        pipeline.set_progress_bar_config(disable=True)

        generator = Generator(device)

        logger.info(f"Worker {worker_id} loaded {model_config.name} successfully")

        # Warmup
        warmup_args = {
            "prompt": "warmup",
            "width": max(512, model_config.min_width),
            "height": max(512, model_config.min_height),
            "num_inference_steps": 1,
            "guidance_scale": model_config.default_guidance,
        }
        _ = pipeline(**warmup_args, output_type="pil")
        logger.info(f"Worker {worker_id} warmup complete")
        
        # Store pipeline info in global
        worker_pipeline = (pipeline, generator, device, gpu_id, model_key)
        
        # Signal ready
        ready_queue.put((worker_id, model_key, gpu_id))
        
        # Main worker loop
        while True:
            try:
                # Wait for request from pipe
                request = pipe.recv()
                
                if request is None:  # Shutdown signal
                    break
                
                gen_args, seed, request_id = request
                
                # Generate image
                logger.info(
                    f"Worker {worker_id} (GPU {gpu_id}, {model_key}) processing: "
                    f"prompt='{gen_args['prompt'][:50]}...' "
                    f"size={gen_args['width']}x{gen_args['height']} "
                    f"steps={gen_args['num_inference_steps']} guidance={gen_args['guidance_scale']} seed={seed}"
                )
                
                if seed is not None:
                    generator.manual_seed(seed)
                else:
                    random_seed = random.randint(0, 2**32 - 1)
                    generator.manual_seed(random_seed)
                    seed = random_seed
                
                start_time = time.time()
                
                # Generate image
                # Use float16 for turbo, bfloat16 for others
                dtype = torch.float16 if model_key == "turbo" else torch.bfloat16
                with torch.amp.autocast('cuda', dtype=dtype):
                    output = pipeline(**gen_args, generator=generator, output_type="pil")
                
                elapsed = time.time() - start_time
                
                # Convert to base64
                buffered = io.BytesIO()
                image = output.images[0]
                
                # Fix for SDXL Turbo potential blank images
                if model_key == "turbo":
                    import numpy as np
                    from PIL import Image
                    # Convert to numpy array and ensure proper range
                    img_array = np.array(image)
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                    image = Image.fromarray(img_array)
                
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                logger.info(f"Worker {worker_id} completed in {elapsed:.2f}s")
                
                # Send result back
                pipe.send((img_str, elapsed, seed))
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                logger.error(traceback.format_exc())
                pipe.send((None, 0, None))
        
    except Exception as e:
        logger.error(f"Worker {worker_id} failed to initialize: {str(e)}")
        logger.error(traceback.format_exc())
        error_queue.put((worker_id, str(e), traceback.format_exc()))
        raise




class MultiModelGenerator:
    """Manages multiple models across multiple GPUs."""

    def __init__(self, model_distribution: Dict[str, int], num_gpus: int):
        self.model_distribution = model_distribution
        self.num_gpus = num_gpus
        self.shutdown_requested = False
        self.worker_processes = []
        self.executor = None
        
        # Track which workers have which models
        self.model_to_workers = {model: [] for model in model_distribution}
        self.worker_to_model = {}
        
        # Worker pipes for communication
        self.worker_pipes = {}  # worker_id -> (parent_conn, child_conn)
        
        # Queues for worker coordination
        self.ready_queue = mp.Queue()
        self.error_queue = mp.Queue()
        
        # Task queues per model
        self.model_queues = {model: asyncio.Queue() for model in model_distribution}
        self.result_futures = {}
        
        # Start workers
        self._start_workers()
        
        # Create thread pool executor for running tasks
        self.executor = ThreadPoolExecutor(max_workers=sum(model_distribution.values()))
        
        # Start task processors
        self.task_processors = []
        for model in model_distribution:
            for _ in range(len(self.model_to_workers[model])):
                processor = asyncio.create_task(self._process_tasks(model))
                self.task_processors.append(processor)

    def _start_workers(self):
        """Start worker processes with assigned models."""
        logger.info(f"Starting workers with distribution: {self.model_distribution}")
        
        # Create worker assignments
        worker_id = 0
        gpu_id = 0
        worker_args = []
        
        for model_key, count in self.model_distribution.items():
            for _ in range(count):
                if gpu_id >= self.num_gpus:
                    raise ValueError(f"Not enough GPUs for requested model distribution")
                
                # Create pipe for this worker
                parent_conn, child_conn = mp.Pipe()
                self.worker_pipes[worker_id] = (parent_conn, child_conn)
                
                worker_args.append((worker_id, model_key, gpu_id, child_conn, self.ready_queue, self.error_queue))
                self.worker_to_model[worker_id] = model_key
                worker_id += 1
                gpu_id += 1
        
        # Create process pool
        processes = []
        for args in worker_args:
            p = mp.Process(target=worker_process, args=args)
            p.start()
            processes.append(p)
        
        # Wait for workers to be ready
        ready_count = 0
        errors = []
        timeout = 120
        start_time = time.time()
        
        while ready_count < len(worker_args):
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Timeout waiting for workers. Only {ready_count}/{len(worker_args)} ready.")
            
            # Check for errors
            try:
                worker_id, error_msg, traceback_str = self.error_queue.get_nowait()
                errors.append(f"Worker {worker_id}: {error_msg}")
                logger.error(f"Worker {worker_id} failed:\n{traceback_str}")
            except queue.Empty:
                pass
            
            # Check for ready workers
            try:
                worker_id, model_key, gpu_id = self.ready_queue.get(timeout=0.1)
                self.model_to_workers[model_key].append(worker_id)
                ready_count += 1
                logger.info(f"Worker {worker_id} ready ({ready_count}/{len(worker_args)})")
            except queue.Empty:
                pass
            
            if errors:
                raise RuntimeError(f"Worker initialization failed:\n" + "\n".join(errors))
        
        # Don't create a pool - we'll use the existing processes
        self.worker_processes = processes
        logger.info("All workers initialized and ready")
        
        # Update global model status
        global model_load_status, model_assignments
        for model, workers in self.model_to_workers.items():
            model_load_status[model] = len(workers) > 0
            model_assignments[model] = workers

    async def _process_tasks(self, model_key: str):
        """Process generation tasks for a specific model."""
        while not self.shutdown_requested:
            try:
                # Get task from queue
                task = await asyncio.wait_for(
                    self.model_queues[model_key].get(), 
                    timeout=1.0
                )
                
                if task is None:  # Shutdown signal
                    break
                
                gen_args, seed, request_id, future = task
                
                # Get available worker for this model
                workers = self.model_to_workers[model_key]
                if not workers:
                    future.set_exception(RuntimeError(f"No workers available for model {model_key}"))
                    continue
                
                # Round-robin worker selection
                worker_id = workers[hash(request_id) % len(workers)]
                
                # Submit to worker via pipe
                try:
                    parent_conn, _ = self.worker_pipes[worker_id]
                    
                    # Send request to worker
                    parent_conn.send((gen_args, seed, request_id))
                    
                    # Wait for response asynchronously
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        parent_conn.recv
                    )
                    
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Task processor error: {e}")

    async def generate(self, model_key: str, gen_args: Dict[str, Any], seed: int | None, request_id: str) -> Tuple[str, float, int]:
        """Submit a generation job for a specific model."""
        if self.shutdown_requested:
            raise RuntimeError("Generator is shutting down")
        
        if model_key not in self.model_to_workers:
            raise ValueError(f"Model {model_key} not loaded")
        
        if not self.model_to_workers[model_key]:
            raise RuntimeError(f"No workers available for model {model_key}")
        
        # Create future for result
        future = asyncio.Future()
        
        # Add to appropriate queue
        await self.model_queues[model_key].put((gen_args, seed, request_id, future))
        
        # Wait for result
        return await future

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all loaded models."""
        status = {}
        for model_key, workers in self.model_to_workers.items():
            status[model_key] = {
                "loaded": len(workers) > 0,
                "worker_count": len(workers),
                "queue_size": self.model_queues[model_key].qsize() if model_key in self.model_queues else 0
            }
        return status

    def shutdown(self):
        """Shutdown all workers."""
        if self.shutdown_requested:
            return
            
        self.shutdown_requested = True
        logger.info("Shutting down multi-model generator")
        
        # Signal shutdown to task processors
        for model in self.model_queues:
            self.model_queues[model].put_nowait(None)
        
        # Cancel task processors
        for processor in self.task_processors:
            processor.cancel()
        
        # Shutdown workers
        shutdown_event.set()
        
        # Send shutdown signal to all workers
        for worker_id, (parent_conn, _) in self.worker_pipes.items():
            try:
                parent_conn.send(None)
            except:
                pass
        
        # Wait for processes to exit
        for p in self.worker_processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        # Close pipes
        for parent_conn, child_conn in self.worker_pipes.values():
            parent_conn.close()
            child_conn.close()
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            
        logger.info("Multi-model generator shutdown complete")


# Global generator instance
shared_generator: Optional[MultiModelGenerator] = None
app: Optional[FastAPI] = None
shutdown_in_progress = False
shutdown_lock = threading.Lock()


def handle_shutdown():
    """Common shutdown handler."""
    global shutdown_in_progress, shared_generator
    
    with shutdown_lock:
        if shutdown_in_progress:
            return
        shutdown_in_progress = True
    
    logger.info("Initiating graceful shutdown...")
    
    shutdown_event.set()
    
    if shared_generator:
        shared_generator.shutdown()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-model image generation server with web interface."
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
        "--models",
        type=str,
        default="flux-schnell:4,sdxl:2,playground:2",
        help="Model distribution as model:count pairs (e.g., flux-schnell:4,sdxl:2)",
    )

    args = parser.parse_args()
    
    # Parse model distribution
    args.model_distribution = {}
    for spec in args.models.split(","):
        if ":" in spec:
            model, count = spec.split(":")
            model = model.strip()
            if model in AVAILABLE_MODELS:
                args.model_distribution[model] = int(count)
            else:
                logger.warning(f"Unknown model: {model}")
        else:
            # Default to 1 instance
            model = spec.strip()
            if model in AVAILABLE_MODELS:
                args.model_distribution[model] = 1
    
    if not args.model_distribution:
        # Default distribution
        args.model_distribution = {"flux-schnell": 4}
    
    # Calculate total GPUs needed
    args.num_gpus = sum(args.model_distribution.values())
    
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    global shared_generator, generation_stats

    try:
        # Initialize generation stats for each model
        for model_key in AVAILABLE_MODELS:
            generation_stats[model_key] = {
                "total_requests": 0,
                "total_completed": 0,
                "average_time": 0.0,
                "dimension_counts": {},
            }
        
        # Initialize the multi-model generator
        logger.info(f"Initializing with model distribution: {server_config.model_distribution}")
        logger.info(f"Total GPUs needed: {server_config.num_gpus}")
        
        available_gpus = torch.cuda.device_count()
        if server_config.num_gpus > available_gpus:
            raise ValueError(f"Requested {server_config.num_gpus} GPUs but only {available_gpus} available")
        
        shared_generator = MultiModelGenerator(
            model_distribution=server_config.model_distribution,
            num_gpus=server_config.num_gpus
        )
        yield
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise
    finally:
        handle_shutdown()
        logger.info("Server shutdown complete")


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan, title="Multi-Model Image Generation Studio")

# Set up static files and templates
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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
        
        for ws in disconnected:
            websocket_connections.remove(ws)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web interface."""
    # Prepare model info for frontend
    models_info = {}
    for key, config in AVAILABLE_MODELS.items():
        models_info[key] = {
            "name": config.name,
            "description": config.description,
            "default_steps": config.default_steps,
            "default_guidance": config.default_guidance,
            "min_width": config.min_width,
            "max_width": config.max_width,
            "min_height": config.min_height,
            "max_height": config.max_height,
        }
    
    return templates.TemplateResponse("index_multimodel.html", {
        "request": request,
        "models": models_info,
        "loaded_models": list(server_config.model_distribution.keys()) if server_config else [],
    })


@app.get("/api/models")
async def get_models():
    """Get available models and their status."""
    if not shared_generator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    model_status = shared_generator.get_model_status()
    
    response = {}
    for key, config in AVAILABLE_MODELS.items():
        status = model_status.get(key, {"loaded": False, "worker_count": 0, "queue_size": 0})
        response[key] = {
            "name": config.name,
            "description": config.description,
            "default_steps": config.default_steps,
            "default_guidance": config.default_guidance,
            "min_width": config.min_width,
            "max_width": config.max_width,
            "min_height": config.min_height,
            "max_height": config.max_height,
            "loaded": status["loaded"],
            "worker_count": status["worker_count"],
            "queue_size": status["queue_size"],
        }
    
    return JSONResponse(response)


@app.get("/api/stats")
async def get_stats():
    """Get server statistics."""
    with stats_lock:
        stats = {
            "models": {},
            "total_requests": 0,
            "total_completed": 0,
        }
        
        for model_key, model_stats in generation_stats.items():
            if model_stats["total_requests"] > 0:
                stats["models"][model_key] = model_stats.copy()
                stats["total_requests"] += model_stats["total_requests"]
                stats["total_completed"] += model_stats["total_completed"]
    
    return JSONResponse(stats)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    with websocket_lock:
        websocket_connections.append(websocket)
    
    try:
        # Send initial model status
        if shared_generator:
            await websocket.send_json({
                "type": "model_status",
                "models": shared_generator.get_model_status()
            })
        
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        with websocket_lock:
            if websocket in websocket_connections:
                websocket_connections.remove(websocket)


@app.post("/v1/images/generations", response_class=JSONResponse)
async def generate_image(request: Request, image_input: TextToImageInput):
    """Generate an image using the specified model."""
    try:
        if not shared_generator:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Get model config
        model_config = AVAILABLE_MODELS[image_input.model]
        
        # Validate dimensions for this model
        if not model_config.supports_dimensions(image_input.width, image_input.height):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_config.name} doesn't support {image_input.width}x{image_input.height}. "
                       f"Supported range: {model_config.min_width}-{model_config.max_width} x "
                       f"{model_config.min_height}-{model_config.max_height}"
            )
        
        # Use model defaults if not specified
        steps = image_input.steps if image_input.steps is not None else model_config.default_steps
        guidance = image_input.guidance if image_input.guidance is not None else model_config.default_guidance
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Broadcast start
        await broadcast_status({
            "type": "generation_start",
            "request_id": request_id,
            "model": image_input.model,
        })
        
        # Prepare arguments
        gen_args = {
            "prompt": image_input.prompt,
            "width": image_input.width,
            "height": image_input.height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }

        # Generate image
        img_str, gen_time, used_seed = await shared_generator.generate(
            image_input.model, gen_args, image_input.seed, request_id
        )
        
        # Update statistics
        with stats_lock:
            model_stats = generation_stats[image_input.model]
            model_stats["total_requests"] += 1
            model_stats["total_completed"] += 1
            total = model_stats["total_completed"]
            model_stats["average_time"] = (
                (model_stats["average_time"] * (total - 1) + gen_time) / total
            )
            
            dim_key = f"{image_input.width}x{image_input.height}"
            if dim_key not in model_stats["dimension_counts"]:
                model_stats["dimension_counts"][dim_key] = 0
            model_stats["dimension_counts"][dim_key] += 1
        
        # Broadcast completion
        await broadcast_status({
            "type": "generation_complete",
            "request_id": request_id,
            "model": image_input.model,
            "gen_time": gen_time,
        })

        return JSONResponse({
            "image": img_str,
            "gen_time": gen_time,
            "seed": used_seed,
            "width": image_input.width,
            "height": image_input.height,
            "model": image_input.model,
            "model_name": model_config.name,
        })
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        logger.error(traceback.format_exc())
        
        await broadcast_status({
            "type": "generation_error",
            "request_id": request_id if 'request_id' in locals() else None,
            "error": str(e),
        })
        
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    server_config = parse_args()
    
    mp.set_start_method("spawn", force=True)

    try:
        config = uvicorn.Config(
            app,
            host=server_config.host,
            port=server_config.port,
            log_level="info",
            access_log=False,
        )
        server = uvicorn.Server(config)
        
        original_signal_handler = server.handle_exit
        
        def custom_signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            handle_shutdown()
            original_signal_handler(sig, frame)
        
        server.handle_exit = custom_signal_handler
        
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        handle_shutdown()