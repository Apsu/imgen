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
from hunyuan_supported_resolutions import HUNYUAN_SUPPORTED_RESOLUTIONS, get_closest_hunyuan_resolution

import torch
import torch.multiprocessing as mp
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from torch import Generator
from diffusers import (
    AutoPipelineForText2Image, 
    StableDiffusionXLPipeline, 
    StableDiffusionPipeline,
    SanaPipeline,
    SanaSprintPipeline,
    HunyuanDiTPipeline, 
    PixArtSigmaPipeline,
    StableCascadePriorPipeline,
    StableCascadeDecoderPipeline,
    LuminaPipeline,
    ChromaPipeline,
    ChromaTransformer2DModel
)
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import logging as diffusers_logging
from compel import Compel, ReturnedEmbeddingsType

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

# Helper functions for compel support
def model_uses_clip(model_key: str) -> bool:
    """Check if a model uses CLIP text encoder(s) for compel processing."""
    clip_models = {
        # SDXL variants (dual CLIP)
        "sdxl", "juggernaut", "realvisxl", "dreamshaper", "anime",
        # Stable Cascade (single CLIP)
        "stable-cascade",
        # Playground (dual CLIP)
        "playground",
        # HunyuanDiT (bilingual CLIP + mT5)
        "hunyuan"
        # Note: FLUX models use T5 which can handle longer prompts natively (512 tokens vs 77)
        # so compel is not needed for FLUX
    }
    return model_key in clip_models

def model_uses_dual_clip(model_key: str) -> bool:
    """Check if a model uses dual CLIP encoders (SDXL-style)."""
    dual_clip_models = {
        # SDXL variants
        "sdxl", "juggernaut", "realvisxl", "dreamshaper", "anime", "playground"
        # Note: FLUX models removed - they use T5 which handles long prompts natively
    }
    return model_key in dual_clip_models

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
    supports_guidance: bool = True  # Whether model supports guidance scale
    supports_negative_prompt: bool = True  # Whether model supports negative prompts
    recommended_resolutions: Optional[List[Tuple[int, int]]] = None
    resolution_constraints: Optional[str] = None
    
    def supports_dimensions(self, width: int, height: int) -> bool:
        return (self.min_width <= width <= self.max_width and 
                self.min_height <= height <= self.max_height)
    
    def get_resolution_info(self) -> Dict[str, Any]:
        """Get resolution information including recommendations."""
        return {
            "min_width": self.min_width,
            "max_width": self.max_width,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "recommended": getattr(self, "recommended_resolutions", None),
            "constraints": getattr(self, "resolution_constraints", None)
        }


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
        description="Fast high-quality generation, no CFG needed",
        supports_guidance=False,  # Schnell doesn't use guidance
        supports_negative_prompt=False  # FLUX models don't support negative prompts
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
        description="Higher quality, supports guidance",
        supports_negative_prompt=False  # FLUX models don't support negative prompts
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
    "juggernaut": ModelConfig(
        name="Juggernaut XL v9",
        model_id="RunDiffusion/Juggernaut-XL-v9",
        pipeline_class=StableDiffusionXLPipeline,
        default_steps=30,
        default_guidance=7.0,
        min_width=512,
        max_width=2048,
        min_height=512,
        max_height=2048,
        vram_gb=10.0,
        description="High-quality portraits and photorealistic imagery"
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
    "dreamshaper": ModelConfig(
        name="DreamShaper XL v2 Turbo",
        model_id="Lykon/dreamshaper-xl-v2-turbo",
        pipeline_class=StableDiffusionXLPipeline,
        default_steps=6,
        default_guidance=2.0,
        min_width=512,
        max_width=2048,
        min_height=512,
        max_height=2048,
        vram_gb=10.0,
        description="Artistic and fantasy-focused (turbo)"
    ),
    "anime": ModelConfig(
        name="Animagine XL 4.0",
        model_id="cagliostrolab/animagine-xl-4.0",
        pipeline_class=StableDiffusionXLPipeline,
        default_steps=25,
        default_guidance=6.0,  # Updated to recommended value
        min_width=512,
        max_width=2048,
        min_height=512,
        max_height=2048,
        vram_gb=10.0,
        description="High-quality anime-style generation v4 (use tag-based prompts)",
        recommended_resolutions=[(1024, 1024), (832, 1216), (1216, 832)],
        resolution_constraints="Best with tag-based prompts (e.g. '1girl, solo, long hair'). Add quality tags: 'masterpiece, absurdres'"
    ),
    # --- Cutting-Edge Diffusion Transformer Models ---
    "sana": ModelConfig(
        name="Sana 1600M 1K",  # Using 1024px version for better quality
        model_id="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        pipeline_class=SanaPipeline,
        default_steps=20,
        default_guidance=4.5,
        min_width=512,
        max_width=1024,  # 1024px version has better quality
        min_height=512,
        max_height=1024,
        vram_gb=12.0,
        description="High-quality 1K generation with Linear DiT (best with complex prompts)",
        resolution_constraints="Works best with complex, detailed prompts. Simple prompts may produce artifacts."
    ),
    "hunyuan": ModelConfig(
        name="HunyuanDiT Bilingual",
        model_id="Tencent-Hunyuan/HunyuanDiT-Diffusers",
        pipeline_class=HunyuanDiTPipeline,
        default_steps=50,
        default_guidance=5.0,  # Updated to match documentation
        min_width=768,  # Constrain to supported sizes
        max_width=1280,
        min_height=768,
        max_height=1280,
        vram_gb=12.0,
        description="Bilingual (EN/CN) understanding with cultural context",
        recommended_resolutions=HUNYUAN_SUPPORTED_RESOLUTIONS,
        resolution_constraints="Only supports specific resolutions. Others will be adjusted automatically."
    ),
    "pixart": ModelConfig(
        name="PixArt-Sigma XL",
        model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        pipeline_class=PixArtSigmaPipeline,
        default_steps=20,
        default_guidance=4.5,
        min_width=512,
        max_width=4096,
        min_height=512,
        max_height=4096,
        vram_gb=8.0,
        description="Ultra-efficient 0.6B param model, exceptional prompt adherence, supports up to 4K",
        recommended_resolutions=[
            (1024, 1024), (2048, 2048), (4096, 4096),  # Square
            (1920, 1080), (3840, 2160),  # 16:9
            (1080, 1920), (2160, 3840),  # 9:16
        ]
    ),
    "sana-sprint": ModelConfig(
        name="Sana Sprint 1.6B",
        model_id="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        pipeline_class=SanaSprintPipeline,
        default_steps=2,
        default_guidance=4.5,
        min_width=512,
        max_width=1024,
        min_height=512,
        max_height=1024,
        vram_gb=8.0,
        description="Ultra-fast 1-4 step generation, NVIDIA/MIT/HF collaboration"
    ),
    "sana-sprint-small": ModelConfig(
        name="Sana Sprint 0.6B",
        model_id="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        pipeline_class=SanaSprintPipeline,
        default_steps=2,
        default_guidance=4.5,
        min_width=512,
        max_width=1024,
        min_height=512,
        max_height=1024,
        vram_gb=5.0,
        description="Smallest ultra-fast model, 1-4 step generation"
    ),
    "stable-cascade": ModelConfig(
        name="Stable Cascade",
        model_id="stabilityai/stable-cascade-prior",  # Will need special handling for decoder
        pipeline_class=StableCascadePriorPipeline,
        default_steps=20,
        default_guidance=4.0,
        min_width=1024,
        max_width=1536,  # Stable Cascade supports up to 1536
        min_height=1024,
        max_height=1536,
        vram_gb=20.0,  # Prior + Decoder
        description="Würstchen v3: Revolutionary two-stage 24x24 latent architecture",
        resolution_constraints="Minimum 1024x1024 required. Two-stage generation (prior + decoder)."
    ),
    "lumina-next": ModelConfig(
        name="Lumina-Next 2B",
        model_id="Alpha-VLLM/Lumina-Next-SFT-diffusers",
        pipeline_class=LuminaPipeline,
        default_steps=60,  # Recommended sampling steps
        default_guidance=4.0,  # Recommended cfg_scale
        min_width=512,
        max_width=2048,
        min_height=512,
        max_height=2048,
        vram_gb=12.0,
        description="2B DiT model with Gemma-2B text encoder, supports 2K resolution",
        recommended_resolutions=[
            (1024, 1024), (512, 2048), (2048, 512),  # Standard resolutions
            (1664, 1664), (1024, 2048), (2048, 1024),  # Extrapolation resolutions
        ]
    ),
    "chroma": ModelConfig(
        name="Chroma 8.9B",
        model_id="lodestones/Chroma",
        pipeline_class=ChromaPipeline,
        default_steps=4,
        default_guidance=3.5,
        min_width=512,
        max_width=2048,
        min_height=512,
        max_height=2048,
        vram_gb=16.0,
        description="Uncensored FLUX variant, 8.9B params, Apache 2.0"
    ),
}


def validate_and_adjust_resolution(model_key: str, width: int, height: int) -> Tuple[int, int, Optional[str]]:
    """
    Validate and potentially adjust resolution for a model.
    Returns: (adjusted_width, adjusted_height, warning_message)
    """
    model_config = AVAILABLE_MODELS[model_key]
    warning = None
    
    # Special handling for HunyuanDiT
    if model_key == "hunyuan":
        # Check if resolution is supported
        if (width, height) not in HUNYUAN_SUPPORTED_RESOLUTIONS:
            new_width, new_height = get_closest_hunyuan_resolution(width, height)
            warning = (f"HunyuanDiT doesn't support {width}x{height}. "
                      f"Adjusted to closest supported resolution: {new_width}x{new_height}")
            width, height = new_width, new_height
    
    # Ensure dimensions are within model bounds
    width = max(model_config.min_width, min(width, model_config.max_width))
    height = max(model_config.min_height, min(height, model_config.max_height))
    
    # Ensure divisible by 16
    width = (width // 16) * 16
    height = (height // 16) * 16
    
    return width, height, warning


def get_model_resolution_info(model_key: str) -> Dict[str, Any]:
    """Get detailed resolution information for a model."""
    config = AVAILABLE_MODELS[model_key]
    info = config.get_resolution_info()
    
    # Add model-specific information
    if model_key == "hunyuan":
        info["supported_resolutions"] = HUNYUAN_SUPPORTED_RESOLUTIONS
        info["auto_adjust"] = True
    elif model_key == "anime":
        info["prompt_format"] = "tag-based"
        info["quality_tags"] = ["masterpiece", "absurdres", "high score", "great score"]
    elif model_key == "sana":
        info["prompt_recommendation"] = "Use complex, detailed prompts for best results"
    
    return info


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
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt. Supports prompt weighting syntax like (beautiful:1.2) for CLIP-based models")
    model: str = Field("flux-schnell", description="Model to use for generation")
    width: int = Field(1024, ge=256, le=4096)
    height: int = Field(768, ge=256, le=4096)
    steps: Optional[int] = Field(None, ge=1, le=100)
    guidance: Optional[float] = Field(None, ge=0.0, le=20.0)
    seed: Optional[int] = Field(None, ge=-2147483648, le=2147483647)
    negative_prompt: Optional[str] = Field(None, max_length=2000, description="Negative prompt for models that support it. Supports prompt weighting syntax")
    num_images: Optional[int] = Field(1, ge=1, le=4, description="Number of images to generate")
    
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
        if model_key == "stable-cascade":
            # Special handling for Stable Cascade two-stage model
            logger.info(f"Loading Stable Cascade two-stage model on GPU {gpu_id}")
            prior = StableCascadePriorPipeline.from_pretrained(
                "stabilityai/stable-cascade-prior",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16",
            ).to(device)
            decoder = StableCascadeDecoderPipeline.from_pretrained(
                "stabilityai/stable-cascade",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                variant="bf16",
            ).to(device)
            # Store both pipelines as a tuple
            pipeline = (prior, decoder)
        elif model_key == "chroma":
            # Special handling for Chroma model using from_single_file
            logger.info(f"Loading Chroma model using from_single_file on GPU {gpu_id}")
            bfl_repo = "black-forest-labs/FLUX.1-dev"
            
            # Load Chroma transformer
            transformer = ChromaTransformer2DModel.from_single_file(
                "https://huggingface.co/lodestones/Chroma/blob/main/chroma-unlocked-v36.safetensors",
                torch_dtype=torch.bfloat16
            )
            
            # Load text encoder and tokenizer from FLUX
            text_encoder = T5EncoderModel.from_pretrained(
                bfl_repo, 
                subfolder="text_encoder_2", 
                torch_dtype=torch.bfloat16
            )
            tokenizer = T5Tokenizer.from_pretrained(
                bfl_repo, 
                subfolder="tokenizer_2"
            )
            
            # Create pipeline
            pipeline = ChromaPipeline.from_pretrained(
                bfl_repo, 
                transformer=transformer, 
                text_encoder=text_encoder, 
                tokenizer=tokenizer, 
                torch_dtype=torch.bfloat16
            ).to(device)
        elif model_key in ["sana-sprint", "sana-sprint-small"]:
            # Special handling for Sana Sprint models according to documentation
            logger.info(f"Loading Sana Sprint model {model_config.name} with recommended dtype settings")
            pipeline = SanaSprintPipeline.from_pretrained(
                model_config.model_id,
                torch_dtype=torch.bfloat16,  # Recommended dtype as per documentation
                use_safetensors=True,
            ).to(device)
        elif model_config.pipeline_class == AutoPipelineForText2Image:
            # AutoPipeline models: FLUX uses bfloat16, Playground uses float16
            if "flux" in model_key:
                dtype = torch.bfloat16
                variant = None
            elif "playground" in model_key:
                dtype = torch.float16
                variant = "fp16" if "fp16" in model_config.model_id else None
            else:
                dtype = torch.bfloat16
                variant = None
                
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_config.model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant=variant,
            ).to(device)
        else:
            # For specific pipeline classes
            # Determine appropriate dtype based on model type
            if model_config.pipeline_class == StableDiffusionXLPipeline:
                # SDXL models should use float16
                dtype = torch.float16
                variant = "fp16"
            elif model_config.pipeline_class == SanaPipeline:
                # Sana needs special handling - see issue #10241
                # Load with float16 for transformer, but components need different dtypes
                dtype = torch.float16  # For transformer
                variant = "bf16"
            elif model_config.pipeline_class == LuminaPipeline:
                # Lumina uses bfloat16
                dtype = torch.bfloat16
                variant = None
            elif model_config.pipeline_class == HunyuanDiTPipeline:
                # HunyuanDiT works better with bfloat16 to avoid quality issues
                dtype = torch.bfloat16
                variant = None
            elif model_config.pipeline_class == PixArtSigmaPipeline:
                # PixArt-Sigma can use float16
                dtype = torch.float16
                variant = None
            else:
                # Default to bfloat16 for other models
                dtype = torch.bfloat16
                variant = None
            
            try:
                pipeline = model_config.pipeline_class.from_pretrained(
                    model_config.model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant=variant,
                ).to(device)
            except ValueError as e:
                if "variant" in str(e):
                    # Try without variant
                    logger.info(f"Model {model_config.name} doesn't have {variant} variant, loading without variant")
                    pipeline = model_config.pipeline_class.from_pretrained(
                        model_config.model_id,
                        torch_dtype=dtype,
                        use_safetensors=True,
                    ).to(device)
                else:
                    raise

        # Optimize model components if available
        # Skip optimization for DiT models and Stable Cascade to avoid compatibility issues
        is_dit_model = model_config.pipeline_class.__name__ in ['SanaPipeline', 'SanaSprintPipeline', 'PixArtSigmaPipeline', 'LuminaPipeline', 'ChromaPipeline']
        
        if model_key == "stable-cascade":
            # Skip optimizations for Stable Cascade two-stage model
            logger.info(f"Worker {worker_id} skipping optimizations for Stable Cascade two-stage model")
            # Disable progress bars for both pipelines
            prior, decoder = pipeline
            prior.set_progress_bar_config(disable=True)
            decoder.set_progress_bar_config(disable=True)
        elif model_key == "chroma":
            # Skip optimizations for Chroma model
            logger.info(f"Worker {worker_id} skipping optimizations for Chroma model")
            # Disable progress bars
            pipeline.set_progress_bar_config(disable=True)
        elif model_key in ["sana-sprint", "sana-sprint-small"]:
            # Skip optimizations for Sana Sprint models
            logger.info(f"Worker {worker_id} skipping optimizations for Sana Sprint model {model_config.name}")
            # Disable progress bars
            pipeline.set_progress_bar_config(disable=True)
        elif model_key == "hunyuan":
            # HunyuanDiT benefits from memory layout optimization
            logger.info(f"Worker {worker_id} applying memory layout optimization for HunyuanDiT")
            if hasattr(pipeline, 'transformer'):
                pipeline.transformer.to(memory_format=torch.channels_last)
            if hasattr(pipeline, 'vae'):
                pipeline.vae.to(memory_format=torch.channels_last)
            # Disable progress bars
            pipeline.set_progress_bar_config(disable=True)
        elif not is_dit_model:
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
        elif model_key == "sana":
            # Special handling for Sana to fix quality issues (see diffusers issue #10241)
            logger.info(f"Worker {worker_id} applying special dtype handling for Sana")
            # Text encoder MUST be bfloat16
            if hasattr(pipeline, 'text_encoder'):
                pipeline.text_encoder = pipeline.text_encoder.to(torch.bfloat16)
            # VAE should be fp32 or bf16 (NOT fp16 to avoid black results)
            if hasattr(pipeline, 'vae'):
                pipeline.vae = pipeline.vae.to(torch.bfloat16)
            # Transformer stays at float16 as loaded
            # Disable progress bars
            pipeline.set_progress_bar_config(disable=True)
        else:
            logger.info(f"Worker {worker_id} skipping advanced optimizations for DiT model {model_config.name}")
            # Disable progress bars
            pipeline.set_progress_bar_config(disable=True)


        generator = Generator(device)

        logger.info(f"Worker {worker_id} loaded {model_config.name} successfully")

        # Warmup
        warmup_args = {
            "prompt": "warmup",
            "width": max(512, model_config.min_width),
            "height": max(512, model_config.min_height),
            "num_inference_steps": max(1, min(model_config.default_steps, 2)),  # Use model default, but cap at 2 for warmup
            "guidance_scale": model_config.default_guidance,
        }
        
        if model_key == "stable-cascade":
            # Special warmup for Stable Cascade
            prior, decoder = pipeline
            prior_output = prior(**warmup_args, output_type="pt")
            # Decoder doesn't accept width/height, only prompt and basic params
            decoder_args = {
                "prompt": warmup_args["prompt"],
                "num_inference_steps": 10,
                "guidance_scale": 0.0,
            }
            _ = decoder(image_embeddings=prior_output.image_embeddings, output_type="pil", **decoder_args)
        elif model_key == "chroma":
            # Warmup for Chroma using recommended settings
            chroma_warmup_args = {
                "prompt": "warmup",
                "guidance_scale": 4.0,
                "num_inference_steps": 4,  # Use fewer steps for warmup
            }
            _ = pipeline(**chroma_warmup_args, output_type="pil")
        elif model_key in ["sana-sprint", "sana-sprint-small"]:
            # Warmup for Sana Sprint using recommended settings
            sana_sprint_warmup_args = {
                "prompt": "warmup",
                "num_inference_steps": 2,  # Default recommended steps
                "guidance_scale": 4.5,     # Default recommended guidance
            }
            _ = pipeline(**sana_sprint_warmup_args, output_type="pil")
        else:
            _ = pipeline(**warmup_args, output_type="pil")
        logger.info(f"Worker {worker_id} warmup complete")
        
        # Create compel processor for CLIP-based models
        compel_proc = None
        if model_uses_clip(model_key):
            try:
                if model_key in ["flux-schnell", "flux-dev", "chroma"]:
                    # FLUX-style models (CLIP + T5) - only use CLIP for compel
                    logger.info(f"Worker {worker_id} creating FLUX compel processor (CLIP only)")
                    # Start with basic compel and handle pooled embeddings manually
                    compel_proc = Compel(
                        tokenizer=pipeline.tokenizer,
                        text_encoder=pipeline.text_encoder,
                        truncate_long_prompts=False
                    )
                elif model_uses_dual_clip(model_key):
                    # SDXL-style dual CLIP encoders - use standard SDXL approach  
                    logger.info(f"Worker {worker_id} creating SDXL compel processor")
                    compel_proc = Compel(
                        tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                        text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True],
                        truncate_long_prompts=False
                    )
                elif model_key == "stable-cascade":
                    # Stable Cascade has special pipeline structure
                    prior, decoder = pipeline
                    logger.info(f"Worker {worker_id} creating Stable Cascade compel processor")
                    compel_proc = Compel(
                        tokenizer=prior.tokenizer,
                        text_encoder=prior.text_encoder,
                        truncate_long_prompts=False
                    )
                else:
                    # Single CLIP encoder (FLUX, Chroma)
                    logger.info(f"Worker {worker_id} creating single CLIP compel processor")
                    compel_proc = Compel(
                        tokenizer=pipeline.tokenizer,
                        text_encoder=pipeline.text_encoder,
                        truncate_long_prompts=False
                    )
                logger.info(f"Worker {worker_id} compel processor created successfully")
            except Exception as e:
                logger.warning(f"Worker {worker_id} failed to create compel processor: {e}")
                compel_proc = None
        
        # Store pipeline info in global
        worker_pipeline = (pipeline, generator, device, gpu_id, model_key, compel_proc)
        
        # Signal ready
        ready_queue.put((worker_id, model_key, gpu_id))
        
        # Main worker loop
        while True:
            try:
                # Wait for request from pipe with timeout to allow checking shutdown
                if pipe.poll(timeout=1.0):
                    request = pipe.recv()
                    
                    if request is None:  # Shutdown signal
                        break
                else:
                    # No request, check if we should shutdown
                    if shutdown_event.is_set():
                        break
                    continue
                
                gen_args, seed, request_id, num_images = request
                
                # Extract compel processor from worker_pipeline
                if worker_pipeline and len(worker_pipeline) >= 6:
                    _, _, _, _, _, compel_proc = worker_pipeline
                else:
                    # Fallback for older format
                    compel_proc = None
                
                # Log generation info before compel processing
                prompt_text = gen_args.get('prompt', 'no prompt')
                compel_available = compel_proc is not None
                logger.info(
                    f"Worker {worker_id} (GPU {gpu_id}, {model_key}) processing: "
                    f"prompt='{str(prompt_text)[:50]}...' "
                    f"size={gen_args['width']}x{gen_args['height']} "
                    f"steps={gen_args['num_inference_steps']} guidance={gen_args['guidance_scale']} "
                    f"num_images={num_images} seed={seed} compel_available={compel_available}"
                )
                
                if seed is not None:
                    generator.manual_seed(seed)
                else:
                    random_seed = random.randint(0, 2147483647)
                    generator.manual_seed(random_seed)
                    seed = random_seed
                
                start_time = time.time()
                
                # Check shutdown before expensive operations
                if shutdown_event.is_set():
                    logger.info(f"Worker {worker_id} received shutdown signal, breaking")
                    break
                
                # Process prompts with compel if available
                if compel_proc is not None:
                    logger.info(f"Worker {worker_id} using compel to process prompts")
                    try:
                        # Extract text prompts
                        prompt = gen_args.get('prompt', '')
                        negative_prompt = gen_args.get('negative_prompt', '')
                        
                        # Store original prompts for models that need them (like Stable Cascade decoder)
                        if prompt:
                            gen_args['original_prompt'] = prompt
                        if negative_prompt:
                            gen_args['original_negative_prompt'] = negative_prompt
                        
                        # Process both prompts with compel to ensure consistent shapes
                        if prompt:
                            # Process positive prompt
                            logger.debug(f"Worker {worker_id} processing positive prompt length: {len(prompt)} chars")
                            embeddings_result = compel_proc(prompt)
                            logger.debug(f"Worker {worker_id} positive embeddings result type: {type(embeddings_result)}")
                            
                            # Initialize variables to track embeddings for shape comparison
                            positive_embeds = None
                            positive_pooled = None
                            
                            if model_key in ["flux-schnell", "flux-dev", "chroma"]:
                                # FLUX-style models - compel returns regular embeddings, need to create pooled
                                positive_embeds = embeddings_result
                                positive_pooled = embeddings_result
                                gen_args['prompt_embeds'] = positive_embeds
                                gen_args['pooled_prompt_embeds'] = positive_pooled
                            elif model_uses_dual_clip(model_key):
                                # SDXL-style models should return tuple when properly configured
                                if isinstance(embeddings_result, (tuple, list)) and len(embeddings_result) == 2:
                                    # Tuple format: (prompt_embeds, pooled_prompt_embeds)
                                    positive_embeds, positive_pooled = embeddings_result
                                    gen_args['prompt_embeds'] = positive_embeds
                                    gen_args['pooled_prompt_embeds'] = positive_pooled
                                    logger.debug(f"Worker {worker_id} SDXL model returned tuple with shapes: embeds {positive_embeds.shape}, pooled {positive_pooled.shape}")
                                else:
                                    # Single tensor - this shouldn't happen with proper config
                                    logger.warning(f"Worker {worker_id} SDXL model returned single tensor instead of tuple")
                                    positive_embeds = embeddings_result
                                    gen_args['prompt_embeds'] = positive_embeds
                            else:
                                # Single encoder models
                                positive_embeds = embeddings_result
                                gen_args['prompt_embeds'] = positive_embeds
                            
                            # Remove text prompt since we're using embeddings
                            gen_args.pop('prompt', None)
                            
                            # Process negative prompt (or create empty one with same processing)
                            # Important: If no negative prompt provided, we need to create embeddings
                            # that match the shape of the positive prompt embeddings
                            if negative_prompt and negative_prompt.strip():
                                negative_text = negative_prompt
                            else:
                                # Create empty negative prompt that will produce same-shaped embeddings
                                # For compel, we can use empty string but ensure it gets same processing
                                negative_text = ""
                            
                            logger.debug(f"Worker {worker_id} processing negative prompt: '{negative_text}' (empty={not bool(negative_text)})")
                            neg_embeddings_result = compel_proc(negative_text)
                            logger.debug(f"Worker {worker_id} negative embeddings result type: {type(neg_embeddings_result)}")
                            
                            # Initialize variables for negative embeddings
                            negative_embeds = None
                            negative_pooled = None
                            
                            if model_key in ["flux-schnell", "flux-dev", "chroma"]:
                                # FLUX-style models - compel returns regular embeddings, need to create pooled
                                negative_embeds = neg_embeddings_result
                                negative_pooled = neg_embeddings_result
                            elif model_uses_dual_clip(model_key):
                                # SDXL-style models should return tuple when properly configured
                                if isinstance(neg_embeddings_result, (tuple, list)) and len(neg_embeddings_result) == 2:
                                    # Tuple format: (negative_prompt_embeds, negative_pooled_prompt_embeds)
                                    negative_embeds, negative_pooled = neg_embeddings_result
                                    logger.debug(f"Worker {worker_id} SDXL negative returned tuple with shapes: embeds {negative_embeds.shape}, pooled {negative_pooled.shape}")
                                else:
                                    # Single tensor - this shouldn't happen with proper config
                                    logger.warning(f"Worker {worker_id} SDXL negative returned single tensor instead of tuple")
                                    negative_embeds = neg_embeddings_result
                            else:
                                # Single encoder models
                                negative_embeds = neg_embeddings_result
                            
                            # Now check if shapes match and fix if needed
                            if positive_embeds is not None and negative_embeds is not None:
                                if hasattr(positive_embeds, 'shape') and hasattr(negative_embeds, 'shape'):
                                    if positive_embeds.shape != negative_embeds.shape:
                                        logger.warning(f"Worker {worker_id} shape mismatch detected: positive {positive_embeds.shape} vs negative {negative_embeds.shape}")
                                        # Expand negative embeddings to match positive embeddings shape
                                        target_seq_len = positive_embeds.shape[1]
                                        current_seq_len = negative_embeds.shape[1]
                                        if target_seq_len > current_seq_len:
                                            # Repeat the last token embedding to pad to target length
                                            padding_needed = target_seq_len - current_seq_len
                                            last_token = negative_embeds[:, -1:, :]  # Shape: [1, 1, hidden_dim]
                                            padding = last_token.repeat(1, padding_needed, 1)  # Shape: [1, padding_needed, hidden_dim]
                                            negative_embeds = torch.cat([negative_embeds, padding], dim=1)
                                            logger.info(f"Worker {worker_id} padded negative embeddings from {current_seq_len} to {target_seq_len} tokens")
                                        elif target_seq_len < current_seq_len:
                                            # Truncate to match
                                            negative_embeds = negative_embeds[:, :target_seq_len, :]
                                            logger.info(f"Worker {worker_id} truncated negative embeddings from {current_seq_len} to {target_seq_len} tokens")
                            
                            # Now set the embeddings in gen_args
                            if model_key in ["flux-schnell", "flux-dev", "chroma"]:
                                gen_args['negative_prompt_embeds'] = negative_embeds
                                gen_args['negative_pooled_prompt_embeds'] = negative_pooled
                            elif model_uses_dual_clip(model_key):
                                gen_args['negative_prompt_embeds'] = negative_embeds
                                if negative_pooled is not None:
                                    gen_args['negative_pooled_prompt_embeds'] = negative_pooled
                            else:
                                gen_args['negative_prompt_embeds'] = negative_embeds
                            
                            # Remove text negative prompt since we're using embeddings
                            gen_args.pop('negative_prompt', None)
                            
                        logger.info(f"Worker {worker_id} compel processing complete - using embeddings")
                        
                        # Log final gen_args keys for debugging
                        embed_keys = [k for k in gen_args.keys() if 'embed' in k]
                        logger.info(f"Worker {worker_id} embedding keys: {embed_keys}")
                    except Exception as e:
                        logger.warning(f"Worker {worker_id} compel processing failed, falling back to text: {e}")
                        # Keep original text prompts on failure
                
                # Add num_images_per_prompt to gen_args (diffusers parameter name)
                gen_args['num_images_per_prompt'] = num_images
                
                # Final shutdown check before generation
                if shutdown_event.is_set():
                    logger.info(f"Worker {worker_id} received shutdown signal before generation, breaking")
                    break
                
                # Remove original prompts from gen_args before pipeline call (only keep for special cases)
                original_prompt = gen_args.pop('original_prompt', None)
                original_negative_prompt = gen_args.pop('original_negative_prompt', None)
                
                # Generate image
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if model_key == "stable-cascade":
                        # Special handling for Stable Cascade two-stage model
                        prior, decoder = pipeline
                        prior_output = prior(**gen_args, generator=generator, output_type="pt")
                        
                        # Handle text prompt vs embeddings for decoder
                        decoder_args = {
                            "image_embeddings": prior_output.image_embeddings,
                            "num_inference_steps": 10,  # Decoder uses fewer steps
                            "guidance_scale": 0.0,  # Decoder doesn't use guidance
                            "generator": generator,
                            "output_type": "pil"
                        }
                        
                        # Add prompt (text or embeddings) to decoder
                        if 'prompt_embeds' in gen_args:
                            # Use embeddings if available - but decoder might not support this
                            # Fall back to original prompt if we stored it
                            if original_prompt:
                                decoder_args["prompt"] = original_prompt
                            else:
                                decoder_args["prompt"] = "beautiful image"  # Fallback
                        else:
                            decoder_args["prompt"] = gen_args.get("prompt", "beautiful image")
                        
                        output = decoder(**decoder_args)
                    elif model_key in ['sana-sprint', 'sana-sprint-small']:
                        # Always pass intermediate_timesteps=None for Sana Sprint
                        gen_args['intermediate_timesteps'] = None
                        output = pipeline(**gen_args, generator=generator, output_type="pil")
                    else:
                        output = pipeline(**gen_args, generator=generator, output_type="pil")
                
                elapsed = time.time() - start_time
                
                # Convert all images to base64
                image_strings = []
                for image in output.images:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    image_strings.append(img_str)
                
                logger.info(f"Worker {worker_id} completed {len(image_strings)} images in {elapsed:.2f}s")
                
                # Send result back
                pipe.send((image_strings, elapsed, seed))
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                logger.error(traceback.format_exc())
                pipe.send(([None], 0, None))  # Return list to match expected format
        
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
        timeout = 300  # Increased for DiT models which take longer to initialize
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
                
                gen_args, seed, request_id, num_images, future = task
                
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
                    parent_conn.send((gen_args, seed, request_id, num_images))
                    
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

    async def generate(self, model_key: str, gen_args: Dict[str, Any], seed: int | None, request_id: str, num_images: int = 1) -> Tuple[List[str], float, int]:
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
        await self.model_queues[model_key].put((gen_args, seed, request_id, num_images, future))
        
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
        
        # Wait for processes to exit with shorter timeout
        for p in self.worker_processes:
            p.join(timeout=2)  # Reduced from 5 to 2 seconds
            if p.is_alive():
                logger.warning(f"Worker process {p.pid} did not exit gracefully, terminating")
                p.terminate()
                p.join(timeout=1)  # Give 1 second after terminate
                if p.is_alive():
                    logger.error(f"Worker process {p.pid} still alive after terminate, killing")
                    p.kill()
        
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
        resolution_info = get_model_resolution_info(key)
        models_info[key] = {
            "name": config.name,
            "description": config.description,
            "default_steps": config.default_steps,
            "default_guidance": config.default_guidance,
            "min_width": config.min_width,
            "max_width": config.max_width,
            "min_height": config.min_height,
            "max_height": config.max_height,
            "supports_guidance": config.supports_guidance,
            "supports_negative_prompt": config.supports_negative_prompt,
            "resolution_info": resolution_info,
            "resolution_constraints": config.resolution_constraints,
            "recommended_resolutions": config.recommended_resolutions,
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
        resolution_info = get_model_resolution_info(key)
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
            "supports_guidance": config.supports_guidance,
            "supports_negative_prompt": config.supports_negative_prompt,
            "resolution_info": resolution_info,
        }
    
    return JSONResponse(response)


@app.get("/api/models/{model_key}/resolution-info")
async def get_model_resolution_details(model_key: str):
    """Get detailed resolution information for a specific model."""
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    
    resolution_info = get_model_resolution_info(model_key)
    config = AVAILABLE_MODELS[model_key]
    
    return JSONResponse({
        "model": model_key,
        "name": config.name,
        "resolution_info": resolution_info,
        "description": config.description,
        "resolution_constraints": config.resolution_constraints,
        "recommended_resolutions": config.recommended_resolutions,
    })


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
        
        # Validate and adjust resolution
        adjusted_width, adjusted_height, resolution_warning = validate_and_adjust_resolution(
            image_input.model, image_input.width, image_input.height
        )
        
        # Log if resolution was adjusted
        if resolution_warning:
            logger.info(resolution_warning)
        
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
        
        # Prepare arguments with adjusted dimensions
        gen_args = {
            "prompt": image_input.prompt,
            "width": adjusted_width,
            "height": adjusted_height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
        
        # Add negative prompt if provided and model supports it
        if image_input.negative_prompt:
            gen_args["negative_prompt"] = image_input.negative_prompt
        
        # Add HunyuanDiT-specific parameters
        if image_input.model == "hunyuan":
            # Disable resolution binning to prevent aspect ratio changes
            gen_args["use_resolution_binning"] = False
            gen_args["original_size"] = (adjusted_width, adjusted_height)
            gen_args["target_size"] = (adjusted_width, adjusted_height)

        # Generate images
        num_images = image_input.num_images if image_input.num_images else 1
        image_strings, gen_time, used_seed = await shared_generator.generate(
            image_input.model, gen_args, image_input.seed, request_id, num_images
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
            
            dim_key = f"{adjusted_width}x{adjusted_height}"
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

        # Build response - maintain backward compatibility for single image
        if num_images == 1:
            response_data = {
                "image": image_strings[0],
                "gen_time": gen_time,
                "seed": used_seed,
                "width": adjusted_width,
                "height": adjusted_height,
                "model": image_input.model,
                "model_name": model_config.name,
            }
        else:
            # Multiple images response
            response_data = {
                "images": image_strings,
                "gen_time": gen_time,
                "seed": used_seed,
                "width": adjusted_width,
                "height": adjusted_height,
                "model": image_input.model,
                "model_name": model_config.name,
                "num_images": num_images,
            }
        
        # Add resolution warning if applicable
        if resolution_warning:
            response_data["resolution_warning"] = resolution_warning
            response_data["requested_width"] = image_input.width
            response_data["requested_height"] = image_input.height
        
        return JSONResponse(response_data)
        
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