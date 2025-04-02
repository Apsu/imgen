import argparse
import asyncio
import base64
import io
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global configuration and state variables
device = torch.device("cuda")
server_config: Optional[argparse.Namespace] = None
shared_generator = None

# Request model definition
class TextToImageInput(BaseModel):
    prompt: str
    width: int = 512
    height: int = 384
    steps: int = 4
    guidance: float = 0.0

class DiffusionGenerator:
    """Diffusion generator abstraction using AutoPipelineForText2Image."""

    def __init__(self, model: str):
        # Set float32 matmul precision to high for better performance on CUDA
        torch.set_float32_matmul_precision("high")
        logger.info(f"Loading model {model}...")

        # Load model pipeline
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
        ).to(device)

        # Hide progress bar after generation for cleaner logs
        self.pipeline.set_progress_bar_config(leave=False)
        logger.info("Pipeline loaded successfully")

    def warmup(self, gen_args: Dict[str, Any], warmup_iterations: int = 1) -> None:
        """Pre-run the model to trigger compilation and warmup."""
        logger.info("Starting warmup...")
        start_time = time.time()

        for _ in range(warmup_iterations):
            _ = self.pipeline(**gen_args, output_type="pil")

        elapsed = time.time() - start_time
        logger.info(f"Warmup complete in {elapsed:.2f} seconds")

    def generate(self, gen_args: Dict[str, Any]) -> Tuple[str, float]:
        """Generate an image from the given arguments.

        Returns:
            Tuple containing base64 encoded image and generation time
        """
        start_time = time.time()

        # Perform the actual image generation
        output = self.pipeline(**gen_args, output_type="pil")
        elapsed = time.time() - start_time

        # Convert the generated image to a base64-encoded PNG
        buffered = io.BytesIO()
        output.images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str, elapsed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FastAPI server for diffusion image generation with xformers flash attention."
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
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown."""
    global shared_generator

    try:
        # Prepare dummy arguments for warmup
        warmup_args = {
            "prompt": "warmup",
            "width": 1024,
            "height": 768,
            "num_inference_steps": 1,
            "guidance_scale": 0.0,
        }

        # Initialize the diffusion generator
        shared_generator = DiffusionGenerator(model=server_config.model)
        shared_generator.warmup(warmup_args, warmup_iterations=1)
        yield
    finally:
        # Cleanup resources if needed
        logger.info("Shutting down server")


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.post("/v1/images/generations", response_class=JSONResponse)
async def generate_image(request: Request, image_input: TextToImageInput):
    """Generate an image based on the provided text input."""
    try:
        # Prepare arguments for the pipeline
        gen_args = {
            "prompt": image_input.prompt,
            "width": image_input.width,
            "height": image_input.height,
            "num_inference_steps": image_input.steps,
            "guidance_scale": image_input.guidance,
        }

        logger.info(f"Generating image with prompt: '{image_input.prompt[:50]}...' "
                    f"at {image_input.width}x{image_input.height} ({image_input.steps} @ {image_input.guidance})")

        # Run generation in a thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        img_str, gen_time = await loop.run_in_executor(
            None, lambda: shared_generator.generate(gen_args)
        )

        logger.info(f"Image generated in {gen_time:.2f}s")

        return JSONResponse({
            "image": img_str,
            "gen_time": gen_time
        })
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Parse command-line arguments and store in global server_config
    server_config = parse_args()

    # Run the server
    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level=logging.WARNING
    )
