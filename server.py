import argparse
import asyncio
import base64
import io
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager

import aiohttp
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global device (CUDA only)
device = torch.device("cuda")

# Global configuration variable.
server_config = None

# Set up Jinja2 templates (ensure you have a "templates" directory).
templates = Jinja2Templates(directory="templates")

def parse_args():
    parser = argparse.ArgumentParser(
        description="FastAPI server for diffusion image generation with xformers flash attention."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
        help="Model path or identifier.",
    )
    return parser.parse_args()

# Request payload for image generation.
class TextToImageInput(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    num_steps: int = 4
    guidance_scale: float = 0.0
    max_sequence_length: int = 256

# Simple HTTP client wrapper.
class HttpClient:
    session: aiohttp.ClientSession = None

    def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()
            self.session = None

    def __call__(self) -> aiohttp.ClientSession:
        assert self.session is not None
        return self.session

# Diffusion generator abstraction using AutoPipelineForText2Image.
class DiffusionGenerator:
    def __init__(self, model: str):
        # Set float32 matmul precision to high for better performance on CUDA.
        torch.set_float32_matmul_precision("high")
        logger.info(f"Loading model {model} ...")
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
        ).to(device)  # Load the model to the specified device (CUDA).

        # Hide progress bar after generation for cleaner logs.
        self.pipeline.set_progress_bar_config(leave=False)

        # Configure inductor settings for performance optimization.
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        # Set memory format to channels_last for better performance on CUDA.
        self.pipeline.transformer.to(memory_format=torch.channels_last)
        self.pipeline.vae.to(memory_format=torch.channels_last)

        logger.info("Compiling pipeline with torch.compile...")
        self.pipeline.transformer = torch.compile(
            self.pipeline.transformer, mode="max-autotune", fullgraph=True
        )
        self.pipeline.vae.decode = torch.compile(
            self.pipeline.vae.decode, mode="max-autotune", fullgraph=True
        )
        logger.info("Pipeline loaded and compiled successfully.")

    def warmup(self, gen_args: dict, warmup_iterations: int = 1):
        logger.info("Starting warmup...")
        start_time = time.time()
        # Run a dummy inference to trigger compilation and warmup.
        for _ in range(warmup_iterations):
            _ = self.pipeline(**gen_args, output_type="pil")
        elapsed = time.time() - start_time
        logger.info(f"Warmup complete in {elapsed:.2f} seconds.")

    def generate(self, gen_args: dict) -> tuple[str, float]:
        start_time = time.time()
        output = self.pipeline(**gen_args, output_type="pil")
        elapsed = time.time() - start_time
        # Convert the generated image to a base64-encoded PNG.
        buffered = io.BytesIO()
        output.images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str, elapsed

# Global variables for HTTP client and the diffusion generator.
http_client = HttpClient()
shared_generator: DiffusionGenerator = None

# Lifespan context manager to handle startup and shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global shared_generator
    http_client.start()
    # Prepare dummy arguments for warmup.
    warmup_args = {
        "prompt": "warmup",
        "width": 512,
        "height": 512,
        "num_inference_steps": 1,
        "guidance_scale": 0.0,
        "max_sequence_length": 256,
    }
    # Initialize the diffusion generator.
    shared_generator = DiffusionGenerator(model=server_config.model)
    shared_generator.warmup(warmup_args, warmup_iterations=1)
    yield
    await http_client.stop()

app = FastAPI(lifespan=lifespan)

# Serve the main page using a Jinja2 template.
@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Image generation endpoint.
@app.post("/v1/images/generations", response_class=HTMLResponse)
async def generate_image(request: Request, image_input: TextToImageInput):
    try:
        # Prepare arguments for the pipeline.
        gen_args = {
            "prompt": image_input.prompt,
            "width": image_input.width,
            "height": image_input.height,
            "num_inference_steps": image_input.num_steps,
            "guidance_scale": image_input.guidance_scale,
            "max_sequence_length": image_input.max_sequence_length,
        }
        loop = asyncio.get_event_loop()
        # Run the generation in a thread executor.
        img_str, gen_time = await loop.run_in_executor(
            None, lambda: shared_generator.generate(gen_args)
        )
        return templates.TemplateResponse(
            "result_fragment.html",
            {"request": request, "image": img_str, "gen_time": gen_time},
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Parse command-line arguments and store in global server_config.
    server_config = parse_args()
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level=logging.WARNING)
