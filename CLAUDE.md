# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based image generation server supporting 15+ diffusion models with multi-GPU support. The server implements a worker pool pattern for efficient parallel image generation across multiple GPUs and includes a web UI for easy interaction.

## Common Development Commands

### Running the Server
```bash
# Basic run (localhost:8000)
python server.py

# With custom configuration
python server.py --host 0.0.0.0 --port 8080 --model "black-forest-labs/FLUX.1-schnell" --num-gpus 4

# Control visible GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python server.py
```

### Testing the API
```bash
curl -X POST "http://localhost:8000/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "width": 1024,
    "height": 768,
    "steps": 4,
    "guidance": 0.0,
    "seed": 42
  }'
```

### Environment Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## High-Level Architecture

### Core Components

1. **Multi-GPU Worker Architecture** (server.py:22-104)
   - Uses multiprocessing to spawn one worker per GPU
   - Each worker loads the diffusion model independently
   - Workers communicate via pipes with the main process
   - Implements warmup on initialization for optimal performance

2. **FastAPI Server** (server.py:106-197)
   - Single endpoint: `POST /v1/images/generations`
   - Accepts GenerateRequest with prompt, dimensions, steps, guidance, and seed
   - Returns base64-encoded PNG images in OpenAI-compatible format
   - Uses round-robin scheduling across GPU workers

3. **Performance Optimizations**
   - Channel-last memory format for transformer and VAE
   - QKV projection fusion enabled
   - High precision float32 matmul for Hopper+ GPUs
   - Bfloat16 precision for model weights

4. **Seed Handling**
   - Each worker initializes with a unique random seed based on worker ID and time
   - When no seed is specified, each request generates a unique random seed
   - Explicit seeds are respected for reproducible generation

### Key Design Patterns

1. **Worker Pool Pattern**: The server maintains a pool of GPU workers, each running in a separate process. This allows for:
   - True parallel execution across GPUs
   - Isolation of GPU memory per worker
   - Fault tolerance (one worker crash doesn't affect others)

2. **Async Request Handling**: FastAPI handles requests asynchronously while workers process synchronously, preventing blocking of the web server.

3. **Pipe-based IPC**: Workers communicate with the main process using multiprocessing pipes, ensuring efficient data transfer without shared memory complexities.

## Important Notes

- The project currently lacks tests, linting configuration, and CI/CD setup
- No Makefile or build scripts exist - all commands must be run directly
- The server respects `CUDA_VISIBLE_DEVICES` environment variable for GPU selection
- Default model requires significant GPU memory (~12GB per worker)
- Server implements proper startup synchronization - workers signal when ready
- Clean shutdown on SIGINT/SIGTERM prevents GPU resource leaks
- Uses event-based synchronization instead of polling loops