# Lambda Image Studio

A high-performance, multi-GPU image generation server supporting multiple state-of-the-art diffusion models. Built with FastAPI and optimized for production workloads.

## Features

- **Multi-Model Support**: 15+ cutting-edge diffusion models including FLUX, PixArt, SDXL variants, and more
- **Multi-GPU Architecture**: Efficient parallel processing across multiple GPUs with worker pool pattern
- **Long Prompt Support**: Compel integration for CLIP-based models supporting prompts beyond 77 tokens
- **High Resolution**: Support for up to 4K resolution (model-dependent)
- **Web UI**: Clean, responsive interface with model-specific presets and advanced options
- **API Compatible**: OpenAI-compatible API endpoint for easy integration

## Model Rankings & Recommendations

Based on extensive testing, here are our model recommendations:

### 🏆 Top Tier - Exceptional Quality

#### 1. **PixArt-Sigma** ⭐⭐⭐⭐⭐
- **Strengths**: Exceptional detail, photorealism, accurate prompt following, supports up to 4K
- **Best for**: High-quality production images, detailed scenes, portraits
- **Recommended settings**: 30-50 steps, guidance 3.5-7.0
- **Resolution**: Excellent at all resolutions up to 4096×4096

#### 2. **FLUX.1-dev** ⭐⭐⭐⭐⭐
- **Strengths**: State-of-the-art quality, excellent prompt adherence, consistent results
- **Best for**: Professional work, complex prompts, artistic compositions
- **Recommended settings**: 50 steps, guidance 3.5
- **Resolution**: Optimal up to 2048×2048

### 🥈 Excellent Tier - Professional Quality

#### 3. **FLUX.1-schnell** ⭐⭐⭐⭐½
- **Strengths**: Fastest high-quality generation (4 steps), great for iteration
- **Best for**: Rapid prototyping, real-time applications
- **Trade-off**: Slightly less detail than FLUX-dev
- **Resolution**: Best at 1024×1024 to 1536×1536

#### 4. **Playground v2.5 (1024px Aesthetic)** ⭐⭐⭐⭐
- **Strengths**: Vibrant colors, artistic style, excellent composition
- **Best for**: Aesthetic imagery, social media content, artistic works
- **Recommended settings**: 50 steps, guidance 3.0

### 🥉 Specialized Models - Domain Excellence

#### 5. **Animagine XL 4.0** ⭐⭐⭐⭐
- **Strengths**: Best-in-class for anime/manga style
- **Best for**: Anime art, character design, manga illustrations
- **Note**: Specialized model - use only for anime content

#### 6. **DreamShaper XL** ⭐⭐⭐⭐
- **Strengths**: Balanced realism and artistic interpretation
- **Best for**: Fantasy art, concept designs, creative imagery

#### 7. **Juggernaut XL v9** ⭐⭐⭐½
- **Strengths**: Photorealistic humans, good general purpose
- **Best for**: Portraits, realistic scenes

#### 8. **RealVisXL V4.0** ⭐⭐⭐½
- **Strengths**: Strong photorealism, good detail
- **Best for**: Product photography, architectural visualization

### ⚡ Fast Generation Models

#### 9. **SDXL Base** ⭐⭐⭐
- **Strengths**: Reliable baseline, wide compatibility
- **Best for**: General purpose when speed matters

#### 10. **SDXL Turbo** ⭐⭐½
- **Strengths**: Ultra-fast (1-4 steps)
- **Trade-off**: Lower quality, less prompt adherence

### 🔬 Experimental/Special Purpose

#### 11. **Sana 1.5** ⭐⭐⭐
- **Note**: Linear multi-scale diffusion transformer
- **Status**: Good potential but needs optimization

#### 12. **HunyuanDiT** ⭐⭐½
- **Note**: Bilingual (English/Chinese) support
- **Limitation**: Strict resolution requirements

#### 13. **Lumina-Next 2B** ⭐⭐½
- **Note**: Next-DiT architecture, 2B parameters
- **Status**: Experimental quality

#### 14. **Stable Cascade** ⭐⭐
- **Note**: Two-stage generation
- **Trade-off**: Slower, mixed results

#### 15. **Chroma** ⭐½
- **Status**: Experimental, limited quality

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/lambdalabs/lambda-image-studio.git
cd lambda-image-studio

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Basic usage (defaults to 8000 port, all available GPUs)
python server.py

# Custom configuration
python server.py --host 0.0.0.0 --port 8080 --num-gpus 4

# Specific GPUs only
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python server.py
```

### Web Interface

Navigate to `http://localhost:8000` to access the web UI.

### API Usage

The server provides an OpenAI-compatible API endpoint:

```bash
curl -X POST "http://localhost:8000/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "model": "pixart",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance": 3.5,
    "num_images": 1
  }'
```

## API Reference

### POST /v1/images/generations

Generate images from text prompts.

**Request Body:**
```json
{
  "prompt": "string (required)",
  "model": "string (optional, default: flux-schnell)",
  "negative_prompt": "string (optional)",
  "width": "integer (optional, model-dependent)",
  "height": "integer (optional, model-dependent)",
  "steps": "integer (optional, model-dependent)",
  "guidance": "float (optional, 0.0 for models without CFG)",
  "seed": "integer (optional, random if not specified)",
  "num_images": "integer (optional, default: 1, max: 4)"
}
```

**Response:**
```json
{
  "created": 1234567890,
  "data": [
    {
      "url": "data:image/png;base64,..."
    }
  ]
}
```

## Model-Specific Notes

### FLUX Models
- Use T5 text encoder (512 token limit)
- No negative prompt support
- FLUX-schnell: No CFG (guidance = 0)
- FLUX-dev: Supports CFG (recommended 3.5)

### PixArt-Sigma
- Supports resolutions up to 4096×4096
- Best quality at 30-50 steps
- Excellent prompt following

### SDXL-based Models
- Support compel for long prompts
- Negative prompts supported
- Optimal at 1024×1024 base resolution

### HunyuanDiT
- Requires exact resolutions (see UI presets)
- Bilingual support (English/Chinese)
- Uses CLIP + T5 encoders

## Advanced Features

### Long Prompt Support (Compel)
CLIP-based models support extended prompts using the compel library:
- Prompt weighting: `"a (beautiful:1.5) sunset"`
- Prompt blending: `"cat AND dog"`
- Negative weights: `"forest (dark:-1.0)"`

### Multi-GPU Configuration
The server automatically distributes workers across available GPUs:
- One worker process per GPU
- Round-robin request distribution
- Automatic GPU detection

### Memory Requirements
- FLUX models: ~15GB VRAM
- PixArt-Sigma: ~12GB VRAM
- SDXL models: ~8GB VRAM
- Sana/HunyuanDiT: ~20GB VRAM

## Troubleshooting

### Out of Memory Errors
- Reduce batch size (num_images)
- Lower resolution
- Use a model with lower memory requirements

### Slow Generation
- Reduce number of steps
- Use faster models (FLUX-schnell, SDXL Turbo)
- Ensure GPU boost clocks are enabled

### Invalid Resolution Errors
- All dimensions must be divisible by 16
- Some models have specific resolution requirements (e.g., HunyuanDiT)
- Use the preset buttons in the UI for valid resolutions

## Architecture

The server uses a multi-process architecture for optimal GPU utilization:

```
Main Process (FastAPI)
    ├── Worker 0 (GPU 0)
    ├── Worker 1 (GPU 1)
    ├── Worker 2 (GPU 2)
    └── Worker N (GPU N)
```

Each worker:
- Loads model independently
- Processes requests synchronously
- Communicates via multiprocessing pipes
- Performs warmup on initialization

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- New models include configuration in `MODEL_CONFIGS`
- UI updates include cache-busting version updates
- Test across different resolutions and settings

## License

[Specify your license here]

## Acknowledgments

Built with:
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [FastAPI](https://fastapi.tiangolo.com/) for the API server
- [Compel](https://github.com/damian0815/compel) for prompt weighting
- Model weights from various sources (see individual model cards)