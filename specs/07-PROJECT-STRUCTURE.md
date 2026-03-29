# 07 - Project Structure and CLI Design

## Directory Layout

```
videoforge/
├── README.md
├── setup.py                       # Package installation
├── requirements.txt               # Python dependencies
├── requirements-rocm.txt          # ROCm-specific overrides
│
├── configs/                       # YAML configuration files
│   ├── data_pipeline.yaml         # Video ingestion settings
│   ├── scene_detection.yaml       # Scene detection tuning
│   ├── caption.yaml               # Captioning model + prompt config
│   ├── train_wan21_lora.yaml      # Training hyperparameters
│   ├── inference.yaml             # Generation settings
│   └── style_tags.yaml            # Show-specific style descriptors
│
├── scripts/                       # User-written scene scripts
│   ├── scene_001.yaml
│   └── scene_002.yaml
│
├── videoforge/                    # Main Python package
│   ├── __init__.py
│   ├── __main__.py                # Entry point: python -m videoforge
│   │
│   ├── data/                      # Data pipeline (Stage 02)
│   │   ├── __init__.py
│   │   ├── __main__.py            # python -m videoforge.data
│   │   ├── preprocess.py          # Video normalization
│   │   ├── scene_detect.py        # Scene boundary detection
│   │   ├── clip_extract.py        # Clip cutting
│   │   ├── clip_filter.py         # Quality filtering
│   │   └── clip_condition.py      # Resize, normalize
│   │
│   ├── caption/                   # Captioning pipeline (Stage 03)
│   │   ├── __init__.py
│   │   ├── __main__.py            # python -m videoforge.caption
│   │   ├── captioner.py           # VLM-based captioning
│   │   ├── enrichment.py          # Merge dialogue + style tags
│   │   ├── review.py              # Interactive caption review
│   │   └── export.py              # Export to .txt sidecar files
│   │
│   ├── train/                     # Training pipeline (Stage 04)
│   │   ├── __init__.py
│   │   ├── __main__.py            # python -m videoforge.train
│   │   ├── config_builder.py      # Generate kohya/OneTrainer configs
│   │   ├── cache_latents.py       # Pre-encode video latents
│   │   ├── cache_text.py          # Pre-encode text embeddings
│   │   └── launcher.py            # Launch training via accelerate
│   │
│   ├── generate/                  # Inference pipeline (Stage 05)
│   │   ├── __init__.py
│   │   ├── __main__.py            # python -m videoforge.generate
│   │   ├── script_parser.py       # Parse scene YAML scripts
│   │   ├── prompt_builder.py      # Build generation prompts
│   │   ├── generator.py           # Run diffusion inference
│   │   └── comfyui_api.py         # Optional ComfyUI API client
│   │
│   ├── postprocess/               # Post-processing (Stage 05 cont.)
│   │   ├── __init__.py
│   │   ├── interpolate.py         # Frame interpolation (RIFE)
│   │   ├── upscale.py             # Video upscaling (Real-ESRGAN)
│   │   └── stitch.py              # Clip concatenation
│   │
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       ├── config.py              # YAML config loading
│       ├── video.py               # FFmpeg wrappers
│       ├── vram.py                # VRAM monitoring helpers
│       └── rocm.py                # ROCm environment validation
│
├── models/                        # Downloaded model weights (gitignored)
│   ├── wan21-1.3b/
│   └── qwen2-vl-7b/
│
├── dataset/                       # Processed training data (gitignored)
│   ├── clips/
│   ├── clip_metadata/
│   ├── subtitles/
│   ├── latent_cache/
│   └── te_cache/
│
├── output/                        # Training outputs (gitignored)
│   └── wan21_lora/
│       ├── wan21_show_style.safetensors
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── samples/
│
├── generated/                     # Generated videos (gitignored)
│   ├── scene_001/
│   └── scene_001_final.mp4
│
└── tests/                         # Test suite
    ├── test_data_pipeline.py
    ├── test_captioning.py
    ├── test_prompt_builder.py
    └── test_video_utils.py
```

## CLI Interface

All operations use a consistent CLI pattern:

```bash
# Top-level help
python -m videoforge --help

# Each subcommand maps to a pipeline stage
python -m videoforge data       [args]   # Data pipeline
python -m videoforge caption    [args]   # Captioning
python -m videoforge train      [args]   # Training
python -m videoforge generate   [args]   # Inference
python -m videoforge postprocess [args]  # Post-processing
python -m videoforge validate   [args]   # Environment validation
```

### Common Arguments

```
--config PATH      YAML config file (overridable by CLI args)
--verbose / -v     Verbose output
--dry-run          Show what would be done without executing
--device           Override device (default: cuda:0)
```

### Entry Point Design

```python
# videoforge/__main__.py
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(prog="videoforge")
    subparsers = parser.add_subparsers(dest="command")

    # Register subcommands
    sub_data = subparsers.add_parser("data", help="Video ingestion and clip extraction")
    sub_caption = subparsers.add_parser("caption", help="Auto-caption training clips")
    sub_train = subparsers.add_parser("train", help="LoRA fine-tuning")
    sub_generate = subparsers.add_parser("generate", help="Generate video from script")
    sub_postprocess = subparsers.add_parser("postprocess", help="Upscale, interpolate, stitch")
    sub_validate = subparsers.add_parser("validate", help="Validate environment setup")

    # Each subcommand adds its own arguments
    # ... (see individual pipeline docs)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to subcommand handler
    handlers = {
        "data": run_data_pipeline,
        "caption": run_captioning,
        "train": run_training,
        "generate": run_generation,
        "postprocess": run_postprocessing,
        "validate": run_validation,
    }
    handlers[args.command](args)

if __name__ == "__main__":
    main()
```

## Configuration Schema

### Master Config (Optional)

A single config that references all pipeline stages:

```yaml
# configs/master.yaml
project:
  name: "my_show_training"
  base_dir: "/home/chuck/videoforge"

environment:
  device: "cuda:0"
  dtype: float16
  rocm_gfx_override: "10.3.0"

data:
  config: "configs/data_pipeline.yaml"
  source_dir: "/data/videos/my_show/"
  dataset_dir: "${project.base_dir}/dataset/"

caption:
  config: "configs/caption.yaml"
  model: "Qwen/Qwen2-VL-7B-Instruct"

train:
  config: "configs/train_wan21_lora.yaml"
  output_dir: "${project.base_dir}/output/"

inference:
  config: "configs/inference.yaml"
  output_dir: "${project.base_dir}/generated/"
```

## Environment Validation

A diagnostic tool to verify the system is ready:

```bash
python -m videoforge validate
```

```
VideoForge Environment Check
═══════════════════════════════════════
✓ Python 3.10.12
✓ Ubuntu 22.04.4 LTS
✓ ROCm 6.3.0 detected
✓ HSA_OVERRIDE_GFX_VERSION=10.3.0
✓ GPU: AMD Radeon RX 6800 XT (16 GB)
✓ PyTorch 2.4.0+rocm6.2
✓ torch.cuda.is_available() = True
✓ VRAM: 16.0 GB total, 15.2 GB free
✓ RAM: 64.0 GB total, 58.3 GB free
✓ Disk: 450 GB free on /home
✓ FFmpeg 6.1.1 available
✓ accelerate 0.33.0
✓ diffusers 0.30.0
✓ transformers 4.44.0
✓ bitsandbytes 0.49.2 (ROCm)
✓ scenedetect 0.6.4
✗ xformers: NOT AVAILABLE (expected on AMD, using SDPA)
═══════════════════════════════════════
Environment is ready.
```

## .gitignore

```gitignore
# Large files
models/
dataset/
output/
generated/

# Cache
__pycache__/
*.pyc
latent_cache/
te_cache/

# Environment
.env
videoforge-env/
*.egg-info/

# System
.DS_Store
*.swp
```

## Dependencies

### requirements.txt

```
# Core ML
torch>=2.4.0
torchvision>=0.19.0
torchaudio>=2.4.0
accelerate>=0.33.0
transformers>=4.44.0
diffusers>=0.30.0
peft>=0.12.0
safetensors>=0.4.0

# Quantization
bitsandbytes>=0.49.0

# Video processing
opencv-python-headless>=4.8.0
ffmpeg-python>=0.2.0
scenedetect[opencv]>=0.6.4

# Captioning
qwen-vl-utils>=0.0.8

# Utilities
pyyaml>=6.0
tqdm>=4.66.0
pillow>=10.0.0
einops>=0.7.0
numpy>=1.24.0
```

### requirements-rocm.txt

```
# ROCm-specific PyTorch install
# Run INSTEAD of the torch lines in requirements.txt
--index-url https://download.pytorch.org/whl/rocm6.2
torch>=2.4.0
torchvision>=0.19.0
torchaudio>=2.4.0
```

## Development Workflow

```bash
# Initial setup
git clone <repo-url> videoforge
cd videoforge
python3.10 -m venv ~/videoforge-env
source ~/videoforge-env/bin/activate
pip install -r requirements-rocm.txt
pip install -r requirements.txt
pip install -e .

# Verify
python -m videoforge validate

# Process videos
python -m videoforge data --config configs/data_pipeline.yaml --input /data/videos/

# Caption clips
python -m videoforge caption --config configs/caption.yaml

# Train
python -m videoforge train --config configs/train_wan21_lora.yaml

# Generate
python -m videoforge generate --script scripts/scene_001.yaml --config configs/inference.yaml
```
