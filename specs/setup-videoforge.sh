#!/usr/bin/env bash
# ============================================================================
# VideoForge Setup Script
# ============================================================================
# Sets up the complete VideoForge environment on Ubuntu Server with AMD ROCm.
# Assumes ROCm is already installed and working (validated via llama.cpp, etc.)
#
# Usage:
#   chmod +x setup-videoforge.sh
#   ./setup-videoforge.sh
#
# Optional environment variables:
#   VIDEOFORGE_DIR    - Project root (default: ~/videoforge)
#   VENV_DIR          - Virtual environment (default: ~/videoforge-env)
#   MODELS_DIR        - Model storage (default: $VIDEOFORGE_DIR/models)
#   SKIP_MODELS       - Set to 1 to skip model downloads
#   SKIP_COMFYUI      - Set to 1 to skip ComfyUI installation
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEOFORGE_DIR="${VIDEOFORGE_DIR:-$HOME/videoforge}"
VENV_DIR="${VENV_DIR:-$HOME/videoforge-env}"
MODELS_DIR="${MODELS_DIR:-$VIDEOFORGE_DIR/models}"
SKIP_MODELS="${SKIP_MODELS:-0}"
SKIP_COMFYUI="${SKIP_COMFYUI:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

log()   { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }
info()  { echo -e "${CYAN}[i]${NC} $*"; }
header(){ echo -e "\n${BOLD}═══════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}═══════════════════════════════════════${NC}"; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
header "Pre-flight Checks"

# Must not be root
if [ "$EUID" -eq 0 ]; then
    err "Do not run this script as root. Run as your normal user."
    exit 1
fi

# Check Ubuntu
if [ -f /etc/os-release ]; then
    . /etc/os-release
    log "OS: $PRETTY_NAME"
else
    warn "Could not detect OS version"
fi

# Check ROCm
if [ -d /opt/rocm ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
    log "ROCm: $ROCM_VERSION"
else
    err "ROCm not found at /opt/rocm. Install ROCm first."
    exit 1
fi

# Check GPU visibility
if command -v rocminfo &>/dev/null; then
    GPU_NAME=$(rocminfo 2>/dev/null | grep -m1 "Marketing Name" | sed 's/.*: *//' || echo "unknown")
    GPU_GFX=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | sed 's/.*Name: *//' || echo "unknown")
    log "GPU: $GPU_NAME ($GPU_GFX)"
else
    warn "rocminfo not found -- cannot verify GPU"
fi

# Check user groups
if id -nG "$USER" | grep -qw video && id -nG "$USER" | grep -qw render; then
    log "User groups: video, render -- OK"
else
    warn "User may not be in 'video' and 'render' groups."
    warn "Run: sudo usermod -aG video,render $USER  (then re-login)"
fi

# Check disk space (want at least 100GB free)
AVAIL_GB=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | tr -d 'G')
log "Disk available: ${AVAIL_GB}GB on $HOME"
if [ "$AVAIL_GB" -lt 100 ]; then
    warn "Less than 100GB free. Model downloads + training data need significant space."
    warn "Recommended: 500GB+ free. Continuing anyway..."
fi

# Check RAM
TOTAL_RAM_GB=$(free -g | awk '/Mem:/ {print $2}')
log "RAM: ${TOTAL_RAM_GB}GB"

# ---------------------------------------------------------------------------
# Environment variables for RDNA2
# ---------------------------------------------------------------------------
header "ROCm Environment Variables"

BASHRC="$HOME/.bashrc"
ENVVARS_MARKER="# === VideoForge ROCm Environment ==="

# Check if we already added these
if grep -q "$ENVVARS_MARKER" "$BASHRC" 2>/dev/null; then
    log "ROCm environment variables already in .bashrc -- skipping"
else
    log "Adding ROCm environment variables to .bashrc"
    cat >> "$BASHRC" << 'ENVBLOCK'

# === VideoForge ROCm Environment ===
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# === End VideoForge ===
ENVBLOCK
    info "Added HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES, PYTORCH_HIP_ALLOC_CONF"
fi

# Source them now for this session
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
log "Environment variables active for this session"

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
header "System Dependencies"

NEEDED_PKGS=()

# Python -- prefer 3.10, accept 3.11
PYTHON_CMD=""
for pyver in python3.10 python3.11 python3; do
    if command -v "$pyver" &>/dev/null; then
        PY_VERSION=$("$pyver" --version 2>&1 | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -le 12 ]; then
            PYTHON_CMD="$pyver"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    warn "Python 3.10-3.12 not found. Will install python3.10."
    NEEDED_PKGS+=(python3.10 python3.10-venv python3.10-dev)
    PYTHON_CMD="python3.10"
else
    log "Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
    # Ensure venv and dev packages are available
    PY_MINOR=$("$PYTHON_CMD" -c 'import sys; print(sys.version_info.minor)')
    if ! "$PYTHON_CMD" -m venv --help &>/dev/null 2>&1; then
        NEEDED_PKGS+=("python3.${PY_MINOR}-venv")
    fi
    # Check for dev headers (needed for some pip builds)
    if ! dpkg -l "python3.${PY_MINOR}-dev" &>/dev/null 2>&1; then
        NEEDED_PKGS+=("python3.${PY_MINOR}-dev")
    fi
fi

# FFmpeg
if command -v ffmpeg &>/dev/null; then
    log "FFmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    NEEDED_PKGS+=(ffmpeg)
fi

# git
if ! command -v git &>/dev/null; then
    NEEDED_PKGS+=(git)
fi

# git-lfs (needed for HuggingFace model downloads)
if ! command -v git-lfs &>/dev/null; then
    NEEDED_PKGS+=(git-lfs)
fi

# Build essentials (some pip packages compile C extensions)
if ! dpkg -l build-essential &>/dev/null 2>&1; then
    NEEDED_PKGS+=(build-essential)
fi

# libgl (needed by OpenCV even in headless mode on some systems)
if ! dpkg -l libgl1 &>/dev/null 2>&1; then
    NEEDED_PKGS+=(libgl1 libglib2.0-0)
fi

if [ ${#NEEDED_PKGS[@]} -gt 0 ]; then
    log "Installing system packages: ${NEEDED_PKGS[*]}"
    sudo apt update
    sudo apt install -y "${NEEDED_PKGS[@]}"
    # Initialize git-lfs if just installed
    if [[ " ${NEEDED_PKGS[*]} " =~ " git-lfs " ]]; then
        git lfs install
    fi
else
    log "All system packages present"
fi

# Whisper needs rust compiler for tokenizers in some cases
if ! command -v rustc &>/dev/null; then
    info "Rust compiler not found. If pip builds fail later, install via:"
    info "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

# ---------------------------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------------------------
header "Python Virtual Environment"

if [ -d "$VENV_DIR" ]; then
    log "Virtual environment already exists at $VENV_DIR"
    info "To recreate: rm -rf $VENV_DIR && re-run this script"
else
    log "Creating virtual environment at $VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "Activated: $(python --version) at $(which python)"

# Upgrade pip
log "Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel 2>&1 | tail -1

# ---------------------------------------------------------------------------
# PyTorch with ROCm
# ---------------------------------------------------------------------------
header "PyTorch + ROCm"

# Detect which ROCm wheel index to use based on ROCm version
ROCM_MAJOR=$(echo "$ROCM_VERSION" | cut -d. -f1)
ROCM_MINOR=$(echo "$ROCM_VERSION" | cut -d. -f2)

# Map ROCm version to available PyTorch wheel index
# PyTorch provides: rocm6.0, rocm6.1, rocm6.2, rocm6.3
if [ "$ROCM_MAJOR" -eq 6 ]; then
    if [ "$ROCM_MINOR" -ge 3 ]; then
        ROCM_WHEEL="rocm6.3"
    elif [ "$ROCM_MINOR" -ge 2 ]; then
        ROCM_WHEEL="rocm6.2"
    elif [ "$ROCM_MINOR" -ge 1 ]; then
        ROCM_WHEEL="rocm6.1"
    else
        ROCM_WHEEL="rocm6.0"
    fi
elif [ "$ROCM_MAJOR" -ge 7 ]; then
    # ROCm 7.x -- try latest available wheel
    ROCM_WHEEL="rocm6.3"
    warn "ROCm $ROCM_VERSION detected. PyTorch may not have matching wheels yet."
    warn "Using $ROCM_WHEEL wheels -- this usually works."
else
    ROCM_WHEEL="rocm6.2"
    warn "Unexpected ROCm version $ROCM_VERSION, defaulting to $ROCM_WHEEL wheels"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${ROCM_WHEEL}"
log "Using PyTorch wheel index: $TORCH_INDEX"

# Check if torch is already installed with ROCm
TORCH_OK=0
if python -c "import torch; assert 'rocm' in torch.__version__ or torch.cuda.is_available()" 2>/dev/null; then
    EXISTING_TORCH=$(python -c "import torch; print(torch.__version__)")
    log "PyTorch already installed: $EXISTING_TORCH"
    TORCH_OK=1
fi

if [ "$TORCH_OK" -eq 0 ]; then
    log "Installing PyTorch with ROCm support..."
    pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" 2>&1 | tail -3
fi

# Validate
log "Validating PyTorch + GPU..."
python << 'PYTEST'
import torch
import sys

print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA/ROCm available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("  FATAL: torch.cuda.is_available() returned False!")
    print("  Check: HSA_OVERRIDE_GFX_VERSION, ROCm install, user groups")
    sys.exit(1)

print(f"  Device: {torch.cuda.get_device_name(0)}")
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"  VRAM: {vram_gb:.1f} GB")

# Quick compute test
x = torch.randn(512, 512, device='cuda')
y = x @ x.T
assert y.shape == (512, 512), "Compute test failed"
print("  Compute test: PASSED")

# SDPA test (our replacement for xformers)
q = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
assert out.shape == (1, 8, 64, 64), "SDPA test failed"
print("  SDPA attention: PASSED")
PYTEST

if [ $? -ne 0 ]; then
    err "PyTorch GPU validation failed. Fix ROCm/GPU issues before continuing."
    exit 1
fi

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
header "Python Dependencies"

log "Installing core ML libraries..."
pip install \
    accelerate \
    transformers \
    diffusers \
    peft \
    safetensors \
    sentencepiece \
    protobuf \
    2>&1 | tail -1

log "Installing bitsandbytes (ROCm quantization)..."
pip install bitsandbytes 2>&1 | tail -1

# Validate bitsandbytes ROCm support
log "Validating bitsandbytes..."
BNB_OK=0
python << 'BNBTEST' && BNB_OK=1
import bitsandbytes as bnb
import torch

# Quick test: create an 8-bit optimizer
param = torch.nn.Parameter(torch.randn(64, 64, device='cuda'))
opt = bnb.optim.AdamW8bit([param], lr=1e-3)
loss = param.sum()
loss.backward()
opt.step()
print(f"  bitsandbytes version: {bnb.__version__}")
print("  8-bit optimizer: PASSED")
BNBTEST

if [ "$BNB_OK" -eq 0 ]; then
    warn "bitsandbytes 8-bit optimizer test failed on ROCm."
    warn "Training will fall back to standard AdamW (uses more VRAM)."
    warn "This is a known issue on some RDNA2 setups."
fi

log "Installing video processing libraries..."
pip install \
    opencv-python-headless \
    ffmpeg-python \
    "scenedetect[opencv]" \
    2>&1 | tail -1

log "Installing captioning dependencies..."
pip install \
    qwen-vl-utils \
    2>&1 | tail -1

log "Installing speech-to-text (Whisper) for subtitle extraction..."
pip install openai-whisper 2>&1 | tail -1

log "Installing utilities..."
pip install \
    pyyaml \
    tqdm \
    pillow \
    einops \
    omegaconf \
    numpy \
    tensorboard \
    2>&1 | tail -1

# ---------------------------------------------------------------------------
# Training frameworks
# ---------------------------------------------------------------------------
header "Training Frameworks"

FRAMEWORKS_DIR="$VIDEOFORGE_DIR/frameworks"
mkdir -p "$FRAMEWORKS_DIR"

# --- kohya-ss/sd-scripts ---
KOHYA_DIR="$FRAMEWORKS_DIR/sd-scripts"
if [ -d "$KOHYA_DIR" ]; then
    log "kohya-ss/sd-scripts already cloned"
    info "To update: cd $KOHYA_DIR && git pull"
else
    log "Cloning kohya-ss/sd-scripts..."
    git clone https://github.com/kohya-ss/sd-scripts.git "$KOHYA_DIR"
fi

log "Installing kohya-ss dependencies..."
# Install kohya requirements but skip torch (we already have ROCm torch)
if [ -f "$KOHYA_DIR/requirements.txt" ]; then
    pip install -r "$KOHYA_DIR/requirements.txt" --no-deps 2>&1 | tail -1
    # Re-install with deps but exclude torch to avoid overwriting ROCm version
    grep -v -i "^torch" "$KOHYA_DIR/requirements.txt" > /tmp/kohya_reqs_filtered.txt || true
    pip install -r /tmp/kohya_reqs_filtered.txt 2>&1 | tail -1
    rm -f /tmp/kohya_reqs_filtered.txt
fi

# --- OneTrainer ---
ONETRAINER_DIR="$FRAMEWORKS_DIR/OneTrainer"
if [ -d "$ONETRAINER_DIR" ]; then
    log "OneTrainer already cloned"
    info "To update: cd $ONETRAINER_DIR && git pull"
else
    log "Cloning OneTrainer..."
    git clone https://github.com/Nerogar/OneTrainer.git "$ONETRAINER_DIR"
fi

info "OneTrainer has its own install script. Run when ready:"
info "  cd $ONETRAINER_DIR && ./install.sh"

# ---------------------------------------------------------------------------
# ComfyUI (inference UI)
# ---------------------------------------------------------------------------
if [ "$SKIP_COMFYUI" -eq 1 ]; then
    info "Skipping ComfyUI installation (SKIP_COMFYUI=1)"
else
    header "ComfyUI (Inference)"

    COMFYUI_DIR="$VIDEOFORGE_DIR/comfyui"
    if [ -d "$COMFYUI_DIR" ]; then
        log "ComfyUI already cloned"
    else
        log "Cloning ComfyUI..."
        git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    fi

    log "Installing ComfyUI dependencies..."
    if [ -f "$COMFYUI_DIR/requirements.txt" ]; then
        grep -v -i "^torch" "$COMFYUI_DIR/requirements.txt" > /tmp/comfy_reqs_filtered.txt || true
        pip install -r /tmp/comfy_reqs_filtered.txt 2>&1 | tail -1
        rm -f /tmp/comfy_reqs_filtered.txt
    fi

    # Install useful custom nodes
    CUSTOM_NODES="$COMFYUI_DIR/custom_nodes"
    mkdir -p "$CUSTOM_NODES"

    if [ ! -d "$CUSTOM_NODES/ComfyUI-VideoHelperSuite" ]; then
        log "Installing ComfyUI-VideoHelperSuite..."
        git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
            "$CUSTOM_NODES/ComfyUI-VideoHelperSuite"
        if [ -f "$CUSTOM_NODES/ComfyUI-VideoHelperSuite/requirements.txt" ]; then
            pip install -r "$CUSTOM_NODES/ComfyUI-VideoHelperSuite/requirements.txt" 2>&1 | tail -1
        fi
    fi

    if [ ! -d "$CUSTOM_NODES/ComfyUI-WanVideoWrapper" ]; then
        log "Installing ComfyUI-WanVideoWrapper..."
        git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
            "$CUSTOM_NODES/ComfyUI-WanVideoWrapper"
        if [ -f "$CUSTOM_NODES/ComfyUI-WanVideoWrapper/requirements.txt" ]; then
            pip install -r "$CUSTOM_NODES/ComfyUI-WanVideoWrapper/requirements.txt" 2>&1 | tail -1
        fi
    fi

    info "Start ComfyUI with:"
    info "  cd $COMFYUI_DIR && python main.py --listen 0.0.0.0 --port 8188"
fi

# ---------------------------------------------------------------------------
# Project directory structure
# ---------------------------------------------------------------------------
header "Project Structure"

log "Creating VideoForge directory tree at $VIDEOFORGE_DIR"

mkdir -p "$VIDEOFORGE_DIR"/{configs,scripts,dataset/{clips,clip_metadata,subtitles,latent_cache,te_cache},output,generated,tests}
mkdir -p "$MODELS_DIR"

# Create placeholder configs
if [ ! -f "$VIDEOFORGE_DIR/configs/data_pipeline.yaml" ]; then
cat > "$VIDEOFORGE_DIR/configs/data_pipeline.yaml" << 'YAML'
# VideoForge Data Pipeline Configuration
data:
  source_dir: "/path/to/your/videos"
  dataset_dir: "./dataset"

preprocessing:
  target_fps: 24
  video_codec: libx264
  crf: 18

scene_detection:
  detector: content
  threshold: 27
  min_scene_length_sec: 1.0
  max_scene_length_sec: 30.0

clip_extraction:
  target_duration_sec: 4.0
  min_duration_sec: 1.0
  max_duration_sec: 8.0
  overlap_sec: 0.5

filtering:
  black_frame_threshold: 0.85
  min_optical_flow: 0.5
  max_optical_flow: 50.0
  text_overlay_threshold: 0.3

conditioning:
  target_width: 848
  target_height: 480
  target_fps: 24
  target_frames: 49
YAML
log "  configs/data_pipeline.yaml"
fi

if [ ! -f "$VIDEOFORGE_DIR/configs/caption.yaml" ]; then
cat > "$VIDEOFORGE_DIR/configs/caption.yaml" << 'YAML'
# VideoForge Captioning Configuration
captioning:
  model: "Qwen/Qwen2-VL-7B-Instruct"
  quantization: 4bit
  dtype: float16
  max_new_tokens: 300
  frames_per_clip: 8

  prompt: |
    Describe this video clip in detail for training a video generation model. Include:
    1. SCENE: Setting, location, time of day, lighting conditions
    2. SUBJECTS: People present, their appearance, clothing, positioning
    3. ACTION: What is happening, movements, gestures, expressions
    4. CAMERA: Camera angle, movement (static, pan, zoom, tracking)
    5. STYLE: Color palette, mood, visual style (cinematic, bright, dark, etc.)
    Write a single flowing paragraph, not a bulleted list. Be specific and visual.
    Do not describe audio or make assumptions about what cannot be seen.

style_tags:
  - "cinematic television scene"
  # Add your show-specific tags here:
  # - "dramatic lighting"
  # - "warm color palette"

dataset_dir: "./dataset"
YAML
log "  configs/caption.yaml"
fi

if [ ! -f "$VIDEOFORGE_DIR/configs/train_wan21_lora.yaml" ]; then
cat > "$VIDEOFORGE_DIR/configs/train_wan21_lora.yaml" << 'YAML'
# VideoForge Training Configuration - Wan 2.1 1.3B LoRA
model:
  name: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  type: text_to_video
  dtype: float16

lora:
  rank: 32
  alpha: 32
  target_modules:
    - "to_q"
    - "to_k"
    - "to_v"
    - "to_out.0"
    - "ff.net.0.proj"
    - "ff.net.2"
  dropout: 0.05

training:
  batch_size: 1
  gradient_accumulation: 4
  gradient_checkpointing: true
  learning_rate: 1.0e-4
  lr_scheduler: cosine
  lr_warmup_steps: 100
  max_train_steps: 3000
  mixed_precision: fp16
  seed: 42
  cache_latents_to_disk: true
  cache_text_encoder_outputs: true
  cpu_offload_text_encoder: true

optimizer:
  name: adamw8bit
  weight_decay: 0.01
  betas: [0.9, 0.999]

dataset:
  path: "./dataset/clips"
  caption_extension: ".txt"
  video_extension: ".mp4"
  resolution: [848, 480]
  frame_count: 49
  fps: 24
  caption_dropout_rate: 0.1

saving:
  output_dir: "./output/wan21_lora"
  save_every_n_steps: 500
  keep_last_n_checkpoints: 3

sampling:
  enabled: true
  every_n_steps: 500
  prompts:
    - "cinematic television scene. A person sits in a dimly lit room, looking contemplative."
    - "cinematic television scene. Two people stand in an office, having an intense conversation."
  num_frames: 49
  width: 848
  height: 480
  guidance_scale: 7.5
  num_inference_steps: 30
YAML
log "  configs/train_wan21_lora.yaml"
fi

if [ ! -f "$VIDEOFORGE_DIR/configs/inference.yaml" ]; then
cat > "$VIDEOFORGE_DIR/configs/inference.yaml" << 'YAML'
# VideoForge Inference Configuration
inference:
  model: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  lora_path: "./output/wan21_lora/wan21_show_style.safetensors"
  lora_weight: 0.8

  width: 848
  height: 480
  fps: 24
  num_frames: 49

  guidance_scale: 6.0
  num_inference_steps: 40
  scheduler: euler_a

  negative_prompt: >
    blurry, low quality, distorted, watermark, text overlay,
    oversaturated, cartoon, anime, 3d render, static image

  output_dir: "./generated"
  output_format: mp4

postprocessing:
  interpolation: false
  upscale: false
  upscale_factor: 2
YAML
log "  configs/inference.yaml"
fi

if [ ! -f "$VIDEOFORGE_DIR/configs/style_tags.yaml" ]; then
cat > "$VIDEOFORGE_DIR/configs/style_tags.yaml" << 'YAML'
# Style tags are prepended to every caption during training and generation.
# They act as trigger words that activate the learned visual style.
# Customize these to match your source material.
style_tags:
  - "cinematic television scene"
  # Examples -- uncomment/modify as appropriate:
  # - "dramatic lighting"
  # - "warm amber tones"
  # - "handheld camera"
  # - "noir style"
  # - "bright sitcom lighting"
  # - "anamorphic widescreen"
YAML
log "  configs/style_tags.yaml"
fi

# Example scene script
if [ ! -f "$VIDEOFORGE_DIR/scripts/example_scene.yaml" ]; then
cat > "$VIDEOFORGE_DIR/scripts/example_scene.yaml" << 'YAML'
# Example scene script for VideoForge generation
# Each clip becomes a separate video generation call
scene:
  title: "Example Scene"
  clips:
    - description: >
        Interior, dimly lit living room, evening. A woman with dark hair
        wearing a blue cardigan sits on a brown leather couch, leaning forward
        with her hands clasped, looking tense. Static camera, medium shot.
      dialogue: "I know what you did."
      duration: 4

    - description: >
        Same room. A man in a grey suit stands near the window, his back
        partially turned. He slowly turns to face the camera with a conflicted
        expression. Static camera with slight push in.
      dialogue: "It's not what you think."
      duration: 4

    - description: >
        Close-up of the woman's face. Her expression shifts from anger to
        hurt. Warm amber lighting from a table lamp highlights her features.
        Static camera, shallow depth of field.
      dialogue: null
      duration: 3
YAML
log "  scripts/example_scene.yaml"
fi

# .gitignore
if [ ! -f "$VIDEOFORGE_DIR/.gitignore" ]; then
cat > "$VIDEOFORGE_DIR/.gitignore" << 'GITIGNORE'
# Models and data (large files)
models/
dataset/
output/
generated/
frameworks/

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/

# Cache
latent_cache/
te_cache/

# Environment
.env
*-env/

# System
.DS_Store
*.swp
*.swo
*~
GITIGNORE
log "  .gitignore"
fi

# ---------------------------------------------------------------------------
# Model downloads
# ---------------------------------------------------------------------------
if [ "$SKIP_MODELS" -eq 1 ]; then
    info "Skipping model downloads (SKIP_MODELS=1)"
else
    header "Model Downloads"

    # Ensure huggingface-cli is available
    if ! command -v huggingface-cli &>/dev/null; then
        pip install huggingface-hub[cli] 2>&1 | tail -1
    fi

    # --- Wan 2.1 T2V 1.3B ---
    WAN_DIR="$MODELS_DIR/Wan2.1-T2V-1.3B-Diffusers"
    if [ -d "$WAN_DIR" ] && [ "$(ls -A "$WAN_DIR" 2>/dev/null)" ]; then
        log "Wan 2.1 1.3B already downloaded"
    else
        log "Downloading Wan 2.1 T2V 1.3B (~5GB)..."
        info "This may take a while depending on your connection."
        huggingface-cli download \
            Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
            --local-dir "$WAN_DIR" \
            --local-dir-use-symlinks False
    fi

    # --- Qwen2-VL-7B (captioning model) ---
    QWEN_DIR="$MODELS_DIR/Qwen2-VL-7B-Instruct"
    if [ -d "$QWEN_DIR" ] && [ "$(ls -A "$QWEN_DIR" 2>/dev/null)" ]; then
        log "Qwen2-VL-7B already downloaded"
    else
        log "Downloading Qwen2-VL-7B-Instruct (~15GB)..."
        info "This is the captioning model. It's large but only needed for data prep."
        huggingface-cli download \
            Qwen/Qwen2-VL-7B-Instruct \
            --local-dir "$QWEN_DIR" \
            --local-dir-use-symlinks False
    fi

    info "Models stored in: $MODELS_DIR"
fi

# ---------------------------------------------------------------------------
# Convenience scripts
# ---------------------------------------------------------------------------
header "Convenience Scripts"

# Activation script
cat > "$VIDEOFORGE_DIR/activate.sh" << ACTIVATE
#!/usr/bin/env bash
# Source this to activate the VideoForge environment:
#   source activate.sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
source "$VENV_DIR/bin/activate"
echo "VideoForge environment activated"
echo "  Python: \$(python --version)"
echo "  PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "  GPU: \$(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
ACTIVATE
chmod +x "$VIDEOFORGE_DIR/activate.sh"
log "activate.sh -- source this to enter the environment"

# Validation script
cat > "$VIDEOFORGE_DIR/validate.sh" << 'VALIDATE'
#!/usr/bin/env bash
# Run a full environment validation
set -e

echo "═══════════════════════════════════════"
echo "  VideoForge Environment Validation"
echo "═══════════════════════════════════════"

python << 'PYCHECK'
import sys
import shutil
import os

def check(label, test_fn):
    try:
        result = test_fn()
        print(f"  OK   {label}: {result}")
        return True
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        return False

results = []

# Python
results.append(check("Python", lambda: sys.version.split()[0]))

# OS
results.append(check("OS", lambda: open("/etc/os-release").read().split("PRETTY_NAME=")[1].split("\n")[0].strip('"')))

# ROCm
results.append(check("ROCm", lambda: open("/opt/rocm/.info/version").read().strip()))

# HSA override
results.append(check("HSA_OVERRIDE_GFX_VERSION", lambda: os.environ.get("HSA_OVERRIDE_GFX_VERSION", "NOT SET")))

# PyTorch
import torch
results.append(check("PyTorch", lambda: torch.__version__))
results.append(check("CUDA/ROCm available", lambda: str(torch.cuda.is_available())))
if torch.cuda.is_available():
    results.append(check("GPU", lambda: torch.cuda.get_device_name(0)))
    results.append(check("VRAM", lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"))

# Compute test
results.append(check("GPU compute", lambda: (
    torch.randn(256, 256, device='cuda') @ torch.randn(256, 256, device='cuda'),
    "PASSED"
)[1]))

# SDPA
results.append(check("SDPA attention", lambda: (
    torch.nn.functional.scaled_dot_product_attention(
        torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16),
        torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16),
        torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16),
    ),
    "PASSED"
)[1]))

# Key libraries
import importlib
for lib, pkg in [
    ("accelerate", "accelerate"),
    ("transformers", "transformers"),
    ("diffusers", "diffusers"),
    ("peft", "peft"),
    ("bitsandbytes", "bitsandbytes"),
    ("cv2", "opencv"),
    ("scenedetect", "scenedetect"),
]:
    results.append(check(pkg, lambda l=lib: importlib.import_module(l).__version__))

# FFmpeg
results.append(check("FFmpeg", lambda: (
    __import__("subprocess").check_output(["ffmpeg", "-version"], stderr=__import__("subprocess").STDOUT)
    .decode().split("\n")[0].split(" ")[2],
)))

# RAM
import psutil
results.append(check("RAM", lambda: f"{psutil.virtual_memory().total / 1024**3:.0f} GB"))

# Disk
disk = shutil.disk_usage(os.path.expanduser("~"))
results.append(check("Disk free", lambda: f"{disk.free / 1024**3:.0f} GB"))

# xformers (should NOT be available)
try:
    import xformers
    print(f"  WARN xformers: installed ({xformers.__version__}) -- unexpected on AMD, may cause issues")
except ImportError:
    print(f"  OK   xformers: not installed (expected on AMD, using SDPA)")

print("═══════════════════════════════════════")
passed = sum(results)
total = len(results)
if passed == total:
    print(f"  All {total} checks passed. Environment is ready.")
else:
    print(f"  {passed}/{total} checks passed. Review failures above.")
PYCHECK
VALIDATE
chmod +x "$VIDEOFORGE_DIR/validate.sh"
log "validate.sh -- run to check everything works"

# VRAM monitor script
cat > "$VIDEOFORGE_DIR/vram-monitor.sh" << 'VRAM'
#!/usr/bin/env bash
# Watch GPU VRAM usage (updates every 2 seconds)
watch -n 2 'cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | while read used; do
    total=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1)
    used_mb=$((used / 1024 / 1024))
    total_mb=$((total / 1024 / 1024))
    pct=$((used * 100 / total))
    echo "VRAM: ${used_mb}MB / ${total_mb}MB (${pct}%)"
done'
VRAM
chmod +x "$VIDEOFORGE_DIR/vram-monitor.sh"
log "vram-monitor.sh -- watch GPU memory usage"

# ComfyUI launcher
if [ "$SKIP_COMFYUI" -ne 1 ]; then
cat > "$VIDEOFORGE_DIR/start-comfyui.sh" << COMFY
#!/usr/bin/env bash
source "$VIDEOFORGE_DIR/activate.sh"
cd "$VIDEOFORGE_DIR/comfyui"
echo "Starting ComfyUI on http://0.0.0.0:8188"
echo "  Access from another machine: http://\$(hostname -I | awk '{print \$1}'):8188"
python main.py --listen 0.0.0.0 --port 8188
COMFY
chmod +x "$VIDEOFORGE_DIR/start-comfyui.sh"
log "start-comfyui.sh -- launch ComfyUI for inference"
fi

# Install psutil for validate.sh
pip install psutil 2>&1 | tail -1

# ---------------------------------------------------------------------------
# Record working versions
# ---------------------------------------------------------------------------
header "Version Lockfile"

python << 'VERSIONS' > "$VIDEOFORGE_DIR/VERSIONS.md"
import torch
import accelerate
import transformers
import diffusers
import peft
import platform
import os
import subprocess

rocm = open("/opt/rocm/.info/version").read().strip() if os.path.exists("/opt/rocm/.info/version") else "unknown"
ffmpeg = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT).decode().split("\n")[0].split(" ")[2]

try:
    import bitsandbytes as bnb
    bnb_ver = bnb.__version__
except:
    bnb_ver = "NOT INSTALLED"

print(f"""# VideoForge - Known Working Versions
# Generated by setup script. Pin these if things break after an update.

| Component | Version |
|-----------|---------|
| OS | {platform.platform()} |
| Python | {platform.python_version()} |
| ROCm | {rocm} |
| PyTorch | {torch.__version__} |
| accelerate | {accelerate.__version__} |
| transformers | {transformers.__version__} |
| diffusers | {diffusers.__version__} |
| peft | {peft.__version__} |
| bitsandbytes | {bnb_ver} |
| FFmpeg | {ffmpeg} |

## Critical Environment Variables
```
HSA_OVERRIDE_GFX_VERSION=10.3.0
HIP_VISIBLE_DEVICES=0
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```
""")
VERSIONS

log "Saved to $VIDEOFORGE_DIR/VERSIONS.md"
cat "$VIDEOFORGE_DIR/VERSIONS.md"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Setup Complete!"

echo ""
log "Project directory: $VIDEOFORGE_DIR"
log "Virtual environment: $VENV_DIR"
log "Models directory: $MODELS_DIR"
echo ""
info "Quick start:"
echo "  cd $VIDEOFORGE_DIR"
echo "  source activate.sh"
echo "  ./validate.sh"
echo ""
info "Next steps:"
echo "  1. Run ./validate.sh to confirm everything works"
echo "  2. Edit configs/data_pipeline.yaml with your video source path"
echo "  3. Edit configs/style_tags.yaml for your show's visual style"
echo "  4. Place source videos and start the data pipeline"
echo ""
info "Convenience scripts:"
echo "  source activate.sh    -- activate environment"
echo "  ./validate.sh         -- check all dependencies"
echo "  ./vram-monitor.sh     -- watch GPU memory"
if [ "$SKIP_COMFYUI" -ne 1 ]; then
echo "  ./start-comfyui.sh    -- launch ComfyUI inference UI"
fi
echo ""
info "Training frameworks installed at:"
echo "  kohya-ss:   $FRAMEWORKS_DIR/sd-scripts"
echo "  OneTrainer: $FRAMEWORKS_DIR/OneTrainer"
echo ""
warn "Remember: Always 'source activate.sh' before running any VideoForge commands."
