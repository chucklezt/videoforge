# 01 - Hardware and Environment Setup

## Hardware Specifications

```
GPU:  AMD Radeon RX 6800 XT
      - 16GB GDDR6 VRAM
      - RDNA2 architecture (gfx1030)
      - 72 Compute Units, 128 ROPs
      - Memory bandwidth: 512 GB/s
      - NOT officially supported by ROCm (community workaround required)

CPU:  Intel Core i7-10700
      - 8 cores / 16 threads
      - 2.9 GHz base, 4.8 GHz boost
      - Used for CPU offloading of model components

RAM:  64GB DDR4
      - Critical for CPU offloading and large dataset handling
      - Allows caching latents and text embeddings in system RAM

OS:   Ubuntu Server 22.04 LTS
      - Headless (no desktop environment needed)
      - SSH access for remote management
```

## ROCm Installation

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Required kernel headers
sudo apt install -y linux-headers-$(uname -r) linux-modules-extra-$(uname -r)

# Add ROCm repository (Ubuntu 22.04 example)
wget https://repo.radeon.com/amdgpu-install/6.3/ubuntu/jammy/amdgpu-install_6.3.60300-1_all.deb
sudo apt install -y ./amdgpu-install_6.3.60300-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to required groups
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Reboot
sudo reboot
```

### The RDNA2 Workaround

The RX 6800 XT (gfx1030) is not officially supported. This environment variable forces ROCm to use the correct ISA target:

```bash
# Add to ~/.bashrc (CRITICAL - required for ALL operations)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Also useful performance tuning
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

**This must be set in every shell session, every script, every systemd service, and every Docker container.**

### Validation

```bash
# Verify ROCm sees the GPU
rocminfo | grep -i "name.*gfx"
# Expected: gfx1030

# Verify HIP
hipinfo
# Should show device details

# Check ROCm version
cat /opt/rocm/.info/version
```

## Python Environment

### Setup

```bash
# Python 3.10 or 3.11 (3.10 is safest for compatibility)
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Create project virtual environment
python3.10 -m venv ~/videoforge-env
source ~/videoforge-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### PyTorch with ROCm

```bash
# Install PyTorch with ROCm 6.2 support
# Check https://pytorch.org/get-started/locally/ for latest compatible version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

### Validate PyTorch + GPU

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")  # Yes, ROCm uses cuda API
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Quick compute test
x = torch.randn(1000, 1000, device='cuda')
y = x @ x.T
print(f"Compute test passed: {y.shape}")
```

### Key Libraries

```bash
# Core ML
pip install accelerate transformers diffusers peft safetensors

# Quantization (ROCm-compatible)
pip install bitsandbytes  # v0.49+ has ROCm support

# Video processing
pip install opencv-python-headless ffmpeg-python scenedetect[opencv]

# Captioning models
pip install qwen-vl-utils

# Utilities
pip install pyyaml tqdm pillow einops omegaconf
```

## Important: What Does NOT Work on AMD/ROCm

| Library/Feature | Status | Alternative |
|----------------|--------|-------------|
| xformers | No AMD support | Use `--sdpa` (PyTorch native scaled dot-product attention) |
| Flash Attention (Dao) | CUDA-only | SDPA handles this automatically |
| CUDA-specific custom kernels | Won't compile | Use PyTorch-native equivalents |
| bitsandbytes < 0.49 | CUDA-only | Use v0.49+ with ROCm support |
| DeepSpeed (some features) | Partial | Use accelerate + FSDP if needed |
| TensorRT | NVIDIA-only | Not needed for training |

## System Services (Optional)

If running training as a background service:

```ini
# /etc/systemd/system/videoforge-train.service
[Unit]
Description=VideoForge Training Job
After=network.target

[Service]
Type=simple
User=chuck
Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="PYTORCH_HIP_ALLOC_CONF=expandable_segments:True"
WorkingDirectory=/home/chuck/videoforge
ExecStart=/home/chuck/videoforge-env/bin/python train.py --config configs/current.yaml
Restart=no

[Install]
WantedBy=multi-user.target
```

## Storage Recommendations

Video training generates significant disk I/O:

- **Source videos:** Plan for 50-200GB depending on dataset size
- **Extracted clips:** 10-50GB (compressed)
- **Latent cache:** 5-20GB (pre-encoded training data)
- **Checkpoints:** 2-5GB per saved checkpoint
- **Total recommended:** 500GB+ free space on SSD

Use an SSD for the latent cache and working directories. Bulk video storage can be on HDD.
