# 04 - Training Pipeline: LoRA Fine-Tuning for Video Generation

## Overview

Fine-tune a pre-trained video diffusion model on the captioned clip dataset using LoRA (Low-Rank Adaptation). This produces a small adapter (~50-200MB) that captures the visual style of the training data without modifying the full model weights.

## Base Model: Wan 2.1 1.3B (Text-to-Video)

See `06-MODEL-SELECTION.md` for full rationale. Summary:
- 1.3 billion parameters (fits in 16GB with quantization)
- Open source (Apache 2.0)
- Generates 2-8 second clips at up to 720p
- Strong temporal coherence
- Supported by kohya-ss/sd-scripts and OneTrainer

## Training Approach

### Why LoRA

- Full fine-tuning requires 3-5x model size in VRAM (impossible at 16GB)
- LoRA trains only small adapter matrices (~0.1-1% of model parameters)
- Produces a portable adapter file that can be loaded/unloaded at inference
- Training time: hours instead of days
- LoRA rank 16-64 captures style and motion patterns effectively

### Training Configuration

```yaml
# config/train_wan21_lora.yaml
model:
  name: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  type: text_to_video
  dtype: float16

lora:
  rank: 32                    # LoRA rank (16-64 typical, higher = more capacity)
  alpha: 32                   # Scaling factor (usually equal to rank)
  target_modules:             # Which layers to attach LoRA to
    - "to_q"
    - "to_k"
    - "to_v"
    - "to_out.0"
    - "ff.net.0.proj"
    - "ff.net.2"
  dropout: 0.05               # Light dropout for regularization

training:
  batch_size: 1               # Batch size 1 is likely necessary for 16GB
  gradient_accumulation: 4    # Effective batch size = 4
  gradient_checkpointing: true
  learning_rate: 1.0e-4
  lr_scheduler: cosine
  lr_warmup_steps: 100
  max_train_steps: 3000       # Adjust based on dataset size
  mixed_precision: fp16        # Use fp16 (bf16 support varies on RDNA2)
  seed: 42

  # Critical VRAM optimizations
  cache_latents_to_disk: true  # Pre-encode videos, cache to SSD
  cache_text_encoder_outputs: true
  cpu_offload_text_encoder: true

optimizer:
  name: adamw8bit             # 8-bit optimizer via bitsandbytes-rocm
  weight_decay: 0.01
  betas: [0.9, 0.999]

dataset:
  path: "/path/to/dataset/clips/"
  caption_extension: ".txt"
  video_extension: ".mp4"
  resolution: [848, 480]
  frame_count: 49              # ~2 sec at 24fps
  fps: 24

  # Data augmentation
  random_crop: false           # Clips are pre-cropped
  flip_augment: false          # Don't flip - text/signs would be mirrored
  caption_dropout_rate: 0.1    # 10% chance of empty caption (improves CFG)

saving:
  output_dir: "./output/wan21_lora"
  save_every_n_steps: 500
  keep_last_n_checkpoints: 3
  save_format: safetensors

sampling:
  enabled: true
  every_n_steps: 500
  prompts:
    - "cinematic television scene, dramatic lighting. A woman sits on a couch in a dimly lit living room, looking worried."
    - "cinematic television scene. Two men stand in an office, one gesturing while speaking."
  num_frames: 49
  width: 848
  height: 480
  guidance_scale: 7.5
  num_inference_steps: 30
```

## Training Tool Options

### Option A: kohya-ss/sd-scripts (Recommended)

Most mature, best community documentation for AMD.

```bash
# Clone and setup
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
pip install -r requirements.txt

# Additional ROCm requirements
pip install bitsandbytes  # v0.49+ for ROCm

# Verify no xformers dependency
# Use --sdpa flag for attention
```

Training command:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0

accelerate launch \
  --mixed_precision fp16 \
  --num_processes 1 \
  wan_train_network.py \
  --pretrained_model_name_or_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --dataset_config dataset_config.toml \
  --output_dir ./output/wan21_lora \
  --output_name wan21_show_style \
  --network_module networks.lora \
  --network_dim 32 \
  --network_alpha 32 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --max_train_steps 3000 \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --mixed_precision fp16 \
  --save_every_n_steps 500 \
  --sample_every_n_steps 500 \
  --optimizer_type adamw8bit \
  --sdpa
```

### Option B: OneTrainer

More user-friendly, auto-detects AMD.

```bash
git clone https://github.com/Nerogar/OneTrainer.git
cd OneTrainer

# Automatic AMD detection during install
./install.sh

# Launch (uses its own config format)
python start.py
```

OneTrainer has a GUI but also accepts JSON configs for CLI usage.

## VRAM Budget (16GB)

Estimated VRAM allocation during training:

```
Component                              VRAM
─────────────────────────────────────────────
Wan 2.1 1.3B (fp16, no text enc)      ~3.5 GB
LoRA adapter weights                   ~0.1 GB
Gradient checkpointing overhead        ~1.0 GB
Optimizer states (8-bit)               ~0.5 GB
Activations (batch size 1)             ~6-8 GB
Video latent (49 frames, 480p)         ~2-3 GB
PyTorch/ROCm overhead                  ~1-2 GB
─────────────────────────────────────────────
Total estimate                         ~14-18 GB
```

If this exceeds 16GB:
1. Reduce `frame_count` to 33 (~1.3 seconds)
2. Reduce resolution to 640x360
3. Reduce LoRA rank to 16
4. Enable CPU offloading for more components

## Latent Caching (Critical for VRAM)

Pre-encode all training videos into latent space and cache to disk. This avoids loading the VAE encoder during training.

```python
# Pre-cache latents (run once before training)
python cache_latents.py \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --dataset /path/to/dataset/clips/ \
  --output /path/to/dataset/latent_cache/ \
  --dtype fp16 \
  --batch_size 1
```

Similarly, cache text encoder outputs:
```python
python cache_text_encoder.py \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --dataset /path/to/dataset/clips/ \
  --output /path/to/dataset/te_cache/ \
  --dtype fp16
```

Both kohya-ss and OneTrainer handle this automatically when configured.

## Training Monitoring

### Loss Curves

Monitor training loss to detect overfitting:

```python
# tensorboard (if using kohya-ss)
pip install tensorboard
tensorboard --logdir ./output/wan21_lora/logs --bind_all

# Or parse log files
grep "loss" training.log | awk '{print NR, $NF}'
```

### Signs of Good Training
- Loss decreases steadily for first ~500-1000 steps
- Loss stabilizes (doesn't keep dropping toward 0)
- Sample images begin showing source material style by step ~1000
- Characters/settings become recognizable by step ~2000

### Signs of Overfitting
- Loss drops to near zero
- Generated samples are exact copies of training clips
- Samples look identical regardless of prompt
- Fix: reduce steps, increase LoRA dropout, reduce rank

### Signs of Underfitting
- Generated samples look generic (like base model)
- No stylistic similarity to training data
- Fix: increase steps, increase LoRA rank, improve captions

## Training Schedule

With ~200-500 training clips on a single RX 6800 XT:

```
Expected training time: 4-12 hours for 3000 steps
Latent caching: 30-60 minutes (one-time)
Text encoder caching: 5-10 minutes (one-time)
```

## Resume Training

Both kohya-ss and OneTrainer support resuming from checkpoints:

```bash
# kohya-ss: add --resume flag
accelerate launch wan_train_network.py \
  --resume ./output/wan21_lora/checkpoint-1500 \
  ... (rest of args)
```

## Output

Training produces:
- `wan21_show_style.safetensors` -- The LoRA adapter file (~50-200MB)
- `checkpoint-*` directories -- Training state for resume
- `samples/` -- Generated samples at each checkpoint
- `training.log` -- Loss curves and training metrics
