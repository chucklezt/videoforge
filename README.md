# VideoForge

Video generation training pipeline for AMD RX 6800 XT (16GB VRAM, RDNA2) on Ubuntu Server with ROCm.

Train LoRA adapters on Seinfeld TV clips to generate new short video clips of George Costanza in the style of the original show, from text prompts.

## Hardware

- **GPU:** AMD Radeon RX 6800 XT (16GB VRAM, RDNA2/gfx1030)
- **CPU:** Intel i7-10700 (8 cores / 16 threads)
- **RAM:** 64GB DDR4
- **OS:** Ubuntu Server 22.04 LTS
- **Stack:** ROCm 6.3+ / PyTorch ROCm 2.5.1 / SDPA attention (no xformers, no CUDA)

## Pipeline Overview

```
[Source Video Files]
        |
        v
[1. Data Pipeline]      -- Extract clips, scenes, filter, normalize    ✅ Complete
        |
        v
[2. Captioning Pipeline] -- Auto-caption each clip with Qwen2.5-VL    ✅ Complete
        |
        v
[3. Training Pipeline]   -- LoRA fine-tune Wan 2.1 1.3B               ✅ Complete (1500 steps)
        |
        v
[4. Inference Pipeline]  -- Text-to-video with trained LoRA           ✅ Working
        |
        v
[Generated Video Clips]
```

## Current Status

### Training (Complete)

Training uses [finetrainers](https://github.com/a-r-r-o-w/finetrainers) 0.2.0.dev0 (GitHub HEAD). The finetrainers repo is cloned at `~/finetrainers/`.

- **Dataset:** 35 captioned Seinfeld clips in `dataset/clips_conditioned/` with `.txt` sidecar captions
- **Model:** Wan2.1-T2V-1.3B (local at `~/videoforge/models/wan21-1.3b`)
- **LoRA:** rank 8, alpha 8, target modules `blocks.*(to_q|to_k|to_v|to_out.0)`
- **Resolution:** 480x480 (reduced from 480x832 to fit 16GB VRAM)
- **Steps:** 1500 (checkpoints at 500/1000/1500)
- **Optimizer:** AdamW, lr 1e-4, constant with warmup (100 steps)
- **Output:** `~/videoforge/output/wan21_lora/`

**Trained LoRA weights:**
- Step 1000: `output/wan21_lora/lora_weights/001000/pytorch_lora_weights.safetensors`
- Step 1500: `output/wan21_lora/lora_weights/001500/pytorch_lora_weights.safetensors`

### Key Training Fixes

1. **`precomputation_once` flag prevents cache wipe:** finetrainers deletes precomputed latent/text caches every epoch by default. With only 35 clips and `gradient_accumulation_steps 4`, an "epoch" is ~9 steps, so caches were wiped constantly, causing re-encoding loops. The `--precomputation_once` flag (added in finetrainers HEAD) caches once and preserves across epochs.

2. **`enable_model_cpu_offload` unloads VAE before training:** Without `--enable_precomputation`, VAE + text encoder + transformer all load simultaneously and OOM on 16GB. Precomputation encodes all clips first (~1 hour), caches to disk, unloads VAE/text encoder, then trains the transformer alone.

3. **Resolution reduction from 480x832 to 480x480:** The original spec called for 848x480 but this OOMed during training even with precomputation. Reducing to 480x480 fits within 16GB.

### Inference (Working)

Inference uses diffusers `WanPipeline` with the trained LoRA adapter. Scripts are in `scripts/`.

**VRAM strategy for 16GB:**
- `pipe.enable_sequential_cpu_offload()` — moves each submodule to GPU only when needed
- `pipe.enable_attention_slicing("max")` — processes attention one head at a time
- VAE loaded in float32 (required by Wan's VAE), pipeline in float16

**Working configurations (step 1500 LoRA, all with sequential CPU offload + attention slicing):**

| Resolution | Max Frames | Duration | s/step | VRAM util | Notes |
|------------|-----------|----------|--------|-----------|-------|
| 480x480 | 33 | 2.1s | ~11s | ~82% | Best per-frame quality |
| 320x320 | 81 | 5.1s | ~12s | ~75% | Good balance |
| 320x240 | 129 | 8.1s | ~19s | 93.3% | Max duration, 1.1GB headroom |
| 320x240 | 150 | 9.4s | -- | OOM | Exceeds 16GB |

**Inference experiments (9 experiments run, step 1500 LoRA):**

| Exp | Resolution | Frames | Duration | Steps | CFG | Time | Size |
|-----|-----------|--------|----------|-------|-----|------|------|
| 1 | 480x480 | 33 | 2.1s | 40 | 7.5 | 9m 36s | 344 KB |
| 2 | 480x480 | 33 | 2.1s | 50 | 8.0 | 9m 31s | 370 KB |
| 3 | 480x480 | 33 | 2.1s | 40 | 6.5 | 7m 41s | 371 KB |
| 4 | 480x480 | 33 | 2.1s | 60 | 6.5 | 11m 20s | 251 KB |
| 5 | 320x320 | 81 | 5.1s | 60 | 6.5 | 12m 44s | 401 KB |
| 6 | 320x240 | 81 | 4.5s | 60 | 7.5 | 8m 19s | 458 KB |
| 7 | 320x240 | 120 | 7.5s | 60 | 7.5 | 15m 38s | 1,016 KB |
| 8 | 320x240 | 129 | 8.1s | 60 | 7.5 | 19m 53s | 1,304 KB |
| 9 | 320x240 | 129 | 8.1s | 60 | 7.5 | 19m 57s | 1,171 KB |

**VRAM profiling (experiments 7-9):**
- VRAM monitoring added via background polling thread (2s interval)
- Sequential CPU offload shows sawtooth pattern: 0 -> 13.9GB -> 0 as modules load/unload
- At 120 frames (320x240): 82.3% utilization, 3GB headroom
- At 129 frames (320x240): 93.3% utilization, 1.1GB headroom -- practical ceiling
- At 150 frames (320x240): OOM (attention tries to allocate 5.8GB with 2GB free)

**Inference findings:**
- Step 1500 LoRA with improved prompts produces decent results; step 1000 shows less detail
- Face quality is the main remaining challenge (common with 1.3B video models)
- Detailed physical descriptions ("short stocky bald man, horseshoe hair pattern, heavy lidded eyes") help with character likeness
- The Monk's Diner setting requires explicit prompt engineering to avoid generic restaurant scenes
- CFG 6.5-7.5 works best; higher CFG follows prompts more closely but can introduce artifacts
- Longer negative prompts don't use more VRAM but can produce more visual variation
- Duration vs quality is the key tradeoff: 480x480 for quality, 320x240 for length (up to 8.1s)

**Generated test videos:** `output/inference/`

---

### Stage 1: Data Pipeline (Complete)

Five-stage pipeline that transforms raw video files into a structured training dataset:

1. **Preprocessing** -- Normalize to MP4/H.264, extract subtitles
2. **Scene Detection** -- Find natural scene boundaries via PySceneDetect
3. **Clip Extraction** -- Cut scenes into 2-8 second training clips with configurable overlap
4. **Filtering** -- Remove black/white frames, static clips, chaotic motion
5. **Conditioning** -- Resize to target resolution, normalize fps, trim frame count

### Stage 2: Captioning Pipeline (Complete)

Auto-generates detailed text captions for each training clip using a vision-language model:

1. **Visual Captioning** -- Qwen2.5-VL-7B-Instruct (bfloat16, ~14GB VRAM) describes each clip
2. **Enrichment** -- Merges style tags and subtitle dialogue into the visual caption
3. **Review** -- Interactive terminal tool to accept, edit, skip, or delete captions
4. **Export** -- Writes `.txt` sidecar files alongside clips for training tools

**Known issue:** captioner requires `fps_sample: 1.0` in `configs/caption.yaml` (not 4.0) and `qwen-vl-utils==0.0.8`. Higher fps values produce tensor shapes that hit unsupported RDNA2 kernels.

---

### Training Launch Command

Run from `~/finetrainers/` after the session setup below:

```bash
cd ~/finetrainers
accelerate launch --mixed_precision bf16 --num_processes 1 train.py \
  --parallel_backend accelerate \
  --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1 \
  --model_name "wan" \
  --pretrained_model_name_or_path "/home/chuck/videoforge/models/wan21-1.3b" \
  --dataset_config "/home/chuck/videoforge/finetrainers_training.json" \
  --dataset_shuffle_buffer_size 10 \
  --dataloader_num_workers 0 \
  --training_type "lora" \
  --seed 42 \
  --batch_size 1 \
  --train_steps 1500 \
  --rank 8 \
  --lora_alpha 8 \
  --target_modules "blocks.*(to_q|to_k|to_v|to_out.0)" \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --checkpointing_steps 500 \
  --checkpointing_limit 2 \
  --enable_slicing \
  --enable_tiling \
  --enable_precomputation \
  --precomputation_items 35 \
  --precomputation_once \
  --optimizer "adamw" \
  --lr 1e-4 \
  --lr_scheduler "constant_with_warmup" \
  --lr_warmup_steps 100 \
  --beta1 0.9 \
  --beta2 0.99 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --flow_weighting_scheme "logit_normal" \
  --output_dir "/home/chuck/videoforge/output/wan21_lora" \
  --tracker_name "finetrainers-wan" \
  --report_to "none"
```

### Inference Example

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python scripts/inference_test.py
```

See `scripts/inference_test.py` for the base script. Experiments 4-9 (`scripts/inference_exp[4-9].py`) iterate on prompt engineering, resolution/duration tradeoffs, and VRAM profiling.

### Session Setup (Required Every Session)

Every terminal session requires these commands before training or inference:

```bash
source ~/videoforge-env/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
pip uninstall torchao -y
pip install datasets==3.3.2
pip uninstall torchcodec -y
```

**Why each step is needed:**
- `torchao 0.16.0` requires `torch.int1` which doesn't exist in PyTorch ROCm 2.5.1 -- breaks diffusers import
- `datasets 4.8.4` hard-requires `torchcodec` for the Video feature type; pinning to 3.3.2 uses `decord` instead
- `torchcodec` is CUDA-only (requires `libnvrtc.so.13`); no ROCm support exists

finetrainers re-installs `torchao` and upgrades `datasets` every time it's installed from GitHub, so these uninstalls must run every session.

---

### CLI

```bash
# Environment validation
python3 -m videoforge validate

# Full data pipeline
python3 -m videoforge data --input /path/to/videos --output ./dataset
python3 -m videoforge data --config configs/data_pipeline.yaml --input /path/to/videos

# Individual stages
python3 -m videoforge data preprocess -i /path/to/videos -o ./dataset/normalized
python3 -m videoforge data scenes -i ./dataset/normalized -o scenes.json
python3 -m videoforge data extract --scenes scenes.json -o ./dataset/clips
python3 -m videoforge data filter -i ./dataset/clips
python3 -m videoforge data condition -i ./dataset/clips -o ./dataset/clips_conditioned

# Captioning pipeline
python3 -m videoforge caption --dataset ./dataset                    # Caption all clips
python3 -m videoforge caption --dataset ./dataset --recaption        # Re-caption everything
python3 -m videoforge caption --clips clip_id_1 clip_id_2           # Caption specific clips
python3 -m videoforge caption review --dataset ./dataset             # Interactive review
python3 -m videoforge caption export --dataset ./dataset             # Export .txt sidecar files
```

### Output Structure

```
dataset/
├── metadata.json
├── scenes.json
├── normalized/          # Format-normalized source videos
├── clips/               # Extracted training clips
├── clips_conditioned/   # Resized/normalized clips + .txt sidecar captions
│   ├── video            # Line-separated absolute paths to .mp4 files (for finetrainers)
│   └── text             # Line-separated captions in same order as video file
├── clip_metadata/       # Per-clip JSON metadata
└── subtitles/           # Extracted subtitle tracks
```

## Project Structure

```
videoforge/
├── configs/                    # YAML configuration files
│   ├── data_pipeline.yaml
│   ├── caption.yaml            # fps_sample must be 1.0 for ROCm compatibility
│   ├── train_wan21_lora.yaml
│   ├── inference.yaml
│   └── style_tags.yaml
├── finetrainers_training.json  # Dataset config for finetrainers
├── scripts/                    # Inference scripts
│   ├── inference_test.py       # Single-run inference (baseline)
│   ├── inference_comparison.py # LoRA checkpoint comparison (step 1000 vs 1500)
│   ├── inference_experiments.py # Parameter sweep (exp 1-3)
│   ├── inference_exp4.py       # Detailed scene prompt, 480x480
│   ├── inference_exp5.py       # Longer generation, 320x320, 81 frames
│   ├── inference_exp6.py       # 320x240, 18fps, CFG 7.5
│   ├── inference_exp7.py       # 120 frames + VRAM monitoring
│   ├── inference_exp8.py       # 129 frames (max duration) + VRAM monitoring
│   └── inference_exp9.py       # Simplified negative prompt + VRAM monitoring
├── specs/                      # Project specifications
├── videoforge/                 # Python package
│   ├── __main__.py             # CLI entry point
│   ├── data/                   # Data pipeline (complete)
│   │   ├── preprocess.py
│   │   ├── scene_detect.py
│   │   ├── clip_extract.py
│   │   ├── clip_filter.py
│   │   └── clip_condition.py
│   ├── caption/                # Captioning pipeline (complete)
│   │   ├── captioner.py        # Qwen2.5-VL video captioning
│   │   ├── enrichment.py       # Style tags + dialogue merging
│   │   ├── review.py           # Interactive caption review
│   │   └── export.py           # .txt sidecar export
│   ├── train/                  # Training pipeline (scaffold; training uses finetrainers)
│   ├── generate/               # Inference pipeline (scripts/ used directly for now)
│   ├── postprocess/            # Post-processing (not yet built)
│   └── utils/                  # Shared utilities
│       ├── config.py
│       ├── video.py
│       ├── rocm.py
│       └── vram.py
├── setup.py
├── requirements.txt
└── requirements-rocm.txt
```

## Setup

```bash
# Full automated setup (ROCm must already be installed)
chmod +x specs/setup-videoforge.sh
./specs/setup-videoforge.sh

# Or manual setup
python3.10 -m venv ~/videoforge-env
source ~/videoforge-env/bin/activate
pip install -r requirements-rocm.txt
pip install -r requirements.txt
pip install -e .

# Clone finetrainers (required for training)
git clone https://github.com/a-r-r-o-w/finetrainers.git ~/finetrainers

python3 -m videoforge validate
```

## Key Constraints

- 16GB VRAM ceiling -- precomputation staging required for Wan2.1 training; sequential CPU offload + attention slicing for inference
- AMD ROCm (RDNA2/gfx1030) -- no xformers, no CUDA kernels, SDPA only
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` required everywhere
- `qwen-vl-utils==0.0.8` required for captioning (0.0.14 breaks video inference on ROCm)
- `torchao`, `torchcodec`, `datasets>=4` must be uninstalled/pinned every session (finetrainers dependency conflicts)
- Open source only -- all models and tools permissively licensed

## Technology Stack

| Component | Tool |
|-----------|------|
| GPU Compute | ROCm 6.3+ / PyTorch ROCm 2.5.1 |
| Base Video Model | Wan 2.1 1.3B (text-to-video) |
| Fine-tuning | LoRA (rank 8) via finetrainers 0.2.0.dev0 |
| Video Captioning | Qwen2.5-VL-7B-Instruct |
| Scene Detection | PySceneDetect |
| Video Processing | FFmpeg |
| Inference | Diffusers WanPipeline + sequential CPU offload |

## Known Issues & Fixes

**Captioning: HIP error on video inference**
- Symptom: `HIP error: no kernel image is available for execution on the device` during captioning
- Cause: `fps_sample: 4.0` generates too many frames, producing tensor shapes that hit unsupported RDNA2 kernels. Also, `qwen_vl_utils` returns `float32` video tensors; passing them to a `bfloat16` model triggers the kernel mismatch.
- Fix: Set `fps_sample: 1.0` in `configs/caption.yaml`. Ensure `videoforge/caption/captioner.py` casts video inputs before inference: `video_inputs = [v.to(dtype=self.dtype) for v in video_inputs]`

**Training: precomputation cache wiped every epoch**
- Symptom: Training re-encodes all clips every ~9 steps, spending most time on precomputation instead of training
- Cause: finetrainers deletes precomputed caches each epoch. With 35 clips and `gradient_accumulation_steps 4`, an epoch is ~9 steps.
- Fix: Use `--precomputation_once` flag (available in finetrainers GitHub HEAD).

**Training: OOM without precomputation**
- Symptom: `HIP out of memory` at step 0 even with gradient checkpointing
- Cause: Without `--enable_precomputation`, VAE + text encoder + transformer all load simultaneously, exceeding 16GB
- Fix: Always use `--enable_precomputation --precomputation_items 35 --precomputation_once`. Also ensure `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is exported before launch.

**Training: OOM at 480x832 resolution**
- Symptom: OOM during training even with precomputation
- Cause: Original spec resolution of 848x480 requires too much VRAM for activations
- Fix: Reduce resolution bucket to `[49, 480, 480]` in dataset config.

**Training: finetrainers breaks on install**
- Symptom: `torch.int1` AttributeError or `ModuleNotFoundError: finetrainers.patches.dependencies`
- Cause: PyPI `finetrainers==0.2.0` is broken. GitHub HEAD includes `patches/dependencies`. `torchao 0.16.0` requires `torch.int1` which doesn't exist in PyTorch ROCm 2.5.1.
- Fix: Install from GitHub (`pip install git+https://github.com/a-r-r-o-w/finetrainers.git`), then `pip uninstall torchao -y`. Run full session setup every session.

**datasets 4.8.4 / torchcodec conflict**
- Symptom: `Could not load libtorchcodec` during training data loading
- Cause: `finetrainers` upgrades `datasets` to 4.8.4 which requires `torchcodec` (CUDA-only, no ROCm support)
- Fix: `pip install datasets==3.3.2 && pip uninstall torchcodec -y` -- must run every session after finetrainers installs

**Inference: OOM at 480x480 with model CPU offload**
- Symptom: `HIP out of memory` during SDPA self-attention at 480x480
- Cause: `enable_model_cpu_offload()` keeps the full transformer on GPU; the attention matrix for 33 frames at 480x480 exceeds remaining VRAM
- Fix: Use `enable_sequential_cpu_offload()` + `enable_attention_slicing("max")`. This is ~3x slower per step but fits 480x480 in 16GB.
