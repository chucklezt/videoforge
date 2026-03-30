# VideoForge

Video generation training pipeline for AMD RX 6800 XT (16GB VRAM, RDNA2) on Ubuntu Server with ROCm.

Train LoRA adapters on video clips from existing content, then generate new short clips from text scripts that match the source material's visual style, characters, and aesthetic.

## Hardware Target

- **GPU:** AMD Radeon RX 6800 XT (16GB VRAM, RDNA2/gfx1030)
- **CPU:** Intel i7-10700 (8 cores / 16 threads)
- **RAM:** 64GB DDR4
- **OS:** Ubuntu Server 22.04 LTS
- **Stack:** ROCm 6.3+ / PyTorch ROCm / SDPA attention (no xformers, no CUDA)

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
[3. Training Pipeline]   -- LoRA fine-tune Wan 2.1 1.3B               ✅ Running
        |
        v
[4. Inference Pipeline]  -- Script-to-video generation                 🔲 Not yet built
        |
        v
[Generated Video Clips]
```

## Current Status

### Stage 1: Data Pipeline (Complete)

Five-stage pipeline that transforms raw video files into a structured training dataset:

1. **Preprocessing** -- Normalize to MP4/H.264, extract subtitles
2. **Scene Detection** -- Find natural scene boundaries via PySceneDetect
3. **Clip Extraction** -- Cut scenes into 2-8 second training clips with configurable overlap
4. **Filtering** -- Remove black/white frames, static clips, chaotic motion
5. **Conditioning** -- Resize to target resolution (848x480), normalize fps, trim frame count

### Stage 2: Captioning Pipeline (Complete)

Auto-generates detailed text captions for each training clip using a vision-language model:

1. **Visual Captioning** -- Qwen2.5-VL-7B-Instruct (bfloat16, ~14GB VRAM) describes each clip
2. **Enrichment** -- Merges style tags and subtitle dialogue into the visual caption
3. **Review** -- Interactive terminal tool to accept, edit, skip, or delete captions
4. **Export** -- Writes `.txt` sidecar files alongside clips for training tools

**Known issue:** captioner requires `fps_sample: 1.0` in `configs/caption.yaml` (not 4.0) and `qwen-vl-utils==0.0.8`. Higher fps values produce tensor shapes that hit unsupported RDNA2 kernels.

### Stage 3: Training Pipeline (Running)

Training uses [finetrainers](https://github.com/a-r-r-o-w/finetrainers) 0.2.0.dev0 (GitHub HEAD), not the VideoForge-native training scaffold. The finetrainers repo is cloned at `~/finetrainers/`. Training runs via `accelerate launch` from that directory.

- **Dataset:** 35 captioned clips in `dataset/clips_conditioned/` with matching `.txt` sidecar files
- **Dataset config:** `~/videoforge/finetrainers_training.json`
- **Model:** Wan2.1-T2V-1.3B (local at `~/videoforge/models/wan21-1.3b`)
- **LoRA rank:** 8, lora_alpha 8
- **Target modules:** `blocks.*(to_q|to_k|to_v|to_out.0)`
- **Resolution bucket:** `[21, 480, 832]` (21 frames, 480x832)
- **Steps:** 1500 (checkpoints at 500/1000/1500)
- **Output:** `~/videoforge/output/wan21_lora/`
- **Active in:** tmux session `train` on chuckai

#### Training Launch Command

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

`--enable_precomputation --precomputation_items 35 --precomputation_once` is required on 16GB VRAM. Without it, VAE + text encoder + transformer all load simultaneously and OOM. Precomputation encodes all clips first (~1 hour), caches to disk, unloads, then trains the transformer alone.

### Stage 4: Inference Pipeline (Not Yet Built)

Script-to-video generation using the trained LoRA adapter.

---

### Session Setup (Required Every Session)

Every terminal session requires these commands before training or captioning:

```bash
source ~/videoforge-env/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
pip uninstall torchao -y
pip install datasets==3.3.2
pip uninstall torchcodec -y
```

**Why each step is needed:**
- `torchao 0.16.0` requires `torch.int1` which doesn't exist in PyTorch ROCm 2.5.1 — breaks diffusers import
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
│   ├── train/                  # Training pipeline (scaffold only; training uses finetrainers)
│   ├── generate/               # Inference pipeline (not yet built)
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

- 16GB VRAM ceiling -- precomputation staging required for Wan2.1 training; quantization, gradient checkpointing, CPU offloading elsewhere
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
| Fine-tuning | LoRA via finetrainers 0.2.0.dev0 (GitHub HEAD) |
| Video Captioning | Qwen2.5-VL-7B-Instruct |
| Scene Detection | PySceneDetect |
| Video Processing | FFmpeg |
| Inference | Diffusers (planned) |

## Lessons Learned

First long debug marathon with several independent fires:

- Captioning broken (HIP error: no kernel image) — root cause was fps_sample: 4.0 in caption.yaml generating too many frames, hitting unsupported RDNA2 kernels. Fix: fps_sample: 1.0 + cast video tensors to bfloat16 before passing to the model. 26/35 clips captioned before the SSH timeout.
- The training scaffold (VideoForge native) was a dead end — it was wired to a CogVideoX diffusers script that can't load Wan model architecture (patch_size: [1,2,2] is a list, not an int, breaks nn.Conv2d). You navigated around this correctly.
- finetrainers 0.2.0 (PyPI) is broken — missing patches/dependencies subpackage. The GitHub HEAD (0.2.0.dev0) is what works, as we already knew from the README.
- torchcodec is CUDA-only — finetrainers 0.2.0 pulls datasets 4.8.4 which requires torchcodec for its Video feature type. Fix confirmed: pip uninstall torchcodec -y + pip install datasets==3.3.2.
- datasets 4.8.4 got re-installed when you installed finetrainers from GitHub. The session-setup uninstall sequence from the README is required every session, and it ran correctly before this training launch.
- rank 8 → rank 16 — I see "trainable parameters": 5898240 in this run vs 11796480 earlier. We're running rank 8, not 16. That's actually fine for this dataset size — less VRAM pressure, still meaningful LoRA.

