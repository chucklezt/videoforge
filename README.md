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

### Stage 3: Training Pipeline (Running)

Training uses [finetrainers](https://github.com/a-r-r-o-w/finetrainers) 0.2.0.dev0, not the VideoForge-native training scaffold. Training runs via `accelerate launch` from `~/finetrainers/train.py`.

- **Dataset:** 35 captioned clips in `dataset/clips_conditioned/` with matching `.txt` sidecar files
- **Model:** Wan2.1-T2V-1.3B (local at `models/wan21-1.3b`)
- **LoRA rank:** 8
- **Steps:** 1500 (checkpoints at 500/1000/1500)
- **Output:** `output/wan21_lora/`
- **Active in:** tmux session `train`

### Stage 4: Inference Pipeline (Not Yet Built)

Script-to-video generation using the trained LoRA adapter.

### Session Setup (Required Every Session)

Every terminal session requires these commands before training or inference:

```bash
pip uninstall torchao -y
pip install datasets==3.3.2
pip uninstall torchcodec -y
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

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
├── clip_metadata/       # Per-clip JSON metadata
└── subtitles/           # Extracted subtitle tracks
```

## Project Structure

```
videoforge/
├── configs/                    # YAML configuration files
│   ├── data_pipeline.yaml
│   ├── caption.yaml
│   ├── train_wan21_lora.yaml
│   ├── inference.yaml
│   └── style_tags.yaml
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
│   ├── train/                  # Training pipeline (scaffold, training uses finetrainers)
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
python3 -m videoforge validate
```

## Key Constraints

- 16GB VRAM ceiling -- quantization, gradient checkpointing, CPU offloading
- AMD ROCm (RDNA2/gfx1030) -- no xformers, no CUDA kernels, SDPA only
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` required everywhere
- Open source only -- all models and tools permissively licensed

## Technology Stack

| Component | Tool |
|-----------|------|
| GPU Compute | ROCm 6.3+ / PyTorch ROCm |
| Base Video Model | Wan 2.1 1.3B (text-to-video) |
| Fine-tuning | LoRA via finetrainers 0.2.0.dev0 |
| Video Captioning | Qwen2.5-VL-7B-Instruct |
| Scene Detection | PySceneDetect |
| Video Processing | FFmpeg |
| Inference | Diffusers (planned) |
