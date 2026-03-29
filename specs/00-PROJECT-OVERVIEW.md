# Project: VideoForge - Open Source Video Generation Training Pipeline

## Mission

Build an open-source video generation training and inference system that runs on consumer AMD hardware. The system trains on clips from existing video content (e.g., TV shows) and generates new short video clips from text scripts that match the visual style, characters, and aesthetic of the source material.

## Hardware Target

- **GPU:** AMD Radeon RX 6800 XT (16GB VRAM, RDNA2/gfx1030)
- **CPU:** Intel i7-10700 (8 cores / 16 threads)
- **RAM:** 64GB DDR4
- **OS:** Ubuntu Server (22.04 or 24.04 LTS)
- **GPU Stack:** AMD ROCm 6.3+ with PyTorch ROCm wheels

## Core Workflow

```
[Source Video Files]
        |
        v
[1. Data Pipeline] -- Extract clips, scenes, audio, subtitles
        |
        v
[2. Captioning Pipeline] -- Auto-caption each clip with detailed descriptions
        |
        v
[3. Training Pipeline] -- LoRA fine-tune a video diffusion model
        |
        v
[4. Inference Pipeline] -- Accept script text, generate video clips
        |
        v
[Generated Video Clips]
```

## Key Design Constraints

1. **16GB VRAM ceiling** -- Every component must fit within 16GB. Use quantization (8-bit/4-bit), gradient checkpointing, CPU offloading, and latent caching aggressively.
2. **AMD ROCm (unofficial RDNA2)** -- No CUDA. No xformers. Use SDPA attention, PyTorch-native ops, and ROCm-compatible libraries only.
3. **Open source only** -- Every model, library, and tool must be open source with a license that permits fine-tuning and local use.
4. **Modular architecture** -- Each pipeline stage is independent. You can re-run captioning without re-extracting clips, or re-train without re-captioning.
5. **CLI-first** -- All operations runnable from command line with YAML/JSON configs. A web UI is a future nice-to-have, not a requirement.

## Target Output

- Generate 2-8 second video clips at 480p-720p resolution, 24fps
- Visual style, color grading, and scene composition should match training data
- Characters and settings should be recognizable from training material
- Generated clips should follow the narrative described in the input script

## What This Is NOT

- Not a real-time video generator
- Not a full episode generator (short clips only)
- Not a deepfake tool (style transfer, not face replacement)
- Not a commercial product (personal research/experimentation)

## Technology Stack Summary

| Component | Tool | Why |
|-----------|------|-----|
| GPU Compute | ROCm 6.3+ / PyTorch ROCm | Only viable AMD ML stack |
| Base Video Model | Wan 2.1 1.3B (text-to-video) | Small enough for 16GB, open source, good quality |
| Fine-tuning | LoRA via kohya-ss/sd-scripts or OneTrainer | Proven AMD/ROCm support |
| Quantization | bitsandbytes-rocm | 8-bit optimizers to fit in VRAM |
| Video Captioning | Qwen2-VL or LLaVA-Video | Open source video understanding |
| Scene Detection | PySceneDetect | Automatic clip boundary detection |
| Video Processing | FFmpeg | Industry standard |
| Inference | ComfyUI + video nodes | Best AMD inference support |
| Orchestration | Python CLI + YAML configs | Simple, debuggable |

## Document Index

| File | Purpose |
|------|---------|
| `setup-videoforge.sh` | **Automated setup script** -- run this first on Ubuntu Server |
| `01-HARDWARE-AND-ENVIRONMENT.md` | ROCm setup, PyTorch install, environment validation |
| `02-DATA-PIPELINE.md` | Video ingestion, scene detection, clip extraction |
| `03-CAPTIONING-PIPELINE.md` | Automated video captioning for training pairs |
| `04-TRAINING-PIPELINE.md` | LoRA fine-tuning configuration and execution |
| `05-INFERENCE-PIPELINE.md` | Script-to-video generation |
| `06-MODEL-SELECTION.md` | Detailed model comparison and selection rationale |
| `07-PROJECT-STRUCTURE.md` | Directory layout, config schema, CLI interface |
| `08-LIMITATIONS-AND-WORKAROUNDS.md` | Known issues, VRAM strategies, fallback plans |
