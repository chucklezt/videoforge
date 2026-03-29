# 06 - Model Selection and Rationale

## The Problem

We need to choose:
1. A **base video diffusion model** for fine-tuning
2. A **vision-language model** for captioning
3. A **training framework** that supports both the model and AMD/ROCm

All must fit within 16GB VRAM (not simultaneously -- captioning and training run separately).

## Video Diffusion Model Comparison

| Model | Parameters | License | Min VRAM (LoRA) | Video Length | Resolution | ROCm Status |
|-------|-----------|---------|-----------------|-------------|------------|-------------|
| **Wan 2.1 T2V 1.3B** | 1.3B | Apache 2.0 | ~14-16GB | 2-8 sec | up to 720p | Works via PyTorch |
| Wan 2.1 T2V 14B | 14B | Apache 2.0 | ~28GB+ | 2-16 sec | up to 1080p | Too large |
| CogVideoX-2B | 2B | Apache 2.0 | ~16-20GB | 2-6 sec | 720p | Untested on ROCm |
| CogVideoX-5B | 5B | CogVideoX License | ~24GB+ | 2-6 sec | 720p | Too large |
| Open-Sora 1.2 | ~700M | Apache 2.0 | ~10-12GB | 2-16 sec | up to 720p | Untested on ROCm |
| AnimateDiff v3 | ~400M (+SD) | Apache 2.0 | ~12-14GB | 1-3 sec | 512p | Likely works |
| Mochi 1 | 10B | Apache 2.0 | ~24GB+ | 2-6 sec | 848x480 | Too large |
| LTX-Video | 2B | Custom open | ~16-18GB | 2-5 sec | 768p | Untested |

## Primary Choice: Wan 2.1 T2V 1.3B

### Why

1. **Fits in 16GB** -- The 1.3B model is small enough for LoRA training with quantization and gradient checkpointing on 16GB VRAM.

2. **Best quality at this size** -- Wan 2.1 demonstrates strong temporal coherence and visual quality despite being relatively small. It punches well above its weight class.

3. **Apache 2.0 license** -- Fully open, no commercial restrictions, no gating.

4. **Diffusers integration** -- First-class support in HuggingFace Diffusers, which uses standard PyTorch ops that work on ROCm.

5. **Training tool support** -- Supported by kohya-ss/sd-scripts and OneTrainer, both of which have confirmed AMD/ROCm compatibility.

6. **Active community** -- Large user base means more guides, troubleshooting resources, and LoRA examples to reference.

7. **Flexible frame counts** -- Supports variable-length generation (1+4k frames: 17, 33, 49, 81, etc.).

### Limitations

- 1.3B is less detailed than the 14B version
- Maximum practical resolution ~480p-720p on 16GB
- Motion complexity limited compared to larger models
- Character consistency across clips is not guaranteed (a general limitation of all current video models)

### Model Card

```
Name:           Wan-AI/Wan2.1-T2V-1.3B-Diffusers
Parameters:     1.3 billion
Architecture:   DiT (Diffusion Transformer) with 3D attention
VAE:            Wan 2.1 Video VAE (spatial + temporal compression)
Text Encoder:   T5-XXL (can be offloaded to CPU or cached)
Scheduler:      Flow Matching (Euler)
Frame format:   1 + 4k (e.g., 49 frames = 12 groups of 4 + 1)
```

## Fallback Choice: Open-Sora 1.2

If Wan 2.1 has ROCm issues:

- Smaller (~700M parameters), more VRAM headroom
- Apache 2.0 license
- Simpler architecture may be more compatible
- Lower quality output but more room for experimentation

## Fallback Choice: AnimateDiff v3 + SD 1.5

If video-native models are problematic on ROCm:

- AnimateDiff adds temporal layers to Stable Diffusion 1.5
- SD 1.5 has the most extensive ROCm testing of any diffusion model
- Lower quality and shorter clips (1-2 seconds) but very likely to work
- Large ecosystem of existing LoRAs and tools

## Vision-Language Model for Captioning

| Model | Parameters | License | VRAM (4-bit) | Video Support | Quality |
|-------|-----------|---------|-------------|---------------|---------|
| **Qwen2-VL-7B-Instruct** | 7B | Apache 2.0 | ~6-8GB | Native video | Excellent |
| InternVL2-8B | 8B | Apache 2.0 | ~6-8GB | Multi-frame | Very good |
| LLaVA-Video-7B | 7B | Apache 2.0 | ~6-8GB | Native video | Good |
| Qwen2-VL-2B-Instruct | 2B | Apache 2.0 | ~2-3GB | Native video | Moderate |

### Primary Choice: Qwen2-VL-7B-Instruct

- Native video input (not just individual frames)
- Excellent scene description capabilities
- Fits comfortably in 16GB at 4-bit quantization
- Apache 2.0 license
- Strong HuggingFace Transformers integration

### Fallback: Qwen2-VL-2B-Instruct

If the 7B model has issues, the 2B version provides:
- Much lower VRAM usage (~2-3GB at 4-bit)
- Faster captioning
- Lower quality descriptions but still usable

## Training Framework

| Framework | Wan 2.1 Support | AMD/ROCm | GUI | Ease of Use |
|-----------|:-:|:-:|:-:|:-:|
| **kohya-ss/sd-scripts** | Yes | Verified | Optional | Medium |
| **OneTrainer** | Yes | Built-in | Yes | Easy |
| SimpleTuner | Yes | Partial (no FLUX) | No | Medium |
| ai-toolkit (this repo) | Yes | Fork only | Yes | Easy |
| HuggingFace Diffusers (raw) | Yes | Via PyTorch | No | Hard |

### Primary: kohya-ss/sd-scripts

- Most mature and widely documented
- Community-verified on AMD GPUs (including RX 6800)
- Supports all the VRAM optimizations we need
- Active development with frequent updates

### Alternative: OneTrainer

- Best AMD auto-detection and setup
- Nicer for beginners
- GUI available for parameter tuning
- Slightly less documentation for advanced video training

### Both are good choices. Start with whichever feels more comfortable.

## Image-to-Video Extension (Future)

Once text-to-video works, consider adding image-to-video (I2V) capability:

- **Wan 2.1 I2V 1.3B** -- Same family, takes a reference image + text prompt
- Enables "generate a key frame as an image, then animate it"
- Better character consistency since the first frame is specified
- Same VRAM requirements as the T2V model

## Model Download

```bash
# Install huggingface CLI
pip install huggingface-hub

# Download Wan 2.1 T2V 1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ./models/wan21-1.3b/

# Download Qwen2-VL-7B for captioning
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir ./models/qwen2-vl-7b/

# Approximate download sizes:
# Wan 2.1 1.3B: ~5 GB
# Qwen2-VL-7B: ~15 GB
# T5-XXL (text encoder, downloaded separately by diffusers): ~10 GB
```
