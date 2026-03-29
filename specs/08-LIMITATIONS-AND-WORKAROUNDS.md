# 08 - Limitations, Risks, and Workarounds

## Hardware Limitations

### 16GB VRAM is Tight

The single biggest constraint. Every design decision flows from this.

| Scenario | VRAM Impact | Mitigation |
|----------|------------|------------|
| Base model in fp16 | ~3.5 GB | Required minimum |
| LoRA training activations | 6-10 GB | Gradient checkpointing |
| Optimizer states | 1-2 GB | 8-bit optimizer (adamw8bit) |
| Video latent (49 frames) | 2-3 GB | Reduce frames or resolution |
| Text encoder | 3-5 GB | Cache to disk, CPU offload |
| **Total training** | **14-18 GB** | **May exceed 16GB** |

### If Training OOMs (Out of Memory)

Apply these mitigations in order until it fits:

1. **Enable latent caching** -- Removes VAE encoder from VRAM during training
2. **Enable text encoder caching** -- Removes text encoder from VRAM
3. **Reduce frame count** -- 49 -> 33 -> 17 frames
4. **Reduce resolution** -- 848x480 -> 640x360
5. **Reduce LoRA rank** -- 32 -> 16 -> 8
6. **Reduce batch size** -- Already at 1? Reduce gradient accumulation
7. **Use gradient checkpointing** -- Should already be enabled
8. **Try fp32 accumulation off** -- `--full_fp16` (may hurt quality)

### If Inference OOMs

1. **Enable CPU offloading** -- `pipe.enable_model_cpu_offload()`
2. **Enable sequential CPU offloading** -- `pipe.enable_sequential_cpu_offload()` (slower but uses less VRAM)
3. **Reduce inference steps** -- 40 -> 30 -> 20
4. **Reduce frame count or resolution**

## AMD/ROCm Limitations

### RDNA2 is Unofficially Supported

**Risk:** AMD could break RDNA2 compatibility in any ROCm update.

**Mitigation:**
- Pin your ROCm version once it works. Don't upgrade ROCm unless necessary.
- Pin PyTorch version similarly.
- Document exact working versions in a `VERSIONS.md` lockfile.

```bash
# Record working versions
cat > VERSIONS.md << 'EOF'
# Known Working Versions
ROCm: 6.3.0
PyTorch: 2.4.0+rocm6.2
Python: 3.10.12
Ubuntu: 22.04.4 LTS
bitsandbytes: 0.49.2
HSA_OVERRIDE_GFX_VERSION: 10.3.0
EOF
```

### No xformers

xformers provides memory-efficient attention on NVIDIA but does not support AMD.

**Mitigation:** PyTorch's built-in SDPA (Scaled Dot-Product Attention) provides similar benefits. It's used automatically when xformers is unavailable. Ensure `--sdpa` flag is set in training tools.

### bitsandbytes ROCm is Preview Quality

8-bit optimizer support on ROCm may have edge cases or stability issues.

**Mitigation:**
- If 8-bit optimizers cause NaN losses or crashes, fall back to standard AdamW (fp32 states)
- This uses ~2x more VRAM for optimizer states, so you may need to reduce other settings
- Alternatively, use Adafactor optimizer which has lower memory usage without quantization

### Flash Attention Not Available

The Tri Dao Flash Attention implementation is CUDA-only.

**Mitigation:**
- PyTorch SDPA may use a flash-attention-like kernel internally on ROCm
- Performance will be lower than CUDA Flash Attention but functional
- AMD has their own Flash Attention fork (CK-based) but it targets MI-series datacenter GPUs

### Potential ROCm Memory Fragmentation

ROCm on consumer GPUs can suffer from memory fragmentation more than CUDA.

**Mitigation:**
```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```
This helps PyTorch manage HIP memory more efficiently.

## Model Limitations

### Video Quality at 1.3B Parameters

The Wan 2.1 1.3B model produces lower quality than the 14B version:
- Less detailed textures and faces
- Simpler motion patterns
- Occasional artifacts in complex scenes

**Mitigation:**
- Post-process with upscaling (Real-ESRGAN)
- Post-process with frame interpolation (RIFE)
- Generate multiple variants (different seeds) and pick the best
- Use image-to-video (I2V) with a generated key frame for better control

### Character Consistency Across Clips

No current open-source video model guarantees character consistency across separate generation calls. The same character may look different in each clip.

**Mitigation:**
- Train with consistent, detailed character descriptions in captions
- Use reference image features (I2V mode) to anchor character appearance
- Post-process: cherry-pick the most consistent clips
- Future: IP-Adapter or character-specific LoRA layers may improve this

### Temporal Coherence Limits

Generated clips are limited to ~2-8 seconds. Longer sequences require stitching, which can introduce visual discontinuity at cut points.

**Mitigation:**
- Design scripts around natural scene cuts (match TV editing style)
- Use crossfade transitions between clips
- Generate overlapping clips and blend the overlapping sections

## Dataset Limitations

### Minimum Dataset Size

Video LoRA training typically needs:
- **Minimum:** 50-100 clips for basic style transfer
- **Recommended:** 200-500 clips for good quality
- **Ideal:** 500-1000+ clips for fine-grained style matching

A single TV season (~10-13 episodes, 22-45 min each) should yield 500-2000+ usable clips after scene detection and filtering.

### Caption Quality Matters More Than Quantity

Poor captions lead to poor generation. Common caption failures:
- Too generic ("a person in a room") -- model can't learn specific styles
- Hallucinated details -- model learns incorrect associations
- Inconsistent terminology -- model gets confused

**Mitigation:** Review at least 50-100 captions manually to ensure the VLM is describing scenes accurately. Correct systematic errors and refine the captioning prompt.

### Copyright Considerations

Training on copyrighted TV content for personal, non-commercial research/experimentation exists in a legal grey area. This project is designed for personal use only.

- Never distribute generated content commercially
- Never distribute the trained LoRA adapter publicly
- Never distribute the training dataset
- This is for personal learning and experimentation only

## Fallback Plans

### If Wan 2.1 Doesn't Work on ROCm

```
Fallback 1: Open-Sora 1.2 (~700M params, simpler arch, more likely to work)
Fallback 2: AnimateDiff v3 + SD 1.5 (most tested on AMD, lowest quality)
Fallback 3: Image generation only (SD 1.5/SDXL LoRA, then animate with separate tool)
```

### If Qwen2-VL Doesn't Work on ROCm

```
Fallback 1: Qwen2-VL-2B (smaller, more likely to work)
Fallback 2: InternVL2-8B (different architecture, may have different compat)
Fallback 3: LLaVA-Video-7B
Fallback 4: Caption from individual frames with any image VLM (less accurate)
Fallback 5: Manual captioning (labor intensive but guaranteed)
```

### If bitsandbytes Doesn't Work on ROCm

```
Fallback 1: Adafactor optimizer (low memory without quantization)
Fallback 2: Standard AdamW + reduce LoRA rank aggressively
Fallback 3: Use optimum-quanto for model quantization instead
```

### If 16GB Absolutely Isn't Enough

```
Option 1: Use cloud GPU for training only (vast.ai, runpod -- $0.20-0.50/hr for 24GB)
Option 2: Use the 14B Wan model on cloud, use 1.3B locally for iteration
Option 3: Train AnimateDiff (smaller, guaranteed to fit)
```

## Performance Expectations

### Realistic Timeline for First Results

```
Day 1:     ROCm + PyTorch setup and validation
Day 2:     Download models, process first batch of source video
Day 3:     Run captioning pipeline, review/correct captions
Day 4-5:   First training run, iterate on hyperparameters
Day 6:     First generated clips, evaluate quality
Day 7+:    Iterate on dataset, captions, and training settings
```

### Realistic Quality Expectations

Be honest about what to expect from a 1.3B model on consumer hardware:

- **Style transfer:** Good -- color palette, lighting, general mood will match
- **Scene composition:** Moderate -- general layouts will match, details may vary
- **Character likeness:** Low-moderate -- recognizable style but not photorealistic
- **Motion quality:** Moderate -- basic movements work, complex action may be choppy
- **Text legibility:** Poor -- any in-scene text will be garbled (a universal limitation)

This is experimental/research-grade output, not production quality. The goal is learning the pipeline and producing interesting results, not broadcast-ready footage.
