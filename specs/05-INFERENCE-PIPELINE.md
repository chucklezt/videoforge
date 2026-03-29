# 05 - Inference Pipeline: Script-to-Video Generation

## Overview

Take a text script describing a scene, and generate a short video clip that matches the visual style learned during training. This is the payoff stage -- where the trained LoRA adapter produces new content.

## Pipeline Flow

```
[Input Script / Scene Description]
        |
        v
[Stage 1: Script Parsing] -- Break script into per-clip prompts
        |
        v
[Stage 2: Prompt Engineering] -- Add style tags, format for model
        |
        v
[Stage 3: Video Generation] -- Run inference with base model + LoRA
        |
        v
[Stage 4: Post-Processing] -- Upscale, interpolate, stitch clips
        |
        v
[Output Video File]
```

## Stage 1: Script Parsing

### Input Format

Users write scenes in a simple structured format:

```yaml
# scripts/scene_001.yaml
scene:
  title: "The Confrontation"
  clips:
    - description: >
        Interior, dimly lit living room, evening. Sarah sits on the brown
        leather couch, leaning forward with her hands clasped, looking tense.
        The camera is static, medium shot.
      dialogue: "I know what you did."
      duration: 4  # seconds

    - description: >
        Same room. David stands near the window, his back partially turned.
        He slowly turns to face Sarah with a conflicted expression.
        Camera holds steady, slight push in.
      dialogue: "It's not what you think."
      duration: 4

    - description: >
        Close-up of Sarah's face. Her expression shifts from anger to
        hurt. Warm amber lighting from the table lamp highlights her features.
        Static camera.
      dialogue: null
      duration: 3
```

### Script Parser

```python
import yaml

def parse_script(script_path):
    """Parse a scene script into individual clip generation prompts."""
    with open(script_path) as f:
        script = yaml.safe_load(f)

    clips = []
    for clip_def in script["scene"]["clips"]:
        clips.append({
            "description": clip_def["description"].strip(),
            "dialogue": clip_def.get("dialogue"),
            "duration": clip_def.get("duration", 4),
            "scene_title": script["scene"]["title"],
        })
    return clips
```

## Stage 2: Prompt Engineering

Transform scene descriptions into prompts that match the caption format used during training.

```python
def build_generation_prompt(clip_def, style_tags):
    """Build a prompt matching the training caption format."""
    parts = []

    # Style prefix (must match what was used in training captions)
    if style_tags:
        parts.append(", ".join(style_tags))

    # Scene description
    parts.append(clip_def["description"])

    # Dialogue (if provided)
    if clip_def.get("dialogue"):
        parts.append(f'The dialogue is: "{clip_def["dialogue"]}"')

    return " ".join(parts)

# Example output:
# "cinematic television scene, dramatic lighting. Interior, dimly lit
#  living room, evening. Sarah sits on the brown leather couch, leaning
#  forward with her hands clasped, looking tense. The camera is static,
#  medium shot. The dialogue is: "I know what you did.""
```

### Prompt Tips for Best Results

- Mirror the language patterns from your training captions
- Include camera direction (static, pan, zoom, tracking)
- Specify lighting and color palette
- Use the same style tags consistently
- Be specific about character positions and actions
- Describe the setting before the action

## Stage 3: Video Generation

### Option A: Diffusers (Direct Python)

```python
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

def generate_clip(prompt, lora_path, output_path, config):
    """Generate a single video clip."""

    # Load base model
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        torch_dtype=torch.float16,
    )

    # Load trained LoRA
    pipe.load_lora_weights(lora_path)

    # VRAM optimization
    pipe.enable_model_cpu_offload()  # Offload idle components to CPU
    # Do NOT use pipe.enable_xformers() -- not available on AMD
    # SDPA is used automatically by PyTorch on ROCm

    # Generate
    output = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, watermark, text overlay",
        num_frames=config["num_frames"],      # 49 for ~2 sec
        width=config["width"],                 # 848
        height=config["height"],               # 480
        guidance_scale=config["guidance_scale"],  # 5.0-8.0
        num_inference_steps=config["steps"],      # 30-50
        generator=torch.Generator(device="cuda").manual_seed(config.get("seed", 42)),
    )

    # Save
    export_to_video(output.frames[0], output_path, fps=config["fps"])
    print(f"Generated: {output_path}")

    # Free VRAM
    del pipe
    torch.cuda.empty_cache()
```

### Option B: ComfyUI (Recommended for Iteration)

ComfyUI has excellent AMD/ROCm support and a node-based workflow that makes it easy to experiment with generation parameters.

```
Installation:
  git clone https://github.com/comfyanonymous/ComfyUI.git
  cd ComfyUI
  pip install -r requirements.txt
  # PyTorch ROCm is already installed in the venv

  # Install video generation nodes
  cd custom_nodes
  git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
  git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git

Usage:
  python main.py --listen 0.0.0.0 --port 8188
  # Access via browser at http://<server-ip>:8188
```

ComfyUI workflow for Wan 2.1 + LoRA:
1. Load Wan 2.1 checkpoint
2. Load LoRA adapter
3. Text encode prompt
4. Sample video latent
5. Decode to video
6. Export to MP4

ComfyUI also supports the API for scripted generation:

```python
import json
import requests

def generate_via_comfyui(prompt, workflow_path, server="http://localhost:8188"):
    """Submit a generation job to ComfyUI API."""
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Update prompt in workflow
    # (node IDs depend on your specific workflow)
    workflow["6"]["inputs"]["text"] = prompt

    response = requests.post(
        f"{server}/prompt",
        json={"prompt": workflow}
    )
    return response.json()
```

### Generation Parameters

```yaml
# config/inference.yaml
inference:
  model: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  lora_path: "./output/wan21_lora/wan21_show_style.safetensors"
  lora_weight: 0.8             # LoRA strength (0.6-1.0, lower = more base model)

  # Video dimensions (match training)
  width: 848
  height: 480
  fps: 24
  num_frames: 49               # ~2 seconds

  # Sampling
  guidance_scale: 6.0          # CFG scale (5-8 typical)
  num_inference_steps: 40      # More steps = better quality, slower
  scheduler: euler_a           # euler_a, ddpm, dpm++

  # Negative prompt
  negative_prompt: >
    blurry, low quality, distorted, watermark, text overlay,
    oversaturated, cartoon, anime, 3d render, static image

  # Batch generation
  seeds: [42, 123, 456]        # Generate multiple variants per prompt

  # Output
  output_dir: "./generated/"
  output_format: mp4
  output_codec: libx264
  output_crf: 18
```

## Stage 4: Post-Processing

### Frame Interpolation (Optional)

Increase smoothness by interpolating between generated frames:

```bash
# RIFE (Real-Time Intermediate Flow Estimation) -- open source
pip install rife-ncnn-vulkan-python
# or
git clone https://github.com/hzwer/Practical-RIFE.git

# Double the frame rate (24fps -> 48fps)
python inference_video.py --exp=1 --video=generated_clip.mp4 --output=interpolated_clip.mp4
```

### Upscaling (Optional)

If generating at 480p, upscale to 720p or 1080p:

```bash
# Real-ESRGAN for video upscaling
pip install realesrgan

# 2x upscale (480p -> 960p)
python inference_realesrgan_video.py \
  -i generated_clip.mp4 \
  -o upscaled_clip.mp4 \
  -s 2 \
  --face_enhance  # Optional: enhance faces
```

Note: Real-ESRGAN may need testing on ROCm. CPU fallback works but is slower.

### Clip Stitching

Combine multiple generated clips into a sequence:

```python
import subprocess

def stitch_clips(clip_paths, output_path, transition="none"):
    """Concatenate clips into a single video."""
    # Create file list for FFmpeg
    list_file = "/tmp/clip_list.txt"
    with open(list_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")

    if transition == "none":
        # Simple concatenation
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264", "-crf", "18",
            output_path
        ]
    elif transition == "crossfade":
        # Crossfade between clips (more complex FFmpeg filter)
        # Build filter chain dynamically based on clip count
        pass  # See FFmpeg xfade filter documentation

    subprocess.run(cmd, check=True)
```

### Add Audio (Future Enhancement)

Audio generation is a separate concern but could be added later:
- **Dialogue:** TTS with voice cloning (e.g., Coqui TTS, Bark)
- **Music/ambient:** MusicGen or AudioCraft
- **Sound effects:** AudioGen

```bash
# Merge generated video with audio track
ffmpeg -i video.mp4 -i audio.wav \
  -c:v copy -c:a aac -b:a 128k \
  output_with_audio.mp4
```

## Full Generation CLI

```bash
# Generate all clips from a script
python -m videoforge.generate \
  --script scripts/scene_001.yaml \
  --config configs/inference.yaml \
  --output generated/scene_001/

# Generate a single clip from a text prompt
python -m videoforge.generate \
  --prompt "cinematic television scene. A man walks into a dark office and sits at the desk." \
  --config configs/inference.yaml \
  --output generated/test_clip.mp4

# Generate with multiple seeds for variation
python -m videoforge.generate \
  --script scripts/scene_001.yaml \
  --config configs/inference.yaml \
  --seeds 42 123 456 789 \
  --output generated/scene_001_variants/

# Stitch clips into final scene
python -m videoforge.postprocess.stitch \
  --input generated/scene_001/ \
  --output generated/scene_001_final.mp4 \
  --transition crossfade \
  --transition_duration 0.5
```

## Generation Time Estimates

On RX 6800 XT (16GB) with Wan 2.1 1.3B + LoRA:

```
Per clip (49 frames, 480p, 40 steps):  ~60-120 seconds
Per clip (49 frames, 480p, 30 steps):  ~45-90 seconds
Per clip (33 frames, 480p, 30 steps):  ~30-60 seconds

A 3-clip scene (~12 seconds total):    ~3-6 minutes
Post-processing (interpolation + stitch): ~1-2 minutes
```

These are estimates -- actual performance depends on ROCm optimization and model-specific factors.
