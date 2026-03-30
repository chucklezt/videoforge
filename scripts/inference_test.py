#!/usr/bin/env python3
"""Wan 2.1 T2V inference with LoRA on ROCm/HIP."""

import os
import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = os.path.expanduser("~/videoforge/models/wan21-1.3b")
LORA_PATH = os.path.expanduser(
    "~/videoforge/output/wan21_lora/lora_weights/001500/pytorch_lora_weights.safetensors"
)
OUTPUT_PATH = os.path.expanduser("~/videoforge/output/inference/test01.mp4")

PROMPT = (
    "George Costanza sitting in a diner booth, gesturing emphatically "
    "while talking, cinematic lighting"
)
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, text overlay, "
    "oversaturated, cartoon, anime, 3d render, static image"
)

# Aggressive VRAM reduction: 33 frames at 320x320
# 33 frames / 16 fps ≈ 2 sec (shorter but fits 16GB)
NUM_FRAMES = 33
WIDTH = 320
HEIGHT = 320
FPS = 16
GUIDANCE_SCALE = 6.0
NUM_STEPS = 30
SEED = 42


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.float32
    )

    print("Loading pipeline...")
    pipe = WanPipeline.from_pretrained(
        MODEL_PATH,
        vae=vae,
        torch_dtype=torch.float16,
    )

    print("Loading LoRA weights...")
    pipe.load_lora_weights(
        os.path.dirname(LORA_PATH),
        weight_name=os.path.basename(LORA_PATH),
    )

    # Sequential CPU offload: each submodule moves to GPU only when needed
    # Slower than enable_model_cpu_offload() but uses far less peak VRAM
    pipe.enable_sequential_cpu_offload()

    # Slice attention computation to reduce peak VRAM during self-attention
    pipe.enable_attention_slicing("max")

    print(f"Generating {NUM_FRAMES} frames at {WIDTH}x{HEIGHT}...")
    output = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=NUM_FRAMES,
        width=WIDTH,
        height=HEIGHT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
    )

    export_to_video(output.frames[0], OUTPUT_PATH, fps=FPS)
    print(f"Saved: {OUTPUT_PATH}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
