#!/usr/bin/env python3
"""Run two inference tests: step 1000 LoRA vs step 1500 LoRA with improved prompt."""

import os
import gc
import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = os.path.expanduser("~/videoforge/models/wan21-1.3b")

NUM_FRAMES = 33
WIDTH = 320
HEIGHT = 320
FPS = 16
GUIDANCE_SCALE = 6.0
NUM_STEPS = 30
SEED = 42

TESTS = [
    {
        "name": "Step 1000 LoRA",
        "lora": os.path.expanduser(
            "~/videoforge/output/wan21_lora/lora_weights/001000/pytorch_lora_weights.safetensors"
        ),
        "output": os.path.expanduser(
            "~/videoforge/output/inference/test_step1000.mp4"
        ),
        "prompt": (
            "George Costanza sitting in a diner booth, gesturing emphatically "
            "while talking, cinematic lighting"
        ),
        "negative": (
            "blurry, low quality, distorted, watermark, text overlay, "
            "oversaturated, cartoon, anime, 3d render, static image"
        ),
    },
    {
        "name": "Step 1500 LoRA + better prompt",
        "lora": os.path.expanduser(
            "~/videoforge/output/wan21_lora/lora_weights/001500/pytorch_lora_weights.safetensors"
        ),
        "output": os.path.expanduser(
            "~/videoforge/output/inference/test_step1500_better_prompt.mp4"
        ),
        "prompt": (
            "George Costanza sitting in a diner booth, gesturing emphatically "
            "while talking, cinematic lighting, detailed face, sharp features, high quality"
        ),
        "negative": (
            "blurry, low quality, distorted, watermark, text overlay, "
            "oversaturated, cartoon, anime, 3d render, static image, "
            "deformed, blurry, mangled face, distorted, ugly"
        ),
    },
]


def run_test(test):
    print(f"\n{'='*60}")
    print(f"TEST: {test['name']}")
    print(f"{'='*60}")
    print(f"LoRA: {test['lora']}")
    print(f"Prompt: {test['prompt']}")
    print(f"Negative: {test['negative']}")
    print(f"Output: {test['output']}")

    print("\nLoading VAE...")
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
        os.path.dirname(test["lora"]),
        weight_name=os.path.basename(test["lora"]),
    )

    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing("max")

    print(f"Generating {NUM_FRAMES} frames at {WIDTH}x{HEIGHT}...")
    output = pipe(
        prompt=test["prompt"],
        negative_prompt=test["negative"],
        num_frames=NUM_FRAMES,
        width=WIDTH,
        height=HEIGHT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
    )

    export_to_video(output.frames[0], test["output"], fps=FPS)
    print(f"Saved: {test['output']}")

    # Full cleanup between runs
    del output, pipe, vae
    gc.collect()
    torch.cuda.empty_cache()


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    for test in TESTS:
        run_test(test)

    print("\n" + "=" * 60)
    print("DONE — both tests complete.")
    for test in TESTS:
        size = os.path.getsize(test["output"])
        print(f"  {test['output']} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
