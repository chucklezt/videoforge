#!/usr/bin/env python3
"""Run three inference experiments with step 1500 LoRA."""

import os
import gc
import time
import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = os.path.expanduser("~/videoforge/models/wan21-1.3b")
LORA_PATH = os.path.expanduser(
    "~/videoforge/output/wan21_lora/lora_weights/001500/pytorch_lora_weights.safetensors"
)

NUM_FRAMES = 33
FPS = 16
SEED = 42

BASE_PROMPT = (
    "George Costanza sitting in a booth at Monk's Diner, a classic New York "
    "coffee shop with vinyl booths and a counter, gesturing emphatically while "
    "talking, casual clothes, detailed face, sharp features, high quality, "
    "TV sitcom lighting"
)
BASE_NEGATIVE = (
    "deformed, blurry, mangled face, distorted, ugly, fancy restaurant, "
    "suit jacket, formal dining, upscale"
)

EXPERIMENTS = [
    {
        "name": "Exp 1 — 480x480, 40 steps, cfg 7.5",
        "output": os.path.expanduser("~/videoforge/output/inference/exp1_resolution.mp4"),
        "width": 480,
        "height": 480,
        "steps": 40,
        "guidance_scale": 7.5,
        "prompt": BASE_PROMPT,
        "negative": BASE_NEGATIVE,
    },
    {
        "name": "Exp 2 — 480x480, 50 steps, cfg 8.0",
        "output": os.path.expanduser("~/videoforge/output/inference/exp2_steps.mp4"),
        "width": 480,
        "height": 480,
        "steps": 50,
        "guidance_scale": 8.0,
        "prompt": BASE_PROMPT,
        "negative": BASE_NEGATIVE,
    },
    {
        "name": "Exp 3 — 480x480, 40 steps, cfg 6.5, Seinfeld prompt",
        "output": os.path.expanduser("~/videoforge/output/inference/exp3_guidance.mp4"),
        "width": 480,
        "height": 480,
        "steps": 40,
        "guidance_scale": 6.5,
        "prompt": (
            BASE_PROMPT + ", Seinfeld TV show aesthetic, 1990s sitcom, "
            "diner booth, coffee cup on table"
        ),
        "negative": BASE_NEGATIVE,
    },
]


def run_experiment(exp):
    print(f"\n{'='*60}")
    print(f"{exp['name']}")
    print(f"{'='*60}")
    print(f"Resolution: {exp['width']}x{exp['height']}")
    print(f"Steps: {exp['steps']}, CFG: {exp['guidance_scale']}")
    print(f"Prompt: {exp['prompt']}")
    print(f"Negative: {exp['negative']}")

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
        os.path.dirname(LORA_PATH),
        weight_name=os.path.basename(LORA_PATH),
    )

    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing("max")

    print(f"Generating {NUM_FRAMES} frames at {exp['width']}x{exp['height']}...")
    t0 = time.time()
    output = pipe(
        prompt=exp["prompt"],
        negative_prompt=exp["negative"],
        num_frames=NUM_FRAMES,
        width=exp["width"],
        height=exp["height"],
        guidance_scale=exp["guidance_scale"],
        num_inference_steps=exp["steps"],
        generator=torch.Generator(device="cpu").manual_seed(SEED),
    )
    elapsed = time.time() - t0

    export_to_video(output.frames[0], exp["output"], fps=FPS)
    size_kb = os.path.getsize(exp["output"]) / 1024
    print(f"Saved: {exp['output']} ({size_kb:.0f} KB)")
    print(f"Inference time: {elapsed:.1f}s ({elapsed/exp['steps']:.2f}s/step)")

    del output, pipe, vae
    gc.collect()
    torch.cuda.empty_cache()

    return {"name": exp["name"], "time": elapsed, "size_kb": size_kb, "steps": exp["steps"]}


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp)
        results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(
            f"  {r['name']}: {r['time']:.1f}s total, "
            f"{r['time']/r['steps']:.2f}s/step, {r['size_kb']:.0f} KB"
        )


if __name__ == "__main__":
    main()
