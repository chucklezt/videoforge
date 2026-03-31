#!/usr/bin/env python3
"""Experiment 5: Longer segment with George Costanza physical description."""

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
OUTPUT_PATH = os.path.expanduser(
    "~/videoforge/output/inference/experiment5_george_long.mp4"
)

PROMPT = (
    "George Costanza from Seinfeld in the diner speaking to Jerry. "
    "Jerry has his back to us and you can see George as he gives a calm lecture. "
    "George says, I can do whatever I want with AI, and open source is where its at. "
    "No one can tell me what I can or cannot do, pants optional. "
    "Keep the same proportions of the face and shape of the head as George tends to "
    "be a short stocky bald man. "
    "Seinfeld TV show aesthetic, 1990s sitcom, Monk's Diner, a classic New York "
    "coffee shop with vinyl booths and a counter, diner booth, coffee cup on table, "
    "casual clothes, TV sitcom lighting, "
    "detailed face, sharp facial features, expressive eyes, high quality, "
    "cinematic lighting, close-up facial detail, "
    "round head, bald, stocky build, short man"
)
NEGATIVE_PROMPT = (
    "deformed, blurry, mangled face, distorted, ugly, disfigured, "
    "fancy restaurant, suit jacket, formal dining, upscale, "
    "low quality, watermark, text overlay, oversaturated, cartoon, anime, 3d render, "
    "extra fingers, extra limbs, missing eyes, featureless face, "
    "tall, thin, full head of hair, long hair"
)

# 81 frames at 320x320 = ~34K attention tokens, similar to 33@480 which works
NUM_FRAMES = 81
WIDTH = 320
HEIGHT = 320
FPS = 16
GUIDANCE_SCALE = 6.5
NUM_STEPS = 60
SEED = 42


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n{'='*60}")
    print(f"Experiment 5 — {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames "
          f"({NUM_FRAMES/FPS:.1f}s), {NUM_STEPS} steps, cfg {GUIDANCE_SCALE}")
    print(f"{'='*60}")
    print(f"Prompt: {PROMPT}")
    print(f"Negative: {NEGATIVE_PROMPT}")

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

    print(f"Generating {NUM_FRAMES} frames at {WIDTH}x{HEIGHT}, {NUM_STEPS} steps...")
    t0 = time.time()
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
    elapsed = time.time() - t0

    export_to_video(output.frames[0], OUTPUT_PATH, fps=FPS)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nSaved: {OUTPUT_PATH} ({size_kb:.0f} KB)")
    print(f"Inference time: {elapsed:.1f}s ({elapsed/NUM_STEPS:.2f}s/step)")
    print(f"Frames: {NUM_FRAMES}, Duration: {NUM_FRAMES/FPS:.1f}s")

    del output, pipe, vae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
