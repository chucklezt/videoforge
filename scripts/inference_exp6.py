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
    "~/videoforge/output/inference/experiment6_george_detailed.mp4"
)

PROMPT = (
    "George Costanza from Seinfeld sitting in a vinyl booth at Monk's Diner, "
    "a classic 1990s New York greasy spoon coffee shop with worn Formica tables, fluorescent lighting, "
    "paper napkin dispenser, ketchup bottle, salt and pepper shakers, and a counter in the background. "
    "George is short and stocky with a round full face, bald on top with a horseshoe pattern of dark hair on the sides, "
    "heavy lidded slightly squinting eyes, prominent nose, weak chin, light Mediterranean complexion. "
    "He wears casual uncool slacks and an open collar shirt, poorly dressed. "
    "He faces the camera diagonally at eye level, medium shot, gesturing emphatically with his hands, mouth open mid-sentence, "
    "giving a stern lecture with an exasperated neurotic expression, hands moving, head bobbing slightly. "
    "Jerry sits across with his back to the camera, only the back of his head and one shoulder visible. "
    "Seinfeld TV show aesthetic, 1990s television sitcom, VHS era color grading, warm diner lighting, "
    "coffee cup and saucer on the table, shallow depth of field, subject in foreground, "
    "detailed face, sharp facial features, expressive heavy lidded eyes, high quality"
)
NEGATIVE_PROMPT = (
    "deformed, blurry, mangled face, distorted, ugly, disfigured, "
    "fancy restaurant, suit jacket, formal dining, upscale restaurant, white tablecloth, "
    "low quality, watermark, text overlay, oversaturated, cartoon, anime, 3d render, "
    "extra fingers, extra limbs, missing eyes, featureless face, "
    "tall, thin, full head of hair, long hair, completely bald, no hair at all, "
    "good looking, handsome, attractive, well dressed, stylish, cool clothes, "
    "tie, blazer, dress shirt, sweater, turtleneck, "
    "slim face, chiseled jaw, narrow face, young, muscular, athletic, "
    "outdoor setting, living room, office, bar, nightclub, apartment"
)

# 81 frames at 320x320 = ~34K attention tokens, similar to 33@480 which works
NUM_FRAMES = 81
WIDTH = 320
HEIGHT = 240
FPS = 18
GUIDANCE_SCALE = 7.5
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
