#!/usr/bin/env python3
"""Experiment 9: 129 frames at 320x240 with VRAM monitoring. Reverting to negative in Exp 4"""

import os
import gc
import time
import threading
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
    "~/videoforge/output/inference/experiment9_george_129f.mp4"
)

PROMPT = (
    "seinfeld sitcom scene, multicam sitcom lighting, 1990s television, george costanza, "
    "george costanza a short stocky bald man wearing glasses and a plaid shirt, "
    "seated at a diner booth with jerry whose back is turned to the camera, "
    "the setting is a cozy diner with warm inviting ambiance, wooden booths and a counter adorned with diner paraphernalia, "
    "red cushioned seat, cash register and condiments on the counter, menu holder, "
    "soft natural lighting casting gentle shadows, "
    "george engaged in conversation leaning slightly forward as he speaks, "
    "expression animated and expressive indicating active discussion, "
    "camera angle close-up focusing on george's face and upper body capturing gestures and facial expressions vividly, "
    "rich earthy color palette enhancing the intimate casual atmosphere, "
    "george gives a stern lecture with an exasperated neurotic expression, hands gesturing emphatically"
)
NEGATIVE_PROMPT = (
    "deformed, blurry, mangled face, distorted, ugly, disfigured, "
    "fancy restaurant, suit jacket, formal dining, upscale, "
    "low quality, watermark, text overlay, oversaturated, cartoon, anime, 3d render, "
    "extra fingers, extra limbs, missing eyes, featureless fac"
)

NUM_FRAMES = 129
WIDTH = 320
HEIGHT = 240
FPS = 16
GUIDANCE_SCALE = 7.5
NUM_STEPS = 60
SEED = 42


# --- VRAM monitoring ---
class VRAMMonitor:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.running = False
        self.log = []  # (timestamp, allocated_mb, reserved_mb, peak_mb)
        self.peak_allocated = 0
        self.peak_reserved = 0
        self._thread = None

    def _sample(self):
        while self.running:
            alloc = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            self.peak_allocated = max(self.peak_allocated, alloc)
            self.peak_reserved = max(self.peak_reserved, reserved)
            self.log.append((time.time(), alloc, reserved))
            time.sleep(self.interval)

    def start(self):
        torch.cuda.reset_peak_memory_stats()
        self.running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()

    def report(self):
        hw_peak = torch.cuda.max_memory_allocated() / 1e6
        hw_peak_reserved = torch.cuda.max_memory_reserved() / 1e6
        total = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"\n{'='*60}")
        print("VRAM REPORT")
        print(f"{'='*60}")
        print(f"Total GPU memory:      {total:,.0f} MB")
        print(f"Peak allocated:        {hw_peak:,.0f} MB  (PyTorch tracker)")
        print(f"Peak reserved:         {hw_peak_reserved:,.0f} MB  (PyTorch tracker)")
        print(f"Peak allocated (poll): {self.peak_allocated:,.0f} MB")
        print(f"Peak reserved (poll):  {self.peak_reserved:,.0f} MB")
        print(f"Headroom at peak:      {total - hw_peak_reserved:,.0f} MB")
        print(f"Utilization at peak:   {hw_peak_reserved / total * 100:.1f}%")
        print(f"Samples collected:     {len(self.log)}")

        # Print phase breakdown from samples
        if self.log:
            print(f"\nVRAM timeline (sampled every {self.interval}s):")
            print(f"{'Time':>8s}  {'Allocated':>10s}  {'Reserved':>10s}")
            # Print every 10th sample to keep it readable
            step = max(1, len(self.log) // 30)
            for i in range(0, len(self.log), step):
                t, a, r = self.log[i]
                elapsed = t - self.log[0][0]
                print(f"{elapsed:7.1f}s  {a:9.0f} MB  {r:9.0f} MB")


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"VRAM: {total_mb:,.0f} MB")

    print(f"\n{'='*60}")
    print(f"Experiment 9 — {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames "
          f"({NUM_FRAMES/FPS:.1f}s), {NUM_STEPS} steps, cfg {GUIDANCE_SCALE}")
    print(f"{'='*60}")
    print(f"Prompt: {PROMPT[:100]}...")
    print(f"Negative: {NEGATIVE_PROMPT[:80]}...")

    monitor = VRAMMonitor(interval=2.0)
    monitor.start()

    print("\nLoading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.float32
    )
    print(f"  [after VAE load] allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    print("Loading pipeline...")
    pipe = WanPipeline.from_pretrained(
        MODEL_PATH,
        vae=vae,
        torch_dtype=torch.float16,
    )
    print(f"  [after pipeline load] allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    print("Loading LoRA weights...")
    pipe.load_lora_weights(
        os.path.dirname(LORA_PATH),
        weight_name=os.path.basename(LORA_PATH),
    )
    print(f"  [after LoRA load] allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing("max")
    print(f"  [after offload setup] allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    print(f"\nGenerating {NUM_FRAMES} frames at {WIDTH}x{HEIGHT}, {NUM_STEPS} steps...")
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
    print(f"  [after generation] allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    export_to_video(output.frames[0], OUTPUT_PATH, fps=FPS)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024

    monitor.stop()

    print(f"\nSaved: {OUTPUT_PATH} ({size_kb:.0f} KB)")
    print(f"Inference time: {elapsed:.1f}s ({elapsed/NUM_STEPS:.2f}s/step)")
    print(f"Frames: {NUM_FRAMES}, Duration: {NUM_FRAMES/FPS:.1f}s")

    monitor.report()

    del output, pipe, vae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
