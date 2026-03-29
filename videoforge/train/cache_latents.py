"""Pre-encode video latents and text encoder outputs for training.

Caching latents to disk avoids loading the VAE encoder during training,
and caching text encoder outputs avoids loading the text encoder.
Both are critical for fitting training within 16GB VRAM.
"""

import json
import logging
from pathlib import Path

import torch
import torchvision.io

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
}


def cache_video_latents(
    model_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "bf16",
    batch_size: int = 1,
    device: str = "cuda:0",
) -> int:
    """Pre-encode all training videos into latent space using the VAE.

    Loads the Wan 2.1 VAE encoder, encodes each video clip, and saves
    the latent tensors to disk. These are loaded during training instead
    of the raw video files.

    Args:
        model_path: Path to the Wan 2.1 model (for VAE weights).
        dataset_dir: Path to dataset with clips_conditioned/ and clip_metadata/.
        output_dir: Path to write cached latent .pt files.
        dtype: Data type for encoding ("bf16", "bfloat16", "fp16", or "fp32").
        batch_size: Number of clips to encode at once.
        device: Torch device string.

    Returns:
        Number of latents cached.
    """
    from diffusers import AutoencoderKLWan

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)

    clips_dir = dataset_dir / "clips_conditioned"
    mp4_files = sorted(clips_dir.glob("*.mp4"))
    if not mp4_files:
        logger.warning("No .mp4 files found in %s", clips_dir)
        return 0

    logger.info("Loading VAE from %s", model_path)
    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch_dtype
    ).to(device)
    vae.eval()

    cached = 0
    for mp4_path in mp4_files:
        out_path = output_dir / f"{mp4_path.stem}.pt"
        if out_path.exists():
            logger.debug("Skipping %s (already cached)", mp4_path.stem)
            continue

        logger.info("Encoding %s (%d/%d)", mp4_path.name, cached + 1, len(mp4_files))

        # Read video: (T, H, W, C) uint8
        video, _, _ = torchvision.io.read_video(str(mp4_path), pts_unit="sec")

        # Normalize 0-255 -> -1 to 1, rearrange to (B, C, T, H, W)
        video = video.to(dtype=torch_dtype, device=device)
        video = video / 127.5 - 1.0
        video = video.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)

        with torch.no_grad():
            latent = vae.encode(video).latent_dist.sample()

        torch.save(latent.cpu(), out_path)
        cached += 1

        del video, latent
        torch.cuda.empty_cache()

    del vae
    torch.cuda.empty_cache()
    logger.info("VAE unloaded, VRAM freed. Cached %d latents.", cached)
    return cached


def cache_text_encoder_outputs(
    model_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "bf16",
    device: str = "cuda:0",
) -> int:
    """Pre-encode all captions through the text encoder (T5-XXL).

    Loads the text encoder, encodes each caption, and saves the
    hidden states to disk. The text encoder can then be offloaded
    during training.

    Args:
        model_path: Path to the Wan 2.1 model (for text encoder weights).
        dataset_dir: Path to dataset with clip_metadata/ containing captions.
        output_dir: Path to write cached text encoder .pt files.
        dtype: Data type for encoding.
        device: Torch device string.

    Returns:
        Number of text embeddings cached.
    """
    from transformers import AutoTokenizer, T5EncoderModel

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)

    metadata_dir = dataset_dir / "clip_metadata"
    meta_files = sorted(metadata_dir.glob("*.json"))
    if not meta_files:
        logger.warning("No metadata JSON files found in %s", metadata_dir)
        return 0

    logger.info("Loading T5 text encoder from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=torch_dtype
    ).to(device)
    text_encoder.eval()

    cached = 0
    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        if not meta.get("filter_passed", True) or meta.get("deleted"):
            continue

        caption = meta.get("caption")
        if not caption:
            logger.debug("Skipping %s (no caption)", meta_path.stem)
            continue

        out_path = output_dir / f"{meta_path.stem}.pt"
        if out_path.exists():
            logger.debug("Skipping %s (already cached)", meta_path.stem)
            continue

        logger.info("Encoding caption for %s (%d/%d)", meta_path.stem, cached + 1, len(meta_files))

        tokens = tokenizer(
            caption,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            hidden_states = text_encoder(**tokens).last_hidden_state

        torch.save(hidden_states.cpu(), out_path)
        cached += 1

        del tokens, hidden_states
        torch.cuda.empty_cache()

    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    logger.info("Text encoder unloaded, VRAM freed. Cached %d embeddings.", cached)
    return cached
