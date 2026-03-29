"""Pre-encode video latents and text encoder outputs for training.

Caching latents to disk avoids loading the VAE encoder during training,
and caching text encoder outputs avoids loading the text encoder.
Both are critical for fitting training within 16GB VRAM.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cache_video_latents(
    model_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "fp16",
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
        dtype: Data type for encoding ("fp16" or "fp32").
        batch_size: Number of clips to encode at once.
        device: Torch device string.

    Returns:
        Number of latents cached.
    """
    # TODO: Load Wan 2.1 VAE encoder
    # TODO: Iterate over clips_conditioned/*.mp4
    # TODO: Read video frames, normalize to [-1, 1]
    # TODO: Encode through VAE to get latent tensors
    # TODO: Save latent tensors as .pt files
    # TODO: Free VAE from VRAM when done
    raise NotImplementedError("latent caching not yet implemented")


def cache_text_encoder_outputs(
    model_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "fp16",
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
    # TODO: Load T5-XXL text encoder from Wan 2.1
    # TODO: Iterate over clip_metadata/*.json, read captions
    # TODO: Tokenize and encode captions
    # TODO: Save hidden states as .pt files
    # TODO: Free text encoder from VRAM when done
    raise NotImplementedError("text encoder caching not yet implemented")
