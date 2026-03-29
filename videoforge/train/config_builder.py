"""Build training configs for kohya-ss/sd-scripts and OneTrainer."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_kohya_config(
    videoforge_config: dict[str, Any],
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Convert a VideoForge training config to kohya-ss dataset_config.toml format.

    Args:
        videoforge_config: Parsed VideoForge train_wan21_lora.yaml config.
        dataset_dir: Path to the dataset directory (clips + captions).
        output_dir: Path for training output (checkpoints, samples).

    Returns:
        Dict representing the kohya-ss TOML config.
    """
    # TODO: Map VideoForge config keys to kohya-ss dataset_config.toml format
    # TODO: Generate accelerate launch arguments
    # TODO: Handle resolution buckets from conditioning config
    raise NotImplementedError("kohya-ss config builder not yet implemented")


def build_onetrainer_config(
    videoforge_config: dict[str, Any],
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Convert a VideoForge training config to OneTrainer JSON format.

    Args:
        videoforge_config: Parsed VideoForge train_wan21_lora.yaml config.
        dataset_dir: Path to the dataset directory (clips + captions).
        output_dir: Path for training output.

    Returns:
        Dict representing the OneTrainer JSON config.
    """
    # TODO: Map VideoForge config keys to OneTrainer JSON format
    raise NotImplementedError("OneTrainer config builder not yet implemented")


def build_accelerate_args(
    videoforge_config: dict[str, Any],
) -> list[str]:
    """Build command-line arguments for accelerate launch.

    Args:
        videoforge_config: Parsed VideoForge training config.

    Returns:
        List of CLI arguments for accelerate launch.
    """
    # TODO: Build accelerate launch args from config
    # - --mixed_precision fp16
    # - --num_processes 1
    raise NotImplementedError("accelerate args builder not yet implemented")


def build_training_command(
    videoforge_config: dict[str, Any],
    dataset_dir: str | Path,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
    framework: str = "kohya",
) -> list[str]:
    """Build the full training command line.

    Args:
        videoforge_config: Parsed VideoForge training config.
        dataset_dir: Path to dataset.
        output_dir: Path for output.
        resume_from: Optional checkpoint path to resume from.
        framework: Training framework ("kohya" or "onetrainer").

    Returns:
        Full command as a list of strings.
    """
    # TODO: Combine accelerate args + training script + config args
    # TODO: Add --resume flag if resume_from is provided
    # TODO: Add --sdpa flag (no xformers on AMD)
    raise NotImplementedError("training command builder not yet implemented")
