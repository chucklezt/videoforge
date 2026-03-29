"""Launch and monitor training jobs."""

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def launch_training(
    config: dict[str, Any],
    dataset_dir: str | Path,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
    framework: str = "kohya",
    dry_run: bool = False,
) -> int:
    """Launch a LoRA training job.

    Builds the training command from config, optionally prints it
    (dry run), then executes it as a subprocess.

    Args:
        config: Parsed VideoForge training config.
        dataset_dir: Path to the prepared dataset.
        output_dir: Path for training output (checkpoints, samples, logs).
        resume_from: Optional checkpoint path to resume training.
        framework: Training framework ("kohya" or "onetrainer").
        dry_run: If True, print the command without executing.

    Returns:
        Process return code (0 = success).
    """
    # TODO: Call config_builder.build_training_command()
    # TODO: Set ROCm environment variables (HSA_OVERRIDE_GFX_VERSION, etc.)
    # TODO: Log the full command
    # TODO: If dry_run, print and return 0
    # TODO: Execute via subprocess.run()
    # TODO: Handle KeyboardInterrupt for graceful stop
    raise NotImplementedError("training launcher not yet implemented")


def validate_training_prereqs(
    config: dict[str, Any],
    dataset_dir: str | Path,
) -> list[str]:
    """Check that all prerequisites for training are met.

    Args:
        config: Parsed VideoForge training config.
        dataset_dir: Path to the dataset.

    Returns:
        List of error messages (empty if all checks pass).
    """
    errors = []
    dataset_dir = Path(dataset_dir)

    # Check dataset exists
    clips_dir = dataset_dir / "clips_conditioned"
    metadata_dir = dataset_dir / "clip_metadata"
    if not clips_dir.exists():
        errors.append(f"Clips directory not found: {clips_dir}")
    if not metadata_dir.exists():
        errors.append(f"Metadata directory not found: {metadata_dir}")

    # Check captions exist
    if metadata_dir.exists():
        import json
        captioned = 0
        total = 0
        for meta_path in metadata_dir.glob("*.json"):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("filter_passed", True) and not meta.get("deleted"):
                total += 1
                if meta.get("caption"):
                    captioned += 1
        if total == 0:
            errors.append("No valid clips found in dataset")
        elif captioned == 0:
            errors.append("No clips have captions. Run the captioning pipeline first.")
        elif captioned < total:
            logger.warning("%d of %d clips are missing captions", total - captioned, total)

    # Check .txt sidecar files exist
    if clips_dir.exists():
        txt_files = list(clips_dir.glob("*.txt"))
        mp4_files = list(clips_dir.glob("*.mp4"))
        if mp4_files and not txt_files:
            errors.append(
                "No .txt caption files found alongside clips. "
                "Run: python -m videoforge caption export --dataset <path>"
            )

    # TODO: Check model weights are downloaded
    # TODO: Check VRAM availability
    # TODO: Check training framework is installed (kohya-ss or OneTrainer)

    return errors
