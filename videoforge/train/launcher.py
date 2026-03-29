"""Launch and monitor training jobs."""

import logging
import os
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
    from videoforge.train.config_builder import build_training_command

    cmd = build_training_command(
        config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        resume_from=resume_from,
        framework=framework,
    )

    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

    logger.info("Training command: %s", " ".join(cmd))

    if dry_run:
        logger.info("[dry run] Would execute the above command")
        return 0

    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        logger.info("Training interrupted")
        return 1


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

    # Check model weights are downloaded
    model_name = config.get("model", {}).get("name", "")
    if model_name:
        model_path = Path(model_name)
        if model_path.exists() or model_name.startswith("/") or model_name.startswith("./"):
            if not model_path.exists():
                errors.append(f"Model path not found: {model_name}")
        else:
            local_model_dirs = [
                Path.home() / "videoforge/models/wan21-1.3b",
                Path(model_name),
            ]
            if any(p.exists() for p in local_model_dirs):
                pass  # local model found, skip cache check
            else:
                from huggingface_hub import try_to_load_from_cache
                cached = try_to_load_from_cache(model_name, "model_index.json")
                if cached is None or (isinstance(cached, str) and not Path(cached).exists()):
                    errors.append(
                        f"Model '{model_name}' not found in HuggingFace cache. "
                        f"Run: huggingface-cli download {model_name}"
                    )

    # Check VRAM availability
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            if vram_gb < 12:
                errors.append(
                    f"Insufficient VRAM: {vram_gb:.1f}GB available, 12GB minimum required"
                )
        else:
            errors.append("No CUDA/ROCm GPU detected")
    except ImportError:
        errors.append("PyTorch not installed")

    # Check training framework is installed
    try:
        import accelerate  # noqa: F401
    except ImportError:
        errors.append(
            "accelerate not installed. Run: pip install accelerate"
        )

    return errors
