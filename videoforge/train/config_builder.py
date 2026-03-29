"""Build training configs for kohya-ss/sd-scripts and OneTrainer."""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_kohya_config(
    videoforge_config: dict[str, Any],
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Convert a VideoForge training config to kohya-ss dataset_config.toml format."""
    raise NotImplementedError("kohya-ss config builder not yet implemented")


def build_onetrainer_config(
    videoforge_config: dict[str, Any],
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Convert a VideoForge training config to OneTrainer JSON format."""
    raise NotImplementedError("OneTrainer config builder not yet implemented")


SCRIPT_NAMES = ["train_cogvideox_lora.py", "train_wan_lora.py", "train_wan_t2v_lora.py"]


def find_diffusers_training_script() -> str:
    """Find the Wan video LoRA training script in a cloned diffusers repo.

    Searches these locations in order:
    1. ~/diffusers
    2. ~/repos/diffusers
    3. ./diffusers
    4. DIFFUSERS_REPO_PATH environment variable

    Returns:
        Path to the training script.

    Raises:
        FileNotFoundError: If the script cannot be found.
    """
    home = Path.home()
    repo_roots = [
        home / "diffusers",
        home / "repos" / "diffusers",
        Path("./diffusers").resolve(),
    ]

    # Script subdirectories to search within each repo root
    script_dirs = ["examples/cogvideo", "examples/wan"]

    env_path = os.environ.get("DIFFUSERS_REPO_PATH")
    if env_path:
        repo_roots.append(Path(env_path))

    searched = []
    for repo_root in repo_roots:
        for script_dir in script_dirs:
            for script_name in SCRIPT_NAMES:
                candidate = repo_root / script_dir / script_name
                searched.append(candidate)
                if candidate.exists():
                    return str(candidate)

    raise FileNotFoundError(
        "Wan LoRA training script not found. The diffusers pip package does not "
        "include training examples -- you need the full repo.\n\n"
        "Clone it:\n"
        "  git clone https://github.com/huggingface/diffusers.git ~/diffusers\n\n"
        "Or set DIFFUSERS_REPO_PATH to your existing clone:\n"
        "  export DIFFUSERS_REPO_PATH=/path/to/diffusers\n\n"
        "Searched:\n" + "\n".join(f"  {c}" for c in searched)
    )


def _get(config: dict, dotted_key: str, default=None):
    """Get a nested config value using dot notation."""
    keys = dotted_key.split(".")
    val = config
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k)
        if val is None:
            return default
    return val


def build_accelerate_args(
    videoforge_config: dict[str, Any],
) -> list[str]:
    """Build command-line arguments for accelerate launch.

    Args:
        videoforge_config: Parsed VideoForge training config.

    Returns:
        List of CLI arguments for accelerate launch.
    """
    mixed_precision = _get(videoforge_config, "training.mixed_precision", "bf16")

    return [
        "accelerate", "launch",
        "--mixed_precision", str(mixed_precision),
        "--num_processes", "1",
    ]


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
    cmd = build_accelerate_args(videoforge_config)

    script_path = find_diffusers_training_script()
    cmd.append(script_path)

    cmd.extend([
        "--pretrained_model_name_or_path", str(_get(videoforge_config, "model.name", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")),
        "--instance_data_dir", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--rank", str(_get(videoforge_config, "lora.rank", 16)),
        "--learning_rate", str(_get(videoforge_config, "training.learning_rate", 1e-4)),
        "--max_train_steps", str(_get(videoforge_config, "training.max_train_steps", 1500)),
        "--gradient_accumulation_steps", str(_get(videoforge_config, "training.gradient_accumulation", 4)),
        "--seed", str(_get(videoforge_config, "training.seed", 42)),
        "--optimizer_type", str(_get(videoforge_config, "optimizer.name", "adamw")),
        "--caption_column", "text",
        "--video_column", "video",
        "--sdpa",
        "--enable_slicing",
        "--enable_tiling",
    ])

    if _get(videoforge_config, "training.gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")

    if resume_from is not None:
        cmd.extend(["--resume_from_checkpoint", str(resume_from)])

    return cmd
