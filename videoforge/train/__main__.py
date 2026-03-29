"""Training pipeline CLI - run via: python -m videoforge.train"""

import argparse
import logging
import sys
from pathlib import Path

from videoforge.utils.config import load_config, get_nested

logger = logging.getLogger("videoforge.train")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_train(args):
    """Run LoRA training."""
    config = load_config(args.config) if args.config else {}

    dataset_dir = Path(args.dataset or get_nested(config, "dataset.path", "./dataset"))
    output_dir = Path(args.output or get_nested(config, "saving.output_dir", "./output/wan21_lora"))

    # Validate prerequisites
    from videoforge.train.launcher import validate_training_prereqs
    errors = validate_training_prereqs(config, dataset_dir)
    if errors:
        for err in errors:
            logger.error(err)
        sys.exit(1)

    logger.info("Training configuration:")
    logger.info("  Dataset: %s", dataset_dir)
    logger.info("  Output: %s", output_dir)
    logger.info("  Model: %s", get_nested(config, "model.name", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
    logger.info("  LoRA rank: %s", get_nested(config, "lora.rank", 32))
    logger.info("  Steps: %s", get_nested(config, "training.max_train_steps", 3000))
    logger.info("  Batch size: %s (x%s accumulation)",
                get_nested(config, "training.batch_size", 1),
                get_nested(config, "training.gradient_accumulation", 4))

    if args.resume:
        logger.info("  Resuming from: %s", args.resume)

    from videoforge.train.launcher import launch_training
    framework = args.framework or "kohya"

    return_code = launch_training(
        config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        resume_from=args.resume,
        framework=framework,
        dry_run=args.dry_run,
    )

    if return_code != 0:
        logger.error("Training exited with code %d", return_code)
        sys.exit(return_code)

    logger.info("Training complete. Output at: %s", output_dir)


def cmd_cache(args):
    """Pre-cache latents and text encoder outputs."""
    config = load_config(args.config) if args.config else {}
    dataset_dir = Path(args.dataset or get_nested(config, "dataset.path", "./dataset"))
    model_path = args.model or get_nested(config, "model.name", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    from videoforge.train.cache_latents import cache_video_latents, cache_text_encoder_outputs

    if args.type in ("latents", "all"):
        latent_dir = dataset_dir / "latent_cache"
        logger.info("Caching video latents to %s", latent_dir)
        if not args.dry_run:
            count = cache_video_latents(model_path, dataset_dir, latent_dir)
            logger.info("Cached %d video latents", count)

    if args.type in ("text", "all"):
        te_dir = dataset_dir / "te_cache"
        logger.info("Caching text encoder outputs to %s", te_dir)
        if not args.dry_run:
            count = cache_text_encoder_outputs(model_path, dataset_dir, te_dir)
            logger.info("Cached %d text embeddings", count)


def main():
    parser = argparse.ArgumentParser(
        prog="videoforge.train",
        description="VideoForge Training Pipeline - LoRA fine-tuning for video generation",
    )
    parser.add_argument("--config", type=str, help="YAML config file (e.g. configs/train_wan21_lora.yaml)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")

    subparsers = parser.add_subparsers(dest="stage")

    # Default: train
    parser.add_argument("--dataset", type=str, help="Dataset directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--framework", type=str, choices=["kohya", "onetrainer"],
                        help="Training framework (default: kohya)")

    # Cache subcommand
    sub_cache = subparsers.add_parser("cache", help="Pre-cache latents and text encoder outputs")
    sub_cache.add_argument("--config", type=str)
    sub_cache.add_argument("--verbose", "-v", action="store_true")
    sub_cache.add_argument("--dry-run", action="store_true")
    sub_cache.add_argument("--dataset", type=str)
    sub_cache.add_argument("--model", type=str, help="Model path or HuggingFace ID")
    sub_cache.add_argument("--type", choices=["latents", "text", "all"], default="all",
                           help="What to cache (default: all)")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.stage == "cache":
        cmd_cache(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
