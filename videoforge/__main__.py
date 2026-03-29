"""VideoForge CLI entry point - run via: python -m videoforge"""

import argparse
import logging
import sys


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_data(args, remaining):
    """Dispatch to data pipeline."""
    from videoforge.data.__main__ import main as data_main
    sys.argv = ["videoforge.data"] + remaining
    data_main()


def cmd_caption(args, remaining):
    """Dispatch to captioning pipeline."""
    from videoforge.caption.__main__ import main as caption_main
    sys.argv = ["videoforge.caption"] + remaining
    caption_main()


def cmd_train(args, remaining):
    """Dispatch to training pipeline."""
    from videoforge.train.__main__ import main as train_main
    sys.argv = ["videoforge.train"] + remaining
    train_main()


def cmd_generate(args, remaining):
    """Dispatch to inference pipeline."""
    print("Inference pipeline not yet implemented.")
    sys.exit(1)


def cmd_postprocess(args, remaining):
    """Dispatch to post-processing pipeline."""
    print("Post-processing pipeline not yet implemented.")
    sys.exit(1)


def cmd_validate(args, remaining):
    """Run environment validation."""
    from videoforge.utils.rocm import check_rocm_env, print_validation_report
    results = check_rocm_env()
    success = print_validation_report(results)
    sys.exit(0 if success else 1)


def main():
    parser = argparse.ArgumentParser(
        prog="videoforge",
        description="VideoForge - Video generation training pipeline for AMD ROCm",
    )
    parser.add_argument("--version", action="version", version="videoforge 0.1.0")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("data", help="Video ingestion and clip extraction", add_help=False)
    subparsers.add_parser("caption", help="Auto-caption training clips", add_help=False)
    subparsers.add_parser("train", help="LoRA fine-tuning", add_help=False)
    subparsers.add_parser("generate", help="Generate video from script", add_help=False)
    subparsers.add_parser("postprocess", help="Upscale, interpolate, stitch", add_help=False)
    subparsers.add_parser("validate", help="Validate environment setup", add_help=False)

    # Parse only the known args so subcommands can have their own args
    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "data": cmd_data,
        "caption": cmd_caption,
        "train": cmd_train,
        "generate": cmd_generate,
        "postprocess": cmd_postprocess,
        "validate": cmd_validate,
    }
    handlers[args.command](args, remaining)


if __name__ == "__main__":
    main()
