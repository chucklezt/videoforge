"""Captioning pipeline CLI - run via: python -m videoforge.caption"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from videoforge.utils.config import load_config, get_nested
from videoforge.caption.captioner import DEFAULT_PROMPT

logger = logging.getLogger("videoforge.caption")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_caption(args):
    """Run captioning on uncaptioned clips."""
    config = load_config(args.config) if args.config else {}
    cap_cfg = config.get("captioning", {})

    dataset_dir = Path(args.dataset or get_nested(config, "dataset_dir", "./dataset"))
    metadata_dir = dataset_dir / "clip_metadata"
    clips_dir = dataset_dir / "clips_conditioned"
    subtitles_dir = dataset_dir / "subtitles"

    if not metadata_dir.exists():
        logger.error("Metadata directory not found: %s", metadata_dir)
        sys.exit(1)

    # Load style tags
    from videoforge.caption.enrichment import load_style_tags
    style_tags_path = args.style_tags or get_nested(config, "style_tags_path", "configs/style_tags.yaml")
    style_tags = load_style_tags(style_tags_path)
    if style_tags:
        logger.info("Style tags: %s", ", ".join(style_tags))

    # Load subtitles if available
    from videoforge.caption.enrichment import load_subtitles, find_dialogue_for_clip
    subtitles_by_source = {}
    if subtitles_dir.exists():
        for srt_file in subtitles_dir.glob("*.srt"):
            subtitles_by_source[srt_file.stem] = load_subtitles(srt_file)
            logger.info("Loaded subtitles: %s", srt_file.name)

    # Find clips to caption
    meta_files = sorted(metadata_dir.glob("*.json"))
    to_caption = []
    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("deleted") or not meta.get("filter_passed", True):
            continue
        if meta.get("caption") and not args.recaption:
            continue
        # Filter to specific clips if requested
        if args.clips and meta["clip_id"] not in args.clips:
            continue
        to_caption.append((meta_path, meta))

    if not to_caption:
        logger.info("No clips need captioning")
        return

    logger.info("Captioning %d clips", len(to_caption))

    if args.dry_run:
        for meta_path, meta in to_caption:
            logger.info("  Would caption: %s", meta["clip_id"])
        return

    # Load model
    from videoforge.caption.captioner import VideoCaptioner
    captioner = VideoCaptioner(
        model_name=cap_cfg.get("model", "Qwen/Qwen2-VL-7B-Instruct"),
        quantization=cap_cfg.get("quantization", "4bit"),
        max_new_tokens=cap_cfg.get("max_new_tokens", 300),
        prompt=cap_cfg.get("prompt", None) or DEFAULT_PROMPT,
        fps_sample=cap_cfg.get("fps_sample", 4.0),
    )
    captioner.load()

    from videoforge.caption.enrichment import enrich_caption

    captioned = 0
    for meta_path, meta in to_caption:
        clip_id = meta["clip_id"]
        clip_path = clips_dir / f"{clip_id}.mp4"

        if not clip_path.exists():
            logger.warning("Clip file missing: %s", clip_path)
            continue

        logger.info("Captioning: %s", clip_id)

        try:
            visual_caption = captioner.caption_clip(clip_path)
        except Exception as e:
            logger.error("  Failed: %s", e)
            continue

        torch.cuda.empty_cache()

        # Find dialogue from subtitles
        source_stem = clip_id.rsplit("_scene", 1)[0]
        dialogue = None
        if source_stem in subtitles_by_source:
            dialogue = find_dialogue_for_clip(
                subtitles_by_source[source_stem],
                meta.get("start_time_sec", 0),
                meta.get("end_time_sec", 0),
            )

        # Enrich caption
        caption = enrich_caption(visual_caption, subtitle_text=dialogue, style_tags=style_tags)

        # Update metadata
        meta["caption"] = caption
        meta["caption_visual"] = visual_caption
        meta["caption_source"] = captioner.model_name
        if dialogue:
            meta["subtitle_text"] = dialogue
        meta["caption_reviewed"] = False

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        captioned += 1
        logger.info("  Done: %s", caption[:100] + "..." if len(caption) > 100 else caption)

    captioner.unload()
    logger.info("Captioning complete: %d clips captioned", captioned)


def cmd_review(args):
    """Interactive caption review."""
    from videoforge.caption.review import review_captions
    config = load_config(args.config) if args.config else {}
    dataset_dir = Path(args.dataset or get_nested(config, "dataset_dir", "./dataset"))
    review_captions(
        dataset_dir / "clip_metadata",
        clips_dir=dataset_dir / "clips_conditioned",
    )


def cmd_export(args):
    """Export captions to .txt files."""
    from videoforge.caption.export import export_captions_txt
    config = load_config(args.config) if args.config else {}
    dataset_dir = Path(args.dataset or get_nested(config, "dataset_dir", "./dataset"))
    output_dir = args.output or str(dataset_dir / "clips_conditioned")
    export_captions_txt(dataset_dir / "clip_metadata", output_dir)


def main():
    parser = argparse.ArgumentParser(
        prog="videoforge.caption",
        description="VideoForge Captioning Pipeline - Auto-caption training clips",
    )
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--dataset", type=str, help="Dataset directory")

    subparsers = parser.add_subparsers(dest="stage")

    # Default: caption
    parser.add_argument("--clips", nargs="*", help="Specific clip IDs to caption")
    parser.add_argument("--recaption", action="store_true", help="Re-caption already captioned clips")
    parser.add_argument("--style-tags", type=str, help="Path to style_tags.yaml")

    # Review
    sub_review = subparsers.add_parser("review", help="Interactive caption review")
    sub_review.add_argument("--config", type=str)
    sub_review.add_argument("--verbose", "-v", action="store_true")
    sub_review.add_argument("--dataset", type=str)

    # Export
    sub_export = subparsers.add_parser("export", help="Export captions to .txt files")
    sub_export.add_argument("--config", type=str)
    sub_export.add_argument("--verbose", "-v", action="store_true")
    sub_export.add_argument("--dataset", type=str)
    sub_export.add_argument("--output", "-o", type=str, help="Output directory for .txt files")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.stage == "review":
        cmd_review(args)
    elif args.stage == "export":
        cmd_export(args)
    else:
        cmd_caption(args)


if __name__ == "__main__":
    main()
