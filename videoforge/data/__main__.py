"""Data pipeline CLI - run via: python -m videoforge.data"""

import argparse
import json
import logging
import sys
from pathlib import Path

from videoforge.utils.config import load_config, get_nested

logger = logging.getLogger("videoforge.data")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")


def cmd_full_pipeline(args):
    """Run the full data pipeline."""
    if not args.config and not args.input:
        logger.error("Provide --config or --input")
        sys.exit(1)

    config = load_config(args.config) if args.config else {}

    source_dir = args.input or get_nested(config, "data.source_dir")
    dataset_dir = args.output or get_nested(config, "data.dataset_dir", "./dataset")

    if not source_dir:
        logger.error("No source directory specified. Use --input or set data.source_dir in config.")
        sys.exit(1)

    dataset_dir = Path(dataset_dir)
    normalized_dir = dataset_dir / "normalized"
    clips_dir = dataset_dir / "clips"
    conditioned_dir = dataset_dir / "clips_conditioned"
    metadata_dir = dataset_dir / "clip_metadata"
    subtitles_dir = dataset_dir / "subtitles"
    scenes_path = dataset_dir / "scenes.json"

    # Stage 1: Preprocessing
    logger.info("=" * 50)
    logger.info("Stage 1: Preprocessing")
    logger.info("=" * 50)
    from videoforge.data.preprocess import run_preprocessing

    fps = get_nested(config, "preprocessing.target_fps", 24)
    crf = get_nested(config, "preprocessing.crf", 18)

    run_preprocessing(
        source_dir, normalized_dir,
        subtitles_dir=subtitles_dir,
        fps=fps, crf=crf,
    )

    # Stage 2: Scene Detection
    logger.info("=" * 50)
    logger.info("Stage 2: Scene Detection")
    logger.info("=" * 50)
    from videoforge.data.scene_detect import detect_scenes_batch

    detector = get_nested(config, "scene_detection.detector", "content")
    threshold = get_nested(config, "scene_detection.threshold", 27.0)
    min_scene = get_nested(config, "scene_detection.min_scene_length_sec", 1.0)
    max_scene = get_nested(config, "scene_detection.max_scene_length_sec", 30.0)

    scenes_data = detect_scenes_batch(
        normalized_dir, scenes_path,
        detector=detector, threshold=threshold,
        min_scene_length_sec=min_scene, max_scene_length_sec=max_scene,
    )

    # Stage 3: Clip Extraction
    logger.info("=" * 50)
    logger.info("Stage 3: Clip Extraction")
    logger.info("=" * 50)
    from videoforge.data.clip_extract import extract_clips_from_scenes

    target_dur = get_nested(config, "clip_extraction.target_duration_sec", 4.0)
    min_dur = get_nested(config, "clip_extraction.min_duration_sec", 1.0)
    max_dur = get_nested(config, "clip_extraction.max_duration_sec", 8.0)
    overlap = get_nested(config, "clip_extraction.overlap_sec", 0.5)

    extract_clips_from_scenes(
        scenes_data, clips_dir, metadata_dir,
        target_duration=target_dur, min_duration=min_dur,
        max_duration=max_dur, overlap=overlap, fps=fps,
    )

    # Stage 4: Filtering
    logger.info("=" * 50)
    logger.info("Stage 4: Clip Filtering")
    logger.info("=" * 50)
    from videoforge.data.clip_filter import filter_clips_batch

    filter_cfg = config.get("filtering", {})
    filter_clips_batch(
        clips_dir, metadata_dir,
        black_threshold=filter_cfg.get("black_frame_threshold", 0.85),
        white_threshold=filter_cfg.get("white_frame_threshold", 0.85),
        min_optical_flow=filter_cfg.get("min_optical_flow", 0.5),
        max_optical_flow=filter_cfg.get("max_optical_flow", 50.0),
        min_duration_sec=filter_cfg.get("min_duration_sec", 1.0),
        dry_run=args.dry_run,
    )

    # Stage 5: Conditioning
    logger.info("=" * 50)
    logger.info("Stage 5: Clip Conditioning")
    logger.info("=" * 50)
    from videoforge.data.clip_condition import condition_clips_batch

    cond_cfg = config.get("conditioning", {})
    target_width = cond_cfg.get("target_width", 848)
    target_height = cond_cfg.get("target_height", 480)
    target_fps = cond_cfg.get("target_fps", 24)
    target_frames = cond_cfg.get("target_frames", 49)
    use_buckets = cond_cfg.get("use_buckets", False)
    buckets = [tuple(b) for b in cond_cfg.get("bucket_resolutions", [])]

    if not args.dry_run:
        condition_clips_batch(
            clips_dir, metadata_dir, conditioned_dir,
            target_width=target_width, target_height=target_height,
            target_fps=target_fps, target_frames=target_frames,
            use_buckets=use_buckets, buckets=buckets if use_buckets else None,
        )

    # Write dataset metadata
    meta_files = sorted((metadata_dir).glob("*.json"))
    total_clips = 0
    passed_clips = 0
    for mf in meta_files:
        with open(mf) as f:
            m = json.load(f)
        total_clips += 1
        if m.get("filter_passed", True):
            passed_clips += 1

    dataset_meta = {
        "source_dir": str(source_dir),
        "total_clips_extracted": total_clips,
        "clips_passed_filter": passed_clips,
        "conditioning": {
            "resolution": [target_width, target_height],
            "fps": target_fps,
            "target_frames": target_frames,
        },
    }
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    logger.info("=" * 50)
    logger.info("Data pipeline complete!")
    logger.info("  Total clips extracted: %d", total_clips)
    logger.info("  Clips passed filter: %d", passed_clips)
    logger.info("  Dataset directory: %s", dataset_dir)
    logger.info("=" * 50)


def cmd_preprocess(args):
    from videoforge.data.preprocess import run_preprocessing
    config = load_config(args.config) if args.config else {}
    source_dir = args.input or get_nested(config, "data.source_dir")
    output_dir = args.output or get_nested(config, "data.dataset_dir", "./dataset") + "/normalized"
    if not source_dir:
        logger.error("Provide --input")
        sys.exit(1)
    run_preprocessing(
        source_dir, output_dir,
        subtitles_dir=args.subtitles_dir,
        fps=get_nested(config, "preprocessing.target_fps", 24),
    )


def cmd_detect_scenes(args):
    from videoforge.data.scene_detect import detect_scenes_batch
    config = load_config(args.config) if args.config else {}
    input_dir = args.input
    output = args.output or "scenes.json"
    if not input_dir:
        logger.error("Provide --input")
        sys.exit(1)
    detect_scenes_batch(
        input_dir, output,
        detector=get_nested(config, "scene_detection.detector", "content"),
        threshold=args.threshold or get_nested(config, "scene_detection.threshold", 27.0),
        min_scene_length_sec=get_nested(config, "scene_detection.min_scene_length_sec", 1.0),
        max_scene_length_sec=get_nested(config, "scene_detection.max_scene_length_sec", 30.0),
    )


def cmd_extract_clips(args):
    from videoforge.data.clip_extract import extract_clips_from_scenes
    config = load_config(args.config) if args.config else {}
    if not args.scenes:
        logger.error("Provide --scenes (scenes.json from scene detection)")
        sys.exit(1)
    with open(args.scenes) as f:
        scenes_data = json.load(f)
    output_dir = args.output or get_nested(config, "data.dataset_dir", "./dataset") + "/clips"
    metadata_dir = args.metadata_dir or str(Path(output_dir).parent / "clip_metadata")
    extract_clips_from_scenes(
        scenes_data, output_dir, metadata_dir,
        target_duration=get_nested(config, "clip_extraction.target_duration_sec", 4.0),
        min_duration=get_nested(config, "clip_extraction.min_duration_sec", 1.0),
        max_duration=get_nested(config, "clip_extraction.max_duration_sec", 8.0),
        overlap=get_nested(config, "clip_extraction.overlap_sec", 0.5),
    )


def cmd_filter_clips(args):
    from videoforge.data.clip_filter import filter_clips_batch
    config = load_config(args.config) if args.config else {}
    clips_dir = args.input
    metadata_dir = args.metadata_dir or str(Path(clips_dir).parent / "clip_metadata")
    if not clips_dir:
        logger.error("Provide --input (clips directory)")
        sys.exit(1)
    filter_cfg = config.get("filtering", {})
    filter_clips_batch(
        clips_dir, metadata_dir,
        black_threshold=filter_cfg.get("black_frame_threshold", 0.85),
        min_optical_flow=filter_cfg.get("min_optical_flow", 0.5),
        max_optical_flow=filter_cfg.get("max_optical_flow", 50.0),
        dry_run=args.dry_run,
    )


def cmd_condition_clips(args):
    from videoforge.data.clip_condition import condition_clips_batch
    config = load_config(args.config) if args.config else {}
    clips_dir = args.input
    output_dir = args.output
    metadata_dir = args.metadata_dir or str(Path(clips_dir).parent / "clip_metadata")
    if not clips_dir or not output_dir:
        logger.error("Provide --input and --output")
        sys.exit(1)
    cond_cfg = config.get("conditioning", {})
    condition_clips_batch(
        clips_dir, metadata_dir, output_dir,
        target_width=cond_cfg.get("target_width", 848),
        target_height=cond_cfg.get("target_height", 480),
        target_fps=cond_cfg.get("target_fps", 24),
        target_frames=cond_cfg.get("target_frames", 49),
    )


def main():
    parser = argparse.ArgumentParser(
        prog="videoforge.data",
        description="VideoForge Data Pipeline - Video ingestion and clip extraction",
    )
    add_common_args(parser)

    subparsers = parser.add_subparsers(dest="stage")

    # Full pipeline (default when no subcommand)
    parser.add_argument("--input", "-i", type=str, help="Source video directory")
    parser.add_argument("--output", "-o", type=str, help="Output dataset directory")

    # Preprocess
    sub_pre = subparsers.add_parser("preprocess", help="Normalize video format")
    add_common_args(sub_pre)
    sub_pre.add_argument("--input", "-i", type=str, required=True)
    sub_pre.add_argument("--output", "-o", type=str)
    sub_pre.add_argument("--subtitles-dir", type=str)

    # Scene detection
    sub_scene = subparsers.add_parser("scenes", help="Detect scene boundaries")
    add_common_args(sub_scene)
    sub_scene.add_argument("--input", "-i", type=str, required=True)
    sub_scene.add_argument("--output", "-o", type=str, default="scenes.json")
    sub_scene.add_argument("--threshold", type=float)

    # Clip extraction
    sub_extract = subparsers.add_parser("extract", help="Extract clips from scenes")
    add_common_args(sub_extract)
    sub_extract.add_argument("--scenes", type=str, required=True, help="scenes.json file")
    sub_extract.add_argument("--output", "-o", type=str)
    sub_extract.add_argument("--metadata-dir", type=str)

    # Filtering
    sub_filter = subparsers.add_parser("filter", help="Filter clips by quality")
    add_common_args(sub_filter)
    sub_filter.add_argument("--input", "-i", type=str, required=True)
    sub_filter.add_argument("--metadata-dir", type=str)

    # Conditioning
    sub_cond = subparsers.add_parser("condition", help="Resize and normalize clips")
    add_common_args(sub_cond)
    sub_cond.add_argument("--input", "-i", type=str, required=True)
    sub_cond.add_argument("--output", "-o", type=str, required=True)
    sub_cond.add_argument("--metadata-dir", type=str)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.stage is None:
        cmd_full_pipeline(args)
    elif args.stage == "preprocess":
        cmd_preprocess(args)
    elif args.stage == "scenes":
        cmd_detect_scenes(args)
    elif args.stage == "extract":
        cmd_extract_clips(args)
    elif args.stage == "filter":
        cmd_filter_clips(args)
    elif args.stage == "condition":
        cmd_condition_clips(args)


if __name__ == "__main__":
    main()
