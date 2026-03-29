"""Stage 1: Video preprocessing - normalize format, extract audio/subtitles."""

import logging
from pathlib import Path

from videoforge.utils.video import (
    extract_audio,
    extract_subtitles,
    get_video_info,
    normalize_video,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts"}


def find_videos(source_dir: str | Path) -> list[Path]:
    """Find all supported video files in a directory."""
    source_dir = Path(source_dir)
    videos = []
    for ext in SUPPORTED_EXTENSIONS:
        videos.extend(source_dir.glob(f"*{ext}"))
        videos.extend(source_dir.glob(f"*{ext.upper()}"))
    return sorted(set(videos))


def preprocess_video(
    input_path: Path,
    output_dir: Path,
    subtitles_dir: Path | None = None,
    fps: int = 24,
    crf: int = 18,
    extract_subs: bool = True,
) -> dict:
    """Preprocess a single video: normalize format, extract subtitles.

    Returns metadata dict about the preprocessed video.
    """
    stem = input_path.stem
    output_path = output_dir / f"{stem}.mp4"
    result = {
        "source": str(input_path),
        "normalized": str(output_path),
        "subtitles": None,
        "audio": None,
    }

    # Get source info
    try:
        info = get_video_info(input_path)
    except Exception as e:
        logger.error("Skipping %s: cannot read video (%s)", input_path.name, e)
        return None

    logger.info(
        "Processing: %s (%.1fs, %dx%d, %.1ffps)",
        input_path.name, info["duration"],
        info.get("width", 0), info.get("height", 0), info.get("fps", 0),
    )

    # Normalize video
    if output_path.exists():
        logger.info("  Normalized file exists, skipping: %s", output_path.name)
    else:
        try:
            normalize_video(input_path, output_path, fps=fps, crf=crf)
            logger.info("  Normalized: %s", output_path.name)
        except Exception as e:
            logger.error("  Failed to normalize %s: %s", input_path.name, e)
            return None

    # Extract subtitles
    if extract_subs and subtitles_dir and info.get("subtitle_count", 0) > 0:
        sub_path = subtitles_dir / f"{stem}.srt"
        if sub_path.exists():
            logger.info("  Subtitles exist, skipping: %s", sub_path.name)
        else:
            extracted = extract_subtitles(input_path, sub_path)
            if extracted:
                result["subtitles"] = str(extracted)
                logger.info("  Extracted subtitles: %s", sub_path.name)
            else:
                logger.info("  No extractable subtitles")
    elif extract_subs and subtitles_dir:
        logger.info("  No subtitle tracks found")

    return result


def run_preprocessing(
    source_dir: str | Path,
    output_dir: str | Path,
    subtitles_dir: str | Path | None = None,
    fps: int = 24,
    crf: int = 18,
) -> list[dict]:
    """Run preprocessing on all videos in source directory."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subtitles_dir:
        subtitles_dir = Path(subtitles_dir)
        subtitles_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(source_dir)
    if not videos:
        logger.warning("No video files found in %s", source_dir)
        return []

    logger.info("Found %d video files to preprocess", len(videos))
    results = []
    for video_path in videos:
        result = preprocess_video(
            video_path, output_dir,
            subtitles_dir=subtitles_dir,
            fps=fps, crf=crf,
        )
        if result is not None:
            results.append(result)

    logger.info("Preprocessing complete: %d videos processed", len(results))
    return results
