"""Stage 5: Clip conditioning - resize and normalize clips for training."""

import json
import logging
from pathlib import Path

from videoforge.utils.video import get_video_info, resize_video

logger = logging.getLogger(__name__)


def select_bucket(
    width: int,
    height: int,
    buckets: list[tuple[int, int]],
) -> tuple[int, int]:
    """Select the best resolution bucket for a given aspect ratio.

    Picks the bucket whose aspect ratio is closest to the source.
    """
    if not buckets:
        return (848, 480)

    source_ratio = width / height if height > 0 else 1.0
    best_bucket = buckets[0]
    best_diff = abs(source_ratio - best_bucket[0] / best_bucket[1])

    for bucket in buckets[1:]:
        diff = abs(source_ratio - bucket[0] / bucket[1])
        if diff < best_diff:
            best_diff = diff
            best_bucket = bucket

    return best_bucket


def condition_clip(
    clip_path: str | Path,
    output_path: str | Path,
    target_width: int = 848,
    target_height: int = 480,
    target_fps: int = 24,
    target_frames: int | None = 49,
    use_buckets: bool = False,
    buckets: list[tuple[int, int]] | None = None,
) -> dict:
    """Condition a single clip: resize, normalize fps, trim frames.

    Returns metadata about the conditioned clip.
    """
    clip_path = Path(clip_path)
    output_path = Path(output_path)

    info = get_video_info(clip_path)

    # Select resolution
    if use_buckets and buckets:
        w, h = select_bucket(info.get("width", 848), info.get("height", 480), buckets)
    else:
        w, h = target_width, target_height

    resize_video(
        clip_path, output_path,
        width=w, height=h,
        fps=target_fps,
        max_frames=target_frames,
    )

    return {
        "resolution": [w, h],
        "fps": target_fps,
        "frame_count": target_frames,
    }


def condition_clips_batch(
    clips_dir: str | Path,
    metadata_dir: str | Path,
    output_dir: str | Path,
    target_width: int = 848,
    target_height: int = 480,
    target_fps: int = 24,
    target_frames: int | None = 49,
    use_buckets: bool = False,
    buckets: list[tuple[int, int]] | None = None,
) -> int:
    """Condition all clips that passed filtering.

    Reads metadata to find passed clips, conditions them, updates metadata.
    Returns count of conditioned clips.
    """
    clips_dir = Path(clips_dir)
    metadata_dir = Path(metadata_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(metadata_dir.glob("*.json"))
    conditioned = 0

    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        # Skip clips that didn't pass filtering
        if not meta.get("filter_passed", True):
            continue

        clip_id = meta["clip_id"]
        clip_path = clips_dir / f"{clip_id}.mp4"
        out_path = output_dir / f"{clip_id}.mp4"

        if out_path.exists():
            logger.debug("  Skipping existing: %s", clip_id)
            conditioned += 1
            continue

        if not clip_path.exists():
            logger.warning("  Clip file missing: %s", clip_path)
            continue

        try:
            result = condition_clip(
                clip_path, out_path,
                target_width=target_width,
                target_height=target_height,
                target_fps=target_fps,
                target_frames=target_frames,
                use_buckets=use_buckets,
                buckets=buckets,
            )

            meta["resolution"] = result["resolution"]
            meta["fps"] = result["fps"]
            meta["frame_count"] = result["frame_count"]

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            conditioned += 1
            logger.debug("  Conditioned: %s -> %s", clip_id, result["resolution"])

        except Exception as e:
            logger.error("  Failed to condition %s: %s", clip_id, e)

    logger.info("Conditioning complete: %d clips conditioned", conditioned)
    return conditioned
