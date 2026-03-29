"""Stage 3: Clip extraction - cut scenes into training-length clips."""

import json
import logging
from datetime import datetime
from pathlib import Path

from videoforge.utils.video import extract_clip

logger = logging.getLogger(__name__)


def split_scene_to_clips(
    start_sec: float,
    end_sec: float,
    target_duration: float = 4.0,
    min_duration: float = 1.0,
    overlap: float = 0.5,
) -> list[tuple[float, float]]:
    """Split a scene into clips of target duration.

    Returns list of (start_sec, duration_sec) tuples.
    """
    duration = end_sec - start_sec

    if duration < min_duration:
        return []

    if duration <= target_duration * 1.5:
        return [(start_sec, duration)]

    clips = []
    current = start_sec
    while current + min_duration <= end_sec:
        clip_dur = min(target_duration, end_sec - current)
        if clip_dur >= min_duration:
            clips.append((current, clip_dur))
        current += target_duration - overlap

    return clips


def extract_clips_from_scenes(
    scenes_data: dict,
    output_dir: str | Path,
    metadata_dir: str | Path,
    target_duration: float = 4.0,
    min_duration: float = 1.0,
    max_duration: float = 8.0,
    overlap: float = 0.5,
    fps: int = 24,
) -> list[dict]:
    """Extract clips from detected scenes.

    Args:
        scenes_data: Dict from scene detection (keyed by video stem).
        output_dir: Directory for output clip files.
        metadata_dir: Directory for clip metadata JSON files.
        target_duration: Target clip length in seconds.
        min_duration: Minimum clip length.
        max_duration: Maximum clip length.
        overlap: Overlap in seconds when splitting long scenes.
        fps: Output framerate.

    Returns:
        List of clip metadata dicts.
    """
    output_dir = Path(output_dir)
    metadata_dir = Path(metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_clips = []
    total_extracted = 0
    total_skipped = 0

    for video_stem, video_data in scenes_data.items():
        source_path = video_data["source"]
        scenes = video_data["scenes"]

        logger.info("Extracting clips from %s (%d scenes)", video_stem, len(scenes))

        for scene in scenes:
            scene_idx = scene["scene_index"]
            clips = split_scene_to_clips(
                scene["start_sec"], scene["end_sec"],
                target_duration=target_duration,
                min_duration=min_duration,
                overlap=overlap,
            )

            for clip_idx, (clip_start, clip_duration) in enumerate(clips):
                if clip_duration > max_duration:
                    clip_duration = max_duration

                clip_id = f"{video_stem}_scene{scene_idx:04d}_clip{clip_idx:03d}"
                clip_path = output_dir / f"{clip_id}.mp4"
                meta_path = metadata_dir / f"{clip_id}.json"

                # Skip if already extracted
                if clip_path.exists() and meta_path.exists():
                    logger.debug("  Skipping existing: %s", clip_id)
                    # Load existing metadata
                    with open(meta_path) as f:
                        all_clips.append(json.load(f))
                    total_skipped += 1
                    continue

                try:
                    extract_clip(
                        source_path, clip_path,
                        start_sec=clip_start,
                        duration_sec=clip_duration,
                        fps=fps,
                    )
                except Exception as e:
                    logger.error("  Failed to extract %s: %s", clip_id, e)
                    continue

                # Build metadata
                meta = {
                    "schema_version": 1,
                    "created_at": datetime.utcnow().isoformat(),
                    "clip_id": clip_id,
                    "source_file": str(source_path),
                    "scene_index": scene_idx,
                    "clip_index": clip_idx,
                    "start_time_sec": round(clip_start, 3),
                    "end_time_sec": round(clip_start + clip_duration, 3),
                    "duration_sec": round(clip_duration, 3),
                    "fps": fps,
                    "caption": None,
                }

                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                all_clips.append(meta)
                total_extracted += 1

    logger.info(
        "Clip extraction complete: %d extracted, %d skipped (already existed)",
        total_extracted, total_skipped,
    )
    return all_clips
