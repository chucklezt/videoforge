"""Stage 2: Scene detection - find natural scene boundaries."""

import json
import logging
from pathlib import Path

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

logger = logging.getLogger(__name__)

DETECTORS = {
    "content": ContentDetector,
    "adaptive": AdaptiveDetector,
    "threshold": ThresholdDetector,
}


def detect_scenes(
    video_path: str | Path,
    detector: str = "content",
    threshold: float = 27.0,
    min_scene_length_sec: float = 1.0,
    max_scene_length_sec: float = 30.0,
) -> list[dict]:
    """Detect scene boundaries in a video.

    Returns list of scene dicts with start/end times.
    """
    video_path = Path(video_path)
    logger.info("Detecting scenes: %s (detector=%s, threshold=%.1f)", video_path.name, detector, threshold)

    video = open_video(str(video_path))
    scene_manager = SceneManager()

    detector_cls = DETECTORS.get(detector, ContentDetector)
    # min_scene_len is in frames; convert from seconds using video fps
    fps = video.frame_rate
    min_scene_frames = int(min_scene_length_sec * fps)

    if detector == "adaptive":
        scene_manager.add_detector(detector_cls(
            adaptive_threshold=threshold,
            min_scene_len=min_scene_frames,
        ))
    elif detector == "threshold":
        scene_manager.add_detector(detector_cls(
            threshold=threshold,
            min_scene_len=min_scene_frames,
        ))
    else:
        scene_manager.add_detector(detector_cls(
            threshold=threshold,
            min_scene_len=min_scene_frames,
        ))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # If no cuts detected, treat the whole video as one scene
    if not scene_list:
        duration = video.duration.get_seconds()
        if duration >= min_scene_length_sec:
            logger.info("  No cuts detected, treating entire video as one scene (%.1fs)", duration)
            scenes = [{
                "scene_index": 0,
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "duration_sec": round(duration, 3),
            }]
            # Still apply max_scene_length splitting
            if duration > max_scene_length_sec:
                scenes = []
                current = 0.0
                while current < duration:
                    sub_end = min(current + max_scene_length_sec, duration)
                    if sub_end - current >= min_scene_length_sec:
                        scenes.append({
                            "scene_index": len(scenes),
                            "start_sec": round(current, 3),
                            "end_sec": round(sub_end, 3),
                            "duration_sec": round(sub_end - current, 3),
                        })
                    current = sub_end
            logger.info("  Found %d scenes (whole video, no cuts)", len(scenes))
            return scenes

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec

        # Force-split scenes longer than max
        if duration > max_scene_length_sec:
            current = start_sec
            sub_idx = 0
            while current < end_sec:
                sub_end = min(current + max_scene_length_sec, end_sec)
                if sub_end - current >= min_scene_length_sec:
                    scenes.append({
                        "scene_index": len(scenes),
                        "start_sec": round(current, 3),
                        "end_sec": round(sub_end, 3),
                        "duration_sec": round(sub_end - current, 3),
                    })
                current = sub_end
                sub_idx += 1
        else:
            scenes.append({
                "scene_index": len(scenes),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(duration, 3),
            })

    logger.info("  Found %d scenes (from %d raw detections)", len(scenes), len(scene_list))
    return scenes


def detect_scenes_batch(
    video_dir: str | Path,
    output_path: str | Path,
    detector: str = "content",
    threshold: float = 27.0,
    min_scene_length_sec: float = 1.0,
    max_scene_length_sec: float = 30.0,
) -> dict:
    """Run scene detection on all videos in a directory.

    Saves results to a JSON file and returns the full result dict.
    """
    video_dir = Path(video_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from videoforge.data.preprocess import find_videos

    videos = find_videos(video_dir)
    if not videos:
        logger.warning("No video files found in %s", video_dir)
        return {}

    all_scenes = {}
    for video_path in videos:
        scenes = detect_scenes(
            video_path,
            detector=detector,
            threshold=threshold,
            min_scene_length_sec=min_scene_length_sec,
            max_scene_length_sec=max_scene_length_sec,
        )
        all_scenes[video_path.stem] = {
            "source": str(video_path),
            "scene_count": len(scenes),
            "scenes": scenes,
        }

    with open(output_path, "w") as f:
        json.dump(all_scenes, f, indent=2)

    total = sum(v["scene_count"] for v in all_scenes.values())
    logger.info("Scene detection complete: %d scenes across %d videos", total, len(all_scenes))
    return all_scenes
