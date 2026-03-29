"""Stage 4: Clip filtering - remove unsuitable clips."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def is_mostly_black(frame: np.ndarray, threshold: float = 0.85, pixel_threshold: int = 20) -> bool:
    """Check if a frame is mostly black."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < pixel_threshold)
    return (dark_pixels / gray.size) > threshold


def is_mostly_white(frame: np.ndarray, threshold: float = 0.85, pixel_threshold: int = 235) -> bool:
    """Check if a frame is mostly white."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray > pixel_threshold)
    return (bright_pixels / gray.size) > threshold


def compute_optical_flow_magnitude(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute mean optical flow magnitude between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


def sample_frames(clip_path: str | Path, count: int = 5) -> list[np.ndarray]:
    """Sample evenly-spaced frames from a clip."""
    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, count, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def filter_clip(
    clip_path: str | Path,
    black_threshold: float = 0.85,
    white_threshold: float = 0.85,
    min_optical_flow: float = 0.5,
    max_optical_flow: float = 50.0,
    min_duration_sec: float = 1.0,
) -> tuple[bool, str]:
    """Check if a clip passes quality filters.

    Returns (passes, reason) where reason explains rejection.
    """
    clip_path = Path(clip_path)

    # Check file exists
    if not clip_path.exists():
        return False, "file_not_found"

    # Duration check
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        return False, "invalid_fps"

    duration = frame_count / fps
    if duration < min_duration_sec:
        return False, f"too_short ({duration:.1f}s)"

    # Sample frames for visual checks
    frames = sample_frames(clip_path, count=5)
    if len(frames) < 2:
        return False, "too_few_frames"

    # Black/white frame check
    black_count = sum(1 for f in frames if is_mostly_black(f, black_threshold))
    if black_count / len(frames) > 0.5:
        return False, "mostly_black"

    white_count = sum(1 for f in frames if is_mostly_white(f, white_threshold))
    if white_count / len(frames) > 0.5:
        return False, "mostly_white"

    # Optical flow check (motion)
    flow_values = []
    for i in range(len(frames) - 1):
        flow_values.append(compute_optical_flow_magnitude(frames[i], frames[i + 1]))

    if flow_values:
        mean_flow = np.mean(flow_values)
        if mean_flow < min_optical_flow:
            return False, f"too_static (flow={mean_flow:.2f})"
        if mean_flow > max_optical_flow:
            return False, f"too_chaotic (flow={mean_flow:.2f})"

    return True, "passed"


def filter_clips_batch(
    clips_dir: str | Path,
    metadata_dir: str | Path,
    black_threshold: float = 0.85,
    white_threshold: float = 0.85,
    min_optical_flow: float = 0.5,
    max_optical_flow: float = 50.0,
    min_duration_sec: float = 1.0,
    dry_run: bool = False,
) -> dict:
    """Filter all clips in a directory.

    Updates metadata JSON files with filter results.
    Returns summary dict with counts.
    """
    clips_dir = Path(clips_dir)
    metadata_dir = Path(metadata_dir)

    meta_files = sorted(metadata_dir.glob("*.json"))
    if not meta_files:
        logger.warning("No metadata files found in %s", metadata_dir)
        return {"total": 0, "passed": 0, "rejected": 0}

    logger.info("Filtering %d clips", len(meta_files))

    passed_count = 0
    rejected_count = 0
    rejection_reasons = {}

    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        clip_id = meta["clip_id"]
        clip_path = clips_dir / f"{clip_id}.mp4"

        passes, reason = filter_clip(
            clip_path,
            black_threshold=black_threshold,
            white_threshold=white_threshold,
            min_optical_flow=min_optical_flow,
            max_optical_flow=max_optical_flow,
            min_duration_sec=min_duration_sec,
        )

        meta["filter_passed"] = passes
        meta["filter_reason"] = reason

        if passes:
            passed_count += 1
            logger.debug("  PASS: %s", clip_id)
        else:
            rejected_count += 1
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            logger.debug("  REJECT: %s (%s)", clip_id, reason)

        if not dry_run:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    summary = {
        "total": len(meta_files),
        "passed": passed_count,
        "rejected": rejected_count,
        "rejection_reasons": rejection_reasons,
    }

    logger.info(
        "Filtering complete: %d passed, %d rejected out of %d",
        passed_count, rejected_count, len(meta_files),
    )
    if rejection_reasons:
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            logger.info("  Rejected: %s (%d clips)", reason, count)

    return summary
