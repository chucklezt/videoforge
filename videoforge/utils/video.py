"""FFmpeg wrapper utilities for video processing."""

import json
import subprocess
from pathlib import Path


def get_video_info(video_path: str | Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    subtitle_streams = []

    for stream in probe.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio" and audio_stream is None:
            audio_stream = stream
        elif stream["codec_type"] == "subtitle":
            subtitle_streams.append(stream)

    info = {
        "path": str(video_path),
        "duration": float(probe.get("format", {}).get("duration", 0)),
        "size_bytes": int(probe.get("format", {}).get("size", 0)),
    }

    if video_stream:
        info["width"] = int(video_stream.get("width", 0))
        info["height"] = int(video_stream.get("height", 0))
        # Parse fps from r_frame_rate (e.g., "24000/1001")
        fps_str = video_stream.get("r_frame_rate", "0/1")
        num, den = fps_str.split("/")
        info["fps"] = float(num) / float(den) if float(den) != 0 else 0.0
        info["codec"] = video_stream.get("codec_name", "unknown")
        info["frame_count"] = int(video_stream.get("nb_frames", 0))

    info["has_audio"] = audio_stream is not None
    info["subtitle_count"] = len(subtitle_streams)

    return info


def run_ffmpeg(args: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    """Run an FFmpeg command."""
    cmd = ["ffmpeg", "-y"] + args
    if quiet:
        cmd.insert(2, "-loglevel")
        cmd.insert(3, "error")
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def extract_clip(
    source_path: str | Path,
    output_path: str | Path,
    start_sec: float,
    duration_sec: float,
    fps: int = 24,
    no_audio: bool = True,
) -> Path:
    """Extract a clip from a video file using FFmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-ss", str(start_sec),
        "-i", str(source_path),
        "-t", str(duration_sec),
        "-c:v", "libx264", "-crf", "18",
        "-r", str(fps),
    ]
    if no_audio:
        args.append("-an")
    args.append(str(output_path))

    run_ffmpeg(args)
    return output_path


def normalize_video(
    input_path: str | Path,
    output_path: str | Path,
    fps: int = 24,
    crf: int = 18,
) -> Path:
    """Normalize a video to MP4/H.264 with consistent settings."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-i", str(input_path),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-c:a", "aac", "-b:a", "128k",
        "-r", str(fps),
        str(output_path),
    ]
    run_ffmpeg(args)
    return output_path


def extract_subtitles(input_path: str | Path, output_path: str | Path, track: int = 0) -> Path | None:
    """Extract a subtitle track from a video file. Returns None if no subtitles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        args = [
            "-i", str(input_path),
            "-map", f"0:s:{track}",
            str(output_path),
        ]
        run_ffmpeg(args)
        return output_path
    except subprocess.CalledProcessError:
        return None


def extract_audio(input_path: str | Path, output_path: str | Path) -> Path | None:
    """Extract audio as 16kHz mono WAV for speech-to-text."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        args = [
            "-i", str(input_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(output_path),
        ]
        run_ffmpeg(args)
        return output_path
    except subprocess.CalledProcessError:
        return None


def resize_video(
    input_path: str | Path,
    output_path: str | Path,
    width: int,
    height: int,
    fps: int = 24,
    max_frames: int | None = None,
) -> Path:
    """Resize a video to target dimensions with padding to maintain aspect ratio."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:-1:-1:color=black"

    args = [
        "-i", str(input_path),
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
    ]
    if max_frames is not None:
        args.extend(["-frames:v", str(max_frames)])
    args.append(str(output_path))

    run_ffmpeg(args)
    return output_path
