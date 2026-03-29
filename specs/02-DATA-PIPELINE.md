# 02 - Data Pipeline: Video Ingestion and Clip Extraction

## Overview

The data pipeline transforms raw video files (e.g., TV show episodes) into a structured dataset of short clips suitable for training a video diffusion model. Each clip is paired with metadata and will later be captioned by the captioning pipeline.

## Pipeline Stages

```
[Raw Video Files (.mp4, .mkv, .avi)]
        |
        v
[Stage 1: Preprocessing] -- Normalize format, extract audio/subtitles
        |
        v
[Stage 2: Scene Detection] -- Find natural scene boundaries
        |
        v
[Stage 3: Clip Extraction] -- Cut into training-length clips
        |
        v
[Stage 4: Filtering] -- Remove unsuitable clips (credits, black frames, etc.)
        |
        v
[Stage 5: Clip Conditioning] -- Resize, normalize framerate, organize
        |
        v
[Structured Dataset Directory]
```

## Stage 1: Preprocessing

### Goal
Normalize all source videos to a consistent format for processing.

### Implementation

```bash
# Normalize to MP4/H.264 with consistent audio
ffmpeg -i input.mkv \
  -c:v libx264 -crf 18 -preset medium \
  -c:a aac -b:a 128k \
  -r 24 \
  output_normalized.mp4
```

### Extract Subtitles (if embedded)

```bash
# Extract subtitle track (for later use as rough captions)
ffmpeg -i input.mkv -map 0:s:0 subtitles.srt

# Or extract all subtitle tracks
ffmpeg -i input.mkv -map 0:s subtitles_%d.srt
```

Subtitles provide dialogue text that can augment auto-generated captions during the captioning stage.

### Extract Audio (for speech-to-text if no subtitles)

```bash
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
```

Use Whisper (open source) for speech-to-text if no subtitle track exists:
```bash
pip install openai-whisper
whisper audio.wav --model base --output_format srt
```

## Stage 2: Scene Detection

### Goal
Find natural scene boundaries (cuts, fades, dissolves) to split videos at semantically meaningful points.

### Tool: PySceneDetect

```bash
pip install scenedetect[opencv]
```

### Usage

```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector

video = open_video("episode_01.mp4")
scene_manager = SceneManager()

# ContentDetector: detects hard cuts based on frame difference
# threshold: lower = more sensitive (more scenes detected)
# Typical range: 20-35 for TV content
scene_manager.add_detector(ContentDetector(threshold=27))

scene_manager.detect_scenes(video)
scene_list = scene_manager.get_scene_list()

for i, (start, end) in enumerate(scene_list):
    print(f"Scene {i:04d}: {start.get_timecode()} -> {end.get_timecode()} "
          f"({end.get_seconds() - start.get_seconds():.1f}s)")
```

### Configuration

```yaml
# config/scene_detection.yaml
scene_detection:
  detector: content          # content | adaptive | threshold
  threshold: 27              # sensitivity (lower = more scenes)
  min_scene_length_sec: 1.0  # ignore scenes shorter than this
  max_scene_length_sec: 30.0 # force-split scenes longer than this
```

## Stage 3: Clip Extraction

### Goal
Extract individual clips from detected scenes, trimmed to training-appropriate lengths.

### Clip Length Strategy

Video diffusion models typically train on short clips:
- **Target length:** 2-6 seconds (49-145 frames at 24fps)
- **Minimum:** 1 second (24 frames)
- **Maximum:** 8 seconds (193 frames)

Scenes longer than the maximum are split into overlapping sub-clips.

### Implementation

```python
import subprocess

def extract_clip(source_path, start_sec, duration_sec, output_path, fps=24):
    """Extract a clip using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", source_path,
        "-t", str(duration_sec),
        "-c:v", "libx264", "-crf", "18",
        "-r", str(fps),
        "-an",  # no audio for training clips
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
```

### Splitting Long Scenes

```python
def split_scene_to_clips(start_sec, end_sec, target_duration=4.0, overlap=0.5):
    """Split a long scene into overlapping clips."""
    clips = []
    duration = end_sec - start_sec

    if duration <= target_duration * 1.5:
        # Short enough to use as-is
        clips.append((start_sec, duration))
    else:
        # Split with overlap
        current = start_sec
        while current + target_duration <= end_sec:
            clips.append((current, target_duration))
            current += target_duration - overlap
        # Handle remainder
        if end_sec - current > 1.0:
            clips.append((current, end_sec - current))

    return clips
```

## Stage 4: Filtering

### Goal
Remove clips that would hurt training quality.

### Filters to Apply

```yaml
# config/clip_filtering.yaml
filtering:
  # Remove clips that are mostly black/white (intros, transitions)
  black_frame_threshold: 0.85    # % of near-black pixels
  white_frame_threshold: 0.85

  # Remove clips with too little motion (static title cards)
  min_optical_flow: 0.5

  # Remove clips with excessive motion (fight scenes may be too chaotic)
  max_optical_flow: 50.0

  # Remove clips shorter than minimum after extraction
  min_duration_sec: 1.0

  # Remove clips with text overlays (credits, chyrons)
  # Use OCR detection, filter if >30% of frames have detected text
  text_overlay_threshold: 0.3

  # Aspect ratio filter (remove letterboxed content if needed)
  allowed_aspect_ratios: ["16:9", "4:3"]
```

### Black Frame Detection

```python
import cv2
import numpy as np

def is_mostly_black(frame, threshold=0.85, pixel_threshold=20):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < pixel_threshold)
    return (dark_pixels / gray.size) > threshold

def filter_black_clips(clip_path, sample_count=5):
    cap = cv2.VideoCapture(clip_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    black_count = 0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and is_mostly_black(frame):
            black_count += 1

    cap.release()
    return black_count / sample_count > 0.5  # Reject if >50% samples are black
```

## Stage 5: Clip Conditioning

### Goal
Resize and normalize all clips to consistent dimensions and framerate for training.

### Resolution Strategy

With 16GB VRAM, target lower resolutions during training:

```yaml
# config/clip_conditioning.yaml
conditioning:
  # Target resolution (width x height)
  # 480p is practical for 16GB VRAM training
  target_width: 848
  target_height: 480

  # Alternatively, use bucket resolutions (multiple aspect ratios)
  use_buckets: true
  bucket_resolutions:
    - [848, 480]   # 16:9 landscape
    - [640, 480]   # 4:3 landscape
    - [480, 848]   # 9:16 portrait (if applicable)
    - [480, 640]   # 3:4 portrait

  # Framerate
  target_fps: 24

  # Frame count (must match model expectations)
  # Wan 2.1 expects specific frame counts: 1 + 4k (e.g., 17, 33, 49, 81)
  target_frames: 49  # ~2 seconds at 24fps

  # Pixel format
  pix_fmt: yuv420p
```

### Resize Implementation

```bash
# Resize maintaining aspect ratio, padding if needed
ffmpeg -i clip_raw.mp4 \
  -vf "scale=848:480:force_original_aspect_ratio=decrease,pad=848:480:-1:-1:color=black" \
  -r 24 \
  -frames:v 49 \
  -c:v libx264 -crf 18 \
  clip_conditioned.mp4
```

## Output: Dataset Directory Structure

```
dataset/
├── metadata.json              # Global dataset info
├── clips/
│   ├── ep01_scene0001_clip001.mp4
│   ├── ep01_scene0001_clip002.mp4
│   ├── ep01_scene0012_clip001.mp4
│   └── ...
├── subtitles/
│   ├── ep01.srt
│   └── ...
└── clip_metadata/
    ├── ep01_scene0001_clip001.json
    └── ...
```

### Clip Metadata Format

```json
{
  "clip_id": "ep01_scene0001_clip001",
  "source_file": "episode_01.mp4",
  "source_episode": "S01E01",
  "scene_index": 1,
  "clip_index": 1,
  "start_time_sec": 45.2,
  "end_time_sec": 49.2,
  "duration_sec": 4.0,
  "resolution": [848, 480],
  "fps": 24,
  "frame_count": 49,
  "subtitle_text": "I told you we should have left earlier.",
  "optical_flow_mean": 3.2,
  "caption": null
}
```

The `caption` field is null at this stage and will be populated by the captioning pipeline (Stage 03).

## CLI Interface

```bash
# Full pipeline
python -m videoforge.data --config configs/data_pipeline.yaml --input /path/to/videos/

# Individual stages
python -m videoforge.data.preprocess --input /path/to/videos/ --output /path/to/normalized/
python -m videoforge.data.detect_scenes --input /path/to/normalized/
python -m videoforge.data.extract_clips --input /path/to/normalized/ --scenes scenes.json
python -m videoforge.data.filter_clips --input /path/to/clips/
python -m videoforge.data.condition_clips --input /path/to/clips/ --output /path/to/dataset/
```
