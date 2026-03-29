# 03 - Captioning Pipeline: Automated Video Description

## Overview

Every training clip needs a text caption that describes what is happening visually. Video diffusion models learn to associate text descriptions with visual content, so caption quality directly impacts generation quality.

This pipeline auto-generates detailed captions for each clip using open-source vision-language models (VLMs).

## Pipeline Flow

```
[Training Clips + Subtitles/Dialogue]
        |
        v
[Stage 1: Frame Sampling] -- Extract key frames from each clip
        |
        v
[Stage 2: Visual Captioning] -- VLM describes the visual content
        |
        v
[Stage 3: Caption Enrichment] -- Merge dialogue, add style tags
        |
        v
[Stage 4: Caption Review] -- Optional manual review/correction
        |
        v
[Captioned Dataset]
```

## Stage 1: Frame Sampling

For each clip, extract representative frames for the VLM to analyze.

```python
import cv2
import numpy as np

def sample_frames(clip_path, num_frames=8):
    """Sample evenly-spaced frames from a clip."""
    cap = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
```

## Stage 2: Visual Captioning with VLM

### Model Choice: Qwen2-VL-7B-Instruct (recommended)

- Open source (Apache 2.0)
- Supports video input natively (not just images)
- 7B parameters -- fits in 16GB VRAM with 4-bit quantization
- Strong scene understanding and description capabilities

### Alternative: InternVL2 or LLaVA-Video

If Qwen2-VL has ROCm issues, these are fallbacks:
- **InternVL2-8B** -- Strong video understanding, Apache 2.0
- **LLaVA-Video-7B** -- Specifically designed for video captioning

### Captioning Implementation

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def load_captioning_model():
    """Load Qwen2-VL with 4-bit quantization to fit in 16GB."""
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return model, processor


def caption_clip(model, processor, clip_path):
    """Generate a detailed caption for a video clip."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": clip_path,
                    "max_pixels": 360 * 420,
                    "fps": 4.0,  # Sample 4 frames per second for captioning
                },
                {
                    "type": "text",
                    "text": CAPTION_PROMPT,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=300)

    output_text = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]
    return output_text
```

### Caption Prompt

The prompt instructs the VLM on what to describe:

```python
CAPTION_PROMPT = """Describe this video clip in detail for training a video generation model. Include:

1. SCENE: Setting, location, time of day, lighting conditions
2. SUBJECTS: People present, their appearance, clothing, positioning
3. ACTION: What is happening, movements, gestures, expressions
4. CAMERA: Camera angle, movement (static, pan, zoom, tracking)
5. STYLE: Color palette, mood, visual style (cinematic, bright, dark, etc.)

Write a single flowing paragraph, not a bulleted list. Be specific and visual.
Do not describe audio or make assumptions about what cannot be seen."""
```

### Example Output

> A medium shot in a dimly lit living room with warm amber lighting from table lamps. A woman with dark hair wearing a blue cardigan sits on a brown leather couch, leaning forward with her hands clasped. Across from her, a man in a grey suit stands near the window, his back partially turned. He turns to face her with a tense expression. The camera is static, slightly below eye level. The color palette is muted with warm tones, creating an intimate, dramatic atmosphere.

## Stage 3: Caption Enrichment

### Merge Dialogue

If subtitles are available, prepend dialogue to the visual caption:

```python
def enrich_caption(visual_caption, subtitle_text=None, style_tags=None):
    """Combine visual caption with dialogue and style information."""
    parts = []

    # Add consistent style prefix (learned during training, used during inference)
    if style_tags:
        parts.append(", ".join(style_tags))

    # Add visual description
    parts.append(visual_caption)

    # Add dialogue if available
    if subtitle_text and subtitle_text.strip():
        parts.append(f'The dialogue is: "{subtitle_text.strip()}"')

    return " ".join(parts)
```

### Style Tags

Define consistent style tags that describe the show's overall look. These become trigger words during inference:

```yaml
# config/style_tags.yaml
style_tags:
  - "cinematic television scene"
  - "dramatic lighting"
  # Add show-specific tags as appropriate:
  # - "noir style"
  # - "bright sitcom lighting"
  # - "handheld camera documentary style"
```

### Final Caption Format

```
cinematic television scene, dramatic lighting. A medium shot in a dimly lit
living room with warm amber lighting from table lamps. A woman with dark hair
wearing a blue cardigan sits on a brown leather couch, leaning forward with
her hands clasped. Across from her, a man in a grey suit stands near the
window. He turns to face her with a tense expression. The camera is static,
slightly below eye level, muted warm tones. The dialogue is: "I told you we
should have left earlier."
```

## Stage 4: Caption Review (Optional)

For best results, review and correct auto-generated captions. This is tedious but significantly improves training quality.

### Simple Review Tool

```python
# caption_review.py -- Simple terminal-based review
import json
import os

def review_captions(metadata_dir):
    files = sorted(os.listdir(metadata_dir))
    for f in files:
        if not f.endswith(".json"):
            continue

        path = os.path.join(metadata_dir, f)
        with open(path) as fh:
            meta = json.load(fh)

        if meta.get("caption_reviewed"):
            continue

        print(f"\n{'='*60}")
        print(f"Clip: {meta['clip_id']}")
        print(f"Caption: {meta['caption']}")
        print(f"{'='*60}")

        action = input("[a]ccept / [e]dit / [s]kip / [d]elete clip > ").strip().lower()

        if action == "a":
            meta["caption_reviewed"] = True
        elif action == "e":
            new_caption = input("New caption: ").strip()
            meta["caption"] = new_caption
            meta["caption_reviewed"] = True
        elif action == "d":
            meta["deleted"] = True
        # skip = leave as-is, review later

        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)
```

For larger datasets, consider building a simple web viewer (Flask/FastAPI) that displays the clip alongside its caption.

## Output

After captioning, each clip's metadata JSON gains a populated `caption` field:

```json
{
  "clip_id": "ep01_scene0001_clip001",
  "caption": "cinematic television scene, dramatic lighting. A medium shot in a dimly lit living room...",
  "caption_source": "qwen2-vl-7b-instruct-4bit",
  "caption_reviewed": false,
  "subtitle_text": "I told you we should have left earlier."
}
```

## VRAM Management

Captioning and training compete for VRAM. **Do not run both simultaneously.**

```
Captioning (Qwen2-VL-7B at 4-bit):  ~6-8 GB VRAM
Training (Wan 2.1 1.3B LoRA):       ~14-16 GB VRAM
```

Run captioning as a batch job first, then free VRAM for training.

## CLI Interface

```bash
# Caption all uncaptioned clips
python -m videoforge.caption --config configs/caption.yaml --dataset /path/to/dataset/

# Caption specific clips
python -m videoforge.caption --clips ep01_scene0001_clip001 ep01_scene0002_clip001

# Review captions interactively
python -m videoforge.caption.review --dataset /path/to/dataset/

# Export captions to txt files (for training tools that expect .txt sidecar files)
python -m videoforge.caption.export --dataset /path/to/dataset/ --format txt
```

## Caption File Export

Many training tools (kohya-ss, OneTrainer) expect caption text files alongside video files:

```
dataset/
├── clips/
│   ├── ep01_scene0001_clip001.mp4
│   ├── ep01_scene0001_clip001.txt    # <- caption text file
│   ├── ep01_scene0012_clip001.mp4
│   ├── ep01_scene0012_clip001.txt
│   └── ...
```

```python
def export_captions_as_txt(metadata_dir, clips_dir):
    """Write .txt caption files alongside clip files."""
    for meta_file in os.listdir(metadata_dir):
        if not meta_file.endswith(".json"):
            continue
        with open(os.path.join(metadata_dir, meta_file)) as f:
            meta = json.load(f)
        if meta.get("deleted") or not meta.get("caption"):
            continue
        txt_path = os.path.join(clips_dir, meta["clip_id"] + ".txt")
        with open(txt_path, "w") as f:
            f.write(meta["caption"])
```
