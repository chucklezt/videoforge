"""Stage 3: Caption enrichment - merge dialogue and style tags."""

import logging
import re
from pathlib import Path

import pysrt

logger = logging.getLogger(__name__)


def load_subtitles(srt_path: str | Path) -> list[dict]:
    """Load an SRT subtitle file into a list of timed text entries."""
    srt_path = Path(srt_path)
    if not srt_path.exists():
        return []

    try:
        subs = pysrt.open(str(srt_path))
    except Exception:
        # Fallback: simple manual parse
        return _parse_srt_simple(srt_path)

    entries = []
    for sub in subs:
        entries.append({
            "start_sec": sub.start.ordinal / 1000.0,
            "end_sec": sub.end.ordinal / 1000.0,
            "text": sub.text.replace("\n", " ").strip(),
        })
    return entries


def _parse_srt_simple(srt_path: Path) -> list[dict]:
    """Simple SRT parser fallback."""
    entries = []
    try:
        text = srt_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    blocks = re.split(r"\n\n+", text.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        time_line = lines[1]
        match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            time_line,
        )
        if not match:
            continue
        g = [int(x) for x in match.groups()]
        start_sec = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end_sec = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        dialogue = " ".join(lines[2:]).strip()
        entries.append({"start_sec": start_sec, "end_sec": end_sec, "text": dialogue})
    return entries


def find_dialogue_for_clip(
    subtitles: list[dict],
    clip_start_sec: float,
    clip_end_sec: float,
    overlap_threshold: float = 0.3,
) -> str | None:
    """Find subtitle text that overlaps with a clip's time range."""
    matching = []
    for sub in subtitles:
        # Calculate overlap
        overlap_start = max(sub["start_sec"], clip_start_sec)
        overlap_end = min(sub["end_sec"], clip_end_sec)
        overlap = max(0, overlap_end - overlap_start)

        sub_duration = sub["end_sec"] - sub["start_sec"]
        if sub_duration > 0 and overlap / sub_duration >= overlap_threshold:
            matching.append(sub["text"])

    if matching:
        return " ".join(matching)
    return None


def enrich_caption(
    visual_caption: str,
    subtitle_text: str | None = None,
    style_tags: list[str] | None = None,
) -> str:
    """Combine visual caption with dialogue and style tags."""
    parts = []

    if style_tags:
        parts.append(", ".join(style_tags))

    parts.append(visual_caption)

    if subtitle_text and subtitle_text.strip():
        parts.append(f'The dialogue is: "{subtitle_text.strip()}"')

    return " ".join(parts)


def load_style_tags(config_path: str | Path) -> list[str]:
    """Load style tags from a YAML config file."""
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        return []

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return data.get("style_tags", [])
