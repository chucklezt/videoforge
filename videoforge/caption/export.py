"""Export captions to .txt sidecar files for training tools."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_captions_txt(
    metadata_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """Write .txt caption files alongside clip files.

    Training tools (kohya-ss, OneTrainer) expect caption text files
    with the same name as the video file.

    Returns count of exported captions.
    """
    metadata_dir = Path(metadata_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for meta_path in sorted(metadata_dir.glob("*.json")):
        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("deleted"):
            continue
        if not meta.get("caption"):
            continue
        if not meta.get("filter_passed", True):
            continue

        txt_path = output_dir / f"{meta['clip_id']}.txt"
        txt_path.write_text(meta["caption"])
        count += 1

    logger.info("Exported %d caption files to %s", count, output_dir)
    return count
