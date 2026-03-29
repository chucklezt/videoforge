"""Stage 4: Interactive caption review."""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def review_captions(metadata_dir: str | Path, clips_dir: str | Path | None = None):
    """Interactive terminal-based caption review.

    For each uncaptioned clip, display the caption and allow the user to
    accept, edit, skip, or delete.
    """
    metadata_dir = Path(metadata_dir)
    files = sorted(metadata_dir.glob("*.json"))

    total = 0
    reviewed = 0
    skipped = 0

    for meta_path in files:
        with open(meta_path) as f:
            meta = json.load(f)

        # Skip deleted clips
        if meta.get("deleted"):
            continue

        # Skip clips without captions
        if not meta.get("caption"):
            continue

        # Skip already reviewed
        if meta.get("caption_reviewed"):
            continue

        total += 1

        print(f"\n{'=' * 60}")
        print(f"Clip: {meta['clip_id']}")
        print(f"Duration: {meta.get('duration_sec', '?')}s")
        if clips_dir:
            clip_path = Path(clips_dir) / f"{meta['clip_id']}.mp4"
            print(f"File: {clip_path}")
        print(f"\nCaption:\n{meta['caption']}")
        print(f"{'=' * 60}")

        while True:
            action = input("[a]ccept / [e]dit / [s]kip / [d]elete > ").strip().lower()
            if action in ("a", "e", "s", "d"):
                break
            print("Invalid choice. Use a/e/s/d.")

        if action == "a":
            meta["caption_reviewed"] = True
            reviewed += 1
        elif action == "e":
            new_caption = input("New caption: ").strip()
            if new_caption:
                meta["caption"] = new_caption
                meta["caption_reviewed"] = True
                reviewed += 1
            else:
                print("Empty caption, skipping.")
                skipped += 1
                continue
        elif action == "d":
            meta["deleted"] = True
            reviewed += 1
        elif action == "s":
            skipped += 1
            continue

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\nReview complete: {reviewed} reviewed, {skipped} skipped, {total} total")
