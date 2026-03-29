"""VRAM monitoring utilities for AMD GPUs."""

from pathlib import Path


def get_vram_usage() -> dict | None:
    """Get current VRAM usage from sysfs (AMD GPUs). Returns None if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return {
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "total_bytes": total,
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "total_gb": total / (1024**3),
                "free_gb": (total - allocated) / (1024**3),
                "utilization_pct": (allocated / total) * 100 if total > 0 else 0,
            }
    except Exception:
        pass

    # Fallback: read from sysfs
    try:
        drm_cards = list(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
        if not drm_cards:
            return None
        used = int(drm_cards[0].read_text().strip())
        total_path = drm_cards[0].parent / "mem_info_vram_total"
        total = int(total_path.read_text().strip())
        return {
            "allocated_bytes": used,
            "reserved_bytes": used,
            "total_bytes": total,
            "allocated_gb": used / (1024**3),
            "reserved_gb": used / (1024**3),
            "total_gb": total / (1024**3),
            "free_gb": (total - used) / (1024**3),
            "utilization_pct": (used / total) * 100 if total > 0 else 0,
        }
    except Exception:
        return None


def log_vram(label: str = "") -> None:
    """Print current VRAM usage."""
    usage = get_vram_usage()
    if usage is None:
        print(f"[VRAM] {label} -- unavailable")
        return
    prefix = f"[VRAM] {label} -- " if label else "[VRAM] "
    print(f"{prefix}{usage['allocated_gb']:.1f}GB / {usage['total_gb']:.1f}GB "
          f"({usage['utilization_pct']:.0f}%) used")
