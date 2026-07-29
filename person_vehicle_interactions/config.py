"""Detection/tracking configuration and cache paths."""

from __future__ import annotations

import dataclasses
from pathlib import Path

CACHE_DIR = Path("cache")
RAW_DIR = CACHE_DIR / "raw"
TRACKS_DIR = CACHE_DIR / "tracks"


@dataclasses.dataclass(frozen=True)
class DetectionConfig:
    """Detection + tracking parameters, pinned for determinism."""

    model_name: str = "yolo11l.pt"
    image_size: int = 1280
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    tracker_name: str = "botsort.yaml"
    target_class_ids: tuple[int, ...] = (0, 2)
    seed: int = 0
