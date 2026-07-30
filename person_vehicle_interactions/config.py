"""Detection/tracking configuration and cache paths."""

from __future__ import annotations

import dataclasses
from pathlib import Path

VIDEOS_DIR = Path("Videos")
CACHE_DIR = Path("cache")
RAW_DIR = CACHE_DIR / "raw"
TRACKS_DIR = CACHE_DIR / "tracks"
VIZ_DIR = CACHE_DIR / "viz"


@dataclasses.dataclass(frozen=True)
class DetectionConfig:
    """Detection + tracking parameters, pinned for determinism."""

    model_name: str = "yolo11l.pt"
    image_size: int = 1280
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    tracker_name: str = "botsort.yaml"
    # COCO class ids to keep; ``None`` means unfiltered (detect/keep all classes).
    # Default: person (0) + vehicle classes car (2), bus (5), truck (7).
    target_class_ids: tuple[int, ...] | None = (0, 2, 5, 7)
    seed: int = 0
