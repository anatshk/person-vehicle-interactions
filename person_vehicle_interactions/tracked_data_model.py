"""Data model for tracked detections and the per-clip track/metadata cache.

A ``TrackedBox`` is one detected+tracked object (person or car) in one frame. Tracks
for a clip are cached as CSV (one row per box) alongside a small JSON metadata sidecar,
so downstream analysis never has to re-run detection.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

# CSV column order for the track cache. Kept explicit so the on-disk schema is stable.
TRACK_CSV_COLUMNS: tuple[str, ...] = (
    "frame",
    "time_seconds",
    "track_id",
    "object_class",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
)

PathLike = Path | str


@dataclasses.dataclass
class TrackedBox:
    """A single tracked detection (person or car) in a single frame.

    Coordinates are pixel values of the axis-aligned box corners (top-left ``x1, y1``,
    bottom-right ``x2, y2``).
    """

    frame: int
    time_seconds: float
    track_id: int
    object_class: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def __post_init__(self) -> None:
        if min(self.x1, self.y1, self.x2, self.y2) < 0.0:
            raise ValueError("Box coordinates must be non-negative.")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Box requires x2 >= x1 and y2 >= y1.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be in [0, 1].")


@dataclasses.dataclass
class ClipMetadata:
    """Per-clip metadata: video properties plus the detection/tracking configuration."""

    clip_id: str
    fps: float
    frame_width: int
    frame_height: int
    frame_count: int
    model_name: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    tracker_name: str
    seed: int


def frame_to_seconds(frame: int, fps: float) -> float:
    """Convert a frame index to a timestamp in seconds using the clip's frame rate."""
    return frame / fps


def save_tracks(boxes: list[TrackedBox], csv_path: PathLike) -> None:
    """Write tracked boxes to CSV, sorted by (frame, track_id) for determinism.

    The header row is always written, even when ``boxes`` is empty.
    """
    ordered_boxes = sorted(boxes, key=lambda box: (box.frame, box.track_id))
    with Path(csv_path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(TRACK_CSV_COLUMNS)
        for box in ordered_boxes:
            writer.writerow(
                [
                    box.frame,
                    box.time_seconds,
                    box.track_id,
                    box.object_class,
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                    box.confidence,
                ]
            )


def load_tracks(csv_path: PathLike) -> list[TrackedBox]:
    """Read tracked boxes back from a CSV cache, restoring native Python types.

    Raises ``FileNotFoundError`` if the cache file does not exist.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            TrackedBox(
                frame=int(row["frame"]),
                time_seconds=float(row["time_seconds"]),
                track_id=int(row["track_id"]),
                object_class=row["object_class"],
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                confidence=float(row["confidence"]),
            )
            for row in reader
        ]


def save_metadata(metadata: ClipMetadata, json_path: PathLike) -> None:
    """Write clip metadata to a JSON sidecar."""
    with Path(json_path).open("w", encoding="utf-8") as json_file:
        json.dump(dataclasses.asdict(metadata), json_file, indent=2)


def load_metadata(json_path: PathLike) -> ClipMetadata:
    """Read clip metadata back from a JSON sidecar."""
    with Path(json_path).open(encoding="utf-8") as json_file:
        return ClipMetadata(**json.load(json_file))
