"""
Raw per-frame model output cache (JSONL) and reprocessing into TrackedBoxes.

The cache is **JSONL** (one JSON object per line), not a single JSON document, so frames
can be appended one line at a time during inference, read back by streaming, and survive
an interrupted run: complete lines stay valid and a partial final line is skipped. A
regular JSON array truncated mid-write would be unparseable. (The small ClipMetadata
sidecar stays regular JSON — written once, atomic; tracks are CSV.)

Each line holds one frame's detections. ``tracks_from_raw`` rebuilds TrackedBoxes from a
raw file without re-running the model — the "reprocess without re-inference" path.
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import json
from pathlib import Path
from typing import TextIO

from person_vehicle_interactions.config import DetectionConfig
from person_vehicle_interactions.detection_processing import (
    build_tracked_boxes,
    COCO_ID_TO_NAME,
    filter_detections_by_class,
)
from person_vehicle_interactions.tracked_data_model import TrackedBox

PathLike = Path | str


@dataclasses.dataclass
class RawFrameDetections:
    """One frame's raw detections (parallel arrays), as cached in the raw JSONL."""

    frame: int
    boxes_xyxy: list[list[float]]
    class_ids: list[int]
    track_ids: list[int | None]
    confidences: list[float]


def write_raw_frame(
    raw_file: TextIO,
    frame_index: int,
    boxes_xyxy: list[list[float]],
    class_ids: list[int],
    track_ids: list[int | None],
    confidences: list[float],
) -> None:
    """Append one frame's detections to an open raw JSONL file."""
    record = {
        "frame": frame_index,
        "boxes_xyxy": boxes_xyxy,
        "class_ids": class_ids,
        "track_ids": track_ids,
        "confidences": confidences,
    }
    raw_file.write(json.dumps(record) + "\n")


def read_raw(raw_path: PathLike) -> Iterator[RawFrameDetections]:
    """Yield RawFrameDetections for each non-empty line of a raw JSONL file."""
    with Path(raw_path).open(encoding="utf-8") as raw_file:
        for line in raw_file:
            if not line.strip():
                continue
            record = json.loads(line)
            yield RawFrameDetections(
                frame=record["frame"],
                boxes_xyxy=record["boxes_xyxy"],
                class_ids=record["class_ids"],
                track_ids=record["track_ids"],
                confidences=record["confidences"],
            )


def tracks_from_raw(
    raw_path: PathLike, fps: float, config: DetectionConfig
) -> list[TrackedBox]:
    """
    Rebuild TrackedBoxes from a raw JSONL file, without re-running the model.

    Converts each frame via ``build_tracked_boxes``, then keeps only the classes in
    ``config.target_class_ids`` (or all classes when it is ``None``).
    """
    tracked_boxes: list[TrackedBox] = []
    for frame in read_raw(raw_path):
        tracked_boxes.extend(
            build_tracked_boxes(
                frame.frame,
                fps,
                frame.boxes_xyxy,
                frame.class_ids,
                frame.track_ids,
                frame.confidences,
            )
        )
    if config.target_class_ids is None:
        return tracked_boxes
    allowed_classes = {
        COCO_ID_TO_NAME[class_id] for class_id in config.target_class_ids
    }
    return filter_detections_by_class(tracked_boxes, allowed_classes)
