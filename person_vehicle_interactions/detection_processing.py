"""Pure detection-processing helpers: build TrackedBoxes and filter them by class."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import resources
import json

from person_vehicle_interactions.tracked_data_model import frame_to_seconds, TrackedBox


def _load_coco_id_to_name() -> dict[int, str]:
    """
    Load the COCO id->name map from the packaged JSON reference.

    Source: the ``yolo11l.pt`` weights' own class names, so the mapping matches exactly
    what the detector emits. ``data/coco_classes.json`` was generated once with::

        from ultralytics import YOLO

        names = YOLO("yolo11l.pt").names  # {0: 'person', 1: 'bicycle', ..., 79: ...}

    It is kept as a data file (not an 80-entry literal) and read here with the stdlib
    only, so this pure module stays free of the heavy ``ultralytics`` dependency.
    """
    resource = resources.files("person_vehicle_interactions").joinpath(
        "data", "coco_classes.json"
    )
    raw_map = json.loads(resource.read_text(encoding="utf-8"))
    return {int(class_id): name for class_id, name in raw_map.items()}


# Full COCO-80 id -> name map (from data/coco_classes.json; see _load_coco_id_to_name).
# Class ids this project uses (grouped in config.py):
#   0  person  — the person in each interaction
#   2  car  /  5  bus  /  7  truck  — the base "vehicle" classes
#   8  boat  — SPECIAL: not a real target. Cars in the low-res night clip
#              (HIu4lM4B8hA_1) misdetect as boat, so boat is added as a per-clip
#              vehicle override (see config.CLIP_VEHICLE_CLASS_OVERRIDES).
COCO_ID_TO_NAME: dict[int, str] = _load_coco_id_to_name()


def build_tracked_boxes(
    frame_index: int,
    fps: float,
    boxes_xyxy: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    track_ids: Sequence[int | None],
    confidences: Sequence[float],
) -> list[TrackedBox]:
    """
    Convert one frame's raw tracker arrays into TrackedBox objects.

    Maps each COCO class id to its name, computes the timestamp from the frame
    rate, and skips detections that have no track id (unconfirmed tracks). Raises
    ``ValueError`` on a class id outside ``COCO_ID_TO_NAME`` (should not happen when
    detection is filtered to those classes).
    """
    time_seconds = frame_to_seconds(frame_index, fps)
    tracked_boxes: list[TrackedBox] = []
    for box_xyxy, class_id, track_id, confidence in zip(
        boxes_xyxy, class_ids, track_ids, confidences
    ):
        if track_id is None:
            continue
        if class_id not in COCO_ID_TO_NAME:
            raise ValueError(
                f"Unexpected class id {class_id}; expected one of "
                f"{sorted(COCO_ID_TO_NAME)}."
            )
        x1, y1, x2, y2 = box_xyxy
        tracked_boxes.append(
            TrackedBox(
                frame=frame_index,
                time_seconds=time_seconds,
                track_id=int(track_id),
                object_class=COCO_ID_TO_NAME[class_id],
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(confidence),
            )
        )
    return tracked_boxes


def filter_detections_by_class(
    boxes: list[TrackedBox], allowed_classes: set[str]
) -> list[TrackedBox]:
    """Return only the boxes whose object_class is allowed, preserving order."""
    return [box for box in boxes if box.object_class in allowed_classes]
