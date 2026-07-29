"""Pure detection-processing helpers: build TrackedBoxes and filter them by class."""

from __future__ import annotations

from collections.abc import Sequence

from person_vehicle_interactions.tracked_data_model import frame_to_seconds, TrackedBox

# COCO class ids we care about, mapped to the names used throughout the pipeline.
COCO_ID_TO_NAME: dict[int, str] = {0: "person", 2: "car"}


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
