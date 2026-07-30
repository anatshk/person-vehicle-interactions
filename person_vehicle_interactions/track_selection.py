"""
Pure selection/geometry helpers for the visualization tools.

Stdlib + our dataclasses only (no cv2/matplotlib), so these are unit-tested in lean CI.
The heavy frame-reading and montage rendering live in ``visualization`` (glue).
"""

from __future__ import annotations

from person_vehicle_interactions.tracked_data_model import TrackedBox


def sample_track_boxes(
    boxes: list[TrackedBox], track_id: int, step: int
) -> list[TrackedBox]:
    """
    Return one track's boxes, sorted by frame, keeping every ``step``-th box.

    ``step`` of 1 keeps all boxes of the track; 2 keeps every other, and so on.
    """
    track_boxes = sorted(
        (box for box in boxes if box.track_id == track_id), key=lambda box: box.frame
    )
    return track_boxes[::step]


def highest_confidence_box_per_track(boxes: list[TrackedBox]) -> list[TrackedBox]:
    """
    Return each track's highest-confidence box, one per track, sorted by track id.

    Ties on confidence are broken by larger box area, then earliest frame, so the
    result is deterministic.
    """
    best_by_track: dict[int, TrackedBox] = {}
    for box in boxes:
        current_best = best_by_track.get(box.track_id)
        if current_best is None or _selection_key(box) > _selection_key(current_best):
            best_by_track[box.track_id] = box
    return [best_by_track[track_id] for track_id in sorted(best_by_track)]


def _selection_key(box: TrackedBox) -> tuple[float, float, int]:
    """Sort key for picking a track's representative box: confidence, area, earliest."""
    return (box.confidence, _box_area(box), -box.frame)


def crop_window(
    center_x: float,
    center_y: float,
    frame_width: int,
    frame_height: int,
    fraction: float = 0.25,
) -> tuple[int, int, int, int]:
    """
    Fixed-size crop window (``fraction`` of frame size) centered on a point.

    Returns integer ``(x1, y1, x2, y2)`` corners of the window. The window always keeps
    its full size and is allowed to extend past the frame (corners may be negative or
    exceed the frame) — the caller pads the out-of-bounds area with black rather than
    shifting or shrinking the window.
    """
    window_width = round(fraction * frame_width)
    window_height = round(fraction * frame_height)
    x1 = round(center_x - window_width / 2)
    y1 = round(center_y - window_height / 2)
    return x1, y1, x1 + window_width, y1 + window_height


def _box_area(box: TrackedBox) -> float:
    """Area of a tracked box in square pixels."""
    return (box.x2 - box.x1) * (box.y2 - box.y1)
