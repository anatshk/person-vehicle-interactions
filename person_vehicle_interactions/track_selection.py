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


SpanFrame = tuple[int, str]


def sample_span_frames(
    start_frame: int, end_frame: int, in_count: int, pad: int
) -> list[SpanFrame]:
    """
    Sample frames across an interaction span plus context frames before and after.
    Returns ``(frame, position)`` pairs in ascending frame order, where position is
    ``"pre"`` (the ``pad`` frames before ``start_frame``), ``"in"`` (``in_count`` frames
    sampled evenly across the inclusive span, endpoints included), or ``"post"`` (the
    ``pad`` frames after ``end_frame``). Frames below 0 are dropped and duplicate ``"in"``
    frames from rounding on a short span are collapsed.
    """
    pre_frames = [
        (frame, "pre") for frame in range(start_frame - pad, start_frame) if frame >= 0
    ]
    in_frames = [
        (frame, "in") for frame in _even_span_frames(start_frame, end_frame, in_count)
    ]
    post_frames = [
        (frame, "post") for frame in range(end_frame + 1, end_frame + 1 + pad)
    ]
    return pre_frames + in_frames + post_frames


def _even_span_frames(start_frame: int, end_frame: int, in_count: int) -> list[int]:
    """Sample ``in_count`` frames evenly across ``[start, end]`` (endpoints included)."""
    if in_count <= 0:
        return []
    if in_count == 1:
        return [start_frame]
    span = end_frame - start_frame
    frames = [
        start_frame + round(step * span / (in_count - 1)) for step in range(in_count)
    ]
    return sorted(dict.fromkeys(frames))


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
