"""Shared factories for building test objects."""

from __future__ import annotations

from person_vehicle_interactions.tracked_data_model import frame_to_seconds, TrackedBox


def make_tracked_box(
    frame: int = 0,
    track_id: int = 1,
    object_class: str = "person",
    x1: float = 10.0,
    y1: float = 20.0,
    x2: float = 30.0,
    y2: float = 40.0,
    confidence: float = 0.9,
    fps: float = 30.0,
) -> TrackedBox:
    """Build a valid TrackedBox with sensible defaults for tests."""
    return TrackedBox(
        frame=frame,
        time_seconds=frame_to_seconds(frame, fps),
        track_id=track_id,
        object_class=object_class,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
    )
