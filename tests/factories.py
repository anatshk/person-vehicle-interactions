"""Shared factories and constants for building test objects."""

from __future__ import annotations

from person_vehicle_interactions.candidate_detection import PredictedWindow
from person_vehicle_interactions.gt_windows import InteractionWindow
from person_vehicle_interactions.interaction_signals import PairFrameSignal
from person_vehicle_interactions.tracked_data_model import frame_to_seconds, TrackedBox

# Every class the tests treat as a vehicle: the global vehicle classes plus ``boat``
# (the per-clip override for the low-res night clip). The production code defines vehicle
# scope by COCO id and per clip (``config.vehicle_class_names_for_clip``), so this flat
# name-set is a test-only convenience and lives here rather than in the code.
VEHICLE_CLASSES: set[str] = {"car", "bus", "truck", "boat"}


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


def make_person(
    frame: int,
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> TrackedBox:
    """Build a person-class TrackedBox."""
    return make_tracked_box(
        frame=frame,
        track_id=track_id,
        object_class="person",
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
    )


def make_vehicle(
    frame: int,
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.8,
    object_class: str = "car",
) -> TrackedBox:
    """Build a vehicle-class TrackedBox (``car`` by default)."""
    return make_tracked_box(
        frame=frame,
        track_id=track_id,
        object_class=object_class,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
    )


def make_pair_signal(
    frame: int,
    overlap: float,
    distance: float,
    person_confidence: float = 0.9,
    vehicle_confidence: float = 0.8,
    fps: float = 30.0,
) -> PairFrameSignal:
    """Build a PairFrameSignal for one frame (time derived from ``fps``)."""
    return PairFrameSignal(
        frame=frame,
        time_seconds=frame_to_seconds(frame, fps),
        overlap=overlap,
        distance=distance,
        person_confidence=person_confidence,
        vehicle_confidence=vehicle_confidence,
    )


def make_person_in_vehicle_boxes(
    frames: range = range(10, 21),
    person_id: int = 1,
    vehicle_id: int = 2,
    person_confidence: float = 0.9,
    vehicle_confidence: float = 0.8,
) -> list[TrackedBox]:
    """
    Tracked boxes for a person sitting fully inside a vehicle across ``frames``.

    The person box (10,10,20,20) is entirely within the vehicle box (0,0,100,100), so the
    normalized overlap is 1.0 on every frame — a clean synthetic interaction.
    """
    boxes: list[TrackedBox] = []
    for frame in frames:
        boxes.append(make_person(frame, person_id, 10, 10, 20, 20, person_confidence))
        boxes.append(
            make_vehicle(frame, vehicle_id, 0, 0, 100, 100, vehicle_confidence)
        )
    return boxes


def make_interaction_gt(
    start_frame: int,
    end_frame: int,
    clip_id: str = "clipA",
    interaction_id: int = 1,
    interaction_type: str = "enter",
    person: str = "a person",
    vehicle: str = "a car",
) -> InteractionWindow:
    """Build a ground-truth InteractionWindow with sensible defaults for tests."""
    return InteractionWindow(
        clip_id=clip_id,
        interaction_id=interaction_id,
        interaction_type=interaction_type,
        start_frame=start_frame,
        end_frame=end_frame,
        person=person,
        vehicle=vehicle,
    )


def make_interaction_prediction(
    start_frame: int,
    end_frame: int,
    person_id: int = 1,
    vehicle_id: int = 2,
) -> PredictedWindow:
    """Build a predicted interaction window with sensible defaults for tests."""
    return PredictedWindow(
        person_id=person_id,
        vehicle_id=vehicle_id,
        start_frame=start_frame,
        end_frame=end_frame,
    )
