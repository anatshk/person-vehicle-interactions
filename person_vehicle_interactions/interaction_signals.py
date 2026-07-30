"""
Pure per-pair interaction signals over tracked boxes.

For a (person, vehicle) pair, computes per-frame **overlap** (fraction of the person box
inside the vehicle box) and **normalized center distance** (÷ vehicle diagonal), plus the
detection confidences — the raw signals for threshold tuning against the ground truth.

Stdlib + our dataclasses only (no cv2/matplotlib), so this is unit-tested in lean CI. The
plotting glue lives elsewhere.
"""

from __future__ import annotations

import dataclasses
import math

from person_vehicle_interactions.tracked_data_model import TrackedBox

PERSON_CLASS: str = "person"


@dataclasses.dataclass
class PairFrameSignal:
    """One frame's signals for a (person, vehicle) pair (both present that frame)."""

    frame: int
    time_seconds: float
    overlap: float  # normalized intersection: person∩vehicle ÷ person area
    distance: float  # center distance ÷ vehicle diagonal
    person_confidence: float
    vehicle_confidence: float


def normalized_intersection(person: TrackedBox, vehicle: TrackedBox) -> float:
    """
    Fraction of the person box that lies inside the vehicle box (0..1).

    Uses person area as the denominator (not IoU), so it stays sensitive when the person
    is far smaller than the vehicle. Returns 0.0 for a degenerate (zero-area) person.
    """
    person_area = (person.x2 - person.x1) * (person.y2 - person.y1)
    if person_area <= 0.0:
        return 0.0
    return _intersection_area(person, vehicle) / person_area


def normalized_center_distance(person: TrackedBox, vehicle: TrackedBox) -> float:
    """
    Distance between box centers, normalized by the vehicle box diagonal.

    Scale- and egomotion-invariant. Returns ``inf`` for a degenerate vehicle diagonal.
    """
    vehicle_diagonal = math.hypot(vehicle.x2 - vehicle.x1, vehicle.y2 - vehicle.y1)
    if vehicle_diagonal <= 0.0:
        return math.inf
    person_center_x = (person.x1 + person.x2) / 2
    person_center_y = (person.y1 + person.y2) / 2
    vehicle_center_x = (vehicle.x1 + vehicle.x2) / 2
    vehicle_center_y = (vehicle.y1 + vehicle.y2) / 2
    center_distance = math.hypot(
        person_center_x - vehicle_center_x, person_center_y - vehicle_center_y
    )
    return center_distance / vehicle_diagonal


def pair_signal_series(
    boxes: list[TrackedBox], person_id: int, vehicle_id: int
) -> list[PairFrameSignal]:
    """
    Per-frame signals for a (person, vehicle) pair, for frames where both are present.

    Returned in frame order.
    """
    person_by_frame = {box.frame: box for box in boxes if box.track_id == person_id}
    vehicle_by_frame = {box.frame: box for box in boxes if box.track_id == vehicle_id}
    signals: list[PairFrameSignal] = []
    for frame in sorted(person_by_frame.keys() & vehicle_by_frame.keys()):
        person = person_by_frame[frame]
        vehicle = vehicle_by_frame[frame]
        signals.append(
            PairFrameSignal(
                frame=frame,
                time_seconds=person.time_seconds,
                overlap=normalized_intersection(person, vehicle),
                distance=normalized_center_distance(person, vehicle),
                person_confidence=person.confidence,
                vehicle_confidence=vehicle.confidence,
            )
        )
    return signals


def person_track_ids(boxes: list[TrackedBox]) -> list[int]:
    """Sorted unique track ids of person-class boxes."""
    return sorted({box.track_id for box in boxes if box.object_class == PERSON_CLASS})


def vehicle_track_ids(boxes: list[TrackedBox], vehicle_classes: set[str]) -> list[int]:
    """Sorted unique track ids of boxes whose class is in ``vehicle_classes``."""
    return sorted(
        {box.track_id for box in boxes if box.object_class in vehicle_classes}
    )


def candidate_pairs(
    boxes: list[TrackedBox],
    vehicle_classes: set[str],
    max_distance: float,
    min_overlap: float,
) -> list[tuple[int, int]]:
    """
    Return (person_id, vehicle_id) pairs that ever come close, sorted.

    A pair is a candidate if, in any shared frame, its overlap ≥ ``min_overlap`` or its
    normalized distance ≤ ``max_distance``. Prunes the many never-interacting pairs before
    plotting / thresholding.
    """
    candidates: list[tuple[int, int]] = []
    vehicle_ids = vehicle_track_ids(boxes, vehicle_classes)
    for person_id in person_track_ids(boxes):
        for vehicle_id in vehicle_ids:
            series = pair_signal_series(boxes, person_id, vehicle_id)
            if any(
                signal.overlap >= min_overlap or signal.distance <= max_distance
                for signal in series
            ):
                candidates.append((person_id, vehicle_id))
    return candidates


def _intersection_area(a: TrackedBox, b: TrackedBox) -> float:
    """Area of the axis-aligned intersection of two boxes (0.0 if disjoint)."""
    overlap_width = min(a.x2, b.x2) - max(a.x1, b.x1)
    overlap_height = min(a.y2, b.y2) - max(a.y1, b.y1)
    if overlap_width <= 0.0 or overlap_height <= 0.0:
        return 0.0
    return overlap_width * overlap_height
