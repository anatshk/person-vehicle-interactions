"""
Pure candidate-interaction detection: thresholds -> predicted interaction windows.

Turns the per-pair overlap/distance/confidence signals into contiguous predicted
windows. A frame counts as *contact* when the person is inside or close to the vehicle
(and both boxes clear a confidence floor); contact frames are grouped into runs, small
gaps are bridged, and only runs long enough to reject momentary pass-bys are kept.

Stdlib + our dataclasses only, so it is unit-tested in lean CI. Thresholds are fit
against the ground truth via LOSO downstream.
"""

from __future__ import annotations

import dataclasses

from person_vehicle_interactions.interaction_signals import (
    pair_signal_series,
    PairFrameSignal,
    person_track_ids,
    vehicle_track_ids,
)
from person_vehicle_interactions.tracked_data_model import TrackedBox

PairId = tuple[int, int]


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """
    Tunable thresholds for candidate detection.

    Attributes:
        min_overlap: Minimum normalized intersection to count a frame as contact.
        max_distance: Maximum normalized center distance to count a frame as contact
            when the boxes do not overlap enough.
        min_duration_frames: Minimum window span (frames, inclusive) to keep a run,
            rejecting momentary contacts / pass-bys.
        min_confidence: Both the person and vehicle box must reach this detection
            confidence for a frame to count as contact (0.0 disables the floor).
        max_gap_frames: Bridge runs separated by at most this many non-contact frames
            into a single window (0 keeps every run separate).
    """

    min_overlap: float
    max_distance: float
    min_duration_frames: int
    min_confidence: float = 0.0
    max_gap_frames: int = 0


@dataclasses.dataclass(frozen=True)
class PredictedWindow:
    """A predicted interaction: a (person, vehicle) pair over an inclusive frame span."""

    person_id: int
    vehicle_id: int
    start_frame: int
    end_frame: int


def frame_is_contact(signal: PairFrameSignal, thresholds: Thresholds) -> bool:
    """
    Whether one frame's signals count as person-vehicle contact.

    Contact requires both boxes to clear the confidence floor and either enough overlap
    or close enough proximity.
    """
    if (
        signal.person_confidence < thresholds.min_confidence
        or signal.vehicle_confidence < thresholds.min_confidence
    ):
        return False
    return (
        signal.overlap >= thresholds.min_overlap
        or signal.distance <= thresholds.max_distance
    )


def pair_series_by_pair(
    boxes: list[TrackedBox], vehicle_classes: set[str]
) -> dict[PairId, list[PairFrameSignal]]:
    """
    Precompute the per-frame signal series for every (person, vehicle) pair that shares a
    frame.

    The signals do not depend on any threshold, so computing them once and thresholding the
    result repeatedly (see :func:`detect_candidates_from_series`) avoids re-scanning the
    tracks for every candidate threshold during fitting. Pairs that never share a frame are
    omitted.
    """
    series_by_pair: dict[PairId, list[PairFrameSignal]] = {}
    vehicle_ids = vehicle_track_ids(boxes, vehicle_classes)
    for person_id in person_track_ids(boxes):
        for vehicle_id in vehicle_ids:
            series = pair_signal_series(boxes, person_id, vehicle_id)
            if series:
                series_by_pair[(person_id, vehicle_id)] = series
    return series_by_pair


def detect_candidates_from_series(
    series_by_pair: dict[PairId, list[PairFrameSignal]],
    thresholds: Thresholds,
) -> list[PredictedWindow]:
    """
    Detect predicted interaction windows from precomputed per-pair signal series.

    Returned sorted by (person_id, vehicle_id, start_frame).
    """
    windows: list[PredictedWindow] = []
    for (person_id, vehicle_id), series in sorted(series_by_pair.items()):
        contact_frames = [
            signal.frame for signal in series if frame_is_contact(signal, thresholds)
        ]
        for start_frame, end_frame in _runs(contact_frames, thresholds.max_gap_frames):
            if end_frame - start_frame + 1 >= thresholds.min_duration_frames:
                windows.append(
                    PredictedWindow(person_id, vehicle_id, start_frame, end_frame)
                )
    return windows


def detect_candidates(
    boxes: list[TrackedBox],
    vehicle_classes: set[str],
    thresholds: Thresholds,
) -> list[PredictedWindow]:
    """
    Detect predicted interaction windows across all (person, vehicle) pairs.

    Returned sorted by (person_id, vehicle_id, start_frame).
    """
    return detect_candidates_from_series(
        pair_series_by_pair(boxes, vehicle_classes), thresholds
    )


def _runs(frames: list[int], max_gap_frames: int) -> list[tuple[int, int]]:
    """
    Group sorted frame indices into (start, end) runs, bridging gaps up to the limit.

    Two consecutive contact frames belong to the same run when the number of missing
    frames between them (``next - prev - 1``) does not exceed ``max_gap_frames``.
    """
    if not frames:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame - previous - 1 <= max_gap_frames:
            previous = frame
        else:
            runs.append((start, previous))
            start = previous = frame
    runs.append((start, previous))
    return runs
