"""Tests for pure candidate-interaction detection (thresholds -> predicted windows)."""

from __future__ import annotations

from person_vehicle_interactions.candidate_detection import (
    detect_candidates,
    detect_candidates_from_series,
    frame_is_contact,
    pair_series_by_pair,
    PredictedWindow,
    Thresholds,
)
from tests.factories import make_pair_signal as _signal
from tests.factories import make_person as _person
from tests.factories import make_vehicle as _vehicle
from tests.factories import VEHICLE_CLASSES

# --- frame_is_contact ---------------------------------------------------------


def test_frame_is_contact_true_on_overlap():
    thresholds = Thresholds(min_overlap=0.2, max_distance=0.5, min_duration_frames=1)
    assert frame_is_contact(_signal(0, overlap=0.3, distance=9.0), thresholds)


def test_frame_is_contact_true_on_proximity_without_overlap():
    thresholds = Thresholds(min_overlap=0.2, max_distance=0.5, min_duration_frames=1)
    assert frame_is_contact(_signal(0, overlap=0.0, distance=0.4), thresholds)


def test_frame_is_contact_false_when_far_and_no_overlap():
    thresholds = Thresholds(min_overlap=0.2, max_distance=0.5, min_duration_frames=1)
    assert not frame_is_contact(_signal(0, overlap=0.1, distance=0.9), thresholds)


def test_frame_is_contact_false_below_confidence_floor():
    thresholds = Thresholds(
        min_overlap=0.2, max_distance=0.5, min_duration_frames=1, min_confidence=0.5
    )
    weak = _signal(0, overlap=0.9, distance=0.0, person_confidence=0.4)
    assert not frame_is_contact(weak, thresholds)


# --- detect_candidates --------------------------------------------------------


def test_detect_sustained_overlap_yields_one_window():
    thresholds = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=3)
    boxes = [
        box
        for frame in range(5)
        for box in (
            _person(frame, 1, 10, 10, 20, 20),
            _vehicle(frame, 2, 0, 0, 100, 100),
        )
    ]
    windows = detect_candidates(boxes, VEHICLE_CLASSES, thresholds)
    assert windows == [
        PredictedWindow(person_id=1, vehicle_id=2, start_frame=0, end_frame=4)
    ]


def test_detect_rejects_momentary_contact_below_min_duration():
    # Person overlaps the car for a single frame only (a pass-by), the rest far away.
    thresholds = Thresholds(min_overlap=0.05, max_distance=0.1, min_duration_frames=3)
    boxes = [
        _person(0, 1, 1000, 1000, 1010, 1010),
        _vehicle(0, 2, 0, 0, 100, 100),
        _person(1, 1, 10, 10, 20, 20),  # single overlapping frame
        _vehicle(1, 2, 0, 0, 100, 100),
        _person(2, 1, 1000, 1000, 1010, 1010),
        _vehicle(2, 2, 0, 0, 100, 100),
    ]
    assert detect_candidates(boxes, VEHICLE_CLASSES, thresholds) == []


def test_detect_bridges_small_gap_but_not_large_gap():
    thresholds = Thresholds(
        min_overlap=0.05, max_distance=0.0, min_duration_frames=2, max_gap_frames=1
    )
    inside = lambda frame: (
        _person(frame, 1, 10, 10, 20, 20),
        _vehicle(frame, 2, 0, 0, 100, 100),
    )
    outside = lambda frame: (
        _person(frame, 1, 1000, 1000, 1010, 1010),
        _vehicle(frame, 2, 0, 0, 100, 100),
    )
    # contact at 0,1 -> gap of 1 (frame 2) -> contact at 3,4 : bridged into 0..4
    boxes = [
        *inside(0),
        *inside(1),
        *outside(2),
        *inside(3),
        *inside(4),
    ]
    assert detect_candidates(boxes, VEHICLE_CLASSES, thresholds) == [
        PredictedWindow(person_id=1, vehicle_id=2, start_frame=0, end_frame=4)
    ]
    # Widen the gap to 2 empty frames (2,3) -> two separate windows.
    boxes_wide = [
        *inside(0),
        *inside(1),
        *outside(2),
        *outside(3),
        *inside(4),
        *inside(5),
    ]
    assert detect_candidates(boxes_wide, VEHICLE_CLASSES, thresholds) == [
        PredictedWindow(person_id=1, vehicle_id=2, start_frame=0, end_frame=1),
        PredictedWindow(person_id=1, vehicle_id=2, start_frame=4, end_frame=5),
    ]


def test_detect_multiple_pairs_sorted():
    thresholds = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=2)
    boxes = []
    for frame in range(2):
        boxes += [
            _person(frame, 1, 10, 10, 20, 20),
            _vehicle(frame, 2, 0, 0, 100, 100),
            _person(frame, 3, 30, 30, 40, 40),
            _vehicle(frame, 4, 0, 0, 100, 100),
        ]
    windows = detect_candidates(boxes, VEHICLE_CLASSES, thresholds)
    assert windows == [
        PredictedWindow(person_id=1, vehicle_id=2, start_frame=0, end_frame=1),
        PredictedWindow(person_id=1, vehicle_id=4, start_frame=0, end_frame=1),
        PredictedWindow(person_id=3, vehicle_id=2, start_frame=0, end_frame=1),
        PredictedWindow(person_id=3, vehicle_id=4, start_frame=0, end_frame=1),
    ]


def test_detect_no_contact_is_empty():
    thresholds = Thresholds(min_overlap=0.05, max_distance=0.1, min_duration_frames=1)
    boxes = [
        _person(0, 1, 1000, 1000, 1010, 1010),
        _vehicle(0, 2, 0, 0, 100, 100),
    ]
    assert detect_candidates(boxes, VEHICLE_CLASSES, thresholds) == []


# --- precomputed series (fitting fast path) -----------------------------------


def test_pair_series_by_pair_covers_every_sharing_pair():
    boxes = [
        _person(0, 1, 10, 10, 20, 20),
        _vehicle(0, 2, 0, 0, 100, 100),
        _person(0, 3, 1000, 1000, 1010, 1010),  # shares frame 0 with vehicle 2 but far
    ]
    series_by_pair = pair_series_by_pair(boxes, VEHICLE_CLASSES)
    assert set(series_by_pair) == {(1, 2), (3, 2)}
    assert series_by_pair[(1, 2)][0].overlap == 1.0


def test_pair_series_by_pair_skips_pairs_that_never_share_a_frame():
    boxes = [
        _person(0, 1, 10, 10, 20, 20),
        _vehicle(5, 2, 0, 0, 100, 100),  # different frame -> no shared frames
    ]
    assert pair_series_by_pair(boxes, VEHICLE_CLASSES) == {}


def test_detect_from_series_matches_detect_candidates():
    thresholds = Thresholds(
        min_overlap=0.05, max_distance=0.5, min_duration_frames=2, max_gap_frames=1
    )
    boxes = []
    for frame in range(5):
        boxes += [
            _person(frame, 1, 10, 10, 20, 20),
            _vehicle(frame, 2, 0, 0, 100, 100),
            _person(frame, 3, 30, 30, 40, 40),
            _vehicle(frame, 4, 0, 0, 100, 100),
        ]
    series_by_pair = pair_series_by_pair(boxes, VEHICLE_CLASSES)
    assert detect_candidates_from_series(
        series_by_pair, thresholds
    ) == detect_candidates(boxes, VEHICLE_CLASSES, thresholds)
