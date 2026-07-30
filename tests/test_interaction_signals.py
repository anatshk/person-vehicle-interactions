"""Tests for the pure per-pair interaction signals (overlap / distance / series)."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.interaction_signals import (
    candidate_pairs,
    normalized_center_distance,
    normalized_intersection,
    pair_signal_series,
    person_track_ids,
    vehicle_track_ids,
)
from tests.factories import make_tracked_box

VEHICLE_CLASSES = {"car", "bus", "truck", "boat"}


def _person(frame, track_id, x1, y1, x2, y2, confidence=0.9):
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


def _vehicle(frame, track_id, x1, y1, x2, y2, confidence=0.8, object_class="car"):
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


def test_normalized_intersection_person_fully_inside_is_one():
    person = _person(0, 1, 10, 10, 20, 20)
    vehicle = _vehicle(0, 2, 0, 0, 100, 100)
    assert normalized_intersection(person, vehicle) == pytest.approx(1.0)


def test_normalized_intersection_disjoint_is_zero():
    person = _person(0, 1, 0, 0, 10, 10)
    vehicle = _vehicle(0, 2, 100, 100, 110, 110)
    assert normalized_intersection(person, vehicle) == 0.0


def test_normalized_intersection_half_inside():
    person = _person(0, 1, 0, 0, 10, 10)  # area 100
    vehicle = _vehicle(0, 2, 5, 0, 100, 100)  # overlaps x:5..10, y:0..10 -> 50
    assert normalized_intersection(person, vehicle) == pytest.approx(0.5)


def test_normalized_center_distance_coincident_is_zero():
    person = _person(0, 1, 10, 10, 20, 20)  # center (15,15)
    vehicle = _vehicle(0, 2, 5, 5, 25, 25)  # center (15,15)
    assert normalized_center_distance(person, vehicle) == pytest.approx(0.0)


def test_normalized_center_distance_normalized_by_vehicle_diagonal():
    vehicle = _vehicle(0, 2, 0, 0, 30, 40)  # diagonal 50, center (15,20)
    person = _person(0, 1, 15, 70, 15, 70)  # center (15,70) -> distance 50
    assert normalized_center_distance(person, vehicle) == pytest.approx(1.0)


def test_pair_signal_series_only_frames_with_both_present_ordered():
    boxes = [
        _person(2, 1, 10, 10, 20, 20),
        _vehicle(2, 2, 0, 0, 100, 100),
        _person(0, 1, 10, 10, 20, 20),
        _vehicle(0, 2, 0, 0, 100, 100),
        _person(1, 1, 10, 10, 20, 20),  # frame 1: no vehicle -> excluded
    ]
    series = pair_signal_series(boxes, person_id=1, vehicle_id=2)
    assert [signal.frame for signal in series] == [0, 2]
    assert series[0].overlap == pytest.approx(1.0)
    assert series[0].person_confidence == pytest.approx(0.9)
    assert series[0].vehicle_confidence == pytest.approx(0.8)


def test_pair_signal_series_missing_pair_is_empty():
    boxes = [_person(0, 1, 0, 0, 1, 1), _vehicle(0, 2, 0, 0, 1, 1)]
    assert pair_signal_series(boxes, person_id=9, vehicle_id=2) == []


def test_person_and_vehicle_track_ids():
    boxes = [
        _person(0, 1, 0, 0, 1, 1),
        _vehicle(0, 2, 0, 0, 1, 1, object_class="car"),
        _vehicle(0, 3, 0, 0, 1, 1, object_class="boat"),
        _person(1, 1, 0, 0, 1, 1),  # duplicate id across frames
    ]
    assert person_track_ids(boxes) == [1]
    assert vehicle_track_ids(boxes, VEHICLE_CLASSES) == [2, 3]


def test_candidate_pairs_keeps_close_drops_far():
    boxes = [
        # person 1 overlaps vehicle 2
        _person(0, 1, 10, 10, 20, 20),
        _vehicle(0, 2, 0, 0, 100, 100),
        # person 3 always far from vehicle 2
        _person(0, 3, 1000, 1000, 1010, 1010),
    ]
    pairs = candidate_pairs(boxes, VEHICLE_CLASSES, max_distance=1.0, min_overlap=0.05)
    assert pairs == [(1, 2)]
