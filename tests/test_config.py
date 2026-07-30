"""Tests for the vehicle-class definition and per-clip class helpers."""

from __future__ import annotations

from person_vehicle_interactions.config import (
    target_class_ids_for_clip,
    VEHICLE_CLASS_IDS,
    vehicle_class_ids_for_clip,
)


def test_vehicle_class_ids_default_for_unknown_clip():
    assert vehicle_class_ids_for_clip("some_clip") == VEHICLE_CLASS_IDS
    assert vehicle_class_ids_for_clip("some_clip") == (2, 5, 7)


def test_vehicle_class_ids_includes_clip_override():
    # HIu4lM4B8hA_1 is a low-res night clip where cars misdetect as 'boat' (8).
    result = vehicle_class_ids_for_clip("HIu4lM4B8hA_1")
    assert set(result) == {2, 5, 7, 8}
    assert list(result) == sorted(result)  # deterministic order


def test_target_class_ids_prepends_person():
    assert target_class_ids_for_clip("some_clip") == (0, 2, 5, 7)
    assert target_class_ids_for_clip("HIu4lM4B8hA_1") == (0, 2, 5, 7, 8)
