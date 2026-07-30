"""Tests for the pure track-selection helpers used by the visualization tools."""

from __future__ import annotations

from person_vehicle_interactions.track_selection import (
    crop_window,
    highest_confidence_box_per_track,
    sample_track_boxes,
)
from tests.factories import make_tracked_box


def test_sample_track_boxes_filters_by_track_and_sorts_by_frame():
    boxes = [
        make_tracked_box(frame=4, track_id=1),
        make_tracked_box(frame=0, track_id=2),
        make_tracked_box(frame=2, track_id=1),
        make_tracked_box(frame=0, track_id=1),
    ]
    result = sample_track_boxes(boxes, track_id=1, step=1)
    assert [box.frame for box in result] == [0, 2, 4]


def test_sample_track_boxes_applies_step():
    boxes = [make_tracked_box(frame=frame, track_id=1) for frame in range(6)]
    result = sample_track_boxes(boxes, track_id=1, step=2)
    assert [box.frame for box in result] == [0, 2, 4]


def test_sample_track_boxes_missing_track_returns_empty():
    boxes = [make_tracked_box(frame=0, track_id=1)]
    assert sample_track_boxes(boxes, track_id=99, step=1) == []


def test_highest_confidence_box_per_track_picks_max_confidence():
    boxes = [
        make_tracked_box(frame=0, track_id=1, confidence=0.6),
        make_tracked_box(frame=1, track_id=1, confidence=0.9),
        make_tracked_box(frame=0, track_id=2, confidence=0.5),
    ]
    result = highest_confidence_box_per_track(boxes)
    assert [box.track_id for box in result] == [1, 2]
    assert result[0].frame == 1
    assert result[0].confidence == 0.9
    assert result[1].frame == 0


def test_highest_confidence_ties_prefer_larger_area():
    boxes = [
        make_tracked_box(frame=0, track_id=1, confidence=0.8, x1=0, y1=0, x2=10, y2=10),
        make_tracked_box(frame=1, track_id=1, confidence=0.8, x1=0, y1=0, x2=20, y2=20),
    ]
    result = highest_confidence_box_per_track(boxes)
    assert len(result) == 1
    assert result[0].frame == 1  # same confidence -> larger box wins


def test_highest_confidence_sorted_by_track_id():
    boxes = [
        make_tracked_box(frame=0, track_id=3, confidence=0.7),
        make_tracked_box(frame=0, track_id=1, confidence=0.7),
    ]
    result = highest_confidence_box_per_track(boxes)
    assert [box.track_id for box in result] == [1, 3]


def test_crop_window_centered_has_expected_size():
    x1, y1, x2, y2 = crop_window(
        center_x=500.0, center_y=400.0, frame_width=1000, frame_height=800
    )
    assert (x2 - x1, y2 - y1) == (250, 200)
    assert (x1, y1, x2, y2) == (375, 300, 625, 500)


def test_crop_window_near_edge_is_not_clamped():
    # Window keeps its full size and is allowed to extend past the frame (glue pads
    # with black), so it never shifts or shrinks at the border.
    x1, y1, x2, y2 = crop_window(
        center_x=0.0, center_y=0.0, frame_width=1000, frame_height=800
    )
    assert (x2 - x1, y2 - y1) == (250, 200)
    assert (x1, y1) == (-125, -100)


def test_crop_window_respects_fraction():
    x1, y1, x2, y2 = crop_window(
        center_x=500.0,
        center_y=500.0,
        frame_width=1000,
        frame_height=1000,
        fraction=0.5,
    )
    assert (x2 - x1, y2 - y1) == (500, 500)
