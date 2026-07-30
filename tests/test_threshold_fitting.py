"""Tests for grid-search threshold fitting (maximize F1 on training clips)."""

from __future__ import annotations

from person_vehicle_interactions.candidate_detection import Thresholds
from person_vehicle_interactions.gt_windows import InteractionWindow
from person_vehicle_interactions.threshold_fitting import (
    fit_thresholds,
    FitResult,
    threshold_grid,
)
from tests.factories import make_tracked_box


def _clip_boxes_person_in_car(person_id=1, vehicle_id=2, frames=range(10, 21)):
    """A person sitting fully inside a car across ``frames`` (overlap == 1.0)."""
    boxes = []
    for frame in frames:
        boxes.append(
            make_tracked_box(
                frame=frame,
                track_id=person_id,
                object_class="person",
                x1=10,
                y1=10,
                x2=20,
                y2=20,
            )
        )
        boxes.append(
            make_tracked_box(
                frame=frame,
                track_id=vehicle_id,
                object_class="car",
                x1=0,
                y1=0,
                x2=100,
                y2=100,
            )
        )
    return boxes


def _gt(start_frame, end_frame, clip_id="clipA"):
    return InteractionWindow(
        clip_id=clip_id,
        interaction_id=1,
        interaction_type="enter",
        start_frame=start_frame,
        end_frame=end_frame,
        person="a person",
        vehicle="a car",
    )


# --- threshold_grid -----------------------------------------------------------


def test_threshold_grid_is_cartesian_product():
    grid = threshold_grid(
        min_overlaps=[0.05, 0.2],
        max_distances=[1.0],
        min_durations=[2, 5],
        min_confidences=[0.0],
        max_gap_frames=[0, 1],
    )
    assert len(grid) == 2 * 1 * 2 * 1 * 2
    assert all(isinstance(thresholds, Thresholds) for thresholds in grid)
    assert (
        Thresholds(
            min_overlap=0.05,
            max_distance=1.0,
            min_duration_frames=2,
            min_confidence=0.0,
            max_gap_frames=0,
        )
        in grid
    )


# --- fit_thresholds -----------------------------------------------------------


def test_fit_picks_thresholds_that_maximize_f1():
    boxes_by_clip = {"clipA": _clip_boxes_person_in_car()}
    vehicle_classes_by_clip = {"clipA": {"car"}}
    gt_by_clip = {"clipA": [_gt(10, 20)]}
    good = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=3)
    bad = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=100)

    result = fit_thresholds(
        boxes_by_clip, vehicle_classes_by_clip, gt_by_clip, grid=[bad, good]
    )

    assert isinstance(result, FitResult)
    assert result.thresholds == good
    assert result.train_result.f1 == 1.0


def test_fit_breaks_ties_by_grid_order():
    # Two thresholds that both detect the interaction perfectly -> first in grid wins.
    boxes_by_clip = {"clipA": _clip_boxes_person_in_car()}
    vehicle_classes_by_clip = {"clipA": {"car"}}
    gt_by_clip = {"clipA": [_gt(10, 20)]}
    first = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=2)
    second = Thresholds(min_overlap=0.1, max_distance=1.0, min_duration_frames=2)

    result = fit_thresholds(
        boxes_by_clip, vehicle_classes_by_clip, gt_by_clip, grid=[first, second]
    )

    assert result.thresholds == first


def test_fit_aggregates_over_multiple_clips():
    boxes_by_clip = {
        "clipA": _clip_boxes_person_in_car(),
        "clipB": _clip_boxes_person_in_car(frames=range(30, 41)),
    }
    vehicle_classes_by_clip = {"clipA": {"car"}, "clipB": {"car"}}
    gt_by_clip = {"clipA": [_gt(10, 20, "clipA")], "clipB": [_gt(30, 40, "clipB")]}
    good = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=3)

    result = fit_thresholds(
        boxes_by_clip, vehicle_classes_by_clip, gt_by_clip, grid=[good]
    )

    assert result.train_result.true_positives == 2
    assert result.train_result.f1 == 1.0
