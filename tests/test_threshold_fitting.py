"""Tests for grid-search threshold fitting (maximize F1 on training clips)."""

from __future__ import annotations

from person_vehicle_interactions.candidate_detection import Thresholds
from person_vehicle_interactions.threshold_fitting import (
    fit_thresholds,
    FitResult,
    threshold_grid,
)
from tests.factories import make_interaction_gt as _gt
from tests.factories import make_person_in_vehicle_boxes

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
    boxes_by_clip = {"clipA": make_person_in_vehicle_boxes()}
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
    boxes_by_clip = {"clipA": make_person_in_vehicle_boxes()}
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
        "clipA": make_person_in_vehicle_boxes(),
        "clipB": make_person_in_vehicle_boxes(frames=range(30, 41)),
    }
    vehicle_classes_by_clip = {"clipA": {"car"}, "clipB": {"car"}}
    gt_by_clip = {"clipA": [_gt(10, 20, "clipA")], "clipB": [_gt(30, 40, "clipB")]}
    good = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=3)

    result = fit_thresholds(
        boxes_by_clip, vehicle_classes_by_clip, gt_by_clip, grid=[good]
    )

    assert result.train_result.true_positives == 2
    assert result.train_result.f1 == 1.0
