"""Tests for predicted-vs-ground-truth interaction evaluation (TP/FP/FN + P/R/F1)."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.candidate_detection import PredictedWindow
from person_vehicle_interactions.evaluation import (
    EvalResult,
    evaluate_clip,
    evaluate_clips,
    windows_overlap,
)
from person_vehicle_interactions.gt_windows import InteractionWindow


def _gt(start_frame, end_frame, interaction_id=1, clip_id="clipA"):
    return InteractionWindow(
        clip_id=clip_id,
        interaction_id=interaction_id,
        interaction_type="enter",
        start_frame=start_frame,
        end_frame=end_frame,
        person="a person",
        vehicle="a car",
    )


def _pred(start_frame, end_frame, person_id=1, vehicle_id=2):
    return PredictedWindow(
        person_id=person_id,
        vehicle_id=vehicle_id,
        start_frame=start_frame,
        end_frame=end_frame,
    )


# --- windows_overlap ----------------------------------------------------------


def test_windows_overlap_true_when_spans_intersect():
    assert windows_overlap(_pred(5, 15), _gt(10, 20))


def test_windows_overlap_true_when_touching_at_one_frame():
    assert windows_overlap(_pred(0, 10), _gt(10, 20))


def test_windows_overlap_false_when_disjoint():
    assert not windows_overlap(_pred(0, 9), _gt(10, 20))


# --- evaluate_clip ------------------------------------------------------------


def test_evaluate_clip_single_match_is_perfect():
    result = evaluate_clip([_pred(8, 22)], [_gt(10, 20)])
    assert result == EvalResult(
        true_positives=1,
        false_positives=0,
        false_negatives=0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
    )


def test_evaluate_clip_miss_and_spurious():
    # One predicted far from the only GT: the GT is missed, the prediction is spurious.
    result = evaluate_clip([_pred(100, 110)], [_gt(10, 20)])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (
        0,
        1,
        1,
    )
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_evaluate_clip_fragmentation_counts_extra_as_false_positive():
    # Two predicted windows both overlap one GT -> one TP, one FP (one-to-one matching).
    result = evaluate_clip([_pred(10, 14), _pred(16, 20)], [_gt(10, 20)])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (
        1,
        1,
        0,
    )
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(2 / 3)


def test_evaluate_clip_no_predictions():
    result = evaluate_clip([], [_gt(10, 20)])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (
        0,
        0,
        1,
    )
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_evaluate_clip_no_ground_truth_with_prediction():
    result = evaluate_clip([_pred(0, 10)], [])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (
        0,
        1,
        0,
    )
    assert result.precision == 0.0


# --- evaluate_clips (aggregate across clips) ----------------------------------


def test_evaluate_clips_aggregates_counts_across_clips():
    predicted_by_clip = {
        "clipA": [_pred(8, 22)],  # matches
        "clipB": [_pred(100, 110)],  # spurious
    }
    gt_by_clip = {
        "clipA": [_gt(10, 20, clip_id="clipA")],
        "clipB": [_gt(10, 20, clip_id="clipB")],  # missed
    }
    result = evaluate_clips(predicted_by_clip, gt_by_clip)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (
        1,
        1,
        1,
    )
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_evaluate_clips_handles_clip_with_gt_but_no_predictions():
    result = evaluate_clips({}, {"clipA": [_gt(10, 20)]})
    assert (result.true_positives, result.false_positives, result.false_negatives) == (
        0,
        0,
        1,
    )
