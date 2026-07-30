"""Tests for predicted-vs-ground-truth interaction evaluation (TP/FP/FN + P/R/F1)."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.evaluation import (
    classify_windows,
    EvalResult,
    evaluate_clip,
    evaluate_clips,
    LabeledWindow,
    windows_overlap,
)
from tests.factories import make_interaction_gt as _gt
from tests.factories import make_interaction_prediction as _pred

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
    expected = EvalResult(
        true_positives=1,
        false_positives=0,
        false_negatives=0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
    )
    assert result == expected


def test_evaluate_clip_miss_and_spurious():
    # One predicted far from the only GT: the GT is missed, the prediction is spurious.
    result = evaluate_clip([_pred(100, 110)], [_gt(10, 20)])
    expected_counts = (0, 1, 1)
    assert (
        result.true_positives,
        result.false_positives,
        result.false_negatives,
    ) == expected_counts
    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)


def test_evaluate_clip_fragmentation_counts_extra_as_false_positive():
    # Two predicted windows both overlap one GT -> one TP, one FP (one-to-one matching).
    result = evaluate_clip([_pred(10, 14), _pred(16, 20)], [_gt(10, 20)])
    expected_counts = (1, 1, 0)
    assert (
        result.true_positives,
        result.false_positives,
        result.false_negatives,
    ) == expected_counts
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(2 / 3)


def test_evaluate_clip_no_predictions():
    result = evaluate_clip([], [_gt(10, 20)])
    expected_counts = (0, 0, 1)
    assert (
        result.true_positives,
        result.false_positives,
        result.false_negatives,
    ) == expected_counts
    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)


def test_evaluate_clip_no_ground_truth_with_prediction():
    result = evaluate_clip([_pred(0, 10)], [])
    expected_counts = (0, 1, 0)
    assert (
        result.true_positives,
        result.false_positives,
        result.false_negatives,
    ) == expected_counts
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
    expected_counts = (1, 1, 1)
    assert (
        result.true_positives,
        result.false_positives,
        result.false_negatives,
    ) == expected_counts
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_evaluate_clips_handles_clip_with_gt_but_no_predictions():
    result = evaluate_clips({}, {"clipA": [_gt(10, 20)]})
    expected_counts = (0, 0, 1)
    assert (
        result.true_positives,
        result.false_positives,
        result.false_negatives,
    ) == expected_counts


# --- classify_windows (per-window TP/FP/FN labels) ----------------------------


def test_classify_windows_single_match_is_tp():
    prediction = _pred(8, 22)
    ground_truth = _gt(10, 20)
    labeled = classify_windows([prediction], [ground_truth])
    assert labeled == [
        LabeledWindow("TP", predicted=prediction, ground_truth=ground_truth)
    ]


def test_classify_windows_miss_and_spurious():
    # Predicted far from the only GT: the prediction is an FP, the GT an unmatched FN.
    prediction = _pred(100, 110)
    ground_truth = _gt(10, 20)
    labeled = classify_windows([prediction], [ground_truth])
    assert labeled == [
        LabeledWindow("FP", predicted=prediction, ground_truth=None),
        LabeledWindow("FN", predicted=None, ground_truth=ground_truth),
    ]


def test_classify_windows_fragmentation_extra_is_fp():
    # Two predictions overlap one GT -> first is TP, the extra is an FP (one-to-one).
    first = _pred(10, 14)
    second = _pred(16, 20)
    ground_truth = _gt(10, 20)
    labeled = classify_windows([first, second], [ground_truth])
    assert labeled == [
        LabeledWindow("TP", predicted=first, ground_truth=ground_truth),
        LabeledWindow("FP", predicted=second, ground_truth=None),
    ]


def test_classify_windows_no_predictions_is_fn():
    ground_truth = _gt(10, 20)
    assert classify_windows([], [ground_truth]) == [
        LabeledWindow("FN", predicted=None, ground_truth=ground_truth)
    ]


def test_classify_windows_no_ground_truth_is_fp():
    prediction = _pred(0, 10)
    assert classify_windows([prediction], []) == [
        LabeledWindow("FP", predicted=prediction, ground_truth=None)
    ]


def test_classify_windows_labels_match_evaluate_counts():
    # The per-window labels must reconcile exactly with evaluate_clip's aggregate counts.
    predicted = [_pred(10, 14), _pred(16, 20), _pred(100, 110)]
    ground_truth = [_gt(10, 20), _gt(50, 60)]
    labels = [window.label for window in classify_windows(predicted, ground_truth)]
    result = evaluate_clip(predicted, ground_truth)
    assert labels.count("TP") == result.true_positives
    assert labels.count("FP") == result.false_positives
    assert labels.count("FN") == result.false_negatives
