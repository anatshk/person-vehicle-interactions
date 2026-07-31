"""Tests for the shipped-vs-LOSO threshold comparison (pure assembly helpers)."""

from __future__ import annotations

from person_vehicle_interactions.candidate_detection import Thresholds
from person_vehicle_interactions.evaluation import EvalResult
from scripts.compare_thresholds import (
    build_comparison,
    overall_result,
    SceneComparison,
    threshold_field_diffs,
)
from scripts.run_loso import FoldReport

SHIPPED = Thresholds(
    min_overlap=0.2,
    max_distance=0.0,
    min_duration_frames=10,
    min_confidence=0.3,
    max_gap_frames=15,
)


def _result(
    true_positives: int, false_positives: int, false_negatives: int
) -> EvalResult:
    """A minimal EvalResult; precision/recall/f1 are unused by these tests."""
    return EvalResult(true_positives, false_positives, false_negatives, 0.0, 0.0, 0.0)


def test_threshold_field_diffs_reports_only_differing_fields():
    fold = Thresholds(
        min_overlap=0.2,
        max_distance=0.0,
        min_duration_frames=5,
        min_confidence=0.3,
        max_gap_frames=5,
    )
    diffs = threshold_field_diffs(fold, SHIPPED)
    assert diffs == {
        "min_duration_frames": (5, 10),
        "max_gap_frames": (5, 15),
    }


def test_threshold_field_diffs_empty_when_identical():
    assert threshold_field_diffs(SHIPPED, SHIPPED) == {}


def test_overall_result_micro_averages_counts():
    overall = overall_result([_result(3, 1, 0), _result(1, 2, 2)])
    assert (
        overall.true_positives,
        overall.false_positives,
        overall.false_negatives,
    ) == (
        4,
        3,
        2,
    )
    assert overall.precision == 4 / 7
    assert overall.recall == 4 / 6
    assert round(overall.f1, 4) == round(2 * (4 / 7) * (4 / 6) / (4 / 7 + 4 / 6), 4)


def test_build_comparison_pairs_scenes_with_diffs():
    fold = Thresholds(
        min_overlap=0.2,
        max_distance=0.0,
        min_duration_frames=5,
        min_confidence=0.3,
        max_gap_frames=15,
    )
    loso_reports = {"ptz": FoldReport(fold, _result(2, 1, 0))}
    single_reports = {"ptz": FoldReport(SHIPPED, _result(1, 0, 1))}

    comparisons = build_comparison(loso_reports, single_reports, SHIPPED)

    assert comparisons == [
        SceneComparison(
            scene="ptz",
            fold_thresholds=fold,
            loso_result=_result(2, 1, 0),
            single_result=_result(1, 0, 1),
            threshold_diffs={"min_duration_frames": (5, 10)},
        )
    ]
