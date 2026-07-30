"""Smoke tests for the LOSO wiring closures (fit / evaluate over cached tracks)."""

from __future__ import annotations

from person_vehicle_interactions.candidate_detection import Thresholds
from scripts.run_loso import FoldReport, make_evaluate, make_fit_thresholds
from tests.factories import make_interaction_gt as _gt
from tests.factories import make_person_in_vehicle_boxes


def test_make_evaluate_returns_fold_report_scoring_test_clips():
    boxes_by_clip = {"clipX": make_person_in_vehicle_boxes()}
    gt_by_clip = {"clipX": [_gt(10, 20, clip_id="clipX")]}
    evaluate = make_evaluate(boxes_by_clip, gt_by_clip)
    thresholds = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=3)

    report = evaluate(("clipX",), thresholds)

    assert isinstance(report, FoldReport)
    assert report.thresholds == thresholds
    assert report.result.f1 == 1.0


def test_make_fit_thresholds_selects_best_from_grid():
    boxes_by_clip = {"clipX": make_person_in_vehicle_boxes()}
    gt_by_clip = {"clipX": [_gt(10, 20, clip_id="clipX")]}
    good = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=3)
    bad = Thresholds(min_overlap=0.05, max_distance=1.0, min_duration_frames=100)
    fit = make_fit_thresholds(boxes_by_clip, gt_by_clip, grid=[bad, good])

    assert fit(("clipX",)) == good
