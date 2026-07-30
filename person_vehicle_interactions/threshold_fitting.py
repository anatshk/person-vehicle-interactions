"""
Grid-search threshold fitting: choose the thresholds that maximize F1 on training clips.

Runs :func:`candidate_detection.detect_candidates` over each training clip's tracks for
every candidate threshold set, scores the aggregate against the ground truth with
:func:`evaluation.evaluate_clips`, and keeps the highest-F1 thresholds. Ties are broken by
grid order (the first candidate reaching the best F1 wins) so the fit is deterministic.

Pure (our dataclasses only); this is the ``fit_thresholds`` half injected into the LOSO
framework. The wiring that loads cached tracks and builds the grid lives in the scripts layer.
"""

from __future__ import annotations

import dataclasses
import itertools

from person_vehicle_interactions.candidate_detection import (
    detect_candidates,
    Thresholds,
)
from person_vehicle_interactions.evaluation import EvalResult, evaluate_clips
from person_vehicle_interactions.gt_windows import InteractionWindow
from person_vehicle_interactions.tracked_data_model import TrackedBox


@dataclasses.dataclass(frozen=True)
class FitResult:
    """The best thresholds found plus their aggregate score on the training clips."""

    thresholds: Thresholds
    train_result: EvalResult


def threshold_grid(
    min_overlaps: list[float],
    max_distances: list[float],
    min_durations: list[int],
    min_confidences: list[float],
    max_gap_frames: list[int],
) -> list[Thresholds]:
    """
    Build the Cartesian product of the per-parameter value lists as ``Thresholds``.

    Ordering follows the argument order (outermost = ``min_overlaps``), which also fixes the
    tie-break order used by :func:`fit_thresholds`.
    """
    return [
        Thresholds(
            min_overlap=min_overlap,
            max_distance=max_distance,
            min_duration_frames=min_duration,
            min_confidence=min_confidence,
            max_gap_frames=max_gap,
        )
        for min_overlap, max_distance, min_duration, min_confidence, max_gap in (
            itertools.product(
                min_overlaps,
                max_distances,
                min_durations,
                min_confidences,
                max_gap_frames,
            )
        )
    ]


def fit_thresholds(
    boxes_by_clip: dict[str, list[TrackedBox]],
    vehicle_classes_by_clip: dict[str, set[str]],
    ground_truth_by_clip: dict[str, list[InteractionWindow]],
    grid: list[Thresholds],
) -> FitResult:
    """
    Return the grid thresholds with the best aggregate F1 over the training clips.

    For each candidate, detect windows on every clip (with that clip's vehicle classes) and
    score the aggregate against the ground truth. The first candidate reaching the maximum
    F1 wins the tie, so the result is deterministic given the grid order.
    """
    best: FitResult | None = None
    for thresholds in grid:
        predicted_by_clip = {
            clip_id: detect_candidates(
                boxes,
                vehicle_classes_by_clip[clip_id],
                thresholds,
            )
            for clip_id, boxes in boxes_by_clip.items()
        }
        train_result = evaluate_clips(predicted_by_clip, ground_truth_by_clip)
        if best is None or train_result.f1 > best.train_result.f1:
            best = FitResult(thresholds=thresholds, train_result=train_result)
    if best is None:
        raise ValueError("threshold grid is empty")
    return best
