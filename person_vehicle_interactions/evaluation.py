"""
Evaluate predicted interaction windows against the ground truth.

Matches predicted windows to ground-truth windows per clip by **temporal overlap**
(one-to-one, greedy) and reports true/false positives, false negatives, and the derived
precision / recall / F1. Type and person/vehicle identity are not checked here — P3 scores
*detection* of an interaction span; enter/exit typing and identity-aware matching come later.

Match rule note: a predicted window matches a GT window on **any shared frame**. If this
proves too lenient (many fragmented detections inflating false positives before the
post-processing merge exists), tighten to an overlap-fraction rule here — the matcher is the
single place to change.

Pure (our dataclasses only), unit-tested in lean CI.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from person_vehicle_interactions.candidate_detection import PredictedWindow
from person_vehicle_interactions.gt_windows import InteractionWindow

Label = Literal["TP", "FP", "FN"]


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """Detection scores over a set of windows."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclasses.dataclass(frozen=True)
class LabeledWindow:
    """
    One window with its match outcome.

    A ``TP`` carries both the predicted window and the ground-truth window it matched; an
    ``FP`` carries only the (unmatched) prediction; an ``FN`` carries only the (unmatched)
    ground-truth window.
    """

    label: Label
    predicted: PredictedWindow | None = None
    ground_truth: InteractionWindow | None = None


def windows_overlap(
    predicted: PredictedWindow, ground_truth: InteractionWindow
) -> bool:
    """Whether two frame spans share at least one frame (inclusive bounds)."""
    return (
        predicted.start_frame <= ground_truth.end_frame
        and ground_truth.start_frame <= predicted.end_frame
    )


def evaluate_clip(
    predicted: list[PredictedWindow],
    ground_truth: list[InteractionWindow],
) -> EvalResult:
    """
    Score one clip's predicted windows against its ground-truth windows.

    Each predicted window claims at most one not-yet-matched overlapping GT window (in
    ascending frame order). Matched GT windows are true positives; unclaimed predicted
    windows are false positives; unmatched GT windows are false negatives.
    """
    true_positives, false_positives = _count_matches(predicted, ground_truth)
    false_negatives = len(ground_truth) - true_positives
    return _result(true_positives, false_positives, false_negatives)


def evaluate_clips(
    predicted_by_clip: dict[str, list[PredictedWindow]],
    ground_truth_by_clip: dict[str, list[InteractionWindow]],
) -> EvalResult:
    """
    Aggregate detection scores across clips.

    Matching is done per clip, then the raw counts are summed and precision / recall / F1
    recomputed from the totals.
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    for clip_id in predicted_by_clip.keys() | ground_truth_by_clip.keys():
        predicted = predicted_by_clip.get(clip_id, [])
        ground_truth = ground_truth_by_clip.get(clip_id, [])
        true_positives, false_positives = _count_matches(predicted, ground_truth)
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += len(ground_truth) - true_positives
    return _result(total_true_positives, total_false_positives, total_false_negatives)


def classify_windows(
    predicted: list[PredictedWindow],
    ground_truth: list[InteractionWindow],
) -> list[LabeledWindow]:
    """
    Label each window by its greedy one-to-one temporal match outcome.

    Predictions are processed in ascending start-frame order, each claiming at most one
    not-yet-matched overlapping GT window: a claimed prediction becomes a ``TP`` (carrying
    the matched GT), an unclaimed one an ``FP``. Ground-truth windows left unmatched become
    ``FN``. Returned as all predictions (in start-frame order) followed by the unmatched GT
    windows (in start-frame order) — the same pairing evaluate_clip counts.
    """
    ordered_predicted = sorted(predicted, key=lambda window: window.start_frame)
    ordered_ground_truth = sorted(ground_truth, key=lambda window: window.start_frame)
    matched_ground_truth: set[int] = set()
    labeled: list[LabeledWindow] = []
    for prediction in ordered_predicted:
        match_index = next(
            (
                index
                for index, gt_window in enumerate(ordered_ground_truth)
                if index not in matched_ground_truth
                and windows_overlap(prediction, gt_window)
            ),
            None,
        )
        if match_index is None:
            labeled.append(LabeledWindow("FP", predicted=prediction))
        else:
            matched_ground_truth.add(match_index)
            labeled.append(
                LabeledWindow(
                    "TP",
                    predicted=prediction,
                    ground_truth=ordered_ground_truth[match_index],
                )
            )
    for index, gt_window in enumerate(ordered_ground_truth):
        if index not in matched_ground_truth:
            labeled.append(LabeledWindow("FN", ground_truth=gt_window))
    return labeled


def _count_matches(
    predicted: list[PredictedWindow],
    ground_truth: list[InteractionWindow],
) -> tuple[int, int]:
    """Greedy one-to-one temporal matching -> (true_positives, false_positives)."""
    labels = [window.label for window in classify_windows(predicted, ground_truth)]
    return labels.count("TP"), labels.count("FP")


def _result(
    true_positives: int, false_positives: int, false_negatives: int
) -> EvalResult:
    """Build an ``EvalResult``, deriving precision / recall / F1 with zero-guards."""
    predicted_total = true_positives + false_positives
    actual_total = true_positives + false_negatives
    precision = true_positives / predicted_total if predicted_total else 0.0
    recall = true_positives / actual_total if actual_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvalResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )
