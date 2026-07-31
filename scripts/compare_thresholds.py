"""
Compare the shipped fit-on-all thresholds against the LOSO fold thresholds.

Two views over the same cached tracks:

* **Threshold values** — how each held-out scene's LOSO fold thresholds differ, field by
  field, from the single shipped set (``config.SHIPPED_THRESHOLDS``).
* **Performance** — per-scene and overall precision / recall / F1 under the **LOSO fold**
  thresholds (leakage-free, the honest generalization estimate) versus the **shipped**
  single set applied to every clip (in-sample, what the deliverable actually emits).

The gap between the two performance columns is the price of shipping one fixed set instead
of per-scene tuning; the LOSO column stays the number to quote for generalization.

    python -m scripts.compare_thresholds
    python -m scripts.compare_thresholds --tracks-dir cache/tracks
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import dataclasses
from pathlib import Path

from person_vehicle_interactions.candidate_detection import Thresholds
from person_vehicle_interactions.config import SHIPPED_THRESHOLDS, TRACKS_DIR
from person_vehicle_interactions.evaluation import EvalResult
from person_vehicle_interactions.gt_windows import load_gt_windows
from person_vehicle_interactions.loso import all_clips, run_loso, SCENE_UNITS
from scripts.run_loso import (
    DEFAULT_GRID,
    FoldReport,
    GT_CSV,
    load_boxes_by_clip,
    make_evaluate,
    make_fit_thresholds,
)

FieldDiffs = dict[str, tuple[object, object]]


@dataclasses.dataclass(frozen=True)
class SceneComparison:
    """One scene's fold thresholds and its LOSO-vs-shipped scores + threshold diffs."""

    scene: str
    fold_thresholds: Thresholds
    loso_result: EvalResult
    single_result: EvalResult
    threshold_diffs: FieldDiffs


def threshold_field_diffs(fold: Thresholds, single: Thresholds) -> FieldDiffs:
    """
    Map each threshold field where ``fold`` differs from ``single`` to ``(fold, single)``.
    Fields that match are omitted, so an empty result means the two sets are identical.
    """
    return {
        field.name: (getattr(fold, field.name), getattr(single, field.name))
        for field in dataclasses.fields(Thresholds)
        if getattr(fold, field.name) != getattr(single, field.name)
    }


def overall_result(results: Iterable[EvalResult]) -> EvalResult:
    """
    Micro-average a set of per-scene results: sum the counts, then re-derive P / R / F1.
    """
    true_positives = sum(result.true_positives for result in results)
    false_positives = sum(result.false_positives for result in results)
    false_negatives = sum(result.false_negatives for result in results)
    predicted_total = true_positives + false_positives
    actual_total = true_positives + false_negatives
    precision = true_positives / predicted_total if predicted_total else 0.0
    recall = true_positives / actual_total if actual_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvalResult(
        true_positives, false_positives, false_negatives, precision, recall, f1
    )


def build_comparison(
    loso_reports: dict[str, FoldReport],
    single_reports: dict[str, FoldReport],
    single_thresholds: Thresholds,
) -> list[SceneComparison]:
    """
    Pair each scene's LOSO fold report with its shipped-thresholds report.
    Scenes follow ``loso_reports`` order; each carries the fold thresholds' differences
    from ``single_thresholds``.
    """
    return [
        SceneComparison(
            scene=scene,
            fold_thresholds=loso_report.thresholds,
            loso_result=loso_report.result,
            single_result=single_reports[scene].result,
            threshold_diffs=threshold_field_diffs(
                loso_report.thresholds, single_thresholds
            ),
        )
        for scene, loso_report in loso_reports.items()
    ]


def _score_row(name: str, result: EvalResult) -> str:
    """A fixed-width P/R/F1 + TP/FP/FN cell for one threshold source."""
    return (
        f"{name:<7} {result.precision:>4.2f} {result.recall:>4.2f} "
        f"{result.f1:>4.2f}  {result.true_positives:>2} "
        f"{result.false_positives:>2} {result.false_negatives:>2}"
    )


def _diffs_caption(diffs: FieldDiffs) -> str:
    """Render a fold's differing threshold fields as ``field fold->shipped`` fragments."""
    if not diffs:
        return "(identical to shipped)"
    return "  ".join(
        f"{field} {fold}->{single}" for field, (fold, single) in diffs.items()
    )


def print_comparison(
    comparisons: list[SceneComparison], single_thresholds: Thresholds
) -> None:
    """Print the threshold-diff and LOSO-vs-shipped performance report."""
    print(f"shipped (fit-on-all): {single_thresholds}\n")
    print("Fold thresholds per held-out scene (differences from shipped):")
    for comparison in comparisons:
        print(f"  {comparison.scene:<16} {_diffs_caption(comparison.threshold_diffs)}")

    print("\nPerformance — LOSO (leakage-free) vs shipped fit-on-all (in-sample):")
    header = "P    R   F1   TP FP FN"
    print(f"{'scene':<16} LOSO: {header}     |  shipped: {header}")
    for comparison in comparisons:
        print(
            f"{comparison.scene:<16} {_score_row('LOSO', comparison.loso_result)}"
            f"     |  {_score_row('shipped', comparison.single_result)}"
        )
    loso_overall = overall_result([c.loso_result for c in comparisons])
    single_overall = overall_result([c.single_result for c in comparisons])
    print(
        f"{'OVERALL':<16} {_score_row('LOSO', loso_overall)}"
        f"     |  {_score_row('shipped', single_overall)}"
    )


def main() -> None:
    """Build and print the shipped-vs-LOSO threshold comparison over cached tracks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-dir", type=Path, default=None)
    parser.add_argument("--gt-csv", type=Path, default=GT_CSV)
    args = parser.parse_args()

    tracks_dir = args.tracks_dir or TRACKS_DIR
    ground_truth_by_clip = load_gt_windows(args.gt_csv)
    boxes_by_clip = load_boxes_by_clip(all_clips(), tracks_dir)
    fit = make_fit_thresholds(boxes_by_clip, ground_truth_by_clip, DEFAULT_GRID)
    evaluate = make_evaluate(boxes_by_clip, ground_truth_by_clip)

    loso_reports = run_loso(fit, evaluate)
    single_reports = {
        scene: evaluate(scene_clips, SHIPPED_THRESHOLDS)
        for scene, scene_clips in SCENE_UNITS.items()
    }
    comparisons = build_comparison(loso_reports, single_reports, SHIPPED_THRESHOLDS)
    print_comparison(comparisons, SHIPPED_THRESHOLDS)


if __name__ == "__main__":
    main()
