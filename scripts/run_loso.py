"""
Run leave-one-scene-out (LOSO) threshold tuning + evaluation over cached tracks.

For each held-out scene, thresholds are grid-search fit on the other scenes' clips and then
scored on the held-out clips, so tuning never sees the scene it is judged on. Prints a
per-scene precision / recall / F1 report plus the micro-averaged overall score.

    python -m scripts.run_loso                 # default grid, cached tracks
    python -m scripts.run_loso --tracks-dir cache/tracks
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
from pathlib import Path

from person_vehicle_interactions.candidate_detection import (
    detect_candidates,
    Thresholds,
)
from person_vehicle_interactions.config import TRACKS_DIR, vehicle_class_names_for_clip
from person_vehicle_interactions.evaluation import EvalResult, evaluate_clips
from person_vehicle_interactions.gt_windows import InteractionWindow, load_gt_windows
from person_vehicle_interactions.loso import all_clips, run_loso
from person_vehicle_interactions.threshold_fitting import fit_thresholds, threshold_grid
from person_vehicle_interactions.tracked_data_model import load_tracks, TrackedBox

GT_CSV = Path("ground_truth/interactions.csv")

# Default threshold search grid. Overlap is normalized intersection (fraction of the person
# inside the vehicle); distance is center distance / vehicle diagonal; durations/gaps are in
# frames. Kept modest so the pure grid search stays fast.
DEFAULT_GRID: list[Thresholds] = threshold_grid(
    min_overlaps=[0.0, 0.05, 0.1, 0.2],
    max_distances=[0.0, 0.5, 1.0, 1.5],
    min_durations=[3, 5, 10],
    min_confidences=[0.0, 0.3],
    max_gap_frames=[0, 5, 15],
)


@dataclasses.dataclass(frozen=True)
class FoldReport:
    """A held-out scene's fitted thresholds and their score on that scene's clips."""

    thresholds: Thresholds
    result: EvalResult


def load_boxes_by_clip(
    clip_ids: tuple[str, ...], tracks_dir: Path
) -> dict[str, list[TrackedBox]]:
    """Load each clip's cached (per-clip vehicle-class filtered) tracks."""
    return {clip_id: load_tracks(tracks_dir / f"{clip_id}.csv") for clip_id in clip_ids}


def make_fit_thresholds(
    boxes_by_clip: dict[str, list[TrackedBox]],
    ground_truth_by_clip: dict[str, list[InteractionWindow]],
    grid: list[Thresholds],
) -> Callable[[tuple[str, ...]], Thresholds]:
    """Build the LOSO ``fit_thresholds`` callable bound to the cached data and grid."""

    def fit(train_clips: tuple[str, ...]) -> Thresholds:
        return fit_thresholds(
            {clip_id: boxes_by_clip[clip_id] for clip_id in train_clips},
            {clip_id: vehicle_class_names_for_clip(clip_id) for clip_id in train_clips},
            {clip_id: ground_truth_by_clip.get(clip_id, []) for clip_id in train_clips},
            grid,
        ).thresholds

    return fit


def make_evaluate(
    boxes_by_clip: dict[str, list[TrackedBox]],
    ground_truth_by_clip: dict[str, list[InteractionWindow]],
) -> Callable[[tuple[str, ...], Thresholds], FoldReport]:
    """Build the LOSO ``evaluate`` callable scoring held-out clips with given thresholds."""

    def evaluate(test_clips: tuple[str, ...], thresholds: Thresholds) -> FoldReport:
        predicted_by_clip = {
            clip_id: detect_candidates(
                boxes_by_clip[clip_id],
                vehicle_class_names_for_clip(clip_id),
                thresholds,
            )
            for clip_id in test_clips
        }
        ground_truth = {
            clip_id: ground_truth_by_clip.get(clip_id, []) for clip_id in test_clips
        }
        return FoldReport(thresholds, evaluate_clips(predicted_by_clip, ground_truth))

    return evaluate


def print_report(reports: dict[str, FoldReport]) -> None:
    """Print per-scene scores plus the micro-averaged overall LOSO score."""
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    print(f"{'scene':<16} {'P':>5} {'R':>5} {'F1':>5}  {'TP':>3} {'FP':>3} {'FN':>3}")
    for scene, report in reports.items():
        result = report.result
        total_true_positives += result.true_positives
        total_false_positives += result.false_positives
        total_false_negatives += result.false_negatives
        print(
            f"{scene:<16} {result.precision:>5.2f} {result.recall:>5.2f} "
            f"{result.f1:>5.2f}  {result.true_positives:>3} "
            f"{result.false_positives:>3} {result.false_negatives:>3}"
        )
    predicted_total = total_true_positives + total_false_positives
    actual_total = total_true_positives + total_false_negatives
    precision = total_true_positives / predicted_total if predicted_total else 0.0
    recall = total_true_positives / actual_total if actual_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(
        f"{'OVERALL':<16} {precision:>5.2f} {recall:>5.2f} {f1:>5.2f}  "
        f"{total_true_positives:>3} {total_false_positives:>3} "
        f"{total_false_negatives:>3}"
    )
    print("\nFitted thresholds per held-out scene:")
    for scene, report in reports.items():
        print(f"  {scene:<16} {report.thresholds}")


def main() -> None:
    """Run LOSO tuning + evaluation over the cached tracks and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-dir", type=Path, default=TRACKS_DIR)
    parser.add_argument("--gt-csv", type=Path, default=GT_CSV)
    args = parser.parse_args()

    ground_truth_by_clip = load_gt_windows(args.gt_csv)
    boxes_by_clip = load_boxes_by_clip(all_clips(), args.tracks_dir)
    reports = run_loso(
        make_fit_thresholds(boxes_by_clip, ground_truth_by_clip, DEFAULT_GRID),
        make_evaluate(boxes_by_clip, ground_truth_by_clip),
    )
    print_report(reports)


if __name__ == "__main__":
    main()
