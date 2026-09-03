"""
One-off experiment: fit the *newly added* threshold combinations on all clips and compare
the best-of-new against the shipped (previously optimal) thresholds.

The original shipped fit sat at an edge of the grid on every axis, so this widens the top
ends of overlap / duration / confidence / gap and runs ONLY the combinations that were not
in the original grid (expanded grid minus original grid). It reports the best-of-new
thresholds and F1 alongside the shipped baseline, both fit/scored on all clips.

    python -m scripts.widen_grid_experiment
"""

from __future__ import annotations

from pathlib import Path

from person_vehicle_interactions.candidate_detection import (
    detect_candidates,
    Thresholds,
)
from person_vehicle_interactions.config import (
    SHIPPED_THRESHOLDS,
    vehicle_class_names_for_clip,
)
from person_vehicle_interactions.evaluation import evaluate_clips
from person_vehicle_interactions.gt_windows import load_gt_windows
from person_vehicle_interactions.loso import all_clips
from person_vehicle_interactions.threshold_fitting import fit_thresholds, threshold_grid
from scripts.run_loso import GT_CSV, load_boxes_by_clip

# Original grid (matches scripts.run_loso.DEFAULT_GRID at experiment time).
ORIGINAL_GRID = threshold_grid(
    min_overlaps=[0.0, 0.05, 0.1, 0.2],
    max_distances=[0.0, 0.5, 1.0, 1.5],
    min_durations=[3, 5, 10],
    min_confidences=[0.0, 0.3],
    max_gap_frames=[0, 5, 15],
)

# Widened grid: top ends extended per the request.
EXPANDED_GRID = threshold_grid(
    min_overlaps=[0.0, 0.05, 0.1, 0.2, 0.4, 0.5],
    max_distances=[0.0, 0.5, 1.0, 1.5],
    min_durations=[3, 5, 10, 15, 20],
    min_confidences=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7],
    max_gap_frames=[0, 5, 15, 20, 25, 30],
)


def new_combinations() -> list[Thresholds]:
    """Return the expanded-grid combinations that were not in the original grid."""
    original = set(ORIGINAL_GRID)
    return [thresholds for thresholds in EXPANDED_GRID if thresholds not in original]


def main() -> None:
    """Fit the new combinations on all clips and compare against the shipped baseline."""
    tracks_dir = Path("cache/tracks")
    ground_truth_by_clip = load_gt_windows(GT_CSV)
    clips = all_clips()
    boxes_by_clip = load_boxes_by_clip(clips, tracks_dir)
    vehicle_classes_by_clip = {
        clip_id: vehicle_class_names_for_clip(clip_id) for clip_id in clips
    }
    gt_by_clip = {clip_id: ground_truth_by_clip.get(clip_id, []) for clip_id in clips}

    grid = new_combinations()
    print(f"original grid size: {len(ORIGINAL_GRID)}")
    print(f"expanded grid size: {len(EXPANDED_GRID)}")
    print(f"new combinations to run: {len(grid)}\n")

    # Best of the new combinations, fit on all clips.
    best_new = fit_thresholds(boxes_by_clip, vehicle_classes_by_clip, gt_by_clip, grid)

    # Shipped baseline, scored on all clips the same way.
    shipped_predicted = {
        clip_id: detect_candidates(
            boxes_by_clip[clip_id], vehicle_classes_by_clip[clip_id], SHIPPED_THRESHOLDS
        )
        for clip_id in clips
    }
    shipped_result = evaluate_clips(shipped_predicted, gt_by_clip)

    def line(label: str, thresholds: Thresholds, result) -> str:
        return (
            f"{label:<14} F1={result.f1:.4f}  P={result.precision:.4f} R={result.recall:.4f}"
            f"  TP={result.true_positives} FP={result.false_positives} FN={result.false_negatives}\n"
            f"               {thresholds}"
        )

    print(line("SHIPPED", SHIPPED_THRESHOLDS, shipped_result))
    print()
    print(line("BEST-OF-NEW", best_new.thresholds, best_new.train_result))
    print()
    delta = best_new.train_result.f1 - shipped_result.f1
    verdict = (
        "new combination BEATS shipped"
        if delta > 1e-9
        else (
            "new combination ties shipped"
            if abs(delta) <= 1e-9
            else "shipped still best (no new combination beats it)"
        )
    )
    print(f"delta F1 (new - shipped): {delta:+.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
