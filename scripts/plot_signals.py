"""
Generate per-pair interaction-signal plots for tuning, with GT windows shaded.

For each clip's cached tracks, finds candidate (person, vehicle) pairs and plots each
pair's overlap / distance / confidence vs frame into ``cache/viz/signals/<clip>/``.

    python -m scripts.plot_signals                 # all clips
    python -m scripts.plot_signals --clip mKzCQKTHizw_0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from person_vehicle_interactions.config import TRACKS_DIR, VIZ_DIR
from person_vehicle_interactions.gt_windows import load_gt_windows
from person_vehicle_interactions.interaction_signals import (
    candidate_pairs,
    pair_signal_series,
)
from person_vehicle_interactions.signal_plots import plot_pair_signals
from person_vehicle_interactions.tracked_data_model import load_tracks

GT_CSV = Path("ground_truth/interactions.csv")
VEHICLE_CLASSES = {"car", "bus", "truck", "boat"}


def plot_clip(
    clip_id: str,
    max_distance: float,
    min_overlap: float,
    tracks_dir: Path,
    viz_dir: Path,
    gt_windows_by_clip: dict,
) -> int:
    """Plot every candidate pair for one clip; return the number of plots written."""
    boxes = load_tracks(tracks_dir / f"{clip_id}.csv")
    gt_windows = gt_windows_by_clip.get(clip_id, [])
    pairs = candidate_pairs(boxes, VEHICLE_CLASSES, max_distance, min_overlap)
    out_dir = viz_dir / "signals" / clip_id
    for person_id, vehicle_id in pairs:
        series = pair_signal_series(boxes, person_id, vehicle_id)
        title = f"{clip_id}  person {person_id} x vehicle {vehicle_id}"
        plot_pair_signals(
            series, gt_windows, title, out_dir / f"p{person_id}_v{vehicle_id}.png"
        )
    print(f"{clip_id}: {len(pairs)} candidate pair(s) -> {out_dir}", flush=True)
    return len(pairs)


def main() -> None:
    """Generate signal plots for one clip or all cached clips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", help="Clip id (default: all cached clips).")
    parser.add_argument("--max-distance", type=float, default=1.0)
    parser.add_argument("--min-overlap", type=float, default=0.05)
    parser.add_argument("--tracks-dir", type=Path, default=TRACKS_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    args = parser.parse_args()

    gt_windows_by_clip = load_gt_windows(GT_CSV)
    if args.clip:
        clip_ids = [args.clip]
    else:
        clip_ids = sorted(
            path.stem
            for path in args.tracks_dir.glob("*.csv")
            if "_allclasses" not in path.stem
        )

    for clip_id in clip_ids:
        plot_clip(
            clip_id,
            args.max_distance,
            args.min_overlap,
            args.tracks_dir,
            args.viz_dir,
            gt_windows_by_clip,
        )


if __name__ == "__main__":
    main()
