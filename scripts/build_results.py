"""
Generate the machine-readable interaction results JSON for the cached clips.

For each clip: load its cached tracks + metadata, detect candidate interaction windows
with the shipped (fit-on-all) threshold set, describe the person and vehicle of each
window with the placeholder captioner, and write a per-clip timestamped results JSON via
``interaction_records.write_clip_records``.

    python -m scripts.build_results                 # all clips, shipped single set
    python -m scripts.build_results --clip gt1125_06
    python -m scripts.build_results --loso          # per-clip LOSO fold thresholds

The shipped deliverable uses ONE fixed threshold set (``SHIPPED_THRESHOLDS``, the default),
so the run is reproducible and works on clips without ground truth. The ``--loso`` fold
thresholds are for honest per-scene evaluation only (leakage-free); see the write-up.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from person_vehicle_interactions.candidate_detection import (
    detect_candidates,
    PredictedWindow,
    Thresholds,
)
from person_vehicle_interactions.config import (
    RESULTS_DIR,
    TRACKS_DIR,
    vehicle_class_names_for_clip,
)
from person_vehicle_interactions.descriptions import placeholder_description
from person_vehicle_interactions.interaction_records import (
    build_clip_records,
    DescribeWindow,
    load_clip_records,
    write_clip_records,
)
from person_vehicle_interactions.loso import all_clips
from person_vehicle_interactions.track_selection import highest_confidence_box_per_track
from person_vehicle_interactions.tracked_data_model import (
    load_metadata,
    load_tracks,
    TrackedBox,
)

PathLike = Path | str

# The single fixed threshold set for the shipped deliverable: fit on ALL clips (not per
# LOSO fold), so the run is reproducible and applies to clips without ground truth. Kept
# in sync with ``fit(all_clips())`` by ``test_shipped_thresholds_match_fit_on_all``.
SHIPPED_THRESHOLDS = Thresholds(
    min_overlap=0.2,
    max_distance=0.0,
    min_duration_frames=10,
    min_confidence=0.3,
    max_gap_frames=15,
)


def make_placeholder_describe_window(boxes: list[TrackedBox]) -> DescribeWindow:
    """
    Build a ``describe_window`` callback that labels each track from its representative box.
    Each track's highest-confidence box supplies the placeholder description, so a window's
    (person, vehicle) descriptions are e.g. ``("person (track 1)", "car (track 2)")``. This
    is the stand-in until a captioner is wired into ``descriptions.describe``.
    """
    box_by_track = {
        box.track_id: box for box in highest_confidence_box_per_track(boxes)
    }

    def describe_window(window: PredictedWindow) -> tuple[str, str]:
        return (
            placeholder_description(box_by_track[window.person_id]),
            placeholder_description(box_by_track[window.vehicle_id]),
        )

    return describe_window


def build_clip_results(
    clip_id: str,
    thresholds: Thresholds,
    tracks_dir: PathLike = TRACKS_DIR,
    results_dir: PathLike = RESULTS_DIR,
    generated_at: datetime.datetime | None = None,
) -> Path:
    """
    Detect one clip's interactions and write its per-clip results JSON; return the path.
    ``generated_at`` defaults to now; pass an explicit value for deterministic output.
    """
    boxes = load_tracks(Path(tracks_dir) / f"{clip_id}.csv")
    metadata = load_metadata(Path(tracks_dir) / f"{clip_id}.meta.json")
    vehicle_classes = vehicle_class_names_for_clip(clip_id)
    windows = detect_candidates(boxes, vehicle_classes, thresholds)
    records = build_clip_records(
        clip_id, windows, metadata.fps, make_placeholder_describe_window(boxes)
    )
    return write_clip_records(records, clip_id, results_dir, generated_at)


def main() -> None:
    """Generate the results JSON for one clip or all clips (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default=None, help="Generate only this clip id.")
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Use per-clip LOSO fold thresholds instead of the shipped single set.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    loso_thresholds = None
    if args.loso:
        from scripts.show_interactions import _thresholds_by_clip

        loso_thresholds = _thresholds_by_clip(fit_on_all=False)
    clip_ids = (args.clip,) if args.clip else all_clips()
    for clip_id in clip_ids:
        thresholds = loso_thresholds[clip_id] if loso_thresholds else SHIPPED_THRESHOLDS
        path = build_clip_results(clip_id, thresholds, results_dir=args.results_dir)
        count = len(load_clip_records(path))
        print(f"{clip_id}: {count} interaction(s) -> {path}")


if __name__ == "__main__":
    main()
