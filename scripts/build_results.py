"""
Classify cached tracks into machine-readable interaction records (the classify stage).

For one clip: load its cached tracks + metadata, detect candidate interaction windows with
a threshold set, describe the person and vehicle of each window with the placeholder
captioner, and write a timestamped per-clip results JSON via
``interaction_records.write_clip_records``.

This is a library used by the deliverable CLI (``scripts.detect_interactions``); it holds
no CLI of its own and is unaware of the LOSO research. The shipped threshold set lives in
``config.SHIPPED_THRESHOLDS``.
"""

from __future__ import annotations

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
    write_clip_records,
)
from person_vehicle_interactions.track_selection import highest_confidence_box_per_track
from person_vehicle_interactions.tracked_data_model import (
    load_metadata,
    load_tracks,
    TrackedBox,
)

PathLike = Path | str


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
