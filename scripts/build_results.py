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

from collections.abc import Callable
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
from person_vehicle_interactions.describers import Describer, KIND_PERSON, KIND_VEHICLE
from person_vehicle_interactions.descriptions import placeholder_description
from person_vehicle_interactions.frame_crops import crop_for_box
from person_vehicle_interactions.interaction_records import (
    build_clip_records,
    DescribeWindow,
    write_clip_records,
)
from person_vehicle_interactions.interaction_signals import PERSON_CLASS
from person_vehicle_interactions.track_selection import highest_confidence_box_per_track
from person_vehicle_interactions.tracked_data_model import (
    load_metadata,
    load_tracks,
    PathLike,
    TrackedBox,
)

# Describe one track's representative box -> its description string.
DescribeBox = Callable[[TrackedBox], str]


def build_describe_window(
    boxes: list[TrackedBox], describe_box: DescribeBox
) -> DescribeWindow:
    """
    Build a ``describe_window`` callback from a per-box describer, cached per ``track_id``.
    Each track is described from its highest-confidence box, and the description is computed
    once and reused (dedup) — a track that recurs across windows is only described a single
    time, which matters when the describer runs a slow model.
    """
    box_by_track = {
        box.track_id: box for box in highest_confidence_box_per_track(boxes)
    }
    described: dict[int, str] = {}

    def describe_track(track_id: int) -> str:
        if track_id not in described:
            described[track_id] = describe_box(box_by_track[track_id])
        return described[track_id]

    def describe_window(window: PredictedWindow) -> tuple[str, str]:
        return describe_track(window.person_id), describe_track(window.vehicle_id)

    return describe_window


def make_placeholder_describe_window(boxes: list[TrackedBox]) -> DescribeWindow:
    """
    Build a model-free ``describe_window`` that labels each track by its class and id, e.g.
    ``("person (track 1)", "car (track 2)")`` — the stand-in when no captioner is selected.
    """
    return build_describe_window(boxes, placeholder_description)


def make_describe_box(
    video_path: PathLike, describer: Describer, pad_fraction: float = 0.1
) -> DescribeBox:
    """
    Build a per-box describer that crops the box from ``video_path`` and captions the crop.
    The crop's ``kind`` (person vs vehicle) is taken from the box class so the describer can
    pick the right prompt / vocabulary. Reads video pixels, so it lives in the glue layer.
    """

    def describe_box(box: TrackedBox) -> str:
        crop = crop_for_box(video_path, box, pad_fraction)
        kind = KIND_PERSON if box.object_class == PERSON_CLASS else KIND_VEHICLE
        return describer(crop, kind)

    return describe_box


def build_clip_results(
    clip_id: str,
    thresholds: Thresholds,
    tracks_dir: PathLike = TRACKS_DIR,
    results_dir: PathLike = RESULTS_DIR,
    generated_at: datetime.datetime | None = None,
    describe_box: DescribeBox | None = None,
) -> Path:
    """
    Detect one clip's interactions and write its per-clip results JSON; return the path.
    ``generated_at`` defaults to now; pass an explicit value for deterministic output.
    ``describe_box`` supplies real descriptions (crop + caption); when ``None`` the model-free
    placeholder is used, so this stays runnable without a video or model.
    """
    boxes = load_tracks(Path(tracks_dir) / f"{clip_id}.csv")
    metadata = load_metadata(Path(tracks_dir) / f"{clip_id}.meta.json")
    vehicle_classes = vehicle_class_names_for_clip(clip_id)
    windows = detect_candidates(boxes, vehicle_classes, thresholds)
    describe_window = (
        build_describe_window(boxes, describe_box)
        if describe_box is not None
        else make_placeholder_describe_window(boxes)
    )
    records = build_clip_records(clip_id, windows, metadata.fps, describe_window)
    return write_clip_records(records, clip_id, results_dir, generated_at)
