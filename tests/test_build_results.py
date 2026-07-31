"""Tests for the results-JSON generation script (describe hook + per-clip glue)."""

from __future__ import annotations

import datetime

import pytest

from person_vehicle_interactions.config import SHIPPED_THRESHOLDS, TRACKS_DIR
from person_vehicle_interactions.interaction_records import load_clip_records
from person_vehicle_interactions.loso import all_clips
from person_vehicle_interactions.tracked_data_model import (
    ClipMetadata,
    save_metadata,
    save_tracks,
)
from scripts.build_results import build_clip_results, make_placeholder_describe_window
from tests.factories import make_interaction_prediction, make_person_in_vehicle_boxes

CLIP_ID = "results_clip"


def _metadata(fps: float = 30.0, frame_count: int = 20) -> ClipMetadata:
    """A minimal ClipMetadata for a synthetic cached clip."""
    return ClipMetadata(
        clip_id=CLIP_ID,
        fps=fps,
        frame_width=100,
        frame_height=100,
        frame_count=frame_count,
        model_name="yolo11l.pt",
        image_size=1280,
        confidence_threshold=0.25,
        iou_threshold=0.7,
        tracker_name="botsort.yaml",
        seed=0,
    )


def test_describe_window_labels_person_and_vehicle_by_track():
    boxes = make_person_in_vehicle_boxes(range(0, 5), person_id=1, vehicle_id=2)
    describe_window = make_placeholder_describe_window(boxes)
    window = make_interaction_prediction(0, 4, person_id=1, vehicle_id=2)
    assert describe_window(window) == ("person (track 1)", "car (track 2)")


def test_build_clip_results_writes_reloadable_json(tmp_path):
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    boxes = make_person_in_vehicle_boxes(range(0, 15), person_id=1, vehicle_id=2)
    save_tracks(boxes, tracks_dir / f"{CLIP_ID}.csv")
    save_metadata(_metadata(), tracks_dir / f"{CLIP_ID}.meta.json")

    path = build_clip_results(
        CLIP_ID,
        SHIPPED_THRESHOLDS,
        tracks_dir=tracks_dir,
        results_dir=tmp_path / "results",
        generated_at=datetime.datetime(2026, 7, 31, 12, 0),
    )

    records = load_clip_records(path)
    assert len(records) == 1
    record = records[0]
    assert record.clip_id == CLIP_ID
    assert (record.person_id, record.vehicle_id) == (1, 2)
    assert (record.start_frame, record.end_frame) == (0, 14)
    assert record.person == "person (track 1)"
    assert record.vehicle == "car (track 2)"


@pytest.mark.integration
def test_shipped_thresholds_match_fit_on_all():
    """SHIPPED_THRESHOLDS must equal fit(all_clips()); skip without the real cache."""
    if not (TRACKS_DIR / f"{all_clips()[0]}.csv").exists():
        pytest.skip("real tracks cache not present")
    from scripts.show_interactions import _thresholds_by_clip

    fitted = _thresholds_by_clip(fit_on_all=True)[all_clips()[0]]
    assert SHIPPED_THRESHOLDS == fitted
