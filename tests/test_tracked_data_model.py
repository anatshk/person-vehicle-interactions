"""Tests for the tracked-box data model and its CSV/JSON cache round-trips."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.tracked_data_model import (
    ClipMetadata,
    frame_to_seconds,
    load_metadata,
    load_tracks,
    save_metadata,
    save_tracks,
    TRACK_CSV_COLUMNS,
)
from tests.factories import make_tracked_box


def make_clip_metadata() -> ClipMetadata:
    """Build a representative ClipMetadata for tests."""
    return ClipMetadata(
        clip_id="sample_clip",
        fps=30.0,
        frame_width=1280,
        frame_height=720,
        frame_count=495,
        model_name="yolo11l.pt",
        image_size=1280,
        confidence_threshold=0.25,
        iou_threshold=0.7,
        tracker_name="botsort.yaml",
        seed=0,
    )


def test_tracked_box_round_trip_csv(tmp_path):
    boxes = [
        make_tracked_box(frame=0, track_id=1, object_class="person"),
        make_tracked_box(frame=15, track_id=2, object_class="car"),
    ]
    csv_path = tmp_path / "sample_clip.csv"
    save_tracks(boxes, csv_path)
    assert load_tracks(csv_path) == boxes


def test_csv_has_expected_header(tmp_path):
    csv_path = tmp_path / "sample_clip.csv"
    save_tracks([make_tracked_box()], csv_path)
    header_line = csv_path.read_text().splitlines()[0]
    assert header_line == ",".join(TRACK_CSV_COLUMNS)


def test_load_restores_dtypes(tmp_path):
    csv_path = tmp_path / "sample_clip.csv"
    save_tracks([make_tracked_box(frame=5, track_id=3)], csv_path)
    box = load_tracks(csv_path)[0]
    assert isinstance(box.frame, int)
    assert isinstance(box.track_id, int)
    assert isinstance(box.object_class, str)
    for float_field in (
        box.time_seconds,
        box.x1,
        box.y1,
        box.x2,
        box.y2,
        box.confidence,
    ):
        assert isinstance(float_field, float)


def test_empty_tracks_round_trip(tmp_path):
    csv_path = tmp_path / "empty.csv"
    save_tracks([], csv_path)
    header_line = csv_path.read_text().splitlines()[0]
    assert header_line == ",".join(TRACK_CSV_COLUMNS)
    assert load_tracks(csv_path) == []


def test_metadata_round_trip_json(tmp_path):
    metadata = make_clip_metadata()
    json_path = tmp_path / "sample_clip.meta.json"
    save_metadata(metadata, json_path)
    assert load_metadata(json_path) == metadata


def test_rows_sorted_deterministically(tmp_path):
    unsorted_boxes = [
        make_tracked_box(frame=2, track_id=1),
        make_tracked_box(frame=1, track_id=2),
        make_tracked_box(frame=1, track_id=1),
    ]
    csv_path = tmp_path / "sample_clip.csv"
    save_tracks(unsorted_boxes, csv_path)
    loaded = load_tracks(csv_path)
    assert [(box.frame, box.track_id) for box in loaded] == [(1, 1), (1, 2), (2, 1)]


def test_multiple_tracks_and_frames(tmp_path):
    boxes = []
    for frame in range(3):
        boxes.append(make_tracked_box(frame=frame, track_id=1, object_class="person"))
        boxes.append(make_tracked_box(frame=frame, track_id=2, object_class="person"))
        boxes.append(make_tracked_box(frame=frame, track_id=3, object_class="car"))
    csv_path = tmp_path / "sample_clip.csv"
    save_tracks(boxes, csv_path)
    loaded = load_tracks(csv_path)
    assert len(loaded) == 9
    assert sum(1 for box in loaded if box.object_class == "person") == 6
    assert sum(1 for box in loaded if box.object_class == "car") == 3
    assert {box.track_id for box in loaded} == {1, 2, 3}


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tracks(tmp_path / "does_not_exist.csv")


def test_frame_to_seconds_helper():
    assert frame_to_seconds(0, 30.0) == 0.0
    assert frame_to_seconds(30, 30.0) == pytest.approx(1.0)
    assert frame_to_seconds(126, 30.0) == pytest.approx(4.2)


def test_rejects_invalid_box():
    with pytest.raises(ValueError):
        make_tracked_box(x1=30.0, x2=10.0)  # x2 < x1
    with pytest.raises(ValueError):
        make_tracked_box(y1=40.0, y2=20.0)  # y2 < y1
    with pytest.raises(ValueError):
        make_tracked_box(x1=-1.0)  # negative coordinate


def test_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        make_tracked_box(confidence=1.5)
    with pytest.raises(ValueError):
        make_tracked_box(confidence=-0.1)
