"""Tests for the raw-detections JSONL cache and tracks_from_raw."""

from __future__ import annotations

from pathlib import Path

import pytest

from person_vehicle_interactions.config import DetectionConfig
from person_vehicle_interactions.raw_detections import (
    read_raw,
    tracks_from_raw,
    write_raw_frame,
)

# One frame = (frame_index, boxes_xyxy, class_ids, track_ids, confidences).
RawFrameInput = tuple[int, list[list[float]], list[int], list[int | None], list[float]]


def write_frames(path: Path, frames: list[RawFrameInput]) -> None:
    """Write frames to a raw JSONL file using write_raw_frame."""
    with path.open("w", encoding="utf-8") as raw_file:
        for frame_index, boxes_xyxy, class_ids, track_ids, confidences in frames:
            write_raw_frame(
                raw_file, frame_index, boxes_xyxy, class_ids, track_ids, confidences
            )


def test_write_read_raw_round_trip(tmp_path):
    path = tmp_path / "clip.jsonl"
    write_frames(
        path,
        [
            (
                0,
                [[10.0, 20.0, 30.0, 40.0]],
                [0],
                [1],
                [0.9],
            ),
            (
                1,
                [[0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 2.0, 2.0]],
                [2, 0],
                [2, 3],
                [0.8, 0.7],
            ),
        ],
    )
    records = list(read_raw(path))
    assert len(records) == 2
    assert records[0].frame == 0
    assert records[1].class_ids == [2, 0]
    assert records[1].boxes_xyxy == [[0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 2.0, 2.0]]
    assert records[1].track_ids == [2, 3]
    assert records[1].confidences == [0.8, 0.7]


def test_tracks_from_raw_builds_boxes(tmp_path):
    path = tmp_path / "clip.jsonl"
    write_frames(
        path,
        [
            (0, [[10.0, 20.0, 30.0, 40.0]], [0], [1], [0.9]),
            (1, [[0.0, 0.0, 5.0, 5.0]], [2], [2], [0.8]),
        ],
    )
    boxes = tracks_from_raw(path, fps=30.0, config=DetectionConfig())
    assert len(boxes) == 2
    assert boxes[0].object_class == "person"
    assert boxes[0].track_id == 1
    assert boxes[1].object_class == "car"
    assert boxes[1].time_seconds == pytest.approx(1 / 30.0)


def test_tracks_from_raw_skips_missing_track_id(tmp_path):
    path = tmp_path / "clip.jsonl"
    write_frames(
        path,
        [
            (
                0,
                [[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 5.0, 5.0]],
                [0, 2],
                [None, 5],
                [0.9, 0.8],
            ),
        ],
    )
    boxes = tracks_from_raw(path, fps=30.0, config=DetectionConfig())
    assert len(boxes) == 1
    assert boxes[0].track_id == 5


def test_tracks_from_raw_filters_classes(tmp_path):
    path = tmp_path / "clip.jsonl"
    write_frames(
        path,
        [
            (
                0,
                [[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 5.0, 5.0]],
                [0, 2],
                [1, 2],
                [0.9, 0.8],
            ),
        ],
    )
    config = DetectionConfig(target_class_ids=(0,))  # person only
    boxes = tracks_from_raw(path, fps=30.0, config=config)
    assert [box.object_class for box in boxes] == ["person"]


def test_tracks_from_raw_empty(tmp_path):
    path = tmp_path / "clip.jsonl"
    path.write_text("", encoding="utf-8")
    assert tracks_from_raw(path, fps=30.0, config=DetectionConfig()) == []
