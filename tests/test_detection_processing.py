"""Tests for the pure detection-processing helpers (no cv2/ultralytics)."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.detection_processing import (
    build_tracked_boxes,
    COCO_ID_TO_NAME,
    filter_detections_by_class,
)
from tests.factories import make_tracked_box


def test_build_tracked_boxes_basic():
    boxes = build_tracked_boxes(
        frame_index=5,
        fps=30.0,
        boxes_xyxy=[(10.0, 20.0, 30.0, 40.0)],
        class_ids=[0],
        track_ids=[7],
        confidences=[0.88],
    )
    assert len(boxes) == 1
    box = boxes[0]
    assert box.frame == 5
    assert box.time_seconds == pytest.approx(5 / 30.0)
    assert box.track_id == 7
    assert box.object_class == "person"
    assert (box.x1, box.y1, box.x2, box.y2) == (10.0, 20.0, 30.0, 40.0)
    assert box.confidence == pytest.approx(0.88)


def test_build_tracked_boxes_maps_class_ids():
    boxes = build_tracked_boxes(
        frame_index=0,
        fps=30.0,
        boxes_xyxy=[(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)],
        class_ids=[0, 2],
        track_ids=[1, 2],
        confidences=[0.9, 0.8],
    )
    assert [box.object_class for box in boxes] == ["person", "car"]


def test_build_tracked_boxes_computes_time_from_fps():
    boxes = build_tracked_boxes(
        frame_index=126,
        fps=30.0,
        boxes_xyxy=[(0.0, 0.0, 1.0, 1.0)],
        class_ids=[2],
        track_ids=[1],
        confidences=[0.5],
    )
    assert boxes[0].time_seconds == pytest.approx(4.2)


def test_build_tracked_boxes_skips_missing_track_id():
    boxes = build_tracked_boxes(
        frame_index=0,
        fps=30.0,
        boxes_xyxy=[(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 2.0, 2.0)],
        class_ids=[0, 2],
        track_ids=[None, 5],
        confidences=[0.9, 0.9],
    )
    assert len(boxes) == 1
    assert boxes[0].track_id == 5
    assert boxes[0].object_class == "car"


def test_build_tracked_boxes_empty_input():
    boxes = build_tracked_boxes(
        frame_index=0,
        fps=30.0,
        boxes_xyxy=[],
        class_ids=[],
        track_ids=[],
        confidences=[],
    )
    assert boxes == []


def test_build_tracked_boxes_maps_full_coco_classes():
    boxes = build_tracked_boxes(
        frame_index=0,
        fps=30.0,
        boxes_xyxy=[(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)],
        class_ids=[7, 8],
        track_ids=[1, 2],
        confidences=[0.9, 0.8],
    )
    assert [box.object_class for box in boxes] == ["truck", "boat"]


def test_coco_id_to_name_covers_full_coco():
    assert len(COCO_ID_TO_NAME) == 80
    assert COCO_ID_TO_NAME[0] == "person"
    assert COCO_ID_TO_NAME[2] == "car"
    assert COCO_ID_TO_NAME[8] == "boat"


def test_build_tracked_boxes_asserts_on_unknown_class_id():
    with pytest.raises(ValueError):
        build_tracked_boxes(
            frame_index=0,
            fps=30.0,
            boxes_xyxy=[(0.0, 0.0, 1.0, 1.0)],
            class_ids=[999],
            track_ids=[1],
            confidences=[0.9],
        )


def test_filter_detections_by_class_keeps_allowed():
    boxes = [
        make_tracked_box(object_class="person"),
        make_tracked_box(object_class="car"),
        make_tracked_box(object_class="person"),
    ]
    result = filter_detections_by_class(boxes, {"car"})
    assert [box.object_class for box in result] == ["car"]


def test_filter_detections_by_class_empty_allowed_returns_empty():
    boxes = [
        make_tracked_box(object_class="person"),
        make_tracked_box(object_class="car"),
    ]
    assert filter_detections_by_class(boxes, set()) == []


def test_filter_detections_by_class_preserves_order():
    boxes = [
        make_tracked_box(object_class="car", track_id=1),
        make_tracked_box(object_class="person", track_id=2),
        make_tracked_box(object_class="car", track_id=3),
    ]
    result = filter_detections_by_class(boxes, {"car", "person"})
    assert [box.track_id for box in result] == [1, 2, 3]
