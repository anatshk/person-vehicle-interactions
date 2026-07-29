"""Tests for clip assembly: VideoProperties and build_clip_metadata."""

from __future__ import annotations

from person_vehicle_interactions.clip_assembly import (
    build_clip_metadata,
    VideoProperties,
)
from person_vehicle_interactions.config import DetectionConfig


def test_build_clip_metadata_maps_video_fields():
    video = VideoProperties(
        fps=30.0, frame_width=1280, frame_height=720, frame_count=495
    )
    metadata = build_clip_metadata("sample_clip", video, DetectionConfig())
    assert metadata.clip_id == "sample_clip"
    assert metadata.fps == 30.0
    assert metadata.frame_width == 1280
    assert metadata.frame_height == 720
    assert metadata.frame_count == 495


def test_build_clip_metadata_uses_config_values():
    video = VideoProperties(
        fps=25.0, frame_width=352, frame_height=288, frame_count=250
    )
    config = DetectionConfig(model_name="yolo11n.pt", image_size=640, seed=7)
    metadata = build_clip_metadata("clip", video, config)
    assert metadata.model_name == "yolo11n.pt"
    assert metadata.image_size == 640
    assert metadata.seed == 7
    assert metadata.confidence_threshold == config.confidence_threshold
    assert metadata.iou_threshold == config.iou_threshold
    assert metadata.tracker_name == config.tracker_name
    assert metadata.fps == 25.0
