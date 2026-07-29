"""Pure clip assembly: video properties and ClipMetadata construction."""

from __future__ import annotations

import dataclasses

from person_vehicle_interactions.config import DetectionConfig
from person_vehicle_interactions.tracked_data_model import ClipMetadata


@dataclasses.dataclass
class VideoProperties:
    """Intrinsic properties of a video clip (what the file is)."""

    fps: float
    frame_width: int
    frame_height: int
    frame_count: int


def build_clip_metadata(
    clip_id: str, video_properties: VideoProperties, config: DetectionConfig
) -> ClipMetadata:
    """Combine a clip id, its video properties, and the run config into ClipMetadata."""
    return ClipMetadata(
        clip_id=clip_id,
        fps=video_properties.fps,
        frame_width=video_properties.frame_width,
        frame_height=video_properties.frame_height,
        frame_count=video_properties.frame_count,
        model_name=config.model_name,
        image_size=config.image_size,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        tracker_name=config.tracker_name,
        seed=config.seed,
    )
