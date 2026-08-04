"""
Pull an image crop for a tracked object: read its video frame and crop its box.

The image-extraction half of the description step, kept separate from any captioner so the
description model can change without touching frame I/O. A ``TrackedBox`` carries the frame
index and box corners, so ``crop_for_box`` turns (video, box) into the object's pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from person_vehicle_interactions.tracked_data_model import TrackedBox

PathLike = Path | str


def read_frame(video_path: PathLike, frame_index: int) -> Any:
    """
    Read a single BGR frame from a video by index.
    Raises ``ValueError`` if the frame cannot be read (bad path / index past the end).
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        read_ok, frame = capture.read()
    finally:
        capture.release()
    if not read_ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def crop_box(frame: Any, box: TrackedBox, pad_fraction: float = 0.1) -> Any:
    """
    Crop ``box`` from ``frame``, padded by ``pad_fraction`` of the box size each side and
    clamped to the frame bounds. Returns the BGR crop (a view into ``frame``).
    """
    frame_height, frame_width = frame.shape[:2]
    pad_x = pad_fraction * (box.x2 - box.x1)
    pad_y = pad_fraction * (box.y2 - box.y1)
    x1 = max(0, round(box.x1 - pad_x))
    y1 = max(0, round(box.y1 - pad_y))
    x2 = min(frame_width, round(box.x2 + pad_x))
    y2 = min(frame_height, round(box.y2 + pad_y))
    return frame[y1:y2, x1:x2]


def crop_for_box(
    video_path: PathLike, box: TrackedBox, pad_fraction: float = 0.1
) -> Any:
    """Read the box's frame and return its (padded) crop — the extract-frame + crop step."""
    return crop_box(read_frame(video_path, box.frame), box, pad_fraction)
