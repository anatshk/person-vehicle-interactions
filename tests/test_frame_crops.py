"""Unit tests for box cropping (the pure part of frame extraction)."""

from __future__ import annotations

import numpy as np

from person_vehicle_interactions.frame_crops import crop_box
from tests.factories import make_tracked_box


def test_crop_box_no_padding_returns_exact_box_region():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    box = make_tracked_box(x1=10, y1=20, x2=30, y2=50)

    crop = crop_box(frame, box, pad_fraction=0.0)

    # rows y1:y2 = 20:50 (30 px), cols x1:x2 = 10:30 (20 px).
    assert crop.shape == (30, 20, 3)


def test_crop_box_padding_expands_within_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = make_tracked_box(x1=40, y1=40, x2=60, y2=60)  # 20x20

    crop = crop_box(frame, box, pad_fraction=0.5)  # +10 px each side

    assert crop.shape == (40, 40, 3)


def test_crop_box_clamps_to_frame_bounds():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    box = make_tracked_box(x1=0, y1=0, x2=48, y2=48)

    crop = crop_box(frame, box, pad_fraction=0.5)  # would overrun on all sides

    assert crop.shape == (50, 50, 3)
