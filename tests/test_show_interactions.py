"""Integration smoke test for the interaction sheet renderer (needs cv2 + matplotlib)."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.candidate_detection import Thresholds
from person_vehicle_interactions.tracked_data_model import (
    ClipMetadata,
    save_metadata,
    save_tracks,
)
from tests.factories import make_interaction_gt as _gt
from tests.factories import make_person, make_vehicle

CLIP_ID = "viz_interaction"
WIDTH = 320
HEIGHT = 240
FRAMES = 10
FPS = 10
# A person sitting fully inside the car for frames 2..6 (overlap 1.0) — one clean window.
INTERACTION_FRAMES = range(2, 7)
THRESHOLDS = Thresholds(
    min_overlap=0.5, max_distance=0.0, min_duration_frames=3, max_gap_frames=0
)


@pytest.fixture
def cached_clip(tmp_path):
    """Write a blank clip plus a tracks cache holding one person-in-vehicle interaction."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    import numpy as np

    videos_dir = tmp_path / "Videos"
    tracks_dir = tmp_path / "tracks"
    videos_dir.mkdir()
    tracks_dir.mkdir()

    writer = cv2.VideoWriter(
        str(videos_dir / f"{CLIP_ID}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    for _ in range(FRAMES):
        writer.write(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()

    boxes = []
    for frame in INTERACTION_FRAMES:
        boxes.append(make_person(frame, 1, 120, 100, 140, 160))
        boxes.append(make_vehicle(frame, 2, 100, 80, 260, 200))
    save_tracks(boxes, tracks_dir / f"{CLIP_ID}.csv")
    save_metadata(
        ClipMetadata(
            clip_id=CLIP_ID,
            fps=FPS,
            frame_width=WIDTH,
            frame_height=HEIGHT,
            frame_count=FRAMES,
            model_name="yolo11l.pt",
            image_size=1280,
            confidence_threshold=0.25,
            iou_threshold=0.7,
            tracker_name="botsort.yaml",
            seed=0,
        ),
        tracks_dir / f"{CLIP_ID}.meta.json",
    )
    return videos_dir, tracks_dir, tmp_path / "viz"


@pytest.mark.integration
def test_render_clip_interactions_writes_tp_and_fn_sheets(cached_clip):
    from scripts.show_interactions import render_clip_interactions

    videos_dir, tracks_dir, viz_dir = cached_clip
    ground_truth = [
        _gt(
            2, 6, clip_id=CLIP_ID, interaction_id=1
        ),  # matches the detected window -> TP
        _gt(
            8, 9, clip_id=CLIP_ID, interaction_id=2
        ),  # no boxes there -> FN (no person)
    ]
    paths = render_clip_interactions(
        CLIP_ID,
        THRESHOLDS,
        ground_truth,
        videos_dir=videos_dir,
        tracks_dir=tracks_dir,
        viz_dir=viz_dir,
    )
    names = sorted(path.name for path in paths)
    assert names == [
        f"{CLIP_ID}_1_2_2-6_TP.png",
        f"{CLIP_ID}_gt2_8-9_FN.png",
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
