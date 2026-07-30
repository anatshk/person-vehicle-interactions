"""Integration smoke tests for visualization (need cv2 + matplotlib; skipped in lean CI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from person_vehicle_interactions.tracked_data_model import (
    ClipMetadata,
    save_metadata,
    save_tracks,
)
from tests.factories import make_tracked_box

CLIP_ID = "viz_synthetic"
WIDTH = 320
HEIGHT = 240
FRAMES = 8
FPS = 10


@pytest.fixture
def cached_clip(tmp_path):
    """Write a blank synthetic clip plus a small tracks cache into tmp dirs."""
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

    boxes = [
        make_tracked_box(
            frame=0, track_id=1, object_class="person", x1=10, y1=10, x2=40, y2=90
        ),
        make_tracked_box(
            frame=2, track_id=1, object_class="person", x1=12, y1=12, x2=60, y2=140
        ),
        make_tracked_box(
            frame=0, track_id=2, object_class="car", x1=100, y1=80, x2=260, y2=200
        ),
    ]
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
def test_show_objects_writes_sheet(cached_clip):
    from person_vehicle_interactions import visualization

    videos_dir, tracks_dir, viz_dir = cached_clip
    out_path = visualization.show_objects(
        CLIP_ID, videos_dir=videos_dir, tracks_dir=tracks_dir, viz_dir=viz_dir
    )
    assert out_path == viz_dir / "objects" / f"{CLIP_ID}.png"
    assert out_path.exists() and out_path.stat().st_size > 0


@pytest.mark.integration
def test_show_track_writes_sheet(cached_clip):
    from person_vehicle_interactions import visualization

    videos_dir, tracks_dir, viz_dir = cached_clip
    out_path = visualization.show_track(
        CLIP_ID,
        track_id=1,
        step=1,
        videos_dir=videos_dir,
        tracks_dir=tracks_dir,
        viz_dir=viz_dir,
    )
    assert out_path == viz_dir / "tracks" / f"{CLIP_ID}_track1.png"
    assert out_path.exists() and out_path.stat().st_size > 0


@pytest.mark.integration
def test_show_track_unknown_track_raises(cached_clip):
    from person_vehicle_interactions import visualization

    videos_dir, tracks_dir, viz_dir = cached_clip
    with pytest.raises(ValueError):
        visualization.show_track(
            CLIP_ID,
            track_id=999,
            step=1,
            videos_dir=videos_dir,
            tracks_dir=tracks_dir,
            viz_dir=viz_dir,
        )
