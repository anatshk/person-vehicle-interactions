"""Integration tests for tracker_engine (need cv2 + ultralytics; skipped in lean CI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from person_vehicle_interactions import tracker_engine
from person_vehicle_interactions.config import DetectionConfig
from person_vehicle_interactions.tracked_data_model import load_metadata, load_tracks

SYNTHETIC_VIDEO = Path("tests/videos/synthetic.mp4")
SYNTHETIC_FPS = 10
SYNTHETIC_FRAMES = 8
SYNTHETIC_WIDTH = 320
SYNTHETIC_HEIGHT = 240


@pytest.fixture(scope="module")
def synthetic_video() -> Path:
    """Create a tiny blank synthetic mp4 once (regenerated if missing)."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    if SYNTHETIC_VIDEO.exists():
        return SYNTHETIC_VIDEO
    SYNTHETIC_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(SYNTHETIC_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        SYNTHETIC_FPS,
        (SYNTHETIC_WIDTH, SYNTHETIC_HEIGHT),
    )
    for _ in range(SYNTHETIC_FRAMES):
        writer.write(np.zeros((SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH, 3), dtype=np.uint8))
    writer.release()
    return SYNTHETIC_VIDEO


@pytest.mark.integration
def test_read_video_properties(synthetic_video):
    props = tracker_engine.read_video_properties(synthetic_video)
    assert props.frame_width == SYNTHETIC_WIDTH
    assert props.frame_height == SYNTHETIC_HEIGHT
    assert props.frame_count == SYNTHETIC_FRAMES
    assert props.fps == pytest.approx(SYNTHETIC_FPS, abs=1)


@pytest.mark.integration
def test_cache_clip_tracks_smoke(synthetic_video, tmp_path):
    pytest.importorskip("ultralytics")
    config = DetectionConfig()
    raw_dir = tmp_path / "raw"
    tracks_dir = tmp_path / "tracks"

    tracks_path, meta_path = tracker_engine.cache_clip_tracks(
        synthetic_video, "synthetic", config, raw_dir=raw_dir, tracks_dir=tracks_dir
    )
    assert tracks_path.exists()
    assert meta_path.exists()
    assert (raw_dir / "synthetic.jsonl").exists()
    # Blank frames -> no detections -> no tracks.
    assert load_tracks(tracks_path) == []
    metadata = load_metadata(meta_path)
    assert metadata.clip_id == "synthetic"
    assert metadata.frame_count == SYNTHETIC_FRAMES


@pytest.mark.integration
def test_cache_clip_tracks_checkpoint_skips_rerun(synthetic_video, tmp_path):
    pytest.importorskip("ultralytics")
    config = DetectionConfig()
    raw_dir = tmp_path / "raw"
    tracks_dir = tmp_path / "tracks"

    tracker_engine.cache_clip_tracks(
        synthetic_video, "synthetic", config, raw_dir=raw_dir, tracks_dir=tracks_dir
    )
    tracks_path = tracks_dir / "synthetic.csv"
    first_mtime = tracks_path.stat().st_mtime_ns
    # Second call should be a no-op (tracks already cached), not rewrite the file.
    tracker_engine.cache_clip_tracks(
        synthetic_video, "synthetic", config, raw_dir=raw_dir, tracks_dir=tracks_dir
    )
    assert tracks_path.stat().st_mtime_ns == first_mtime
