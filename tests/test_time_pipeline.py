"""Integration smoke test for the timing harness (needs cv2 + ultralytics; skipped in lean CI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from person_vehicle_interactions.pipeline_timing import ClipTiming
from scripts.time_pipeline import time_clip

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
def test_time_clip_returns_timing_with_stage_seconds(synthetic_video):
    pytest.importorskip("ultralytics")

    timing = time_clip(synthetic_video, ground_truth_by_clip={})

    assert isinstance(timing, ClipTiming)
    assert timing.clip_id == "synthetic"
    assert timing.width == SYNTHETIC_WIDTH
    assert timing.height == SYNTHETIC_HEIGHT
    assert timing.frame_count == SYNTHETIC_FRAMES
    assert timing.detect_track_seconds >= 0.0
    assert timing.process_seconds >= 0.0
    assert timing.candidate_seconds >= 0.0
    # No ground truth supplied -> evaluation stage skipped.
    assert timing.eval_seconds is None
