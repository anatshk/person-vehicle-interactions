"""Unit tests for the pure timing helpers (table formatting + timing wrapper)."""

from __future__ import annotations

from person_vehicle_interactions.pipeline_timing import (
    ClipTiming,
    format_timings_table,
    time_call,
)


def _make_timing(
    clip_id: str = "clipA",
    frame_count: int = 300,
    fps: float = 30.0,
    eval_seconds: float | None = 0.001,
) -> ClipTiming:
    return ClipTiming(
        clip_id=clip_id,
        width=1920,
        height=1080,
        fps=fps,
        frame_count=frame_count,
        size_mb=12.5,
        detect_track_seconds=4.2,
        process_seconds=0.03,
        candidate_seconds=0.002,
        eval_seconds=eval_seconds,
    )


def test_duration_seconds_from_frames_and_fps():
    timing = _make_timing(frame_count=300, fps=30.0)
    assert timing.duration_seconds == 10.0


def test_duration_seconds_is_zero_when_fps_missing():
    timing = _make_timing(frame_count=300, fps=0.0)
    assert timing.duration_seconds == 0.0


def test_time_call_returns_result_and_nonnegative_elapsed():
    result, elapsed = time_call(lambda: 21 * 2)
    assert result == 42
    assert elapsed >= 0.0


def test_format_timings_table_has_header_and_one_row_per_timing():
    table = format_timings_table([_make_timing(clip_id="clipA"), _make_timing("clipB")])
    lines = table.splitlines()
    assert lines[0].startswith("| clip")
    assert "resolution" in lines[0]
    assert "detect_track_s" in lines[0]
    # header + separator + two data rows.
    assert len(lines) == 4
    assert "clipA" in table and "clipB" in table
    assert "1920x1080" in table


def test_format_timings_table_renders_missing_eval_as_na():
    table = format_timings_table([_make_timing(eval_seconds=None)])
    assert "n/a" in table
