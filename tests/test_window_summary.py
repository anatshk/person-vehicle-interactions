"""Tests for per-window signal-value summaries over a PairFrameSignal series."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.window_summary import summarize_window
from tests.factories import make_pair_signal


def test_summarize_window_aggregates_signal_values():
    series = [
        make_pair_signal(
            frame=10,
            overlap=0.2,
            distance=1.5,
            person_confidence=0.8,
            vehicle_confidence=0.7,
        ),
        make_pair_signal(
            frame=11,
            overlap=0.6,
            distance=0.4,
            person_confidence=0.9,
            vehicle_confidence=0.9,
        ),
        make_pair_signal(
            frame=12,
            overlap=0.4,
            distance=0.9,
            person_confidence=1.0,
            vehicle_confidence=0.8,
        ),
    ]
    summary = summarize_window(series, start_frame=10, end_frame=12, fps=30.0)
    assert summary.peak_overlap == pytest.approx(0.6)
    assert summary.min_distance == pytest.approx(0.4)
    assert summary.mean_person_confidence == pytest.approx(0.9)
    assert summary.mean_vehicle_confidence == pytest.approx(0.8)
    assert summary.duration_frames == 3
    assert summary.duration_seconds == pytest.approx(3 / 30.0)


def test_summarize_window_ignores_signals_outside_span():
    series = [
        make_pair_signal(frame=8, overlap=0.99, distance=0.01),  # before span
        make_pair_signal(frame=10, overlap=0.3, distance=1.0),
        make_pair_signal(frame=11, overlap=0.5, distance=0.5),
        make_pair_signal(frame=20, overlap=0.95, distance=0.02),  # after span
    ]
    summary = summarize_window(series, start_frame=10, end_frame=11, fps=25.0)
    assert summary.peak_overlap == pytest.approx(0.5)
    assert summary.min_distance == pytest.approx(0.5)
    assert summary.duration_frames == 2
    assert summary.duration_seconds == pytest.approx(2 / 25.0)


def test_summarize_window_duration_spans_the_full_window_not_just_present_frames():
    # A bridged gap (frame 11 absent) still counts toward the inclusive frame span.
    series = [
        make_pair_signal(frame=10, overlap=0.3, distance=1.0),
        make_pair_signal(frame=12, overlap=0.5, distance=0.5),
    ]
    summary = summarize_window(series, start_frame=10, end_frame=12, fps=30.0)
    assert summary.duration_frames == 3
    assert summary.duration_seconds == pytest.approx(3 / 30.0)


def test_summarize_window_raises_when_no_signals_in_span():
    series = [make_pair_signal(frame=5, overlap=0.5, distance=0.5)]
    with pytest.raises(ValueError):
        summarize_window(series, start_frame=10, end_frame=20, fps=30.0)
