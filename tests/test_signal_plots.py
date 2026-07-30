"""Integration smoke test for signal plotting (needs matplotlib; skipped in lean CI)."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.gt_windows import InteractionWindow
from person_vehicle_interactions.interaction_signals import PairFrameSignal


@pytest.mark.integration
def test_plot_pair_signals_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from person_vehicle_interactions.signal_plots import plot_pair_signals

    series = [
        PairFrameSignal(
            frame=frame,
            time_seconds=frame / 30.0,
            overlap=0.1 * frame,
            distance=1.0 - 0.1 * frame,
            person_confidence=0.9,
            vehicle_confidence=0.8,
        )
        for frame in range(5)
    ]
    windows = [
        InteractionWindow("clip", 1, "enter", 2, 4, "person", "vehicle"),
    ]
    out_path = tmp_path / "signals" / "clip_p1_v2.png"
    result = plot_pair_signals(series, windows, "clip p1 v2", out_path)
    assert result == out_path
    assert out_path.exists() and out_path.stat().st_size > 0
