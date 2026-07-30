"""
Summarize a predicted interaction window's signal values.

Reduces the per-frame :class:`PairFrameSignal` series inside a window's inclusive frame
span to a few headline numbers — peak overlap, closest approach, mean detection
confidences, and duration (frames + seconds) — for the ``results.json`` records and the
``show_interactions`` sheet captions. Pure (our dataclasses only), unit-tested in lean CI.
"""

from __future__ import annotations

import dataclasses

from person_vehicle_interactions.interaction_signals import PairFrameSignal


@dataclasses.dataclass(frozen=True)
class WindowSummary:
    """
    Headline signal values over one interaction window.
    Aggregated across the frames of the window's inclusive span that are present in the
    pair's signal series; the duration covers the full span regardless of bridged gaps.
    """

    peak_overlap: float
    min_distance: float
    mean_person_confidence: float
    mean_vehicle_confidence: float
    duration_frames: int
    duration_seconds: float


def summarize_window(
    series: list[PairFrameSignal],
    start_frame: int,
    end_frame: int,
    fps: float,
) -> WindowSummary:
    """
    Summarize the signals of one (person, vehicle) pair over an inclusive frame span.
    Only frames within ``[start_frame, end_frame]`` are aggregated. The duration spans the
    full inclusive window (``end_frame - start_frame + 1`` frames, converted via ``fps``),
    so bridged gaps still count.
    Raises:
        ValueError: If no signal frames fall within the span.
    """
    in_window = [
        signal for signal in series if start_frame <= signal.frame <= end_frame
    ]
    if not in_window:
        raise ValueError(f"No signal frames in window [{start_frame}, {end_frame}].")
    duration_frames = end_frame - start_frame + 1
    return WindowSummary(
        peak_overlap=max(signal.overlap for signal in in_window),
        min_distance=min(signal.distance for signal in in_window),
        mean_person_confidence=(
            sum(signal.person_confidence for signal in in_window) / len(in_window)
        ),
        mean_vehicle_confidence=(
            sum(signal.vehicle_confidence for signal in in_window) / len(in_window)
        ),
        duration_frames=duration_frames,
        duration_seconds=duration_frames / fps,
    )
