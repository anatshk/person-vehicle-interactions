"""
Glue: plot per-pair interaction signals vs frame, with ground-truth windows shaded.

The tuning graph — overlap + detection confidences on the left axis (0..1) and normalized
center-distance on the right axis, with GT interaction spans shaded, so thresholds can be
read off against the truth. matplotlib is imported lazily (Agg), so the module stays
importable where it is absent (the smoke test skips via importorskip).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from person_vehicle_interactions.gt_windows import InteractionWindow
from person_vehicle_interactions.interaction_signals import PairFrameSignal

PathLike = Path | str

# Distances above this many vehicle-diagonals aren't interesting for tuning; cap the axis
# so the near-contact region stays readable (spikes clip off-axis).
DISTANCE_AXIS_CAP: float = 3.0


def plot_pair_signals(
    series: list[PairFrameSignal],
    gt_windows: list[InteractionWindow],
    title: str,
    out_path: PathLike,
) -> Path:
    """
    Plot one (person, vehicle) pair's signals vs frame; shade GT windows; save a PNG.

    Left axis: overlap + person/vehicle confidence (0..1). Right axis: normalized distance.
    Returns the written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [signal.frame for signal in series]
    figure, overlap_axis = plt.subplots(figsize=(10, 4))
    distance_axis = overlap_axis.twinx()

    lines: list[Any] = []
    if frames:
        lines += overlap_axis.plot(
            frames, [s.overlap for s in series], color="tab:blue", label="overlap"
        )
        lines += overlap_axis.plot(
            frames,
            [s.person_confidence for s in series],
            color="tab:green",
            linestyle="--",
            label="person conf",
        )
        lines += overlap_axis.plot(
            frames,
            [s.vehicle_confidence for s in series],
            color="tab:olive",
            linestyle=":",
            label="vehicle conf",
        )
        lines += distance_axis.plot(
            frames, [s.distance for s in series], color="tab:red", label="distance"
        )

    overlap_axis.set_xlabel("frame")
    overlap_axis.set_ylabel("overlap / confidence")
    overlap_axis.set_ylim(0.0, 1.05)
    distance_axis.set_ylabel("normalized distance")
    distance_axis.set_ylim(0.0, DISTANCE_AXIS_CAP)

    shaded_label_used = False
    for window in gt_windows:
        overlap_axis.axvspan(
            window.start_frame,
            window.end_frame,
            color="tab:orange",
            alpha=0.15,
            label=None if shaded_label_used else "GT window",
        )
        shaded_label_used = True

    handles, labels = overlap_axis.get_legend_handles_labels()
    line_labels = [line.get_label() for line in lines]
    overlap_axis.legend(
        lines + handles[len(line_labels) :],
        line_labels + labels[len(line_labels) :],
        loc="upper right",
        fontsize=8,
    )
    overlap_axis.set_title(title, fontsize=10)
    overlap_axis.grid(True, alpha=0.3)
    figure.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return out_path
