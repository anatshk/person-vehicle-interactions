"""
Ground-truth interaction windows: load ``interactions.csv`` into per-clip windows.

Used to shade the true interaction spans on the signal-tuning plots and, later, for
evaluation. Pure (stdlib ``csv`` only).
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

from person_vehicle_interactions.tracked_data_model import PathLike


@dataclasses.dataclass
class InteractionWindow:
    """One ground-truth interaction: a frame span with type + descriptions."""

    clip_id: str
    interaction_id: int
    interaction_type: str
    start_frame: int
    end_frame: int
    person: str
    vehicle: str


def load_gt_windows(csv_path: PathLike) -> dict[str, list[InteractionWindow]]:
    """
    Load ground-truth interactions grouped by clip id.

    Returns a mapping ``clip_id -> [InteractionWindow, ...]`` sorted by start frame.
    """
    windows_by_clip: dict[str, list[InteractionWindow]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            window = InteractionWindow(
                clip_id=row["clip_id"],
                interaction_id=int(row["interaction_id"]),
                interaction_type=row["type"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                person=row["person"],
                vehicle=row["vehicle"],
            )
            windows_by_clip.setdefault(window.clip_id, []).append(window)
    for windows in windows_by_clip.values():
        windows.sort(key=lambda window: window.start_frame)
    return windows_by_clip
