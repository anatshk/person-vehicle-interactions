"""
Pure helpers for timing the pipeline stages and rendering the results as a table.

Kept free of cv2 / ultralytics so it unit-tests in lean CI; the glue that actually runs
the stages and measures them lives in ``scripts/time_pipeline.py``. ``ClipTiming`` holds one
clip's video properties alongside the per-stage wall-clock seconds, and
``format_timings_table`` renders a list of them as a Markdown table for the write-up.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import time
from typing import TypeVar

Result = TypeVar("Result")

# Column header -> the ClipTiming attribute (or property) it renders.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("clip", "clip_id"),
    ("resolution", "resolution"),
    ("fps", "fps"),
    ("frames", "frame_count"),
    ("duration_s", "duration_seconds"),
    ("size_mb", "size_mb"),
    ("detect_track_s", "detect_track_seconds"),
    ("process_s", "process_seconds"),
    ("candidate_s", "candidate_seconds"),
    ("eval_s", "eval_seconds"),
)


@dataclasses.dataclass(frozen=True)
class ClipTiming:
    """One clip's video properties and the wall-clock seconds each stage took."""

    clip_id: str
    width: int
    height: int
    fps: float
    frame_count: int
    size_mb: float
    detect_track_seconds: float
    process_seconds: float
    candidate_seconds: float
    eval_seconds: (
        float | None
    )  # None when the clip has no ground truth to evaluate against.

    @property
    def resolution(self) -> str:
        """The frame size as ``WIDTHxHEIGHT`` (e.g. ``1920x1080``)."""
        return f"{self.width}x{self.height}"

    @property
    def duration_seconds(self) -> float:
        """Clip length in seconds (frames / fps); 0.0 when fps is unknown."""
        return self.frame_count / self.fps if self.fps else 0.0


def time_call(function: Callable[[], Result]) -> tuple[Result, float]:
    """Run ``function`` and return ``(result, elapsed_seconds)`` (wall-clock)."""
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _format_cell(value: object) -> str:
    """Render one cell: fixed decimals for floats, ``n/a`` for a missing value."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_timings_table(rows: list[ClipTiming]) -> str:
    """
    Render clip timings as a Markdown table, one row per clip, columns per ``_COLUMNS``.
    Floats use three decimals and a missing evaluation time shows as ``n/a``.
    """
    headers = [header for header, _ in _COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [_format_cell(getattr(row, attribute)) for _, attribute in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
