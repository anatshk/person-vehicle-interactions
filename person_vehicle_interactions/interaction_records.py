"""
Machine-readable interaction records: predicted windows -> per-clip results JSON.

Turns the predicted (person, vehicle) interaction windows for one clip into serializable
records — clip id, track ids, frame range, time span (seconds, via the clip fps), and the
person/vehicle descriptions — and writes them to a timestamped per-clip JSON artifact. The
description of each window is supplied by an injected ``describe_window`` callback, so this
module stays pure and testable while the actual captioning lives in the glue layer.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import datetime
import json
from pathlib import Path

from person_vehicle_interactions.candidate_detection import PredictedWindow
from person_vehicle_interactions.config import RESULTS_DIR
from person_vehicle_interactions.tracked_data_model import frame_to_seconds, PathLike

# Maps a predicted window to its (person_description, vehicle_description).
DescribeWindow = Callable[[PredictedWindow], tuple[str, str]]


@dataclasses.dataclass(frozen=True)
class InteractionRecord:
    """One detected person-vehicle interaction, ready to serialize to the results JSON."""

    clip_id: str
    person_id: int
    vehicle_id: int
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    person: str
    vehicle: str


def build_clip_records(
    clip_id: str,
    windows: list[PredictedWindow],
    fps: float,
    describe_window: DescribeWindow,
) -> list[InteractionRecord]:
    """
    Build one clip's interaction records from its predicted windows.
    Windows are ordered by (start_frame, person_id, vehicle_id) for determinism; each
    window's time span is derived from ``fps`` and its descriptions from ``describe_window``.
    """
    records: list[InteractionRecord] = []
    for window in sorted(
        windows, key=lambda w: (w.start_frame, w.person_id, w.vehicle_id)
    ):
        person_description, vehicle_description = describe_window(window)
        records.append(
            InteractionRecord(
                clip_id=clip_id,
                person_id=window.person_id,
                vehicle_id=window.vehicle_id,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                start_seconds=frame_to_seconds(window.start_frame, fps),
                end_seconds=frame_to_seconds(window.end_frame, fps),
                person=person_description,
                vehicle=vehicle_description,
            )
        )
    return records


def write_clip_records(
    records: list[InteractionRecord],
    clip_id: str,
    results_dir: PathLike = RESULTS_DIR,
    generated_at: datetime.datetime | None = None,
) -> Path:
    """
    Write one clip's records to ``<results_dir>/<clip_id>_interactions_<YYYYMMDDHHMM>.json``.
    ``generated_at`` defaults to now; pass an explicit value for deterministic output.
    Returns the path written.
    """
    generated_at = generated_at or datetime.datetime.now()
    out_path = (
        Path(results_dir) / f"{clip_id}_interactions_{generated_at:%Y%m%d%H%M}.json"
    )
    payload = {
        "clip_id": clip_id,
        "generated_at": generated_at.isoformat(),
        "interactions": [dataclasses.asdict(record) for record in records],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)
    return out_path


def load_clip_records(path: PathLike) -> list[InteractionRecord]:
    """Read a per-clip results JSON back into ``InteractionRecord`` objects."""
    with Path(path).open(encoding="utf-8") as json_file:
        payload = json.load(json_file)
    return [InteractionRecord(**record) for record in payload["interactions"]]
