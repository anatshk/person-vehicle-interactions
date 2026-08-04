"""Tests for building and (de)serializing machine-readable interaction records."""

from __future__ import annotations

import datetime
import json

from person_vehicle_interactions.candidate_detection import PredictedWindow
from person_vehicle_interactions.interaction_records import (
    build_clip_records,
    InteractionRecord,
    load_clip_records,
    write_clip_records,
)
from tests.factories import make_interaction_prediction as _pred

GENERATED_AT = datetime.datetime(2026, 7, 30, 13, 45)


def _describe_window(window: PredictedWindow) -> tuple[str, str]:
    return f"person {window.person_id}", f"vehicle {window.vehicle_id}"


def test_build_clip_records_converts_frames_to_seconds_and_applies_descriptions():
    records = build_clip_records(
        "clipA",
        [_pred(10, 20, person_id=1, vehicle_id=2)],
        fps=10.0,
        describe_window=_describe_window,
    )
    assert records == [
        InteractionRecord(
            clip_id="clipA",
            person_id=1,
            vehicle_id=2,
            start_frame=10,
            end_frame=20,
            start_seconds=1.0,
            end_seconds=2.0,
            person="person 1",
            vehicle="vehicle 2",
        )
    ]


def test_build_clip_records_sorts_by_start_frame_then_ids():
    windows = [
        _pred(30, 40, person_id=3, vehicle_id=1),
        _pred(10, 20, person_id=2, vehicle_id=5),
        _pred(10, 20, person_id=1, vehicle_id=9),
    ]
    records = build_clip_records(
        "c", windows, fps=30.0, describe_window=_describe_window
    )
    keys = [(r.start_frame, r.person_id, r.vehicle_id) for r in records]
    assert keys == [(10, 1, 9), (10, 2, 5), (30, 3, 1)]


def test_build_clip_records_empty_windows_is_empty():
    assert build_clip_records("c", [], fps=30.0, describe_window=_describe_window) == []


def test_write_clip_records_filename_and_payload(tmp_path):
    records = build_clip_records(
        "clipA", [_pred(10, 20)], fps=10.0, describe_window=_describe_window
    )
    path = write_clip_records(
        records, "clipA", results_dir=tmp_path, generated_at=GENERATED_AT, method="fast"
    )
    assert path == tmp_path / "clipA_interactions_fast_202607301345.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["clip_id"] == "clipA"
    assert payload["generated_at"] == "2026-07-30T13:45:00"
    assert len(payload["interactions"]) == 1
    assert payload["interactions"][0]["person"] == "person 1"


def test_write_then_load_round_trips(tmp_path):
    records = build_clip_records(
        "clipA",
        [_pred(10, 20), _pred(30, 40, person_id=3, vehicle_id=4)],
        fps=10.0,
        describe_window=_describe_window,
    )
    path = write_clip_records(
        records, "clipA", results_dir=tmp_path, generated_at=GENERATED_AT
    )
    assert load_clip_records(path) == records
