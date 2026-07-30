"""Tests for the ground-truth interaction-window loader."""

from __future__ import annotations

from person_vehicle_interactions.gt_windows import load_gt_windows

CSV_TEXT = (
    "clip_id,interaction_id,type,start_sec,end_sec,start_frame,end_frame,person,vehicle,notes\n"
    'clipA,1,enter,6.41,10.01,192,300,"male, gray shirt","white SUV","reaches car"\n'
    'clipA,2,exit,7.61,17.22,228,516,"male, black shirt","black SUV","closes door"\n'
    'clipB,1,other,0.0,4.2,0,126,"female","gray car","removing cover"\n'
)


def _write_csv(tmp_path):
    path = tmp_path / "interactions.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


def test_load_gt_windows_groups_by_clip(tmp_path):
    windows = load_gt_windows(_write_csv(tmp_path))
    assert set(windows) == {"clipA", "clipB"}
    assert len(windows["clipA"]) == 2
    assert len(windows["clipB"]) == 1


def test_load_gt_windows_parses_fields(tmp_path):
    windows = load_gt_windows(_write_csv(tmp_path))
    first = windows["clipA"][0]
    assert first.interaction_id == 1
    assert first.interaction_type == "enter"
    assert first.start_frame == 192
    assert first.end_frame == 300
    assert first.person == "male, gray shirt"
    assert first.vehicle == "white SUV"


def test_load_gt_windows_orders_by_start_frame(tmp_path):
    windows = load_gt_windows(_write_csv(tmp_path))
    assert [w.start_frame for w in windows["clipA"]] == [192, 228]
