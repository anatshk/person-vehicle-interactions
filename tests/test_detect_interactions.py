"""Tests for the deliverable CLI: input gathering, stage dispatch, per-clip output."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from person_vehicle_interactions.interaction_records import load_clip_records
from person_vehicle_interactions.tracked_data_model import (
    ClipMetadata,
    save_metadata,
    save_tracks,
)
from scripts import detect_interactions
from scripts.detect_interactions import (
    gather_inputs,
    normalize_argv,
    process_all,
    process_tracks_classify,
    process_video_full,
)
from tests.factories import make_person_in_vehicle_boxes

GENERATED_AT = datetime.datetime(2026, 7, 31, 12, 0)


def _write_synthetic_tracks(clip_dir: Path, clip_id: str) -> Path:
    """Write a synthetic person-in-vehicle tracks cache into ``clip_dir``; return its csv."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    boxes = make_person_in_vehicle_boxes(range(0, 15), person_id=1, vehicle_id=2)
    tracks_path = clip_dir / f"{clip_id}.csv"
    save_tracks(boxes, tracks_path)
    save_metadata(
        ClipMetadata(
            clip_id=clip_id,
            fps=30.0,
            frame_width=100,
            frame_height=100,
            frame_count=20,
            model_name="yolo11l.pt",
            image_size=1280,
            confidence_threshold=0.25,
            iou_threshold=0.7,
            tracker_name="botsort.yaml",
            seed=0,
        ),
        clip_dir / f"{clip_id}.meta.json",
    )
    return tracks_path


def test_normalize_argv_defaults_to_run_pipeline():
    assert normalize_argv(["clip.mp4"]) == ["run", "clip.mp4"]


def test_normalize_argv_leaves_explicit_command_and_flags():
    assert normalize_argv(["detect-and-track", "clip.mp4"]) == [
        "detect-and-track",
        "clip.mp4",
    ]
    assert normalize_argv(["classify-tracks", "t.csv"]) == ["classify-tracks", "t.csv"]
    assert normalize_argv(["-h"]) == ["-h"]


def test_gather_inputs_single_file_and_folder(tmp_path):
    single = tmp_path / "one.mp4"
    single.write_bytes(b"")
    assert gather_inputs(single, ".mp4") == [single]

    folder = tmp_path / "vids"
    folder.mkdir()
    (folder / "b.mp4").write_bytes(b"")
    (folder / "a.mp4").write_bytes(b"")
    (folder / "note.txt").write_text("x")
    assert gather_inputs(folder, ".mp4") == [folder / "a.mp4", folder / "b.mp4"]


def test_gather_inputs_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        gather_inputs(tmp_path / "nope.mp4", ".mp4")


def test_gather_inputs_rejects_wrong_extension_file(tmp_path):
    not_mp4 = tmp_path / "clip.mov"
    not_mp4.write_bytes(b"")
    with pytest.raises(ValueError, match=".mp4"):
        gather_inputs(not_mp4, ".mp4")


def test_process_tracks_classify_copies_tracks_and_writes_under_per_clip_folder(
    tmp_path,
):
    clip_id = "myclip"
    tracks_path = _write_synthetic_tracks(tmp_path / "tracks", clip_id)
    results_dir = tmp_path / "out"

    path = process_tracks_classify(
        tracks_path, results_dir=results_dir, generated_at=GENERATED_AT
    )

    clip_dir = results_dir / clip_id
    assert path.parent == clip_dir
    assert (clip_dir / f"{clip_id}.csv").exists()
    assert (clip_dir / f"{clip_id}.meta.json").exists()
    records = load_clip_records(path)
    assert len(records) == 1
    assert records[0].clip_id == clip_id
    assert records[0].person == "person (track 1)"


def test_process_tracks_classify_is_safe_when_tracks_already_in_place(tmp_path):
    clip_id = "inplace"
    results_dir = tmp_path / "out"
    tracks_path = _write_synthetic_tracks(results_dir / clip_id, clip_id)

    path = process_tracks_classify(
        tracks_path, results_dir=results_dir, generated_at=GENERATED_AT
    )

    assert path.parent == results_dir / clip_id
    assert len(load_clip_records(path)) == 1


def test_process_video_full_detects_then_classifies(tmp_path, monkeypatch):
    clip_id = "phoneclip"

    def fake_detect(video_path: Path, clip_dir: Path, force: bool = False) -> Path:
        return _write_synthetic_tracks(clip_dir, video_path.stem)

    monkeypatch.setattr(detect_interactions, "_detect_and_track", fake_detect)
    video_path = tmp_path / "videos" / f"{clip_id}.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")
    results_dir = tmp_path / "out"

    path = process_video_full(
        video_path, results_dir=results_dir, generated_at=GENERATED_AT
    )

    assert path.parent == results_dir / clip_id
    records = load_clip_records(path)
    assert len(records) == 1
    assert records[0].clip_id == clip_id


def test_process_video_full_uses_describer_for_descriptions(tmp_path, monkeypatch):
    clip_id = "describedclip"

    def fake_detect(video_path: Path, clip_dir: Path, force: bool = False) -> Path:
        return _write_synthetic_tracks(clip_dir, video_path.stem)

    monkeypatch.setattr(detect_interactions, "_detect_and_track", fake_detect)
    # Skip real video I/O: the crop is irrelevant to the injected describer.
    monkeypatch.setattr("scripts.build_results.crop_for_box", lambda *a, **k: object())
    video_path = tmp_path / "videos" / f"{clip_id}.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"")

    def describer(crop, kind):
        return f"a {kind} (described)"

    path = process_video_full(
        video_path,
        results_dir=tmp_path / "out",
        generated_at=GENERATED_AT,
        describer=describer,
    )

    record = load_clip_records(path)[0]
    assert record.person == "a person (described)"
    assert record.vehicle == "a vehicle (described)"


def test_build_describer_placeholder_is_none():
    assert detect_interactions.build_describer("placeholder") is None


def test_process_all_reports_failures_and_continues(tmp_path, capsys):
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    good.write_text("x")
    bad.write_text("x")

    def process(path: Path) -> Path:
        if path.name == "bad.csv":
            raise ValueError("boom")
        return path

    failures = process_all([bad, good], process)

    assert failures == 1
    output = capsys.readouterr().out
    assert "bad.csv: ERROR" in output
    assert "good.csv" in output
