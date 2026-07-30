"""Tests for the leave-one-scene-out (LOSO) split + orchestration framework."""

from __future__ import annotations

import pytest

from person_vehicle_interactions.loso import (
    all_clips,
    loso_splits,
    run_loso,
    scene_for_clip,
    SCENE_UNITS,
)


def test_scene_for_clip_maps_clips_to_their_scene():
    assert scene_for_clip("mKzCQKTHizw_0") == scene_for_clip("mKzCQKTHizw_1")
    assert scene_for_clip("NmlzoaDcOuI_1") == scene_for_clip("NmlzoaDcOuI_6")
    assert scene_for_clip("gt1125_06") != scene_for_clip("1THkHYIQ_bY_0")


def test_scene_for_clip_unknown_raises():
    with pytest.raises(KeyError):
        scene_for_clip("not_a_clip")


def test_all_clips_are_the_union_of_scene_units():
    expected = {clip for clips in SCENE_UNITS.values() for clip in clips}
    assert set(all_clips()) == expected
    assert len(all_clips()) == len(set(all_clips()))  # no duplicates


def test_loso_splits_one_per_scene_and_partition():
    splits = loso_splits()
    assert len(splits) == len(SCENE_UNITS)
    for split in splits:
        assert set(split.test_clips) == set(SCENE_UNITS[split.held_out_scene])
        assert set(split.train_clips).isdisjoint(split.test_clips)
        assert set(split.train_clips) | set(split.test_clips) == set(all_clips())


def test_run_loso_injects_train_and_test_clips():
    def fit_thresholds(train_clips):
        return {"trained_on": tuple(sorted(train_clips))}

    def evaluate(test_clips, thresholds):
        return {"test_clips": tuple(sorted(test_clips)), "thresholds": thresholds}

    results = run_loso(fit_thresholds, evaluate)
    assert set(results) == set(SCENE_UNITS)
    # The held-out scene's clips are the test clips, and were excluded from training.
    ptz_result = (
        results["ptz"] if "ptz" in results else results[scene_for_clip("mKzCQKTHizw_0")]
    )
    assert set(ptz_result["test_clips"]) == set(
        SCENE_UNITS[scene_for_clip("mKzCQKTHizw_0")]
    )
    assert "mKzCQKTHizw_0" not in ptz_result["thresholds"]["trained_on"]
