"""
Leave-one-scene-out (LOSO) cross-validation framework.

Thresholds are fit on all-but-one scene and tested on the held-out scene, so tuning never
sees the scene it's judged on. Clips from the same physical scene are grouped so a scene is
never split across train/test (no leakage).

The split + orchestration here is pure; the actual threshold-fitting and evaluation are
passed in as callables, so this module has no dependency on the signal / detection code.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import TypeVar

# Scene units: clips grouped by the physical scene they come from (see PLAN). Same-scene
# clips stay together so LOSO never leaks a scene across the train/test boundary.
SCENE_UNITS: dict[str, tuple[str, ...]] = {
    "cctv_night": ("1THkHYIQ_bY_0",),
    "aerial_4k": ("gt1125_06",),
    "grayscale_cctv": ("HIu4lM4B8hA_1",),
    "indoor_ceiling": ("iMGR_0AG3a8_2_3",),
    "ptz": ("mKzCQKTHizw_0", "mKzCQKTHizw_1"),
    "elevated_fixed": ("NmlzoaDcOuI_1", "NmlzoaDcOuI_6"),
}

Thresholds = TypeVar("Thresholds")
Result = TypeVar("Result")


@dataclasses.dataclass(frozen=True)
class LosoSplit:
    """One LOSO fold: a held-out scene's clips (test) vs all the other clips (train)."""

    held_out_scene: str
    train_clips: tuple[str, ...]
    test_clips: tuple[str, ...]


def all_clips() -> tuple[str, ...]:
    """All clip ids across every scene unit, sorted."""
    return tuple(sorted(clip for clips in SCENE_UNITS.values() for clip in clips))


def scene_for_clip(clip_id: str) -> str:
    """Return the scene unit a clip belongs to, or raise ``KeyError`` if unknown."""
    for scene, clips in SCENE_UNITS.items():
        if clip_id in clips:
            return scene
    raise KeyError(f"Clip {clip_id!r} is not in any scene unit.")


def loso_splits() -> list[LosoSplit]:
    """
    Build one LOSO fold per scene unit.

    Each fold's test set is a scene's clips; its train set is every other scene's clips.
    """
    clips = all_clips()
    splits: list[LosoSplit] = []
    for scene, scene_clips in SCENE_UNITS.items():
        test_clips = tuple(sorted(scene_clips))
        train_clips = tuple(clip for clip in clips if clip not in scene_clips)
        splits.append(LosoSplit(scene, train_clips, test_clips))
    return splits


def run_loso(
    fit_thresholds: Callable[[tuple[str, ...]], Thresholds],
    evaluate: Callable[[tuple[str, ...], Thresholds], Result],
) -> dict[str, Result]:
    """
    Run LOSO cross-validation.

    For each fold: ``fit_thresholds(train_clips)`` produces thresholds, then
    ``evaluate(test_clips, thresholds)`` scores the held-out scene. Returns a mapping of
    held-out scene -> evaluation result.
    """
    results: dict[str, Result] = {}
    for split in loso_splits():
        thresholds = fit_thresholds(split.train_clips)
        results[split.held_out_scene] = evaluate(split.test_clips, thresholds)
    return results
