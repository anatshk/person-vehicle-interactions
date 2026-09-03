"""Detection/tracking configuration and cache paths."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from person_vehicle_interactions.candidate_detection import Thresholds

VIDEOS_DIR = Path("Videos")
CACHE_DIR = Path("cache")
RAW_DIR = CACHE_DIR / "raw"
TRACKS_DIR = CACHE_DIR / "tracks"
VIZ_DIR = CACHE_DIR / "viz"
RESULTS_DIR = CACHE_DIR / "results"

# COCO class id for the person we look for interactions with.
PERSON_CLASS_ID: int = 0

# Which COCO classes count as a "vehicle" for interaction reasoning: car, bus, truck.
VEHICLE_CLASS_IDS: tuple[int, ...] = (2, 5, 7)

# Per-clip extra vehicle classes, keyed by clip id. Used where the scene causes the
# detector to misclassify vehicles (e.g. cars labeled 'boat' in a low-res night clip).
# TODO: expand the vehicle definition if the video quality is low — ideally drive this
#       from clip properties (resolution / lighting) instead of a hardcoded per-clip map.
CLIP_VEHICLE_CLASS_OVERRIDES: dict[str, tuple[int, ...]] = {
    "HIu4lM4B8hA_1": (8,),  # boat: low-res night clip where cars misdetect as boat
}


def vehicle_class_ids_for_clip(clip_id: str) -> tuple[int, ...]:
    """
    Return the vehicle class ids for a clip: the base set plus any per-clip override.

    The result is sorted and de-duplicated so it is deterministic.
    """
    extra_class_ids = CLIP_VEHICLE_CLASS_OVERRIDES.get(clip_id, ())
    return tuple(sorted(set(VEHICLE_CLASS_IDS) | set(extra_class_ids)))


def target_class_ids_for_clip(clip_id: str) -> tuple[int, ...]:
    """Return the detection keep-set for a clip: person plus its vehicle classes."""
    return (PERSON_CLASS_ID,) + vehicle_class_ids_for_clip(clip_id)


def vehicle_class_names_for_clip(clip_id: str) -> set[str]:
    """
    Return the vehicle class *names* for a clip (COCO names of its vehicle class ids).

    The single source of truth for "which classes count as a vehicle" for consumers that
    work with class names (e.g. tracked boxes / signals) rather than ids.
    """
    from person_vehicle_interactions.detection_processing import COCO_ID_TO_NAME

    return {
        COCO_ID_TO_NAME[class_id] for class_id in vehicle_class_ids_for_clip(clip_id)
    }


@dataclasses.dataclass(frozen=True)
class DetectionConfig:
    """Detection + tracking parameters, pinned for determinism."""

    model_name: str = "yolo11l.pt"
    image_size: int = 1280
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    tracker_name: str = "botsort.yaml"
    # COCO class ids to keep; ``None`` means unfiltered (detect/keep all classes).
    # Default: person + the base vehicle classes; per-clip runs use
    # ``target_class_ids_for_clip`` to add clip-specific vehicle classes.
    target_class_ids: tuple[int, ...] | None = (PERSON_CLASS_ID,) + VEHICLE_CLASS_IDS
    seed: int = 0


# The single fixed interaction thresholds shipped with the deliverable: fit on ALL clips
# (not per LOSO fold), so a run is reproducible and applies to clips without ground truth.
# LOSO is a research-only concern; the deliverable never sees it. Kept in sync with
# ``fit(all_clips())`` by ``test_shipped_thresholds_match_fit_on_all``.
# TODO: change the shipped thresholds to the better values found by the widened grid search
#   (min_overlap=0.4, max_distance=0.0, min_duration_frames=3, min_confidence=0.5,
#   max_gap_frames=15), which beat these on the fit-on-all score (F1 0.85 vs 0.71). This must
#   be done together with widening DEFAULT_GRID to include those values, or
#   test_shipped_thresholds_match_fit_on_all will fail. See scripts/widen_grid_experiment.py
#   and the "Last Minute Checks" section of docs/DETAILED_WRITE_UP.md.
SHIPPED_THRESHOLDS = Thresholds(
    min_overlap=0.2,
    max_distance=0.0,
    min_duration_frames=10,
    min_confidence=0.3,
    max_gap_frames=15,
)
