"""
Free-text descriptions of the interacting person and vehicle.

``describe`` is the single swap point for a captioner / vision-language model: it takes an
image crop and returns a free-text description. Until a captioner is selected (see the
write-up), it returns a fixed placeholder, and ``placeholder_description`` builds a
description from a tracked box's class and id so the results JSON is populated without a
model. Swapping in the real captioner is then a one-module change.
"""

from __future__ import annotations

from typing import Any

from person_vehicle_interactions.tracked_data_model import TrackedBox

PLACEHOLDER_DESCRIPTION = "unknown"


def describe(crop: Any) -> str:
    """
    Describe an image crop in free text (captioner hook — placeholder until one is chosen).
    Returns a fixed placeholder; swap the body for a vision-language model call to enable
    real descriptions.
    """
    del crop  # Unused until a captioner is wired in.
    return PLACEHOLDER_DESCRIPTION


def placeholder_description(box: TrackedBox) -> str:
    """Describe a tracked object by its class and track id, e.g. ``"car (track 7)"``."""
    return f"{box.object_class} (track {box.track_id})"
