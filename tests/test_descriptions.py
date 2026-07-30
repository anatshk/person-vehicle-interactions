"""Tests for the interaction description hook and placeholder builder."""

from __future__ import annotations

from person_vehicle_interactions.descriptions import describe, placeholder_description
from tests.factories import make_tracked_box


def test_describe_returns_a_nonempty_string():
    # Placeholder captioner hook: shape only, until a real model is wired in.
    result = describe(crop=None)
    assert isinstance(result, str)
    assert result


def test_placeholder_description_uses_class_and_track_id():
    box = make_tracked_box(track_id=7, object_class="car")
    assert placeholder_description(box) == "car (track 7)"
