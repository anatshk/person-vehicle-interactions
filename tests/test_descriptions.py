"""Tests for the interaction description hook and placeholder builder."""

from __future__ import annotations

from person_vehicle_interactions.descriptions import (
    describe,
    placeholder_description,
    PLACEHOLDER_DESCRIPTION,
)
from tests.factories import make_tracked_box


def test_describe_returns_the_placeholder_until_a_captioner_is_wired():
    # The hook is a stub for now; assert it yields the documented placeholder verbatim.
    assert describe(crop=None) == PLACEHOLDER_DESCRIPTION


def test_placeholder_description_uses_class_and_track_id():
    box = make_tracked_box(track_id=7, object_class="car")
    assert placeholder_description(box) == "car (track 7)"
