"""Unit tests for open-vocab attribute tagging (model injected, never loaded here)."""

from __future__ import annotations

from typing import Any

from person_vehicle_interactions.attribute_tagger import tag_image


class _FakeBoxes:
    """Stand-in for a YOLO-World ``result.boxes``: class indices + scores."""

    def __init__(self, class_ids: list[int], scores: list[float]) -> None:
        self.cls = class_ids
        self.conf = scores


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _FakeYoloWorld:
    """Records the vocabulary/conf it was called with and returns fixed detections."""

    def __init__(self, class_ids: list[int], scores: list[float]) -> None:
        self._result = _FakeResult(_FakeBoxes(class_ids, scores))
        self.seen_vocabulary: list[str] | None = None
        self.seen_confidence: float | None = None

    def set_classes(self, vocabulary: list[str]) -> None:
        self.seen_vocabulary = list(vocabulary)

    def predict(
        self, image: Any, verbose: bool = True, conf: float = 0.25
    ) -> list[Any]:
        self.seen_confidence = conf
        return [self._result]


def test_tag_image_maps_class_ids_to_phrases_sorted_by_score():
    model = _FakeYoloWorld(class_ids=[2, 0], scores=[0.4, 0.9])
    vocabulary = ["a man", "a woman", "a child"]

    tags = tag_image(model, image=None, vocabulary=vocabulary, top_k=3)

    # Class 0 ("a man") scored 0.9, class 2 ("a child") scored 0.4 — highest first.
    assert tags == [("a man", 0.9), ("a child", 0.4)]
    assert model.seen_vocabulary == vocabulary


def test_tag_image_respects_top_k():
    model = _FakeYoloWorld(class_ids=[0, 1, 2], scores=[0.5, 0.8, 0.2])
    vocabulary = ["a", "b", "c"]

    tags = tag_image(model, image=None, vocabulary=vocabulary, top_k=1)

    assert tags == [("b", 0.8)]


def test_tag_image_returns_empty_when_nothing_detected():
    model = _FakeYoloWorld(class_ids=[], scores=[])

    tags = tag_image(model, image=None, vocabulary=["a red car"], confidence=0.5)

    assert tags == []
    assert model.seen_confidence == 0.5
