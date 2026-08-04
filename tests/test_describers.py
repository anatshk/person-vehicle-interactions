"""Unit tests for the describers (pure templating + backends with injected fake models)."""

from __future__ import annotations

from typing import Any

import pytest

from person_vehicle_interactions.describers import (
    describe_from_tags,
    KIND_PERSON,
    KIND_VEHICLE,
    moondream_describer,
    PERSON_PROMPT,
    VEHICLE_PROMPT,
    yoloworld_describer,
)


def test_describe_from_tags_templates_winners():
    assert describe_from_tags(["silver", "sedan"], KIND_VEHICLE) == "a silver sedan"
    assert describe_from_tags(["man", "in dark clothing"], KIND_PERSON) == (
        "a man in dark clothing"
    )


def test_describe_from_tags_falls_back_to_kind_when_empty():
    assert describe_from_tags([], KIND_PERSON) == "a person"
    assert describe_from_tags([], KIND_VEHICLE) == "a vehicle"


class _FakeTagger:
    """YOLO-World stand-in: returns the first vocabulary phrase as the top tag."""

    def __init__(self) -> None:
        self.seen_vocabularies: list[list[str]] = []

    def set_classes(self, vocabulary: list[str]) -> None:
        self.seen_vocabularies.append(list(vocabulary))
        self._vocabulary = list(vocabulary)

    def predict(
        self, image: Any, verbose: bool = True, conf: float = 0.25
    ) -> list[Any]:
        top = self._vocabulary[0]

        class _Boxes:
            cls = [0]
            conf = [0.9]

        class _Result:
            boxes = _Boxes()

        del top
        return [_Result()]


def test_yoloworld_describer_probes_each_dimension_and_templates():
    model = _FakeTagger()
    describe = yoloworld_describer(model)

    description = describe(object(), KIND_VEHICLE)

    # Fake picks each dimension's first phrase: colour "black", type "sedan".
    assert description == "a black sedan"
    # Two vehicle dimensions were probed (colour, then type).
    assert len(model.seen_vocabularies) == 2


class _FakeCaptioner:
    """moondream2 stand-in: records the prompt, returns a fixed answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.seen_prompt: str | None = None

    def query(self, image: Any, prompt: str, reasoning: bool = False) -> dict[str, str]:
        self.seen_prompt = prompt
        return {"answer": self.answer}


def test_moondream_describer_uses_kind_prompt_and_returns_answer():
    image_module = pytest.importorskip("PIL.Image")
    crop = image_module.new("RGB", (2, 2))
    model = _FakeCaptioner("a woman in a red coat walking with a child")
    describe = moondream_describer(model)

    person = describe(crop, KIND_PERSON)

    assert person == "a woman in a red coat walking with a child"
    assert model.seen_prompt == PERSON_PROMPT


def test_moondream_describer_uses_vehicle_prompt_for_vehicles():
    image_module = pytest.importorskip("PIL.Image")
    crop = image_module.new("RGB", (2, 2))
    model = _FakeCaptioner("a silver mercedes sedan")
    describe = moondream_describer(model)

    describe(crop, KIND_VEHICLE)

    assert model.seen_prompt == VEHICLE_PROMPT
