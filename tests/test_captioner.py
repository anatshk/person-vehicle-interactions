"""Unit tests for the moondream2 captioner wrapper (model injected, never loaded here)."""

from __future__ import annotations

from typing import Any

import pytest

from person_vehicle_interactions.captioner import query_frame

# numpy + Pillow are not installed in lean CI (only lint/test tools) — skip the module there,
# matching the other numpy-dependent tests; it runs where the runtime deps are present.
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


class _FakeModel:
    """Stand-in for a loaded moondream2 model: records its call, returns a fixed answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.seen: tuple[Any, str] | None = None

    def query(self, image: Any, prompt: str, reasoning: bool = False) -> dict[str, str]:
        self.seen = (image, prompt)
        self.reasoning = reasoning
        return {"answer": self.answer}


def test_query_frame_passes_prompt_and_returns_stripped_answer():
    model = _FakeModel("  a silver sedan  ")
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    result = query_frame(model, frame, "Describe the vehicle.")

    assert result == "a silver sedan"
    assert model.seen is not None
    image, prompt = model.seen
    assert prompt == "Describe the vehicle."
    assert isinstance(image, Image.Image)
    assert model.reasoning is False  # short answer, no chain-of-thought


def test_query_frame_converts_bgr_numpy_to_rgb():
    model = _FakeModel("x")
    # One pixel, OpenCV BGR order (b=10, g=20, r=30); RGB should read back (30, 20, 10).
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)

    query_frame(model, frame, "?")

    image, _ = model.seen
    assert image.convert("RGB").getpixel((0, 0)) == (30, 20, 10)


def test_query_frame_accepts_pil_image_unchanged():
    model = _FakeModel("y")
    pil_image = Image.new("RGB", (2, 2), (1, 2, 3))

    query_frame(model, pil_image, "?")

    image, _ = model.seen
    assert image is pil_image
