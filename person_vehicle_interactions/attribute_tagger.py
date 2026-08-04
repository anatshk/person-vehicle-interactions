"""
Open-vocabulary attribute tagging with YOLO-World.

The "description" of an object is the highest-scoring phrase(s) from a vocabulary you design
(e.g. ``["a man", "a woman"]`` or ``["a red car", "a white car", ...]``). YOLO-World scores
arbitrary text prompts against an image in a single fast forward pass — a deterministic,
CPU-cheap stand-in for a captioner. The model is loaded once via ``load_tagger`` and injected
into ``tag_image`` so one model serves many crops.

Note: YOLO-World returns one best phrase per detected box, so probe one attribute *dimension*
at a time (gender, top colour, vehicle type, ...) and template the winners into a description.
"""

from __future__ import annotations

from typing import Any

DEFAULT_WEIGHTS = "yolov8s-world.pt"


def load_tagger(weights: str = DEFAULT_WEIGHTS) -> Any:
    """Load a YOLO-World open-vocab model once (weights auto-downloaded on first use)."""
    from ultralytics import YOLOWorld

    return YOLOWorld(weights)


def tag_image(
    model: Any,
    image: Any,
    vocabulary: list[str],
    top_k: int = 1,
    confidence: float = 0.01,
) -> list[tuple[str, float]]:
    """
    Score ``vocabulary`` phrases against ``image`` with a preloaded YOLO-World ``model``.

    Returns up to ``top_k`` ``(phrase, score)`` pairs, highest score first (empty if nothing
    clears ``confidence``). ``model`` is injected, not loaded, so one model serves many crops.
    """
    model.set_classes(vocabulary)
    result = model.predict(image, verbose=False, conf=confidence)[0]
    tags = [
        (vocabulary[int(class_id)], float(score))
        for class_id, score in zip(result.boxes.cls, result.boxes.conf)
    ]
    tags.sort(key=lambda tag: -tag[1])
    return tags[:top_k]
