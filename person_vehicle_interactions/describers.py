"""
Real person/vehicle describers: turn an image crop into a short free-text description.

A ``Describer`` is ``(crop, kind) -> str`` where ``kind`` is ``"person"`` or ``"vehicle"``,
so one object serves both. Two backends are provided, both loaded once and injected:

- **fast** (``yoloworld_describer``): open-vocab YOLO-World scores a small attribute
  vocabulary per dimension (gender / clothing, or colour / type) and templates the winners
  into a phrase. Deterministic, ~0.2 s/crop, offline — the default backend.
- **detailed** (``moondream_describer``): the moondream2 VLM answers a free-text prompt about
  the crop. Much richer, but ~90 s-5 min/crop on CPU — opt-in via ``--detailed``.

The templating (``describe_from_tags``) is pure and unit-tested; the backends are thin glue
over ``attribute_tagger`` / ``captioner`` and import those (and their heavy models) lazily.

TODO[FUTURE]: description-as-FP-filter. Beyond describing, ask the VLM a discriminating
yes/no on the *union* crop of the person+vehicle boxes — "is the person entering/exiting the
vehicle, or just passing by?" — and drop candidates answered "passing by". This gates the
hard pass-by false positives that geometry can't (a person crossing in front of a parked car
reads like an entry by bbox overlap alone). Proof of concept (validated on real crops, not
yet wired): moondream on the union crop answered the ``NmlzoaDcOuI_1`` p66 passer-by
"passing by" on all 3 sampled frames, and the ``NmlzoaDcOuI_6`` p2 real interaction
"interacting" 4/4. Deferred here because it is slow (moondream, ~300-575 s/query under CPU
contention) and belongs behind its own opt-in flag.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

# A crop describer: (image_crop, kind) -> short description. ``kind`` is KIND_PERSON/VEHICLE.
Describer = Callable[[Any, str], str]

KIND_PERSON = "person"
KIND_VEHICLE = "vehicle"

# Moondream prompts, one per kind — shape the answer by wording the prompt.
PERSON_PROMPT = (
    "Describe this person in a few words: apparent gender, clothing, and colours."
)
VEHICLE_PROMPT = "Describe this vehicle in a few words: its type and colour."


@dataclasses.dataclass(frozen=True)
class AttributeDimension:
    """One attribute to probe (e.g. gender): the open-vocab phrases scored against a crop."""

    vocabulary: list[str]


# Attribute dimensions probed per kind for the fast (YOLO-World) backend. Order matters —
# winners are templated left to right ("a" + winners) into e.g. "a man in dark clothing" or
# "a silver sedan". The vocabularies are deliberately small and are a tuning knob, not a
# contract (open-vocab on low-res CCTV is attribute-noisy; see the write-up).
PERSON_DIMENSIONS: tuple[AttributeDimension, ...] = (
    AttributeDimension(["man", "woman"]),
    AttributeDimension(
        ["in dark clothing", "in light clothing", "in colourful clothing"]
    ),
)
VEHICLE_DIMENSIONS: tuple[AttributeDimension, ...] = (
    AttributeDimension(["black", "white", "silver", "grey", "red", "blue"]),
    AttributeDimension(["sedan", "SUV", "truck", "bus", "van", "pickup truck"]),
)


def describe_from_tags(winners: list[str], kind: str) -> str:
    """
    Template the winning attribute phrases into a description, e.g. ``"a silver sedan"``.
    Falls back to the bare kind (``"a person"`` / ``"a vehicle"``) when nothing was tagged.
    """
    if not winners:
        return f"a {kind}"
    return "a " + " ".join(winners)


def _winning_tags(
    tag_crop: Callable[[Any, list[str]], list[tuple[str, float]]],
    crop: Any,
    dimensions: tuple[AttributeDimension, ...],
) -> list[str]:
    """Score each dimension's vocabulary against ``crop`` and keep the top phrase per one."""
    winners: list[str] = []
    for dimension in dimensions:
        tags = tag_crop(crop, dimension.vocabulary)
        if tags:
            winners.append(tags[0][0])
    return winners


def yoloworld_describer(model: Any) -> Describer:
    """
    Build a fast describer over a preloaded YOLO-World ``model`` (injected, not loaded here).
    Probes the per-kind attribute dimensions and templates the winners into a phrase.
    """
    from person_vehicle_interactions.attribute_tagger import tag_image

    def describe(crop: Any, kind: str) -> str:
        dimensions = PERSON_DIMENSIONS if kind == KIND_PERSON else VEHICLE_DIMENSIONS
        winners = _winning_tags(
            lambda crop_image, vocabulary: tag_image(
                model, crop_image, vocabulary, top_k=1
            ),
            crop,
            dimensions,
        )
        return describe_from_tags(winners, kind)

    return describe


def moondream_describer(model: Any) -> Describer:
    """
    Build a detailed describer over a preloaded moondream2 ``model`` (injected, not loaded).
    Asks the per-kind free-text prompt and returns the model's answer verbatim.
    """
    from person_vehicle_interactions.captioner import query_frame

    def describe(crop: Any, kind: str) -> str:
        prompt = PERSON_PROMPT if kind == KIND_PERSON else VEHICLE_PROMPT
        return query_frame(model, crop, prompt)

    return describe
