"""
Local moondream2 vision-language captioner, isolated from the rest of the pipeline.

Loads the moondream2 model once (pinned revision, weights auto-downloaded) and answers a
free-text prompt about a single image frame. The model is created by ``load_captioner`` and
then *passed into* ``query_frame``, so one loaded model can serve many frames / clips
without reloading. Deliberately pipeline-agnostic: it knows nothing about tracks, windows,
or records, and imports the heavy model stack lazily so importing this module stays cheap.

The structured shape of an answer comes from the prompt (tune the prompt to make moondream
emit parseable output); ``query_frame`` returns the raw answer text.
"""

from __future__ import annotations

from typing import Any

MODEL_ID = "vikhyatk/moondream2"
# Pinned revision — moondream2's weights and modeling code change between dated releases, so
# fixing this keeps descriptions reproducible. Auto-downloaded from Hugging Face on first
# use; document it as an external asset (like ``yolo11l.pt``).
MODEL_REVISION = "2025-06-21"


def load_captioner(revision: str = MODEL_REVISION, device: str = "cpu") -> Any:
    """
    Load the moondream2 model once (weights auto-downloaded for the pinned revision).

    Returns the model (moved to ``device``, in eval mode) to pass to ``query_frame``. The
    heavy imports live here so importing this module is cheap and the dependency stays
    quarantined to callers that caption.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model


def query_frame(model: Any, frame: Any, prompt: str, reasoning: bool = False) -> str:
    """
    Ask ``prompt`` about a single image ``frame`` using a preloaded ``model``.

    ``frame`` is a PIL image or an OpenCV-style BGR ``numpy`` array (converted to RGB here).
    ``model`` is injected rather than loaded so one model serves many frames. ``reasoning``
    is off by default — we want a short, direct answer, not a chain-of-thought trace.
    Returns the answer text (stripped); shape it via the prompt.
    """
    answer = model.query(_to_pil(frame), prompt, reasoning=reasoning)
    return answer["answer"].strip()


def _to_pil(frame: Any) -> Any:
    """Return ``frame`` as a PIL image, converting a BGR (OpenCV) numpy array to RGB."""
    from PIL import Image

    if isinstance(frame, Image.Image):
        return frame
    return Image.fromarray(frame[:, :, ::-1])
