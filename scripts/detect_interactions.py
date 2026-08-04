"""
Person-vehicle interaction detector — the deliverable CLI.

Runs the full pipeline on a video (or a folder of videos): detect + track people and
vehicles, then classify their interactions into a machine-readable results JSON. The two
stages can also be run on their own. Each clip's outputs — cached tracks *and* its results
JSON — are written to a per-clip folder ``<results-dir>/<clip_id>/`` so every clip's
artifacts stay together.

    python -m scripts.detect_interactions <video|folder>                 # full (default)
    python -m scripts.detect_interactions detect-and-track <video|folder>  # video -> tracks
    python -m scripts.detect_interactions classify-tracks <tracks|folder>  # tracks -> results

The full/detect stages run YOLO11-l + BoT-SORT (needs the model). Interactions are decided
with ``config.SHIPPED_THRESHOLDS``; a bad input in a batch is reported and skipped so the
rest still run.

Both ``run`` and ``classify-tracks`` describe each interaction's person + vehicle with
``--fast`` (YOLO-World open-vocab, the default) or ``--detailed`` (moondream2 VLM,
slower/richer). Descriptions need the video pixels: ``run`` has the video as its input, while
``classify-tracks`` finds it at ``<videos-dir>/<clip>.mp4`` (default ``Videos/``) or a
``--video`` override, and additionally accepts ``--placeholder`` to stay model-free and
offline. A backend that fails to load, or a missing video, falls back to placeholder
descriptions with a warning rather than aborting the run.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import functools
from pathlib import Path
import shutil
import sys

from person_vehicle_interactions.config import (
    DetectionConfig,
    RESULTS_DIR,
    SHIPPED_THRESHOLDS,
    target_class_ids_for_clip,
    VIDEOS_DIR,
)
from person_vehicle_interactions.describers import Describer
from scripts.build_results import build_clip_results, make_describe_box

VIDEO_SUFFIX = ".mp4"
TRACKS_SUFFIX = ".csv"
COMMANDS = ("run", "detect-and-track", "classify-tracks")


def normalize_argv(argv: list[str]) -> list[str]:
    """
    Default to the full ``run`` pipeline when no subcommand is given.
    A first token that is neither a known command nor a flag is treated as the ``path`` of
    an implicit ``run``, so ``detect_interactions clip.mp4`` means ``run clip.mp4``.
    """
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        return ["run", *argv]
    return list(argv)


def gather_inputs(path: Path, suffix: str) -> list[Path]:
    """
    Resolve an input path to the files to process: a folder's ``*suffix`` files (sorted) or
    the single file itself.

    Raises ``FileNotFoundError`` if the path does not exist and ``ValueError`` if a single
    file has the wrong extension (e.g. a non-``.mp4`` video), so the CLI can exit gracefully
    instead of failing deep inside detection.
    """
    if path.is_dir():
        return sorted(path.glob(f"*{suffix}"))
    if path.is_file():
        if path.suffix.lower() != suffix:
            raise ValueError(f"Expected a {suffix} file, got '{path.name}'")
        return [path]
    raise FileNotFoundError(f"No such file or directory: {path}")


def _detect_and_track(video_path: Path, clip_dir: Path, force: bool = False) -> Path:
    """
    Detect + track one video into ``clip_dir`` (cached, reused unless ``force``).
    Returns the tracks csv path. Imports the tracker lazily so the classify stage stays
    free of the heavy detection dependencies.
    """
    from person_vehicle_interactions.tracker_engine import cache_clip_tracks

    clip_id = video_path.stem
    config = dataclasses.replace(
        DetectionConfig(), target_class_ids=target_class_ids_for_clip(clip_id)
    )
    tracks_path, _ = cache_clip_tracks(
        video_path, clip_id, config, raw_dir=clip_dir, tracks_dir=clip_dir, force=force
    )
    return tracks_path


def process_video_full(
    video_path: Path,
    results_dir: Path = RESULTS_DIR,
    force: bool = False,
    generated_at=None,
    describer: Describer | None = None,
    method: str = "placeholder",
) -> Path:
    """
    Full pipeline for one video: detect + track, then classify. Returns the results JSON.
    ``describer`` (when given) captions each interaction's person + vehicle from the video
    pixels; otherwise the model-free placeholder descriptions are used. ``method`` labels the
    backend in the output filename.
    """
    clip_dir = Path(results_dir) / video_path.stem
    _detect_and_track(video_path, clip_dir, force)
    describe_box = (
        make_describe_box(video_path, describer) if describer is not None else None
    )
    return build_clip_results(
        video_path.stem,
        SHIPPED_THRESHOLDS,
        tracks_dir=clip_dir,
        results_dir=clip_dir,
        generated_at=generated_at,
        describe_box=describe_box,
        method=method,
    )


def build_describer(backend: str) -> Describer | None:
    """
    Load the descriptions backend once and return its describer (``None`` for placeholder).
    ``fast`` = YOLO-World open-vocab (default), ``detailed`` = moondream2 VLM. Heavy imports
    are lazy, so the placeholder path and ``--help`` need neither model.
    """
    if backend == "placeholder":
        return None
    if backend == "detailed":
        from person_vehicle_interactions.captioner import load_captioner
        from person_vehicle_interactions.describers import moondream_describer

        return moondream_describer(load_captioner())
    from person_vehicle_interactions.attribute_tagger import load_tagger
    from person_vehicle_interactions.describers import yoloworld_describer

    return yoloworld_describer(load_tagger())


def process_video_detect(
    video_path: Path, results_dir: Path = RESULTS_DIR, force: bool = False
) -> Path:
    """Detect + track one video into its per-clip folder. Returns the tracks csv path."""
    clip_dir = Path(results_dir) / video_path.stem
    return _detect_and_track(video_path, clip_dir, force)


def _copy_tracks_into(tracks_path: Path, clip_dir: Path) -> None:
    """
    Copy a clip's tracks csv + metadata into ``clip_dir`` (created if needed).
    A no-op for a file already in ``clip_dir``, so re-classifying in place is safe.
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    for source in (tracks_path, tracks_path.with_suffix(".meta.json")):
        if source.exists() and source.parent != clip_dir:
            shutil.copy2(source, clip_dir / source.name)


def process_tracks_classify(
    tracks_path: Path,
    results_dir: Path = RESULTS_DIR,
    generated_at=None,
    describer: Describer | None = None,
    method: str = "placeholder",
    videos_dir: Path = VIDEOS_DIR,
    video: Path | None = None,
) -> Path:
    """
    Classify one cached tracks file into its per-clip folder. Returns the results JSON.
    The source tracks (csv + metadata) are copied into the folder so it holds the same
    tracks-plus-results bundle the full pipeline produces.

    ``describer`` (when given) captions each interaction's person + vehicle from the clip's
    video pixels: the ``video`` override, else ``videos_dir/<clip>.mp4``. If no describer is
    selected, or its video can't be found, the model-free placeholder descriptions are used
    (with a warning in the missing-video case) so this never crashes. ``method`` labels the
    backend in the output filename.
    """
    clip_id = tracks_path.stem
    clip_dir = Path(results_dir) / clip_id
    _copy_tracks_into(tracks_path, clip_dir)
    describe_box = None
    if describer is not None:
        clip_video = video if video is not None else Path(videos_dir) / f"{clip_id}.mp4"
        if clip_video.exists():
            describe_box = make_describe_box(clip_video, describer)
        else:
            print(
                f"warning: no video for '{clip_id}' at {clip_video}; using placeholders"
            )
            method = "placeholder"
    return build_clip_results(
        clip_id,
        SHIPPED_THRESHOLDS,
        tracks_dir=clip_dir,
        results_dir=clip_dir,
        generated_at=generated_at,
        describe_box=describe_box,
        method=method,
    )


def process_all(inputs: list[Path], process: Callable[[Path], Path]) -> int:
    """
    Run ``process`` over each input, reporting each result; return the failure count.
    A failure is reported and skipped so one bad clip never aborts the batch.
    """
    failures = 0
    for input_path in inputs:
        try:
            output_path = process(input_path)
        except (
            Exception
        ) as error:  # noqa: BLE001 - batch boundary: report and continue.
            failures += 1
            print(f"{input_path.name}: ERROR - {error}")
        else:
            print(f"{input_path.name}: -> {output_path}")
    return failures


def _build_parser() -> argparse.ArgumentParser:
    """Build the subcommand parser (run / detect-and-track / classify-tracks)."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="Full pipeline: video -> tracks -> results."
    )
    detect = subparsers.add_parser(
        "detect-and-track", help="Detect + track a video into cached tracks."
    )
    classify = subparsers.add_parser(
        "classify-tracks", help="Classify cached tracks into a results JSON."
    )
    for stage in (run, detect):
        stage.add_argument(
            "path", type=Path, help="A video file or a folder of videos."
        )
        stage.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
        stage.add_argument(
            "--force", action="store_true", help="Re-detect even if tracks are cached."
        )
    run.add_argument(
        "--fast",
        dest="backend",
        action="store_const",
        const="fast",
        default="fast",
        help="Fast YOLO-World open-vocab descriptions (default).",
    )
    run.add_argument(
        "--detailed",
        dest="backend",
        action="store_const",
        const="detailed",
        help="Detailed moondream2 VLM descriptions (slow, higher quality).",
    )
    classify.add_argument(
        "path", type=Path, help="A tracks csv or a folder of cached tracks."
    )
    classify.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    classify.add_argument(
        "--fast",
        dest="backend",
        action="store_const",
        const="fast",
        default="fast",
        help="Fast YOLO-World open-vocab descriptions (default); needs the clip's video.",
    )
    classify.add_argument(
        "--detailed",
        dest="backend",
        action="store_const",
        const="detailed",
        help="Detailed moondream2 VLM descriptions (slow); needs the clip's video.",
    )
    classify.add_argument(
        "--placeholder",
        dest="backend",
        action="store_const",
        const="placeholder",
        help="Model-free placeholder descriptions (offline; no video needed).",
    )
    classify.add_argument(
        "--videos-dir",
        type=Path,
        default=VIDEOS_DIR,
        help="Folder to find each clip's <clip>.mp4 for descriptions (default: Videos/).",
    )
    classify.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Explicit video for a single clip (overrides --videos-dir lookup).",
    )
    return parser


def _describer_for(args: argparse.Namespace) -> tuple[Describer | None, str]:
    """
    Build the chosen descriptions backend, returning ``(describer, method)``.
    Falls back to placeholders (``method="placeholder"``) if the backend can't load — a missing
    model / dependency prints a warning and continues rather than aborting the run.
    """
    backend = getattr(args, "backend", "placeholder")
    try:
        return build_describer(backend), backend
    except Exception as error:  # noqa: BLE001 - a missing model must not abort the run.
        print(
            f"warning: '{backend}' descriptions unavailable ({error}); using placeholders"
        )
        return None, "placeholder"


def _processor_for(args: argparse.Namespace) -> Callable[[Path], Path]:
    """Bind the per-clip processor for the chosen stage to its CLI options."""
    if args.command == "detect-and-track":
        return functools.partial(
            process_video_detect, results_dir=args.results_dir, force=args.force
        )
    if args.command == "classify-tracks":
        describer, method = _describer_for(args)
        return functools.partial(
            process_tracks_classify,
            results_dir=args.results_dir,
            describer=describer,
            method=method,
            videos_dir=args.videos_dir,
            video=args.video,
        )
    describer, method = _describer_for(args)
    return functools.partial(
        process_video_full,
        results_dir=args.results_dir,
        force=args.force,
        describer=describer,
        method=method,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the chosen stage over one input or a folder of inputs."""
    argv = normalize_argv(sys.argv[1:] if argv is None else list(argv))
    args = _build_parser().parse_args(argv)
    suffix = VIDEO_SUFFIX if args.command != "classify-tracks" else TRACKS_SUFFIX
    try:
        inputs = gather_inputs(args.path, suffix)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
    if not inputs:
        print(f"No {suffix} files found in {args.path}")
        return
    failures = process_all(inputs, _processor_for(args))
    print(f"Done: {len(inputs) - failures}/{len(inputs)} clip(s) succeeded.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
