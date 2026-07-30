"""
Render TP / FP / FN interaction contact sheets from the cached tracks + source clips.

For each clip: fit thresholds (LOSO fold-fitted per clip by default, or one set fit on all
clips with ``--fit-on-all``), detect candidate interaction windows, match them against the
ground truth, and render one montage per window — the predicted (person, vehicle) boxes
drawn across the span plus context frames, captioned with the deciding thresholds and the
actual signal values, so each true/false positive and false negative can be eyeballed.

    python -m scripts.show_interactions                 # all clips, LOSO fold thresholds
    python -m scripts.show_interactions --clip gt1125_06
    python -m scripts.show_interactions --fit-on-all    # one threshold set for all clips

cv2 / matplotlib are pulled in lazily via ``visualization`` (integration test skips them).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from person_vehicle_interactions.candidate_detection import (
    detect_candidates,
    pair_series_by_pair,
    Thresholds,
)
from person_vehicle_interactions.config import (
    TRACKS_DIR,
    vehicle_class_names_for_clip,
    VIDEOS_DIR,
    VIZ_DIR,
)
from person_vehicle_interactions.evaluation import classify_windows, LabeledWindow
from person_vehicle_interactions.gt_windows import InteractionWindow, load_gt_windows
from person_vehicle_interactions.interaction_signals import (
    pair_signal_series,
    PERSON_CLASS,
)
from person_vehicle_interactions.loso import all_clips, fold_thresholds_by_clip
from person_vehicle_interactions.track_selection import sample_span_frames
from person_vehicle_interactions.tracked_data_model import (
    frame_to_seconds,
    load_metadata,
    load_tracks,
    TrackedBox,
)
from person_vehicle_interactions.visualization import (
    BoxAnnotation,
    FrameTile,
    PERSON_COLOR_BGR,
    render_frame_sheet,
    VEHICLE_COLOR_BGR,
)
from person_vehicle_interactions.window_summary import summarize_window, WindowSummary
from scripts.run_loso import (
    DEFAULT_GRID,
    GT_CSV,
    load_boxes_by_clip,
    make_fit_thresholds,
)

PathLike = Path | str

# How many frames to sample across a window's span, plus context frames each side.
IN_SAMPLE_COUNT = 5
PAD = 3
COLUMNS = 5


def render_clip_interactions(
    clip_id: str,
    thresholds: Thresholds,
    ground_truth: list[InteractionWindow],
    videos_dir: PathLike = VIDEOS_DIR,
    tracks_dir: PathLike = TRACKS_DIR,
    viz_dir: PathLike = VIZ_DIR,
) -> list[Path]:
    """
    Render one sheet per labeled interaction window (TP/FP/FN) for a clip.
    Returns the paths of the PNGs written.
    """
    boxes = load_tracks(Path(tracks_dir) / f"{clip_id}.csv")
    metadata = load_metadata(Path(tracks_dir) / f"{clip_id}.meta.json")
    vehicle_classes = vehicle_class_names_for_clip(clip_id)
    predicted = detect_candidates(boxes, vehicle_classes, thresholds)
    paths: list[Path] = []
    for labeled in classify_windows(predicted, ground_truth):
        path = _render_window(
            clip_id,
            labeled,
            boxes,
            metadata.fps,
            metadata.frame_count,
            vehicle_classes,
            thresholds,
            Path(videos_dir) / f"{clip_id}.mp4",
            Path(viz_dir) / "interactions",
        )
        if path is not None:
            paths.append(path)
    return paths


def _render_window(
    clip_id: str,
    labeled: LabeledWindow,
    boxes: list[TrackedBox],
    fps: float,
    frame_count: int,
    vehicle_classes: set[str],
    thresholds: Thresholds,
    video_path: Path,
    out_dir: Path,
) -> Path | None:
    """Render one labeled window's sheet; returns its path, or None if nothing to draw."""
    if labeled.label in ("TP", "FP"):
        tiles, pair_label, values, out_name = _predicted_sheet(
            clip_id, labeled, boxes, fps, frame_count
        )
    else:
        tiles, pair_label, values, out_name = _false_negative_sheet(
            clip_id, labeled, boxes, fps, frame_count, vehicle_classes
        )
    if not tiles:
        return None
    span = _label_span(labeled)
    title = _title(clip_id, labeled.label, pair_label, span, fps, thresholds, values)
    return render_frame_sheet(video_path, tiles, title, out_dir / out_name, COLUMNS)


def _predicted_sheet(
    clip_id: str,
    labeled: LabeledWindow,
    boxes: list[TrackedBox],
    fps: float,
    frame_count: int,
) -> tuple[list[FrameTile], str, str, str]:
    """Tiles/labels for a TP or FP: the predicted person+vehicle across the span."""
    window = labeled.predicted
    person_id, vehicle_id = window.person_id, window.vehicle_id
    start, end = window.start_frame, window.end_frame
    person_by_frame = {box.frame: box for box in boxes if box.track_id == person_id}
    vehicle_by_frame = {box.frame: box for box in boxes if box.track_id == vehicle_id}
    signal_by_frame = {
        signal.frame: signal
        for signal in pair_signal_series(boxes, person_id, vehicle_id)
    }
    tiles: list[FrameTile] = []
    for frame, position in _span_frames(start, end, frame_count):
        annotations: list[BoxAnnotation] = []
        if frame in person_by_frame:
            annotations.append(
                _annotation(person_by_frame[frame], PERSON_COLOR_BGR, f"p{person_id}")
            )
        if frame in vehicle_by_frame:
            annotations.append(
                _annotation(
                    vehicle_by_frame[frame], VEHICLE_COLOR_BGR, f"v{vehicle_id}"
                )
            )
        signal = signal_by_frame.get(frame)
        metric = f"ov={signal.overlap:.2f} d={signal.distance:.2f}" if signal else "—"
        tiles.append(FrameTile(frame, f"f{frame} {position}\n{metric}", annotations))
    summary = summarize_window(
        pair_signal_series(boxes, person_id, vehicle_id), start, end, fps
    )
    values = f"peakOv={summary.peak_overlap:.2f} minDist={summary.min_distance:.2f}"
    pair_label = f"p{person_id}×v{vehicle_id}"
    out_name = f"{clip_id}_{person_id}_{vehicle_id}_{start}-{end}_{labeled.label}.png"
    return tiles, pair_label, values, out_name


def _false_negative_sheet(
    clip_id: str,
    labeled: LabeledWindow,
    boxes: list[TrackedBox],
    fps: float,
    frame_count: int,
    vehicle_classes: set[str],
) -> tuple[list[FrameTile], str, str, str]:
    """Tiles/labels for an FN: the GT span with every person+vehicle box drawn."""
    gt = labeled.ground_truth
    start, end = gt.start_frame, gt.end_frame
    person_by_frame: dict[int, list[TrackedBox]] = defaultdict(list)
    vehicle_by_frame: dict[int, list[TrackedBox]] = defaultdict(list)
    for box in boxes:
        if box.object_class == PERSON_CLASS:
            person_by_frame[box.frame].append(box)
        elif box.object_class in vehicle_classes:
            vehicle_by_frame[box.frame].append(box)
    tiles: list[FrameTile] = []
    for frame, position in _span_frames(start, end, frame_count):
        annotations = [
            _annotation(box, PERSON_COLOR_BGR, f"p{box.track_id}")
            for box in person_by_frame.get(frame, [])
        ] + [
            _annotation(box, VEHICLE_COLOR_BGR, f"v{box.track_id}")
            for box in vehicle_by_frame.get(frame, [])
        ]
        tiles.append(FrameTile(frame, f"f{frame} {position}", annotations))
    values = _false_negative_values(boxes, vehicle_classes, start, end, fps)
    out_name = f"{clip_id}_gt{gt.interaction_id}_{start}-{end}_FN.png"
    return tiles, "gt", values, out_name


def _false_negative_values(
    boxes: list[TrackedBox],
    vehicle_classes: set[str],
    start: int,
    end: int,
    fps: float,
) -> str:
    """Explain an FN: no person detected, or the closest sub-threshold pair's values."""
    person_present = any(
        box.object_class == PERSON_CLASS and start <= box.frame <= end for box in boxes
    )
    if not person_present:
        return "person not detected in span"
    best = _closest_pair_over_span(boxes, vehicle_classes, start, end, fps)
    if best is None:
        return "no candidate person+vehicle pair in span"
    (person_id, vehicle_id), summary = best
    return (
        f"sub-threshold: peakOv={summary.peak_overlap:.2f} "
        f"minDist={summary.min_distance:.2f} (best p{person_id}×v{vehicle_id})"
    )


def _closest_pair_over_span(
    boxes: list[TrackedBox],
    vehicle_classes: set[str],
    start: int,
    end: int,
    fps: float,
) -> tuple[tuple[int, int], WindowSummary] | None:
    """Return the pair with the highest peak overlap over the span, if any share it."""
    best: tuple[tuple[int, int], WindowSummary] | None = None
    for pair, series in pair_series_by_pair(boxes, vehicle_classes).items():
        if not any(start <= signal.frame <= end for signal in series):
            continue
        summary = summarize_window(series, start, end, fps)
        if best is None or summary.peak_overlap > best[1].peak_overlap:
            best = (pair, summary)
    return best


def _span_frames(start: int, end: int, frame_count: int) -> list[tuple[int, str]]:
    """Sampled span frames (with pre/post context) clamped to the clip's frame range."""
    return [
        (frame, position)
        for frame, position in sample_span_frames(start, end, IN_SAMPLE_COUNT, PAD)
        if 0 <= frame < frame_count
    ]


def _annotation(
    box: TrackedBox, color: tuple[int, int, int], label: str
) -> BoxAnnotation:
    """Build a BoxAnnotation from a tracked box."""
    return BoxAnnotation((box.x1, box.y1, box.x2, box.y2), color, label)


def _label_span(labeled: LabeledWindow) -> tuple[int, int]:
    """The (start, end) frame span a labeled window is titled by."""
    window = labeled.predicted or labeled.ground_truth
    return window.start_frame, window.end_frame


def _title(
    clip_id: str,
    label: str,
    pair_label: str,
    span: tuple[int, int],
    fps: float,
    thresholds: Thresholds,
    values: str,
) -> str:
    """Build the sheet's suptitle: clip, pair, span, label, thresholds, and values."""
    start, end = span
    duration = end - start + 1
    seconds = frame_to_seconds(duration, fps)
    return (
        f"{clip_id} | {pair_label} | f{start}-{end} ({duration}f/{seconds:.1f}s) | "
        f"{label} | thr: {_threshold_caption(thresholds)} | {values}"
    )


def _threshold_caption(thresholds: Thresholds) -> str:
    """The deciding thresholds as a compact caption fragment."""
    return (
        f"ov≥{thresholds.min_overlap:g} dist≤{thresholds.max_distance:g} "
        f"dur≥{thresholds.min_duration_frames} conf≥{thresholds.min_confidence:g} "
        f"gap≤{thresholds.max_gap_frames}"
    )


def _thresholds_by_clip(fit_on_all: bool) -> dict[str, Thresholds]:
    """Per-clip thresholds: LOSO fold-fitted by default, or one set fit on all clips."""
    ground_truth_by_clip = load_gt_windows(GT_CSV)
    boxes_by_clip = load_boxes_by_clip(all_clips(), TRACKS_DIR)
    fit = make_fit_thresholds(boxes_by_clip, ground_truth_by_clip, DEFAULT_GRID)
    if fit_on_all:
        shared = fit(all_clips())
        return {clip_id: shared for clip_id in all_clips()}
    return fold_thresholds_by_clip(fit)


def main() -> None:
    """Render interaction sheets for one clip or all clips (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default=None, help="Render only this clip id.")
    parser.add_argument(
        "--fit-on-all",
        action="store_true",
        help="Use one threshold set fit on all clips (shippable view) instead of LOSO.",
    )
    parser.add_argument("--gt-csv", type=Path, default=GT_CSV)
    args = parser.parse_args()

    ground_truth_by_clip = load_gt_windows(args.gt_csv)
    thresholds_by_clip = _thresholds_by_clip(args.fit_on_all)
    clip_ids = (args.clip,) if args.clip else all_clips()
    for clip_id in clip_ids:
        paths = render_clip_interactions(
            clip_id,
            thresholds_by_clip[clip_id],
            ground_truth_by_clip.get(clip_id, []),
        )
        print(f"{clip_id}: {len(paths)} sheet(s)")


if __name__ == "__main__":
    main()
