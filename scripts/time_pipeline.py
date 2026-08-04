"""
Time each pipeline stage per clip and print a Markdown table for the write-up.

For every video in ``Videos/`` (or a folder / single file passed on the command line) this
runs the real stages and records wall-clock seconds against the clip's video properties:

  detect+track (fused YOLO ``model.track`` inference) -> process (raw JSONL -> TrackedBoxes)
  -> candidate-detection -> evaluation (vs ground truth, when the clip has any).

Detection and tracking are one fused ``model.track()`` call, so they are timed together as
``detect_track_s`` (the decision recorded in the write-up). Inference is always run fresh
(the track cache is bypassed) so the number reflects real compute, not a cache hit; each run
reloads the YOLO weights, so a one-off model-load time is printed above the table to subtract.

    python -m scripts.time_pipeline [path] [--gt-csv ground_truth/interactions.csv]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import tempfile

from person_vehicle_interactions.config import (
    DetectionConfig,
    SHIPPED_THRESHOLDS,
    target_class_ids_for_clip,
    vehicle_class_names_for_clip,
    VIDEOS_DIR,
)
from person_vehicle_interactions.pipeline_timing import (
    ClipTiming,
    format_timings_table,
    time_call,
)

GT_CSV = Path("ground_truth/interactions.csv")
VIDEO_SUFFIX = ".mp4"


def time_clip(video_path: Path, ground_truth_by_clip: dict) -> ClipTiming:
    """
    Run and time every stage for one video, returning its ``ClipTiming``.
    Heavy deps (cv2 / ultralytics) are imported lazily so importing this module stays cheap.
    """
    from person_vehicle_interactions.candidate_detection import detect_candidates
    from person_vehicle_interactions.clip_assembly import build_clip_metadata
    from person_vehicle_interactions.evaluation import evaluate_clip
    from person_vehicle_interactions.raw_detections import tracks_from_raw
    from person_vehicle_interactions.tracker_engine import (
        read_video_properties,
        run_inference,
    )

    clip_id = video_path.stem
    config = dataclasses.replace(
        DetectionConfig(), target_class_ids=target_class_ids_for_clip(clip_id)
    )
    properties = read_video_properties(video_path)
    size_mb = video_path.stat().st_size / 1_000_000

    with tempfile.TemporaryDirectory() as scratch:
        raw_path = Path(scratch) / f"{clip_id}.jsonl"
        _, detect_track_seconds = time_call(
            lambda: run_inference(video_path, config, raw_path)
        )
        boxes, process_seconds = time_call(
            lambda: tracks_from_raw(raw_path, properties.fps, config)
        )
        # Metadata is part of the processing stage; build it so timing is comparable.
        build_clip_metadata(clip_id, properties, config)

    vehicle_classes = vehicle_class_names_for_clip(clip_id)
    windows, candidate_seconds = time_call(
        lambda: detect_candidates(boxes, vehicle_classes, SHIPPED_THRESHOLDS)
    )

    eval_seconds: float | None = None
    ground_truth = ground_truth_by_clip.get(clip_id)
    if ground_truth is not None:
        _, eval_seconds = time_call(lambda: evaluate_clip(windows, ground_truth))

    return ClipTiming(
        clip_id=clip_id,
        width=properties.frame_width,
        height=properties.frame_height,
        fps=properties.fps,
        frame_count=properties.frame_count,
        size_mb=size_mb,
        detect_track_seconds=detect_track_seconds,
        process_seconds=process_seconds,
        candidate_seconds=candidate_seconds,
        eval_seconds=eval_seconds,
    )


def measure_model_load_seconds() -> float:
    """Time a single fresh YOLO weight load — the fixed per-clip inference overhead."""
    from ultralytics import YOLO

    _, seconds = time_call(lambda: YOLO(DetectionConfig().model_name))
    return seconds


def gather_videos(path: Path) -> list[Path]:
    """Resolve a path to the videos to time: a folder's mp4s (sorted) or the single file."""
    if path.is_dir():
        return sorted(path.glob(f"*{VIDEO_SUFFIX}"))
    return [path]


def main(argv: list[str] | None = None) -> None:
    """Time the pipeline over each input video and print the model-load line + table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=VIDEOS_DIR)
    parser.add_argument("--gt-csv", type=Path, default=GT_CSV)
    args = parser.parse_args(argv)

    from person_vehicle_interactions.gt_windows import load_gt_windows

    ground_truth_by_clip = (
        load_gt_windows(args.gt_csv) if Path(args.gt_csv).exists() else {}
    )
    videos = gather_videos(args.path)
    if not videos:
        print(f"No {VIDEO_SUFFIX} files found in {args.path}")
        return

    print(
        f"model_load_s (one-off, included in each detect_track_s): "
        f"{measure_model_load_seconds():.3f}"
    )
    rows = [time_clip(video_path, ground_truth_by_clip) for video_path in videos]
    print(format_timings_table(rows))


if __name__ == "__main__":
    main()
