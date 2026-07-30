"""
Glue: run detection + tracking over a clip and cache raw output + processed tracks.

cv2 and ultralytics are imported lazily inside the functions, so this module (and its
integration tests) stays importable where the heavy deps are absent — those tests skip
via ``pytest.importorskip``.
"""

from __future__ import annotations

from pathlib import Path
import random

from person_vehicle_interactions.clip_assembly import (
    build_clip_metadata,
    VideoProperties,
)
from person_vehicle_interactions.config import DetectionConfig, RAW_DIR, TRACKS_DIR
from person_vehicle_interactions.raw_detections import tracks_from_raw, write_raw_frame
from person_vehicle_interactions.tracked_data_model import save_metadata, save_tracks

PathLike = Path | str


def set_seeds(seed: int) -> None:
    """Seed Python, numpy, and torch RNGs for deterministic inference."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_video_properties(video_path: PathLike) -> VideoProperties:
    """Read fps, dimensions, and frame count from a video via cv2."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        return VideoProperties(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            frame_height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def run_inference(
    video_path: PathLike, config: DetectionConfig, raw_path: PathLike
) -> None:
    """
    Run YOLO tracking over a clip, appending per-frame raw detections to ``raw_path``.

    Writes to a ``.partial`` file and renames it on success, so a final raw file always
    represents a fully-processed clip.
    """
    from ultralytics import YOLO

    set_seeds(config.seed)
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = raw_path.with_name(raw_path.name + ".partial")

    target_classes = (
        None if config.target_class_ids is None else list(config.target_class_ids)
    )
    model = YOLO(config.model_name)
    results = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        classes=target_classes,
        imgsz=config.image_size,
        conf=config.confidence_threshold,
        iou=config.iou_threshold,
        tracker=config.tracker_name,
        verbose=False,
    )
    with partial_path.open("w", encoding="utf-8") as raw_file:
        for frame_index, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                write_raw_frame(raw_file, frame_index, [], [], [], [])
                continue
            write_raw_frame(
                raw_file,
                frame_index,
                boxes.xyxy.tolist(),
                [int(class_id) for class_id in boxes.cls.tolist()],
                [int(track_id) for track_id in boxes.id.tolist()],
                [float(confidence) for confidence in boxes.conf.tolist()],
            )
    partial_path.replace(raw_path)


def cache_clip_tracks(
    video_path: PathLike,
    clip_id: str,
    config: DetectionConfig,
    raw_dir: PathLike = RAW_DIR,
    tracks_dir: PathLike = TRACKS_DIR,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Cache a clip's tracks, reusing checkpoints.

    Skips entirely if the tracks csv + meta already exist (unless ``force``); reuses an
    existing raw file (re-derives tracks without re-inference). Returns
    ``(tracks_csv_path, meta_json_path)``.
    """
    raw_path = Path(raw_dir) / f"{clip_id}.jsonl"
    tracks_path = Path(tracks_dir) / f"{clip_id}.csv"
    meta_path = Path(tracks_dir) / f"{clip_id}.meta.json"

    if not force and tracks_path.exists() and meta_path.exists():
        return tracks_path, meta_path

    video_properties = read_video_properties(video_path)
    if force or not raw_path.exists():
        run_inference(video_path, config, raw_path)

    tracked_boxes = tracks_from_raw(raw_path, video_properties.fps, config)
    metadata = build_clip_metadata(clip_id, video_properties, config)

    Path(tracks_dir).mkdir(parents=True, exist_ok=True)
    save_tracks(tracked_boxes, tracks_path)
    save_metadata(metadata, meta_path)
    return tracks_path, meta_path
