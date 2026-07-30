"""
Glue: render track-inspection contact sheets from the cached tracks + source clips.

``show_objects`` gives a per-clip gallery (one tile per tracked object at its largest
box, labeled by track id) to eyeball what was tracked. ``show_track`` follows a single
object over time in fixed 0.25-frame crops with its box drawn, to judge track quality.

cv2 and matplotlib are imported lazily inside the functions, so the module stays
importable where the heavy deps are absent (integration tests skip via importorskip).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from person_vehicle_interactions.config import TRACKS_DIR, VIDEOS_DIR, VIZ_DIR
from person_vehicle_interactions.track_selection import (
    crop_window,
    highest_confidence_box_per_track,
    sample_track_boxes,
)
from person_vehicle_interactions.tracked_data_model import (
    load_metadata,
    load_tracks,
    TrackedBox,
)

PathLike = Path | str

BOX_COLOR_BGR = (0, 255, 0)
PERSON_COLOR_BGR = (0, 255, 0)  # green
VEHICLE_COLOR_BGR = (0, 0, 255)  # red


@dataclasses.dataclass
class BoxAnnotation:
    """One box to draw on a frame: pixel corners, BGR color, and a short text label."""

    box_xyxy: tuple[float, float, float, float]
    color_bgr: tuple[int, int, int]
    label: str


@dataclasses.dataclass
class FrameTile:
    """One montage tile: a source frame index, its caption, and boxes to draw on it."""

    frame_index: int
    caption: str
    annotations: list[BoxAnnotation]


def show_objects(
    clip_id: str,
    videos_dir: PathLike = VIDEOS_DIR,
    tracks_dir: PathLike = TRACKS_DIR,
    viz_dir: PathLike = VIZ_DIR,
    columns: int = 4,
) -> Path:
    """
    Render one tile per tracked object (its highest-confidence-frame crop), labeled by
    track id.

    Returns the path of the written PNG (``<viz_dir>/objects/<clip_id>.png``).
    """
    boxes = load_tracks(Path(tracks_dir) / f"{clip_id}.csv")
    representative_boxes = highest_confidence_box_per_track(boxes)
    if not representative_boxes:
        raise ValueError(f"No tracked objects for clip {clip_id!r}.")

    capture = _open_capture(Path(videos_dir) / f"{clip_id}.mp4")
    tiles = []
    titles = []
    try:
        for box in representative_boxes:
            frame = _read_frame(capture, box.frame)
            tiles.append(_tight_crop(frame, box))
            titles.append(f"id{box.track_id} {box.object_class} {box.confidence:.2f}")
    finally:
        capture.release()

    out_path = Path(viz_dir) / "objects" / f"{clip_id}.png"
    suptitle = f"{clip_id} — {len(representative_boxes)} tracked objects"
    return _montage(tiles, titles, out_path, columns, suptitle)


def show_track(
    clip_id: str,
    track_id: int,
    step: int,
    videos_dir: PathLike = VIDEOS_DIR,
    tracks_dir: PathLike = TRACKS_DIR,
    viz_dir: PathLike = VIZ_DIR,
    fraction: float = 0.25,
    columns: int = 5,
) -> Path:
    """
    Follow one object over time: every ``step``-th frame as a 0.25-frame crop, box drawn.

    Returns the path of the written PNG
    (``<viz_dir>/tracks/<clip_id>_track<track_id>.png``).
    """
    metadata = load_metadata(Path(tracks_dir) / f"{clip_id}.meta.json")
    boxes = load_tracks(Path(tracks_dir) / f"{clip_id}.csv")
    sampled_boxes = sample_track_boxes(boxes, track_id, step)
    if not sampled_boxes:
        raise ValueError(f"No boxes for track {track_id} in clip {clip_id!r}.")

    capture = _open_capture(Path(videos_dir) / f"{clip_id}.mp4")
    tiles = []
    titles = []
    try:
        for box in sampled_boxes:
            frame = _read_frame(capture, box.frame)
            center_x = (box.x1 + box.x2) / 2
            center_y = (box.y1 + box.y2) / 2
            window = crop_window(
                center_x,
                center_y,
                metadata.frame_width,
                metadata.frame_height,
                fraction,
            )
            crop = _black_pad_crop(frame, *window)
            _draw_box(
                crop,
                (
                    box.x1 - window[0],
                    box.y1 - window[1],
                    box.x2 - window[0],
                    box.y2 - window[1],
                ),
            )
            tiles.append(crop)
            # Frame index out of the clip's total frame count, e.g. "f5/45".
            titles.append(f"f{box.frame}/{metadata.frame_count}")
    finally:
        capture.release()

    out_path = Path(viz_dir) / "tracks" / f"{clip_id}_track{track_id}.png"
    suptitle = f"{clip_id} — track {track_id} ({sampled_boxes[0].object_class})"
    return _montage(tiles, titles, out_path, columns, suptitle)


def render_frame_sheet(
    video_path: PathLike,
    tiles: list[FrameTile],
    title: str,
    out_path: PathLike,
    columns: int = 5,
) -> Path:
    """
    Render annotated frames as a titled montage and save it as a PNG; return the path.
    Each tile's frame is read from ``video_path``, its boxes are drawn on it, and the tile
    is captioned. Frame indices must be within the clip's frame count.
    """
    capture = _open_capture(video_path)
    frames = []
    captions = []
    try:
        for tile in tiles:
            frame = _read_frame(capture, tile.frame_index)
            for annotation in tile.annotations:
                _draw_box(
                    frame,
                    annotation.box_xyxy,
                    color=annotation.color_bgr,
                    label=annotation.label,
                )
            frames.append(frame)
            captions.append(tile.caption)
    finally:
        capture.release()
    return _montage(frames, captions, out_path, columns, title)


def _open_capture(video_path: PathLike) -> Any:
    """Open a cv2 VideoCapture, raising FileNotFoundError if the clip won't open."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path!r}.")
    return capture


def _read_frame(capture: Any, frame_index: int) -> Any:
    """Seek to and read a single frame (BGR ndarray) from an open capture."""
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    read_ok, frame = capture.read()
    if not read_ok:
        raise ValueError(f"Could not read frame {frame_index}.")
    return frame


def _tight_crop(frame: Any, box: TrackedBox) -> Any:
    """Crop a frame to a box, clamped to the frame bounds."""
    height, width = frame.shape[:2]
    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(width, int(round(box.x2)))
    y2 = min(height, int(round(box.y2)))
    return frame[y1:y2, x1:x2]


def _black_pad_crop(frame: Any, x1: int, y1: int, x2: int, y2: int) -> Any:
    """
    Crop a fixed window from a frame, padding out-of-bounds area with black.

    The window keeps its full ``(x2 - x1, y2 - y1)`` size regardless of frame edges.
    """
    import numpy as np

    canvas = np.zeros((y2 - y1, x2 - x1, frame.shape[2]), dtype=frame.dtype)
    source_x1 = max(x1, 0)
    source_y1 = max(y1, 0)
    source_x2 = min(x2, frame.shape[1])
    source_y2 = min(y2, frame.shape[0])
    if source_x2 > source_x1 and source_y2 > source_y1:
        canvas[source_y1 - y1 : source_y2 - y1, source_x1 - x1 : source_x2 - x1] = (
            frame[source_y1:source_y2, source_x1:source_x2]
        )
    return canvas


def _draw_box(
    frame: Any,
    box_xyxy: tuple[float, float, float, float],
    color: tuple[int, int, int] = BOX_COLOR_BGR,
    label: str | None = None,
) -> None:
    """Draw a (optionally labeled) rectangle for one box, relative to ``frame``, in place."""
    import cv2

    x1, y1, x2, y2 = (int(round(coordinate)) for coordinate in box_xyxy)
    thickness = max(1, round(min(frame.shape[:2]) / 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        font_scale = max(0.4, min(frame.shape[:2]) / 600)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _montage(
    tiles: list[Any],
    titles: list[str],
    out_path: PathLike,
    columns: int,
    suptitle: str,
) -> Path:
    """Lay out BGR tiles in a titled grid and save it as a PNG; return the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    row_count = (len(tiles) + columns - 1) // columns
    figure, axes = plt.subplots(
        row_count, columns, figsize=(columns * 3, row_count * 3)
    )
    flat_axes = np.array(axes).reshape(-1)
    for axis, tile, title in zip(flat_axes, tiles, titles):
        axis.imshow(tile[..., ::-1])  # BGR -> RGB for matplotlib.
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    for axis in flat_axes[len(tiles) :]:
        axis.axis("off")
    figure.suptitle(suptitle)
    figure.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return out_path
