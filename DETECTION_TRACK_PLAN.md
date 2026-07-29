# Detection & Tracking Plan (P1 remainder)

Plans the **detect → track → cache** glue that sits on top of the existing
`tracked_data_model.py` (`TrackedBox` / `ClipMetadata` + CSV/JSON cache). Function-level
design + planned tests. TDD: pure logic first (unit), glue via integration tests.

## Design principle: split pure logic from heavy glue

- **Pure modules** — stdlib + our dataclasses only, **no** `cv2` / `ultralytics` imports
  → unit-tested, run in lean CI.
- **Glue module** — imports `cv2` + `ultralytics` → integration-tested; those tests use
  `pytest.importorskip("cv2" / "ultralytics")` so lean CI **skips** them instead of
  failing on the missing heavy deps.

This keeps the pure conversion/filter logic (the interesting part) fully unit-tested and
CI-green, while the model/video plumbing is exercised by integration tests run locally.

## Modules & functions

### `config.py` — constants/defaults
`MODEL_NAME = "yolo11l.pt"`, `IMAGE_SIZE = 1280`, `CONFIDENCE_THRESHOLD = 0.25`,
`IOU_THRESHOLD = 0.7`, `TRACKER_NAME = "botsort.yaml"`, `TARGET_CLASS_IDS = (0, 2)`,
`SEED = 0`, `CACHE_DIR = Path("cache")`.

### `detection_processing.py` — PURE (unit-tested, lean CI)
- `COCO_ID_TO_NAME: dict[int, str] = {0: "person", 2: "car"}`
- `build_tracked_boxes(frame_index, fps, boxes_xyxy, class_ids, track_ids, confidences) -> list[TrackedBox]`
  Convert one frame's raw tracker arrays into `TrackedBox`es: map class id → name,
  compute `time_seconds` via `frame_to_seconds`, **skip detections without a track_id**.
- `filter_detections_by_class(boxes, allowed_classes: set[str]) -> list[TrackedBox]`
  Keep only boxes whose `object_class` is allowed (order-preserving).  ["filter by label"]

### `tracker_engine.py` — GLUE (imports `cv2` + `ultralytics`)
- `read_video_properties(video_path) -> VideoProperties`  ["loading a clip"]
  fps, width, height, frame_count via `cv2.VideoCapture`. `VideoProperties` = small dataclass.
- `track_clip(video_path, fps, config) -> list[TrackedBox]`  ["using the model"]
  Run `model.track(source=..., stream=True, persist=True, classes=TARGET_CLASS_IDS,
  imgsz=..., conf=..., iou=..., tracker=..., verbose=False)`; per-frame Result → extract
  arrays → `build_tracked_boxes` → accumulate; apply `filter_detections_by_class` as a
  safeguard.
- `cache_clip_tracks(video_path, clip_id, config, cache_dir) -> tuple[Path, Path]`  ["save to cache"]
  Orchestrate: `read_video_properties` → `track_clip` → assemble `ClipMetadata` →
  `save_tracks` + `save_metadata`; return the (csv_path, meta_path).

### `scripts/run_tracking.py` — CLI
Run `cache_clip_tracks` for one clip or every file in `Videos/`; write to `cache/`
(gitignored).

## Determinism

Set torch / numpy / random seeds (`SEED`); pin `imgsz` / `conf` / `iou` / tracker; CPU
inference. `yolo11l.pt` auto-downloads on first run (documented as an external asset).

## Planned tests

### `tests/test_detection_processing.py` — unit (lean CI)
1. `test_build_tracked_boxes_basic` — arrays → correct `TrackedBox` fields.
2. `test_build_tracked_boxes_maps_class_ids` — 0 → person, 2 → car.
3. `test_build_tracked_boxes_computes_time_from_fps` — `time_seconds == frame / fps`.
4. `test_build_tracked_boxes_skips_missing_track_id` — `None` id dropped.
5. `test_build_tracked_boxes_empty_input` — empty arrays → `[]`.
6. `test_build_tracked_boxes_asserts_on_unknown_class_id` — id outside `{0, 2}` raises.
7. `test_filter_detections_by_class_keeps_allowed`.
8. `test_filter_detections_by_class_empty_allowed_returns_empty`.
9. `test_filter_detections_by_class_preserves_order`.

### `tests/test_tracker_engine.py` — integration (importorskip; skipped in lean CI)
9. `test_read_video_properties` — write a tiny synthetic mp4 (`cv2.VideoWriter`, known
   fps/size/N frames), assert properties read back correctly.
10. `test_cache_clip_tracks_smoke` — run the pipeline on a tiny synthetic video (no real
    objects → empty/near-empty tracks), assert CSV + meta files are written and reload.
    Needs cv2 + ultralytics (+ weights) → local/manual.

## CI update

- Register a `markers = ["integration: needs cv2/ultralytics/model weights"]` entry in
  `pyproject.toml`.
- Integration tests self-skip via `pytest.importorskip(...)`, so lean CI stays green
  running only the pure unit tests.

## Resolved decisions

- `track_clip` returns a **`list`** (clips are short: ≤600 frames).
- Detections with class ids outside `{0, 2}` → **assert/raise** in `build_tracked_boxes`
  (defensive; the `classes` filter should already prevent it).
- `VideoProperties` vs `ClipMetadata`: **separate `VideoProperties`** (video-intrinsic
  fields only); `ClipMetadata` stays flat/unchanged; `cache_clip_tracks` maps the 4 fields.
