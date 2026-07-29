# Detection & Tracking Plan (P1 remainder)

Plans the **detect → track → cache** glue on top of `detection_processing.py` (merged)
and `tracked_data_model.py`. **Session goal:** an overnight batch script that populates the
cache for all clips, **resumable via checkpoints**. TDD: pure logic unit-tested, glue via
integration tests.

## Design principle: split pure logic from heavy glue

- **Pure modules** — stdlib + our dataclasses only, **no** `cv2` / `ultralytics` → unit-tested
  in lean CI.
- **Glue module** — imports `cv2` + `ultralytics` → integration-tested; tests use
  `pytest.importorskip(...)` so lean CI **skips** them.

## Checkpointing & cache layout (session goal)

Two-level cache so an interrupted overnight run never redoes expensive work:

```
cache/
  raw/<clip_id>.jsonl         # per-frame MODEL output (xyxy/cls/id/conf), appended per frame
  tracks/<clip_id>.csv        # processed TrackedBoxes
  tracks/<clip_id>.meta.json  # ClipMetadata
```

- **Save point 1 — raw model output** (`raw/<clip_id>.jsonl`): appended **per frame** during
  inference, finalized on completion (temp → rename). The expensive artifact.
- **Save point 2 — processed tracks** (`tracks/`): derived from raw, written last (atomic).
- **Resume logic** (batch script): tracks exist → **skip**; raw exists but tracks missing →
  **re-derive tracks from raw, no re-inference**; neither → full run.
- **Limitation:** `model.track()` is stateful across frames, so a clip killed *mid-inference*
  re-runs (no mid-stream resume). Clips are short, so the clip is the atomic unit.

## Modules & functions

### `config.py` — DetectionConfig + paths
`DetectionConfig` frozen dataclass (`model_name="yolo11l.pt"`, `image_size=1280`,
`confidence_threshold=0.25`, `iou_threshold=0.7`, `tracker_name="botsort.yaml"`,
`target_class_ids=(0, 2)`, `seed=0`); `CACHE_DIR`, `RAW_DIR`, `TRACKS_DIR`.

### `detection_processing.py` — PURE ✅ (merged)
`build_tracked_boxes`, `filter_detections_by_class`, `COCO_ID_TO_NAME`.

### `clip_assembly.py` — NEW, PURE (unit-tested)
- `VideoProperties` dataclass (`fps, frame_width, frame_height, frame_count`).
- `build_clip_metadata(clip_id, video_properties, config) -> ClipMetadata`.

### `raw_detections.py` — NEW, PURE (unit-tested)
- `write_raw_frame(file, frame_index, boxes_xyxy, class_ids, track_ids, confidences)` —
  append one JSONL line.
- `read_raw(raw_path)` — iterate frame records.
- `tracks_from_raw(raw_path, fps, config) -> list[TrackedBox]` — read JSONL →
  `build_tracked_boxes` per frame → `filter_detections_by_class`. The "reprocess without
  re-inference" path.

### `tracker_engine.py` — NEW, GLUE (`cv2` + `ultralytics`)
- `set_seeds(seed)` — torch / numpy / random.
- `read_video_properties(video_path) -> VideoProperties` — via `cv2.VideoCapture`.
- `run_inference(video_path, config, raw_path)` — fresh `YOLO(config.model_name)` **per call**;
  `.track(stream=True, persist=True, classes=..., imgsz=..., conf=..., iou=..., tracker=...,
  verbose=False)`; per frame → extract arrays → `write_raw_frame` (append to
  `raw_path.partial`, rename to final on completion).
- `cache_clip_tracks(video_path, clip_id, config) -> tuple[Path, Path]` — **checkpoint
  orchestration**: skip if tracks exist; else ensure raw (run_inference if missing) →
  `tracks_from_raw` → `build_clip_metadata` → `save_tracks` / `save_metadata`.

### `scripts/build_cache.py` — NEW (the overnight script)
Iterate `Videos/`, call `cache_clip_tracks` per clip with checkpoint-skip, log progress
(clip, frames, elapsed). `--force` to rebuild.

## Determinism
Seeds via `set_seeds`; pinned imgsz/conf/iou/tracker; CPU. `yolo11l.pt` auto-downloads
(documented as an external asset).

## Planned tests

### Pure (lean CI)
- `test_clip_assembly.py` — `build_clip_metadata` maps VideoProperties + config → `ClipMetadata`.
- `test_raw_detections.py` — raw JSONL write→read round-trip; `tracks_from_raw` builds correct
  `TrackedBox`es (skips id-less, filters classes).

### Integration (`importorskip`, `@pytest.mark.integration`, skipped in lean CI)
- `test_read_video_properties` — synthetic mp4 (generated once into `tests/videos/`,
  gitignored, regenerated if missing) → correct props.
- `test_cache_clip_tracks_smoke` — pipeline on the synthetic clip (empty tracks OK); assert
  raw + tracks + meta written and reload; a re-run **skips** (checkpoint).

## CI / repo
- Register `markers = ["integration: needs cv2/ultralytics/model weights"]` in `pyproject.toml`.
- `.gitignore`: `cache/`, `tests/videos/`.

## Resolved decisions
- Tracks/`track_clip` return a **list** (clips short).
- Unknown class id → **raise** in `build_tracked_boxes`.
- **Separate `VideoProperties`** in a **new pure module** (`clip_assembly.py`).
- Model loaded **per call** (fresh `YOLO(...)`).
- Integration video **synthetic**, created once into `tests/videos/` (gitignored).
- Overlay **deferred** — the cache holds box + class + track_id, so overlays are recoverable.
- **Two-level checkpoint cache** (raw model output + processed tracks) for resumable overnight runs.
