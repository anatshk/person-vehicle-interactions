# Anat Shkolyar - Person-Vehicle Interaction Task

This is a high-level description + setup and run instructions for my results.
Detailed description in the write-up file.

## Description (High-Level)

1. Detection and Tracking - using Ultralytics YOLO11 + the built-in **BoT-SORT** tracker
   (has camera-motion compensation, which helps the PTZ/aerial clips). Pretrained model
   (COCO), using person and car/truck/bus/boat classes.
2. Extract features from the tracked objects: for every (person, vehicle) pair over time,
   the **normalized overlap** (box intersection ÷ person-box area), the **normalized
   center-distance** (÷ vehicle-box diagonal), and the per-frame **detection confidences**.
3. Extract interactions based on metric thresholds (thresholds found using LOSO - see
   write-up file): a candidate is a pair that stays in contact (overlap/distance) for a
   minimum duration above a confidence floor, with short gaps bridged.
4. Get descriptions for the objects involved in the interaction (open-vocab YOLO-World by
   default, or a moondream2 VLM).

Details on the development are in the write-up: [WRITE_UP.md](WRITE_UP.md).

## TLDR - How to run it

### Environment Setup

CPU-only, Python 3.12. Install CPU-only torch first, then the pinned dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

External model weights download automatically on first use (no manual step):

- `yolo11l.pt` — the detector.
- `yolov8s-world.pt` — YOLO-World, for the fast (`--fast`) descriptions.
- `moondream2` (@ 2025-06-21, ~3.7 GB) — for the detailed (`--detailed`) descriptions.

### Run Instructions

Full pipeline (detect + track → classify → describe) on a single clip or a whole folder:

```
python -m scripts.detect_interactions Videos/            # every clip in the folder
python -m scripts.detect_interactions Videos/<clip>.mp4  # a single clip
```

Description backend for the full run: `--fast` (YOLO-World open-vocab, the default),
`--detailed` (moondream2 VLM, slower but richer), or `--placeholder` (model-free labels).
Other flags: `--results-dir`, `--force` (re-detect, ignoring the cache). Use `-h` for the
full help.

The two stages can also be run on their own:

```
python -m scripts.detect_interactions detect-and-track <video|folder>  # video -> tracks
python -m scripts.detect_interactions classify-tracks <tracks|folder>  # tracks -> results
```

Runtimes (CPU): detection + tracking dominates (per-frame inference at `imgsz=1280`) — the
4K aerial clip is by far the slowest, the small CCTV clips are quick. `--detailed` is the
other slow part (the VLM runs ~minutes per crop). A per-stage timing table is in the
write-up (produced by `python -m scripts.time_pipeline`).

## My Outputs

Per-clip results are committed under [`outputs/`](outputs/) (one folder per clip). Running
the pipeline yourself also writes them to the git-ignored `cache/results/<clip_id>/`.

Each clip writes a timestamped JSON, `<clip_id>_interactions_<YYYYMMDDHHMM>.json`. Example
(illustrative):

```json
{
  "clip_id": "NmlzoaDcOuI_6",
  "generated_at": "2026-08-04T20:00:00",
  "interactions": [
    {
      "clip_id": "NmlzoaDcOuI_6",
      "person_id": 2,
      "vehicle_id": 5,
      "start_frame": 120,
      "end_frame": 168,
      "start_seconds": 4.0,
      "end_seconds": 5.6,
      "person": "a man in dark clothing",
      "vehicle": "a silver sedan"
    }
  ]
}
```

Each interaction carries the clip id, the person and vehicle **track ids**, the **frame
range** and matching **time span** (seconds), and a short **description** of each.

To visualize an interaction, `show_interactions` renders annotated contact sheets (person
in green, vehicle in red, drawn across the span plus context frames):

```
python -m scripts.show_interactions [--clip <clip_id>] [--fit-on-all]
```

Sheets are written to `cache/viz/interactions/`.
