# Anat Shkolyar - Person-Vehicle Interaction Task

This is a high-level description + setup and run instructions for my results.
The write-up comes in two forms: a one-page [executive summary](WRITE_UP.md) and the full
[detailed write-up](docs/DETAILED_WRITE_UP.md).

## Deliverables
- Code - this is the repository - [https://github.com/anatshk/person-vehicle-interactions](https://github.com/anatshk/person-vehicle-interactions)
- Outputs - in the repo, under the [outputs](https://github.com/anatshk/person-vehicle-interactions/tree/main/outputs) folder. Each clip has a folder that contains:
  - `<clip_id>.csv` file - these are the cached detections that allow us to skip the long-running detect-track step.
  - `<clip_id>.meta.json` file - properties of the video along with detect-track parameters used.
  - `<clip_id>_interactions_fast_<YYYYMMDDHHMM>.json` - required results, descriptions made by a fast but unreliable model.
  - `<clip_id>_interactions_detailed_<YYYYMMDDHHMM>.json` - same as above, however the model provides reliable descriptions at a significantly longer runtime cost.
- Visualizations -
  - [`outputs/sheets`](https://github.com/anatshk/person-vehicle-interactions/tree/main/outputs/sheets) contains images of the interactions found per clip.
  - [`docs/images`](https://github.com/anatshk/person-vehicle-interactions/tree/main/docs/images) has a few examples used in the [detailed write-up](docs/DETAILED_WRITE_UP.md)
- [WRITE_UP.md](WRITE_UP.md) - one-page executive summary (approach, results, assumptions, limitations, next steps).
- [docs/DETAILED_WRITE_UP.md](docs/DETAILED_WRITE_UP.md) - the full write-up: development process, trade-offs, decisions, and per-clip analysis.

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

Details on the development are in the write-up: the [executive summary](WRITE_UP.md) or the
full [detailed write-up](docs/DETAILED_WRITE_UP.md).

## TLDR - How to run it

### Environment Setup

Python 3.12. Development was **CPU-only**, so that is the tested path - the GPU install below
is provided for convenience but is **not verified here**, so no promises it behaves identically.
Install torch first, then the pinned dependencies:

```
python -m venv .venv
source .venv/bin/activate

# CPU-only (tested):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# --- or --- GPU / CUDA (untested here):
# pip install torch torchvision            # default CUDA build from PyPI

pip install -r requirements.txt
```

External model weights download automatically on first use (no manual step):

- `yolo11l.pt` - the detector. Ultralytics (**AGPL-3.0**; commercial use needs a license).
- `yolov8s-world.pt` - YOLO-World, for the fast (`--fast`) descriptions. Ultralytics (**AGPL-3.0**).
- `moondream2` (@ 2025-06-21, ~3.7 GB) - for the detailed (`--detailed`) descriptions (**Apache-2.0**).

No external services or network APIs are used at inference time - everything runs locally.

### Run Instructions

Full pipeline (detect + track → classify → describe) on a single clip or a whole folder:

```
python -m scripts.detect_interactions path/to/video/folder    # every .mp4 in the folder
python -m scripts.detect_interactions path/to/video/clip.mp4  # a single clip
```

Only `.mp4` inputs are supported for now; any other file (or a single non-mp4 path) exits
gracefully with a clear message instead of crashing.

Description backend for the full run: `--fast` (YOLO-World open-vocab, the default) or
`--detailed` (moondream2 VLM, slower but richer). If the chosen backend cannot load, it falls
back to model-free placeholder labels with a warning. Other flags: `--results-dir`, `--force`
(re-detect, ignoring the cache).

Full `run` help (`python -m scripts.detect_interactions run -h`):

```
usage: detect_interactions.py run [-h] [--results-dir RESULTS_DIR] [--force]
                                  [--fast] [--detailed]
                                  path

positional arguments:
  path                  A video file or a folder of videos.

options:
  -h, --help            show this help message and exit
  --results-dir RESULTS_DIR
  --force               Re-detect even if tracks are cached.
  --fast                Fast YOLO-World open-vocab descriptions (default).
  --detailed            Detailed moondream2 VLM descriptions (slow, higher
                        quality).
```

The two stages can also be run on their own:

```
python -m scripts.detect_interactions detect-and-track <video|folder>  # video -> tracks
python -m scripts.detect_interactions classify-tracks <tracks|folder>  # tracks -> results
```

`classify-tracks` takes the same `--fast` (default) / `--detailed` description backends as the
full run, so you can regenerate real descriptions from the committed `outputs/` tracks without
re-running detection. It finds each clip's video at `Videos/<clip>.mp4` (override with
`--videos-dir` / `--video`); add `--placeholder` to stay fully offline (no video, model-free
labels).

Runtimes (CPU): detection + tracking dominates (per-frame inference at `imgsz=1280`) - the
4K aerial clip is by far the slowest, the small CCTV clips are quick. `--detailed` is the
other slow part (the VLM runs ~minutes per crop). A per-stage timing table + analysis is in
[`docs/timing_findings.txt`](docs/timing_findings.txt) (produced by `python -m scripts.time_pipeline`).

## My Outputs

Per-clip results are committed under [`outputs/`](outputs/) (one folder per clip). Running
the pipeline yourself also writes them to the git-ignored `cache/results/<clip_id>/`.

The committed outputs hold **15 interactions across the 8 clips** (both `--fast` and
`--detailed` produce the same windows; only the descriptions differ):

| clip | interactions |
|---|---|
| `1THkHYIQ_bY_0` | 2 |
| `gt1125_06` | 5 |
| `HIu4lM4B8hA_1` | 0 (expected - see the CCTV detection-recall limitation in the write-up) |
| `iMGR_0AG3a8_2_3` | 1 |
| `mKzCQKTHizw_0` | 2 |
| `mKzCQKTHizw_1` | 1 |
| `NmlzoaDcOuI_1` | 3 |
| `NmlzoaDcOuI_6` | 1 |

These are produced with the single shipped threshold set (`config.SHIPPED_THRESHOLDS`); the
precision/recall figures in the write-up come from a separate leakage-free LOSO evaluation.

Each clip writes a timestamped JSON, `<clip_id>_interactions_<YYYYMMDDHHMM>.json`. Example
(`NmlzoaDcOuI_6`, run with `--detailed`):

```json
{
  "clip_id": "NmlzoaDcOuI_6",
  "generated_at": "2026-08-04T20:00:00",
  "interactions": [
    {
      "clip_id": "NmlzoaDcOuI_6",
      "person_id": 2,
      "vehicle_id": 1,
      "start_frame": 0,
      "end_frame": 42,
      "start_seconds": 0.0,
      "end_seconds": 7.0,
      "person": "Male, wearing green, black, and white.",
      "vehicle": "Red four-door sedan"
    }
  ]
}
```

Each interaction carries the clip id, the person and vehicle **track ids**, the **frame
range** and matching **time span** (seconds), and a short **description** of each. The
committed `outputs/` use the default `--fast` backend; `--detailed` (moondream2, shown above)
is more accurate but much slower on CPU.

To visualize an interaction, `show_interactions` renders annotated contact sheets (person
in green, vehicle in red, drawn across the span plus context frames):

```
python -m scripts.show_interactions [--clip <clip_id>]
```

Sheets are written to `cache/viz/interactions/`; rendered sheets for every clip are committed
under [`outputs/sheets/`](outputs/sheets/) (see its README for the title/metric legend).
Example - the `NmlzoaDcOuI_6` person↔car interaction (person 2 × vehicle 1, frames 0-42):

![Example interaction sheet](docs/images/example_interaction_sheet.png)
