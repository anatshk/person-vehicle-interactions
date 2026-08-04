# Anat Shkolyar - Person-Vehicle Interaction Task

Executive summary. The full development narrative, decisions, figures, and per-clip analysis are
in the detailed write-up: [docs/DETAILED_WRITE_UP.md](docs/DETAILED_WRITE_UP.md). Run instructions
are in the [README](README.md).

Note: development was CPU-only (personal laptop), so the pipeline is offline/batch, not real-time.

## Approach

A classic **detect -> track -> reason** pipeline, chosen over an end-to-end video/VLM model for
determinism, inspectability, and graceful degradation (the brief prioritizes reproducibility).

1. **Detect + track** - Ultralytics **YOLO11** (COCO-pretrained) + the built-in **BoT-SORT**
   tracker (camera-motion compensation helps the PTZ/aerial clips). Person + vehicle classes.
2. **Pairwise signals** - for every (person, vehicle) pair over time: **normalized overlap**
   (intersection / person-box area), **normalized center-distance** (/ vehicle diagonal,
   scale-invariant), and per-frame **detection confidence**.
3. **Candidate detection** - threshold the signals into contact windows, with temporal gates
   (minimum duration, bridged gaps). Thresholds fit on all clips
   (`config.SHIPPED_THRESHOLDS`) and validated with leave-one-scene-out (LOSO) CV.
4. **Describe** - each person and vehicle is captioned by a flag-selectable backend:
   `--fast` (YOLO-World open-vocab, sane runtime) or `--detailed` (moondream2 VLM, slower but
   accurate). Descriptions are deduplicated per track.

Output per interaction: `clip_id`, frame range + time span, a person description, and a vehicle
description (machine-readable JSON under [`outputs/`](outputs/)).

## Results

- LOSO (leakage-free): **precision 0.62 / recall 0.77 / F1 0.69** (10 TP, 6 FP, 3 FN) against
  13 ground-truth interactions across 6 scene units.
- The thresholds generalize: **5 of 6 LOSO folds produce the shipped threshold set** (not overfit).
- Committed outputs hold **15 interactions across the 8 clips**.

## Key assumptions and scope decisions

- A **"vehicle" is a car** (COCO `car`; every GT interaction is with a car); the keep-set is
  broadened to truck/bus (and boat for one low-res clip) so a mislabeled car is not dropped.
- An **interaction is a discrete enter/exit/other contact event**, not the state of riding; a
  person merely passing by, or already seated, is not an interaction.
- Each output record is **one person paired with one vehicle**; concurrent people become
  separate records.
- Interaction type (enter/exit/other) is annotated in the GT but **not emitted** - the brief
  only asks to *list* interactions.

## Limitations

- **Single-camera ambiguity** - a person passing in front of a parked car overlaps its box and
  can score as an interaction.
- **Broken tracking -> fragmentation** - ID switches split one real interaction into several
  windows (the main source of false positives).
- **Static occupant** - a seated person has sustained overlap with no mount/dismount, so overlap
  alone can mistake them for someone entering.
- **CCTV detection recall** - low-res/grayscale clips miss people and misclassify cars, capping
  recall regardless of the interaction logic.

## Next steps

- An **enter/exit transition signal** (overlap rising low->high vs falling high->low) to fix the
  static-occupant false trigger and enable enter/exit/other typing in one signal.
- Use **descriptions as a merge key** to unify fragmented tracks, and as a pass-by FP filter
  (validated: moondream labels a passer-by "passing by" vs a real interaction "interacting").
- A **higher-recall / CCTV-tuned detector** and a **stronger tracker** (or a video-native
  detect+track model) for the recall misses and ID switches.
- **GPU** for the detailed VLM descriptions (minutes per crop on CPU today).

See [docs/DETAILED_WRITE_UP.md](docs/DETAILED_WRITE_UP.md) for the full reasoning, figures, and
per-clip breakdown.
