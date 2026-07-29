# Plan — Person-Vehicle Interaction Detection

Living plan for the pipeline. Stages/components level (not function signatures yet).
Ground truth lives in [ground_truth/](ground_truth/); rules in [CLAUDE.md](CLAUDE.md);
decision history in [WORKLOG.md](WORKLOG.md).

## Objective

For each MP4 clip, output a machine-readable list of person↔car interactions
(`enter` / `exit` / `other`), each with `clip_id`, frame range + time span, a person
description, and a vehicle description. Pass-by is not an interaction.

## Locked decisions

- **Detector:** Ultralytics **YOLO11-l**, COCO-pretrained, no training. Detect **person
  (0)** and **car (2)** only, filtered at inference (`classes=[0, 2]`).
- **Vehicle scope:** `car` only (see GT scope decision).
- **Detection resolution:** start with **high `imgsz`** (e.g. 1280) to recover cut-off /
  distant people. Adaptive per clip.
- **Tracking:** Ultralytics `model.track()` with **BoT-SORT** (default; has camera-motion
  compensation — helps the PTZ / aerial clips). ByteTrack as an alternative to try.
- **Overlap metric:** **normalized intersection = intersection ÷ person-box area**
  (fraction of the person inside the car box) — far more sensitive than IoU when a person
  is much smaller than a car.
- **Distance metric:** center-distance **normalized by car bbox size** (e.g. ÷ car
  diagonal) — scale- and egomotion-invariant, no calibration needed.
- **Time axis:** frames, converted to seconds via each clip's true fps.
- **Tuning/eval methodology:** **leave-one-scene-out (LOSO) cross-validation** over the 6
  independent scene units (below); report per-scene. Keep scene groups intact — no leakage.
- **Caching:** persist per-clip tracks to disk so the analysis / graph / tuning loop never
  re-runs detection.

## Scene units (for LOSO, no leakage)

1. `1THkHYIQ_bY_0` — fixed CCTV, night
2. `gt1125_06` — 4K aerial/drone
3. `HIu4lM4B8hA_1` — grayscale CCTV
4. `iMGR_0AG3a8_2_3` — indoor ceiling CCTV
5. `mKzCQKTHizw_0` + `mKzCQKTHizw_1` — PTZ, same scene (different angle)
6. `NmlzoaDcOuI_1` + `NmlzoaDcOuI_6` — fixed elevated, same camera (different car)

## Pipeline stages

1. **Decode** — iterate frames; read true fps + frame count.
2. **Detect** — YOLO11-l per frame → person/car boxes (high `imgsz`).
3. **Track** — BoT-SORT → stable `person_X` / `car_X` boxes over time.
4. **Cache tracks** — write per-clip tracks (JSON/parquet) with boxes + frame/time.
5. **Pairwise signals** — for every (person, car) pair over time: normalized-intersection
   overlap and normalized center-distance.
6. **Graphs** — plot overlap/distance vs. time per pair → visually select candidates.
7. **Candidate detection** — threshold the signals (see below) into interaction candidates.
8. **Type classification** — label each candidate `enter` / `exit` / `other`
   (**deferred second stage**; overlap/distance find *candidates*, not the type — enter vs
   exit needs track birth/death-near-car logic).
9. **Emit** — per-clip JSON + combined `results.json`.

## Interaction logic & thresholds

- Contact window = signals cross the thresholds, per the GT contact-window definition
  (starts at box overlap, ends at separation / person invisible inside).
- **Tunable thresholds** (to fit via LOSO against the GT):
  - min **overlap** (normalized intersection) to count as contact;
  - max **proximity** (normalized distance) when boxes don't overlap;
  - min **duration** (frames/seconds) to reject momentary contacts / pass-bys.
- **Merge rule:** collapse same-person + same-vehicle engagements whose boxes never
  separate into one interaction (see GT merge rule).
- **Hard negatives:** people passing in front of a car (box overlap, no interaction) —
  `NmlzoaDcOuI_1` — must be rejected by the duration/overlap thresholds.

## Output & tooling

- Per-clip JSON + combined `results.json`.
- Per interaction: `clip_id`, frame range, time span (s), interaction type, person
  description, vehicle description.

### `show_interaction` (visualization)

- A tool that consumes the **output records** (clip_id, frame range, person/car
  descriptions) and renders the interaction: annotated frames / a short clip with the
  interacting `person_X` and `car_X` boxes highlighted and labeled over the frame range.
- Doubles as (a) a validation aid while tuning thresholds and (b) the assignment's
  optional annotated-frame deliverable.

### Descriptions (YOLO-only)

- **YOLO does not caption** — it's a detector. Descriptions come from the **YOLO family
  only**, fully local:
  - base detector → class (person/car) + track id + bbox;
  - **open-vocabulary YOLO-World / YOLOE** → attribute tags via local text prompts
    (e.g. "red car", "person in white shirt") — deterministic, offline, no external service.
- **Tradeoff (noted, not taken):** an **LLM API** could replace the local open-vocab
  model for richer free-text descriptions. The tradeoff is **network connectivity + per-call
  cost** (API) **vs. an extra local model + memory/compute** (open-vocab YOLO). We keep it
  local for the submission — reproducible, offline, no cost or external dependency.

## Determinism & reproducibility

- Pin weights, `imgsz`, `conf`, `iou`, tracker config; fix seeds. CPU inference is
  deterministic. Document `yolo11l.pt` as an auto-downloaded external asset.

## Evaluation

- Compare predicted interactions to `ground_truth/interactions.csv`: temporal overlap of
  windows + interaction-type match, reported per scene (LOSO).

## Testing (TDD)

- Core logic tested first with **synthetic tracks**: overlap/distance computation,
  candidate detection, the merge rule, type classification. Glue/IO not exhaustively tested.

## Deferred / future improvements (noted, not built now)

- **SAHI tiling + external tracker** (`boxmot`) for the 4K aerial clip if high-`imgsz`
  recall is poor there (`model.track()` doesn't compose with SAHI directly).
- **Pixel→meter calibration** — considered, but not used now; normalized metrics avoid the
  need. Could approximate px/m from a known car length where a clean reference exists.
- **LLM-API description backend** — alternative to the local open-vocab model (see the
  Descriptions tradeoff); not taken for the submission.

## Suggested phasing

- **P0** ✅ — env setup (`requirements.txt`, `pyproject.toml` for Black/isort/pytest); CI added.
- **P1** 🔨 — detect + track + cache tracks. Done: track/metadata **data model + CSV cache**
  (TDD). Next: `tracker_engine.py` (YOLO11-l + BoT-SORT) + smoke test (+ optional overlay).
- **P2** — pairwise overlap/distance signals + graphs.
- **P3** — threshold candidate detection + LOSO tuning + merge rule.
- **P4** — enter/exit/other type classification.
- **P5** — JSON output + `show_interaction` visualization + evaluation vs. GT.
- **P6 (deferred)** — SAHI + external tracker; px2m; richer descriptions; visualizations.
