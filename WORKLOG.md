# Work-with-Claude Log

A running, reverse-chronological log (newest first) of what was asked, decisions
made (with alternatives considered), outcomes, and reflections. Feeds the final
write-up. One entry per working session or significant request.

Entry template:

```
## YYYY-MM-DD — Session N: <short title>

**Asked:** <what I was asked to do>

**Decisions:**
- <decision> — <rationale>. _Alternatives: <options considered>._

**Outcome:** <what changed / current state>

**Reflect / next:** <observations, follow-ups>
```

---

## 2026-07-29 — Session 3: Pipeline plan (PLAN.md)

**Asked:** Frame-refine the GT, then converge the pipeline design and draft a plan
(reviewer leading).

**Decisions:**
- **Detector:** YOLO11-l (mature, well-supported). _Alternatives: YOLO26 (newest,
  NMS-free, best small-object) and YOLOv8; YOLO11 chosen to de-risk._
- **Detection res:** high `imgsz` to start; **SAHI tiling deferred** as an improvement
  for the 4K aerial clip (doesn't compose with `model.track()`).
- **Tracking:** `model.track()` with **BoT-SORT** (camera-motion compensation for PTZ).
- **Overlap metric:** normalized intersection (intersection ÷ person area), not IoU.
- **Distance metric:** center-distance normalized by car bbox size (scale/egomotion-
  invariant); **pixel→meter calibration deferred** (noted as considered, not used).
- **Tuning/eval:** leave-one-scene-out CV over 6 scene units; keep scene groups intact.
- **Cache tracks** to disk so the tuning/graph loop never re-runs detection.
- Enter/exit/other **type classification staged separately** from candidate detection.
- **`show_interaction`** visualizer: consumes the output records to render annotated
  interaction frames/clips (validation aid + optional deliverable).
- **Descriptions YOLO-only**: base detector (class + id) + open-vocab YOLO-World/YOLOE for
  attribute tags, all local. Noted tradeoff: an LLM API could replace the open-vocab model
  (network + cost vs. extra local model + memory) — not taken for the submission.

**Outcome:** [PLAN.md](PLAN.md) drafted (objective, locked decisions, scene units,
staged pipeline, thresholds, output, determinism, TDD, deferrals, P0–P6 phasing).
Committed GT earlier this session (`38ac9e8`).

**Reflect / next:** On plan approval, start **P0** (env setup) then **P1** (detect +
track + cache) under TDD.

---

## 2026-07-29 — Session 2: Pipeline plan & ground-truth setup

**Asked:** Discuss and agree the approach; set up a ground-truth annotation workflow
driven by watching each clip.

**Decisions:**
- Approach: classic **detect → track → reason** pipeline (deterministic/reproducible)
  over a VLM/captioning approach.
- Model stack: **Ultralytics YOLO + ByteTrack**.
- Output: **JSON**, per-clip files + a combined `results.json`.
- Preprocessing: **downscale large frames** for detection; compute times from each
  clip's **real fps**.
- Descriptions richness: **tabled** until after clip review.
- Establish **ground truth first**, via an interactive dialogue: Claude names a clip,
  the reviewer watches it externally and reports summary + interaction times, Claude
  records it. Thumbnails deferred (extract specific frames on request).

**Refined definitions (during review):**
- Interaction time span = **vehicle-contact window only** (starts when person/vehicle
  boxes overlap, ends when person is invisible inside or boxes separate).
- **Interaction = discrete event** (enter/exit/other contact), **not** the state of
  riding/occupying — continuous riding/driving with no visible mount/dismount is a
  pass-by.
- **Vehicle scope** = COCO `car` class **only** for this assignment (every interaction
  is with a car); kept configurable but defaults to `car`.
- Clips from the same scene noted for train/val grouping: `mKzCQKTHizw_0`/`_1` (same
  scene, different angle) and `NmlzoaDcOuI_1`/`_6` (same camera, different car).

**Outcome:** All 8 clips annotated in
[ground_truth/ground_truth.md](ground_truth/ground_truth.md) (human-readable) and
[ground_truth/interactions.csv](ground_truth/interactions.csv) (eval mirror), then
**frame-refined** via labeled ffmpeg contact sheets (reviewer picked exact start/end
frames per interaction; script in scratchpad `frame_sheet.sh`). Applying the
contact-window + same-person/same-vehicle **merge rules** gives **13 interactions total**
(6 enter, 6 exit, 1 other) — merges collapsed clip 3 (exit+other) and clip 4
(exit+trunk+rear-door) into single windows. Data spans fixed CCTV, aerial/drone,
PTZ/moving cameras; night/day; 4K to 352×288; jump-cuts, occlusions, truncated events,
overlapping/concurrent interactions, a driver swap, and pass-by hard negatives.

**Reflect / next:** Finalize the description approach (tabled), then begin TDD on the
interaction-reasoning logic against this GT. Key challenges to design for: camera
egomotion (PTZ clips), low fps (6 fps clips), heavy occlusion / partial-frame vehicles,
concurrent/overlapping interactions on the same vehicle, and **small-object detection on
the 4K aerial clip** (`gt1125_06`) — prefer tiled/sliced inference (SAHI-style) or
high-res detection there rather than the default downscale (preprocessing should be
adaptive per clip).

**Tunable interaction thresholds to define in PLAN.md (parked for the plan discussion):**
- min person↔car box **overlap** (IoU or % of person box inside car box) to count as contact;
- max person↔car box **proximity/distance** (normalized by frame size) when boxes don't overlap;
- min interaction **duration** (frames / seconds) to filter momentary/spurious contacts
  (clip `NmlzoaDcOuI_1` has people passing in front of the car — box overlap without
  interaction — as hard negatives for this tuning);
- **merge rule**: collapse same-person + same-vehicle engagements whose boxes stay close
  into one interaction (can't separate them yet). Possible future aid: a "vehicle
  open/closed" door-state classifier to split them back apart.

---

## 2026-07-29 — Session 1: Project setup & ground rules

**Asked:** Initialize a git repo, ignore the `Videos/` folder, and define code-style
and general workstyle rules for the project. Then set up a work-with-Claude log.

**Decisions:**
- **venv + pip** for environment/dependencies — simplest reproducible setup for a
  reviewer (`python -m venv .venv && pip install -r requirements.txt`).
  _Alternatives: Poetry (stronger lockfile reproducibility, but requires reviewers to
  have Poetry installed)._
- **isort with `profile="google"`** for import ordering.
  _Alternatives: Ruff (all-in-one linter), flake8-import-order._
- **Google Python Style Guide** + **Black** (line length 88) + **type hints required**
  + Google-style docstrings.
- **`pyproject.toml` for tool config only**; runtime deps pinned in `requirements.txt`.
- **TDD** (red-green-refactor): tests first, confirm they fail, then implement.
- **Plan-first** workstyle: propose a short plan before non-trivial changes.
- **Conventional Commits**; commit only when explicitly asked.

**Outcome:** git repo initialized on branch `main`; `.gitignore` (ignores `Videos/`
+ Python/venv/editor noise); `CLAUDE.md` capturing all rules above; this `WORKLOG.md`
created and seeded. Nothing committed yet.

**Reflect / next:** Scaffold `pyproject.toml` (Black/isort/pytest config) and
`requirements.txt` when development begins. First feature work will start with tests
per TDD, following a proposed plan.
