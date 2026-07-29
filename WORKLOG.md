# Work-with-Claude Log

A running, reverse-chronological log (newest first) of **concise highlights** — a short
narrative of what was done and the decisions made. Feeds the final write-up. Alternatives
are included only when they add real information (ask first).

Entry template:

```
## YYYY-MM-DD — Session N: <short title>

**Narrative:** <highlights — e.g. "module X + tests", "added CI">

**Decisions:** <key decisions>

**Next:** <what's next>
```

---

## 2026-07-29 — Session 6: shared test factory + detection processing

**Narrative:** Extracted the duplicated `TrackedBox` builder into a shared
`tests/factories.py` (new tests package). Added the `detection_processing` module + tests
(`build_tracked_boxes`, `filter_detections_by_class`). Cleared PR-review nits (grouped
imports, docstring style) and a CI isort/black config clash. Landed via PRs #4/#5.

**Decisions:** Pure/glue split so pure logic runs in lean CI without torch; isort
`force_single_line=false` made Black-compatible (`multi_line_output=3`); multi-line
docstrings start with a newline.

**Next:** `tracker_engine` glue + an overnight, checkpointed cache-building batch script
(see DETECTION_TRACK_PLAN.md).

---

## 2026-07-29 — Session 5: P1 track data model, CI, PR workflow

**Asked:** Implement P1 under TDD; add CI; move to a PR-based flow with a periodic docs PR.

**Decisions / work:**
- **P1 track data model** (`tracked_data_model.py`): `TrackedBox` + `ClipMetadata`
  dataclasses, CSV track cache + JSON metadata sidecar, `frame_to_seconds`. Built TDD
  (11 tests: round-trip, dtype restoration, deterministic ordering, validation). Green;
  black / isort / mypy clean.
- **Cache = CSV** (not parquet) via the stdlib `csv` module — inspectable, clean
  native-type round-trips, one fewer dependency (dropped `pyarrow`).
- **CI** (`.github/workflows/ci.yml`): isort / black / pytest on PRs + pushes to `main`;
  lean install (pinned tools only) since current tests are pure-Python. CI bundled with
  the P1 code so it runs on the PR that introduces the code it checks.
- **PR-based workflow**: work lands via reviewed PRs merged on GitHub;
  `WORKLOG.md` / `PLAN.md` updates are **batched into a periodic docs PR**.

**Outcome:** two PRs opened — #1 (docs: rules, checklist, this worklog/plan update) and
#2 (P1 code + CI). Local `main` clean.

**Reflect / next:** after the PRs merge, wire `tracker_engine.py` (YOLO11-l + BoT-SORT,
high `imgsz`) with a smoke test, then **P2** (pairwise overlap/distance signals + graphs).

---

## 2026-07-29 — Session 4: P0 environment setup

**Asked:** Start P0 — create the venv + requirements; reviewer runs the install in a
separate terminal.

**Decisions / setup:**
- **Python 3.12.3**, venv + pip.
- **CPU-only torch** (no CUDA): `torch==2.13.0+cpu` / `torchvision==0.28.0+cpu`. The
  default CUDA wheel was installed first by mistake, then swapped for the CPU build.
  Keeps the footprint small and matches the plan's **deterministic CPU inference**
  assumption (no GPU non-determinism to manage).
- Dependencies **exact-pinned** to the tested environment in `requirements.txt`
  (runtime: ultralytics 8.4.110, opencv 5.0.0.93, numpy 2.5.1, pandas 3.0.5,
  pyarrow 25.0.0, matplotlib 3.11.1) and `requirements-dev.txt` (black, isort, pytest,
  mypy). No full lock file — avoids the `+cpu` local-version portability trap; torch
  CPU-install noted for reviewers.
- `pyproject.toml` holds tool config only (Black 88 / isort google / pytest / mypy).

**Outcome:** environment installed and sanity-checked (imports OK, ultralytics 8.4.110).

**Reflect / next:** P1 — detect → track → cache tracks, under TDD (core geometry/cache
logic tested first; detection/tracking glue kept light).

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
