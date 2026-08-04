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

## 2026-08-04 — Session 15: deliverable docs finalized

**Narrative:** Closed out the written deliverables (PRs #52–#60). Aligned README and write-up
with what is actually committed under `outputs/`, fixed the README Deliverables nesting, and
**split the write-up into a ≤2-page `WRITE_UP.md` executive summary plus a full
`docs/DETAILED_WRITE_UP.md`** — the assignment caps the write-up at two pages, and the
material worth keeping (limitation images, GT totals, alternatives considered, the external
clip) did not fit. Surfaced decisions and findings from `docs/PLAN.md` and this log into the
write-up, so the reasoning is in the deliverable rather than only in the working notes.
Slimmed `CLAUDE.md` to project-specific guidance and marked `CURRENT_STATUS.md` as untracked
working state, not part of the deliverable.

**Decisions:** Write-up **split rather than trimmed** — PR #54 (trim toward the page limit)
was closed in favour of the two-document split. `CURRENT_STATUS.md` is **session state, not a
deliverable**: gitignored, so a fresh clone won't have it.

**Next:** make the repo public — the last unchecked box in `docs/TODO.md`.

## 2026-08-04 — Session 14: real descriptions from cached tracks

**Narrative:** Closed the session-13 gap — `classify-tracks` was model-free (placeholder
captions only), so only the full `run` could produce real descriptions. It now takes
`--fast`(default)/`--detailed`/`--placeholder` and resolves the sibling video (`--video`, else
`<videos-dir>/<clip>.mp4`) to cut crops, reusing `_describer_for` and `make_describe_box`
(PR #51). A missing video degrades to a placeholder with a warning instead of crashing. TDD:
4 new tests, full suite 162 passing. **Verified end-to-end on the shortest clip
(`NmlzoaDcOuI_6`), both run modes × both backends, all real captions** — cached
`classify-tracks --fast` 9s and `--detailed` 7m52s, matching the full `run --fast` (3m) and
`run --detailed` (~10m) exactly, confirming `classify-tracks` does not silently re-detect.

**Decisions:** Both entry points must emit real descriptions for both backends — a cached
path that can only produce placeholders is not a usable second run mode.

**Next:** deliverable docs pass (→ Session 15).

## 2026-08-04 — Session 13: `--detailed` outputs + repo root cleanup

**Narrative:** Generated and committed the `--detailed` (moondream) results for all 8 clips —
15 interactions — alongside the `--fast` results already on `main` (PR #50), under the
filename convention
`outputs/<clip>/<clip>_interactions_<fast|detailed|placeholder>_<ts>.json`. Landed the
descriptions wiring itself (#46), the committed tracks/sheets/timing findings (#43), the
`show_interactions` fit-on-all default (#44), and the README/write-up fills (#42, #45, #47,
#48). Tidied the repo root (#49): planning docs moved under `docs/`, the assignment PDF to
`docs/reference/`. Pinned `clip` (git SHA) and `ftfy==6.3.1`.

**Decisions:** Ship **both** backends' outputs rather than picking one, with the method in the
filename so a reader can tell which produced a given result. Identified the gap that
`classify-tracks` could not produce real descriptions from cached tracks — carried into
Session 14.

**Next:** wire real descriptions into `classify-tracks` and verify both run modes end-to-end.

## 2026-07-31 — Session 12: descriptions backend chosen + external clip run

**Narrative:** Selected the description backend by benchmarking on real crops. Built isolated,
model-injected modules: `frame_crops.py` (`read_frame`/`crop_box`/`crop_for_box`, unit-tested),
`attribute_tagger.py` (YOLO-World `load_tagger`/`tag_image`), and reused `captioner.py`
(moondream2 `load_captioner`/`query_frame`). Verified live that **moondream2 @ 2025-06-21 needs
transformers 4.x** (5.x's `all_tied_weights_keys` breaks its trust-remote-code modeling) and
loads on CPU **without `device_map`/`accelerate`**. Side-by-side on identical crops: **moondream
is far more accurate** (clothing/vehicle colour, type, even brand + context) where YOLO-World v2-L
gives systematically wrong colours; YOLO-World is ~500× faster (~0.2s vs ~90s–5min/query) but
attribute-weak on low-res CCTV. Downsampled the external phone clip (1080p portrait → 720×720p
12 fps, 126 MB → 5.85 MB) and ran the full pipeline → **14 candidate interactions** (GT from user:
4 people exiting one dark car), surfacing pass-by FPs + fragmentation.

**Decisions:** **Dual, flag-selectable description backend — `--detailed` (moondream2) vs `--fast`
(YOLO-World v2-L)**; fast for realtime, detailed for accuracy; moondream run only on interaction
participants, **dedup per track**, cached, offline. **Moondream FP-filter VALIDATED** — on the
union(person,vehicle) crop it labelled the `NmlzoaDcOuI_1` p66 passer-by "passing by" (all frames)
and a real interaction "interacting". moondream weights (~3.7 GB) documented as an auto-downloaded
external asset. Moondream PR #36 was closed then un-shelved (decision reversed).

**Next:** wire `describe_window` behind the flag (crop → backend → cache per track); sort PR
structure (reuse PR #36 captioner + commit the new modules); render the external clip's sheets.

## 2026-07-31 — Session 11: deliverable CLI + shipped-vs-LOSO threshold comparison

**Narrative:** Turned the results glue into the actual **deliverable CLI**
`scripts/detect_interactions.py` (PR #34): full pipeline by default (video → detect+track →
classify), staged subcommands `detect-and-track` / `classify-tracks`, single file **or folder**,
**per-clip output folder** `<results-dir>/<clip_id>/` (tracks + results together; classify-tracks
copies the source tracks in), per-clip error-continue, and a graceful non-`.mp4` exit. Demoted
`build_results.py` to the classify-stage library and moved the pinned `SHIPPED_THRESHOLDS`
(fit-on-all) into `config.py`. Added `scripts/compare_thresholds.py` (PR #35, stacked) reporting
per-scene/overall P/R/F1 under **LOSO fold** (honest) vs **shipped** (in-sample) thresholds plus
the threshold-field diffs.

**Decisions:** the shipped deliverable uses **one fixed fit-on-all threshold set** and is
**LOSO-unaware** (`--loso` removed). Comparison finding: **5/6 LOSO folds are identical to the
shipped set**; only `cctv_night` differs (`min_confidence` 0.0 vs 0.3), where the shipped floor
drops one FP. Combined `results.json` + folder/robustness hardening deferred to final cleanup.

**Next:** real descriptions (→ Session 12).

## 2026-07-30 — Session 10: output layer + interaction contact sheets

**Narrative:** Built the output/inspection layer over the tuned detector (PRs #27–#31):
`evaluation.classify_windows` exposing per-window **TP/FP/FN** `LabeledWindow`s (evaluate_*
re-expressed on it, counts unchanged); `window_summary.summarize_window` (peak overlap, min
distance, mean/min/max person+vehicle confidence, duration); `interaction_records`
(`InteractionRecord` + timestamped per-clip `results.json` writer/loader) with a
`descriptions.describe(crop)` **placeholder captioner hook** + `placeholder_description`; and
`scripts/show_interactions.py`, a **TP/FP/FN contact-sheet renderer** (per-clip
`detect_candidates → classify_windows`, person(green)/vehicle(red) boxes drawn across the span
+ pre/post context, tiles captioned `ov=/d=`, title carrying thresholds and peakOv/minDist).
Reused/added pure helpers `loso.fold_thresholds_by_clip` and `track_selection.sample_span_frames`.

**Sheets generated (all 8 clips, LOSO fold thresholds):** **19 sheets = 10 TP / 6 FP / 3 FN**
(reconciles with the LOSO report), in `cache/viz/interactions/` as
`<clip>_<personid>_<vehicleid>_<start>-<end>_<TP|FP>.png` (FN: `<clip>_gt<id>_<start>-<end>_FN`).

**Visual analysis (user, in progress):**
- **`NmlzoaDcOuI_1` p66×v1 f93–102 labeled TP is really an FP** — a passer-by walking in front
  of the car, coinciding in time with a real driver-swap interaction (GT #2). Two root causes:
  (1) `classify_windows` matches predictions to GT by **temporal overlap only, not identity**;
  (2) **single-camera bbox ambiguity** — passing-in-front looks identical to entering by
  overlap alone, worst in this multi-person driver-swap. GT is correct; the detector locked
  onto the wrong nearby person and the temporal-only metric credited it. → limitations material.
- **`gt1125_06` p111×v5 f369–381 labeled FP is really a TP** — the contiguous continuation
  (person ID switch 18→111, same vehicle v5, boundary f369) of the TP p18×v5 f239–369, both
  inside GT #2 (enter, f228–516, black SUV). GT is correct; the second fragment can't re-claim
  the already-matched GT under **one-to-one** matching, so it scores FP. `p114×v29 388–412`
  likely a third fragment on the same event.
- **ptz `mKzCQKTHizw_0` GT #1 — label swap (occupancy trigger), NOT fragmentation.** The FP
  `45×49 f271–289` is the *real* enter (woman runs in + gets into the gray car, peakOv 0.98);
  the TP `51×49 f262–287` is a **different person already seated inside** (p45 f190–289 vs p51
  f262–287, co-present 262–287) — sustained overlap, no mount/dismount → should be FP.
  One-to-one gave GT #1 to the earlier occupant (f262<f271), swapping the labels. **Not a GT
  issue.** New failure mode: seated occupant ≈ enterer by overlap alone → needs an **enter/exit
  transition** signature (overlap low→high / high→low), not "overlap high for N frames".
- **iMGR `_gt2` exit vehicle = merge of car boxes 130+132** (stacked; 130 upper/rear flaky, 132
  lower/front) — recorded in `tracking_notes.md` so the cluttered FN sheet's relevant vehicle is
  explicit; useful for the description/merge + identity eval. (The woman herself is the
  documented detection-recall miss → "person not detected in span".)
- **Net:** the temporal-only, one-to-one matcher mislabels **both ways** — false TPs (identity:
  hard p66; occupancy: ptz) and false FPs (fragmentation: gt1125). → motivates description-based
  interaction merge + identity-aware matching + an enter/exit transition signal.

**Ideas raised (running list in CURRENT_STATUS.md):** expand the title's shortcut names into an
explicit legend; add **per-frame detection confidence** to each tile (help weed out FPs); build
the **description pipeline as a merge key** to unify fragmented interactions.

**Decisions:** temporal-overlap matching kept for now (identity-aware matching is future work);
`results.json` is **per-clip only**, timestamped filename; descriptions injected via a
`describe_window` callback so the records core stays captioner-agnostic; enter/exit/other typing
stays **dropped** (not required). Sheets use LOSO fold thresholds by default (`--fit-on-all`
option) so labels match the report.

**Next:** `results.json` generation script on real clips (glue: detect + `describe_window` via
`placeholder_description` + `write_clip_records`); then captioner selection, interaction merge,
and deliverables (README, ≤2-page write-up, make repo public).

---

## 2026-07-30 — Session 9: class-filter config + per-clip vehicle classes

**Narrative:** Made the detection class filter configurable and separated a "vehicle"
definition from it. `config.target_class_ids` is now optional (`None` = unfiltered), the
COCO id→name map is the full COCO-80 loaded from `data/coco_classes.json` (generated from
`YOLO('yolo11l.pt').names`), and vehicles are `VEHICLE_CLASS_IDS = (car, bus, truck)` with
per-clip overrides. Visualization sheets now pick each track's **highest-confidence** frame
and `show_track` labels tiles `f<frame>/<total>`.

**The `HIu4lM4B8hA_1` issue (why per-clip overrides exist):** its `show_objects` sheet had
**0 vehicles** under a car-scoped filter — only 2 persons. Re-running detection **unfiltered**
showed the cars *are* detected, but YOLO labels them **`boat`** in that low-res grayscale
night scene:

- Before — unfiltered detection (cars appear, but as `boat`):
  ![HIu all classes](images/HIu4lM4B8hA_1_all_classes.png)
- After — per-clip override adds `boat` to the vehicle set, so the cars are kept:
  ![HIu with boat](images/HIu4lM4B8hA_1_vehicles_with_boat.png)

**Solution:** `CLIP_VEHICLE_CLASS_OVERRIDES["HIu4lM4B8hA_1"] = (8,)` (boat), applied per
clip by `build_cache`; a `--force` rebuild recovered the cars (5 boat "vehicle" tracks).
**TODO:** drive this from clip properties (resolution / lighting) rather than a hardcoded
map — expand the vehicle definition automatically when video quality is low.

**Decisions:** boat stays a **per-clip** override (not a global vehicle class) to avoid
polluting other clips; COCO map kept as an external data file (stdlib load, no ultralytics
in the pure module); object sheets select by confidence, not box size.

**Object-noting review (all 8 clips):** went clip-by-clip through the `show_objects` sheets
recording tracking issues → [ground_truth/tracking_notes.md](../ground_truth/tracking_notes.md).
**ID switches are pervasive** — interacting objects fragment into 2–11 track ids (worst: the
cover-remover in `1THkHYIQ_bY_0` ≈ 11 ids); plus spurious tracks (open car doors, car-cover
artifacts) and a couple of missed people. **Decision:** don't retune/rerun tracking
(expensive); instead **merge interactions in postprocessing** using person description +
vehicle description + interaction location (+ temporal adjacency) — so descriptions/location
become **merge keys**, computed early.

**iMGR missed woman — detection vs tracking:** `iMGR_0AG3a8_2_3` GT #2 (woman exits black car,
f128–132) is untracked. Two checks: (1) an **unfiltered** run adds no person (not a
misclassification); (2) **detection-only** `predict` on frames 126–134 gives **0 person
detections during the exit** (isolated single hits at f132/f134 only). → a **detection-recall
miss**, not a tracking edge case. Deferred to a higher-recall detector / SAHI tiling.

**Next:** P2 signals (overlap/distance/confidence per pair) + threshold-tuning graphs with GT
overlay; interaction-level merge (P3) keyed on descriptions + location.

---

## 2026-07-30 — Session 8: overnight cache results + visualization tools

**Results of overnight run:** PR #9 merged; `scripts/build_cache.py` ran over all 8 clips
and populated the two-level cache (`cache/raw/*.jsonl` + `cache/tracks/*.csv` + meta) for
every clip. Track-row counts per clip (a rough tracking-density sanity signal):

| clip | track rows | note |
|---|---|---|
| gt1125_06 | 17,537 | 4K clip, dense — expected |
| iMGR_0AG3a8_2_3 | 3,103 | |
| 1THkHYIQ_bY_0 | 1,734 | |
| mKzCQKTHizw_0 | 617 | |
| NmlzoaDcOuI_6 | 584 | |
| mKzCQKTHizw_1 | 342 | |
| NmlzoaDcOuI_1 | 286 | |
| HIu4lM4B8hA_1 | 47 | very sparse — flagged for eyeball check |

**Adding visualization tools:** Track-inspection contact sheets over the cached tracks,
to eyeball tracking quality before building P2 signals. Split pure/glue as usual:
- **`track_selection.py`** (pure, unit-tested): `sample_track_boxes` (one track, every
  `step`-th frame), `largest_box_per_track` (each track's max-area box, ties → earliest
  frame), `crop_window` (fixed `fraction`·frame window around a center; corners may exceed
  the frame — caller black-pads rather than shifting/shrinking).
- **`visualization.py`** (glue, lazy cv2/matplotlib, integration smoke test): `show_objects`
  (one tile per tracked object at its largest-box crop, labeled `id<n> <class>`) and
  `show_track` (a single object over time in 0.25-frame crops with its box drawn). Sheets
  are matplotlib grids (the Python equivalent of the ffmpeg contact sheets used for GT
  frame-refinement). Output → `cache/viz/{objects,tracks}/`.

Ran `show_objects` over all 8 clips. **Finding:** `HIu4lM4B8hA_1` yields only 2 tracked
objects, both persons — **zero cars** (small-FOV low-res night clip), a detection gap to
note for interaction detection. Object counts elsewhere look sane (e.g. 44 cars in the 4K
`gt1125_06`).

**Decisions:** `show_track` keeps the fixed 0.25-frame crop (context around the object,
uniform tile size) over a tight-box-then-pad crop; `show_objects` draws no box (the
largest-box crop *is* the object, so a border would add nothing).

**Next:** pick specific track ids per clip → `show_track` sheets; cache sanity-check; P2.

---

## 2026-07-29 — Session 7: pure cache modules

**Narrative:** Added the pure half of the tracker/cache work — `config.DetectionConfig`,
`clip_assembly` (`VideoProperties` + `build_clip_metadata`), and `raw_detections` (raw
per-frame cache + `tracks_from_raw`). Unit-tested; PR #7.

**Decisions:** Raw per-frame model output is stored as **JSONL** (one JSON object per
line), so frames append one at a time, stream-read, and survive an interrupted run
(complete lines stay valid; a partial last line is skipped — a truncated JSON array
wouldn't parse at all). Regular **JSON** is kept for the small `ClipMetadata` sidecar
(written once, atomic); tracks are CSV.

**Next:** the glue — `tracker_engine` + overnight `build_cache.py`.

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
[ground_truth/ground_truth.md](../ground_truth/ground_truth.md) (human-readable) and
[ground_truth/interactions.csv](../ground_truth/interactions.csv) (eval mirror), then
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
