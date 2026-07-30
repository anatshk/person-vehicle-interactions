# Tracking Notes — per-clip `show_objects` review

Reviewer + Claude go clip-by-clip through `cache/viz/objects/<clip>.png` and record
tracking issues: **ID switches** (one physical object with several track ids),
**misclassifications**, **missed objects**, and **spurious tracks**. Feeds the decision on
whether a track-merge pass is needed before P2, and the write-up's limitations section.

Object counts are the number of distinct track ids (one tile per id). Review order is by
ascending object count.

| clip | objects | reviewed | key issues |
|---|---|---|---|
| mKzCQKTHizw_1 | 4 | ✅ | 1 car split across ids 1/4/7 (ID switch); person ok |
| HIu4lM4B8hA_1 | 7 | ✅ | cars=`boat`; id28/39 same car (break); id31=door, id4=bg (spurious); male person missed |
| mKzCQKTHizw_0 | 9 | ✅ | person id1/29 same; woman id2/9/45 same (ID switches); cars id11 parked, id49 interacted; id50/51 bg |
| NmlzoaDcOuI_1 | 26 | ✅ | car id1/41; passenger-exit woman id24/39/57; driver-exit id67/72 (ID switches); id40=driver door (spurious); many bg |
| NmlzoaDcOuI_6 | 26 | ✅ | interacting red car id1 + green-shirt person id2 both CLEAN (single ids); id54 maybe dup/pass-by, id25 pass-by car; rest bg |
| 1THkHYIQ_bY_0 | 40 | ✅ | car id5/35; **person uncovering split across ~11 ids** (extreme); rest bg + car-cover artifacts |
| iMGR_0AG3a8_2_3 | 47 | ✅ | white car id3/138/140/236; white-car woman id5/33/55/93/137; black car id130/132/230 (ID switches); woman exiting black car MISSED; rest parked (unfiltered probe requested) |
| gt1125_06 | 59 | ✅ | clear 4K; interacting: white car id6, black car id5, black shirt person id18/111, tan/gray person id21; bg scooter-kids & sitting person heavily fragmented |

---

## Cross-clip summary (all 8 reviewed)

- **ID switches are the dominant issue.** Interacting objects fragment into multiple ids:
  | clip | worst fragmentation of an interacting object |
  |---|---|
  | 1THkHYIQ_bY_0 | cover-remover ≈ **11 ids** (car 2) |
  | iMGR_0AG3a8_2_3 | white-car woman 5, white car 4, black car 3 |
  | NmlzoaDcOuI_1 | passenger-exit woman 3 (car 2, driver 2) |
  | mKzCQKTHizw_0 | woman 3 (non-interacting person 2) |
  | mKzCQKTHizw_1 | car 3 |
  | gt1125_06 | black-shirt person 2 (interacting people otherwise clean) |
  | HIu4lM4B8hA_1 | driving-away car 2 |
  | NmlzoaDcOuI_6 | **none** — interacting pair are single ids (clean) |
- **Spurious tracks:** open **car doors** tracked as objects (HIu id31, NmlzoaDcOuI_1 id40),
  **car-cover artifacts** (1THkHYIQ_bY_0), stray background objects.
- **Missed detections:** HIu male person; iMGR black-car woman (**genuine miss**, confirmed
  by unfiltered probe — not a misclassification).
- **Misclassification:** HIu cars → `boat` (handled by per-clip override). iMGR: none
  (unfiltered adds only 2 trucks; person count unchanged).
- Fragmentation is **scene/motion dependent** (worst in night/occluded/PTZ; clean in the
  well-lit static `NmlzoaDcOuI_6`).

## Strategy / decision (2026-07-30)

ID switches are **pervasive**: across the reviewed clips, nearly every *interacting* object
is fragmented into 2–5 ids (worst: the cover-remover in `1THkHYIQ_bY_0` ≈ 11 ids), plus
occasional door/artifact/background ghosts and a few missed people. `NmlzoaDcOuI_6` is the
clean exception (interacting pair = single ids).

**Decision:** do **not** retune BoT-SORT or re-run inference to fix this (expensive rerun).
Instead, handle fragmentation **at the interaction level in postprocessing**:
1. Detect candidate interactions on the fragmented tracks (P2/P3 signals + thresholds).
2. **Merge** candidates that are the same real interaction, using: **person description**
   + **vehicle description** + the interaction's **physical location** (+ temporal
   adjacency). This absorbs ID switches (same person/car/place → one interaction) without
   touching tracking.

Implication: the pipeline must produce, per candidate, a **person description**, a
**vehicle description**, and an **interaction location** — these become merge keys, not just
output fields. (Extends the existing PLAN "merge rule"; fold into PLAN on the next docs pass.)

---

## Notes per clip

<!-- Filled in as we review each sheet. -->

### mKzCQKTHizw_1 (4 objects)
- **ID switch:** ids **1, 4, 7** are all the **same car** (one physical car → 3 track ids).
- Person: OK (1 id, correct).
- → 4 tracked ids = 2 real objects (1 car + 1 person).

### HIu4lM4B8hA_1 (7 objects) — low-res grayscale night
- Vehicles detected as **`boat`** (per-clip override adds boat; see WORKLOG session 9).
- **id1** — the main car.
- **id28, id39** — same car (the one driving away); **ID switch / tracking break**.
- **id31** — the **door of the main car** tracked as its own object (spurious sub-part).
- **id4** — unclear, **probably a background object** (spurious).
- **id46** — the **female** person (OK).
- **Missed:** the **male** person is not tracked at all. Detection-only check (f86–112, GT
  entry f90–108): detected as `person` only fleetingly (f87–88, incl. 2 persons at f88),
  then **0 persons through the entry (f90–108)** — only the car (as `boat`); f100+ returns
  nothing (scene cut / view clears). → **detection-recall miss**, like iMGR. (Sheet:
  `cache/viz/detect_check/HIu4lM4B8hA_1_detect_86_112.png`.)
- → real objects ≈ 2 cars + 2 people; 3 of the 5 "boat" ids are spurious/duplicate, and a
  person is missing. Hard clip (low-res night).

### mKzCQKTHizw_0 (9 objects) — PTZ, same scene as mKzCQKTHizw_1 (different angle)
- **id1, id29** — same person (a **non-interacting** person); ID switch.
- **id2, id9, id45** — same person (the **woman who interacts** with the car); ID switch (3 ids).
- **id11** — a **parked car**, not interacted with.
- **id49** — the **car the woman interacts with**.
- **id50, id51** — background persons, **not relevant** to the task.
- → 9 ids = ~2 cars + 2 relevant people (+2 bg); two ID-switch clusters (2 and 3 ids).

### NmlzoaDcOuI_1 (26 objects) — fixed elevated; busy, many pass-bys
Only the relevant objects noted (rest are background people, not itemized):
- **id1, id41** — the **car**; ID switch.
- **id24, id39, id57** — the person **getting out of the passenger side** (woman, short hair); ID switch (3 ids).
- **id67, id72** — the person **leaving the driver side**; ID switch.
- **id40** — the **driver door being open**, tracked as its own object (spurious sub-part).
- → 1 car + 2 interacting people, each fragmented; 1 door ghost; ~19 background people.

### NmlzoaDcOuI_6 (26 objects) — same camera as _1, different car
- **id1** — the **red car** (interacting).
- **id2** — the person in the **green shirt** (interacting).
- **id54** — uncertain: maybe the **same car as id1** or another car driving by.
- **id25** — a car **driving by** (not interacting).
- All other people are background.
- → **The interacting pair (id1 car + id2 person) is cleanly tracked as single ids** — a
  clean counter-example to the fragmentation seen elsewhere (so ID switches are scene/
  motion dependent, not universal).

### 1THkHYIQ_bY_0 (40 objects) — night CCTV; GT = removing car cover (`other`)
- **id5, id35** — the **car** (uncovered); ID switch.
- **Person removing the cover** — split across **~11 ids**: 27, 100, 109, 120, 121, 129,
  134, 138, 150, 151, 152. **Extreme fragmentation** (crouching / low-light / occluded by
  the cover → tracker repeatedly loses and re-acquires).
- Others: background cars/people, plus **artifacts on the car cover** detected as objects.
- → worst fragmentation case so far: one interacting person = ~11 ids.

### iMGR_0AG3a8_2_3 (47 objects) — indoor ceiling CCTV; mostly parked cars
- **White car** — id3, 138, 140, 236 (4 ids); ID switch.
- **Woman of the white car** — id5, 33, 55, 93, 137 (5 ids); ID switch.
- **Black car** — id130, 132, 230 (3 ids); ID switch. For **GT #2 (the exit)** the interaction
  vehicle is the **merge of car boxes 130 + 132** (stacked: 130 upper/rear y≈181–292, 132
  lower/front y≈280–399 → together the whole SUV; 130 is flaky, drops at f128/f130). Noted
  because the FN sheet shows many cars — this identifies which boxes are the relevant vehicle.
- **Woman getting out of the black car** — **MISSED** (not tracked).
- Rest: parked cars.
- **Unfiltered probe result:** all-classes run gives car 40, person 7, **truck 2** (49
  total) vs filtered car 40 / person 7 (47). Person count is unchanged → the missed woman
  is **not** hiding under another class: she's a **genuine detection miss** (occlusion /
  top-down angle), unlike HIu's car→boat *misclassification*. So no per-clip override helps
  here — it's a recall gap (higher-recall detector / SAHI tiling territory, deferred).
- **Detection-only check (no tracking), GT frames 126–134** (interaction #2, exit f128–132):
  YOLO `predict` per frame detects **only cars (17–19) and nothing else** during f126–131;
  the single "person" at f132/f134 is a **hand-only partial box** (not a legitimate person).
  **Zero non-car/person classes** appear on her in any frame → she isn't mislabeled, she's
  simply undetected. **Confirmed a detection-recall miss**, not a tracking edge case. (Sheet:
  `cache/viz/detect_check/iMGR_0AG3a8_2_3_detect_126_134.png`.)

### gt1125_06 (59 objects) — 4K aerial/drone, "wonderfully clear"; parking lot
Interacting objects (rest are parked cars / pedestrians):
- **white car** — id6.
- **black car** — id5.
- **person, black shirt** — id18, 111 (ID switch).
- **person, tan/gray shirt** — id21 (single id).
Background but noted as heavily fragmented (non-interacting):
- kid on scooter (white shirt) — id24.
- kid on scooter (black shirt) — id26, 42, 114, 146 (4 ids).
- person sitting down — id27, 35, 81, 95, 126, 130 (6 ids).
- → even in a clear 4K clip, non-interacting objects fragment a lot; interacting people are
  1–2 ids here (cleaner than the night/PTZ clips).
