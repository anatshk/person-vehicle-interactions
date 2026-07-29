# Ground Truth — Person-Vehicle Interactions

Human-annotated ground truth, established by watching each clip in an external viewer.
Used later to evaluate the automated pipeline. Machine-readable mirror lives in
`interactions.csv`.

Time spans capture the **vehicle-contact window only** (the period the person is
physically engaged with the vehicle), not the person's full trajectory before/after.
Operationally, the contact window **starts** when the person and vehicle detection
boxes first overlap (person reaches the vehicle) and **ends** when the person becomes
invisible inside the vehicle **or** the person and vehicle boxes separate again.

Interaction types: `enter` (person gets into a vehicle), `exit` (person gets out of a
vehicle), `other` (deliberate physical engagement with the vehicle that isn't ingress/
egress — e.g. loading cargo, opening the hood, removing a cover) — incidental pass-by is
**not** an interaction.

**Scope decision (interaction types):** GT annotates *all* engagements, including
`other`. Whether `other` events are reported by the pipeline is a documented,
configurable flag (**default: broad** = report enter/exit/other). Alternatives
considered: narrow scope (enter/exit only, matching the brief's explicit examples) —
rejected as the default because it would discard defensible interactions, but preserved
as a toggle.

**Scope decision (vehicle classes):** for this assignment a "vehicle" is the COCO
**`car`** class **only**. Every interaction in this dataset is with a car (no trucks,
buses, motorcycles, or bicycles are involved as interaction targets), so restricting to
`car` avoids spurious detections without losing any real interaction. The counted class
set remains a documented, configurable project setting (default: `car`); it can be
widened to other COCO vehicle classes if the data changes. Note: kick-scooters have no
COCO class and are not treated as vehicles.

**Interaction = event, not state:** an interaction is a discrete event — `enter`
(board/get in), `exit` (alight/get out), or `other` contact — **not** the ongoing state
of riding/occupying. A person continuously riding a bicycle/scooter (or driving through)
with no visible mount/dismount is a **pass-by** and is not logged. A visible mount =
`enter`, dismount = `exit`.

**Merge rule (same person + same vehicle):** consecutive engagements by the *same person*
with the *same vehicle* are recorded as a **single interaction** when their detection
boxes stay close/overlapping throughout — a detector can't separate them at this stage
(no reliable door-open/closed or person-vs-vehicle-part signal). They are kept separate
only when the person's box **clearly separates** from the vehicle between engagements.
When merged, the type is the most salient event (e.g. `exit` over a following `other`).
Future improvement: a "vehicle open/closed" (door-state) classifier could enable finer
separation. See clip `HIu4lM4B8hA_1` (merged exit) for an applied example.

## Clip inventory

| clip_id | resolution | fps | duration (s) | frames |
|---|---|---|---|---|
| 1THkHYIQ_bY_0 | 1280×720 | 30.00 | 16.50 | 495 |
| gt1125_06 | 3840×2160 | 29.97 | 20.02 | 600 |
| HIu4lM4B8hA_1 | 352×288 | 25.00 | 10.00 | 250 |
| iMGR_0AG3a8_2_3 | 960×720 | 6.67 | 27.30 | 182 |
| mKzCQKTHizw_0 | 640×360 | 29.97 | 10.01 | 300 |
| mKzCQKTHizw_1 | 640×360 | 29.97 | 9.21 | 276 |
| NmlzoaDcOuI_1 | 1280×720 | 6.00 | 18.00 | 108 |
| NmlzoaDcOuI_6 | 1280×720 | 6.00 | 17.00 | 102 |

---

## Per-clip annotations

### 1THkHYIQ_bY_0

- **Camera POV:** fixed, static, high viewpoint (pole/building-mounted), filmed at night.
- **Scene:** a man removes a fabric cover from a private car, viewed from the car's left
  side. A floodlight faces the camera (possible glare/artifacts). Night → car reads
  gray/silver.
- **# interactions:** 1 (`other`; debatable — see scope decision).

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 | other | 0.0 | 4.2 | male, gray shirt, dark pants, dark hair | private car, gray/silver at night | Removing car cover (contact window 0–4.2 s, f0–f126). Afterward the cover catches on the undercarriage and he untangles it, then walks off carrying it (not counted). Counts as `other` only under broad scope. |

### gt1125_06

- **Camera POV:** aerial / drone view over a parking lot, daytime.
- **Scene:** busy lot with many parked cars and ≥5 people. Two men walk side by side
  toward the cars and each get into one. Background (non-interacting): a person sitting
  on the sidewalk, 2 kids riding scooters, and a cyclist appearing in the last frames.
- **# interactions:** 2 (both `enter`).

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 | enter | 6.41 | 10.01 | male, gray/brown shirt, white shorts (left of the pair) | white SUV with black roof, parked parallel to road (illegally) | Reaches car f192, fully inside f300. |
| 2 | enter | 7.61 | 17.22 | male, black shirt, light shorts (right of the pair) | black SUV, parked legally, perpendicular to road | Reaches car f228, closes door f516. |

### HIu4lM4B8hA_1

- **Camera POV:** fixed CCTV, wall/building-mounted, low quality, grayscale.
- **Scene:** a car theft. The female driver exits a parked car leaving the door open and
  walks around it; meanwhile a male enters through the open door and drives off. Two
  people interact with the **same** car (viewed from the rear), with overlapping time
  windows.
- **# interactions:** 2 (`exit`, `enter`). #1 and #2 overlap in time.
- **Merge note:** the female's exit and subsequent walk-around/touch are recorded as a
  **single `exit` contact window** — from this rear angle her detection box stays
  overlapping the car throughout, so the two can't be separated into distinct windows.

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 | exit | 0.0 | 7.44 | female (the driver) | parked car, viewed from the rear; grayscale → color N/A | Gets out (leaves door open) then walks around and touches the car — single contact window f0–f186. |
| 2 | enter | 3.6 | 4.32 | male, enters from the left | same car | In through the open door, f90–f108 (not visible inside by ~f108; end approximate, to tune vs. model). Drives off later (occupant/driving state, not counted). Overlaps #1. |

### iMGR_0AG3a8_2_3

- **Camera POV:** fixed CCTV, ceiling-mounted, indoor parking garage, low quality, RGB.
- **Scene:** a woman in white unloads a white car (exits, opens the trunk for a stroller,
  then takes a child from the rear door). Separately, a black car stops mid-lot and a
  woman in dark clothing gets out before it drives away. **Note:** a jump-cut artifact
  at ~15 s (content discontinuity) — expect tracks to break there.
- **# interactions:** 2 (1 for the white car, 1 for the black car). The white-car
  interaction overlaps the black-car exit in time but involves different people/cars.

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 | exit | 0.0 | 27.15 | woman in white | white car, viewed from the left | Merged exit + trunk + rear-door: exits driver side (f0), opens the trunk for a stroller, then opens the rear door and takes out a child — her box never separates from the car, so it's a single contact window **f0–f181** (runs to clip end / truncated; jump-cut at f100 within). |
| 2 | exit | 19.20 | 19.80 | woman in dark clothing | black car (stops mid-lot) | Gets out **f128–f132**; car then drives away (driver stays inside — not counted). Overlaps #1. |

### mKzCQKTHizw_0

- **Camera POV:** pole/wall-mounted CCTV, color, daytime, decent quality, but **PTZ /
  moving** — it pans and appears to zoom to follow the moving person (auto or manual).
  Camera egomotion present.
- **Scene:** a woman runs to a gray car that already has an open door and gets in, just
  as the clip ends. The car is viewed from the right and partly out of frame (only the
  rear wheel, rear door, and half the front door are visible).
- **# interactions:** 1 (`enter`).
- **Scene grouping:** same underlying scene as clip 6 (`mKzCQKTHizw_1`) — keep both on the
  **same side** of any train/val split to avoid leakage.

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 | enter | 9.08 | 9.98 | woman, gray top, khaki pants | gray car, right side, partially out of frame | Reaches the open door f272 and gets in; clip ends f299 (interaction truncated by video end). Moving/PTZ camera. |

### mKzCQKTHizw_1

- **Camera POV:** PTZ / moving CCTV, color, daytime — a **different angle of the same
  scene as `mKzCQKTHizw_0`** (clip 5).
- **Scene:** the same gray car, heavily cut off at the top of the frame, door already
  open (no person visible when it opens at ~2 s). A woman runs to the car and gets in,
  then it drives away.
- **# interactions:** 1 (`enter`).
- **Scene grouping:** same underlying scene as clip 5 — keep both on the **same side** of
  any train/val split to avoid leakage.

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 | enter | 4.50 | 6.24 | woman (running; likely the same person as clip 5) | gray car, heavily cut off at top of frame | Door opens ~2 s (no person visible — not logged); runs in, box first overlaps the car f135, inside/door closed by f187. Moving/PTZ camera. |

### NmlzoaDcOuI_6

- **Camera POV:** fixed, color, daytime, slightly above human height, low frame rate —
  **same camera/viewpoint as `NmlzoaDcOuI_1`** (clip 7), different car.
- **Scene:** a man exits a **red** car (viewed from behind-right, same angle as clip 7)
  and walks away.
- **# interactions:** 1 (`exit`).
- **Scene grouping:** shares the camera/location with clip 7 — keep both on the **same
  side** of any train/val split to avoid leakage.

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 | exit | 0.0 | 7.0 | man, green shirt, dark pants | red car, behind-right view | Exiting at clip start (f0); box stays on the car through the door-close (~f30) and separates when he walks away f42. |

### NmlzoaDcOuI_1

- **Camera POV:** fixed, color, daytime, slightly above human height, low frame rate
  (jumpy).
- **Scene:** a **driver swap**. A short-haired woman exits the passenger side, walks
  around the back, and later gets in on the driver side — but only after a long-haired
  woman (the driver) gets out of the driver door. Gray car viewed from behind-right
  (both right wheels + trunk visible). The driver door is open but obstructed early on
  (no visible person → not logged).
- **# interactions:** 3. #2 and #3 share the driver door and overlap in time (the swap).
- **Scene grouping:** shares the camera/location with clip 8 (`NmlzoaDcOuI_6`, a red car)
  — keep both on the **same side** of any train/val split to avoid leakage.
- **Test case (hard negatives):** other people pass in front of the car and their boxes
  overlap it *without* interacting — useful for tuning the overlap/duration thresholds
  (pass-by vs. real interaction).

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 | exit | 2.17 | 5.33 | woman, short hair, white shirt, dark pants | gray car, behind-right view (passenger door) | Exits passenger side (f13); box stays on the car as she walks around the back, separates when she leaves frame f32 (returns ~8 s near driver door). |
| 2 | exit | 12.0 | 16.67 | woman, long hair | same car (driver door) | Exits the driver door (f72), walks around the front; box separates f100. |
| 3 | enter | 11.0 | 12.83 | woman, short hair (same as #1) | same car (driver door) | Enters the driver door f66–f77 (not visible inside by f77), in parallel with #2 (driver swap). |

<!-- One section per clip, filled during review. Template:

### <clip_id>

- **Camera POV:** <fixed CCTV / dashcam / handheld / drone / ...>
- **Scene:** <short description>
- **# interactions:** <n>

| # | type | start (s) | end (s) | person | vehicle | notes |
|---|------|-----------|---------|--------|---------|-------|
| 1 |      |           |         |        |         |       |

-->
