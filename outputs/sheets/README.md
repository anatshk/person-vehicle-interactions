# Interaction contact sheets

One JPEG per interaction window, produced by `scripts.show_interactions` with the shipped
thresholds. Each sheet is a montage of sampled frames with the person box drawn in **green**
(`p<id>`) and the vehicle box in **red** (`v<id>`).

## Filename

```
<clip_id>_<person_id>_<vehicle_id>_<start>-<end>_<TP|FP>.jpg   # a predicted interaction
<clip_id>_gt<gt_id>_<start>-<end>_FN.jpg                       # a missed ground-truth window
```

`<start>-<end>` is the interaction's frame range; `TP`/`FP`/`FN` is how it scored against the
ground truth (true positive / false positive / false negative).

## Sheet title

```
<clip_id> | p<person>×v<vehicle> | f<start>-<end> (<N>f/<S>s) | <TP|FP|FN> | thr: <...> | <values>
```

- `p<person>×v<vehicle>` — the person and vehicle **track ids** in the pair (`gt` on an FN sheet).
- `f<start>-<end> (<N>f/<S>s)` — the window's frame range, its length in frames and seconds.
- `TP|FP|FN` — the match label vs ground truth.
- `thr: ov≥ dist≤ dur≥ conf≥ gap≤` — the deciding **thresholds** (the shipped set):
  - `ov≥`   — min normalized **overlap** to count as contact,
  - `dist≤` — max normalized center-**distance** when the boxes don't overlap,
  - `dur≥`  — min **duration** in frames,
  - `conf≥` — min detection **confidence**,
  - `gap≤`  — max frame **gap** bridged when merging a contact.
- `<values>` — what the window actually measured:
  - `peakOv=` the peak overlap and `minDist=` the minimum distance over the window; or,
  - on an FN, `person not detected in span` / `sub-threshold: peakOv=.. minDist=.. (best p..×v..)`.

## Per-tile caption

```
f<frame> <pre|in|post>
ov=<overlap> d=<distance>          (TP/FP tiles only)
```

- `f<frame>` — the frame number; `pre` / `in` / `post` — a context frame just before the span,
  one of 5 frames sampled evenly **inside** the span, or a pad frame just after it (3 each side).
- `ov=` / `d=` — that frame's overlap and distance.

## Metric definitions

- **overlap** = `intersection(person, vehicle) ÷ person-box area` — the fraction of the person
  inside the vehicle box (more sensitive than IoU when a person is far smaller than a car).
- **distance** = `center-to-center distance ÷ vehicle-box diagonal` — scale- and
  egomotion-invariant, no calibration needed.

Both are the signals the thresholds above are applied to. Colours: **person = green**,
**vehicle = red**.
