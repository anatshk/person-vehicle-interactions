# Outputs

Cached **detection + tracking** outputs, committed so reviewers can skip the slow detection
pass (~73 min total on this CPU — see [`../docs/timing_findings.txt`](../docs/timing_findings.txt)).

## Structure

```
outputs/
  <clip_id>/
    <clip_id>.csv        # tracked person/vehicle boxes per frame (the detection+tracking output)
    <clip_id>.meta.json  # clip + detection config: fps, resolution, frame count, model, seed, ...
    <clip_id>_interactions_<method>_<YYYYMMDDHHMM>.json   # results JSON; <method> = the
                         # description backend: fast (YOLO-World) / detailed (moondream2) /
                         # placeholder. Both fast + detailed results are committed here.
  sheets/
    README.md            # legend: what each sheet's title / tile captions / metrics mean
    <clip_id>/*.jpg      # annotated interaction contact sheets, one per interaction window (JPEG q90)
```

Classify a clip's cached tracks straight into a results JSON, no detection needed:

```
python -m scripts.detect_interactions classify-tracks outputs/<clip_id>/<clip_id>.csv
```

The per-clip results JSON is written alongside the tracks. (`classify-tracks` uses model-free
placeholder descriptions; real descriptions come from the full `run` with `--fast`/`--detailed`.)

## Thresholds

Two different kinds of threshold are involved, and **both are global — the same for every clip,
neither is LOSO**:

- The `confidence_threshold` (0.25) and `iou_threshold` (0.7) in each `.meta.json` are the
  **YOLO detection / NMS** thresholds used during detection — fixed detector settings, not
  interaction thresholds.
- The **interaction** thresholds (overlap / distance / duration / confidence / gap) are the
  single shipped set fit on *all* clips (`config.SHIPPED_THRESHOLDS`), applied at classify time.
  The per-clip LOSO fold thresholds are a leakage-free **evaluation** device only (see the
  write-up) and are never used to produce these outputs.
