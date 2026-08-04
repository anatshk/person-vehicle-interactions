# Outputs

Cached **detection + tracking** outputs, one folder per clip
(`<clip_id>/<clip_id>.csv` + `<clip_id>.meta.json`), committed so reviewers can skip the slow
detection pass (~73 min total on this CPU — see [`../docs/timing_findings.txt`](../docs/timing_findings.txt)).

Classify a clip's cached tracks straight into a results JSON, no detection needed:

```
python -m scripts.detect_interactions classify-tracks outputs/<clip_id>/<clip_id>.csv
```

The per-clip results JSON (`<clip_id>_interactions_<YYYYMMDDHHMM>.json`) is written alongside
the tracks. (`classify-tracks` uses model-free placeholder descriptions; real descriptions
come from the full `run` with `--fast`/`--detailed`.)
