# Project: Person-Vehicle Interaction Detection

Take-home assignment. Given short MP4 clips (in `Videos/`, no audio), write Python that takes
a clip as input and outputs a machine-readable list of human-vehicle interactions — a person
entering or exiting a vehicle counts; a person merely passing by does not.

Per interaction the output must include `clip_id`, the time span(s) or frame range(s), a
description of the person(s) involved, and a description of the vehicle involved.

Deliverables: reproducible Python source plus a README with setup and run instructions, the
machine-readable output artifact, and a write-up (≤2 pages) covering approach, assumptions,
limitations, and next steps.

## Deviations from my global conventions

- **This is a take-home, not production.** Keep scope reasonable and favor a clear, working
  pipeline over completeness. Readability beats cleverness at this size.
- **The handover file here is `CURRENT_STATUS.md`**, not `NEXT_STEPS.md`.
- **`docs/WORKLOG.md` feeds the final write-up**, so decisions recorded there need to be
  reusable prose, not just notes to self. When an ambiguity is resolved, document the
  decision *and the alternatives considered* in the write-up.
- **mypy is part of the check loop** (`.venv/bin/mypy person_vehicle_interactions`), not just
  Black and isort.
- PR comment replies are tagged `[Claude-Opus-4.8]`.
- Stacked-PR caution is not theoretical here: #14 was marked merged by GitHub but its commits
  were silently dropped from `main` when its squash-merged base branch was deleted, and it had
  to be re-landed. Land base PRs first; keep stacks shallow.

## Local state — git-ignored, not in the repo

- **`Videos/`** — the provided input clips. Supplied separately; never commit them.
- **`*.pt`** — YOLO / world-model weights, auto-downloaded on first run.
- **`cache/`** — generated track cache. `tests/videos/` — synthetic test video.
- **`outputs/`** and **`ground_truth/`** hold committed results and hand-labeled GT; check
  before regenerating, since a rerun can churn them.

## Running

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

python -m scripts.detect_interactions <video|folder>       # full pipeline
python -m scripts.detect_interactions detect-and-track <v>  # video -> tracks
python -m scripts.detect_interactions classify-tracks <t>   # tracks -> results
python -m scripts.show_interactions [--clip <clip_id>]      # inspect results
python -m scripts.time_pipeline                             # -> docs/timing_findings.txt
```

Checks before a commit:

```bash
.venv/bin/pytest -q
.venv/bin/black --check person_vehicle_interactions tests
.venv/bin/isort --check-only person_vehicle_interactions tests
.venv/bin/mypy person_vehicle_interactions
```

Tests marked `integration` need cv2/ultralytics/model weights and are skipped in lean CI.
