# TODO — Assignment Deliverables Checklist

Concise tracker of what the assignment requires. Verify all before submitting.

**Status (2026-08-04):** everything below is done **except making the repo public** — the one
remaining pre-submission action. Both run modes (full `run` and cached-tracks
`classify-tracks`) now emit real `--fast`/`--detailed` descriptions, verified end-to-end on
`NmlzoaDcOuI_6`.

## Deliverables
- [ ] Public **GitHub repo** (submit the link) — still private; make public before submitting
- [x] **Python source** that reproduces the results
- [x] **README**: environment setup, run commands, external assets (e.g. `yolo11l.pt`)
- [x] **Machine-readable output**: list of all person-vehicle interactions (committed under `outputs/`)
- [x] **Write-up (≤2 pages)**: approach, assumptions, limitations, next steps
- [x] Optional: visualizations / annotated frames (committed under `outputs/sheets/`)

## Output fields (per interaction)
- [x] `clip_id`
- [x] time span(s) or frame range(s)
- [x] person description
- [x] vehicle description

## Rules / compliance
- [x] Deterministic & reproducible; **seed fixed**
- [x] Document any external services / APIs / pretrained models used
- [x] Ambiguities resolved and **documented** (decision + alternatives) — see WORKLOG / GT
- [x] Scope kept reasonable (take-home, not production)
