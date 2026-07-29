# Project: Person-Vehicle Interaction Detection

Take-home assignment. Given short MP4 clips (in `Videos/`, no audio), write Python
that takes a clip as input and outputs a machine-readable list of human-vehicle
interactions (e.g. a person entering/exiting a vehicle counts; a person merely
passing by does not).

Per interaction the output must include: `clip_id`, time span(s) or frame range(s),
a description of the person(s) involved, and a description of the vehicle involved.

Deliverables: reproducible Python source + README (setup & run instructions),
machine-readable output artifact, and a brief write-up (≤2 pages) covering approach,
assumptions, limitations, and next steps. `Videos/` is git-ignored input data.

## Code style

- Follow the **Google Python Style Guide** (https://google.github.io/styleguide/pyguide.html).
- Format with **Black**, line length **88**.
- Sort imports with **isort** using `profile = "google"` (keep compatible with Black).
- **Type hints are required** on all function/method signatures.
- **Google-style docstrings** on modules and public functions/classes.
- Prefer clear, readable code over cleverness; this is a take-home, not production.

## Environment & dependencies

- Use **venv + pip** (not Poetry).
- Runtime dependencies are declared in **`requirements.txt`** with pinned versions.
- **`pyproject.toml` is used for tool configuration only** (Black, isort, pytest, etc.),
  not for dependency management.
- Standard setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Determinism & reproducibility

- Optimize for clarity, determinism, and reproducibility.
- If any randomness is involved, **fix a seed**.
- Pin dependencies (e.g. `requirements.txt`) and document external models/services in
  the README.

## Testing (TDD)

- Work **test-driven**. For each feature, follow the red-green-refactor cycle:
  1. **Red** — write the test(s) first and run them to confirm they *fail* for the
     right reason before writing any implementation.
  2. **Green** — write the minimal code to make the tests pass.
  3. **Refactor** — clean up while keeping the tests green.
- Use **pytest** for the core interaction-detection logic.
- Do not write implementation code for a feature before its failing test exists.
- Glue/IO code does not need exhaustive tests, but core logic must be covered first.

## Workstyle

- **Plan first, then execute**: for any non-trivial change, propose a short plan and
  wait for approval before writing code.
- When an ambiguity arises in the task, resolve it, document the decision and the
  alternatives considered (in the write-up), and proceed.
- Keep scope reasonable — favor a clear, working pipeline.

## Work log

- Maintain **`WORKLOG.md`** (reverse-chronological, newest first).
- At the end of each working session or significant request, add/update an entry using
  the template in that file: **Asked / Decisions (+ alternatives) / Outcome / Reflect-next**.
- This log feeds the final write-up (assumptions and alternatives considered).

## Commits & git

- Use **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, ...).
- Keep commits small and focused.
- Keep messages concise: a good title, a short "Set up/..." sentence if useful, and
  **terse bullet points** — name the file/change, don't explain it, no "Add" prefix.
- **Commit only when explicitly asked.** Never commit automatically.
