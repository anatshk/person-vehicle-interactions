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
- Sort imports with **isort** using `profile = "google"` **plus `force_single_line = false`**
  — group names from the same module into one parenthesized `from x import (a, b, ...)`
  block (keep compatible with Black).
- **Type hints are required** on all function/method signatures.
- **Google-style docstrings** on modules and public functions/classes. **Multi-line
  docstrings start with a line break** after the opening `"""` (summary on the next line,
  not the first); single-line docstrings stay on one line. _(A deliberate deviation from
  PEP 257 / Google's summary-on-first-line; not auto-enforced by Black/isort.)_
- Use **meaningful, descriptive variable and function names** (per the Google style
  guide, but worth repeating) — avoid abbreviations and single-letter names except
  conventional short-lived loop indices.
- Prefer clear, readable code over cleverness; this is a take-home, not production.

## Making changes

- **Minimal diffs:** make the smallest change that accomplishes the task. Don't refactor,
  rename, or reformat code unrelated to the request.
- **Confirm before multi-file edits:** if a change would touch several files (e.g. a
  rename or signature change with multiple call sites), don't apply it automatically —
  pause, surface the affected scope (like PyCharm's rename warning when the scope is
  large), and proceed only on confirmation.
- **Reuse before adding:** prefer reusing existing code; generalize an existing function
  rather than adding a near-duplicate.

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
- **Task transitions:** when moving from one task/phase to the next, **remind me to compact
  and clear the conversation context**, and first **write/update a handover file** (e.g.
  `CURRENT_STATUS.md`) capturing current state + next steps, so the fresh context resumes
  cleanly.

## Work log

- Maintain **`WORKLOG.md`** (reverse-chronological, newest first) — **concise highlights**,
  not blow-by-blow. Each entry: a short **narrative** of what was done (e.g. "module X +
  tests", "added CI") and the **decisions** made. Skip mechanics that PRs/CI already capture.
- **Alternatives:** include them only when they add real information — **ask before adding**,
  don't include by default.
- This log feeds the final write-up.
- `WORKLOG.md` and `PLAN.md` updates are batched into a **periodic docs PR**, not committed
  loosely to `main`.

## Commits & git

- Work lands via **reviewed PRs** merged on GitHub (feature branches off `main`).
- **Merge strategy:** prefer **"Rebase and merge"** on GitHub (not squash). For **stacked
  PRs**, **land the base PR before the one stacked on it** — merging a stacked PR into a
  base branch that is then squash-merged/deleted can **silently drop the stacked PR's
  commits from `main`** even though GitHub marks it "merged" (this happened with #14 and had
  to be re-landed). Keep stacks shallow, or rebase onto `main` and retarget as bases land.
- **PR comment replies** are posted via `gh` under the authenticated (human) account, so
  tag each reply with **`[Claude-<model>]`** (currently `[Claude-Opus-4.8]`) so it's
  clearly Claude replying, not the reviewer replying to themselves.
- Use **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, ...).
- Keep commits small and focused.
- Keep messages concise: a good title, a short "Set up/..." sentence if useful, and
  **terse bullet points** — name the file/change, don't explain it, no "Add" prefix.
- **Commit only when explicitly asked.** Never commit automatically.
