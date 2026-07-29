# Work-with-Claude Log

A running, reverse-chronological log (newest first) of what was asked, decisions
made (with alternatives considered), outcomes, and reflections. Feeds the final
write-up. One entry per working session or significant request.

Entry template:

```
## YYYY-MM-DD — Session N: <short title>

**Asked:** <what I was asked to do>

**Decisions:**
- <decision> — <rationale>. _Alternatives: <options considered>._

**Outcome:** <what changed / current state>

**Reflect / next:** <observations, follow-ups>
```

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
