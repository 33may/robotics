---
name: loc-new
description: Scaffold a new localization candidate realization from the _template — folder, vendor clone with provenance, env recipe, hypothesis README, journal seed. Use when starting a new localization algorithm/candidate for the locbench loop (phases 2–3 of the playbook), e.g. "scaffold rtabmap", "new localization candidate", "loc-new <name>".
---

# /loc-new — scaffold a localization candidate

Creates `logic/oli/reason/localization/realizations/<name>/` ready for the development loop.
This skill is phases 2–3 *setup* of the playbook — read
`logic/oli/reason/localization/realizations/AGENTS.md` (the playbook) AND
`logic/oli/reason/localization/AGENTS.md` (the contract) before running it. All rules there
bind here; this file only sequences the mechanical steps.

## Session marker — do this FIRST

Open the session with one line so it is collectable later for rule extraction
([[locdev-session-keyword]]):

```
#locdev candidate=<name>  — loc-new scaffold
```

## Inputs (ask if missing — one question at a time)

1. **name** — snake_case, valid Python identifier (it becomes a package: `realizations/<name>/`).
2. **the research pick** — approach + upstream (repo URL / library), from `docs/research/localization.md`,
   the board, or an invention proposal. No pick ⇒ stop: run the research phase first, this skill
   does not choose algorithms.

## Steps

1. **Preconditions**
   - `realizations/<name>/` must not exist; name not `_template`, valid identifier.
   - Read the playbook's build-strategy checklist; run it against the upstream pick
     (license, alive, ROS-free path, known-map mode, stack fit). Any red flag → stop and
     surface to Anton before scaffolding.

2. **Copy the template**
   - `cp -r realizations/_template realizations/<name>` (drop `__pycache__` if present).
   - Replace every `<name>` placeholder across the copied files with the candidate name.
   - Rename the module class `TemplateModule` → `<PascalName>Module` in `module.py` (including
     the `build(config)` registry entrypoint's return) and in `test_contract.py`'s absolute
     import (both the class name and the `_template` path segment); keep it the conforming
     no-op stub — adaptation of the real algorithm is loop work (phase 3–4), not scaffold work.

3. **Vendor the upstream** (only when adapting a repo; pip-pinned libs skip this)
   - `git clone --depth 1 <url> realizations/<name>/vendor/<repo>`
   - Capture `git -C … rev-parse HEAD` **before** `rm -rf vendor/<repo>/.git` (snapshot, never
     a nested repo/submodule).
   - Write `vendor/UPSTREAM.md`: url, sha, license, why this repo (vs the alternatives), empty
     "our patches" list.
   - Append fat upstream assets (models/datasets/sample data) to `realizations/.gitignore`
     as `<name>/vendor/<paths>` entries.

4. **Env recipe**
   - `environment.yml`: env name `bench-<name>`; add the candidate's pinned deps
     (prefer pip-pinned; `git+…@sha` when unpackaged); `build.sh` for compile/patch steps.

5. **README = the hypothesis (phase 2, before building)**
   - Fill Approach / Hypothesis / Expected failure modes from the research pick. The hypothesis
     must reference the board: what should this do better than what's been tried?

6. **Seed the journal**
   - Append `it-0` to `JOURNAL.md`: change = "scaffolded from _template @ <git-sha>",
     hypothesis = the README's, decision = n/a, next = first bring-up step.

7. **Validate the scaffold**
   - The untouched stub is pure — run its `test_contract.py` in the brain env:
     `conda run -n brain python -m pytest logic/oli/reason/localization/realizations/<name>/ -q`
     → must be green (a scaffolded candidate starts contract-green, accuracy-red).
   - Once locbench's `env create` exists: `locbench env create <name>` + rerun the test inside
     `bench-<name>` is the full 3.3 dry-run.

8. **Report** — folder tree, upstream provenance line, the hypothesis, and the next loop step
   (adapt `module.py` in phase 3–4 via `/loc-iterate` when it exists; until then, pair-mode by hand).

## Hard boundaries (from the playbook — repeated because scaffold time is when they get broken)

- Never scaffold without a written hypothesis — phase 2 precedes phase 3.
- Never pick the algorithm inside this skill.
- Never touch `logic/locbench/`, episode sets, other candidates' folders.
- Deleting a botched scaffold is Anton's call, not yours — report, don't `rm`.
