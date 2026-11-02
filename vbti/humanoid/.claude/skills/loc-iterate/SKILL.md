---
name: loc-iterate
description: Run ONE pair-mode development cycle on a locbench localization candidate — orient from disk, propose a single change, run --smoke, read the report AND look at the plots, stop at the improve/pivot/abandon decision, journal it. Use when developing/iterating a localization realization for the locbench loop (playbook phases 3–7), e.g. "iterate rtabmap", "loc-iterate <name>", "improve the reference candidate", "run a locbench cycle".
---

# /loc-iterate — one pair-mode cycle on a locbench candidate

Runs phases **3–7** of the localization dev-loop **in pair mode**: you do all the mechanical
work and present analysis; **Anton makes the improve/pivot/abandon call.** Nothing is decided
autonomously — there are no Promoted Rules yet, so every phase-7 verdict stops for Anton (this is
the capture-first bootstrap: memory [[locdev-bootstrap-capture-plan]]).

**Read first (all bind here; this file only sequences the steps):**
- `logic/oli/reason/localization/realizations/AGENTS.md` — the playbook (the 7-phase loop, the
  iteration discipline, the autonomy boundary, Promoted Rules).
- `logic/locbench/AGENTS.md` — the oracle (CLI surface + the integrity invariants you must NEVER
  touch).
- `logic/oli/reason/localization/AGENTS.md` — the module contract (two sockets, invariants).

## Session marker — do this FIRST, every run

Open the session with one line so it is collectable later for rule extraction
([[locdev-session-keyword]]):

```
#locdev candidate=<name>  — loc-iterate pair cycle
```

## Input

- **name** — an existing `realizations/<name>/`. If it doesn't exist, stop and point at `/loc-new`.

## Preconditions (check, don't assume)

- `realizations/<name>/` exists; its `bench-<name>` env exists (`locbench env create <name>` if
  not — the run boots the brain there and hard-errors otherwise).
- Contract is green in the candidate env:
  `conda run -n bench-<name> pytest logic/oli/reason/localization/realizations/<name>/ -q`.
  Red contract ⇒ fix that first; accuracy work is meaningless until the module conforms.

## Steps (one cycle)

1. **Orient — purely from disk (resume-safe).** Read, in order: `README.md` (current truth +
   hypothesis), the **tail of `JOURNAL.md`** (last iteration's decision + next step), and the
   **latest run dir** under `logic/locbench/runs/<name>/` (its `report.json` + plots) if one
   exists. Re-running this skill after a killed session must reconstruct state from these alone —
   never from memory of the prior turn.

2. **Detect the phase — bring-up vs tune.** If the candidate has never completed a smoke episode
   without crashing at coverage > 0, you are in **bring-up**: ignore accuracy gates entirely,
   chase "runs at all". Otherwise you are **tuning** against the gates. `report.json` tells you
   which — say which one out loud before proposing anything.

3. **Propose exactly ONE change.** One `config.yaml` value **or** one code fix in `module.py` —
   never both, never several. State it in one line (what + why) and the hypothesis ("expect this
   to move <metric> because …"). Wait for Anton's go before editing. Tunables live in
   `config.yaml`; a hardcoded magic number in `module.py` is a defect to fix, not add.

4. **Apply** the one change.

5. **Run the smoke grind:** `python -m humanoid.logic.locbench run <name> --smoke 3`. It boots the
   whole stack (World + brain-in-`bench-<name>` + evaluator) as one subprocess; the exit code is
   the verdict, artifacts land in a fresh `runs/<name>/<run-id>/`.

6. **Analyze — plots are senses.** Read `report.json` (tier, per-episode pos/yaw/coverage, failed
   gates) AND **open the plot PNGs with the Read tool** (overlay = GT vs est on the map with LOST
   stretches marked; error timeline vs gate lines) — cite their absolute paths. Name the failure
   mode in **one sentence** ("LOST for ~12 s entering the north aisle", not "max err 0.4 m").

7. **STOP for the phase-7 decision.** Present: phase (bring-up/tune), the one-sentence failure
   mode, the key plot, and your recommendation (improve → next single change / pivot → different
   approach / abandon → gate looks infeasible). Then **wait for Anton.** Do not decide. (When
   Promoted Rules exist, some of these become unattended — not yet.)

8. **Journal + commit** after Anton decides. Append one `JOURNAL.md` entry in the schema
   (hypothesis → change → run-id → result vs previous → **decision (anton)** → reasoning → next).
   Commit the change + journal entry together on the branch (one iteration = one commit).

## Full-certify path

`--smoke` PASS means **nothing** until the full set confirms it. On a smoke PASS, propose the full
run (`run <name>` with no `--smoke`) before any tier claim or board row. Board rows come from full
runs only.

## Hard boundaries (from the playbook — repeated because they get broken mid-iteration)

- **Oracle integrity.** Never edit `logic/locbench/`, gate thresholds, or `episodes/<scene>.json`
  to make a candidate pass. Believe a gate is wrong? Journal it and PAUSE for Anton — never act.
- **One change per iteration.** This is what makes the journal causally readable. No exceptions.
- **Stop at the decision.** Phase-7 verdicts are Anton's in pair mode. An unattended verdict with
  no Promoted Rule authorizing it is a bug, not initiative.
- **Never touch other candidates' folders, or the harness.** Your scope is `realizations/<name>/`.
- **Deleting anything — envs, maps, runs, the realization — is Anton's call.** Report, don't `rm`.
