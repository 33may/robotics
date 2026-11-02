# realizations/ — candidate localizers & the development-loop playbook

One folder = one candidate localization algorithm. This file is the **playbook** every dev-flow
agent reads before touching a realization: the loop protocol, the discipline, and the
**Promoted Rules** distilled from real iterations. It starts thin **on purpose** — nobody knows
the SLAM improve-vs-abandon rules for this project yet (locdev-flow design D1), so the playbook
*captures* judgment from journaled decisions instead of pretending to encode it upfront.

Technical contract rules live in `../AGENTS.md` (two sockets, invariants, conformance layers) —
read that first; this file never duplicates it. The oracle is **locbench**
(`logic/locbench/`, change `may-173-locbench-harness`): frozen episodes, raw two-tier gates,
`report.json` + plots. Your job is to make a candidate pass it — never to change it.

## Who decides what (the autonomy boundary)

| action | who |
|---|---|
| research, scaffolding, vendor clone, env build, code edits, smoke + full runs, journaling, commits | agent, autonomous |
| phase-7 verdicts (improve / pivot / abandon) | Anton in pair mode; unattended only via a **Promoted Rule** (cite its id in the journal), else **PAUSE + ping** |
| promoting a rule into this file | agent proposes, Anton approves |
| DEPLOY tier declaration, Stage-2 `--localizer` flip | **always Anton** |
| deleting anything — envs, maps, runs, realizations | **always Anton** |

## The loop (one phase at a time; never skip 6 or 7)

| # | phase | do | done when |
|---|---|---|---|
| 1 | Research | read `docs/research/localization.md`, the board, every prior realization's README + JOURNAL tail | you can say why THIS candidate, why now |
| 2 | Scope | write the README: approach, hypothesis ("expect PASS because …"), expected failure modes | hypothesis on disk **before** building |
| 3 | Build | find a working implementation and adapt it (checklist below); `/loc-new` scaffolds the folder | `module.py` wraps the algorithm behind `LocalizationModule` |
| 4 | Refine | env recipe builds; in-folder `test_contract.py` green **inside the candidate env** | bring-up: `step()` emits poses, coverage > 0, no crash |
| 5 | Run | `locbench run <name> --smoke=3` to grind; the full episode set only to certify | `report.json` + plots exist for the run |
| 6 | Analyze | read `report.json` AND **look at the plots** (overlay PNG, error timeline) | the failure mode is *named in one sentence*, not just numbers quoted |
| 7 | Decide | improve / pivot / abandon — feasibility of the gate is the question, not just "what next tweak" | JOURNAL entry appended, decision + reasoning + who |

**Bring-up before tune.** Until a candidate completes an episode without crashing and with
coverage > 0, you are in bring-up: ignore accuracy gates entirely, chase "runs at all".
Only then switch to tuning against the gates. The report tells you which phase you are in.

## Build strategy — adapt-existing-first

Order of preference — journal which you took and why:

1. **Maintained library with Python bindings**, pip-installable at a pinned version.
2. **Working repo cloned into `vendor/`** and patched in place.
3. **From scratch** — last resort; requires a journal entry explaining why 1 and 2 failed.

Adoption checklist, before any code lands:

- **License**: BSD/MIT/Apache fine; GPL/AGPL or unclear → flag to Anton before adopting.
- **Alive**: commits within ~2 years, issues get answers, builds reproduce.
- **ROS-free path exists** — the brain imports no ROS (invariance boundary).
- **Known-map localization mode** actually exists (not mapping-only demos).
- Runs on our stack: Linux, Python ≥ 3.10 or C++ with usable bindings, no exotic hardware.

Vendor hygiene (all mandatory):

- strip the clone's `.git/` — `vendor/` is a snapshot, never a nested repo or submodule;
- write `vendor/UPSTREAM.md`: url, commit sha, license, why this repo, list of our patches;
- gitignore fat upstream assets (models, datasets, sample bags); code + patches are committed;
- keep patches minimal — every divergence from upstream is a line in `UPSTREAM.md`.

## Iteration discipline

- **ONE change per iteration.** One config value, or one code fix — never both, never several.
  This is what makes the journal causally readable and the loop debuggable.
- **All tunables live in `config.yaml`.** A tuning iteration's diff is one file; parameter
  history is reconstructable from git. Hardcoded magic numbers in `module.py` are a defect.
- **Smoke-grind, full-certify.** Iterate on `--smoke=3`; a smoke-PASS means nothing until the
  full episode set confirms it. Board rows and tier claims come from full runs only.
- **Plots are senses.** Every analyze phase reads the overlay and timeline images, not only
  `report.json`. "LOST for 12 s entering the north aisle" lives in the plot, not in max-err.
- **Oracle integrity.** Never edit `logic/locbench/`, gate thresholds, or episode sets. If you
  believe the harness or a gate is wrong, journal it and PAUSE for Anton — never act on it.
- **Commit per iteration** on the work branch (change + journal entry together), squash at merge.

## Conformance testing — where it runs

- The shared checker is `../testing.py::verify_module_contract` — every realization's
  `test_contract.py` calls it. `stop()` must tolerate a failed `start()` (see `../AGENTS.md`).
- **Pure realizations** (`_template/`, `reference/` — stdlib/numpy only) are also imported by
  the repo's brain-marked suite (`tests/oli/reason/localization/test_realizations_template.py`).
- **Dep-heavy candidates** (RTAB-Map, cuVSLAM, …) must NOT be importable from the brain env —
  their `test_contract.py` runs inside their own env:
  `conda run -n bench-<name> pytest logic/oli/reason/localization/realizations/<name>/`.
  Never add a brain-env import (incl. static canaries in `test_architecture.py`) for these.

## Folder anatomy

| file | role |
|---|---|
| `module.py` | the `LocalizationModule` adapter + `build(config) -> Module` registry entrypoint (`config` = parsed `config.yaml` + overrides — every knob routes through it) |
| `config.yaml` | ALL tunables; the unit of a tuning iteration |
| `environment.yml` (+ `build.sh`) | recipe for the disposable `bench-<name>` env |
| `lock.yml` | post-build `conda env export` — committed, "what exactly was built" |
| `vendor/` | adapted upstream (see hygiene above); absent when pip-pinning suffices |
| `map/` | cached `map_dir` from the mapping pass — gitignored, hash pinned in reports |
| `test_contract.py` | conformance gate, runs in the candidate's env |
| `README.md` | current truth: approach, hypothesis, status, best full-run numbers |
| `JOURNAL.md` | append-only iteration log (schema below) — the capture instrument |

## JOURNAL.md — entry schema (append-only, one entry per iteration)

```markdown
## it-<N> — <YYYY-MM-DD> — run <run-id|none>
hypothesis: <what we believed going in>
change:     <the ONE thing changed>
result:     <verdict + key numbers, vs previous iteration>
decision:   <improve|pivot|abandon|promote> (anton | agent:rule-<id>)
reasoning:  <why — this is the payload the promotion protocol mines>
next:       <the next hypothesis>
```

`decision (who)` is the autonomy audit trail: pair-mode entries say `anton`; unattended entries
must cite the Promoted Rule that authorized them — an entry with neither means the run should
have paused, and that is a harness bug to report.

## Promotion protocol (how this playbook grows)

1. While journaling, check past entries (this and other realizations) for the same decision
   pattern. Seen **twice or more** → draft a rule.
2. Propose it to Anton: the rule, the journal entries that evidence it, the phase it applies to.
3. On approval, append it under Promoted Rules as `rule-<NNN>` with links to its evidence.
4. Rules that stop matching reality get retired (struck through, reason noted) — never silently
   deleted.

## Promoted Rules

*(empty — rules are earned from journaled iterations, not written in advance)*

| id | phase | rule | evidence |
|---|---|---|---|
