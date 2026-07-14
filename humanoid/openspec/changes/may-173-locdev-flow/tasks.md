# Tasks — locdev-flow (build order; gated on locbench §8 reference-triplet green)

## 1. OpenSpec + coordination

- [x] 1.1 `proposal.md` / `design.md` authored from the 2026-07-13 user-story session — Anton to review
- [ ] 1.2 `specs/localization-dev-flow/spec.md` delta (SHALL requirements + scenarios, may-149 format)
- [ ] 1.3 Move locbench task 11.1 here (note in the sibling change's tasks.md) — build gate revised (Anton, 2026-07-13): **§2–§3 build now in parallel** (locbench-independent: playbook + template/scaffolder); §4 waits for the report schema to settle; §5+ waits for locbench §8
- [ ] 1.4 Branch `33may/may-173-locdev-flow` off main; merge at working states

## 2. Playbook v1 — `realizations/AGENTS.md` (D1, D3–D6)

- [x] 2.1 The 7-phase loop written as agent instructions: per phase — inputs, outputs, done-criteria, what to read (report fields, which plots, journal)
- [x] 2.2 Build strategy section: adapt-existing-first checklist (repo vitality, license, bindings, ROS-free path), vendor hygiene (strip `.git`, `UPSTREAM.md`, fat-asset gitignore)
- [x] 2.3 Iteration discipline section: one-change rule, config.yaml-only tunables, smoke-grind/full-certify, bring-up-before-tune
- [x] 2.4 Hard rules: oracle integrity (never touch locbench/gates/episodes), deletion always Anton, DEPLOY/Stage-2 always Anton (+ conformance-split: pure realizations in the brain suite, dep-heavy ones tested inside their own env)
- [x] 2.5 JOURNAL.md schema (D5) + promotion protocol (journal pattern ×2 → proposed rule → Anton approves → Promoted Rules section); Promoted Rules starts empty
- [ ] 2.6 One-line `CLAUDE.md` shim (done); Anton reviews the playbook end-to-end

## 3. Realization template + `/loc-new` (D3, D9)

- [x] 3.1 `realizations/_template/`: `module.py` stub (contract-conformant no-op + `build(config)` registry entrypoint), `config.yaml`, `environment.yml`/`build.sh` seeds, README/JOURNAL seeds with schema headers — TDD'd via `tests/oli/reason/localization/test_realizations_template.py` (brain-marked, 5 green) + in-folder `test_contract.py` (absolute import — relative `.module` breaks standalone runs on dual-namespace identity)
- [x] 3.2 `/loc-new <name>` skill: scaffold from template → vendor clone + `UPSTREAM.md` + gitignore entries → env recipe draft → README hypothesis section filled from the research pick
- [ ] 3.3 Dry-run: contract-conformance half DONE (template green standalone + in-suite); `locbench env create` half blocked until locbench §7 lands

## 4. Pair-mode loop — `/loc-iterate` (v1; D1, D2, D4)

- [ ] 4.1 Skill flow: orient (README + JOURNAL tail + last report/plots) → phase detect (bring-up vs tune) → propose the one change → edit → `locbench run --smoke` → read report + look at plots → failure-mode summary → **stop for the phase-7 discussion** → journal entry (decision+who+reasoning) → commit
- [ ] 4.2 Workflow-tool fan-outs inside the skill for mechanical stretches only (parallel log/plot analysis, research lookups) — no decisions in subagents
- [ ] 4.3 Full-certify path: on smoke-PASS, run full 10 before claiming the tier; board row only from full runs
- [ ] 4.4 Resume-safety: killed session → re-running `/loc-iterate` re-orients purely from disk state (journal + run dir); verify by killing one mid-iteration

## 5. Flow self-test on the reference candidate (the gate for v1)

- [ ] 5.1 Mis-configure `realizations/reference/` (e.g. 0.2 m bias in config) and run `/loc-iterate` pair-mode cycles until PASS: proves orient → diagnose → one-change → rerun → journal end-to-end with zero SLAM deps
- [ ] 5.2 Review the produced JOURNAL with Anton: entries causally readable, decisions + reasoning captured — the capture instrument works before real candidates exist

## 6. Earned autonomy — `loc-enforce` workflow (v2; D1, D2, D7) — gated on Promoted Rules being non-empty

- [ ] 6.1 `loc-enforce.js`: loop of iteration-subagents (sense → decide-within-playbook → act → read), budget-capped, stop on PASS / budget / non-covered decision
- [ ] 6.2 Pause + Telegram ping (summary + key plot + recommendation) via the telegram plugin; resume on reply; verify pause state survives a dead session
- [ ] 6.3 Decision audit: every v2 journal entry cites the playbook rule that authorized it; entry without a rule ⇒ the run should have paused — test with a rule-free playbook (must pause on iteration 1)
- [ ] 6.4 Rerun the §5 self-test unattended: mis-configured reference → PASS with zero pings once the needed rules exist

## 7. Portfolio + invention protocol (D8, D9)

- [ ] 7.1 Playbook portfolio section: sequential-only, board as ranking, insight-carryover rule (read prior journals before scoping the next candidate), env/map removables *report* (sizes, never delete)
- [ ] 7.2 Invention protocol: research session against board failure evidence → `/loc-new` scaffold → Anton glances at README before any grind
- [ ] 7.3 DEPLOY/Stage-2 stop wired into both skill + workflow (present evidence, wait for Anton)

## 8. Docs, memory, daily note

- [ ] 8.1 Update memory: refresh stale `architecture-locbench-harness` leaf (still describes the old recorder/bag design) + new leaf for the locdev flow (capture-not-encode, graduated autonomy, this change)
- [ ] 8.2 Daily note block (draft → approve → append)

## 9. Carried from locbench §11 (harness archived 2026-07-14; §11.1 was already absorbed as this change's playbook)

- [x] 9.1 `logic/locbench/AGENTS.md` — architecture (3 processes, W1–W5 wires), CLI surface (episodes/run/score/board/env), invariants (raw scoring/no alignment, all-episodes-pass, sim-time timeouts); one-line `CLAUDE.md` shim (locbench §11.2)
- [x] 9.2 Architecture-guard extension in `tests/oli/reason/test_architecture.py`: `logic/oli/` never imports `logic.locbench` (the oracle drives the brain, not the reverse). Joins the existing realizations-import + service-purity guards (locbench §11.3)
- [x] 9.3 Memory: `architecture-locbench-harness` refreshed to as-built (episode-sets pivot, ship+archive, env contract); `locbench-run-env-contract` + `architecture-locdev-flow` leaves cover the rest (merges 8.1) (locbench §11.4)
