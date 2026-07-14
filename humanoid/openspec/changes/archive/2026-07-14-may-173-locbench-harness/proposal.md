# Proposal — locbench: the localization development & evaluation loop

## Why

MAY-173's endgame is a real localizer behind the brain's localization seam (`logic/oli/reason/localization/`) — today the seam runs on Isaac ground truth. The research (`docs/research/localization.md`) fixed the candidate landscape (RTAB-Map first) and *measured* the accuracy budget nav actually needs (E1/E2: constant bias < 0.1 m is binding). The module refactor (`may-173-reason-module-separation`, merged) froze the contract candidates must fill: `LocalizationModule.start/step/stop` over `LocalizationIn → LocalizationOut`, map-frame output.

What's missing is the **engine around that contract** — and it has two lives, in order:

1. **Development enforcer.** No SLAM adapter works on the first try. The loop is the red/green harness an (autonomous) agent iterates against: run episodes → read `report.json` → fix the adapter/env → rerun. "It works" stops being a judgment call and becomes a computable verdict — the same move TDD makes for code.
2. **Evaluation engine.** Once candidates *work*, the identical loop becomes the selector: same frozen episode set, same gates, scoreboard row per candidate → pick the winner for phase 2 deployment.

Live-loop by design: the candidate runs **in the loop** during real sim drives — no recording infrastructure now, and Stage 2 (robot drives *on* the candidate's poses — which IS the demo condition) is one seam-swap away.

## What Changes

- **NEW — `logic/locbench/` package.** The loop engine, sibling of `oli/` — a tool, not part of the robot. Statistics/gates/episode logic brain-pure (numpy + stdlib); the runner orchestrates a live Isaac session.
- **NEW — frozen episode sets.** Spawn/goal pairs seeded-randomly sampled from the baked occupancy free space (min separation, min route length), rendered on the map PNG for Anton's approval, frozen to `episodes/<scene>.json`. An evaluation = 10 episodes (flag): spawn robot at S, Nav drives to G on ground truth, episode ends on GT arrival (0.3 m) or timeout (90 s, marked).
- **NEW — candidate hosting: IN the brain, shadow mode (Stage 1, passive).** A candidate = `reason/localization/realizations/<name>/` — the `LocalizationModule` adapter + env recipe (`environment.yml`/`build.sh`) + post-build lockfile. The brain gains a **localization host** (frame-channel client → `LocalizationIn` → `module.step` on a side thread → latest pose out on telemetry): localization is brain logic and runs inside the brain process (Anton, 2026-07-13 — "no internal logic outside the brain"). Stage 1 = `--shadow <name>`: Nav drives on GT, the candidate is measured only, **warm start** per episode (`Setup.initial_pose` = spawn pose). For bench runs the whole brain boots inside the disposable `bench-<name>` conda env (brain + candidate deps; no Docker; removal = `conda env remove`). What the bench tests IS the deployment configuration — no transfer gap.
- **NEW — brain service seam (goal in / telemetry out).** The brain runs isolated (`brain_main`, own process); a small socket seam (`logic/oli/service/`) lets any client send `GoalCoordinate`s and read pose/path/status/est-pose/obs telemetry. The evaluator (locbench's pure-client process) scripts episodes through it; dev_app later migrates onto the same seam to become pure visuals (follow-up change).
- **NEW — scorer + two-tier gates.** Per tick: raw map-frame compare vs GT (**no anchor/alignment — constant bias stays visible**), position [m] + yaw [deg] error, coverage, rate. Per episode: mean/median/p95/max. Verdict: **every episode must pass**. Tiers: **PASS** = measured budget (mean < 0.10 m, max < 0.15 m, yaw < 10°, coverage ≥ 95%); **DEPLOY** = margin tier (numbers set in design). Output: `report.json` + GT-vs-est overlay on the occupancy PNG (committed); raw pose pairs gitignored.
- **NEW — reference candidate.** GT passthrough with injectable bias/noise/dropout, run through the full hosting path. Harness acceptance: passes clean, fails at 0.2 m bias, fails at low coverage — the loop proves itself before any SLAM exists.
- **NEW — Stage 2, sequenced: the flag flip.** After Stage 1 runs green: `--localizer <name>` — Nav consumes the in-brain module's pose (GT stays evaluator-side as judge); success = GT-verified arrival. Unlocked per candidate only on a Stage-1 PASS. Stage 2 green = the phase-2 demo condition demonstrated.
- **NEW — CLI, the agent surface.** `p -m humanoid.logic.locbench {episodes, run, score, board, env}` — idempotent, file logs, exit codes, machine-readable reports.
- **NOT in scope:** real SLAM adapters (that is what the loop is *for*, immediately after), bag recording/offline replay (deferred; the runner can grow `--record`), real-robot episodes.

## Capabilities

### New Capabilities

- `localization-bench`: the localization development & evaluation loop. Defines frozen episode sets, live in-the-loop candidate hosting behind the merged `LocalizationModule` contract, raw two-tier gated scoring, the self-validating reference candidate, Stage-2 closed-loop verification, and the CLI an autonomous agent iterates against. Observable behavior: any `realizations/<name>/` satisfying the contract gets a reproducible pass/fail verdict from the same episode set — first to force it into working shape, then to rank it.

### Modified Capabilities

- None. Additive: existing channels/contracts consumed read-only; `reason/localization/` gains only the `realizations/` convention (adapters are never imported by the brain env — architecture guard extended accordingly).

## Impact

- **New code**: `logic/locbench/`; `reason/localization/realizations/reference/`; tests under `tests/locbench/` (pure parts `brain`-marked).
- **Existing code**: no behavior changes in `logic/oli/` or `logic/simulation/`; Stage 2 adds one additive `Localizer` realization behind the existing seam.
- **Data**: `episodes/<scene>.json` + reports/plots committed; pose-pair CSVs gitignored.
- **Envs**: loop engine in `brain`; episodes need a live `isaac` World (launcher run); `bench-<name>` envs created/removed by `locbench env` — never touching `brain|isaac|limx|hum`.
- **Linear**: [MAY-173](https://linear.app/may33/issue/MAY-173) — this change builds the engine; adapters (RTAB-Map first) and the live seam swap are the follow-ups it enables.
