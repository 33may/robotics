# locbench/ — the localization evaluation oracle

Gives any `reason/localization/realizations/<name>/` candidate a **reproducible two-tier
pass/fail verdict** from frozen episodes. This is the measuring stick the AI dev-loop
(`may-173-locdev-flow`) optimises against — candidates change, the oracle does not. Design +
rationale: the archived change `openspec/changes/archive/2026-07-14-may-173-locbench-harness/`
(D1–D14). The dev-loop that *uses* this: `reason/localization/realizations/AGENTS.md`.

## Three processes, five wires

A bench run is the normal Oli stack (booted by `logic/oli/launcher.py` as ONE subprocess) plus
a third pure-client process that scores it:

| proc | role | talks |
|---|---|---|
| **P1 World** (Isaac, `bench-<cand>`-agnostic) | applies GlideCmd, reports obs + camera frames + GT pose | W1, W2, W3 |
| **P2 Brain** (`--service --shadow <cand>`, in `bench-<cand>`) | Nav + in-brain localization **host** running the candidate on a side thread | W1, W2, W4, W5 |
| **P3 Evaluator** (`runner.py`, plain client) | drives episodes, logs (est, GT) pairs, scores | W4, W5, W3 |

- **W1** obs / GlideCmd — World ⇄ Brain (the normal control loop).
- **W2** camera frames — World → Brain (feed the localization host).
- **W3** ground truth — the evaluator's truth source. Stage-1 shadow: GT is derived from
  telemetry (`_gt_from_telemetry`) and **republished** to a bench-only socket
  (`Setup.calibration["gt_feed_socket"]`) that the reference candidate replays; real candidates
  ignore it.
- **W4** goal-in — evaluator → Brain (`service/goal_channel.py`, 33-byte struct, latest-wins).
- **W5** telemetry-out — Brain → evaluator (`service/telemetry.py`, JSON datagram: pose, path,
  goal, est `LocalizationOut`, loop rate, loc state/error).

The evaluator injects ALL its I/O (send_goal / start-stop / gt_latest / telemetry) — it never
imports a socket type, which is why `evaluator.py` unit-tests against fakes.

## CLI surface (`python -m humanoid.logic.locbench …`)

| cmd | does |
|---|---|
| `episodes <scene> [--seed N] [--freeze]` | sample → render the approval PNG → (`--freeze`) write `episodes/<scene>.json` |
| `env {create,remove} <cand> [--force]` | build/destroy the disposable `bench-<cand>` conda env + freeze `lock.yml` |
| `run <cand> [--scene] [--smoke N] [--episodes N] [--timeout S] [--shadow-config F]` | boot the stack in `bench-<cand>`, evaluate, write a run dir; exit code = verdict |
| `score <run-dir>` | recompute stats/plots/report offline from the stored pairs (same numbers) |
| `board` | MD scoreboard — latest report per candidate |

## Module map

| file | holds |
|---|---|
| `__main__.py` | CLI dispatch + `SCENES` registry (map_dir, scene_usd, camera cadence, zones) |
| `episodes.py` | seeded, constraint-checked episode sampling; versioned freeze/load |
| `render.py` | episode-set render on the baked map (routes via the **deployment** `build_planner`) |
| `envs.py` | `bench-<name>` conda env create/remove + `lock.yml` (conda behind an injected runner) |
| `runner.py` | `run_bench` (spawns the launcher), `write_run_artifacts`, `score_run_dir`, `board` |
| `evaluator.py` | `Evaluator` — episode loop (transit → warm-start → scored leg → teardown), pure client |
| `pairs.py` | (est, GT) nearest-stamp association, warm-up exclusion, LOST handling |
| `stats.py` | per-episode pos/yaw error, coverage, rates — **raw map frame** |
| `verdict.py` | two-tier PASS/DEPLOY, weakest-episode roll-up |
| `report.py` / `plots.py` | `report.json` (+ provenance) / overlay · timeline · run-sheet · CDF |

## Invariants — the oracle's integrity (dev-loop agents: NEVER weaken these)

- **Raw map-frame scoring, NO alignment / refit.** A constant bias must stay visible — it is the
  measured nav-killing failure mode. Never anchor est to GT before scoring.
- **Every episode must pass; run tier = the weakest episode.** No averaging — one lost aisle is a
  failed demo. Timeout/crashed episodes fail both tiers.
- **Sim-time timeouts.** Episode budgets run on GT **sim** stamps (sim is ~0.03–0.3× real under
  cameras); wall-clock is only a frozen-sim backstop.
- **Frozen episode sets are versioned + committed.** Never resample per candidate — a candidate
  competes on the same course as every other. Editing gates or episodes to make a candidate pass
  is tampering, not tuning.
- **One env per run, hard-required.** `run` boots the brain in `bench-<cand>`; missing env →
  exit 3, no fallback (so a run is reconstructable from its `lock.yml`).
- **Dependency direction is one-way.** locbench imports the brain to drive it; the brain
  **never** imports locbench (guarded in `tests/oli/reason/test_architecture.py`). locbench is
  *not* `brain`-marked — matplotlib / subprocess / conda tooling live here and must stay OUT of
  every brain/robot boot.

`runs/` policy: commit `report.json` + plots; gitignore `pairs.csv`.
