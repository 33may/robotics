# Design — locbench, the localization development & evaluation loop (localization-bench)

## Context

The reason refactor (merged, `may-173-reason-module-separation`) froze the seam: `LocalizationModule.start(Setup)/step(LocalizationIn)→LocalizationOut/stop()`, map-frame output, pose⇔not-LOST. The research fixed candidates (RTAB-Map first) and the measured budget (E1/E2: constant bias <0.1 m binding). Design re-locked with Anton 2026-07-13 (ground-up Q&A, this session): **live-loop first**, **localization runs in-brain**, the loop's first life is a *development enforcer* (agent iterates adapters red→green against it), its second life the *evaluation engine*.

Ground rules inherited: invariance boundary, single entrypoint, no Docker (disposable conda envs), TDD in-repo, MD tables.

## The system (3 processes)

```
P1 ISAAC WORLD ◀──W1 obs/cmds──▶ P2 BRAIN ◀──W4 goals / W5 telemetry──▶ P3 EVALUATOR
      │                          ┌─────────────────────────┐            (locbench, pure client)
      │ W2 frames                │ Orchestrator: Nav→glide  │            sends episode goals
      └─────────────────────────▶│ + LOCALIZATION HOST:     │            reads GT (W3) + est (W5)
        (brain = frame consumer) │   frames→LocalizationIn→ │            logs pairs → scores
      W3 GT ───────────────────▶ │   module.step (thread) → │            never touches frames
        (to evaluator only)      │   latest pose→telemetry  │
                                 └─────────────────────────┘
```

| wire | carries | status |
|---|---|---|
| W1 | `Observation` →, `GlideCmd` ← (`/tmp/oli-world.sock`) | exists |
| W2 | `CameraFrame` → brain (localization host) | exists (`comm/camera_stream.py`) — consumer moves |
| W3 | GT pose → evaluator | exists (`comm/debug_pose.py`) |
| W4 | goal set/clear (`GoalCoordinate`) → `Nav.set_goal` | **NEW** (`oli/service/`) |
| W5 | pose, path, goal status, est `LocalizationOut`, loop-rate, `(Observation, Intent)` | **NEW** (`oli/service/`) |

## Decisions

### D1 — Live in-the-loop evaluation, candidate passive in Stage 1

Every evaluation is a real sim session: robot drives spawn→goal steered by **ground truth**; the candidate consumes the live sensor stream and its poses are only logged. No recording format, real-time throughput is tested for free, Stage 2 is a flag flip. Cost accepted: evals need Isaac up; runs are statistically (not bitwise) comparable — the frozen episode set + all-episodes-pass gating absorbs that.

### D2 — Frozen seeded episode sets

`locbench episodes <scene>` samples spawn/goal pairs from the baked occupancy free space (seeded; min pair separation, min route length, reachability-checked via `plan_path`), renders them on the map PNG for Anton's approval, freezes to `episodes/<scene>.json` (versioned, committed). An eval = 10 episodes (`--episodes` flag). Episode ends on GT arrival (≤0.3 m) or timeout (90 s → marked `timeout`, pairs still scored, episode fails the verdict).

### D3 — Episode transit instead of teleport (no World changes)

The robot reaches each episode's spawn S by **gliding there on GT (unscored transit leg)**, then the scored S→G episode starts. Keeps the World untouched; costs sim minutes per eval. Alternative (World-side teleport) rejected for v1.

### D4 — Warm start

`LocalizationSetup.initial_pose` = the episode's spawn pose (GT at episode start); fresh `start(...)/stop()` lifecycle **per episode**. Matches deployment (boot at a known dock). Cold-start/relocalization = later episode variant, additive.

### D5 — Brain isolation + the goal/telemetry wire (Anton, 2026-07-13)

The brain runs as its **own process** (`brain_main`) — never inside a client. New brain-side seam, `logic/oli/service/` (flag-gated in `brain_main`, socket patterns like `comm/debug_pose.py`): **goal channel in** (`GoalCoordinate` set/clear → `Nav.set_goal/clear_goal` — the seam the refactor built gains a wire; a goal *source* is anyone who can send: the evaluator scripts it, dev_app converts a map click and sends) and **telemetry channel out** (pose, path, goal status, est pose, brain-loop rate, `(Observation, Intent)` from the recorder hook). dev_app today hosts the brain in-process (`brain_link.py`); it migrates onto this seam to become pure visuals — own follow-up change; its embedded mode coexists meanwhile and locbench never uses it.

### D6 — Localization runs IN the brain; shadow mode (Anton, 2026-07-13)

"No internal logic outside the brain." The brain grows a **localization host**: a frame-channel client (the brain becomes the frame consumer — it owns its senses), `LocalizationIn` assembly (frames + in-process obs/intent), `module.step()` on a side thread, latest `LocalizationOut` exported on telemetry and readable by Nav through a thin in-process `Localizer` adapter.

- **Stage 1 = shadow mode**: `brain_main --localizer gt --shadow <name>` — Nav drives on GT, the candidate runs in shadow, measured only.
- **Stage 2 = the flag flip**: `--localizer <name>` — Nav consumes the module's pose; GT stays evaluator-side as judge. No new machinery between stages; what the bench tests IS the deployment configuration.
- **Accepted risks, eyes open** (Anton): a GIL-holding binding stalls the control loop — *measured*, brain-loop rate/jitter is a report metric; a segfaulting candidate kills the brain — run marked `crashed`, rebooted. Fine for sim/eval now; re-examined for the real robot.
- **Out-of-process module host = fallback only**, built iff a candidate's deps cannot coexist even in the merged env. A later major refactor (every component its own process, continuous latest-signal reading) may revisit hosting globally — not before.

### D7 — Candidate = a `realizations/<name>/` folder

`logic/oli/reason/localization/realizations/<name>/` holds: the `LocalizationModule` adapter (+ optional `build_map`), `environment.yml`/`build.sh`, post-build `lock.yml`, `README.md`, `JOURNAL.md`. Selected by name via `--shadow`/`--localizer` (registry in the localization package, lazy imports — realizations are never imported unless selected; architecture guard: nothing brain-marked imports `realizations`).

### D8 — Envs: the brain boots inside `bench-<name>`

One process ⇒ one env: for a bench run the whole brain runs in the candidate's disposable env (`conda run -n bench-<name> brain_main …`) — brain deps (numpy/stdlib in glide) + candidate stack together. `locbench env create|remove <name>` builds it from the realization's recipe (+ exports `lock.yml`, committed — "what exactly was built" stays answerable); `remove` leaves no trace; hard guard refuses `brain|isaac|limx|hum`. No Docker.

### D9 — Mapping pass: live, per candidate, cached

SLAM candidates need their own map representation (RTAB-Map = feature DB; the occupancy grid can't feed it). `locbench map <candidate> <scene>`: one scripted **coverage drive** (goal list in `episodes/<scene>.json`) with the candidate's mapping mode running in-brain; output `map_dir` cached under the realization (gitignored, content-hash recorded). Eval runs require an existing `map_dir` and pin its hash into provenance. Re-map only when scene or mapping logic changes.

### D10 — Scoring: raw map-frame compare, no alignment ever

Module output is map-frame by contract → the evaluator compares raw per tick (nearest GT by stamp, Δt ≤ 100 ms): position error [m], yaw error [deg], coverage (% ticks answered, 2 s warmup excluded), pose rate [Hz], brain-loop rate. No Umeyama/anchor fitting anywhere — constant bias must stay visible (E1's nav-killing failure mode). Per episode: mean/median/p95/max.

### D11 — Verdict: every episode passes; two tiers

| tier | mean pos | max pos | yaw | coverage | meaning |
|---|---|---|---|---|---|
| **PASS** | <0.10 m | <0.15 m | <10° | ≥95% | the measured E1/E2 nav budget — "works in sim" |
| **DEPLOY** | <0.07 m | <0.12 m | <7° | ≥98% | ~30% sim→real margin — "take it to phase 2" *(provisional — locked after the first real candidate experiment)* |

A candidate's tier = the highest tier **all** episodes clear (timeout/crashed episodes fail both). One lost aisle = failed demo, so no averaging across episodes.

### D12 — Artifacts & traceability (locked with Anton)

Per run dir `runs/<candidate>/<run-id>/`: `report.json` (stats, verdicts, provenance: adapter git hash, `lock.yml` hash, map_dir hash, episode-set version, seed) + plots — per-episode overlay (GT vs est on the map, LOST stretches marked), error timeline, per-run wide contact sheet (ultrawide grid) + distribution vs gate lines. Committed: reports + plots; gitignored: `pairs.csv`. Traceability stack: `report.json` (machine) → `JOURNAL.md` (append-only per iteration: change → numbers → next hypothesis) → `README.md` (current truth) → `realizations/AGENTS.md` (the loop protocol for implementing agents) → memory tree `localization-slam` node (cross-session lessons via /reflect).

### D13 — Reference candidate proves the loop

`realizations/reference/` — a `LocalizationModule` that replays GT (debug-pose socket path passed via `Setup.calibration["debug_pose_socket"]`, a declared test-only key) with injectable constant bias / Gaussian noise / dropout / delay via its config. Runs the FULL path: in-brain shadow host, telemetry out, evaluator scoring — in a plain brain-compatible env. Acceptance triplet: clean → PASS; 0.2 m bias → fails max-pos; 20% dropout → fails coverage. Validates host + wire + scorer with zero SLAM dependencies.

### D14 — CLI is the agent surface

`p -m humanoid.logic.locbench {episodes, env, map, run, score, board}` — every command idempotent, per-run file logs, meaningful exit codes, machine-readable outputs. `run` boots World + Brain via the Supervisor (single-entrypoint rule) and attaches the evaluator. The agent loop is strictly: `run → read report.json/log → edit realization → rerun`, budget-capped. `score` recomputes stats/plots offline from stored pairs.

## Risks / Trade-offs

- **Isaac non-determinism** → runs statistically comparable only; frozen episodes + all-pass gating absorb it; recording (`--record`) stays the escape hatch if it bites.
- **Sim-in-the-loop iteration speed** (~10–15 min/eval) → `--episodes 3` smoke mode for grinding; full 10 for verdicts.
- **GIL stall / segfault in-brain** → accepted eyes-open (D6); measured, visible, rebooted. Real-robot hosting re-examined later.
- **Merged env fails for some candidate** → the out-of-process host fallback, built then, module code unchanged (host-agnostic contract).
- **Committed plots accumulate** over hundreds of iterations → see Open Questions.

## Migration Plan

Additive. Follow-ups this change enables: (1) agent-driven adapters (RTAB-Map first) inside the loop; (2) dev_app migration onto the goal/telemetry seam (pure visuals); (3) winner's live wiring = the same `--localizer <name>` flag on the real robot stack (MAY-173 endgame); (4) optional `--record` → frozen bags if iteration speed demands.

## Open Questions

- Plot-commit policy under heavy iteration: commit all (as decided) vs reports always + plots only on `--keep` runs. Default: commit all; revisit at first repo-bloat sign.
- DEPLOY tier numbers (D11) — **deferred (Anton, 2026-07-13)**: defined together after the first real candidate experiment ("now I have no idea" — same capture-not-encode stance as locdev-flow). Until then the DEPLOY row is provisional and verdicts gate on PASS only.
- Transit legs (D3): if per-eval sim time hurts, revisit World-side teleport as a debug channel.
