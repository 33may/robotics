# Tasks — locbench (build order; TDD throughout, `brain`-marked where pure)

## 1. OpenSpec + branch

- [x] 1.1 `proposal.md` / `design.md` rewritten on the settled architecture (in-brain hosting, 3 processes, shadow mode) — Anton-approved 2026-07-13
- [ ] 1.2 `specs/localization-bench/spec.md` delta (SHALL requirements + scenarios, may-149 format)
- [x] 1.3 Branch `33may/may-173-locbench` off main; merge to main at working states (Anton's standing rule)

## 2. Brain service seam — W4 goals in / W5 telemetry out (D5)

- [x] 2.1 TDD: `logic/oli/service/` protocol — encode/decode goal set/clear (`GoalCoordinate`), telemetry record (stamp, pose, path, goal status, est `LocalizationOut`, loop-rate, `(Observation, Intent)`); socket patterns per `comm/debug_pose.py`
- [x] 2.2 TDD: `GoalChannelServer` → `Nav.set_goal/clear_goal` against a fake Nav; latest-wins, malformed-message tolerance
- [x] 2.3 TDD: `TelemetryServer` fed from the Orchestrator recorder hook + Nav path/status + localization host output; client sees a coherent latest snapshot
- [x] 2.4 `brain_main --service` flag boots both; architecture guard: `service/` imports stay brain-pure (no isaacsim/limxsdk, no devapp)
- [ ] 2.5 Integration smoke vs live glide session: send goal over the wire → robot drives; telemetry streams (Anton at the controls)

## 3. Episode sets (D2, D3)

- [ ] 3.1 TDD: `episodes.py` — seeded sampling from occupancy free space (min pair separation, min route length, `plan_path` reachability incl. transit from origin), freeze/load `episodes/<scene>.json` (versioned)
- [ ] 3.2 TDD: eval episode list + mapping-pass coverage goal list in one file per scene
- [ ] 3.3 `locbench episodes warehouse` → render spawns/goals/routes on the baked map PNG — show inline + Anton approves → freeze v1 (committed)

## 4. Scoring core — pure, sim-free (D10, D11)

- [ ] 4.1 TDD: `pairs.py` — (est `LocalizationOut`, GT) pair log write/read; nearest-stamp association (Δt ≤ 100 ms), 2 s warmup exclusion
- [ ] 4.2 TDD: `stats.py` — per-tick pos/yaw error (raw map-frame, NO alignment), coverage, pose rate, brain-loop rate → per-episode mean/median/p95/max
- [ ] 4.3 TDD: two-tier verdict — PASS/DEPLOY thresholds, all-episodes-pass, timeout/crashed episodes fail both; verify a constant 0.2 m bias fails max-pos (the E1 failure mode stays visible)
- [ ] 4.4 TDD: `report.py` — `report.json` with numbers, per-episode + run verdicts, provenance (adapter git hash, `lock.yml` hash, map_dir hash, episode-set version, seed, timings)
- [ ] 4.5 `plots.py`: episode overlay (GT vs est on map PNG, LOST stretches marked), error timeline w/ gate lines, run contact sheet (ultrawide grid), distribution vs tiers — show inline + cite paths

## 5. Localization host in the brain (D6, D7)

- [ ] 5.1 TDD: realization registry in `reason/localization/` — resolve `--shadow`/`--localizer <name>` → lazy-import `realizations/<name>/` (never imported unless selected); architecture guard: nothing brain-marked imports `realizations`
- [ ] 5.2 TDD: host loop against a fake module — frame client → `LocalizationIn` assembly (nearest obs, latest intent, frame-paced) → `step()` on side thread → latest `LocalizationOut` readable; slow module ⇒ dropped frames counted (rate/coverage), never a blocked control loop (fake with GIL-friendly sleep)
- [ ] 5.3 TDD: per-episode lifecycle over the wire — evaluator can command `start(Setup)/stop()` between episodes (warm start = spawn pose); `verify_module_contract` semantics on violations → episode marked `crashed`
- [ ] 5.4 Thin in-process `Localizer` adapter over the host's latest pose (Stage 2 seam, dormant in Stage 1); brain-loop rate/jitter metric exported on telemetry
- [ ] 5.5 `brain_main --localizer gt --shadow <name>` wiring; brain boots headless in an arbitrary conda env (no torch needed in glide — verify import surface)

## 6. Evaluator (D1, D14)

- [ ] 6.1 TDD against fakes: episode loop — transit goal → arrival watch (GT ≤ 0.3 m) → `start` → goal → pair logging → arrival/timeout (90 s, `--timeout`) → `stop` → next; crash detection (brain death ⇒ episode `crashed`, reboot, continue)
- [ ] 6.2 `locbench run <name> --scene warehouse [--episodes N] [--smoke=3]` — Supervisor boots World + Brain(`--service --shadow <name>`) in `bench-<name>`, attaches evaluator (single-entrypoint rule); per-run dir + file log + exit code = verdict
- [ ] 6.3 `locbench score <run-dir>` — recompute stats/plots offline from stored pairs; `locbench board` — MD scoreboard from committed reports (+ per-candidate report history)
- [ ] 6.4 `runs/` gitignore: commit `report.json` + plots, ignore `pairs.csv`

## 7. Env tooling (D8)

- [ ] 7.1 TDD: `envs.py` — `locbench env create|remove <name>` from the realization recipe; post-build `conda env export` → `lock.yml` (committed); `remove` leaves no trace; hard guard refuses `brain|isaac|limx|hum`
- [ ] 7.2 `bench-reference` path: reference candidate must run in a plain brain-compatible env (proves the contract needs nothing special)

## 8. Reference candidate + harness self-validation (D13) — the gate for everything above

- [ ] 8.1 `realizations/reference/`: `LocalizationModule` replaying GT via `Setup.calibration["debug_pose_socket"]`; injectable constant bias / Gaussian noise / dropout / delay via config; `README.md` + `JOURNAL.md` seeded (the exemplar realization)
- [ ] 8.2 Acceptance triplet vs live sim: clean → **PASS**; 0.2 m bias → **fails max-pos**; 20% dropout → **fails coverage** — full path (in-brain shadow host → telemetry → evaluator → report)
- [ ] 8.3 Freeze the clean-reference run as the standing baseline row on the board

## 9. Mapping pass (D9)

- [ ] 9.1 TDD: `locbench map <name> <scene>` — coverage drive from the scene's mapping goal list, module in mapping mode (`build_map`) in-brain; `map_dir` cached under the realization (gitignored, content-hash into provenance); refuse `run` without a `map_dir` when the realization declares one
- [ ] 9.2 Reference candidate declares no map (GT replay) — verify the no-map path

## 10. Stage 2 — closed loop (D6), gated on Stage-1 PASS

- [ ] 10.1 TDD: `--closed-loop` — brain boots `--localizer <name>` (Nav on the module's pose via the 5.4 adapter); evaluator refuses unless the candidate's latest full run is PASS
- [ ] 10.2 Success gate: GT-verified arrival per episode, no timeout; report gains `closed_loop` verdict block
- [ ] 10.3 Reference candidate closed-loop: clean passes; 0.2 m bias visibly degrades/fails — empirical check that Stage-1 gates predict closed-loop success (the E1/E2 claim)

## 11. Docs, guard, memory, daily note

- [ ] 11.1 `realizations/AGENTS.md` — the loop protocol for implementing agents: candidate checklist, iterate rules (smoke → full → closed-loop), JOURNAL/README/memory split, budget caps; one-line `CLAUDE.md` shim
- [ ] 11.2 `logic/locbench/AGENTS.md` — architecture (3 processes, wires), CLI surface, invariants (raw scoring, no alignment; all-episodes-pass); shim
- [ ] 11.3 Architecture guard extensions in `tests/oli/reason/test_architecture.py`: no brain-marked import of `realizations/`; `service/` purity; locbench never imported by `logic/oli/`
- [ ] 11.4 Memory: update `architecture-locbench-harness` (build outcome + gotchas); daily note block (draft → approve → append)
