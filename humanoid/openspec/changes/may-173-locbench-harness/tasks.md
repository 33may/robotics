## 1. OpenSpec scaffolding

- [x] 1.1 Author `proposal.md` / `design.md` / `tasks.md` / `specs/localization-bench/spec.md`
- [ ] 1.2 Structure check against existing changes (no `openspec` CLI on this machine — manual review; Anton approves the proposal)
- [ ] 1.3 Work continues on branch `33may/nav-localization-research` (natural MAY-173 continuation; Anton-confirmed)

## 2. Package scaffold + shared types (D3, D4)

- [ ] 2.1 `logic/locbench/` package skeleton + `__main__.py` CLI dispatcher (`record|export|run|score|board|env`); `tests/locbench/` with `brain` marker on pure modules
- [ ] 2.2 TDD: bag model — `bag.py` (read/write `gt.jsonl`, `intrinsics.json`, `meta.json`; rgb/depth png IO with uint16-mm depth, 0=invalid round-trip)
- [ ] 2.3 TDD: trajectory model — `trajectory.py` (TUM trajectory file read/write: `stamp tx ty tz qx qy qz qw`; SE(3)→SE(2) projection)

## 3. Scorer (D6, D7, D8) — pure, TDD-first

- [ ] 3.1 TDD: stamp association (nearest-neighbor on shared clock, Δt ≤ 100 ms; unmatched → no-fix)
- [ ] 3.2 TDD: constant camera→base transform from `camera_mounts.py` home pose (D8); applied before anchoring
- [ ] 3.3 TDD: one-time SE(2) anchor fit on the first ~2 s of matched pairs (D6); verify a constant-bias trajectory is NOT absorbed (bias survives the anchor when injected after t=2 s; whole-run bias pins the anchor — test both)
- [ ] 3.4 TDD: per-stamp position/yaw errors → mean/max/p95 + fix coverage (2 s warmup exclusion)
- [ ] 3.5 TDD: gate evaluation (mean<0.10 m, max<0.15 m, yaw<10°, coverage≥95%) → `report.json` with numbers, verdicts, provenance (bag id, candidate, anchor, versions)
- [ ] 3.6 Overlay plot: GT vs est on the baked occupancy PNG (`assets/envs/warehouse_nvidia/nav_maps/v1/`); show inline + cite path

## 4. Reference candidate + harness self-validation (D4, D9)

- [ ] 4.1 `candidates/reference/`: `run.py` implementing `map`/`localize` by replaying `gt.jsonl` → TUM trajectory (runs in plain `brain` env — proves the contract needs no special env)
- [ ] 4.2 Injectable corruption via candidate config: constant bias, Gaussian noise, dropout, stamp delay
- [ ] 4.3 TDD acceptance triplet on a synthetic bag: clean reference **passes**; 0.2 m bias **fails** (max-pos gate); 20% dropout **fails** (coverage gate)
- [ ] 4.4 `locbench run reference --bag <synthetic>` end-to-end: subprocess, log file, exit code, `report.json` — the full loop with zero SLAM code

## 5. Recorder (D1, D2, D3)

- [ ] 5.1 TDD against fakes: frame-channel + debug-pose clients → bag writer (dedup by stamp, per-stream dirs, achieved-fps/drop stats into `meta.json`)
- [ ] 5.2 `locbench record --bag <dir> --fps N --duration S` CLI; graceful stop; both cameras recorded when present
- [ ] 5.3 Integration smoke vs a live glide session (`launcher --sim isaac --mode glide` + `--cameras --debug-pose`): frames + GT land, stamps interleave on one clock

## 6. TUM RGB-D exporter (D3)

- [ ] 6.1 TDD: bag → TUM layout (`rgb/`, `depth/`, `rgb.txt`, `depth.txt`, `groundtruth.txt`; mm → factor-5000 depth; GT SE(2) → SE(3) with recorded constants)
- [ ] 6.2 Round-trip check: export a synthetic bag, re-read with `trajectory.py`, scores clean via the reference candidate

## 7. Scripted-goal driver (D2) — ⚠ blocked on the Nav execution gate

- [ ] 7.1 Goal-list script format (`drives/<scene>_<pass>.json`: ordered GoalCoordinates + dwell) + driver that feeds `Nav.set_goal` and advances on arrival
- [ ] 7.2 Map-pass sweep + eval-pass drive lists for the warehouse (aisle coverage / representative route) — Anton reviews the routes
- [ ] 7.3 Dry-run in sim: both drives complete without manual input

## 8. Candidate envs + runner (D4, D5, D10)

- [ ] 8.1 `locbench env create|remove <candidate>` → `bench-<name>` from the candidate's env spec; `remove` verified to leave no trace; guard refuses to touch `brain|isaac|limx|hum`
- [ ] 8.2 TDD: runner — `locbench run <candidate> --bag <scene>`: map pass → localize pass → score; subprocess via `conda run`, log capture, timeout, nonzero-exit surfacing
- [ ] 8.3 `locbench board`: aggregate `report.json`s → MD pipe-table scoreboard (per candidate: gates, mean/max/p95, coverage, runtime)

## 9. Freeze warehouse bags v1 (needs §5 + §7)

- [ ] 9.1 Record `map` + `eval` bags in the warehouse scene (Anton launches; recorder attached); verify meta (fps, drops) sane
- [ ] 9.2 Score the clean reference against the frozen eval bag — the standing harness baseline
- [ ] 9.3 Freeze: bags read-only under `assets/bags/warehouse/`, gitignore entry, provenance in `meta.json`

## 10. Docs, memory, daily note

- [ ] 10.1 `logic/locbench/AGENTS.md` (+ one-line `CLAUDE.md` shim): contract, CLI surface, invariants, how the agent loop drives it
- [ ] 10.2 Agent memory: update `architecture-locbench-harness` with build outcome + any gotchas
- [ ] 10.3 Daily note block (draft → approve → append)
