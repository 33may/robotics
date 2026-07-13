## Why

MAY-173's endgame is a real localizer behind the brain's `Localizer` seam (`logic/oli/reason/nav/localizer.py`) — today the seam runs on Isaac ground truth. The localization research (`docs/research/localization.md`) surveyed the candidate landscape and measured the accuracy budget the nav stack actually needs (E1/E2 experiments: constant bias < 0.1 m is the binding constraint; noise and rate are cheap). What is missing is the machinery that makes "this candidate works" a **computable pass/fail** instead of a judgment call.

Isaac already gives us ground truth, and the nav stack drives the robot. **locbench** closes the loop: record what a localizer would see at runtime, replay it to any candidate implementation behind a fixed contract, and score the estimate against ground truth with gates derived from the measured budget. Once this harness exists, implementing and iterating the actual SLAM candidates (RTAB-Map first, per the research) becomes an autonomous agent loop — run → read report → fix → rerun — with no human in the scoring path.

## What Changes

- **NEW — `logic/locbench/` package.** A benchmarking tool, sibling of `oli/` — not part of the robot. Core (scoring, contract, formats) is brain-pure (numpy + stdlib + PIL/yaml lazy, like `occupancy_io.py`).
- **NEW — bag recorder.** A **socket client** of the two existing World channels — the camera frame channel (`CameraFrame`: stamp_ns, RGB, depth[m], intrinsics) and the debug-pose channel (GT stamp, x, y, yaw; same sim clock). Writes a frozen on-disk bag (RGB png + uint16-mm depth png + `gt.jsonl` + `intrinsics.json` + `meta.json`). Zero World-side changes.
- **NEW — scripted-goal driver.** Feeds `Nav` a fixed `GoalCoordinate` list so the two bag passes per scene (`map` coverage sweep + `eval` drive) are reproducible and re-recordable. Depends on the in-flight Nav execution gate.
- **NEW — TUM RGB-D exporter.** Bag → the standard TUM layout (`rgb/`, `depth/`, `rgb.txt`, `depth.txt`, `groundtruth.txt`) so candidate tools ingest with near-zero glue.
- **NEW — candidate adapter contract.** A candidate is a directory (`logic/locbench/candidates/<name>/`) with an env spec and a `run.py` exposing exactly two operations: `map(bag_dir) → map_dir` and `localize(bag_dir, map_dir) → est_traj` (TUM trajectory file). Files in, files out; candidates run as subprocesses in **per-candidate disposable conda envs** (`bench-<name>`) — no Docker, nothing installed into `brain`/`isaac`/`limx`/`hum`, removal is `conda env remove` (see agent memory `feedback-sandbox-env-policy`).
- **NEW — scorer + gates.** Est camera SE(3) → base SE(2) via the constant glide-mode mount transform (`camera_mounts.py`), anchored into the map frame **once** (fixed transform from the first ~2 s — never a per-run trajectory refit, which would hide exactly the constant bias E1 says kills nav). Per-stamp position [m] + yaw [deg] error vs GT → mean/max/p95 + fix coverage → pass/fail gates: mean pos < 0.10 m, max pos < 0.15 m, yaw < 10°, coverage ≥ 95%. Output: machine-readable `report.json` + GT-vs-est overlay plot on the baked occupancy PNG.
- **NEW — GT-passthrough reference candidate.** Replays `gt.jsonl` with injectable corruption (bias, noise, dropout, delay). This is how the harness itself is validated: it must pass clean, fail at 0.2 m injected bias, and fail at low coverage — before any real SLAM code exists.
- **NEW — CLI, the autonomous-agent surface.** `p -m humanoid.logic.locbench {record, export, run, score, board, env}` — idempotent steps, file logs, exit codes, `report.json`. The later agent loop drives only this surface.
- **NOT in scope:** implementing any real SLAM candidate (that is the agent's job *inside* this harness, a follow-up change), and the live in-brain wiring of the winner (the MAY-173 endgame, also follow-up).

## Capabilities

### New Capabilities

- `localization-bench`: the localization evaluation harness. Defines the frozen bag as the single candidate input, the two-function candidate adapter contract with per-candidate disposable envs, fixed-anchor SE(2) scoring against measured gates, the self-validating reference candidate, and the CLI surface an autonomous agent iterates against. Externally observable behavior: any candidate directory that satisfies the contract gets an identical, reproducible pass/fail verdict from the same frozen bags.

### Modified Capabilities

- None. `locbench` is additive; it consumes existing channels/contracts (`CameraFrame`, debug-pose) read-only and never touches the brain or World code paths.

## Impact

- **New code**: `logic/locbench/` (+ `logic/locbench/candidates/reference/`); tests under `tests/locbench/` (pure parts `brain`-marked).
- **Data**: frozen bags under `assets/bags/<scene>/<pass>_vN/` — gitignored (multi-GB RGBD).
- **Existing code**: no changes to `logic/oli/` or `logic/simulation/`; the recorder and driver are clients of existing sockets/APIs.
- **Envs**: harness core runs in `brain`; recording session needs a live `isaac` World (normal launcher run); candidate envs are created/destroyed by `locbench env`.
- **Linear**: [MAY-173](https://linear.app/may33/issue/MAY-173) — this change reframes it as "prove a localizer against the GT gate"; the swap-in of the winner stays the MAY-173 endgame.
