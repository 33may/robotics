# Design — locbench, the localization evaluation harness (localization-bench)

## Context

The nav brain is complete and world-invariant (`reason/nav/`: costmap → A* → pure-pursuit) and runs on ground truth through the `Localizer` seam. The localization research fixed the candidate landscape (RTAB-Map now / cuVSLAM later) and — via closed-loop experiments on the real nav stack — the tolerance budget: **constant position bias < 0.1 m is binding**; Gaussian noise and fix rate are comparatively cheap. Design locked with Anton 2026-07-13 (agent memory `architecture-locbench-harness`): build the **abstract harness + validation loop now**; specific SLAM implementations are a later autonomous-agent job inside it.

Ground rules inherited from the repo: invariance boundary (nothing here imports `isaacsim`/`limxsdk`; the recorder talks to sockets), no Docker (per-candidate disposable conda envs), TDD in-repo, MD tables in docs.

## Goals / Non-Goals

**Goals**

- Make "this localizer works" computable: frozen input → contract → gates → `report.json`.
- Record the *runtime interface*, not Isaac internals — the same recorder must work against the real robot later.
- Self-validating: the harness proves itself with a reference candidate before any SLAM code exists.
- Agent-drivable end to end: every step a CLI command with file logs and machine-readable output.

**Non-Goals**

- Implementing RTAB-Map/cuVSLAM/etc. adapters (follow-up change, agent-driven).
- Live out-of-process localizer node + bus wiring into the brain (MAY-173 endgame, follow-up).
- Real-robot recording (interface-compatible by construction; untested here).
- Online/streaming evaluation — the bench is offline replay by design.

## Decisions

### D1 — The recorder is a socket client, not a World plugin

It subscribes to the existing camera frame channel (`CameraStreamReader`) and debug-pose channel — the exact surfaces a real localizer consumer sees, latest-wins drops included. Zero changes to `glide_world_main.py` or the brain; runs in any env; swapping Isaac for the real robot later changes nothing in the recorder.

### D2 — Frozen two-pass bags, scripted drives

Per scene, two bags: `map` (coverage sweep) and `eval` (representative drive), both produced by a scripted `GoalCoordinate` list fed to `Nav` (reproducible; doubles as a demo artifact). Isaac rendering is not deterministic — freezing the bags quarantines that at record time; everything downstream is exactly replayable.

### D3 — Bag format: raw on disk + TUM RGB-D export

Bag = `rgb/<stamp>.png` + `depth/<stamp>.png` (uint16 millimeters — matches the wire codec exactly, 0 = invalid) + `gt.jsonl` + `intrinsics.json` + `meta.json` (achieved fps, drop stats, scene, pass). The TUM RGB-D layout is the lingua franca candidate tools already ingest (`rgb.txt`/`depth.txt`/`groundtruth.txt`, depth factor 5000); the exporter converts mm → TUM scale. Both cameras are recorded as separate streams; candidates declare which they consume.

### D4 — Candidate contract: two functions, files in / files out

`map(bag_dir) → map_dir` and `localize(bag_dir, map_dir) → est_traj.txt` (TUM trajectory: `stamp tx ty tz qx qy qz qw`). Candidates run as **subprocesses** (`conda run -n bench-<name> python run.py …`) — never imported into our processes, so their ABI/deps can be arbitrary. Failure = nonzero exit + captured log; the scorer never knows which tool ran.

### D5 — Per-candidate disposable conda envs, no Docker

Each candidate dir declares its env (`environment.yml` or `setup.sh`); `locbench env create/remove <name>` manages `bench-<name>`. Anton's two requirements (memory `feedback-sandbox-env-policy`): (a) removal leaves the machine as if it was never there; (b) a picked candidate works immediately in our env, no container plumbing. Nothing is ever installed into `brain`/`isaac`/`limx`/`hum`.

### D6 — Fixed one-time anchor, never per-run trajectory alignment

A SLAM estimate lives in its own map frame. We anchor it into the Isaac/world frame with **one rigid SE(2) transform fit on the first ~2 s of matched pairs**, then score everything else against that fixed anchor. A per-run Umeyama/full-trajectory fit would absorb exactly the constant bias E1 measured as the nav-killing failure mode — the metric must keep it visible.

### D7 — Gates from the measured budget, machine-readable verdicts

Per matched stamp (nearest-neighbor on the shared sim clock, Δt ≤ 100 ms): position error [m], yaw error [deg]. Report mean/max/p95 + **fix coverage** (fraction of eval GT stamps, after a 2 s warmup, with a fix). Gates: mean pos < 0.10 m, max pos < 0.15 m, yaw < 10°, coverage ≥ 95% — from the E1/E2 budget (memory `experiment-nav-pose-tolerance`). Output per run: `report.json` (all numbers + gate verdicts + provenance) and an overlay plot (GT vs est on the baked occupancy PNG) for human eyes.

### D8 — Camera→base is one constant matrix in glide mode

SLAM estimates the **camera** trajectory; the gate is on the **base** SE(2) pose. In glide the joints hold the home pose, so camera-in-base is a constant transform derived from `camera_mounts.py` — one fixed matrix applied by the scorer, recorded into `meta.json` at record time. (When Oli steps for real, this becomes per-frame FK from `Observation` — out of scope here, seam noted in code.)

### D9 — The reference candidate proves the harness

`candidates/reference/` replays `gt.jsonl` through the full contract (subprocess, TUM output) with injectable corruption: constant bias, Gaussian noise, dropout, stamp delay. Harness acceptance = reference **passes clean**, **fails at 0.2 m bias**, **fails at < 95% coverage**. This validates the scorer catches precisely the E1 failure mode, with zero SLAM dependencies.

### D10 — The CLI is the autonomous-loop surface

`p -m humanoid.logic.locbench {record, export, run, score, board, env}`. Every command: idempotent, logs to a file under the run dir, meaningful exit code, machine-readable artifact. The later agent loop is strictly `run → read report.json/logs → edit candidate → rerun`, budget-capped — no bespoke agent plumbing inside the harness.

## Risks / Trade-offs

- **Bag size** (720p RGBD ≈ 1–2 GB/min/camera) → configurable record fps (default ~10–15 Hz, plenty for SLAM) + uint16-mm png depth; bags gitignored, meta records what was kept.
- **Latest-wins frame drops under recorder load** → accepted: that IS the runtime interface; `meta.json` reports achieved fps so a starved recording is visible, not silent.
- **Stamp association** (frames stamp on render ticks, GT on cmd ticks) → same sim clock, ≤ 10 ms granularity; nearest-neighbor with Δt ≤ 100 ms is safe at glide speeds.
- **Scripted driver blocked on the Nav execution gate** (in flight) → only the *record* step depends on it; scorer/contract/reference/exporter build and validate on synthetic bags meanwhile.
- **Offline bench ≠ live wiring** — a candidate passing offline still needs the OOP node + bus bridge (follow-up); the contract keeps that gap small (same inputs, same SE(2) out).

## Migration Plan

Additive — nothing migrates. Follow-ups after this change: (1) agent-driven candidate implementations (RTAB-Map first), (2) winner's live node + `Localizer` seam wiring (MAY-173 endgame).

## Open Questions

- Does the eval pass eventually also want a manual joystick drive as a second, harder trajectory? (Scripted locked for v1; additive later.)
- Head cam only vs both cams as the default candidate input — v1 records both, RTAB-Map will start on head; revisit after first scoreboard.
