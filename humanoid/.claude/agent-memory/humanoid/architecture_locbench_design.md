---
name: architecture-locbench-harness
description: Locked locbench design (2026-07-13) — SLAM eval harness: socket-client recorder, frozen 2-pass bags, 2-function adapter contract, fixed-anchor SE(2) scoring, agent-drivable CLI
metadata:
  type: project
---

Design for **locbench** (proposed `logic/locbench/`) — the evaluation harness + pluggable
localization architecture that lets an autonomous agent implement/iterate SLAM candidates
against Isaac ground truth (MAY-173 track). Locked with Anton 2026-07-13.

- **Recorder = socket CLIENT, not World plugin** — taps the existing frame channel
  (`CameraFrame`: stamp_ns + RGB + depth[m] + intrinsics) and debug-pose channel (GT stamp,x,y,yaw,
  same sim clock). Zero World-side changes; records exactly what a real localizer sees at runtime.
- **Frozen 2-pass bags per scene**: `map` pass (coverage sweep) + `eval` pass — both driven by
  **scripted GoalCoordinate lists through Nav** (Anton's pick: reproducible, doubles as demo).
  Isaac isn't deterministic; frozen bags are. Export to TUM RGB-D for tool ingestion.
- **Adapter contract (UPDATED 13-07)**: candidates implement `LocalizationModule`
  (start/step/stop over `LocalizationIn`→`LocalizationOut`, see
  [[architecture-reason-module-separation]]); the bench's map/localize runner wraps the module —
  files in/out at the runner level. Scorer is tool-agnostic. In glide, joints hold home pose →
  camera→base is a CONSTANT transform (camera_mounts.py).
- **Scoring (UPDATED 13-07)**: the module contract requires **map-frame output** (module gets map
  artifacts + initial-pose hint, owns internal alignment) → scorer compares RAW, **no anchor at
  all** — constant bias fully visible. Gates from [[experiment-nav-pose-tolerance]]: mean pos
  <0.10 m, max <0.15 m, yaw <10°, fix-coverage ≥95%. Output `report.json` + GT-vs-est overlay on
  the baked occupancy PNG.
- **Scope decision (Anton)**: build the ABSTRACT harness + validation loop now; the agent
  implements specific SLAM candidates inside it later. Harness is proven by a built-in
  **GT-passthrough reference candidate** with injectable noise/bias/dropout — must pass clean and
  fail at 0.2 m bias (validates the scorer catches the E1 failure mode). Not by hand-running RTAB-Map.
- **Sandboxing**: per-candidate disposable conda envs, no Docker — see [[feedback-sandbox-env-policy]].
- **CLI = the autonomous-loop surface**: `p -m humanoid.logic.locbench {record,export,run,score,board}`,
  file logs + machine-readable report.json, so the agent loop is run→read→fix→rerun, budget-capped.

Links: [[research-nav-localization]], [[nav-brain-localizer-seam]].
