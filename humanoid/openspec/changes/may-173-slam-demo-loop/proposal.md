# Proposal — slam-demo-loop: explore → bake → deploy, end to end

## Why

MAY-173 must land a reliable-localization demo on the NVIDIA stack. The experiment phase concluded on 2026-07-16 with a causal verdict: the v1 map failure was **frame rate alone** — the same pixels that break cuVSLAM at 5 Hz track at 2–3 cm ATE / zero teleports at 30 Hz (cell experiment, ep0/ep1/ep2, identical-pixel bag variants). The capture recipe is locked (**30 Hz, stereo, visual-only**), the offline bake chain is proven (`cell_audit.sh` mirrors the overnight full bake), and cuSFM is eliminated on evidence.

What has never run, even once, is the **deployment side**: localizing a live robot against a baked map. Every result so far is offline odometry replay. The demo — and the credibility of the whole stack — lives in that untested half.

The demo is designed under one governing rule: **no privileged information** — the sim is treated as the real world. The robot may use only what a physical robot would have: its cameras, an operator's hints, and artifacts built from its own recordings. This rule has a structural payoff: since the occupancy grid is built from cuVSLAM's own poses, **the map frame IS the world frame**. No GT registration exists anywhere in the demo loop; GT survives only as a hidden validation oracle (ghost trail + error readout in dev_app), never as an input.

The demo scenario (Anton's scope, 2026-07-16):

1. **Phase 1 — Exploration.** Robot starts at a known spawn. Operator teleoperates through the warehouse recording stereo at 30 Hz (the locked recipe).
2. **Phase 2 — Offline build.** The recording is baked into three artifacts: cuVSLAM keyframe map (localization memory), cuVGL BoW index (global reloc), occupancy grid from depth + cuVSLAM poses (planning + UI map).
3. **Phase 3 — Deployment.** Robot spawns at a known position. dev_app shows the SLAM-built occupancy map; the operator clicks a goal; the robot plans a path and walks it — **steering on its estimated pose**, GT injection off, exactly the real deployment flow.

Scale strategy: prove the complete loop on the **demo zone** (the ep0/ep1/ep2 rack-aisle area) first, then scale to the full warehouse as a repetition, not a new build.

## What Changes

- **NEW — P3 live localizer (`logic/oli/loc/`).** A PyCuVSLAM sidecar process (own env, launcher-supervised — the PolicyRunner pattern): consumes the existing camera stream, `localize_in_map` snap from a pose hint, then continuous map-anchored tracking; publishes pose + tracking state (OK/degraded/LOST). The brain imports nothing from it — the invariance boundary is untouched.
- **NEW — stereo streams on the camera channel.** `CameraPublisher` gains `head_left`/`head_right` (cams exist in the body; only the publisher surface grows).
- **NEW — dev_app LOC MODE.** Pose-source toggle GT → localizer; hint at flip time = operator "you are here" click (demo) or current GT (validation rig); mapped-zone shading; GT ghost trail + live |est−GT| readout + loss/reloc markers (validation instruments, display-only); session log to disk for post-run autopsy.
- **NEW — P1 teleop-record + capture-writer surgery.** Recorder hooked into the teleop session; writer made big-walk-capable: stereo-only option, JPEG q95 (cuVSLAM is grayscale-internal), async bounded-queue writer; `.jpg` support in `rosbag_synth`.
- **NEW — P2 occupancy-from-depth.** Recorded depth + baked cuVSLAM poses → 2D occupancy grid **in map frame** (pure numpy, offline). Replaces the Isaac-GUI occupancy export in the demo loop; the GUI map remains eval/planning-GT tooling only.
- **NEW — P3 autonomy: click-goal → plan → follow.** Goal click on the SLAM occupancy map → path plan (reuse the coverage-route planner machinery on the SLAM grid) → brain-side follower emits GLIDE_CMD steering on the **estimated** pose. This closes the untested loop: localization error feeds back into control.
- **REUSE — the entire offline bake chain** (container `create_map_offline.py` + cuVGL BoW build + `cell_audit.sh`/`map_audit.py`) verbatim as the P2 production path. PyCuVSLAM map-format compatibility with the container bake is tested first; if incompatible, P2 additionally rebuilds the runtime map via PyCuVSLAM `save_map` from the same dump (identical pixels — equal validity, both audited).
- **NOT in scope:** cuVGL kidnap runtime (stretch — demo starts known-position), full-warehouse scale (scaling step after the zone demo passes), scene dressing (separate agent, off critical path), real-robot deployment, walk-policy locomotion (glide is the demo gait).

## Capabilities

### New Capabilities

- `slam-demo-loop`: the explore→bake→deploy demonstration loop under the no-privileged-information rule. Defines the three-phase scenario, the map-frame-is-world convention, the live localizer process + LOC MODE pose-source switch, the SLAM-built occupancy artifact, click-goal autonomy on estimated pose, and the validation instruments (GT as hidden oracle only). Observable behavior: an operator records a walk, bakes artifacts offline, then watches the robot navigate to a clicked goal while localizing purely against those artifacts.

### Modified Capabilities

- None in `openspec/specs/` (locbench/locdev specs are archived siblings; this change consumes their artifacts, does not modify them).

## Impact

- **New code**: `logic/oli/loc/` (localizer sidecar main + pose channel), `logic/simulation/mapping/occupancy_from_depth.py`, dev_app LOC MODE (map_render overlays, panel toggle, localizer source), teleop-record entry, follower module emitting GLIDE_CMD from est pose.
- **Modified code**: `logic/oli/comm/camera_publisher.py` (+codec/protocol tests) for stereo; `logic/simulation/mapping/recorder.py` (async/JPEG/stereo-only); `rosbag_synth.py` (`.jpg`); launcher Supervisor (+localizer subsystem).
- **Envs**: `bench-cuvslam` (py3.11) becomes a runtime env for the localizer process; no new env.
- **Artifacts**: per-map `registration.json` (bake-time GT survey) is **validation-only** — used to draw the GT ghost in map frame, never read by the demo loop.
- **Milestone (tomorrow's deliverable, 2026-07-17):** P1→P2→P3 full loop on the demo zone — teleop-recorded walk, baked artifacts, known-start deployment, click-goal, robot walks there while localizing. Stretch: cuVGL kidnap, full warehouse.
- Linear: [MAY-173](https://linear.app/may33/issue/MAY-173). Branch: `33may/may-173-nvidia-loc-stack`.
