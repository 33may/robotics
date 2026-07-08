---
name: limx-kinematic-projection-bus-relay
description: kinematic_projection is a REQUIRED MROS bus relay (not just a physics converter) — RobotState and RobotCmd only flow between sim and policy peers when it runs; IMU goes direct.
metadata:
  type: reference
---

`kinematic_projection` (the 9.4 MB ELF in `humanoid-mujoco-sim/prebuild/`) is not optional plumbing — it is the **MROS bus relay** that makes `RobotState` and `RobotCmd` flow between a sim peer and a policy peer. Without it on the bus, those topics silently deliver zero. Empirically mapped 2026-06-22 during MAY-147 Phase 7.

## The real bus topology

| Topic | Path | Needs kinematic_projection? |
|---|---|---|
| `ImuData` | sim →(direct)→ policy | **No** — goes peer-to-peer |
| `RobotState` | sim → **kinematic_projection** → policy | **Yes** — it subscribes to the sim's state, does AB↔PR conversion, and **republishes** on the topic policy peers read via `subscribeRobotState` |
| `RobotCmd` | policy → **kinematic_projection** → sim | **Yes** — converts policy PR cmd and republishes as `RobotCmdForSim` that the sim peer reads via `subscribeRobotCmdForSim` |

## How it was proven

- A minimal sim peer (`Robot(Humanoid, True)` publishing `publishRobotStateForSim` + `publishImuDataForSim` in a loop, no Isaac/UDS) + the MAY-145 probe as a policy peer: probe received **IMU (≈484 Hz) but ZERO RobotState**.
- Add `kinematic_projection` to the bus (`MROS_ETC_PATH=.../prebuild/etc ROBOT_TYPE=HU_D04_01 ./kinematic_projection`): probe immediately received **RobotState (≈483 Hz)** too.
- This also explains the deploy stand controller hanging on "Waiting for robot state data" and our sidecar's `cmd_recv=0` without it — both the state→policy and cmd→sim directions are gated on the relay.

## Launch

```
MROS_IP_LIST=127.0.0.x ROBOT_TYPE=HU_D04_01 \
  MROS_ETC_PATH=<humanoid-mujoco-sim>/prebuild/etc MROS_LOG_LEVEL=0 \
  <humanoid-mujoco-sim>/prebuild/kinematic_projection
```
It logs "joint calibration finished" after ~3 s; only then does it relay. Per-robot config lives in `prebuild/etc/kinematic_projection/HU_D04_01/` (twisted ankle + waist parallel-mechanism models).

## Open architectural question for the Isaac bridge (MAY-147)

This **partially overturns design D9**, which deemed `kinematic_projection` out of scope because Isaac runs serial USD = PR space. That was right for *physics* but wrong for *bus routing*: deploy-python cannot see Isaac's state, and Isaac cannot see deploy's cmd, unless the relay is on the bus.

But it does AB↔PR conversion assuming the sim is in **AB (parallel) space** (MuJoCo's MJCF). Our Isaac sim is in **PR (serial) space**. So naively spawning it would double-convert the ankle/waist joints. Options:
1. Spawn kinematic_projection anyway and accept mis-converted ankle/waist (other 27 joints pass through) — measure how bad.
2. ❌ **IMPOSSIBLE in Python** — a pure-Python identity relay cannot work: limxsdk's Python `Robot` exposes no `subscribeRobotStateForSim` (can't tap the sim's raw state) and `subscribeRobotCmd` is self-loopback only (can't tap the policy's raw cmd). kinematic_projection relays via a lower-level C++ MROS API not surfaced in the Python wheel. Proven 2026-06-22 (MAY-147). Don't re-attempt.
3. Configure kinematic_projection with an identity/serial model (untested).
4. **ONNX-direct**: run the exported policy inside the Isaac process (PR obs → PR actions applied via Oli's PD law), skipping MROS/kinproj entirely — aligns with the invariant-interface endgame ([[project_invariant_oli_interface]]).

Related: [[limx-sdk-role-gating]], [[isaac-pd-implicit-drive]], [[vendor-humanoid-mujoco-sim]].
