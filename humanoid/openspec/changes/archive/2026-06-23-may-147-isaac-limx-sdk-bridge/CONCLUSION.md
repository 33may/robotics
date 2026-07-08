# Conclusion — superseded by `may-147-oli-deployment-interface`

**Status:** closed 2026-06-23. Superseded by `may-147-oli-deployment-interface`.

## What this change set out to do

Make Isaac a drop-in peer on LimX's MROS bus — publish `RobotState`/`ImuData`, subscribe `RobotCmd` — so any `humanoid-rl-deploy-python` controller could drive Isaac as if it were the MuJoCo sim, via a Py 3.8 sidecar bridging the limxsdk ABI gap.

## Why it was closed (the load-bearing finding)

The bridge-as-spine premise was overturned. `RobotState`/`RobotCmd` only flow between MROS peers through `kinematic_projection`, a C++-level relay **not surfaced in the limxsdk Python wheel**; a pure-Python identity relay is impossible (limxsdk exposes no `subscribeRobotStateForSim`, and `subscribeRobotCmd` is self-loopback only). Proven 2026-06-22 — see memory `limx-kinematic-projection-bus-relay`. Driving Isaac from deploy-python therefore cannot work without also running `kinematic_projection`, which double-converts the serial-USD ankle/waist joints.

The deeper realization: we should not conform to LimX's bus at all. The successor defines our **own** schema-invariant contract between an independent World process and a Reason+Action brain — which also delivers the actual goal (a deployment-invariant interface that drives sim and real from one codebase), not just "Isaac looks like MuJoCo."

## What was built and validated (phases 1–6, all green)

- `bridge/protocol.py` — fixed-size UDS wire (HELLO/CMD/STATE_IMU), byte-identical across CPython 3.8.18 and 3.11.14.
- `bridge/sidecar.py` — Py 3.8 limxsdk process (sim-peer role), socket server loop, signal/shutdown lifecycle.
- `bridge/__init__.py` `OliBridge` — Py 3.11 client + sidecar spawn/connect/handshake/poll/close.
- `oli.py` — Isaac HU_D04_01 articulation: USD load+pin, IMU, PR↔Isaac permutation, **PD realized via PhysX implicit drive** (the key actuation finding — see memory `isaac-pd-implicit-drive`). Standalone + cross-env two-process smoke both passed.
- `_research/` audits: `joint_name_audit`, `isaac_dof_dump`, `imu_prim_audit`.

## What carries forward (reused, not discarded)

- `protocol.py` → the Communication wire of the successor (`Observation`≅STATE_IMU, `PolicyOut`≅CMD).
- `OliBridge`/`sidecar` transport halves → re-homed (World=server, Brain=client; roles inverted).
- `oli.py` core + implicit-PD drive → the Sim World body (slimmed: permutation relocates to `SimComm`, in-class bridge I/O stripped).
- `sidecar.py` limxsdk parts → the future `RealComm` edge (role flips sim-peer → real-robot).

## What dies

- Isaac-as-MROS-sim-peer framing; the `kinematic_projection` relay path; `load_oli.py --bridge` deploy-driving mode; the bridge-validation smoke scripts (`smoke_oli_bridge.py`, `fake_driver_smoke.py`, `e2e_deploy.sh`).

Successor: `openspec/changes/may-147-oli-deployment-interface/`.
