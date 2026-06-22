---
name: vendor-humanoid-mujoco-sim
description: LimX humanoid-mujoco-sim cloned at humanoid/vendor/ — asset source, SDK, reference simulator; index doc at humanoid/docs/vendor/humanoid-mujoco-sim.md
metadata:
  type: reference
---

# humanoid-mujoco-sim vendor repo

**Location:** `/home/may33/projects/ml_portfolio/robotics/humanoid/vendor/humanoid-mujoco-sim/`
**Index doc:** `/home/may33/projects/ml_portfolio/robotics/humanoid/docs/vendor/humanoid-mujoco-sim.md`
**Cloned:** 2026-06-18 (MAY-136). Recursive — three submodules pinned (see index § 2).

LimX's own MuJoCo harness. For us this repo is the **canonical bundle** that holds the things we'll keep coming back to as the humanoid project grows. We don't run MuJoCo, but every piece below points at something we will (or may) reuse.

## What lives here, in our terms

| Component | What it gives us |
|---|---|
| `humanoid-description/HU_D04_description/usd/` | Source-of-truth USD assets for Oli. Layered `base → physics → sensor`. Drop-in candidate for Isaac. |
| `humanoid-description/HU_D04_description/urdf/` | Serial-equivalent kinematics. Authoritative for joint names + DOF order on the SDK side. |
| `humanoid-description/HU_D04_description/xml/` (MJCF) | Reference for inertia, friction, damping. Topology differs from URDF/USD (parallel mechanism — Achilles rods). |
| `humanoid-description/HU_D04_description/meshes/HU_D04_01/` | STL pool shared by all three asset views. |
| `simulator.py` | Reference implementation of the SDK ↔ sim contract. PD-with-feedforward law per joint, IMU + state publish layout. Spec for our own Isaac bridge if/when we mirror it. |
| `prebuild/kinematic_projection` (+ `etc/...`) | Parallel ↔ serial coordinate translator. Used by MuJoCo path. Relevant only if we ever author a parallel USD for high-fidelity locomotion sim2real. |
| `limxsdk-lowlevel/include/limxsdk/*.h` | C++ headers; canonical packet layouts (`humanoid.h`, `datatypes.h`). |
| `limxsdk-lowlevel/python3/*/limxsdk-4.0.1-py3-none-any.whl` | Python SDK wheel. **Pinned to Python 3.8 ABI** (ships `libpython3.8.so.1.0`). Will not work cleanly in `isaac` env (3.11) or `lerobot` env (3.12) — needs its own env. |
| `robot-joystick/` | Real-hardware joystick pairing. Sim-only sessions don't need it. |

## Relationship to our project plan

- **Isaac Sim asset import** (MAY-136 and any downstream Isaac work) → pull from `humanoid-description/HU_D04_description/usd/` or `urdf/`. Don't use MJCF.
- **SDK boundary** (any sim2real story, including the Q8 "swap 127.0.0.1 ↔ 10.192.1.2" plan) → `limxsdk-lowlevel` wheel is the wire format. Our Isaac driver code should accept the same `RobotCmd{mode, q, dq, tau, Kp, Kd}` packet so policy code is identical sim ↔ real.
- **Joint/DOF reference** (sweep scripts, observation layout, policy I/O) → URDF order is authoritative.
- **Parallel mechanism fidelity** (only if locomotion sim2real becomes a headline) → `kinematic_projection` + a parallel USD we'd have to author.
- **MJCF inertia/friction values** → useful as starting-point physics tuning numbers if Isaac USD physics defaults feel off.

## When to come back to this memory

- Starting any Isaac Sim work on Oli — point at the USD path.
- Designing the Isaac actuator interface — read § 5 of the index doc, mirror the PD+ff packet.
- Choosing a Python env for SDK use — remember the 3.8 ABI pin.
- Considering MuJoCo for anything — sanity-check by reading § 4 (topology) and § 6 (kinematic_projection) first.
- If asked "where do the Oli assets come from" — this repo.

Linked memories: [[vendor-humanoid-rl-deploy-cpp]] (deleted submodule from prior session — may be re-cloned), [[oli-sdk-interfaces]] (Q1–Q9 summary on MAY-137).
