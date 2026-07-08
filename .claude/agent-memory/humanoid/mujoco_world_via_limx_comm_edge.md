---
name: mujoco-world-via-limx-comm-edge
description: MuJoCo World mode = a limxsdk Communication edge (LimxBody behind WorldComm) reusing the unchanged sim+brain; this edge IS the deferred RealComm.
metadata:
  type: project
---

Decision (Anton, 2026-06-25): run Oli in the LimX MuJoCo sim under our may-147
architecture by building a **Communication edge**, NOT by rebuilding the sim. The MuJoCo
process (`vendor/humanoid-mujoco-sim/simulator.py` + the `kinematic_projection` ELF) runs
UNCHANGED — it already walks (MAY-142, 2026-06-19).

Topology (3 processes): Brain (py3.11 `brain`, unchanged) ⇄ UDS 3 PR contracts ⇄ LimX
edge (py3.8 `limx`, NEW: `logic/simulation/mujoco/limx_world_main.py` + `limx_body.py`) ⇄
MROS bus ⇄ sim.

Key realization: **to limxsdk the MuJoCo sim is indistinguishable from the real robot**, so
this edge IS the deferred `RealComm` (design Non-Goal said stub-only) — building it now buys
sim/real parity. PR↔AB and physics stay inside the sim process, never our code; the edge
only maps limxsdk PR RobotState/RobotCmd ↔ our PR Observation/PolicyOut.

Reuse: `SimComm` was promoted to engine-agnostic `WorldComm` (`logic/oli/comm/world.py`;
`isaacsim/sim_comm.py` is now a back-compat shim). `LimxBody` implements the SAME duck-typed
body protocol as the Isaac `Oli` (`dof_names`/`read_joints_isaac`/`read_imu`/`apply_isaac` —
`_isaac` suffix just means "native order"), so WorldComm + the whole Brain are reused as-is.

Load-bearing integration facts (verified against the wheel):
- Edge = limxsdk POLICY peer: `import limxsdk.robot.Robot as Robot; Robot(RobotType.Humanoid)`
  (is_sim=False = policy), `robot.init("127.0.0.1")`. Subscribe cbs fire on an SDK thread →
  store-latest. `getMotorNames()`/`getMotorNumber()` → 31.
- `RobotCmd` is a streaming setpoint (NOT latched) → the edge republishes the held PolicyOut
  at 1 kHz (answers the design's "RealComm 1 kHz republish" open question: YES, edge owns it).
  `mode=[0]*31` (torque-position hybrid), `parallel_solve_required=[True]*31`.
- `ImuData.quat` is (w,x,y,z); RobotState/Cmd are PR space (post-ELF). WorldComm's name-based
  PR↔native permutation absorbs the head_yaw/head_pitch order swap for free.
- Run order: sim FIRST (it spawns the ELF), THEN the edge — the ELF must be up or
  RobotState/RobotCmd deliver zero (see [[limx_kinematic_projection_bus_relay]]), then brain.

**Why:** deployment-invariant thesis — one brain drives Isaac, MuJoCo, real. MuJoCo gives a
known-good walker to validate the Brain and isolate the Isaac topple to physics fidelity
([[isaac_walk_physics_fidelity]]).
**How to apply:** add a new World by writing a body (the duck-typed protocol) behind
WorldComm; keep the bus/SDK out of the brain. Track as its own Linear change (RealComm pulled
forward). Spec: `docs/superpowers/specs/2026-06-25-mujoco-limx-world-design.md`.
