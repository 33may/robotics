# Design — MuJoCo World via a LimX Communication edge

Date: 2026-06-25 · Owner: Anton (architect) · Status: approved topology, pre-implementation

## Context

Oli already walks in the LimX MuJoCo simulator (MAY-142, 2026-06-19) under the deploy
stack as-shipped: `simulator.py` (MuJoCo + PD) + the `kinematic_projection` ELF (PR↔AB)
+ a `walk_controller` policy peer on the MROS bus, steered by the keyboard `robot-joystick`.

We want Oli operable in that same MuJoCo sim, but driven by **our** deployment-invariant
brain (Reason + Action + our joystick teleop), inside the may-147 architecture — *without*
rebuilding the physics or the parallel-mechanism math.

The realization that shapes this design: **to `limxsdk`, the MuJoCo sim is
indistinguishable from the real robot.** Both are MROS peers that publish
`RobotState`/`ImuData` and consume `RobotCmd`. So the Communication edge that talks to the
MuJoCo sim *is* the real/limx edge (the deferred `RealComm`), just pointed at the sim. The
bus and the PR↔AB ELF live entirely on the world side of the seam, behind the existing
deploy process — never in the brain, never in our new code. This is the architecture-native
home for the bus (design D4: "Communication is the only world-aware layer, deployed at the
world edge").

## Goals / Non-Goals

**Goals**
- Oli stands then walks in the MuJoCo viewer, steered by **our** teleop joystick, with the
  deploy's policy controller fully replaced by our Reason + Action brain.
- Reuse the unchanged MuJoCo process (`simulator.py` + ELF + PD) and the unchanged Brain.
- Build the limx Communication edge early — this is the first real exercise of `RealComm`,
  buying sim/real interface parity ahead of hardware.

**Non-Goals**
- Rebuilding MuJoCo physics, the PD law, or the PR↔AB projection (all reused as-is).
- Touching the Isaac World, the contracts, the Brain, or the joystick teleop.
- A byte-identical (vs schema-identical) wire; multi-machine; the physical-robot edge.

## Topology (3 processes, matches real 1:1)

```text
Brain (py3.11 `brain`, UNCHANGED)              MuJoCo sim (py3.8 `limx`, AS-IS)
 joystick → reason → action(ONNX) →            simulator.py + kinematic_projection ELF
   PolicyOut / Observation                       (AB physics + deploy PD — walks today)
        │  UDS SEQPACKET, 3 PR contracts                 ▲   MROS bus (localhost)
        ▼                                                │   RobotState/ImuData ↓  RobotCmd ↑
   LimX edge (py3.8 `limx`, NEW) ────────────────────────┘
   WorldComm(server) + LimxBody(policy-role limxsdk peer)
```

- **MuJoCo sim** — run unchanged: `python vendor/humanoid-mujoco-sim/simulator.py`
  (env `limx`, py3.8, `limxsdk-4.0.1`, `mujoco 3.2.3`; existing working-tree patches:
  XML quat fix, `stand.autostart`, freeze-until-first-cmd). It spawns the ELF itself.
- **LimX edge** (NEW) — our py3.8 process. A limxsdk **policy-role** peer (drop-in for the
  deploy `walk_controller`, minus the ONNX) that bridges the bus to our UDS contracts.
- **Brain** — `python logic/oli/brain_main.py` unchanged; connects over UDS as the client.

## Components

### MuJoCo sim — reused, not built
Owns physics, PD (`tau = Kp(q*−q) + Kd(dq*−dq) + tau_ff` into `mjData.ctrl`), and the ELF
that converts our PR `RobotCmd` → AB actuation and AB sensors → PR `RobotState`. The edge
sees only PR-space `RobotState`/`RobotCmd`. **No AB math in our code.**

### `LimxBody` — NEW (`logic/simulation/mujoco/limx_body.py`)
A limxsdk policy peer that implements the existing **body protocol** so `WorldComm` can use
it exactly like the Isaac `Oli`:

| Body protocol (Isaac/native order) | `LimxBody` realization (over the bus) |
|---|---|
| `dof_names` | `robot.getMotorNames()` (SDK order; set-equal to `PR_ORDER`) |
| `read_joints_isaac() -> (q, dq, tau)` | latest `RobotState.{q,dq,tau}` snapshot → np arrays |
| `read_imu() -> (acc, gyro, quat_wxyz)` | latest `ImuData.{acc,gyro,quat}` (quat already wxyz) |
| `apply_isaac(q_des, dq_des, tau_ff, kp, kd)` | build `RobotCmd`, `publishRobotCmd` |

limxsdk facts (verified against the wheel at `envs/limx/.../limxsdk/`):
- `robot = Robot(RobotType.Humanoid)` — `is_sim=False` ⇒ **policy role**. `robot.init("127.0.0.1")`.
- Subscribe callbacks fire on an SDK thread → store-latest into a slot; loop snapshots it.
- `RobotState`: `stamp(ns), q[31], dq[31], tau[31], motor_names`. `ImuData`: `acc[3], gyro[3], quat[4]=wxyz`.
- `RobotCmd`: `stamp, mode[31] (=0), q, dq, tau, Kp, Kd, motor_names, parallel_solve_required[31]=True`.
- **`RobotCmd` is a streaming setpoint, not latched — must be republished every tick.**
- `publishRobotCmd` is loop-thread-safe; SensorJoy is **not** subscribed (the brain owns the joystick).

### `WorldComm` — reused (promote out of `isaacsim/`)
`SimComm` is already body-agnostic and imports no isaacsim. Promote it to a shared module
(`logic/oli/comm/world.py` as `WorldComm`; keep `SimComm = WorldComm` alias in
`isaacsim/sim_comm.py` for the Isaac path). It owns the **name-based** PR↔native permutation
(so the head_yaw/head_pitch order swap is handled automatically), serves the brain over UDS,
and applies `PolicyOut` via `body.apply_isaac`. Must import cleanly under py3.8 (pure
numpy + stdlib — verify `codec`/`protocol`/`contracts` carry no 3.9+ runtime syntax).

### `limx_world_main` — NEW (`logic/simulation/mujoco/limx_world_main.py`)
The edge entrypoint, sibling of `sim_world_main.py` but **the sim steps itself** (no
`world.step()` here). Build `LimxBody` + `WorldComm`; wait for the first bus samples; serve
the brain; then run a `limxsdk Rate(1000)` loop:

```text
each 1 kHz tick:
  WorldComm.publish(stamp = latest RobotState.stamp)   # PR Observation → brain (D8 pacing)
  cmd = WorldComm.receive_latest()                     # drain newest PolicyOut (latest-wins)
  if cmd: last = cmd
  if last:  WorldComm.apply(last)                      # → LimxBody → publishRobotCmd (republish/hold)
  else:     publish a stand-hold/zero cmd              # pre-first-PolicyOut
  rate.sleep()
```

### Brain — reused unchanged
`brain_main.py` with `--joystick socket` (live teleop) or `--walk-after`/`--vx` (scripted).
Stamp-paced at 100 Hz off the Observation stamp (D8). Emits 100 Hz `PolicyOut`; the edge
republishes it at 1 kHz.

## Decisions

- **LD1 — The edge is a limxsdk policy-role peer (`is_sim=False`), a drop-in for the deploy
  `walk_controller` minus the ONNX.** It does not run physics or projection; it relays.
- **LD2 — Reuse `WorldComm` via a `LimxBody`, not a new transport.** The body protocol is the
  seam; the bus is "just another body." No new wire, no new permutation logic.
- **LD3 — The edge owns the 1 kHz `RobotCmd` republish (hold last `PolicyOut`).** Answers the
  design's open `RealComm` question: brain sends 100 Hz actions, the py3.8 edge streams them
  at 1 kHz. Live per-tick torque clamp is a deferred real-fidelity refinement, not needed in sim.
- **LD4 — The joystick stays in the Brain; the bus carries no `SensorJoy`.** Velocity command
  enters at Reasoning (our teleop), flows into the obs vector inside the PolicyRunner.
- **LD5 — Name-based PR↔SDK permutation** (existing `WorldComm` behaviour) — robust to the
  head order swap and any SDK reordering, as long as the name sets match.
- **LD6 — Pace by `RobotState.stamp` (bus/sim time), per D8.** Identical mechanism to Isaac.

## Error handling

- **Startup:** edge blocks until the first `RobotState` and `ImuData` arrive; brain blocks on
  `connect()` until the edge's UDS server is up (existing retry in `BrainComm.connect`).
- **Brain stalls / disconnects:** edge keeps streaming the last cmd briefly, then a watchdog
  switches to a damping cmd (`Kp=0`, small `Kd`, `q_des=q`) — fail-safe (design RC10). On UDS
  EOF the edge logs and either holds-damp or exits cleanly.
- **Sim/bus loss:** if no `RobotState` for N ms, edge logs and exits (sim is authoritative).

## Testing (committed TDD, no throwaway scripts)

- **`LimxBody` unit** — inject a `FakeRobot` (no SDK) exposing the limxsdk method surface
  (`getMotorNames`, `subscribeRobotState/ImuData`, `publishRobotCmd`); assert the body
  protocol maps fields correctly and `apply_isaac` builds a well-formed `RobotCmd`
  (mode 0, lengths 31, motor_names echoed). Runs in any env (no limxsdk import at test time).
- **`WorldComm` ⇄ `BrainComm` loopback** — already covered by `tests/oli/comm/`; re-run after
  the promote/alias to prove the Isaac path is unbroken.
- **Edge-loop unit** — drive `limx_world_main`'s step function with a fake body + fake comm:
  assert republish-every-tick, hold-last, and watchdog-damp transitions.
- **py3.8 import test** — assert `WorldComm`/`codec`/`protocol`/`contracts` import under the
  `limx` env.
- **Integration milestone (manual)** — `simulator.py` up → `limx_world_main` up → brain up;
  Oli stands then walks in the viewer under joystick. This is the acceptance gate, not a unit.

## Acceptance criterion

With three processes running — `simulator.py` (+ELF), our `limx_world_main` edge, and our
Brain — **Oli stands and then walks in the MuJoCo viewport, steered by our teleop joystick**
(vx/vy/wz, STAND→WALK), with no LimX policy controller in the loop.

## Risks / Open items

- **py3.8 compatibility** of `WorldComm`/`codec`/`contracts` — verify no 3.9+ runtime syntax;
  the wire was designed 3.8∩3.11, so low risk.
- **`parallel_solve_required` / `mode=0`** — the example sets `[True]*31` and `mode=[0]*31`;
  the deploy `walk_controller` omits `parallel_solve_required`. Verify the sim's
  `subscribeRobotCmdForSim` handler accepts our cmd identically (smoke test early).
- **Role-gating timing** — the ELF (started by `simulator.py`) must be up before the policy
  peer's RobotState/RobotCmd flow (memory `limx-kinematic-projection-bus-relay`); sequence the
  launch (sim first, then edge).
- **1 kHz UDS load** — Observation publish can be throttled (e.g. 200–1000 Hz) independently of
  the 1 kHz bus republish if needed.

## Scope note

This builds the deferred `RealComm` edge ahead of the may-147 plan (which scoped it
"stub only"). Track as its own Linear change (e.g. "Oli LimX/MuJoCo Communication edge"),
distinct from the Isaac may-147 milestone.
