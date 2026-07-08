# Design — Oli deployment interface

## Context

Isaac is the project's primary 3D environment, but it cannot share a process with `limxsdk` (ABI-locked to CPython 3.8; Isaac runs 3.11). The previous change `may-147-isaac-limx-sdk-bridge` tried to make Isaac a peer on LimX's MROS bus. That deadlocked: `RobotState`/`RobotCmd` only flow between MROS peers through `kinematic_projection`, which is a C++-level relay not surfaced in the limxsdk Python wheel, and a pure-Python identity relay is impossible (proven 2026-06-22 — see memory `limx-kinematic-projection-bus-relay`).

The pivot: stop conforming to LimX's bus; define **our own** schema-invariant contract between an independent **World** process and a **Reason+Action brain**. The forced constraints that shape the design:

- **`limxsdk` = Py 3.8**, and the brain will grow heavy modern dependencies (SLAM, VLA, CUDA) that cannot live in 3.8. So on the **real** path the brain and the SDK edge cannot share a process — a process boundary is forced there.
- **North star**: end-of-summer-2026 reasoning demo + parallel acting research, both required to run on sim and real from one codebase. Interface parity is therefore the priority, above speed-to-first-demo.

Already built and reusable: `oli.py` (Isaac articulation + implicit-PD drive, ~90% of the Sim World body), `bridge/protocol.py` (a cross-version 3.8∩3.11 UDS wire), and the `OliBridge`/`sidecar` transport plumbing.

## Goals / Non-Goals

**Goals**
- A deployment-invariant brain (Reason+Action) authored once, driving sim and real unchanged.
- Sim↔real **interface parity**: the World is an independent process in both, so the brain↔world boundary is exercised identically — timing, latency, and serialization bugs surface in sim, not on hardware.
- A falsifiable first proof: Oli walking in Isaac on the LimX walk ONNX, through the interface.

**Non-Goals (deferred; the seam must not preclude them)**
- Building `RealComm` (py3.8 limxsdk edge) — interface/stub only this change.
- A vectorized in-process RL training path — Oli stays reusable for it, not built here.
- Reasoning beyond teleop (SLAM / VLA / autonomy).
- PR↔AB parallel-mechanism conversion — sim runs serial/PR; ankle/waist fidelity gap accepted and documented.
- Distributed / multi-machine; byte-identical (vs schema-identical) wire.

## Architecture (C4)

**Level 1 — Context.** Two objects: the **Robot** (our system) observes and actuates the **World** (external: Isaac sim *or* physical robot).

**Level 2 — Containers.** A **Robot Runtime** (the brain: orchestration, loop, logging; exposes `connect(world)`) talks to the **World** over the Communication boundary. The World is a separate process; its env differs (sim = isaac py3.11, real = robot + limx py3.8).

**Level 3 — Components inside the Robot Runtime.** `Communication` (edge adapter; `SimComm`/`RealComm`) ⇄ `Reasoning` (Teleop + JoystickAdapter, fed by the Operator's joystick) ⇄ `Action` (PolicyRunner/ONNX). An `Orchestrator` conducts `read → reason → act → write`, owns logging/recording (the only thing that sees every contract), and the watchdog. Communication is our component but is deployed at the world edge.

```text
            ┌──────────────────────────┐
            │  BRAIN process (py3.11+)  │   env CONSTANT across sim & real
            │  Comm-client + Reason +   │   (onnxruntime/scipy; later torch/VLA)
            │  Action(ONNX) + Orchestr. │   imports NO isaacsim, NO limxsdk
            └─────────────▲─────────────┘
                          │  schema-invariant contract (UDS; encoding TBD per world)
          ┌───────────────┴───────────────┐
          ▼                               ▼
 ┌──────────────────────┐      ┌──────────────────────────────┐
 │ SIM WORLD (py3.11)   │      │ REAL WORLD (py3.8)            │
 │ Isaac + Oli +        │      │ robot + limxsdk EDGE          │
 │ SimComm server       │      │ (RealComm) — DEFERRED         │
 └──────────────────────┘      └──────────────────────────────┘
```

## Contracts (canonical PR, 31 joints, wxyz quat)

```text
Observation   World→[Comm]→Reason    ≅ RobotState+ImuData : stamp, q[31], dq[31],
                                       tau[31], imu{acc[3], gyro[3], quat_wxyz[4]}
PolicyIn      Reason→Action          observation + intent{mode∈{STAND,WALK}, v_x, v_y, w_z}
PolicyOut     Action→[Comm]→World    ≅ RobotCmd : q_des, dq_des, tau_ff, kp, kd, mode  [31]
```

`Observation ≅ STATE_IMU` and `PolicyOut ≅ CMD` in `bridge/protocol.py` — so the existing wire serves both seams. Adopting LimX's RobotState/RobotCmd shape as our invariant buys free sim/real fidelity. Policy memory (history ring, last_actions) lives inside the PolicyRunner and crosses no contract.

## Run cycle — async, two clocks, latest-wins

```text
World clock  1 kHz, free-running, AUTHORITATIVE (never waits for the brain)
Brain clock  policy @100 Hz, paced by WORLD-STAMP Δ≥10 ms (trained decimation=10)
Wire         latest-wins both ways; non-blocking; World holds last PolicyOut
```

Hot path (also the Phase-3 obs-builder spec): World reads articulation+IMU → SimComm permutes Isaac→PR → emits Observation; brain drains-latest, and when the stamp has advanced ≥10 ms sim it steps: Reason adds intent → PolicyRunner encodes obs[102] (`ang_vel·0.25 | gravity=Rᵀ[0,0,-1] (quat wxyz→xyzw) | [v_x,v_y,w_z] | (q−default)·1 | dq·0.05 | last_actions`), pushes into the 5-deep ring → obs[510], runs ONNX, clips, torque-clamps on live q/dq, resolves `q_des=a·scale+default`, stores last_actions, emits PolicyOut; SimComm permutes PR→Isaac and applies via the implicit PD drive each 1 kHz tick until the next PolicyOut.

## Decisions

- **D1 — World is an independent process in sim AND real.** Alternative: co-locate the brain in Isaac's interpreter for sim (one process, no IPC, fastest). Rejected: it hides every serialization/latency/jitter bug until hardware. Paying the IPC cost in sim is the whole point — it makes "works in sim" imply "works in real" at the integration layer.
- **D2 — Brain runs in a dedicated `brain` Py 3.11 env.** Alternative: reuse the `isaac` env + onnxruntime. Rejected: the brain's future torch/VLA/CUDA pins must not collide with Isaac's; a clean env also models the real deployment honestly (brain never imports `isaacsim`).
- **D3 — Three canonical-PR contracts; policy memory excluded.** Alternative: pass the encoded 510-vector or include last_actions in the contract. Rejected: that leaks policy specifics across the edge and couples Comm to the policy. The runner owns encoding + memory.
- **D4 — Communication is the only world-aware layer; the PR↔Isaac permutation relocates out of `oli.py` into `SimComm`.** Alternative: keep the permutation in Oli. Rejected: it would make the body partly policy/contract-aware; the clean line is policy-specifics in the brain, world-specifics in Comm, pure PR between.
- **D5 — Single Reason→Policy seam.** There is no separate `Command` contract between Reason and Action; Reasoning consumes (Observation, joystick) and emits PolicyIn directly. Alternative: a Reason→Command→Action chain. Rejected as redundant — it is always Reason(process the world)→Policy.
- **D6 — PolicyOut is a fully resolved RobotCmd (all logic in the brain).** Alternative: PolicyOut = the 31 raw policy actions, resolved by World/Comm. Rejected: resolution (scale/default/gains) is policy-specific and must stay in the brain so Comm/World stay policy-agnostic. Matches LimX (controller resolves, robot applies).
- **D7 — Async free-run, latest-wins, held command.** Alternative: synchronous lock-step (brain steps the world; deterministic). Rejected: not real-faithful and contradicts D1. Determinism for RL training is recovered via a separate in-process tap (non-goal here).
- **D8 — Pace the policy by world-stamp Δ, not brain wall-clock.** The walk policy was trained at decimation=10 (10 ms steps); its 5-deep history spans 50 ms of sim time. Wall-clock pacing under RTF<1 would feed a wrong dt and smear the history → distribution shift → fall. Stamp-pacing keeps the trained dt exact at any RTF and is identical on real (sim-time = wall-time). This is the load-bearing decision for transfer.
- **D9 — Freeze-until-command startup.** The World emits Observations of the static pose but does not step until the first PolicyOut. Alternatives: spawn pinned then release at runtime (PhysX re-init pain); free base + stand-hold default cmd (finite-Kp droop risk). Freeze is simplest and removes the free-fall window entirely; the brain bootstraps off the frozen-pose Observation.
- **D10 — World = server, Brain = client.** Inverts the old bridge (where Isaac was the client). Matches real: the robot is the always-on endpoint a controller `init()`s into.
- **D11 — Reuse the bridge transport; re-home the two halves.** OliBridge already does the world-side direction (send state / recv cmd) and `sidecar`'s socket loop does the brain-side (send cmd / recv state). The new design re-homes them (World=server, Brain=client) and inverts client/server. The wire (`protocol.py`) is unchanged and already 3.8∩3.11.

## Risks / Trade-offs

- **Isaac can't sustain 1 kHz wall-clock with rendering (RTF<1)** → D8 stamp-pacing makes the brain RTF-robust; render decimation keeps the viewport cheap.
- **Free-base balance under finite-Kp PD** (does Oli actually stay up?) → freeze-until-cmd + STAND-first benign initial condition; empirical, validated at the milestone.
- **Cold history transient at stand→walk** (history replicated ×5, last_actions=0) → mitigated by entering walk from a stable stance; expect a brief transient, not a fall.
- **Torque-limit clamp degrades from 1 kHz (deploy) to 100 Hz (us)** → applied on the latest Observation each policy step; sim-acceptable; a fidelity note for real.
- **Async non-determinism hurts RL training** → out of scope; training uses a separate in-process vectorized tap to Oli, not this contract.
- **Brain stall mid-stride** (held cmd unsafe) → watchdog fail-safe damping (RC10); minimal in sim, vital on real.

## Migration Plan

1. Close `may-147-isaac-limx-sdk-bridge` with a Conclusion (what was learned: kinproj/identity-relay impossibility; what is reused: protocol + transport + Oli) and `openspec archive` it.
2. Stand up the `brain` Py 3.11 env (onnxruntime/scipy/numpy).
3. Reuse map (see proposal Impact): slim `oli.py`; relocate permutation to `SimComm`; re-home transport (World=server, brain=client); reframe `protocol.py`.
4. Build the brain (contracts → Comm client → Reason/Teleop → Action/PolicyRunner → Orchestrator), then `sim_world_main`, then run the walk milestone.
5. Rollback: the change is additive (new `oli/` package + a slimmed `oli.py`); reverting restores the bridge change from archive.

## Open Questions

- **RealComm 1 kHz republish ownership**: does the py3.8 edge own the 1 kHz cmd republish + live torque clamp (brain sends 100 Hz actions), or does the brain send full cmd packets and the edge just relays? Sim doesn't care; decide when real lands.
- **Joystick source on real**: limxsdk `SensorJoy` over the bus vs a direct device read; both enter at Reasoning, but the source plumbing differs.
- **Head joint order**: `oli.py` PR_ORDER has head_pitch/head_yaw swapped vs the corpus `sdk_joint_order` (numerically harmless today — head params symmetric). Fix systematically later, not in this change.
- **Wire encoding sim vs real**: schema-invariant is committed; byte-identical is not (sim has not done PR↔AB and may carry different terms). Refine the encoding during development.
