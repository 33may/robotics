## Why

We are building the **Oli runtime**: a deployment-invariant control system for the HU_D04_01 humanoid. Reasoning and Action logic is authored **once** and runs **unchanged** whether Oli's body is the Isaac simulation or the physical robot — only a thin Communication edge swaps underneath. This is the precondition for the end-of-summer-2026 reasoning demo plus the parallel acting research, both of which must run on sim and real from the same codebase.

The prior approach (make Isaac a peer on LimX's MROS bus — `may-147-isaac-limx-sdk-bridge`) deadlocked: a pure-Python identity relay is impossible, and `kinematic_projection` cannot be reproduced in the limxsdk Python wheel (proven 2026-06-22). The pivot is to stop conforming to LimX's bus and instead define **our own** schema-invariant contract between an independent World process and a Reason+Action brain. The hard `limxsdk` Py 3.8 ABI vs modern-brain Py 3.11+ (SLAM/VLA/CUDA) split forces a process boundary on the real path; modeling the **sim** World as an independent process too gives us sim↔real interface parity, so behavior validated in sim transfers to hardware with no logic change.

## What Changes

- **NEW — the Oli runtime**, a two-process architecture over one schema-invariant contract:
  - **World** (independent process): Sim = Isaac + Oli articulation; Real = robot. Free-runs at 1 kHz, authoritative body.
  - **Communication**: the only world-aware layer (joint reorder, parallel↔serial mechanism, unit/quat conventions). `SimComm` (py3.11 → Isaac) now; `RealComm` (py3.8 limxsdk → robot) deferred.
  - **Reasoning** (brain): `Observation` + joystick → `PolicyIn`. Foundation = Teleop + JoystickAdapter.
  - **Action / PolicyRunner** (brain): all policy logic — obs encoding, ONNX, resolution to a PD command. STAND (analytic) + WALK (ONNX) modes.
  - **Orchestrator/Runtime** (brain): conducts the brain loop, paces by world-stamp, wires concrete impls, logs/records every contract, watchdog.
- **Three invariant contracts** in canonical PR space: `Observation` (≅ RobotState+ImuData), `PolicyIn`, `PolicyOut` (≅ RobotCmd).
- **NEW dependency**: a dedicated `brain` Py 3.11 conda env (onnxruntime, scipy, numpy; later torch/VLA). The brain imports neither `isaacsim` nor `limxsdk` — that is the enforceable invariant boundary.
- **REUSE** from the bridge change: `bridge/protocol.py` (the wire), `OliBridge`/`sidecar` transport plumbing (re-homed: World=server, Brain=client), and the entire `oli.py` Isaac articulation + implicit-PD drive.
- **REMOVE / archive (BREAKING vs the bridge change)**: Isaac-as-MROS-sim-peer framing, `kinematic_projection` relay path, `load_oli.py --bridge` deploy-driving mode, and the bridge-validation smoke scripts. `may-147-oli-deployment-interface` **supersedes** `may-147-isaac-limx-sdk-bridge`.
- **MILESTONE**: Oli stands then **walks** in the Isaac viewport, driven by the LimX walk ONNX, steered by a joystick command through Teleop.

## Capabilities

### New Capabilities

- `oli-deployment-interface`: A deployment-invariant humanoid runtime. Defines the three canonical-PR contracts (Observation/PolicyIn/PolicyOut), the World↔Brain two-process model and its async free-running run cycle (latest-wins, world-stamp pacing, freeze-until-cmd, watchdog), the responsibilities and boundaries of each component (World, Communication, Reasoning, Action, Orchestrator), and the externally observable behavior: the same brain binary drives Isaac and the robot, and the walk ONNX makes Oli walk in Isaac through this interface.

### Modified Capabilities

- None. `openspec/specs/` has no deployed capabilities yet; this is the first.

## Impact

- **New code** under `humanoid/logic/oli/` (invariant brain — imports no `isaacsim`/`limxsdk`): `contracts.py`, `comm/` (base + protocol + client), `reason/` (teleop), `action/` (policy_runner), `runtime.py`, `brain_main.py`.
- **Modified code** under `humanoid/logic/simulation/isaacsim/`: `oli.py` slimmed to a pure articulation (PR↔Isaac permutation relocates to `SimComm`; in-class bridge I/O stripped); `sim_comm.py` (new server) + `sim_world_main.py` (from `load_oli.py`).
- **Deferred** (interface/stub only): `RealComm` py3.8 edge (repurposed from `sidecar.py`, role flips sim-peer → real-robot).
- **New env**: `brain` (Py 3.11). `isaac` env reused for the World; `limx` env reused for the deferred RealComm.
- **Reuses** the LimX walk `policy.onnx` artifact verbatim and the vendor `walk_controller.py`/`stand_controller.py` as the authoritative obs/action reference.
- **Predecessor**: `may-147-isaac-limx-sdk-bridge` (closed with a conclusion + archived; its built transport + articulation are reused, not discarded).
- Linear: [MAY-147](https://linear.app/may33/issue/MAY-147). Parent: MAY-141 (Oli walking in simulation).
