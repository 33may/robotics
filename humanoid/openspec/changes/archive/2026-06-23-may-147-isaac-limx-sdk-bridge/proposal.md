## Why

Isaac Sim is the project's primary 3D environment spine for the humanoid stack, but right now it cannot exchange packets with any LimX policy code or LimX deploy stack because the SDK that defines that wire (the limxsdk wheel) is hard-locked to Python 3.8 — it ships its own `libpython3.8.so.1.0` next to `_robot.so`. Isaac Sim 5.x runs on Python 3.11; no Isaac Sim version supports Python 3.8, and no Python 3.10/3.11 wheel of limxsdk is published. The two cannot share a process.

LimX's own MuJoCo simulator (`humanoid-mujoco-sim`) sidesteps this by being a sim peer on their proprietary MROS bus — it publishes `RobotState`/`ImuData`, subscribes to `RobotCmd`, and any external policy or deploy-python (also Py 3.8) talks to it through the SDK as if it were a real robot. For Isaac to slot into the same ecosystem, Isaac must look like a sim peer too — same packet schema, same rates, same `init("127.0.0.1")` endpoint pattern. Without that, MAY-148 (port standing+walking policy to Isaac) cannot start, and the year's RL/teleop work cannot use Isaac as its physics surface.

This change introduces a two-process bridge that lets Isaac act as a MROS sim peer. A Python 3.8 sidecar in the `limx` conda env owns the `limxsdk.Robot()` connection and handles the bus; Isaac Sim in its Python 3.11 env owns physics; they exchange `RobotCmd`/`RobotState`/`ImuData` over a local IPC channel at ≥1 kHz. From any SDK consumer's perspective the bridge is invisible — they construct `Robot(...)` and `init("127.0.0.1")` as usual.

This is the highest-risk task of the week. It is also the foundation that unlocks every downstream Isaac-on-Oli initiative.

Linear: [MAY-147](https://linear.app/may33/issue/MAY-147/wire-limxsdk-loop-in-isaac-robotstateimu-pub-robotcmd-sub)
Parent: [MAY-141](https://linear.app/may33/issue/MAY-141/oli-walking-in-simulation)
Was blocked by: [MAY-145](https://linear.app/may33/issue/MAY-145/capture-robotcmd-wire-shape-via-sim-role-probe) — resolved 2026-06-22

## What Changes

The capability has two reusable pieces — a robot component and a bus bridge that plugs into it:

- **`Oli` — a reusable Isaac Sim component** (`humanoid/logic/simulation/isaacsim/oli.py`). Single class that owns USD loading + root pinning + articulation init + IMU sensor at `base_link` + (optional) PD actuator law + (optional) MROS bridge wiring. Host apps construct `Oli(world, bridge=...)` once and call `oli.tick()` inside their own physics loop. `bridge=None` is a supported mode — the same class powers RL training, ONNX eval, recon, nav, and SLAM apps that don't talk to the LimX SDK at all.
- **`OliBridge` — the MROS sim-peer plumbing** (`humanoid/logic/simulation/isaacsim/bridge/`). Two factory constructors:
  - `OliBridge.spawn_sidecar(ip=...)` — context-manager that starts the Py 3.8 sidecar process and connects to it (one-line host-app integration).
  - `OliBridge.connect(socket=...)` — attaches to an already-running sidecar (for "keep the sidecar alive across Isaac sessions" workflows).
- **Python 3.8 sidecar** (`bridge/sidecar.py`, runs in `limx` env): constructs `Robot(RobotType.Humanoid, True)`, calls `robot.init(<ip>)`, subscribes `subscribeRobotCmdForSim`, publishes `RobotState` + `ImuData` from incoming IPC frames. Pure bus ↔ IPC relay; no physics.
- **Binary IPC protocol** (`bridge/protocol.py`) — three message types (`HELLO`, `CMD`, `STATE_IMU`) over AF_UNIX SEQPACKET, fixed-size `struct` payloads byte-identical between CPython 3.8 and CPython 3.11.
- **Smoke-test app** (`humanoid/logic/simulation/isaacsim/load_oli.py`) — ~20-line reference example that composes `Oli` + `OliBridge.spawn_sidecar`. Replaces the current 116-line smoke loader.
- **Documentation**: `bridge/README.md` with workflows mapping, `Oli` API reference, and a "Known limitations" section. Cross-links from the two existing vendor docs.

The bridge is end-to-end smoke-tested against LimX's own `humanoid-rl-deploy-python` damping + stand controllers.

### Workflows the design supports

| Workflow | `Oli` | `OliBridge` | n_envs | Sidecar processes |
|---|---|---|---|---|
| MAY-147 — deploy-python controllers vs Isaac | ✓ | ✓ (spawn or connect) | 1 | 1 |
| RL training (future) — policy + sim co-located | partial (use `ArticulationView` for vectorization) | ✗ | 1 → 4096 | 0 |
| ONNX policy eval in Isaac | ✓ | ✗ | 1 | 0 |
| Recon / nav / SLAM apps with Oli in the scene | ✓ | optional (✓ if scripting via SDK, ✗ otherwise) | 1 | 0 or 1 |
| Real-robot deploy (LimX's own stack, unchanged) | ✗ | ✗ | n/a | n/a |

This is the same pattern LimX themselves use for TRON1 (ONNX as the cross-process boundary, SDK only in the deploy process). The bridge is therefore optional plumbing for cases where a Py 3.8 SDK consumer needs to drive Isaac live.

## Capabilities

### New Capabilities

- `isaac-limx-sdk-bridge`: A two-process pattern presenting Isaac Sim as a sim peer on the LimX MROS bus, plus a reusable `Oli` Python component that any Isaac-based host app can drop into its scene. Defines the IPC protocol, joint-permutation contract, PD-actuation law, lifecycle/handshake, the `Oli`/`OliBridge` API surface, and the externally observable behavior (any deploy-python should connect as if Isaac were the MuJoCo sim; any Isaac app should be able to import `Oli` with one line).

## Impact

- **New code** under `humanoid/logic/simulation/isaacsim/`: `oli.py` (the `Oli` class), `bridge/` (protocol, sidecar, `OliBridge` class), reworked `load_oli.py` (smoke demo).
- **No upstream vendor changes** — the `limxsdk` wheel is used as-is; the sidecar is a thin Python wrapper around it.
- **New dependencies**: stdlib only for v1 (`socket`, `struct`, `subprocess`, `signal`, `select`). Optional `mmap`/`multiprocessing.shared_memory` path documented as deferred D-decision if jitter is observed.
- **New conda env dependency**: the existing `limx` env (Py 3.8, has `limxsdk-4.0.1` installed) — already provisioned.
- **Direct consumer**: MAY-148 (port standing+walking policy to Isaac). Without this change, MAY-148 cannot begin.
- **Downstream consumers**: any Isaac-based humanoid initiative — RL training, ONNX policy eval, recon, nav, SLAM. All of them want to drop Oli into their scene with one line; only the live-deploy-controller workflows additionally want the bridge.
- **External ask in parallel**: a separate, asynchronous request to LimX for `limxsdk` wheels built against CPython 3.10 and 3.11 with proper ABI tags. If they deliver, only `OliBridge` and the sidecar are removed; `Oli` itself stays intact and gets a single-process SDK path instead of the IPC path.
- **Reversibility**: high — the bridge surface is one folder + one class. `Oli` is the load-bearing API and is bridge-agnostic by construction.
