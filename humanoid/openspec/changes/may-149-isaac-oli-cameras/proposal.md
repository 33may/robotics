## Why

Block-2 (semantic environment & navigation foundation, [MAY-170](https://linear.app/may33/issue/MAY-170)) needs visual input in the sim: 3D reconstruction, SLAM, and the semantic map all consume RGBD frames. Oli's real perception surface is two Intel RealSense **D435i** RGBD cameras — one on the chest, one on the head (`oli-corpus://user-manual#1.4`). Isaac has no cameras today, so no perception code can be developed against the sim.

The cameras must reach the brain through **Communication**, exactly as proprioception does — so the deployment-invariant guarantee holds: the same brain consumes one `CameraFrame` contract whether frames come from Isaac (`SimComm`) or the real RealSense stack (`RealComm`, deferred). Building this now, alongside the cameras, keeps the sensor surface and its transport co-designed instead of bolted on later.

## What Changes

- **NEW — two cameras baked into the robot USD sensor layer.** Camera prims are authored into `HU_D04_01_sensor.usd` (the composition layer that already holds the IMU), parented to the moving links so they track the body: chest cam on `waist_pitch_link` (35° pitched down), head cam on `head_pitch_link` (horizontal). Mount poses from Manual §1.4.1 (given in `base_link`, converted to each parent's local frame at nominal pose); intrinsics match D435i (RGB ≈69°×42° FOV). Authored by a committed **build script** (mirrors the existing `build_rl_usd.py`), not by hand-editing the binary crate.
- **NEW — the World renders both cameras.** RGB + depth per camera at a configurable resolution/rate (default **1280×720 @ 30 Hz**, matching the real `realsense_mros.yaml`), exposed in-process as numpy arrays.
- **NEW — invariant contract `CameraFrame`.** A brain-facing snapshot (stamp, camera name, RGB, depth, intrinsics) in a world-order-free form — a fourth contract alongside `Observation`/`PolicyIn`/`PolicyOut`.
- **NEW — a dedicated frame-streaming channel.** A second `AF_UNIX` **`SOCK_STREAM`** socket carries frames, separate from the 1 kHz `SEQPACKET` control channel (a single 720p frame is ~2.8 MB — it cannot ride a SEQPACKET datagram). `SimComm` reads the Isaac cameras → `CameraFrame` → stream; `BrainComm` gains a latest-wins camera read.
- **NEW — smoke test.** Render each camera in a scene, save RGB + depth frames, verify pose and FOV.
- **DEFERRED (interface/stub only)** — `RealComm` camera edge (real RealSense → `CameraFrame`). No hardware yet; the stub locks the contract so the real path drops in unchanged.

## Capabilities

### New Capabilities

- `oli-perception`: Oli's visual sensor surface. Defines the two robot-mounted RGBD cameras (mounts baked into the robot USD, D435i intrinsics, RGB+depth render in the World), the invariant `CameraFrame` contract, and the dedicated World→brain frame-streaming channel. Externally observable behavior: the same brain consumes camera frames identically whether the World is Isaac (`SimComm`) or the real robot (`RealComm`, deferred), and the frame channel never stalls the control loop.

### Modified Capabilities

- None. `oli-perception` is additive; it extends the sensor surface of the `oli-deployment-interface` change (not yet in `openspec/specs/`) without altering its three existing contracts.

## Impact

- **Robot USD**: cameras authored into `assets/oli/usd/configuration/HU_D04_01_sensor.usd` by a new `logic/simulation/isaacsim/build_camera_usd.py`; the `_rl` variant sensor layer gets the same treatment.
- **Contracts** (`logic/oli/contracts.py`): add `CameraFrame` (pure; no numpy dtype coupling beyond arrays).
- **Communication** (`logic/oli/comm/`): a frame wire (extend `protocol.py` or new `frame_protocol.py`) for length-prefixed image messages; `codec.py` gains `CameraFrame`↔wire; `client.py` (`BrainComm`) gains a camera-stream read; `base.py` grows the camera read method.
- **World** (`logic/simulation/isaacsim/`): `sim_comm.py` (`SimComm`) reads the Isaac `Camera` sensors and publishes `CameraFrame`; `sim_world_main.py` opens the frame channel and paces camera reads at the render rate (decoupled from the 1 kHz control loop). `oli.py` stays a pure articulation — cameras live in the USD, read via the sensor API.
- **Deferred**: `RealComm` camera path (py3.8) — stub only.
- **Env**: `isaac` (render, via `isaacsim.sensors.camera.Camera` — Isaac Sim, not IsaacLab); `brain` (consume). No new dependencies (numpy already present).
- **Linear**: [MAY-149](https://linear.app/may33/issue/MAY-149). Parent MAY-143. Unblocks block-2 perception work (MAY-171/173).
