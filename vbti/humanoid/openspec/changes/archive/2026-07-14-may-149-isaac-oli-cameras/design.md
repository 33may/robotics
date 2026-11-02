# Design — Oli simulated cameras (oli-perception)

## Context

Oli's real perception is two Intel RealSense **D435i** RGBD cameras: chest and head (`oli-corpus://user-manual#1.4`, real config `oli-corpus://oli-main-2.2.12#install/etc/realsense_mros/realsense_mros.yaml`, 1280×720 @ 30 Hz color+depth). The Isaac World has none. Block-2 perception (3D reconstruction, SLAM, semantic map — MAY-170/171/173) needs these streams in sim.

Current state to build on:
- The two-process runtime (`may-147-oli-deployment-interface`): **World** (Isaac, py3.11) is the authoritative body and the connection **server**; the **brain** (py3.11, imports no `isaacsim`/`limxsdk`) is the client. Today they exchange three canonical-PR contracts over one `AF_UNIX` **`SEQPACKET`** socket with tiny fixed-size struct payloads (`Observation` = 428 B).
- The robot USD is a layered binary crate: `HU_D04_01.usd → configuration/HU_D04_01_sensor.usd → …_physics.usd → …_base.usd`. The **sensor layer** already carries the IMU.
- `oli.py` is a pure articulation; `SimComm` (server) owns the Isaac↔PR permutation and body I/O; `build_rl_usd.py` is the precedent for authoring a USD variant from code.

Constraints: brain stays SDK-free (invariance boundary); World defaults must not regress the walk/stand path; vendor USD layers stay pristine (we edit the project asset copy under `assets/oli/usd/`).

## Goals / Non-Goals

**Goals:**
- Two RGBD cameras mounted on Oli's moving links in the Isaac World, faithful to the manual's poses and D435i intrinsics.
- A single invariant `CameraFrame` contract the brain consumes identically in sim and (future) real.
- A frame transport that never stalls the 1 kHz control loop.
- A smoke test proving pose + FOV + RGBD render.

**Non-Goals:**
- No `RealComm` camera implementation (stub/interface only — no hardware).
- No perception/SLAM/reconstruction logic — this only produces frames for those consumers.
- No policy change; cameras feed perception, not the walk ONNX.
- No IsaacLab (`sensors.CameraCfg`) — this project is raw Isaac Sim.
- No camera IMU (D435i's built-in IMU is irrelevant; Oli's policy IMU is separate).

## Decisions

### D1 — Cameras are robot embodiment, baked into the USD sensor layer
The mounts are a fixed physical fact of the robot (Manual §1.4.1), so they belong with the asset, not with scene-assembly code. We author the two `Camera` prims into `HU_D04_01_sensor.usd` — the same composition layer that already holds the IMU. *Alternatives:* (a) attach cameras at runtime in Python — rejected: mounts scatter into code and drift from the asset; (b) free world-pose props — rejected: they wouldn't track body motion (the whole point). The World still *owns reading* the cameras; the USD only fixes where they are.

### D2 — Parent links: chest → `waist_pitch_link`, head → `head_pitch_link`
There is **no "chest" link** — Oli's torso mass lives in `waist_pitch_link` (11.9 kg). The chest cam parents there; the head cam parents to `head_pitch_link` (top of the neck chain). The manual gives poses in `base_link`; we convert to each parent's local frame at nominal (zero) joint pose using the URDF/MJCF offsets:
- chest local ≈ `[0.092, 0.0175, 0.4336 − 0.15939]` under `waist_pitch_link`, pitched 35° down;
- head local ≈ `[0.0615−(−0.013), 0.0175, 0.652 − 0.58729]` under `head_pitch_link`, horizontal.
Exact offsets are recomputed from the description in the build script, not hard-copied.

### D3 — Author via a committed, idempotent build script
The USD is `PXR-USDC` binary, so we edit through the `pxr.Usd` API in a script (`build_camera_usd.py`), mirroring `build_rl_usd.py`. Re-runnable and diff-friendly (**the tracked script is the source of truth, not the binary**). Note: `assets/oli/usd/` is a **symlink into the vendor submodule** — it *is* the vendor `usd/` dir, the established asset space (`build_rl_usd.py` already generates there). We treat that vendor tree as our own working repo; the USDs are untracked, regenerable build artifacts, so re-running the script restores the cameras after any submodule reset. Applies to both the default and `_rl` sensor layers.

### D4 — Isaac Sim `Camera` sensor, RGB + depth annotators
Use `isaacsim.sensors.camera.Camera` (Isaac Sim, not Lab). RGB via the `rgb` annotator; depth via `distance_to_image_plane` (planar Z in meters — matches RealSense depth convention, not radial distance). Intrinsics set from D435i nominal FOV (RGB ≈69°×42°) at the configured resolution.

### D5 — `CameraFrame`: minimal invariant contract, intrinsics only
`CameraFrame(stamp_ns, name, rgb[H,W,3] uint8, depth[H,W], intrinsics{fx,fy,cx,cy,width,height})`, world-order-free and unscaled. **Extrinsics are NOT in the contract** — the camera pose is derivable brain-side by forward kinematics from the `Observation` joint states plus the static mount table, and that derivation is identical sim vs real. Keeping pose out of the frame preserves invariance (the real robot has no ground-truth camera pose either) and keeps the payload lean.

### D6 — Dedicated `SOCK_STREAM` frame channel, separate from control
Frames get their own `AF_UNIX` **`SOCK_STREAM`** socket, distinct from the `SEQPACKET` control channel. Rationale: a 720p frame is ~2.8 MB — orders of magnitude past a SEQPACKET datagram — and arrives at ~30 Hz vs control at 1 kHz. Wire framing: a small fixed header (magic/version/msg-type/seq) + per-array length-prefixed blocks (shape, dtype tag, raw bytes). The brain read is **latest-wins** (drain to newest, non-blocking) so a slow consumer never backs up the World. *Alternatives:* chunk over SEQPACKET (ugly manual reassembly — rejected); shared memory (`/dev/shm`, zero-copy — deferred as the escape hatch if bandwidth bites, D-risk below). The `payload_len` field already reserved in `protocol.py` "for STREAM future" anticipated this.

### D7 — Camera cadence decoupled from the 1 kHz control loop
Rendering is expensive; the World renders/publishes cameras on a ~30 Hz sub-tick, independent of the 1 kHz physics/control step and the SEQPACKET Observation stream. Frame publishing must never block the control step.

### D8 — Resolution/rate configurable, default 1280×720 @ 30 Hz
Default matches the real `realsense_mros.yaml`. A CLI/config knob drops resolution (e.g. 640×360) or rate if the sim frame-rate suffers. The contract carries whatever is rendered; the brain cannot request more than the World produces.

### D9 — `RealComm` camera edge deferred to a stub
Interface/stub only, no hardware. The stub exists to lock the `CameraFrame` contract so the real RealSense→`CameraFrame` path drops in later without touching the brain.

### D10 — The mount table is one World-agnostic constant
The static per-camera mounts (base_link→camera transform + D435i intrinsics) live as a single pure constant in `logic/oli/` — importable in both the brain and the `isaac` env (like walkmatch `spec.py`). Both `build_camera_usd.py` (Isaac) and the brain's FK-based extrinsics derivation (D5) read it; a future MuJoCo or real edge reads the same. Mounts are never re-typed per World — that shared source is exactly what makes the FK-derived camera pose match ground truth identically across sim and real. The USD/MJCF/physical mount is the per-World *rendering placement*; the mount table is the *invariant truth* behind it.

## Risks / Trade-offs

- **Rendering tanks sim throughput** → decouple camera cadence (D7); configurable resolution (D8); render headless where possible; publish only when a consumer is connected.
- **Depth semantics mismatch** (Isaac radial vs planar; units) → use `distance_to_image_plane` (planar Z, meters); assert range in the smoke test against a known scene.
- **Mount transform error** (base_link → parent-local) → recompute offsets from the description in the build script; smoke test verifies pose + FOV visually and numerically against a placed target.
- **Bandwidth** ~390 MB/s at native 720p float32 depth ×2 cams → localhost UDS handles it; mitigations: encode depth as **uint16 millimeters** (halves it, matches RealSense native), downscale (D8), or shared memory (D6 escape hatch).
- **Editing binary USD corrupts the asset** → the build script is idempotent and runs on the project asset copy; vendor layers pristine; regenerate from script on any doubt.
- **Cameras always in the USD add cost even when unused** → gate rendering/publishing behind a World flag so the walk/stand path is unaffected by default.

## Migration Plan

Purely additive. Order: (1) `build_camera_usd.py` authors the sensor-layer cameras; (2) `CameraFrame` in `contracts.py`; (3) frame wire + codec; (4) `SimComm` camera read + World frame channel (flag-gated); (5) `BrainComm` camera read + smoke consumer; (6) `RealComm` stub. Rollback = don't run the build script / don't open the frame channel; nothing in the existing control path changes.

## Open Questions

- **Depth wire encoding**: float32 meters vs uint16 millimeters? Leaning uint16 mm (bandwidth + RealSense parity); final call in the codec task.
- **Exact D435i intrinsics**: nominal FOV-derived fx/fy is fine for sim; pull real calibration only if a consumer needs metric accuracy.
- **Frame-channel gating**: cameras always-publish vs publish-on-subscribe — decide when wiring the World main.
