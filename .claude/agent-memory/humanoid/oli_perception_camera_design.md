---
name: oli-perception-camera-design
description: MAY-149 simulated cameras — the design decisions and gotchas behind the `may-149-isaac-oli-cameras` OpenSpec change (bake into USD sensor layer, CameraFrame contract, dedicated frame channel).
metadata:
  type: project
---

MAY-149 = simulate Oli's two RealSense **D435i** RGBD cameras (chest + head) in Isaac. Shaped
2026-07-01 with Anton as OpenSpec change `may-149-isaac-oli-cameras`, new capability
`oli-perception` (validated `--strict`). It is NOT the arm-project hardware camera work — those
memories (`camera_udev_setup`, `camera_viewer_preset_fix`) are a different robot.

**Why:** block-2 (semantic env / SLAM / reconstruction, MAY-170/171/173) needs RGBD in sim.

**Key decisions (full rationale in the change's design.md, D1–D9):**
- **Bake cameras into the robot USD**, not runtime-attach — mounts are a fixed robot fact (Manual
  §1.4.1). Home = the **sensor layer** `HU_D04_01_sensor.usd` (already holds the IMU); USD is
  `PXR-USDC` binary so author via a committed idempotent `build_camera_usd.py` (mirrors
  `build_rl_usd.py`), on the project copy under `assets/oli/usd/`, vendor pristine.
- **GOTCHA — there is no "chest" link.** Oli's torso mass is `waist_pitch_link` (11.9 kg). Chest cam
  parents to `waist_pitch_link` (35° down), head cam to `head_pitch_link` (horizontal). Manual poses
  are in `base_link`; convert to parent-local at nominal pose from URDF/MJCF offsets.
- **`CameraFrame`** = 4th invariant contract (stamp, name, rgb, depth, intrinsics). **Intrinsics
  only, NO extrinsics** — camera pose is derivable brain-side by FK from `Observation` + static
  mount table (identical sim vs real, preserves invariance).
- **Dedicated frame transport:** a 2nd `AF_UNIX` **SOCK_STREAM** socket, separate from the 1 kHz
  `SEQPACKET` control channel (a 720p frame ≈2.8 MB ≫ SEQPACKET datagram cap; 30 Hz vs 1 kHz).
  Latest-wins read; never stalls control. Shared-memory is the deferred escape hatch if bandwidth bites.
- **Render** via `isaacsim.sensors.camera.Camera` (Isaac Sim, NOT IsaacLab CameraCfg — ticket was
  wrong); RGB + `distance_to_image_plane` depth (planar Z, m). Default 1280×720 @ 30 Hz (matches real
  `realsense_mros.yaml`), config knob to downscale. Camera cadence decoupled from the control loop.
- **`RealComm` camera edge deferred** to a stub that locks the contract.

Specs: `oli-corpus://user-manual#1.4.1` (mounts), `…realsense_mros/realsense_mros.yaml` (res/rate).
Communication is the only camera-aware layer (SimComm reads Isaac cams → CameraFrame). Relates to
[[project_invariant_oli_interface]], [[isaac_oli_smoke_loader]], [[feedback_tests_in_repo_tdd]].

**BUILT + Isaac-verified 2026-07-02.** Full stack TDD-green (~55 camera tests) and proven end-to-end
from a real Isaac World (`frame_smoke.py`: chest+head both 1280×720, sim-time stamps, finite depth).
Module map:
- `logic/oli/camera_mounts.py` — shared mount table + `rgb_intrinsics` (pure, both envs).
- `logic/oli/contracts.py` — `CameraFrame` / `CameraIntrinsics`.
- `logic/oli/comm/frame_protocol.py` — wire framing; `frame_name(header)` peeks the stream name.
- `logic/oli/comm/codec.py` — `encode/decode_camera_frame` (depth → uint16 mm on the wire).
- `logic/oli/comm/frame_channel.py` — `FrameChannelServer` (per-NAME mailbox) / `Client`; shared
  `recv_frame(conn)` reassembly helper.
- `logic/oli/comm/camera_stream.py` — **`CameraStreamReader`**: the multi-stream CONSUMER. Owns its
  receiver thread, demuxes by camera name into per-stream latest-wins slots; `read(name)` →
  `contracts.CameraFrame`, non-blocking + non-consuming. The dev app's IsaacCameraSource wraps THIS,
  NOT the raw `FrameChannelClient.read_latest()` (single-slot, clobbers with 2 cams).
- `logic/oli/comm/camera_publisher.py` — **`CameraPublisher`**: World-side, engine-agnostic (duck-typed
  Oli), `publish(tick, stamp_ns)` reads every cam → CameraFrame → serves on the render sub-tick.
- `logic/simulation/isaacsim/oli.py` — `Oli(cameras=True, camera_resolution=…)`, `_attach_cameras`,
  `read_camera_rgbd(name)`, `camera_intrinsics(name)`.
- `logic/simulation/isaacsim/sim_world_main.py` — `--cameras/--camera-socket/--camera-res`; publishes
  inside `_actuate_and_step` ONLY on render steps; 8-tick camera warmup before serve.
- `logic/oli/frame_smoke.py` — brain-side §8.3 consumer (paced by `brain_main.py --mode stand`).
- Default frame socket `/tmp/oli-world-frames.sock`; stream names `["chest","head"]` from `CAMERAS`.

**KEY REFINEMENT (single-slot clobber):** the original frame mailbox was single-slot — publishing 2
cameras a tick clobbered one. Fix: **per-camera-name latest-wins on BOTH ends** — server keeps a
`{name: bytes}` mailbox (peeks the wire name), consumer demuxes by name. See
[[isaac_camera_first_render_not_ready]] for the crash-on-first-read gotcha. Glide (MAY-172) hosts the
same `CameraPublisher` behind `--cameras`. NOT committed yet (2026-07-02) — all 3 flows commit together.
