## 1. OpenSpec scaffolding & validation

- [x] 1.1 Author `tasks.md` (this file)
- [x] 1.2 `openspec validate may-149-isaac-oli-cameras --strict` — must report valid
- [x] 1.3 Branch `33may/may-149-oli-cameras` off the current work; capability `oli-perception`

## 2. Shared mount table & mount math (D2, D5, D10)

- [x] 2.1 Define the shared World-agnostic mount table in pure `logic/oli/` (base_link→camera pos+orientation + D435i intrinsics per camera), importable in both the brain and `isaac` envs; consumed by the brain FK and `build_camera_usd.py`
- [x] 2.2 TDD: pure helper converting a `base_link`-frame mount (pos + pitch) → parent link local frame at nominal pose from URDF/MJCF offsets (chest→`waist_pitch_link`, head→`head_pitch_link`)
- [x] 2.3 Unit-test chest ([0.092, 0.0175, 0.4336], 35° down) and head ([0.0615, 0.0175, 0.652], 0°) against hand-computed local transforms

## 3. Bake cameras into the USD sensor layer (D1, D3)

- [x] 3.1 `logic/simulation/isaacsim/build_camera_usd.py`: author two `Camera` prims into `assets/oli/usd/configuration/HU_D04_01_sensor.usd` under the chest/head parents, using the §2 transforms and D435i intrinsics (RGB ≈69°×42° FOV)
- [x] 3.2 Idempotent: re-running yields exactly two camera prims (no duplication); operates on the symlinked vendor USD asset space (treated as our own working repo), regenerable from the tracked build script
- [x] 3.3 Apply the same bake to the `_rl` sensor layer (`HU_D04_01_rl_sensor.usd`)
- [x] 3.4 Load the USD in Isaac; assert both camera prims exist at the expected world poses at nominal joint pose

## 4. World camera render (D4, D7, D8)

- [x] 4.1 Wrap the baked prims with `isaacsim.sensors.camera.Camera`; attach `rgb` + `distance_to_image_plane` (planar Z, meters) annotators — in `oli.py` `_attach_cameras`
- [x] 4.2 Configurable resolution via `Oli(camera_resolution=…)` (default 1280×720); flag-gated `cameras=` so the walk path pays no render cost
- [x] 4.3 Render cameras on the render sub-tick decoupled from the 1 kHz control step: publish only when `world.step(render=True)` (the existing `--render-every` gate) — wired in `sim_world_main._actuate_and_step`
- [x] 4.4 Expose per-camera RGB (uint8 H×W×3) + depth (float32, meters) as in-process numpy — `oli.read_camera_rgbd()`; proven by `camera_smoke.py`

## 5. CameraFrame contract (D5)

- [x] 5.1 TDD: `CameraFrame` dataclass in `logic/oli/contracts.py` (stamp_ns, name, rgb, depth, intrinsics{fx,fy,cx,cy,width,height}); shape/type asserts; extrinsics deliberately excluded — `CameraIntrinsics` re-homed to `contracts.py`
- [x] 5.2 Verify the brain import graph stays pure (no `isaacsim`/`limxsdk`) with `CameraFrame` added

## 6. Frame wire + codec (D6)

- [x] 6.1 TDD: frame wire framing — fixed header (magic/version/type/seq/stamp/name/res/intrinsics/lengths) + RGB + depth payloads; `frame_protocol.py`, round-trip + garbage-reject tests, pure stdlib
- [x] 6.2 Depth wire encoding: **uint16 millimeters** (RealSense-native, halves bandwidth, 0=invalid preserved); implemented in codec, ~1 mm lossy by design
- [x] 6.3 TDD: `codec` `CameraFrame`↔wire round-trip (rgb exact + depth ≤1 mm + intrinsics + name + stamp + depth-zero)

## 7. SimComm publish + World frame channel (D6, D7)

> Transport module `comm/frame_channel.py` (`FrameChannelServer`/`Client`, thread + latest-wins mailbox) DONE + loopback-tested. Render-tick wiring lands in `CameraPublisher` (`comm/camera_publisher.py`) + `sim_world_main.py`. Design refinement (discovered mid-build): a single-slot mailbox CLOBBERS with 2 cameras, so BOTH ends were made **per-camera-name** — server keeps a `{name: bytes}` mailbox (peeks the wire name), and the consumer is the new `CameraStreamReader` (`comm/camera_stream.py`) that demuxes by name into per-stream latest-wins slots. Verified end-to-end in Isaac (`frame_smoke.py`: chest+head both 1280×720, sim-time stamps, finite depth).

- [x] 7.1 `CameraPublisher` reads the Isaac cameras → `CameraFrame` each render tick (engine-agnostic, duck-typed Oli; tolerates a not-ready camera so it never crashes the World)
- [x] 7.2 World opens a second `AF_UNIX` `SOCK_STREAM` frame channel (server), flag-gated (`--cameras`); separate from the SEQPACKET control socket
- [x] 7.3 Frame channel never backs up the World: `publish()` is O(1) latest-wins PER camera name; absent/slow consumer + non-blocking verified on a real UDS (render-tick publish call wired in §4)
- [x] 7.4 (new) Multi-stream demux both ends: `CameraStreamReader` (consumer) + per-name server mailbox — two cameras no longer clobber (`test_camera_stream.py`, `test_camera_publisher.py`)

## 8. BrainComm read + smoke consumer (D6)

- [x] 8.1 `comm/base.py`: `Comm` ABC grows `read_camera_frame()` (concrete default None; latest-wins)
- [x] 8.2 `BrainComm` (client) connects the (opt-in) frame channel; drain-to-newest non-blocking read → decoded `CameraFrame`; loopback test with `SimComm` + `FrameChannelServer`
- [x] 8.3 Minimal brain-side smoke consumer: `logic/oli/frame_smoke.py` connects the frame channel via `CameraStreamReader`, reads N frames/stream, saves RGB + depth — VERIFIED in Isaac (both streams 4/4 @ 1280×720)

## 9. RealComm camera stub (D9)

- [x] 9.1 `RealCameraSource` stub (`logic/simulation/real/real_camera.py`) locks the `CameraFrame` signature (raises `NotImplementedError`, no hardware); test pins deferral

## 10. End-to-end smoke & verification (spec: verifiable pose/FOV)

- [x] 10.1 Smoke test: `camera_smoke.py` (World-side, target cube) + `frame_smoke.py` (brain-side, over the channel) render each camera and save RGB + depth (png/npy)
- [x] 10.2 `camera_smoke.py` asserts each camera's world pose == base∘mount (FK cross-check) within tolerance AND horizontal FOV ≈ 69° (D435i); PASS
- [x] 10.3 Show saved frames inline (Read tool) + cite absolute paths — chest (35° down, near floor) + head (0° fwd, horizon) shown; /tmp/oli_frame_smoke/

## 11. Docs & memory

- [x] 11.1 AI-native doc `logic/oli/comm/CAMERA_STREAMING.md` (producer + transport surface: data flow, entry points, invariants, failure modes, tests); complements the dev app's consumer `devapp/sources/README.md`
- [x] 11.2 Agent memory: updated `oli-perception-camera-design` (build outcome + module map + per-name refinement); new `isaac-camera-first-render-not-ready` gotcha; index updated
- [x] 11.3 Update today's humanoid daily note (draft → approve → write)
