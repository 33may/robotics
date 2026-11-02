# Tasks — slam-demo-loop

Authored incrementally (module-by-module rule, 2026-07-16): only the active
phase carries checkboxes; later phases are headers that get detailed when we
reach them. Design refs D1–D9 in `design.md`.

## 1. Phase 1 — teleop recording (Robot-side sink, D9)

Deliverable: teleop freely in the launcher stack while a Robot-side recorder
process persists a bake-grade 30 Hz dump; dev_app RecordPanel starts/stops and
monitors it; recording survives dev_app crash. JPEG-vs-PNG verdict lands from
the parallel agent experiment; the writer supports both (codec slot).

- [x] 1.1 Stereo streams: `head_left`/`head_right` on the camera channel — `CameraPublisher` + `Oli(stereo_cameras=…)` wiring + glide_world flag (TDD on publisher/codec suite)
- [x] 1.2 `DriveRecorder` async mode: bounded RAM queue + encoder worker pool; codec param `png|jpeg` (q95); depth stays uint16 PNG; line-flushed jsonl (crash-safe tail); backpressure = block, never drop (TDD, pure) — measured on a real 720p warehouse frame: PNG 239 ms vs JPEG q95 2 ms; async end-to-end 178 frame-sets/s (bar: 30). JPEG q95 proven cuVSLAM-safe by cell test (ep0 0.023→0.029 m, ep1 0.016→0.015 m, 0 jumps, 3.8× smaller)
- [ ] 1.3 `rosbag_synth` accepts `.jpg` frames alongside `.png` (TDD)
- [x] 1.4 `recorder_main.py` (Robot side): drains camera channel + GT debug-pose channel → neutral dump; `recording/fk.py` (base∘mount FK, constant-offset-safe via mount recovery — 8 tests) + `recording/session.py` (dedupe, stamp↔pose join, gap accounting — 10 tests) + process integration test (SIGTERM = graceful save); status.json heartbeat, idle-timeout finalize; recorder BINDS its own pose path `/tmp/oli-record-pose.sock` (glide_world `--debug-pose` now repeatable — nav map + recorder can't share a bound datagram socket)
- [x] 1.5 dev_app `RecordPanel` (ProcessLauncher pattern): out-dir/codec controls, Start/Stop+save, status.json monitor (frames/stamps/gaps/skipped), teardown = graceful stop; registered when a camera socket is present; launcher grew `--stereo-cameras` (dry-run verified boot plan)
- [x] 1.6 Acceptance: live stack (fixed-vx glide, full warehouse scene) + recorder end-to-end → 746 stamps @ 30.0 Hz sim (median dt 33.0 ms), 0 gaps, graceful idle-timeout save → bag → container cuVSLAM → **ATE 0.0105 m, 0 jumps, maxstep 3 cm** (best of the day; FK poses bake-grade). Found + fixed en route: glide World tick-gated cameras ran ~5.6 Hz sim → `--camera-hz` sim-time gate; locbench had the same bug (`camera_every: 16` ≈ 7 Hz sim — pre-16-07 bench scores were rate-handicapped). Human-joystick GUI session = the actual demo-zone walk (usage, not build)
- [x] 1.7 JPEG verdict landed → recorder_main defaults to jpeg q95 (cell test: ep0 0.023→0.029 m, ep1 0.016→0.015 m, 0 jumps, 3.8× smaller; DriveRecorder default stays png for scripted-dump compat)

## 2. Phase 2 — offline artifact build (bake + occupancy-from-depth, D1)

Deliverable: from ONE teleop dump, every artifact Phase 3 consumes — cuVSLAM
keyframe map + cuVGL BoW + a robot-derived occupancy grid — via a single
reusable flow with a clear in/out contract. No GT anywhere in the build path.

- [x] 2.1 cuVGL BoW map on `teleop_v1_demo` bake via the existing container flow (`--steps_to_run cuvgl`, 135 s → `cuvgl_map/` bow_index.pb 16 MB + vocabulary + keyframes 1.7 G). Container `depth`+`occupancy` steps skipped deliberately: they need a FoundationStereo TRT engine we don't have — and we recorded real sensor depth. Est-vs-GT plot (`data/maps/teleop_v1_demo_est_vs_gt.png`): mean 16 cm / p95 25.6 / max 27 over 83 m, 0 jumps; pure drift-stairs, no relocalization on revisit (the exact gap Phase 3 LocalizeInMap closes)
- [x] 2.2 Occupancy via the VENDOR pipeline (pivot, Anton 16-07: "check what nvidia propose"): `nvblox_inject.py` (TDD, 5 convention tests) maps recorded sensor depth onto the bake's keyframes — head depth → `depth/head_left/` (2.5 cm offset, sub-cell) + chest (35° down, near-field) as NEW keyframes composed from the bake's OWN head_left poses ∘ static FK offset; runtime convention self-check against the meta's `sensor_to_vehicle`. Then container `--steps_to_run occupancy` (nvblox fuse_cusfm, 30 s) → ROS trinary PNG+YAML → existing `occupancy_io.convert_ros_map` (**unknown = blocked**, Anton 16-07). Hand-rolled projector (`occupancy_from_depth.py`) was the dead end that taught the conventions: bake base_link is yaw-flipped; TUM poses compose badly with external extrinsics — frames_meta poses don't
- [x] 2.3 Verified on `teleop_v1_demo`: 680×225 @ 5 cm; driven path blocked 3/2268 (0.1%) raw, 1.7% @ 0.25 m inflation, 2.5% @ 0.35 m (only tightest shelf-corner turns); artifact loads via `load_occupancy`, world (0,0) free
- [ ] 2.4 One-script offline flow (`bake_map`): dump → bag synth → edex+compute_poses → cuvgl → occupancy_from_depth → audit; documented IN (dump dir) / OUT (map dir Phase 3 consumes)

## 3. Phase 3 — deployment: live localizer + LOC MODE (D3, D4, D6)

(to be detailed at Phase 2 boundary)

## 4. Phase 3 — autonomy: click-goal → plan → follow on est pose (D7)

(to be detailed)

## 5. E2E zone demo + validation instruments (D8)

(to be detailed)
