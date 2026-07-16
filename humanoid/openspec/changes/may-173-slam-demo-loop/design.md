# Design — slam-demo-loop

Decisions carry the evidence that forced them. Tasks reference these as D1…D9.

## D1 — No privileged information; map frame IS the world frame

The sim is treated as the real world. The demo loop may consume only: camera frames, operator hints (clicks), and artifacts baked from the robot's own recordings. Consequence: the occupancy grid is built from cuVSLAM poses, so its frame is the *map* frame — dev_app displays it, the planner plans in it, the localizer reports in it. There is no world/GT frame anywhere in the loop. GT exists solely as a **hidden validation oracle** (D8). The Isaac-GUI occupancy map is demoted to eval tooling (GT route planning for scripted experiment drives) and is never shipped to the demo.

## D2 — Capture recipe is frozen: 30 Hz, stereo, visual-only

Evidence (2026-07-16 cell experiment, identical-pixel variants from one master recording per cell):

| cell | 5 Hz | 30 Hz |
|---|---|---|
| ep0 | ATE 1.04 m, 3 jumps | ATE 0.023 m, 0 jumps |
| ep1 | ATE 2.32 m, 11 jumps | ATE 0.016 m, 0 jumps |
| ep2 | ATE 0.014 m, 0 jumps | ATE 0.030 m, 0 jumps |

IMU is not consumable by the offline bake toolchain (verified: the greps that suggested otherwise matched "max**imu**m"); NVIDIA's IMU story is runtime-only bridging. IMU rows stay in dumps as a reserve. No dressing, no cuSFM (eliminated 2026-07-15: full-drive PGO made ATE 2× worse via aliased matches).

## D3 — Localizer is its own process in its own env (PolicyRunner pattern)

PyCuVSLAM (`bench-cuvslam` env, py3.11, verified: `Tracker.localize_in_map`, `save_map`, `SlamLocalizationSettings` all present) runs as a launcher-supervised sidecar. It consumes `CameraStreamReader` (the camera channel was designed for exactly this consumer class) and publishes pose + state on its own channel. The brain never imports `cuvslam`; Comm stays the world edge; the invariance boundary is untouched. Crash isolation: a localizer death degrades to LOST, never takes down the stack.

## D4 — Two cuVSLAM modes, both exercised

- **Tracking (VO/SLAM)**: survives motion — proven offline by D2.
- **Map-relative (`localize_in_map`)**: recognizes place in a prior map — the demo mode, untested anywhere. Its distinct failure modes (approach direction never mapped, feature-poor localization, wrong-aisle aliasing on bad hint) are exactly what LOC MODE exploration must surface *before* the scripted demo.

## D5 — Map-format bridge tested first; rebuild fallback is equally valid

Unknown: whether the container bake's map loads in PyCuVSLAM `localize_in_map`. This is task #2 — the build's biggest risk retires in the first hour. Fallback (no schedule impact): rebuild runtime maps via PyCuVSLAM `save_map` from the **same master dumps** — identical pixels, and the rebuild is audited with the same `map_audit.py` gate before use. Container bake remains the production P2 path either way.

## D6 — Hint semantics: operator click, not GT read

`LocalizeInMap` requires a pose hint (corpus: no unlocalized loading — nvidia-corpus://cuvslam-api/cpp/a00138#intro). Demo: the operator clicks "you are here" on the occupancy map — a human hint, as in real deployments. Validation rig only: current GT pose as hint (faster iteration). cuVGL kidnap (hint from BoW retrieval) is the stretch tier, container-side.

## D7 — Estimated pose closes the control loop

The follower steers on the localizer's pose, not GT. This is deliberately the hard part: localization error feeds control, which changes viewpoint, which changes localization. The path: goal click → plan on SLAM occupancy → brain-side follower emits GLIDE_CMD (already in the World spine) from est pose. LOC MODE teleop (human as follower) de-risks this loop before autonomy does it.

## D8 — GT is a display-only oracle with an offline-computed transform

To draw the GT ghost and |est−GT| readout, GT (sim frame) must map into the map frame. `registration.json` per map — Umeyama SE(2) from bake keyframes vs recorded GT (the `map_audit.py` machinery) — computed at bake time, consumed **only** by dev_app validation overlays and the session-log autopsy. Nothing in the demo loop reads it.

## D9 — Writer surgery makes P1 honest at scale

Current recorder: 4 cams × sync PNG ≈ 1–2.4 s/frame — cell-scale only. Surgery: stereo-only option (bake consumes left/right only), JPEG q95 (cuVSLAM grayscale-internal; q95 transparent), async bounded-queue writer thread (sim step never blocks on disk). `rosbag_synth` learns `.jpg`. All flag-gated; existing cell dumps and tests stay valid.
