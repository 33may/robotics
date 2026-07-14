# cuvslam — current truth

> Written in phase 2 (Scope), BEFORE building — the hypothesis is the point. Keep this file
> current: it is what phase-1 research reads about this candidate later.

## Approach

cuVSLAM standalone visual(-inertial) odometry via **PyCuVSLAM** (`vendor/pycuvslam`, NVIDIA's
CUDA-accelerated tracker — see `vendor/UPSTREAM.md`). ROS-free Python API, **not** Isaac ROS.
Known-start only: `start()` seeds the tracker from `LocalizationSetup.initial_pose` and fixes
the cuVSLAM-frame → map-frame SE(3) transform once; `step()` runs one odometry update per
`LocalizationIn` (RGBD head/chest frames + optional IMU), transforms into OUR map frame and
flattens SE(3) → SE(2). In stack terms this is a pure **L1** (local odometry) candidate —
no loop closure (L3), no global relocalization (L4), no map anchoring after t=0.

## Hypothesis

**Bring-up PASS, gate FAIL.** Expect the candidate to run end-to-end through `locbench run`
(coverage > 0, no crash) and to FAIL the 0.15 m max-pos gate on long aisles — pure VO drift is
unbounded and nothing re-anchors it. That failure is the point: nothing on the board has
measured a *real* localizer yet (only the GT-replay reference), and today's deliverable is the
**drift-vs-distance curve**, which sizes how much L3/L5 correction the future cuVSLAM(L1) +
RTAB-Map(L2–L5) combo must supply. Do not tune for the gate.

## Expected failure modes

- **Unbounded drift** — pos error grows with distance travelled; max-pos gate breached by aisle end.
- **Feature-poor walls** — warehouse aisles with blank surfaces starve the tracker → DRIFTING/LOST.
- **Motion blur / low frame cadence** — bench camera cadence may under-feed a 30 FPS-happy tracker.
- **Input-mode mismatch** — our frames are RGBD (head/chest); cuVSLAM's preferred modes are
  stereo/RGBD/multicam — a wrong mapping (e.g. depth scale, distortion model) silently wrecks odometry.

## Status

| field | value |
|---|---|
| phase | bring-up PASS (run 20260714-130101: 3/3 arrived, coverage 1.00, no crash) — drift NOT yet measured: est frozen at warm start by a suspected host frame-starvation bug (JOURNAL it-4), awaiting Anton's call |
| best full run | — (smoke only) |
| tier | FAIL (expected pre-tuning; numbers not yet meaningful) |
| map | none (pure VO — no map artifacts; `map_dir` unused) |
