# Conclusion — Isaac RGBD cameras through Communication: shipped

**Status:** closed 2026-07-14. All 33 tasks complete.

## What it delivered

Gave Isaac a camera surface that reaches the brain through **Communication** as a single
`CameraFrame` contract — the same shape the real RealSense stack will emit — so perception code
(3D reconstruction, SLAM, semantic map) can be developed against the sim without breaking the
deployment-invariance guarantee. Two cameras (chest + head), frames flow World → `SimComm` →
brain exactly like proprioception.

## Why it matters downstream

This is the sensor tap the whole MAY-173 localization arc consumes: locbench's in-brain
localization host is fed by these `CameraFrame`s (W2), and every candidate localizer's
`LocalizationIn` is frame-paced off them. Shipping cameras co-designed with their transport (not
bolted on) is what let the harness treat "what the localizer sees" as identical in sim and on the
robot.

## Deferred

None — 33/33. The real-camera edge (`RealComm`) is out of scope by design (a later change).
