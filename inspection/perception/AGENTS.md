# perception — eye-in-hand D405

## Purpose
Everything the wrist camera produces: frame bundles, hand-eye
calibration, and the segmented object primitive that fills the object
slot in `cell/`.

## Files
- `capture.py` — arm pose + D405 frame bundle → disk; owns the on-disk
  bundle format. Stationary captures only.
- `handeye.py` — ChArUco collect/solve for T_flange_cam (IR-left stream;
  D405 color is unrectified, never feed its coefficients to solvePnP).

## Contracts & decisions
- Capture only while stationary — motion corrupts depth by up to 5 cm.
- Output toward `cell/` is a segmented, inflated object primitive, not a
  raw cloud.
- No TSDF / dense reconstruction, by decision: pose error is 3-6 mm
  against ~80 mm objects — fusion would smear, a primitive is honest.

## Does NOT belong here
- Viewpoint selection (view/), collision checking (cell/), planning
  (motion/), loop control (run/).
