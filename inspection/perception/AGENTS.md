# perception — eye-in-hand D405

## Purpose
Everything the wrist camera produces: frame bundles, hand-eye
calibration, and the segmented object primitive that fills the object
slot in `cell/`.

## Files
- `camera.py` — D405 pipeline layer (open/metadata/bundle/live viewer),
  migrated from vbti.logic.cameras.d405_still; owns the UR wrist serial and
  `t_flange_cam()` — loads the latest dated calib/ artifact (OpenCV camera
  convention: +Z boresight, left-eye frame).
- `capture.py` — UR5e pose + D405 frame bundle → disk; owns the on-disk
  bundle format. Stationary captures only. Pose = RTDE joints → UR5eIK
  flange FK; meta carries joints_rad, T_base_flange, T_base_cam.
- `handeye.py` — ChArUco collect/solve for T_flange_cam on the UR5e
  (IR-left stream; D405 color is unrectified, never feed its coefficients
  to solvePnP). Collect is hands-free: freedrive + stationarity gate
  (RTDE qd) + novelty gate + accept/reject beeps. FK = UR5eIK.fk (flange,
  nominal DH); raw joints saved per pair for later re-FK.

## Contracts & decisions
- Capture only while stationary — motion corrupts depth by up to 5 cm.
- Output toward `cell/` is a segmented, inflated object primitive, not a
  raw cloud.
- No TSDF / dense reconstruction, by decision: pose error is 3-6 mm
  against ~80 mm objects — fusion would smear, a primitive is honest.

## Does NOT belong here
- Viewpoint selection (view/), collision checking (cell/), planning
  (motion/), loop control (run/).
