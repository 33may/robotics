# perception — eye-in-hand D405

## Purpose
Everything the wrist camera produces: the camera layer, posed frame bundles
on disk, hand-eye calibration, and diagnostic overlays of the identity chain.

## Files
- `camera.py`: D405 pipeline (open, metadata, aligned grab, bundle, live
  viewer); owns `WRIST_SERIAL`. `t_flange_cam()` delegates to
  `cell/geometry.py:load_T_flange_cam` (OpenCV convention: +Z boresight,
  left-eye frame).
- `capture.py`: UR5e pose + D405 bundle → disk; owns the bundle format
  (`save_bundle`, `save_mask`, `read_pose`, CLI `snap`). Pose = RTDE joints →
  `UR5eIK` flange FK; meta carries `joints_rad`, `T_base_flange`, `T_base_cam`.
- `handeye.py`: ChArUco `collect`/`solve` for T_flange_cam on IR-left
  (D405 color is unrectified: never feed its coefficients to solvePnP).
  Collect is hands-free: freedrive + stationarity gate (RTDE qd) + novelty
  gate + accept/reject beeps. Raw joints saved per pair for later re-FK.
- `overlay.py`: `save_chain`: rgb / chain_prompt / chain_mask / chain_kept
  images beside a capture (the UI `chain` tab). Drawing only, no model.

## Contracts & decisions
- Capture only while stationary: motion corrupts depth by up to 5 cm.
- Images on disk are written upright: a half-turned capture is rotated in
  `save_bundle`/`save_chain` (`cell/geometry.py:is_half_turn`, `rotate180`).
- No TSDF / dense reconstruction, by decision: pose error is 3-6 mm
  against ~80 mm objects; fusion would smear.
- Hand-eye FK is the flange (nominal DH), not the pendant TCP (RG2 tip).

## Does NOT belong here
- Segmentation / identity policy (`run/segmenter.py`), cloud geometry
  (`cell/geometry.py`), viewpoint selection (`view/`), planning (`motion/`),
  loop control (`run/`).
