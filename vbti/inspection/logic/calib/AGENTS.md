# calib — registration & calibration artifacts

## Purpose
Scripts and data that pin frames to reality. Run rarely, versioned
forever: the numbers here are load-bearing for everything downstream.

## Files
Scripts (CLIs, run by hand; all but `plane_fit`/`probe_fit` talk to the robot or camera):
- `tap_probe.py`: freedrive tap logger: press "s", print the TCP point (UR `base` frame).
- `probe_fit.py`: probed touch points → base→table transform (`pose:` line for `cell.yaml`).
- `plane_fit.py`: corner taps → oriented safety planes (`planes:` snippet for `cell.yaml`).
- `frame_check.py`: read-only check that the model's FK agrees with the controller
  (flange→TCP vector must be constant across poses). Never moves the arm.
- `depth_tare.py`: D405 Tare depth calibration over the ChArUco board; `apply` writes
  the camera, `restore` undoes it. Outputs under `d405_tare/<MMDD-HHMM>/`.
- `depth_val.py`: depth error on a recorded run: one yellow dot, truth from ray
  intersection (no depth), per-view depth error + table-plane check.
- `depth_eval.py`: end-to-end depth eval: survey + fixed cells, then `depth_val`, compare to baseline.

Data (dated, committed):
- `T_flange_cam_2026-08-19.npy` (+ `.yaml` provenance): hand-eye result (solver: `logic/perception/handeye.py`).
- `table_probe_2026-08-18.yaml`: table touch points (input to `probe_fit.py`).
- `planes_2026-08-31.yaml`: safety-plane tap session (input to `plane_fit.py`).
- `d405_tare/`: Tare candidates and camera calibration backups.

## Contracts & decisions
- Data files are source, not build products: commit them, never overwrite;
  a new calibration is a new dated file.
- `logic/cell/geometry.py:load_T_flange_cam` loads the newest `T_flange_cam_*.npy`
  at runtime (sorted by name). A run records the resolved file + sha256
  (`logic/record/schema.py`). Everything else feeds `logic/cell/cell.yaml` by hand-pasted numbers.
- Frames: pendant/`tap_probe` readings are the UR controller `base` frame;
  the URDF `base_link` is `base` rotated pi about Z (`frame_check.py` exists
  because this hid a 180 deg error).

## Does NOT belong here
- Runtime perception (`logic/perception/`), the world model itself (`logic/cell/`),
  code imported by the live pipeline.
