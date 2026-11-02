# `logic/cameras`

## Purpose

`logic/cameras` is the shared camera hardware layer. It exists so viewing, capture, inference, evaluation, and depth preparation use the same logical camera names and physical camera mapping.

## Files

| File | Purpose |
|---|---|
| `cameras.py` | Core preset registry and reusable camera API. |
| `view_cameras.py` | Live OpenCV grid viewer over shared presets. |
| `capture.py` | Interactive multi-camera snapshot capture. |
| `calibrate.py` | Visual overlay alignment against reference images. |
| `check_usb.py` | RealSense USB/topology diagnostics. |
| `reset_camera.py` | Hardware-reset RealSense devices. |
| `test_view_cameras.py` | Basic viewer/preset tests. |

## Main API

Defined in `cameras.py`:

- `CAMERA_PRESETS`
- `DEFAULT_PRESET = "realsense"`
- `init_cameras(camera_config=None, width=640, height=480, fps=30)`
- `capture_frames(cameras)`
- `get_latest_depth(cameras)`
- `stop_cameras(cameras)`
- `build_grid_frame(...)`
- `show_camera_grid(...)`

## Docs

- `presets.md` - all camera presets and naming rules.
- `commands.md` - viewer, capture, calibration, USB, reset commands.
- `integration.md` - how cameras connect to datasets, inference, evaluation, and depth.

## Critical Rule

Camera names are model-interface names. If physical cameras are swapped but logical names stay wrong, the model receives the wrong observation under the right key. Verify mapping before collecting data or evaluating a checkpoint.
