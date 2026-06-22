# Camera viewer preset fix

Date: 2026-05-29

`vbti/logic/cameras/view_cameras.py` was originally doing raw RealSense discovery and starting streams directly, which ignored `logic/cameras/cameras.py` presets, camera names/order, OpenCV backends, and per-camera settings like depth/fps overrides.

Fix pattern:
- Keep `view_cameras.py` as a thin CLI wrapper over shared camera utilities.
- Add `--preset` with default `DEFAULT_PRESET` and choices from `CAMERA_PRESETS`.
- Use `init_cameras()`, `capture_frames()`, `build_grid_frame()`, and `stop_cameras()` from `cameras.py`.
- Keep hardware-free tests focused on CLI preset parsing.

Validation used: from `vbti/logic/cameras`, run `python -m unittest test_view_cameras.py && python -m py_compile view_cameras.py cameras.py`.
