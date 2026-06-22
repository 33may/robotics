# Camera Commands

## Live Viewer

```bash
python -m vbti.logic.cameras.view_cameras
python -m vbti.logic.cameras.view_cameras --preset realsense_depth
python -m vbti.logic.cameras.view_cameras --preset opencv --fps 15
python -m vbti.logic.cameras.view_cameras --preset opencv --width 320 --height 240
```

Arguments:

| Arg | Default | Meaning |
|---|---|---|
| `--preset` | `realsense` | Key from `CAMERA_PRESETS`. |
| `--width` | `640` | Requested capture width. |
| `--height` | `480` | Requested capture height. |
| `--fps` | `30` | Requested capture FPS. |

Controls:

- `q`: quit.

## Snapshot Capture

```bash
python -m vbti.logic.cameras.capture --save_dir ./captures/cup_center
python -m vbti.logic.cameras.capture --save_dir ./captures/no_cup
python -m vbti.logic.cameras.capture --save_dir ./captures/cup_left --preset opencv
```

Arguments:

| Arg | Default | Meaning |
|---|---|---|
| `--save_dir` | required | Output folder. |
| `--preset` | `realsense` | Camera preset. |

Controls:

- `c`: capture all cameras.
- `q` / Esc: quit.

Outputs per capture index:

```text
000_top.png
000_left.png
000_right.png
000_gripper.png
000_meta.json
```

## Visual Alignment Overlay

```bash
python -m vbti.logic.cameras.calibrate --ref_dir /tmp/current_cams
python -m vbti.logic.cameras.calibrate --ref_dir /tmp/current_cams --camera top
python -m vbti.logic.cameras.calibrate --ref_dir /tmp/current_cams --preset opencv
```

Arguments:

| Arg | Default | Meaning |
|---|---|---|
| `--ref_dir` | required | Folder with reference images. |
| `--camera` | optional | Restrict to one logical camera. |
| `--preset` | `realsense` | Camera preset. |

Reference files may be named `<camera>.png` or `*_<camera>.png`.

Controls:

- `n`: next camera.
- `p`: previous camera.
- `space`: advance.
- `q` / Esc: quit.

This is for visual placement drift, not intrinsic calibration.

## USB Diagnostics

```bash
python -m vbti.logic.cameras.check_usb
```

Reports:

- `usbfs_memory_mb`;
- RealSense serial/name/firmware/USB speed;
- filtered `lsusb -t` topology.

If four cameras timeout, check:

- USB 2 vs USB 3 path;
- `usbfs_memory_mb` too low;
- gripper cable flex;
- camera body/cable swap evidence.

## RealSense Reset

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/cameras/reset_camera.py
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/cameras/reset_camera.py 128422270260
```

No serial resets all detected RealSense devices. With a serial, only that camera is reset.

Warning: current script includes sudo/udev trigger behavior. Read it before modifying or running in a different environment.

## Test Viewer Parsing

```bash
cd /home/may33/projects/ml_portfolio/robotics/vbti/logic/cameras
python -m unittest test_view_cameras.py
python -m py_compile view_cameras.py cameras.py
```
