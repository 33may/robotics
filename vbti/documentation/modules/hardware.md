# Hardware Module

## Scope

Hardware code lives in:

- `logic/cameras/`
- `logic/servos/`
- `udev/99-realsense-cams.rules`

It manages the physical SO-101/Feetech servo setup and four RealSense D405 cameras.

## Cameras

Main file: `logic/cameras/cameras.py`.

Shared API:

- `CAMERA_PRESETS`
- `init_cameras(camera_config=None, width=640, height=480, fps=30)`
- `capture_frames(cameras)`
- `get_latest_depth(cameras)`
- `stop_cameras(cameras)`
- `build_grid_frame(...)`
- `show_camera_grid(...)`

## Camera Presets

| Preset | Purpose |
|---|---|
| `realsense` | Four RGB D405 cameras by RealSense serial. Default/canonical for many real runs. |
| `realsense_depth` | Same cameras with aligned gripper depth. |
| `opencv` | Four RGB cameras through `/dev/cam_*_raw` symlinks. |
| `opencv-3cam` | Reduced left/right/gripper OpenCV layout. |
| `opencv_depth` | Fixed RGB cameras by OpenCV, gripper by RealSense depth. |
| `sim_3_cam`, `sim_4cam_*` | Simulation-style camera names for sim-compatible schemas. |

Camera names are part of the model interface. Do not swap physical cameras without updating the preset or recording/inference schema.

## Camera Commands

Live view:

```bash
python -m vbti.logic.cameras.view_cameras --preset realsense
python -m vbti.logic.cameras.view_cameras --preset realsense_depth
python -m vbti.logic.cameras.view_cameras --preset opencv --fps 15
```

Capture snapshots:

```bash
python -m vbti.logic.cameras.capture --save_dir ./captures/cup_center
python -m vbti.logic.cameras.capture --save_dir ./captures/cup_left --preset opencv
```

Controls: `c` captures, `q` or Esc quits.

Visual alignment against references:

```bash
python -m vbti.logic.cameras.calibrate --ref_dir /tmp/current_cams
python -m vbti.logic.cameras.calibrate --ref_dir /tmp/current_cams --camera top
```

This is visual placement alignment, not geometric camera calibration.

USB diagnostics:

```bash
python -m vbti.logic.cameras.check_usb
```

RealSense reset:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/cameras/reset_camera.py
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/cameras/reset_camera.py 128422270260
```

## Udev Symlinks

Install rules:

```bash
sudo cp vbti/udev/99-realsense-cams.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Use `/dev/cam_*` instead of `/dev/video*`; video numbers can shuffle. The `_raw` symlinks are used by the `opencv` preset.

Known caveat: code, udev rules, and physical camera swaps must be verified together. There is evidence of a historical gripper/left serial mismatch between `cameras.py` and udev notes.

## Camera Depth Workflow

Depth only comes from RealSense/librealsense, not plain OpenCV/V4L2.

When a camera config has `depth=True`, `init_cameras()` starts color and z16 depth streams at the same FPS, aligns depth to color, and stores the last depth map. `get_latest_depth()` returns uint16 depth maps; multiply by the RealSense scale for meters.

Inference/eval converts gripper depth to turbo RGB and injects it as:

```text
observation.images.gripper_depth
```

Canonical close-range clip is `0.05-0.20 m`.

## Servos

Main files:

- `logic/servos/scan_all.py`
- `logic/servos/profiles.py`
- `logic/servos/load_calibration.py`
- `logic/servos/rest.py`
- `logic/servos/calibrate_interactive.py`
- `logic/servos/quick_recalib.py`
- `logic/servos/unlock_all.py`
- `logic/servos/change_id.py`
- `logic/servos/factory_reset_motors.py`

Joint order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Motor IDs:

| ID | Joint |
|---|---|
| 1 | `shoulder_pan` |
| 2 | `shoulder_lift` |
| 3 | `elbow_flex` |
| 4 | `wrist_flex` |
| 5 | `wrist_roll` |
| 6 | `gripper` |

Baud rate is `1000000`.

## Servo Commands

Scan all `/dev/ttyACM*` ports:

```bash
python -m vbti.logic.servos.scan_all
```

Use voltage to identify arms:

- leader: about 5V;
- follower: about 12V.

Profile manager:

```bash
python -m vbti.logic.servos.profiles list
python -m vbti.logic.servos.profiles show frodeo-test
python -m vbti.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
python -m vbti.logic.servos.profiles export frodeo-test
python -m vbti.logic.servos.profiles register my-new --description "New calib"
python -m vbti.logic.servos.profiles sync
python -m vbti.logic.servos.profiles activate frodeo-test
```

Backward-compatible loader:

```bash
python -m vbti.logic.servos.load_calibration --port /dev/ttyACM1 --robot_id frodeo-test
```

Move to rest:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/rest.py --port=/dev/ttyACM1 --speed=3.0 --fps=30
```

Rest pose in degrees:

```text
shoulder_pan: 0
shoulder_lift: -95
elbow_flex: 95
wrist_flex: 45
wrist_roll: 0
gripper: 0
```

EEPROM unlock:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/unlock_all.py --port=/dev/ttyACM1
```

Change ID:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/change_id.py /dev/ttyACM1 6 1
```

Factory reset:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/factory_reset_motors.py
```

Warning: `factory_reset_motors.py` has hardcoded `PORT`, `MOTORS_TO_RESET`, and `TEMP_ID`. Read/edit before running.

Interactive recalibration:

```bash
python -m vbti.logic.servos.calibrate_interactive --port=/dev/ttyACM1 --name=sim_accurate --base=frodeo-test
```

Quick recalibration:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/quick_recalib.py \
  --port=/dev/ttyACM1 \
  --old_profile=frodeo-test \
  --new_profile=sim_accurate
```

## Safety Rules

- Run `scan_all` before hardware debugging.
- Pass `--port` explicitly; defaults differ between scripts.
- Treat `profiles load`, `calibrate_interactive`, `quick_recalib`, `unlock_all`, `change_id`, and `factory_reset_motors` as EEPROM/register-writing operations.
- Do not delete profiles unless intentionally removing cache and backup entries.
- If a model fails, verify servo calibration and camera placement before changing model code.
- Use X11 for data collection if LeRobot keyboard controls fail under Wayland.
