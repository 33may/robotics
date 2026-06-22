# Runbook: Preflight Before Robot Work

Use before data collection, real inference, or protocol evaluation.

## 1. Activate Environment

Most real robot tools run from the robotics workspace with the LeRobot env:

```bash
cd /home/may33/projects/ml_portfolio/robotics
conda activate lerobot
```

## 2. Scan Servos

```bash
python -m vbti.logic.servos.scan_all
```

Check:

- follower arm is visible;
- all IDs 1-6 respond;
- no hardware error flags;
- voltage identifies leader/follower correctly.

Expected voltage clue:

- leader: about 5V;
- follower: about 12V.

## 3. Load Known Calibration If Needed

```bash
python -m vbti.logic.servos.profiles list
python -m vbti.logic.servos.profiles show frodeo-test
python -m vbti.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
```

Use the actual follower port from the scan. Do not blindly use `/dev/ttyACM1` if scan shows a different mapping.

## 4. Move Robot To Rest

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/rest.py --port=/dev/ttyACM1 --speed=3.0 --fps=30
```

## 5. Check Cameras

RGB-only:

```bash
python -m vbti.logic.cameras.view_cameras --preset realsense
```

OpenCV symlink path:

```bash
python -m vbti.logic.cameras.view_cameras --preset opencv
```

Depth path:

```bash
python -m vbti.logic.cameras.view_cameras --preset opencv_depth
```

Check:

- logical names match physical views;
- gripper camera works under motion;
- top camera sees the protocol workspace;
- no frozen/black frames.

## 6. USB Diagnostics If Cameras Are Unstable

```bash
python -m vbti.logic.cameras.check_usb
```

If RealSense is stuck:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/cameras/reset_camera.py
```

## 7. Decide The Mode

| Task | Camera preset |
|---|---|
| RGB data collection via LeRobot | Usually OpenCV camera config or known recording command. |
| RGB inference/eval | `realsense` or `opencv`. |
| Depth inference/eval | `opencv_depth` or `realsense_depth`. |
| Protocol eval | preset must match checkpoint schema. |

## Stop Conditions

Do not continue if:

- any servo ID is missing;
- follower voltage/port is uncertain;
- calibration profile is unknown;
- a camera logical name is wrong;
- gripper camera disconnects during arm motion;
- depth checkpoint is about to run without depth camera support.
