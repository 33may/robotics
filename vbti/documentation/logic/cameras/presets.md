# Camera Presets

Presets are defined in `logic/cameras/cameras.py` and consumed by camera tools, inference, and evaluation.

## Preset Table

| Preset | Cameras | Backend | Use |
|---|---|---|---|
| `realsense` | `top`, `left`, `right`, `gripper` | RealSense serials | Canonical four-RGB D405 setup. |
| `realsense_depth` | `top`, `left`, `right`, `gripper` | RealSense serials | Four RGB cameras plus aligned gripper depth. |
| `opencv` | `top`, `left`, `right`, `gripper` | `/dev/cam_*_raw` | Stable V4L2 RGB through udev symlinks. |
| `opencv-3cam` | `left`, `right`, `gripper` | `/dev/cam_*` | Older/reduced three-camera layout. |
| `opencv_depth` | `top`, `left`, `right`, `gripper` | OpenCV fixed cams + RealSense gripper | RGB through OpenCV, depth through gripper D405. |
| `sim_3_cam` | `wrist_cam`, `left_cam`, `right_cam` | OpenCV symlinks | Simulation-style naming. |
| `sim_3_cam_opencv` | `wrist_cam`, `left_cam`, `right_cam` | OpenCV raw symlinks | Simulation-style three-camera raw V4L2. |
| `sim_4cam_opencv` | `top_cam`, `left_cam`, `right_cam`, `wrist_cam` | OpenCV raw symlinks | Simulation-style four-camera raw V4L2. |
| `sim_4cam_realsense` | `top_cam`, `left_cam`, `right_cam`, `wrist_cam` | Stable camera symlinks | Simulation-style four-camera real setup. |

## RealSense Preset

`realsense` maps logical camera names to D405 serial numbers. This is useful when RealSense device discovery is stable and depth/reset functionality is needed.

Use it for:

- full RealSense RGB viewing;
- inference/evaluation when RealSense stability is acceptable;
- depth variants.

## OpenCV Preset

`opencv` maps logical camera names to `/dev/cam_*_raw` symlinks. This avoids `/dev/video*` number shuffling.

Use it for:

- RGB-only real inference/evaluation;
- lower-overhead V4L2 access;
- workflows where librealsense instability is not needed.

## Depth Presets

Depth requires librealsense. OpenCV cannot access D405 z16 depth.

Depth presets work by ensuring the gripper camera is opened through RealSense and configured with `depth=True`. The rest of the fixed cameras may still use OpenCV in `opencv_depth`.

## Udev Mapping

Rules live in `udev/99-realsense-cams.rules`.

Install:

```bash
sudo cp vbti/udev/99-realsense-cams.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Use symlinks instead of numeric `/dev/video*` paths. Numeric video devices can change after unplug, reset, or reboot.

## Known Mapping Caveat

There is historical evidence of a gripper/left serial mismatch between `cameras.py` and udev memory/rules. Before important data collection or evaluation, verify physical view with:

```bash
python -m vbti.logic.cameras.view_cameras --preset realsense
python -m vbti.logic.cameras.view_cameras --preset opencv
```

Do not assume the mapping is correct just because the code opens all cameras.
