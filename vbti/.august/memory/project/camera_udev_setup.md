---
name: Camera udev rules and stream types
description: RealSense D405 udev symlinks, UYVY vs YUYV stream mapping, LeRobot opencv camera issues
type: project
---

## RealSense D405 V4L2 Stream Types

Each D405 exposes 6 `/dev/video*` nodes, 3 with data:
- `ATTR{index}==2` → UYVY (raw, lower quality)
- `ATTR{index}==4` → YUYV (processed/ISP, better quality)
- Z16 = depth stream

## Udev Symlinks

Installed at `/etc/udev/rules.d/99-realsense-cams.rules` (source: `vbti/udev/99-realsense-cams.rules`).
Creates stable symlinks by V4L2 serial + index:

| Symlink | V4L2 Serial | RS Serial | Camera |
|---------|------------|-----------|--------|
| `/dev/cam_top` | 125423070759 | 123622270073 | top |
| `/dev/cam_gripper` | 130523070141 | 128422270260 | gripper |
| `/dev/cam_right` | 125423070468 | 126122270644 | right |
| `/dev/cam_left` | 125423070032 | 123622270367 | left |

`cam_*` → processed (YUYV, index==4), `cam_*_raw` → raw (UYVY, index==2).

**Why:** `/dev/video*` numbers shuffle on every USB re-enumeration. Symlinks are stable.

**How to apply:** Use `/dev/cam_top` etc. in both LeRobot and custom inference scripts instead of `/dev/videoN`.

## LeRobot OpenCV Camera Issues

1. `cv2.setNumThreads(1)` was global — starved 4-camera setups. Changed to `setNumThreads(4)` in `lerobot/src/lerobot/cameras/opencv/camera_opencv.py`.
2. Gripper camera (on moving arm) prone to USB micro-disconnects. LeRobot's read thread crashes after 11 consecutive failures. Custom inference script survives via last-frame fallback.
