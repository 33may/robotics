# 1. Cameras

**What you have:** four Intel RealSense D405 — `top`, `left`, `right`, `gripper` (wrist). Every dataset, training config and inference run refers to cameras by these four names, so they must always point at the same physical camera.

**What you need to do:** bind each physical camera to its name. Cameras are addressed in two ways — by serial in `logic/cameras/cameras.py` (`realsense*` presets, used for live view and inference) and by udev symlink `/dev/cam_<name>_raw` (`opencv*` presets, used by `lerobot-record`). Put your serials in both. Four cameras need a larger USB buffer than the kernel default.

```bash
# USB buffer: add usbcore.usbfs_memory_mb=1000 to GRUB_CMDLINE_LINUX in /etc/default/grub
sudo update-grub && sudo reboot

rs-enumerate-devices -s                              # serial of each D405
# edit CAMERA_PRESETS in logic/cameras/cameras.py with your serials
# edit udev/99-realsense-cams.rules with the V4L2 serials:
udevadm info -q property -n /dev/videoN | grep ID_SERIAL_SHORT
sudo cp udev/99-realsense-cams.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
python -m vla.logic.cameras.check_usb
python -m vla.logic.cameras.view_cameras --preset realsense
```

**Done when:**
- `ls /dev/cam_*` shows `cam_top_raw`, `cam_left_raw`, `cam_right_raw`, `cam_gripper_raw`
- `check_usb` lists all 4 cameras on USB 3.x and `usbfs_memory_mb` ≥ 1000
- `view_cameras` shows all 4 cameras, each under its correct name
- the grid runs stable

## Camera commands

| Command | What it does |
|---|---|
| `python -m vla.logic.cameras.view_cameras --preset <p> [--width --height --fps]` | Live grid of a preset (`realsense`, `realsense_depth`, `opencv`, `opencv_depth`, …). `q` quits. |
| `python -m vla.logic.cameras.capture --save_dir <dir> [--preset <p>]` | Snapshot all cameras on `c` → `NNN_<camera>.png` + `NNN_meta.json`. |
| `python -m vla.logic.cameras.calibrate --ref_dir <dir> [--camera <name>]` | Overlay live view on reference shots to re-place a camera after it moved (placement drift, not intrinsics). `n`/`p` switch camera. |
| `python -m vla.logic.cameras.check_usb` | `usbfs_memory_mb`, per-camera serial / firmware / USB speed, USB topology. |
| `python -m vla.logic.cameras.reset_camera [<serial>]` | Hardware-reset one RealSense (or all). Uses sudo + udev trigger. |

Presets live in `CAMERA_PRESETS` (`logic/cameras/cameras.py`); full reference: `docs/modules/cameras/`.
