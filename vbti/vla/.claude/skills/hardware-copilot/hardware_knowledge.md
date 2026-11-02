# Hardware knowledge

## Topology

- Two SO-101 arms: leader (teleop input, ~5 V) and follower (executes actions, ~12 V). Identify them by the voltage `scan_all` reports; `/dev/ttyACM*` numbers change across boots.
- Servos: Feetech STS3215, IDs 1-6 = `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`, baud 1,000,000. Same order as dataset action/state.
- Cameras: 4x RealSense D405, logical names `top`, `left`, `right`, `gripper`. Serials live in `CAMERA_PRESETS` (`logic/cameras/cameras.py`) and in `udev/99-realsense-cams.rules` (`/dev/cam_<name>_raw`). Keep both in sync.
- OpenCV can read D405 RGB but not z16 depth. Depth needs librealsense on the gripper camera (`realsense_depth`, `opencv_depth`).
- There is a recorded gripper/left serial mismatch between `cameras.py` and the udev rules. Verify each view under its name with `view_cameras` before collection or eval.

## Servo registers (STS3215)

| Register | Meaning |
|---|---|
| 31-32 | `homing_offset`, signed 16-bit |
| 33 | operating mode |
| 55 | EEPROM lock (0 unlocked, 1 locked) |
| 56 | present position |
| 62 | voltage (x0.1 V) |
| 63 | temperature (°C) |
| 65 | hardware error flags (`hw_err`) |

`hw_err` persists across power cycles; clearing it needs a factory reset.

## Factory reset without bus collision

A reset servo comes back as ID 1, which collides with `shoulder_pan`. `factory_reset_motors.py` does:

1. Move ID 1 to temporary ID 7.
2. Send RESET (raw `FF FF <id> 02 06 <checksum>`) to the target; it becomes ID 1.
3. Change ID 1 to the target ID.
4. Move ID 7 back to ID 1.
5. Reload the calibration profile (`profiles load`).

On failure the script prints recovery commands. Set `PORT`, `MOTORS_TO_RESET` at the top of the file first.

## "No status packet"

Check in order:

1. Wrong port: `scan_all` / `lerobot-find-port`.
2. Loose USB or servo daisy-chain cable.
3. Power: the follower needs its own supply.
4. Two servos with the same ID.
5. `hw_err` set: scan, factory reset if needed.
6. EEPROM locked (register 55): `unlock_all` before writing.

## Calibration

- A calibration defines the joint zero and range. Datasets, checkpoints and inference are only consistent under the profile they were recorded with.
- Profiles: LeRobot cache `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<name>.json`, registry `calibration/registry.json`, backups `calibration/profiles/`.
- Re-verify (`scan_all`, `profiles show`) after any servo replacement, ID change or factory reset.
- Sim/real alignment needs the calibration zero pose to match the sim zero. `calibrate_interactive` and `quick_recalib` exist for that.

## USB

```
per camera: 640 x 480 x 3 B x 30 fps ≈ 27.6 MB/s
4 cameras:  ≈ 110 MB/s
USB 2.0 practical: ~35-40 MB/s  -> four cameras need USB 3.x
```

- Kernel buffer: `usbcore.usbfs_memory_mb=1000` in `GRUB_CMDLINE_LINUX`, then `update-grub` and reboot. `check_usb` reports the current value.
- Frame timeouts: `reset_camera` (one serial or all), then retry. Check that each D405 enumerates at 5000M in `check_usb`.
- Fallback when bandwidth is short: lower FPS or resolution (`view_cameras --fps 15`).

## Safety for replay and inference

- `max_relative_target` clamps per-step joint deltas (10.0 default in replay and inference).
- `replay --dry_run` first.
- Return to rest before and after runs.
- Stop if the robot or cameras misbehave; do not keep running trials on broken hardware.
