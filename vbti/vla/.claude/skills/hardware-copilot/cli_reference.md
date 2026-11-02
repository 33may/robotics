# Hardware CLI reference

`W` = writes servo EEPROM or moves the arm. Ask before running.

## Servos: `python -m vla.logic.servos.<module>`

| Command | Args | Does | |
|---|---|---|---|
| `scan_all` | none (scans every `/dev/ttyACM*`) | Per servo: position, voltage, temp, hw_err, EEPROM lock, mode | |
| `profiles list` | | Profiles, active one, cache presence | |
| `profiles show` | `NAME` | Per-joint ID, homing offset, range, degree limits | |
| `profiles load` | `NAME --port P` | Write a profile to the motors | W |
| `profiles sync` / `activate` / `export` / `register` | | Registry and LeRobot cache management | |
| `profiles delete` | `NAME` | Removes registry, cache and backup entries | |
| `load_calibration` | `[--port] [--robot_id]` | Old wrapper around `profiles load`; uses the active profile if no id | W |
| `rest` | `--port P [--robot_id] [--speed] [--fps]` | Follower to rest pose | W |
| `unlock_all` | `[--port P]` (all ports if omitted) | Unlock EEPROM on IDs 1-6 | W |
| `change_id` | `PORT CURRENT_ID NEW_ID` (positional) | Change one servo ID; checks target is free, asks to confirm | W |
| `factory_reset_motors` | none; edit `PORT`, `MOTORS_TO_RESET`, `TEMP_ID` in the file | Factory reset without ID collision | W |
| `quick_recalib` | `--port --old_profile --new_profile` | Take the current pose as zero, write a new profile | W |
| `calibrate_interactive` | `--port --name --base` | Curses TUI to nudge per-joint zero offsets | W |

Rest pose: `shoulder_pan 0, shoulder_lift -95, elbow_flex 95, wrist_flex 45, wrist_roll 0, gripper 0`.

Profile storage: LeRobot cache `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`, registry `calibration/registry.json`, backups `calibration/profiles/`. `quick_recalib.py` references `so101_follower` while `profiles.py` uses `so_follower`; check which one the installed LeRobot reads.

## Cameras: `python -m vla.logic.cameras.<module>`

| Command | Args | Does |
|---|---|---|
| `view_cameras` | `[--preset realsense] [--width 640] [--height 480] [--fps 30]` | Live grid; `q` quits |
| `capture` | `--save_dir D [--preset P]` | `c` saves `NNN_<camera>.png` + `NNN_meta.json` |
| `calibrate` | `--ref_dir D [--camera C] [--preset P]` | Overlay live view on reference shots to re-place a moved camera |
| `check_usb` | | `usbfs_memory_mb`, serial / firmware / USB speed per D405, `lsusb -t` |
| `reset_camera` | `[SERIAL]` | Hardware reset one or all D405s; runs `sudo udevadm trigger` |

Presets (`CAMERA_PRESETS` in `logic/cameras/cameras.py`):

| Preset | Backend | Use |
|---|---|---|
| `realsense` | D405 by serial | Default, four RGB |
| `realsense_depth` | D405 by serial | Four RGB + gripper depth |
| `opencv` | `/dev/cam_*_raw` symlinks | RGB, V4L2; what `lerobot-record` uses |
| `opencv_depth` | OpenCV fixed cams + RealSense gripper | RGB via OpenCV, depth via gripper D405 |
| `sim_4cam_opencv`, `sim_4cam_realsense`, `sim_3_cam*`, `opencv-3cam` | | Sim-style names / older layouts |

## Inference and replay

| Command | Args | Does | |
|---|---|---|---|
| `inference.run_real_inference preview` | `[--camera_config] [--width] [--height] [--fps]` | Camera grid, no model, no robot | |
| `inference.run_real_inference run` | `--checkpoint --port --cameras --task --depth --action_horizon --max_steps --record ...` | Policy on the robot; flags in `docs/tutorial/07_inference.md` | W |
| `inference.eval_engine` | `--checkpoint --protocol --port --cameras --depth --action_horizon --record --resume` | Protocol evaluation session | W |
| `dataset.replay_utils show` | `DATASET_PATH [--episode_idx]` | Print an episode's actions, no robot | |
| `dataset.replay_utils replay` | `DATASET_PATH --port [--episode_idx] [--max_relative_target=10.0] [--speed] [--dry_run]` | Replay an episode | W |
| `dataset.replay_utils goto` | `--port [--<joint>=deg] [--duration]` | Move to a joint pose | W |

## LeRobot CLIs

| Command | Does |
|---|---|
| `lerobot-find-port` | Find a serial port (interactive, unplug the cable) |
| `lerobot-find-cameras realsense` / `opencv` | List cameras with serials / device paths |
| `lerobot-calibrate` | Calibrate leader (`--teleop.*`) or follower (`--robot.*`) |
| `lerobot-teleoperate` | Teleop without recording |
| `lerobot-record` | Teleop recording into a LeRobot dataset |

`rs-enumerate-devices -s` lists D405 serials. `udevadm info -q property -n /dev/videoN | grep ID_SERIAL_SHORT` gives the V4L2 serial for the udev rules.
