---
name: hardware-copilot
description: Hardware copilot for the vla framework's SO-101 arms, Feetech STS3215 servos, RealSense D405 cameras and real-robot deployment. Use for servo scans, calibration profiles, USB/camera issues, teleoperation and recording, rest pose, servo recovery, and running a policy on the real robot. Triggers on servo, motor, calibration, USB, camera, RealSense, teleop, record, deploy, inference, real robot, port.
---

# Hardware Copilot

Servo state, calibration, cameras, teleop/recording and real-robot inference. A model is only judged after the hardware is known good. Paths are relative to `vbti/vla`; commands run there in the `vla` conda env.

## Session start

1. If `$ARGUMENTS` names an issue, go to it.
2. Otherwise ask what hardware is involved.
3. Before any diagnosis: `python -m vla.logic.servos.scan_all` and `python -m vla.logic.servos.profiles list`.

## References

- [cli_reference.md](cli_reference.md): hardware commands and which ones write to motors.
- [hardware_knowledge.md](hardware_knowledge.md): topology, servo registers, recovery, USB, power.
- `docs/tutorial/01_cameras.md`, `02_robot.md`, `03_teleoperation.md`, `07_inference.md`; `docs/runbooks/preflight.md`; `docs/modules/{cameras,servos}/`.

## Workflows

### Preflight (before recording, inference or eval)

```bash
python -m vla.logic.servos.scan_all                         # ports, IDs 1-6, voltage, hw_err, lock
python -m vla.logic.servos.profiles list                    # active calibration
python -m vla.logic.servos.rest --port=<follower_port>      # follower to rest
python -m vla.logic.cameras.check_usb                       # USB 3.x, usbfs_memory_mb
python -m vla.logic.cameras.view_cameras --preset realsense # each view under its correct name
```

Identify leader/follower by voltage (leader ~5 V, follower ~12 V), not by port number.

### Calibrate and register

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<follower_port> --robot.id=<name>
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<leader_port>  --teleop.id=<name>
python -m vla.logic.servos.profiles sync
python -m vla.logic.servos.profiles activate <name>
```

### Record demonstrations

`lerobot-record` with the `/dev/cam_<name>_raw` udev symlinks; full command in `docs/tutorial/03_teleoperation.md`. Record from X11 (LeRobot keyboard controls fail under Wayland). Keep the task text identical across sessions.

### Servo recovery

```bash
python -m vla.logic.servos.scan_all
python -m vla.logic.servos.unlock_all --port=<port>
python -m vla.logic.servos.change_id <port> <current_id> <new_id>
python -m vla.logic.servos.factory_reset_motors     # edit PORT / MOTORS_TO_RESET at the top first
python -m vla.logic.servos.profiles load <name> --port <port>
```

Pick the one step needed; do not run the sequence blindly. Procedure in [hardware_knowledge.md](hardware_knowledge.md).

### Run a policy

```bash
python -m vla.logic.cameras.view_cameras --preset <preset>
python -m vla.logic.inference.run_real_inference run \
    --checkpoint=experiments/<exp>/<vNNN>/<run_name>/checkpoints/<NNNNNN>/pretrained_model \
    --port=<follower_port> --cameras=<preset> --task="<task text>" \
    --depth=<true|false> --action_horizon=10 --max_steps=10000 --move_to_start=true
python -m vla.logic.servos.rest --port=<follower_port>
```

Camera names in the preset must match the training dataset; the task text must match training. `run` moves the follower to rest with `max_relative_target=10.0` regardless of the flag (`KNOWN_ISSUES.md`).

Protocol evaluation: `python -m vla.logic.inference.eval_engine`, see `docs/tutorial/06_evaluation.md`.

## Rules

- Scan before operating; scan again after any ID, calibration or reset change.
- Ask before anything that writes EEPROM or moves the arm: `profiles load`, `unlock_all`, `change_id`, `factory_reset_motors`, `quick_recalib`, `calibrate_interactive`, `replay_utils replay/goto`, `rest`, inference.
- Pass `--port` explicitly. Script defaults differ (`/dev/ttyACM0` vs `/dev/ttyACM1`).
- `replay_utils replay --dry_run` before a real replay. Keep `max_relative_target` set.
- Address cameras by RealSense serial or `/dev/cam_*` symlink, never by `/dev/video*` number.
- Depth needs librealsense on the gripper camera. `--depth=true` only for depth-trained checkpoints.
- Erratic servos under load: check the follower power supply before software.
- Cameras dropping during motion: suspect the gripper cable / USB path before software.
- Isolate one variable at a time (one camera, then two, ...).
- Persistent system fixes (GRUB `usbfs_memory_mb`, udev rules), not runtime `echo`.
