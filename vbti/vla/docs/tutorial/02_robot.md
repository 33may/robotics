# 2. Robot

**What you have:** two SO-101 arms on Feetech STS3215 servos — a leader (you move it) and a follower (it copies). Joint meaning and zero positions come from the calibration; every dataset's `action` / `observation.state` is only valid for the calibration it was recorded with.

**What you need to do:** find which USB port is which arm, calibrate both with LeRobot, and register the follower calibration as a named profile so recordings and inference can refer to it.

```bash
python -m vla.logic.servos.scan_all
lerobot-calibrate --robot.type=so101_follower --robot.port=<follower_port> --robot.id=<name>
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<leader_port>  --teleop.id=<name>
python -m vla.logic.servos.profiles sync
python -m vla.logic.servos.profiles activate <name>
python -m vla.logic.servos.rest --port=<follower_port> --robot_id=<name>
```

**Done when:**
- `scan_all` shows both ports, 6 servos each, all responding, no hardware errors
- `profiles list` shows your profile as active
- `rest` moves the follower to its rest pose

## Robot commands

| Command | What it does |
|---|---|
| `python -m vla.logic.servos.scan_all` | Every port: per-servo ID, position, voltage, temperature, HW error, EEPROM lock |
| `python -m vla.logic.servos.profiles list \| show \| load \| export \| register \| sync \| activate \| delete` | Calibration profiles: LeRobot cache + `calibration/registry.json`; `load` writes a profile to the motors' EEPROM |
| `python -m vla.logic.servos.rest --port <p>` | Move the follower to rest |
| `python -m vla.logic.servos.calibrate_interactive` | TUI: nudge per-joint offsets until 0° matches the simulation zero |
| `python -m vla.logic.servos.quick_recalib` | Set the current pose as the new zero |
| `python -m vla.logic.servos.unlock_all` | Unlock EEPROM on all servos so calibration can be written |
| `python -m vla.logic.servos.change_id` | Change a servo's ID (new / replaced motor) |
| `python -m vla.logic.servos.factory_reset_motors` | Factory-reset specific motors |

Full reference: `docs/modules/servos/`.
