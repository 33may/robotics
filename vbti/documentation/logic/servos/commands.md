# Servo Commands

## Scan All Ports

```bash
python -m vbti.logic.servos.scan_all
```

Scans sorted `/dev/ttyACM*` ports and prints each expected motor ID with:

- position;
- voltage;
- temperature;
- hardware error flag;
- EEPROM lock state;
- control mode.

Use voltage to identify arms:

- leader/teleop arm: about 5V;
- follower/robot arm: about 12V.

## List And Show Profiles

```bash
python -m vbti.logic.servos.profiles list
python -m vbti.logic.servos.profiles show frodeo-test
```

`show` prints calibration values per joint: ID, homing offset, min/max ranges, and degree limits.

## Load Calibration Profile

```bash
python -m vbti.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
```

This writes calibration values to the motors. Treat it as an EEPROM-writing operation.

Backward-compatible wrapper:

```bash
python -m vbti.logic.servos.load_calibration --port /dev/ttyACM1 --robot_id frodeo-test
```

If `robot_id` is omitted, it uses the active/default profile.

## Move To Rest

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/rest.py --port=/dev/ttyACM1 --speed=3.0 --fps=30
```

Rest pose:

```text
shoulder_pan:    0.0
shoulder_lift: -95.0
elbow_flex:     95.0
wrist_flex:     45.0
wrist_roll:      0.0
gripper:         0.0
```

Pass `--port` explicitly. Defaults differ from other scripts and may not match the follower on a given boot.

## Active Workflow Before Evaluation

```bash
python -m vbti.logic.servos.scan_all
python -m vbti.logic.servos.profiles list
python -m vbti.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/rest.py --port=/dev/ttyACM1
```

Only after this should model behavior be interpreted as model behavior rather than hidden hardware state.
