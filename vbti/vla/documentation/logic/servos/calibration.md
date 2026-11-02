# Servo Calibration Profiles

## Storage

The profile manager uses:

```text
LeRobot cache:
~/.cache/huggingface/lerobot/calibration/robots/so_follower

Project registry:
vbti/calibration/registry.json

Project backups:
vbti/calibration/profiles
```

Known caveat: `quick_recalib.py` references `so101_follower`, while `profiles.py` references `so_follower`. Verify which path your current LeRobot install expects.

## Profile Commands

```bash
python -m vbti.logic.servos.profiles list
python -m vbti.logic.servos.profiles show frodeo-test
python -m vbti.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
python -m vbti.logic.servos.profiles export frodeo-test
python -m vbti.logic.servos.profiles register my-new --description "New calibration"
python -m vbti.logic.servos.profiles sync
python -m vbti.logic.servos.profiles activate frodeo-test
python -m vbti.logic.servos.profiles delete old-profile
```

`delete` removes registry/cache/backup entries. Do not use it as cleanup unless explicitly intended.

## Profile Fields

Each joint stores:

- motor `id`;
- `drive_mode`;
- `homing_offset`;
- `range_min`;
- `range_max`.

These values define how raw Feetech encoder ticks become the joint degrees used by LeRobot datasets and inference.

## Interactive Calibration

```bash
python -m vbti.logic.servos.calibrate_interactive --port=/dev/ttyACM1 --name=sim_accurate --base=frodeo-test
```

Main controls:

- arrows / `1`-`6`: select joint;
- Enter: tune selected joint;
- `z`: all-zero test;
- `s`: save;
- `q`: quit.

Tune controls:

- `+` / `-`: small nudge;
- `>` / `<`: larger nudge;
- `[` / `]`: large nudge;
- `z`: set current joint to zero;
- `a`, `b`: range-related actions depending on TUI context.

This writes homing offset and range registers. Do not run casually.

## Quick Recalibration

Use when manually placing the robot at simulation zero and deriving new offsets:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/quick_recalib.py \
  --port=/dev/ttyACM1 \
  --old_profile=frodeo-test \
  --new_profile=sim_accurate
```

Typical sequence:

```bash
python -m vbti.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/quick_recalib.py --port=/dev/ttyACM1 --old_profile=frodeo-test --new_profile=sim_accurate
python -m vbti.logic.servos.profiles load sim_accurate --port /dev/ttyACM1
```

## Calibration And Sim-Real

Dataset joint values are degrees relative to the calibration profile used during recording. Simulation joint values have a different zero/range convention. Sim-real alignment is not only radians-to-degrees conversion; calibration zero poses must align.
