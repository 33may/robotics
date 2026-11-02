# Servo Recovery And Safety

## EEPROM Unlock

```bash
python -m vla.logic.servos.unlock_all
python -m vla.logic.servos.unlock_all --port=/dev/ttyACM1
```

Unlocks EEPROM register on IDs 1-6. Use before operations that need persistent motor writes.

## Change Motor ID

```bash
python -m vla.logic.servos.change_id /dev/ttyACM1 <current_id> <new_id>
```

The script:

- verifies the current ID responds;
- checks that the target ID is free;
- asks for confirmation;
- unlocks EEPROM;
- writes the new ID;
- verifies the new ID responds.

## Factory Reset

```bash
python -m vla.logic.servos.factory_reset_motors
```

Warning: read the constants at the top of the file before running:

```python
PORT = "/dev/ttyACM1"
MOTORS_TO_RESET = [3]
TEMP_ID = 7
```

The script temporarily moves ID 1 out of the way, resets the target, restores IDs, and prints recovery commands if something fails.

## Recovery Workflow

```bash
python -m vla.logic.servos.scan_all
python -m vla.logic.servos.unlock_all --port=/dev/ttyACM1
python -m vla.logic.servos.change_id /dev/ttyACM1 <current_id> <new_id>
python -m vla.logic.servos.factory_reset_motors
python -m vla.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
```

Do not run the whole workflow blindly. Pick the exact recovery step needed.

## Safety Checklist

- Identify leader/follower by voltage before writing to motors.
- Pass `--port` explicitly.
- Keep one known-good calibration profile exported.
- After ID or calibration changes, run `scan_all` and `profiles show`.
- Return to rest before inference/evaluation.
- If a servo reports hardware error, diagnose hardware before running a model.
