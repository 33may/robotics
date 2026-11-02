# `logic/servos`

## Purpose

`logic/servos` is the SO-101/Feetech hardware utility layer. It makes servo state inspectable and recoverable so model failures are not confused with calibration or motor problems.

## Files

| File | Purpose |
|---|---|
| `scan_all.py` | Scan `/dev/ttyACM*` and print motor voltage/temp/error/lock/mode. |
| `profiles.py` | Calibration profile registry and EEPROM loading. |
| `load_calibration.py` | Backward-compatible wrapper around profile loading. |
| `rest.py` | Smoothly move follower to safe rest pose. |
| `calibrate_interactive.py` | Curses TUI for zero-offset tuning. |
| `quick_recalib.py` | Manual “current pose is zero” recalibration helper. |
| `unlock_all.py` | Unlock EEPROM on servo IDs 1-6. |
| `change_id.py` | Safely change one motor ID. |
| `factory_reset_motors.py` | Reset selected motor while avoiding ID collision. |

## Docs

- `commands.md` - routine commands.
- `calibration.md` - profile system and recalibration workflows.
- `recovery.md` - ID changes, unlock, factory reset, safety notes.

## Joint And ID Contract

| ID | Joint |
|---|---|
| 1 | `shoulder_pan` |
| 2 | `shoulder_lift` |
| 3 | `elbow_flex` |
| 4 | `wrist_flex` |
| 5 | `wrist_roll` |
| 6 | `gripper` |

Baud: `1000000`.

This joint order is the same order used by datasets, inference, and evaluation.
