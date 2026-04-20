# Calibration Profile System & Interactive Calibration TUI

**Date:** 2026-04-15
**Status:** Draft
**Scope:** Profile management + interactive keyframe-based calibration tool. Dataset transform is deferred to a future session.

---

## Problem

The SO-ARM101's joint zero positions don't precisely match the simulation. Each joint is off by up to ~10 degrees. The standard LeRobot calibration ("hold at middle, press ENTER") is too imprecise for sim-to-real alignment. We need:

1. A way to manage multiple calibration profiles (old models use old calibration, new models use new)
2. A precision calibration tool that lets you define exact joint positions via keyframes

## Constraints

- Old calibration profiles (`frodeo-test`, `frodeo`) must remain untouched and functional
- Profiles must work with existing LeRobot infrastructure (same JSON format, same cache location)
- The TUI must work over SSH (no browser/GUI dependencies)
- Feetech STS3215 servos, 12-bit encoder (4096 steps), protocol 0

---

## Part 1: Calibration Profile System

### Storage

Profiles remain in the LeRobot cache where the framework expects them:
```
~/.cache/huggingface/lerobot/calibration/robots/so101_follower/{name}.json
```

### Registry

A project-local registry at `calibration/registry.json` tracks metadata:
```json
{
  "profiles": {
    "frodeo-test": {
      "description": "Original calibration, horizontal gripper at wrist_roll=0",
      "created": "2026-03-15",
      "status": "active"
    },
    "frodeo-v2": {
      "description": "Sim-matched calibration, vertical gripper at wrist_roll=0",
      "created": "2026-04-15",
      "parent": "frodeo-test",
      "status": "active"
    }
  },
  "default": "frodeo-v2"
}
```

Fields:
- `description`: Free-text purpose of this profile
- `created`: Date created
- `parent`: Optional, which profile this was derived from
- `status`: `active` or `archived`

### CLI — `vbti.logic.servos.profiles`

| Command | Action |
|---------|--------|
| `profiles list` | Show all profiles from registry with status |
| `profiles load <name> --port=PORT` | Write profile to motor EEPROM (replaces current `load_calibration.py`) |
| `profiles show <name>` | Print calibration values per joint |
| `profiles export <name>` | Copy JSON from cache to `calibration/profiles/` for version control |

All commands use `python -m vbti.logic.servos.profiles <command>`.

The `load` command:
1. Reads JSON from LeRobot cache
2. Connects to motors via FeetechMotorsBus
3. Disables torque
4. Writes homing_offset, range_min, range_max per motor
5. Prints verification readback

### Registering existing profiles

On first run of `profiles list`, if `calibration/registry.json` doesn't exist, scan the LeRobot cache directory for existing `.json` files and create the registry with them (status=active, no description — user can fill in later).

---

## Part 2: Interactive Calibration TUI

### Entry point

```bash
python -m vbti.logic.servos.calibrate_interactive --port=/dev/ttyACM1 --name=frodeo-v2
```

If `--name` already exists, loads it as starting point (edit mode). Otherwise starts fresh.

### Technology

Plain Python with standard library only. Terminal control via `curses` or raw ANSI escape codes. No external TUI framework dependency.

### Joint Selection Screen

```
┌─ Calibration: frodeo-v2 ──────────────────────┐
│  [1] shoulder_pan    ✓ calibrated              │
│  [2] shoulder_lift   ✓ calibrated              │
│  [3] elbow_flex      ● active                  │
│  [4] wrist_flex      ○ pending                 │
│  [5] wrist_roll      ○ pending                 │
│  [6] gripper         ○ pending                 │
│                                                │
│  [s] Save profile   [q] Quit without saving    │
└────────────────────────────────────────────────┘
```

Status per joint:
- `○ pending` — not yet calibrated in this session
- `● active` — currently selected
- `✓ calibrated` — keyframes set and fit accepted

### Per-Joint Flow

#### Phase 1: Range Discovery

Torque disabled on the selected joint. User physically moves the joint through its full range of motion.

```
┌─ elbow_flex ─ Range Discovery ────────────────┐
│  Move joint through full range, press ENTER    │
│                                                │
│  Raw encoder: 2051 (live)                      │
│  Min seen:     841                             │
│  Max seen:    3066                             │
│                                                │
│  [ENTER] Done   [r] Reset min/max              │
└────────────────────────────────────────────────┘
```

Continuously reads Present_Position and tracks min/max.

#### Phase 2: Keyframe Marking

User physically moves the joint to known positions and marks them with target degree values.

```
┌─ elbow_flex ─ Keyframes ──────────────────────┐
│  Raw encoder: 2051 (live)                      │
│                                                │
│  Keyframes:                                    │
│    #1:  encoder 2051  →   0.0°                 │
│    #2:  encoder 3045  →  89.5°                 │
│    #3:  encoder  841  → -106.0°                │
│                                                │
│  [m] Mark current position                     │
│  [d] Delete keyframe                           │
│  [f] Fit calibration                           │
│  [b] Back to joint list                        │
└────────────────────────────────────────────────┘
```

On `m`: prompt for target degrees (e.g., `0`, `-10`, `89.5`). Records (raw_encoder, target_degrees) pair.

Minimum 2 keyframes required to fit. More keyframes = better precision via least-squares.

#### Phase 3: Fit & Verify

Computes `homing_offset`, `range_min`, `range_max` from keyframes using least-squares fit.

```
┌─ elbow_flex ─ Fit Result ─────────────────────┐
│  homing_offset: 1099    range: [841, 3066]     │
│                                                │
│  Keyframe residuals:                           │
│    #1:   0.0° target →  0.3° actual  (Δ 0.3°) │
│    #2:  89.5° target → 89.1° actual  (Δ 0.4°) │
│    #3: -106.0° target→-106.2° actual (Δ 0.2°) │
│                                                │
│  [t] Toggle torque (test by commanding degrees)│
│  [+/-] Nudge homing_offset ±1 tick             │
│  [a] Accept   [r] Redo keyframes               │
└────────────────────────────────────────────────┘
```

**Torque test mode (`t`):** Enables torque, prompts for a degree value, commands the joint to move there. User visually verifies. Press `t` again to disable torque and return to adjustment.

**Nudge (`+/-`):** Fine-adjusts homing_offset by ±1 encoder tick (~0.088°). Residuals update live.

**Accept (`a`):** Stores calibration for this joint, returns to joint selection.

### Fitting Algorithm

Given keyframes `[(encoder_1, degrees_1), (encoder_2, degrees_2), ...]` and discovered `range_min`, `range_max`:

The LeRobot DEGREES normalization (from `motors_bus.py`):
```python
# Normalize (encoder → degrees):
mid = (range_min + range_max) / 2
degrees = (reported_position - mid) * 360.0 / 4095

# Where:
reported_position = raw_position - homing_offset

# So full chain:
degrees = (raw_position - homing_offset - mid) * 360.0 / 4095
```

Given keyframes `[(raw_encoder_i, target_degrees_i), ...]` and `range_min`, `range_max` from Phase 1:

```
mid = (range_min + range_max) / 2
scale = 360.0 / 4095

For each keyframe:
  target_degrees_i = (raw_encoder_i - homing_offset - mid) * scale

Solving for homing_offset:
  homing_offset = raw_encoder_i - mid - (target_degrees_i / scale)
```

With 1 keyframe: exact solution. With 2+: least-squares average (since `homing_offset` is a single scalar, multiple keyframes overdetermine it — the mean minimizes total residual).

`range_min` and `range_max` come from Phase 1 (physical range discovery). `homing_offset` is then fitted from keyframes. If the user marks keyframes at the extremes (e.g., "this encoder position = my max degrees"), the tool can optionally adjust `range_min`/`range_max` to match, then re-fit `homing_offset`.

### Save

On `s` from the joint selection screen:
1. Write JSON to `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/{name}.json`
2. Update `calibration/registry.json` (add or update entry)
3. Optionally write EEPROM immediately: "Write to motors now? [y/n]"

---

## File Structure

```
vbti/logic/servos/
├── profiles.py                  # Profile management CLI (list, load, show, export)
├── calibrate_interactive.py     # Interactive TUI
├── load_calibration.py          # Existing (kept for backward compat, delegates to profiles.load)
├── scan_all.py                  # Existing
└── factory_reset_motors.py      # Existing

calibration/
├── registry.json                # Profile metadata registry
└── profiles/                    # Version-controlled backups of calibration JSONs
    ├── frodeo-test.json
    └── frodeo-v2.json
```

---

## Out of Scope (Deferred)

- Dataset transform utility (remap datasets between calibration profiles)
- Leader arm calibration (this spec covers follower only)
- Automatic sim-matching (computing offsets from URDF/MJCF programmatically)
