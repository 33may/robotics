# Calibration Profile System & Interactive TUI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a calibration profile manager and an interactive TUI for precision keyframe-based joint calibration on the SO-ARM101 robot.

**Architecture:** Two modules in `vbti/logic/servos/`: `profiles.py` (profile CRUD + registry) and `calibrate_interactive.py` (curses-based TUI). Both use the LeRobot `FeetechMotorsBus` API for servo communication. Profiles are stored in the standard LeRobot cache with a project-local registry for metadata.

**Tech Stack:** Python 3.12, LeRobot FeetechMotorsBus API, curses (stdlib), numpy (for least-squares fit), fire (CLI)

**Spec:** `docs/superpowers/specs/2026-04-15-calibration-system-design.md`

---

## File Structure

```
vbti/logic/servos/
├── profiles.py                  # NEW — Profile management CLI
├── calibrate_interactive.py     # NEW — Interactive calibration TUI
├── load_calibration.py          # EXISTING — update to delegate to profiles
├── scan_all.py                  # EXISTING — no changes
└── factory_reset_motors.py      # EXISTING — no changes

calibration/
├── registry.json                # NEW — Profile metadata
└── profiles/                    # NEW — Version-controlled backups
```

---

### Task 1: Profile Storage & Registry

**Files:**
- Create: `vbti/logic/servos/profiles.py`
- Create: `calibration/registry.json`
- Create: `calibration/profiles/` (directory)

- [ ] **Step 1: Create the registry JSON with existing profiles**

Create `calibration/registry.json`:
```json
{
  "profiles": {
    "frodeo-test": {
      "description": "Primary test calibration",
      "created": "2026-03-15",
      "status": "active"
    },
    "frodeo": {
      "description": "Original calibration",
      "created": "2026-03-01",
      "status": "active"
    }
  },
  "default": "frodeo-test"
}
```

Create `calibration/profiles/` directory:
```bash
mkdir -p calibration/profiles
```

- [ ] **Step 2: Write `profiles.py` — constants and helpers**

Create `vbti/logic/servos/profiles.py`:
```python
#!/usr/bin/env python3
"""Manage calibration profiles for SO-ARM101.

Usage:
    python -m vbti.logic.servos.profiles list
    python -m vbti.logic.servos.profiles show <name>
    python -m vbti.logic.servos.profiles load <name> --port=/dev/ttyACM1
    python -m vbti.logic.servos.profiles export <name>
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# LeRobot cache location for follower calibration files
LEROBOT_CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"

# Project-local paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # vbti/logic/servos -> project root
REGISTRY_PATH = PROJECT_ROOT / "calibration" / "registry.json"
PROFILES_BACKUP_DIR = PROJECT_ROOT / "calibration" / "profiles"


def _calib_path(name: str) -> Path:
    """Path to a calibration JSON in the LeRobot cache."""
    return LEROBOT_CALIB_DIR / f"{name}.json"


def _load_registry() -> dict:
    """Load the registry, creating it from cache scan if missing."""
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())

    # Bootstrap: scan existing profiles in LeRobot cache
    registry = {"profiles": {}, "default": None}
    if LEROBOT_CALIB_DIR.exists():
        for f in sorted(LEROBOT_CALIB_DIR.glob("*.json")):
            registry["profiles"][f.stem] = {
                "description": "",
                "created": "",
                "status": "active",
            }
    if registry["profiles"] and not registry["default"]:
        registry["default"] = next(iter(registry["profiles"]))

    _save_registry(registry)
    return registry


def _save_registry(registry: dict) -> None:
    """Write the registry JSON."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")


def _load_calib_json(name: str) -> dict:
    """Load a calibration JSON from the LeRobot cache."""
    path = _calib_path(name)
    if not path.exists():
        raise FileNotFoundError(f"No calibration file: {path}")
    return json.loads(path.read_text())
```

- [ ] **Step 3: Write `profiles.py` — `list` command**

Append to `profiles.py`:
```python
def list_profiles() -> None:
    """Show all known calibration profiles."""
    registry = _load_registry()
    default = registry.get("default", "")
    print(f"{'Name':<20} {'Status':<10} {'Description':<40} {'Default'}")
    print("-" * 80)
    for name, info in registry["profiles"].items():
        is_default = "*" if name == default else ""
        exists = "✓" if _calib_path(name).exists() else "✗ missing"
        status = info.get("status", "?")
        desc = info.get("description", "")
        print(f"{name:<20} {status:<10} {desc:<40} {is_default}")
        if exists != "✓":
            print(f"  WARNING: {exists} at {_calib_path(name)}")
```

- [ ] **Step 4: Write `profiles.py` — `show` command**

Append to `profiles.py`:
```python
def show(name: str) -> None:
    """Print calibration values for a profile."""
    calib = _load_calib_json(name)
    print(f"Profile: {name}")
    print(f"{'Joint':<18} {'ID':>3} {'Drive':>6} {'Offset':>8} {'Min':>6} {'Max':>6}")
    print("-" * 55)
    for joint in JOINT_NAMES:
        if joint not in calib:
            print(f"{joint:<18} --- not in file ---")
            continue
        c = calib[joint]
        print(
            f"{joint:<18} {c['id']:>3} {c['drive_mode']:>6} "
            f"{c['homing_offset']:>8} {c['range_min']:>6} {c['range_max']:>6}"
        )
```

- [ ] **Step 5: Write `profiles.py` — `load` command**

Append to `profiles.py`:
```python
def load(name: str, port: str = "/dev/ttyACM1") -> None:
    """Write a calibration profile to motor EEPROM."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    calib_json = _load_calib_json(name)
    calibration = {}
    motors = {}
    for joint in JOINT_NAMES:
        c = calib_json[joint]
        calibration[joint] = MotorCalibration(
            id=c["id"],
            drive_mode=c["drive_mode"],
            homing_offset=c["homing_offset"],
            range_min=c["range_min"],
            range_max=c["range_max"],
        )
        norm = MotorNormMode.RANGE_0_100 if joint == "gripper" else MotorNormMode.DEGREES
        motors[joint] = Motor(c["id"], "sts3215", norm)

    bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
    bus.connect()

    print(f"Loading profile '{name}' to motors on {port}...")
    with bus.torque_disabled():
        bus.write_calibration(calibration)

    for joint, cal in calibration.items():
        print(
            f"  {joint:15s}  homing_offset={cal.homing_offset:5d}  "
            f"range=[{cal.range_min}, {cal.range_max}]"
        )

    bus.disconnect()
    print("Done.")
```

- [ ] **Step 6: Write `profiles.py` — `export` command**

Append to `profiles.py`:
```python
def export(name: str) -> None:
    """Copy a calibration JSON from LeRobot cache to calibration/profiles/ for version control."""
    src = _calib_path(name)
    if not src.exists():
        print(f"No calibration file found: {src}")
        return
    PROFILES_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = PROFILES_BACKUP_DIR / f"{name}.json"
    shutil.copy2(src, dst)
    print(f"Exported: {src} → {dst}")
```

- [ ] **Step 7: Write `profiles.py` — `register` helper and fire CLI**

Append to `profiles.py`:
```python
def register(name: str, description: str = "", parent: str = "") -> None:
    """Add or update a profile entry in the registry."""
    from datetime import date
    registry = _load_registry()
    entry = registry["profiles"].get(name, {})
    entry["description"] = description or entry.get("description", "")
    entry["created"] = entry.get("created", "") or date.today().isoformat()
    entry["status"] = "active"
    if parent:
        entry["parent"] = parent
    registry["profiles"][name] = entry
    _save_registry(registry)
    print(f"Registered profile '{name}' in registry.")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "list": list_profiles,
        "show": show,
        "load": load,
        "export": export,
        "register": register,
    })
```

- [ ] **Step 8: Test profiles CLI manually**

```bash
cd ~/projects/ml_portfolio/robotics
conda activate lerobot
python -m vbti.logic.servos.profiles list
python -m vbti.logic.servos.profiles show frodeo-test
python -m vbti.logic.servos.profiles export frodeo-test
ls calibration/profiles/
```

Expected: `list` shows frodeo-test and frodeo. `show` prints joint values. `export` copies to `calibration/profiles/frodeo-test.json`.

- [ ] **Step 9: Commit**

```bash
git add vbti/logic/servos/profiles.py calibration/registry.json calibration/profiles/
git commit -m "feat: add calibration profile management system"
```

---

### Task 2: Interactive Calibration TUI — Bus Helpers

**Files:**
- Create: `vbti/logic/servos/calibrate_interactive.py`

This task builds the servo communication layer that the TUI will use. No curses yet — just the functions that read/write servos.

- [ ] **Step 1: Create `calibrate_interactive.py` with imports and constants**

```python
#!/usr/bin/env python3
"""Interactive keyframe-based calibration TUI for SO-ARM101.

Usage:
    python -m vbti.logic.servos.calibrate_interactive --port=/dev/ttyACM1 --name=frodeo-v2
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .profiles import (
    JOINT_NAMES,
    LEROBOT_CALIB_DIR,
    _calib_path,
    _load_registry,
    _save_registry,
    register,
)

ENCODER_RESOLUTION = 4096
ENCODER_MAX = ENCODER_RESOLUTION - 1  # 4095
DEGREES_PER_TICK = 360.0 / ENCODER_MAX


@dataclass
class Keyframe:
    """A reference point: (raw encoder value, target degrees)."""
    raw_encoder: int
    target_degrees: float


@dataclass
class JointCalibState:
    """Calibration state for a single joint."""
    name: str
    motor_id: int
    # Phase 1: range discovery
    range_min: int | None = None
    range_max: int | None = None
    # Phase 2: keyframes
    keyframes: list[Keyframe] = field(default_factory=list)
    # Phase 3: computed calibration
    homing_offset: int | None = None
    accepted: bool = False
```

- [ ] **Step 2: Write bus connection helper**

Append to `calibrate_interactive.py`:
```python
def _make_bus(port: str) -> FeetechMotorsBus:
    """Create a FeetechMotorsBus for all 6 SO-ARM101 joints (no calibration)."""
    motors = {}
    for i, name in enumerate(JOINT_NAMES):
        norm = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES
        motors[name] = Motor(i + 1, "sts3215", norm)
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect()
    return bus


def read_raw_position(bus: FeetechMotorsBus, joint: str) -> int:
    """Read raw encoder value for one joint (no calibration applied)."""
    vals = bus.sync_read("Present_Position", [joint], normalize=False)
    return vals[joint]


def read_all_raw(bus: FeetechMotorsBus) -> dict[str, int]:
    """Read raw encoder values for all joints."""
    return bus.sync_read("Present_Position", normalize=False)
```

- [ ] **Step 3: Write the keyframe fitting function**

Append to `calibrate_interactive.py`:
```python
def fit_homing_offset(state: JointCalibState) -> int:
    """Compute homing_offset from keyframes via least-squares.

    The LeRobot DEGREES normalization is:
        degrees = (raw - homing_offset - midpoint) * 360.0 / 4095
    where midpoint = (range_min + range_max) / 2

    Solving for homing_offset per keyframe:
        homing_offset = raw - midpoint - (target_degrees / DEGREES_PER_TICK)

    With multiple keyframes, average for least-squares optimal.
    """
    if not state.keyframes:
        raise ValueError("Need at least 1 keyframe")
    if state.range_min is None or state.range_max is None:
        raise ValueError("Range not discovered yet")

    mid = (state.range_min + state.range_max) / 2.0
    offsets = []
    for kf in state.keyframes:
        offset = kf.raw_encoder - mid - (kf.target_degrees / DEGREES_PER_TICK)
        offsets.append(offset)

    return int(round(np.mean(offsets)))


def compute_residuals(state: JointCalibState) -> list[float]:
    """Compute degree residuals for each keyframe given current calibration."""
    if state.homing_offset is None or state.range_min is None or state.range_max is None:
        return []
    mid = (state.range_min + state.range_max) / 2.0
    residuals = []
    for kf in state.keyframes:
        actual_deg = (kf.raw_encoder - state.homing_offset - mid) * DEGREES_PER_TICK
        residuals.append(actual_deg - kf.target_degrees)
    return residuals


def raw_to_degrees(raw: int, state: JointCalibState) -> float | None:
    """Convert raw encoder to degrees using current calibration state."""
    if state.homing_offset is None or state.range_min is None or state.range_max is None:
        return None
    mid = (state.range_min + state.range_max) / 2.0
    return (raw - state.homing_offset - mid) * DEGREES_PER_TICK
```

- [ ] **Step 4: Write the save function**

Append to `calibrate_interactive.py`:
```python
def save_profile(name: str, states: dict[str, JointCalibState], description: str = "") -> Path:
    """Save calibration to LeRobot cache and update registry."""
    calib = {}
    for joint_name in JOINT_NAMES:
        s = states[joint_name]
        calib[joint_name] = {
            "id": s.motor_id,
            "drive_mode": 0,
            "homing_offset": s.homing_offset if s.homing_offset is not None else 0,
            "range_min": s.range_min if s.range_min is not None else 0,
            "range_max": s.range_max if s.range_max is not None else ENCODER_MAX,
        }

    out_path = _calib_path(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calib, indent=4) + "\n")
    print(f"\nSaved calibration to {out_path}")

    register(name, description=description)
    return out_path
```

- [ ] **Step 5: Commit**

```bash
git add vbti/logic/servos/calibrate_interactive.py
git commit -m "feat: add calibration TUI data model and fitting logic"
```

---

### Task 3: Interactive Calibration TUI — Curses Interface

**Files:**
- Modify: `vbti/logic/servos/calibrate_interactive.py`

- [ ] **Step 1: Write the joint selection screen**

Append to `calibrate_interactive.py`:
```python
import curses
import select
import sys


def _draw_joint_list(stdscr, states: dict[str, JointCalibState], selected: int, name: str):
    """Draw the joint selection screen."""
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(0, 0, f"─ Calibration: {name} ─", curses.A_BOLD)
    stdscr.addstr(1, 0, "")

    for i, joint in enumerate(JOINT_NAMES):
        s = states[joint]
        if s.accepted:
            marker = "✓"
        elif i == selected:
            marker = "●"
        else:
            marker = "○"
        status = "calibrated" if s.accepted else ("pending" if s.homing_offset is None else "fitted")
        line = f"  [{i+1}] {marker} {joint:<18} {status}"
        attr = curses.A_REVERSE if i == selected else 0
        if i + 2 < h:
            stdscr.addstr(i + 2, 0, line[:w-1], attr)

    footer_y = len(JOINT_NAMES) + 3
    if footer_y < h:
        stdscr.addstr(footer_y, 0, "  [Enter] Calibrate joint   [s] Save   [q] Quit")
    stdscr.refresh()
```

- [ ] **Step 2: Write Phase 1 — range discovery screen**

Append to `calibrate_interactive.py`:
```python
def _phase_range_discovery(stdscr, bus: FeetechMotorsBus, state: JointCalibState):
    """Phase 1: User moves joint through full range, we record min/max."""
    bus.disable_torque([state.name])
    seen_min = None
    seen_max = None

    stdscr.nodelay(True)
    stdscr.clear()

    while True:
        raw = read_raw_position(bus, state.name)
        if seen_min is None:
            seen_min = raw
            seen_max = raw
        seen_min = min(seen_min, raw)
        seen_max = max(seen_max, raw)

        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f"─ {state.name} ─ Range Discovery ─", curses.A_BOLD)
        stdscr.addstr(1, 0, "  Move joint through full range, press ENTER when done")
        stdscr.addstr(3, 0, f"  Raw encoder: {raw:>5}")
        stdscr.addstr(4, 0, f"  Min seen:    {seen_min:>5}")
        stdscr.addstr(5, 0, f"  Max seen:    {seen_max:>5}")
        stdscr.addstr(6, 0, f"  Range:       {seen_max - seen_min:>5} ticks  ({(seen_max - seen_min) * DEGREES_PER_TICK:.1f}°)")
        stdscr.addstr(8, 0, "  [Enter] Done   [r] Reset min/max")
        stdscr.refresh()

        key = stdscr.getch()
        if key == ord('\n') or key == curses.KEY_ENTER:
            break
        elif key == ord('r'):
            seen_min = raw
            seen_max = raw

        time.sleep(0.05)

    stdscr.nodelay(False)
    state.range_min = seen_min
    state.range_max = seen_max
```

- [ ] **Step 3: Write Phase 2 — keyframe marking screen**

Append to `calibrate_interactive.py`:
```python
def _prompt_float(stdscr, prompt: str, row: int) -> float | None:
    """Prompt user for a float value using curses. Returns None if cancelled."""
    curses.echo()
    stdscr.addstr(row, 0, f"  {prompt}: ")
    stdscr.refresh()
    try:
        inp = stdscr.getstr(row, len(prompt) + 4, 20).decode().strip()
        if not inp:
            return None
        return float(inp)
    except (ValueError, curses.error):
        return None
    finally:
        curses.noecho()


def _phase_keyframes(stdscr, bus: FeetechMotorsBus, state: JointCalibState):
    """Phase 2: User marks positions with target degree values."""
    bus.disable_torque([state.name])
    stdscr.nodelay(True)

    while True:
        raw = read_raw_position(bus, state.name)

        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f"─ {state.name} ─ Keyframes ─", curses.A_BOLD)
        stdscr.addstr(1, 0, f"  Raw encoder: {raw:>5} (live)")
        stdscr.addstr(2, 0, f"  Range: [{state.range_min}, {state.range_max}]")

        row = 4
        stdscr.addstr(row, 0, "  Keyframes:")
        row += 1
        for i, kf in enumerate(state.keyframes):
            if row < h - 3:
                stdscr.addstr(row, 0, f"    #{i+1}:  encoder {kf.raw_encoder:>5}  →  {kf.target_degrees:>8.1f}°")
                row += 1

        if not state.keyframes:
            stdscr.addstr(row, 0, "    (none yet)")
            row += 1

        row += 1
        if row < h:
            stdscr.addstr(row, 0, "  [m] Mark position  [d] Delete last  [f] Fit  [b] Back")
        stdscr.refresh()

        key = stdscr.getch()
        if key == ord('m'):
            stdscr.nodelay(False)
            current_raw = read_raw_position(bus, state.name)
            deg = _prompt_float(stdscr, "Target degrees", row + 1)
            if deg is not None:
                state.keyframes.append(Keyframe(raw_encoder=current_raw, target_degrees=deg))
            stdscr.nodelay(True)
        elif key == ord('d'):
            if state.keyframes:
                state.keyframes.pop()
        elif key == ord('f'):
            if len(state.keyframes) >= 1:
                break
            # Not enough keyframes — flash a message
            stdscr.nodelay(False)
            stdscr.addstr(row + 1, 0, "  Need at least 1 keyframe!")
            stdscr.refresh()
            time.sleep(1)
            stdscr.nodelay(True)
        elif key == ord('b'):
            return  # go back without fitting

        time.sleep(0.05)

    stdscr.nodelay(False)
    state.homing_offset = fit_homing_offset(state)
```

- [ ] **Step 4: Write Phase 3 — fit verification and nudge screen**

Append to `calibrate_interactive.py`:
```python
def _phase_verify(stdscr, bus: FeetechMotorsBus, state: JointCalibState):
    """Phase 3: Show fit results, allow nudging and torque-test."""
    torque_on = False
    stdscr.nodelay(True)

    while True:
        raw = read_raw_position(bus, state.name)
        residuals = compute_residuals(state)
        current_deg = raw_to_degrees(raw, state)

        stdscr.clear()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f"─ {state.name} ─ Verify Fit ─", curses.A_BOLD)
        stdscr.addstr(1, 0, f"  homing_offset: {state.homing_offset}   range: [{state.range_min}, {state.range_max}]")
        if current_deg is not None:
            stdscr.addstr(2, 0, f"  Live: raw={raw}  degrees={current_deg:.1f}°")
        else:
            stdscr.addstr(2, 0, f"  Live: raw={raw}")

        row = 4
        stdscr.addstr(row, 0, "  Keyframe residuals:")
        row += 1
        for i, (kf, res) in enumerate(zip(state.keyframes, residuals)):
            actual = kf.target_degrees + res
            if row < h - 4:
                stdscr.addstr(row, 0, f"    #{i+1}: {kf.target_degrees:>7.1f}° target → {actual:>7.1f}° actual  (Δ {res:>+.2f}°)")
                row += 1

        row += 1
        torque_str = "ON — type degrees to test" if torque_on else "off"
        if row < h:
            stdscr.addstr(row, 0, f"  Torque: {torque_str}")
        row += 1
        if row < h:
            stdscr.addstr(row, 0, "  [+/-] Nudge offset ±1   [t] Toggle torque   [a] Accept   [r] Redo keyframes")
        stdscr.refresh()

        key = stdscr.getch()
        if key == ord('+') or key == ord('='):
            state.homing_offset += 1
        elif key == ord('-') or key == ord('_'):
            state.homing_offset -= 1
        elif key == ord('t'):
            if torque_on:
                bus.disable_torque([state.name])
                torque_on = False
            else:
                # Enable torque and prompt for a degree command
                stdscr.nodelay(False)
                deg = _prompt_float(stdscr, "Go to degrees", row + 1)
                if deg is not None:
                    # Temporarily write calibration for this joint so
                    # the bus can unnormalize the degree command.
                    # Preserve any existing calibration for other joints.
                    if bus.calibration is None:
                        bus.calibration = {}
                    bus.calibration[state.name] = MotorCalibration(
                        id=state.motor_id,
                        drive_mode=0,
                        homing_offset=state.homing_offset,
                        range_min=state.range_min,
                        range_max=state.range_max,
                    )
                    bus.enable_torque([state.name])
                    bus.write("Goal_Position", state.name, deg)
                    torque_on = True
                stdscr.nodelay(True)
        elif key == ord('a'):
            if torque_on:
                bus.disable_torque([state.name])
            state.accepted = True
            return
        elif key == ord('r'):
            if torque_on:
                bus.disable_torque([state.name])
            state.keyframes.clear()
            state.homing_offset = None
            state.accepted = False
            return  # caller will re-enter keyframe phase

        time.sleep(0.05)

    stdscr.nodelay(False)
```

- [ ] **Step 5: Write the main TUI loop**

Append to `calibrate_interactive.py`:
```python
def _calibrate_joint(stdscr, bus: FeetechMotorsBus, state: JointCalibState):
    """Full calibration flow for one joint."""
    # Phase 1: Range discovery (skip if already done)
    if state.range_min is None:
        _phase_range_discovery(stdscr, bus, state)

    # Phase 2 + 3 loop: keyframes → fit → verify (may loop if user redoes)
    while not state.accepted:
        if not state.keyframes:
            _phase_keyframes(stdscr, bus, state)
            if state.homing_offset is None:
                return  # user pressed 'b' to go back
        _phase_verify(stdscr, bus, state)


def _tui_main(stdscr, port: str, name: str):
    """Main curses TUI loop."""
    curses.curs_set(0)

    # Initialize bus
    bus = _make_bus(port)

    # Initialize joint states (load existing profile if editing)
    states = {}
    existing_calib = None
    if _calib_path(name).exists():
        try:
            existing_calib = json.loads(_calib_path(name).read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    for i, joint in enumerate(JOINT_NAMES):
        motor_id = i + 1
        if existing_calib and joint in existing_calib:
            c = existing_calib[joint]
            states[joint] = JointCalibState(
                name=joint,
                motor_id=motor_id,
                range_min=c["range_min"],
                range_max=c["range_max"],
                homing_offset=c["homing_offset"],
                accepted=True,  # pre-loaded joints marked as done
            )
        else:
            states[joint] = JointCalibState(name=joint, motor_id=motor_id)

    selected = 0

    try:
        while True:
            _draw_joint_list(stdscr, states, selected, name)
            key = stdscr.getch()

            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(JOINT_NAMES) - 1:
                selected += 1
            elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')):
                selected = key - ord('1')
            elif key == ord('\n') or key == curses.KEY_ENTER:
                joint = JOINT_NAMES[selected]
                # Reset accepted state if re-entering
                states[joint].accepted = False
                _calibrate_joint(stdscr, bus, states[joint])
            elif key == ord('s'):
                # Save
                all_done = all(s.accepted for s in states.values())
                if not all_done:
                    pending = [j for j, s in states.items() if not s.accepted]
                    stdscr.addstr(len(JOINT_NAMES) + 5, 0, f"  Warning: uncalibrated joints: {', '.join(pending)}")
                    stdscr.addstr(len(JOINT_NAMES) + 6, 0, "  Press 's' again to save anyway, any other key to cancel")
                    stdscr.refresh()
                    confirm = stdscr.getch()
                    if confirm != ord('s'):
                        continue

                curses.endwin()
                save_profile(name, states)

                # Ask about writing to EEPROM
                resp = input("Write calibration to motors now? [y/n]: ").strip().lower()
                if resp == 'y':
                    from .profiles import load as load_profile
                    load_profile(name, port)

                return
            elif key == ord('q'):
                curses.endwin()
                resp = input("Quit without saving? [y/n]: ").strip().lower()
                if resp == 'y':
                    return
                # Re-enter curses
                stdscr = curses.initscr()
                curses.noecho()
                curses.cbreak()
                stdscr.keypad(True)
                curses.curs_set(0)
    finally:
        bus.disconnect(disable_torque=True)


def main(port: str = "/dev/ttyACM1", name: str = "new-profile"):
    """Launch the interactive calibration TUI.

    Args:
        port: Serial port for the robot arm.
        name: Profile name to create or edit.
    """
    curses.wrapper(lambda stdscr: _tui_main(stdscr, port, name))


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
```

- [ ] **Step 6: Test the TUI manually with the real robot**

```bash
cd ~/projects/ml_portfolio/robotics
conda activate lerobot
python -m vbti.logic.servos.calibrate_interactive --port=/dev/ttyACM1 --name=test-tui
```

Test:
1. Joint selection screen appears with arrow keys working
2. Press Enter on a joint → range discovery phase starts
3. Move joint, see min/max update live
4. Press Enter → keyframe phase
5. Move to a position, press `m`, enter `0` → keyframe recorded
6. Press `f` → fit computed, verify screen shows residuals
7. Press `a` → back to joint list with ✓
8. Press `q` → exits cleanly

- [ ] **Step 7: Commit**

```bash
git add vbti/logic/servos/calibrate_interactive.py
git commit -m "feat: add interactive keyframe-based calibration TUI"
```

---

### Task 4: Wire Up `__init__.py` and Update `load_calibration.py`

**Files:**
- Modify: `vbti/logic/servos/load_calibration.py`

- [ ] **Step 1: Update `load_calibration.py` to delegate to profiles**

Replace the body of `load_calibration.py`:
```python
#!/usr/bin/env python3
"""Load calibration from JSON file to motor EEPROM.

Delegates to profiles.load(). Kept for backward compatibility.

Usage:
    python -m vbti.logic.servos.load_calibration --port /dev/ttyACM1 --robot_id frodeo-test
"""
from .profiles import load


def main(port: str = "/dev/ttyACM1", robot_id: str = "frodeo-test"):
    load(name=robot_id, port=port)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
```

- [ ] **Step 2: Test backward compatibility**

```bash
python -m vbti.logic.servos.load_calibration --port=/dev/ttyACM1 --robot_id=frodeo-test
```

Expected: same behavior as before — loads calibration to motors.

- [ ] **Step 3: Commit**

```bash
git add vbti/logic/servos/load_calibration.py
git commit -m "refactor: delegate load_calibration to profiles module"
```

---

### Task 5: End-to-End Calibration Test

This is a manual integration test — create a new profile using the TUI and verify it works.

- [ ] **Step 1: Create a new calibration profile via TUI**

```bash
python -m vbti.logic.servos.calibrate_interactive --port=/dev/ttyACM1 --name=frodeo-v2
```

For each joint:
1. Move through full range → record limits
2. Mark keyframe at 0° position (match sim visually)
3. Mark 1-2 more keyframes at known angles
4. Fit and verify residuals are < 1°
5. Accept

Save when all 6 joints are done.

- [ ] **Step 2: Verify profile appears in registry**

```bash
python -m vbti.logic.servos.profiles list
python -m vbti.logic.servos.profiles show frodeo-v2
```

- [ ] **Step 3: Load profile and test with goto**

```bash
python -m vbti.logic.servos.profiles load frodeo-v2 --port=/dev/ttyACM1
python vbti/logic/dataset/replay_utils.py goto --port=/dev/ttyACM1 --robot_id=frodeo-v2 --wrist_roll=0
```

Verify: wrist_roll=0 now produces vertical gripper orientation.

- [ ] **Step 4: Verify old profile still works**

```bash
python -m vbti.logic.servos.profiles load frodeo-test --port=/dev/ttyACM1
python vbti/logic/dataset/replay_utils.py goto --port=/dev/ttyACM1 --robot_id=frodeo-test --wrist_roll=0
```

Verify: wrist_roll=0 produces horizontal gripper (old behavior).

- [ ] **Step 5: Export and commit new profile**

```bash
python -m vbti.logic.servos.profiles export frodeo-v2
git add calibration/
git commit -m "feat: add frodeo-v2 sim-matched calibration profile"
```
