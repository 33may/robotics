"""Interactive servo calibration with curses TUI.

Bus helpers, data model, fitting logic, save, and TUI screens.
"""
from __future__ import annotations

import curses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from vbti.logic.servos.profiles import (
    JOINT_NAMES,
    _calib_path,
    register,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODER_RESOLUTION = 4096
ENCODER_MAX = ENCODER_RESOLUTION - 1  # 4095
DEGREES_PER_TICK = 360.0 / ENCODER_MAX

# Motor ID mapping for SO-ARM101
MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    raw_encoder: int
    target_degrees: float


@dataclass
class JointCalibState:
    name: str
    motor_id: int
    range_min: int | None = None
    range_max: int | None = None
    keyframes: list[Keyframe] = field(default_factory=list)
    homing_offset: int | None = None
    accepted: bool = False


# ---------------------------------------------------------------------------
# Bus helpers
# ---------------------------------------------------------------------------

def _make_bus(port: str) -> FeetechMotorsBus:
    """Create a FeetechMotorsBus for all 6 SO-ARM101 joints (no calibration)."""
    motors = {}
    for joint in JOINT_NAMES:
        norm = MotorNormMode.RANGE_0_100 if joint == "gripper" else MotorNormMode.DEGREES
        motors[joint] = Motor(id=MOTOR_IDS[joint], model="sts3215", norm_mode=norm)
    return FeetechMotorsBus(port=port, motors=motors)


def read_raw_position(bus: FeetechMotorsBus, joint: str) -> int:
    """Read raw encoder value for a single joint."""
    result = bus.sync_read("Present_Position", [joint], normalize=False)
    return int(result[joint])


def read_all_raw(bus: FeetechMotorsBus) -> dict[str, int]:
    """Read raw encoder values for all joints."""
    result = bus.sync_read("Present_Position", normalize=False)
    return {name: int(result[name]) for name in JOINT_NAMES}


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def fit_homing_offset(state: JointCalibState) -> int:
    """Compute homing_offset from keyframes using least-squares averaging.

    LeRobot DEGREES normalization:
        degrees = (raw - homing_offset - midpoint) * 360.0 / 4095
    Solving for homing_offset per keyframe:
        homing_offset = raw - midpoint - (target_degrees / DEGREES_PER_TICK)
    With multiple keyframes we average (least-squares optimal for a single parameter).
    """
    if state.range_min is None or state.range_max is None:
        raise ValueError(f"range_min/range_max not set for joint '{state.name}'")
    if not state.keyframes:
        raise ValueError(f"No keyframes for joint '{state.name}'")

    midpoint = (state.range_min + state.range_max) / 2.0
    offsets = []
    for kf in state.keyframes:
        offset = kf.raw_encoder - midpoint - (kf.target_degrees / DEGREES_PER_TICK)
        offsets.append(offset)
    return int(round(np.mean(offsets)))


def compute_residuals(state: JointCalibState) -> list[float]:
    """Compute degree error per keyframe given current calibration values."""
    if state.homing_offset is None or state.range_min is None or state.range_max is None:
        raise ValueError("Calibration incomplete — need homing_offset, range_min, range_max")

    midpoint = (state.range_min + state.range_max) / 2.0
    residuals = []
    for kf in state.keyframes:
        predicted = (kf.raw_encoder - state.homing_offset - midpoint) * DEGREES_PER_TICK
        residuals.append(predicted - kf.target_degrees)
    return residuals


def raw_to_degrees(raw: int, state: JointCalibState) -> float | None:
    """Convert raw encoder to degrees using current calibration state.

    Returns None if calibration is incomplete.
    """
    if state.homing_offset is None or state.range_min is None or state.range_max is None:
        return None
    midpoint = (state.range_min + state.range_max) / 2.0
    return (raw - state.homing_offset - midpoint) * DEGREES_PER_TICK


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_profile(
    name: str,
    states: dict[str, JointCalibState],
    description: str = "",
) -> Path:
    """Build calibration JSON from states and save to LeRobot cache + registry."""
    calib_data: dict[str, dict] = {}
    for joint, st in states.items():
        if st.homing_offset is None or st.range_min is None or st.range_max is None:
            raise ValueError(f"Joint '{joint}' calibration incomplete")
        calib_data[joint] = {
            "id": st.motor_id,
            "drive_mode": 0,
            "homing_offset": st.homing_offset,
            "range_min": st.range_min,
            "range_max": st.range_max,
        }

    out_path = _calib_path(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calib_data, indent=4) + "\n")

    register(name, description)
    return out_path


# ---------------------------------------------------------------------------
# TUI helpers
# ---------------------------------------------------------------------------

def _draw_joint_list(
    stdscr,
    states: dict[str, JointCalibState],
    selected: int,
    name: str,
) -> None:
    """Draw the joint selection screen."""
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    title = f"─ Calibration: {name} ─"
    stdscr.addstr(0, max(0, (w - len(title)) // 2), title, curses.A_BOLD)

    row = 2
    for i, jname in enumerate(JOINT_NAMES):
        st = states[jname]
        if st.accepted:
            marker = "✓"
        elif i == selected:
            marker = "●"
        else:
            marker = "○"
        label = f"  {marker} {jname:18s}"
        if st.accepted and st.homing_offset is not None:
            label += f"  offset={st.homing_offset}  range=[{st.range_min}, {st.range_max}]"
        attr = curses.A_REVERSE if i == selected else 0
        stdscr.addstr(row + i, 1, label[:w - 2], attr)

    footer = "[Enter] Calibrate joint   [s] Save   [q] Quit"
    if row + len(JOINT_NAMES) + 2 < h:
        stdscr.addstr(row + len(JOINT_NAMES) + 2, 1, footer)
    stdscr.refresh()


# ---------------------------------------------------------------------------
# Phase 1 — Range discovery
# ---------------------------------------------------------------------------

def _phase_range_discovery(stdscr, bus: FeetechMotorsBus, state: JointCalibState) -> None:
    """Move joint through full range to discover min/max encoder values."""
    bus.disable_torque([state.name])
    stdscr.nodelay(True)
    raw = read_raw_position(bus, state.name)
    mn, mx = raw, raw

    try:
        while True:
            raw = read_raw_position(bus, state.name)
            mn = min(mn, raw)
            mx = max(mx, raw)

            stdscr.clear()
            stdscr.addstr(0, 0, f"── Phase 1: Range Discovery — {state.name} ──", curses.A_BOLD)
            stdscr.addstr(2, 2, "Move joint through full range, press ENTER when done")
            stdscr.addstr(4, 2, f"Raw encoder:  {raw:5d}")
            stdscr.addstr(5, 2, f"Min seen:     {mn:5d}")
            stdscr.addstr(6, 2, f"Max seen:     {mx:5d}")
            rng = mx - mn
            stdscr.addstr(7, 2, f"Range:        {rng:5d} ticks  ({rng * DEGREES_PER_TICK:.1f}°)")
            stdscr.addstr(9, 2, "[Enter] Accept   [r] Reset min/max")
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                state.range_min = mn
                state.range_max = mx
                break
            elif key in (ord("r"), ord("R")):
                raw = read_raw_position(bus, state.name)
                mn, mx = raw, raw

            time.sleep(0.05)
    finally:
        stdscr.nodelay(False)


# ---------------------------------------------------------------------------
# Input helper
# ---------------------------------------------------------------------------

def _prompt_float(stdscr, prompt: str, row: int) -> float | None:
    """Prompt user for a float value. Returns None if empty or invalid."""
    curses.echo()
    try:
        stdscr.addstr(row, 2, prompt)
        stdscr.clrtoeol()
        stdscr.refresh()
        raw = stdscr.getstr(row, 2 + len(prompt), 20)
        text = raw.decode().strip()
        if not text:
            return None
        return float(text)
    except (ValueError, UnicodeDecodeError):
        return None
    finally:
        curses.noecho()


# ---------------------------------------------------------------------------
# Phase 2 — Keyframe collection
# ---------------------------------------------------------------------------

def _phase_keyframes(stdscr, bus: FeetechMotorsBus, state: JointCalibState) -> None:
    """Collect keyframes mapping encoder positions to known degree values."""
    bus.disable_torque([state.name])
    stdscr.nodelay(True)

    try:
        while True:
            raw = read_raw_position(bus, state.name)

            stdscr.clear()
            stdscr.addstr(0, 0, f"── Phase 2: Keyframes — {state.name} ──", curses.A_BOLD)
            stdscr.addstr(2, 2, f"Raw encoder:  {raw:5d}")
            stdscr.addstr(3, 2, f"Range:        [{state.range_min}, {state.range_max}]")
            stdscr.addstr(5, 2, "Keyframes:")

            row = 6
            for i, kf in enumerate(state.keyframes):
                stdscr.addstr(row + i, 4, f"{i + 1}. encoder={kf.raw_encoder:5d}  → {kf.target_degrees:.1f}°")
            row += max(len(state.keyframes), 1)

            row += 1
            stdscr.addstr(row, 2, "[m] Mark position  [d] Delete last  [f] Fit & verify  [b] Back")
            stdscr.refresh()

            key = stdscr.getch()
            if key == ord("m"):
                stdscr.nodelay(False)
                raw_now = read_raw_position(bus, state.name)
                deg = _prompt_float(stdscr, f"Target degrees for encoder {raw_now}: ", row + 2)
                if deg is not None:
                    state.keyframes.append(Keyframe(raw_encoder=raw_now, target_degrees=deg))
                stdscr.nodelay(True)
            elif key == ord("d"):
                if state.keyframes:
                    state.keyframes.pop()
            elif key == ord("f"):
                if state.keyframes:
                    state.homing_offset = fit_homing_offset(state)
                    break
            elif key == ord("b"):
                break

            time.sleep(0.05)
    finally:
        stdscr.nodelay(False)


# ---------------------------------------------------------------------------
# Phase 3 — Verify and fine-tune
# ---------------------------------------------------------------------------

def _set_joint_calib_on_bus(bus: FeetechMotorsBus, state: JointCalibState) -> None:
    """Write a single joint's calibration to the bus (for torque commands)."""
    assert state.homing_offset is not None
    assert state.range_min is not None
    assert state.range_max is not None
    if bus.calibration is None:
        bus.calibration = {}
    bus.calibration[state.name] = MotorCalibration(
        id=state.motor_id,
        drive_mode=0,
        homing_offset=state.homing_offset,
        range_min=state.range_min,
        range_max=state.range_max,
    )


def _prompt_global_pose(
    stdscr,
    bus: FeetechMotorsBus,
    all_states: dict[str, JointCalibState],
) -> None:
    """Enable torque on all calibrated joints, prompt target degrees for each."""
    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.addstr(0, 0, "── Global Pose Test ──", curses.A_BOLD)
    stdscr.addstr(2, 2, "Enter target degrees per joint (empty = skip, 'q' = cancel):")

    targets: dict[str, float] = {}
    row = 4
    for jname in JOINT_NAMES:
        st = all_states[jname]
        calibrated = st.homing_offset is not None and st.range_min is not None
        status = f"(offset={st.homing_offset})" if calibrated else "(not calibrated — skip)"
        stdscr.addstr(row, 2, f"{jname:18s} {status}")
        if calibrated:
            curses.echo()
            stdscr.addstr(row, 55, "→ ")
            stdscr.refresh()
            try:
                raw_input = stdscr.getstr(row, 57, 10).decode().strip()
            except (curses.error, UnicodeDecodeError):
                raw_input = ""
            curses.noecho()
            if raw_input.lower() == "q":
                return
            if raw_input:
                try:
                    targets[jname] = float(raw_input)
                except ValueError:
                    pass
        row += 1

    if not targets:
        stdscr.addstr(row + 1, 2, "No targets — press any key")
        stdscr.refresh()
        stdscr.getch()
        return

    # Set calibration and command all target joints
    for jname, deg_val in targets.items():
        _set_joint_calib_on_bus(bus, all_states[jname])

    target_joints = list(targets.keys())
    bus.enable_torque(target_joints)
    for jname, deg_val in targets.items():
        bus.write("Goal_Position", jname, deg_val)

    row += 2
    stdscr.addstr(row, 2, "Holding pose — press any key to release")
    stdscr.refresh()
    stdscr.getch()

    bus.disable_torque(target_joints)


def _phase_verify(
    stdscr,
    bus: FeetechMotorsBus,
    state: JointCalibState,
    all_states: dict[str, JointCalibState],
) -> None:
    """Verify calibration with live readout, tweak offset, and test torque."""
    stdscr.nodelay(True)
    torque_on = False

    try:
        while True:
            raw = read_raw_position(bus, state.name)
            deg = raw_to_degrees(raw, state)

            stdscr.clear()
            stdscr.addstr(0, 0, f"── Phase 3: Verify — {state.name} ──", curses.A_BOLD)
            stdscr.addstr(2, 2, f"Homing offset: {state.homing_offset}")
            stdscr.addstr(3, 2, f"Range:         [{state.range_min}, {state.range_max}]")
            stdscr.addstr(4, 2, f"Raw encoder:   {raw:5d}")
            deg_str = f"{deg:.2f}°" if deg is not None else "N/A"
            stdscr.addstr(5, 2, f"Degrees:       {deg_str}")
            stdscr.addstr(6, 2, f"Torque:        {'ON' if torque_on else 'off'}")

            # Residuals table
            row = 8
            if state.keyframes and state.homing_offset is not None:
                residuals = compute_residuals(state)
                stdscr.addstr(row, 2, f"{'#':>3}  {'Target':>8}  {'Actual':>8}  {'Delta':>8}")
                row += 1
                for i, (kf, res) in enumerate(zip(state.keyframes, residuals)):
                    actual = kf.target_degrees + res
                    stdscr.addstr(row, 2, f"{i+1:>3}  {kf.target_degrees:>8.1f}  {actual:>8.1f}  {res:>+8.2f}")
                    row += 1

            row += 1
            stdscr.addstr(row, 2, "[+/-] Tweak offset  [t] Joint torque  [g] Global pose  [a] Accept  [r] Redo")
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("+"), ord("=")):
                if state.homing_offset is not None:
                    state.homing_offset += 1
            elif key in (ord("-"), ord("_")):
                if state.homing_offset is not None:
                    state.homing_offset -= 1
            elif key == ord("t"):
                if torque_on:
                    bus.disable_torque([state.name])
                    torque_on = False
                else:
                    stdscr.nodelay(False)
                    deg_val = _prompt_float(stdscr, "Go to degrees: ", row + 2)
                    if deg_val is not None:
                        _set_joint_calib_on_bus(bus, state)
                        bus.enable_torque([state.name])
                        bus.write("Goal_Position", state.name, deg_val)
                        torque_on = True
                    stdscr.nodelay(True)
            elif key == ord("g"):
                if torque_on:
                    bus.disable_torque([state.name])
                    torque_on = False
                stdscr.nodelay(False)
                _prompt_global_pose(stdscr, bus, all_states)
                stdscr.nodelay(True)
            elif key == ord("a"):
                if torque_on:
                    bus.disable_torque([state.name])
                    torque_on = False
                state.accepted = True
                return
            elif key == ord("r"):
                if torque_on:
                    bus.disable_torque([state.name])
                    torque_on = False
                state.keyframes.clear()
                state.homing_offset = None
                state.accepted = False
                return

            time.sleep(0.05)
    finally:
        stdscr.nodelay(False)


# ---------------------------------------------------------------------------
# Joint calibration flow
# ---------------------------------------------------------------------------

def _calibrate_joint(
    stdscr,
    bus: FeetechMotorsBus,
    state: JointCalibState,
    all_states: dict[str, JointCalibState],
) -> None:
    """Run the full calibration flow for a single joint."""
    if state.range_min is None:
        _phase_range_discovery(stdscr, bus, state)

    while not state.accepted:
        if not state.keyframes:
            _phase_keyframes(stdscr, bus, state)
            # User pressed 'b' to go back without fitting
            if state.homing_offset is None:
                return
        _phase_verify(stdscr, bus, state, all_states)


# ---------------------------------------------------------------------------
# Main TUI loop
# ---------------------------------------------------------------------------

def _tui_main(stdscr, port: str, name: str) -> None:
    """Main curses loop — joint selection, calibration, save."""
    curses.curs_set(0)
    bus = _make_bus(port)
    bus.connect()

    # Initialize states
    states: dict[str, JointCalibState] = {}
    existing_data = {}
    calib_file = _calib_path(name)
    if calib_file.exists():
        existing_data = json.loads(calib_file.read_text())

    for jname in JOINT_NAMES:
        st = JointCalibState(name=jname, motor_id=MOTOR_IDS[jname])
        if jname in existing_data:
            j = existing_data[jname]
            st.homing_offset = j.get("homing_offset")
            st.range_min = j.get("range_min")
            st.range_max = j.get("range_max")
            st.accepted = True
        states[jname] = st

    selected = 0

    try:
        while True:
            _draw_joint_list(stdscr, states, selected, name)
            key = stdscr.getch()

            if key == curses.KEY_UP:
                selected = (selected - 1) % len(JOINT_NAMES)
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % len(JOINT_NAMES)
            elif ord("1") <= key <= ord("6"):
                selected = key - ord("1")
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                jname = JOINT_NAMES[selected]
                states[jname].accepted = False
                _calibrate_joint(stdscr, bus, states[jname], states)
            elif key == ord("s"):
                # Check for uncalibrated joints
                uncalibrated = [
                    j for j, st in states.items()
                    if not st.accepted or st.homing_offset is None
                ]
                if uncalibrated:
                    stdscr.addstr(
                        len(JOINT_NAMES) + 5, 1,
                        f"Warning: {len(uncalibrated)} joints uncalibrated. Press 's' again to save anyway.",
                    )
                    stdscr.refresh()
                    confirm = stdscr.getch()
                    if confirm != ord("s"):
                        continue

                curses.endwin()
                # Only save joints that have calibration
                saveable = {
                    j: st for j, st in states.items()
                    if st.homing_offset is not None and st.range_min is not None
                }
                if not saveable:
                    print("No calibrated joints to save.")
                    return
                out = save_profile(name, saveable)
                print(f"Saved to {out}")
                ans = input("Write to motors now? [y/n] ").strip().lower()
                if ans == "y":
                    from vbti.logic.servos.profiles import load as load_profile
                    load_profile(name, port=port)
                return
            elif key == ord("q"):
                curses.endwin()
                ans = input("Quit without saving? [y/n] ").strip().lower()
                if ans == "y":
                    return
                # Re-init curses
                stdscr = curses.initscr()
                curses.noecho()
                curses.cbreak()
                stdscr.keypad(True)
                curses.curs_set(0)
    finally:
        bus.disconnect(disable_torque=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(port: str = "/dev/ttyACM1", name: str = "new-profile") -> None:
    """Launch the interactive calibration TUI."""
    curses.wrapper(lambda stdscr: _tui_main(stdscr, port, name))


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
