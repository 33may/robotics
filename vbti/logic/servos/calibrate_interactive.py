"""Interactive servo calibration with curses TUI.

Flow:
  Phase 1 — Range discovery (all joints, like LeRobot)
  Phase 2 — Zero tuning (per joint: torque on at 0°, nudge +/- until sim-accurate)
  Save   — Adjust ranges for final offsets, write JSON + EEPROM
"""
from __future__ import annotations

import _curses
import curses
import json
import time
from dataclasses import dataclass

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from vbti.logic.servos.profiles import (
    JOINT_NAMES,
    _calib_path,
    register,
)

ENCODER_MAX = 4095
HALF_TURN = 2047

MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
    "wrist_flex": 4, "wrist_roll": 5, "gripper": 6,
}


@dataclass
class JointState:
    name: str
    motor_id: int
    # Raw physical range (discovered with offset=0)
    phys_min: int | None = None
    phys_max: int | None = None
    # Homing offset (tuned in phase 2)
    homing_offset: int | None = None
    zero_tuned: bool = False


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

def _make_bus(port: str) -> FeetechMotorsBus:
    motors = {}
    for joint in JOINT_NAMES:
        norm = MotorNormMode.RANGE_0_100 if joint == "gripper" else MotorNormMode.DEGREES
        motors[joint] = Motor(id=MOTOR_IDS[joint], model="sts3215", norm_mode=norm)
    return FeetechMotorsBus(port=port, motors=motors)


def _reset_all(bus: FeetechMotorsBus) -> None:
    """Reset all servos: offset=0, limits=full range."""
    bus.disable_torque()
    for joint in JOINT_NAMES:
        bus.write("Homing_Offset", joint, 0, normalize=False)
        bus.write("Min_Position_Limit", joint, 0, normalize=False)
        bus.write("Max_Position_Limit", joint, ENCODER_MAX, normalize=False)
    bus.calibration = {}
    time.sleep(0.1)


def _write_offset(bus: FeetechMotorsBus, joint: str, offset: int) -> None:
    """Write homing_offset to EEPROM (torque must be off)."""
    bus.disable_torque([joint])
    bus.write("Homing_Offset", joint, offset)
    time.sleep(0.05)


def _command_degrees(bus: FeetechMotorsBus, joint: str, state: JointState, deg: float) -> None:
    """Set calibration on bus and command joint to a degree position."""
    assert state.homing_offset is not None and state.phys_min is not None and state.phys_max is not None
    # Compute range in reported space (after offset)
    rmin = state.phys_min - state.homing_offset
    rmax = state.phys_max - state.homing_offset
    cal = MotorCalibration(
        id=state.motor_id, drive_mode=0,
        homing_offset=state.homing_offset, range_min=rmin, range_max=rmax,
    )
    if bus.calibration is None:
        bus.calibration = {}
    bus.calibration[state.name] = cal
    bus.enable_torque([joint])
    bus.write("Goal_Position", joint, deg)


# ---------------------------------------------------------------------------
# Phase 1 — Range discovery (all joints at once)
# ---------------------------------------------------------------------------

def _phase_range(stdscr, bus: FeetechMotorsBus, states: dict[str, JointState]) -> None:
    """Sweep all joints by hand to discover physical min/max."""
    _reset_all(bus)

    mins = {j: 99999 for j in JOINT_NAMES}
    maxs = {j: 0 for j in JOINT_NAMES}

    stdscr.nodelay(True)
    try:
        while True:
            positions = bus.sync_read("Present_Position", normalize=False)

            stdscr.clear()
            stdscr.addstr(0, 0, "── Phase 1: Range Discovery (all joints) ──", curses.A_BOLD)
            stdscr.addstr(2, 2, "Move ALL joints through their full range of motion.")
            stdscr.addstr(3, 2, "Press ENTER when done.")

            row = 5
            for joint in JOINT_NAMES:
                raw = int(positions[joint])
                mins[joint] = min(mins[joint], raw)
                maxs[joint] = max(maxs[joint], raw)
                rng = maxs[joint] - mins[joint]
                stdscr.addstr(row, 2, f"{joint:18s}  raw={raw:5d}  min={mins[joint]:5d}  max={maxs[joint]:5d}  ({rng:4d})")
                row += 1

            stdscr.addstr(row + 1, 2, "[Enter] Done   [r] Reset")
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                break
            elif key in (ord("r"), ord("R")):
                positions = bus.sync_read("Present_Position", normalize=False)
                for joint in JOINT_NAMES:
                    raw = int(positions[joint])
                    mins[joint] = raw
                    maxs[joint] = raw

            time.sleep(0.05)
    finally:
        stdscr.nodelay(False)

    # Store physical ranges and compute initial offsets (LeRobot style: current pos → half turn)
    for joint in JOINT_NAMES:
        states[joint].phys_min = mins[joint]
        states[joint].phys_max = maxs[joint]
        # Initial offset: midpoint of range → half turn
        mid = (mins[joint] + maxs[joint]) // 2
        states[joint].homing_offset = mid - HALF_TURN


# ---------------------------------------------------------------------------
# Phase 2 — Zero tuning (per joint)
# ---------------------------------------------------------------------------

def _phase_zero_tune(
    stdscr,
    bus: FeetechMotorsBus,
    state: JointState,
    all_states: dict[str, JointState],
    profile_name: str,
) -> None:
    """Torque on at 0°, nudge +/- to match sim zero, accept."""
    assert state.homing_offset is not None

    # Write offset to EEPROM and command 0°
    _write_offset(bus, state.name, state.homing_offset)
    _command_degrees(bus, state.name, state, 0.0)

    stdscr.nodelay(True)
    try:
        while True:
            raw = bus.sync_read("Present_Position", [state.name], normalize=False)[state.name]

            stdscr.clear()
            stdscr.addstr(0, 0, f"── Zero Tuning: {state.name} ──", curses.A_BOLD)
            stdscr.addstr(2, 2, f"Homing offset:  {state.homing_offset}")
            stdscr.addstr(3, 2, f"Physical range: [{state.phys_min}, {state.phys_max}]")
            stdscr.addstr(4, 2, f"Raw encoder:    {int(raw)}")
            stdscr.addstr(6, 2, "Motor is holding at 0°. Nudge until it matches sim zero.")
            stdscr.addstr(8, 2, "[+/-] Nudge offset (motor moves)")
            stdscr.addstr(9, 2, "[g] Global pose (all joints)   [z] All zeros")
            stdscr.addstr(10, 2, "[a] Accept   [b] Back")
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("+"), ord("=")):
                state.homing_offset += 1
                _write_offset(bus, state.name, state.homing_offset)
                _command_degrees(bus, state.name, state, 0.0)
            elif key in (ord("-"), ord("_")):
                state.homing_offset -= 1
                _write_offset(bus, state.name, state.homing_offset)
                _command_degrees(bus, state.name, state, 0.0)
            elif key == ord("g"):
                bus.disable_torque([state.name])
                stdscr.nodelay(False)
                _global_pose(stdscr, bus, all_states)
                # Re-enable this joint at 0°
                _write_offset(bus, state.name, state.homing_offset)
                _command_degrees(bus, state.name, state, 0.0)
                stdscr.nodelay(True)
            elif key == ord("z"):
                bus.disable_torque([state.name])
                stdscr.nodelay(False)
                _all_zeros(stdscr, bus, all_states)
                # Re-enable this joint at 0°
                _write_offset(bus, state.name, state.homing_offset)
                _command_degrees(bus, state.name, state, 0.0)
                stdscr.nodelay(True)
            elif key == ord("a"):
                bus.disable_torque([state.name])
                state.zero_tuned = True
                _autosave(profile_name, all_states)
                return
            elif key == ord("b"):
                bus.disable_torque([state.name])
                return

            time.sleep(0.05)
    finally:
        stdscr.nodelay(False)


# ---------------------------------------------------------------------------
# Global pose helpers
# ---------------------------------------------------------------------------

def _all_zeros(stdscr, bus: FeetechMotorsBus, all_states: dict[str, JointState]) -> None:
    """Command all tuned joints to 0°, hold until keypress."""
    ready = [j for j in JOINT_NAMES if all_states[j].homing_offset is not None and all_states[j].phys_min is not None]
    if not ready:
        return

    for j in ready:
        assert all_states[j].homing_offset is not None
        _write_offset(bus, j, all_states[j].homing_offset)
        _command_degrees(bus, j, all_states[j], 0.0)

    stdscr.clear()
    stdscr.addstr(0, 0, "── All Zeros ──", curses.A_BOLD)
    row = 2
    for j in ready:
        stdscr.addstr(row, 2, f"{j:18s} → 0°  (offset={all_states[j].homing_offset})")
        row += 1
    stdscr.addstr(row + 1, 2, "Press any key to release")
    stdscr.refresh()
    stdscr.getch()
    bus.disable_torque(ready)


def _global_pose(stdscr, bus: FeetechMotorsBus, all_states: dict[str, JointState]) -> None:
    """Prompt degrees per joint, hold pose."""
    stdscr.clear()
    stdscr.addstr(0, 0, "── Global Pose ──", curses.A_BOLD)
    stdscr.addstr(2, 2, "Enter degrees per joint (empty=skip, q=cancel, z=all zeros):")

    targets: dict[str, float] = {}
    row = 4
    for j in JOINT_NAMES:
        st = all_states[j]
        has_cal = st.homing_offset is not None and st.phys_min is not None
        info = f"(offset={st.homing_offset})" if has_cal else "(skip)"
        stdscr.addstr(row, 2, f"{j:18s} {info}")
        if has_cal:
            curses.echo()
            stdscr.addstr(row, 50, "→ ")
            stdscr.refresh()
            try:
                inp = stdscr.getstr(row, 52, 10).decode().strip()
            except (curses.error, UnicodeDecodeError):
                inp = ""
            curses.noecho()
            if inp.lower() == "q":
                return
            if inp.lower() == "z":
                targets = {j2: 0.0 for j2 in JOINT_NAMES if all_states[j2].homing_offset is not None and all_states[j2].phys_min is not None}
                break
            if inp:
                try:
                    targets[j] = float(inp)
                except ValueError:
                    pass
        row += 1

    if not targets:
        return

    for j, deg in targets.items():
        assert all_states[j].homing_offset is not None
        _write_offset(bus, j, all_states[j].homing_offset)
        _command_degrees(bus, j, all_states[j], deg)

    stdscr.clear()
    stdscr.addstr(0, 0, "── Holding Pose ──", curses.A_BOLD)
    r = 2
    for j, deg in targets.items():
        stdscr.addstr(r, 2, f"{j:18s} → {deg:.1f}°")
        r += 1
    stdscr.addstr(r + 1, 2, "Press any key to release")
    stdscr.refresh()
    stdscr.getch()
    bus.disable_torque(list(targets.keys()))


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _compute_final_calibration(states: dict[str, JointState]) -> dict[str, dict]:
    """Compute final calibration with ranges adjusted for homing_offset."""
    calib = {}
    for joint in JOINT_NAMES:
        st = states[joint]
        if st.homing_offset is None or st.phys_min is None or st.phys_max is None:
            continue
        # Range in reported space: reported = physical - offset
        calib[joint] = {
            "id": st.motor_id,
            "drive_mode": 0,
            "homing_offset": st.homing_offset,
            "range_min": st.phys_min - st.homing_offset,
            "range_max": st.phys_max - st.homing_offset,
        }
    return calib


def _autosave(name: str, all_states: dict[str, JointState]) -> None:
    calib = _compute_final_calibration(all_states)
    if not calib:
        return
    out = _calib_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=4) + "\n")


# ---------------------------------------------------------------------------
# Main TUI
# ---------------------------------------------------------------------------

def _draw_main(stdscr, states: dict[str, JointState], name: str, selected: int, phase1_done: bool) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(0, 0, f"── Calibration: {name} ──", curses.A_BOLD)

    row = 2
    for i, j in enumerate(JOINT_NAMES):
        st = states[j]
        if st.zero_tuned:
            marker = "✓"
        elif i == selected:
            marker = "●"
        else:
            marker = "○"
        label = f"  {marker} {j:18s}"
        if st.homing_offset is not None:
            label += f"  offset={st.homing_offset}"
        if st.phys_min is not None:
            label += f"  range=[{st.phys_min},{st.phys_max}]"
        attr = curses.A_REVERSE if i == selected else 0
        stdscr.addstr(row + i, 1, label[:w - 2], attr)

    footer_row = row + len(JOINT_NAMES) + 2
    if not phase1_done:
        stdscr.addstr(footer_row, 1, "[r] Run range discovery first")
    else:
        stdscr.addstr(footer_row, 1, "[Enter] Tune zero   [g] Global pose   [z] All zeros   [s] Save   [q] Quit")
    stdscr.refresh()


def _tui_main(stdscr, port: str, name: str) -> None:
    curses.curs_set(0)
    bus = _make_bus(port)
    bus.connect()

    states: dict[str, JointState] = {}
    phase1_done = False

    # Load existing profile if present
    calib_file = _calib_path(name)
    if calib_file.exists():
        data = json.loads(calib_file.read_text())
        for j in JOINT_NAMES:
            if j in data:
                d = data[j]
                # Reconstruct physical range from stored range + offset
                offset = d["homing_offset"]
                states[j] = JointState(
                    name=j, motor_id=MOTOR_IDS[j],
                    phys_min=d["range_min"] + offset,
                    phys_max=d["range_max"] + offset,
                    homing_offset=offset,
                    zero_tuned=True,
                )
            else:
                states[j] = JointState(name=j, motor_id=MOTOR_IDS[j])
        phase1_done = all(states[j].phys_min is not None for j in JOINT_NAMES)
    else:
        for j in JOINT_NAMES:
            states[j] = JointState(name=j, motor_id=MOTOR_IDS[j])

    selected = 0

    try:
        while True:
            _draw_main(stdscr, states, name, selected, phase1_done)
            key = stdscr.getch()

            if key == curses.KEY_UP:
                selected = (selected - 1) % len(JOINT_NAMES)
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % len(JOINT_NAMES)
            elif ord("1") <= key <= ord("6"):
                selected = key - ord("1")
            elif key == ord("r"):
                _phase_range(stdscr, bus, states)
                phase1_done = True
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")) and phase1_done:
                j = JOINT_NAMES[selected]
                _phase_zero_tune(stdscr, bus, states[j], states, name)
            elif key == ord("g") and phase1_done:
                _global_pose(stdscr, bus, states)
            elif key == ord("z") and phase1_done:
                _all_zeros(stdscr, bus, states)
            elif key == ord("s"):
                calib = _compute_final_calibration(states)
                if not calib:
                    continue
                out = _calib_path(name)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(calib, indent=4) + "\n")
                register(name)
                curses.endwin()
                print(f"Saved to {out}")
                ans = input("Write to motors now? [y/n] ").strip().lower()
                if ans == "y":
                    from vbti.logic.servos.profiles import load as load_profile
                    load_profile(name, port=port)
                print("Done.")
                return
            elif key == ord("q"):
                return
    finally:
        bus.disconnect(disable_torque=True)


def main(port: str = "/dev/ttyACM1", name: str = "new-profile") -> None:
    """Launch the interactive calibration TUI."""
    try:
        curses.wrapper(lambda stdscr: _tui_main(stdscr, port, name))
    except _curses.error:
        pass


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
