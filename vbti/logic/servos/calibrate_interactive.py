"""Interactive servo calibration TUI.

Starts from an existing calibration (e.g. frodeo-test), loads it to servos,
commands each joint to 0°, then lets you nudge the offset until the physical
position matches simulation zero. Only the homing_offset changes — range stays.
"""
from __future__ import annotations

import _curses
import curses
import json
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from vbti.logic.servos.profiles import (
    JOINT_NAMES,
    LEROBOT_CALIB_DIR,
    _calib_path,
    register,
)

MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
    "wrist_flex": 4, "wrist_roll": 5, "gripper": 6,
}


# ---------------------------------------------------------------------------
# Bus helpers
# ---------------------------------------------------------------------------

def _make_bus(port: str, calib: dict) -> FeetechMotorsBus:
    """Create bus with calibration loaded."""
    motors = {}
    calibration = {}
    for joint in JOINT_NAMES:
        j = calib[joint]
        norm = MotorNormMode.RANGE_0_100 if joint == "gripper" else MotorNormMode.DEGREES
        motors[joint] = Motor(id=j["id"], model="sts3215", norm_mode=norm)
        calibration[joint] = MotorCalibration(
            id=j["id"], drive_mode=j["drive_mode"],
            homing_offset=j["homing_offset"],
            range_min=j["range_min"], range_max=j["range_max"],
        )
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
    return bus


def _load_calib_to_servos(bus: FeetechMotorsBus) -> None:
    """Write current bus calibration to servo EEPROM."""
    assert bus.calibration is not None
    with bus.torque_disabled():
        bus.write_calibration(bus.calibration)
    time.sleep(0.1)


def _apply_nudge(bus: FeetechMotorsBus, calib: dict, joint: str, step: int) -> None:
    """Nudge offset by step, shift range to match, write to EEPROM, re-command 0°.

    When offset changes by +step, reported space shifts by -step at the same
    physical position. So range_min/max must also shift by -step to keep
    pointing at the same physical limits.
    """
    assert bus.calibration is not None
    c = calib[joint]
    new_offset = c["homing_offset"] + step
    new_min = c["range_min"] - step
    new_max = c["range_max"] - step

    # Update JSON dict
    c["homing_offset"] = new_offset
    c["range_min"] = new_min
    c["range_max"] = new_max

    # Update bus calibration
    old_cal = bus.calibration[joint]
    bus.calibration[joint] = MotorCalibration(
        id=old_cal.id, drive_mode=old_cal.drive_mode,
        homing_offset=new_offset, range_min=new_min, range_max=new_max,
    )

    # Write to servo EEPROM
    bus.disable_torque([joint])
    bus.write("Homing_Offset", joint, new_offset)
    bus.write("Min_Position_Limit", joint, new_min)
    bus.write("Max_Position_Limit", joint, new_max)
    time.sleep(0.05)
    bus.enable_torque([joint])
    bus.write("Goal_Position", joint, 0.0)


def _command_all_zeros(bus: FeetechMotorsBus) -> None:
    """Command all joints to 0°."""
    bus.enable_torque()
    for joint in JOINT_NAMES:
        bus.write("Goal_Position", joint, 0.0)


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

def _draw_main(stdscr, calib: dict, name: str, selected: int, tuned: dict[str, bool]) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    stdscr.addstr(0, 0, f"── Calibration: {name} ──", curses.A_BOLD)

    row = 2
    for i, j in enumerate(JOINT_NAMES):
        c = calib[j]
        marker = "✓" if tuned.get(j) else ("●" if i == selected else "○")
        mid = (c["range_min"] + c["range_max"]) / 2
        label = f"  {marker} {j:18s}  offset={c['homing_offset']:>5}  mid={mid:.0f}  range=[{c['range_min']},{c['range_max']}]"
        attr = curses.A_REVERSE if i == selected else 0
        stdscr.addstr(row + i, 1, label[:w - 2], attr)

    footer_row = row + len(JOINT_NAMES) + 2
    stdscr.addstr(footer_row, 1, "[Enter] Tune zero   [z] All zeros   [s] Save & exit   [q] Quit")
    stdscr.refresh()


def _tune_joint(stdscr, bus: FeetechMotorsBus, calib: dict, joint: str, name: str, tuned: dict[str, bool]) -> None:
    """Nudge offset until 0° matches sim zero."""
    assert bus.calibration is not None
    # Command this joint to 0°
    bus.enable_torque([joint])
    bus.write("Goal_Position", joint, 0.0)

    offset = calib[joint]["homing_offset"]
    stdscr.nodelay(True)

    try:
        while True:
            raw = int(bus.sync_read("Present_Position", [joint], normalize=False)[joint])
            c = calib[joint]
            mid = (c["range_min"] + c["range_max"]) / 2
            deg = (raw - mid) * 360.0 / 4095

            stdscr.clear()
            stdscr.addstr(0, 0, f"── Zero Tuning: {joint} ──", curses.A_BOLD)
            stdscr.addstr(2, 2, f"Homing offset:  {offset}")
            stdscr.addstr(3, 2, f"Range:          [{c['range_min']}, {c['range_max']}]  mid={mid:.0f}")
            stdscr.addstr(4, 2, f"Raw encoder:    {raw}")
            stdscr.addstr(5, 2, f"Current degrees:{deg:>8.1f}°")
            stdscr.addstr(7, 2, "Motor holding at 0°. Nudge until it matches sim zero.")
            stdscr.addstr(9, 2, "[+/-] ±1 tick    [>/<] ±10 ticks    []/[] ±100 ticks")
            stdscr.addstr(10, 2, "[z] All zeros    [a] Accept    [b] Back")
            stdscr.refresh()

            key = stdscr.getch()
            step = 0
            if key in (ord("+"), ord("=")):
                step = 1
            elif key in (ord("-"), ord("_")):
                step = -1
            elif key in (ord(">"), ord(".")):
                step = 10
            elif key in (ord("<"), ord(",")):
                step = -10
            elif key == ord("]"):
                step = 100
            elif key == ord("["):
                step = -100

            if step != 0:
                _apply_nudge(bus, calib, joint, step)
                offset = calib[joint]["homing_offset"]
            elif key == ord("z"):
                stdscr.nodelay(False)
                _load_calib_to_servos(bus)
                _command_all_zeros(bus)
                stdscr.clear()
                stdscr.addstr(0, 0, "── All Zeros ──", curses.A_BOLD)
                row = 2
                for j in JOINT_NAMES:
                    stdscr.addstr(row, 2, f"{j:18s} → 0°  (offset={calib[j]['homing_offset']})")
                    row += 1
                stdscr.addstr(row + 1, 2, "Press any key to release")
                stdscr.refresh()
                stdscr.getch()
                bus.disable_torque()
                # Re-enable just this joint
                bus.enable_torque([joint])
                bus.write("Goal_Position", joint, 0.0)
                stdscr.nodelay(True)
            elif key == ord("a"):
                bus.disable_torque([joint])
                tuned[joint] = True
                # Autosave
                out = _calib_path(name)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(calib, indent=4) + "\n")
                return
            elif key == ord("b"):
                bus.disable_torque([joint])
                return

            time.sleep(0.05)
    finally:
        stdscr.nodelay(False)


def _tui_main(stdscr, port: str, name: str, base_profile: str) -> None:
    curses.curs_set(0)

    # Load base calibration
    base_path = LEROBOT_CALIB_DIR / f"{base_profile}.json"
    if not base_path.exists():
        raise FileNotFoundError(f"Base profile not found: {base_path}")
    calib = json.loads(base_path.read_text())

    # If target profile already exists, load it instead (resume)
    target_path = _calib_path(name)
    if target_path.exists() and name != base_profile:
        calib = json.loads(target_path.read_text())

    # Track which joints have been tuned
    tuned: dict[str, bool] = {j: False for j in JOINT_NAMES}

    # Create bus with this calibration and load to servos
    bus = _make_bus(port, calib)
    bus.connect()
    _load_calib_to_servos(bus)

    selected = 0

    try:
        while True:
            _draw_main(stdscr, calib, name, selected, tuned)
            key = stdscr.getch()

            if key == curses.KEY_UP:
                selected = (selected - 1) % len(JOINT_NAMES)
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % len(JOINT_NAMES)
            elif ord("1") <= key <= ord("6"):
                selected = key - ord("1")
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                _tune_joint(stdscr, bus, calib, JOINT_NAMES[selected], name, tuned)
            elif key == ord("z"):
                _load_calib_to_servos(bus)
                _command_all_zeros(bus)
                stdscr.clear()
                stdscr.addstr(0, 0, "── All Zeros ──", curses.A_BOLD)
                row = 2
                for j in JOINT_NAMES:
                    stdscr.addstr(row, 2, f"{j:18s} → 0°  (offset={calib[j]['homing_offset']})")
                    row += 1
                stdscr.addstr(row + 1, 2, "Press any key to release")
                stdscr.refresh()
                stdscr.getch()
                bus.disable_torque()
            elif key == ord("s"):
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


def main(port: str = "/dev/ttyACM1", name: str = "sim_accurate", base: str = "frodeo-test") -> None:
    """Launch the interactive calibration TUI.

    Args:
        port: Serial port for the robot arm.
        name: Profile name to create.
        base: Existing profile to start from (default: frodeo-test).
    """
    try:
        curses.wrapper(lambda stdscr: _tui_main(stdscr, port, name, base))
    except _curses.error:
        pass


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
