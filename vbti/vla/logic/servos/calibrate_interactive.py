"""Interactive servo calibration TUI.

Starts from an existing calibration (e.g. frodeo-test), uses SOFollower
(same as goto) to command joints, then lets you nudge offsets until 0°
matches simulation zero.
"""
from __future__ import annotations

import _curses
import curses
import json
import time

from lerobot.motors import MotorCalibration
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from vbti.logic.servos.profiles import (
    JOINT_NAMES,
    LEROBOT_CALIB_DIR,
    _calib_path,
    register,
)


# ---------------------------------------------------------------------------
# Robot helpers (use SOFollower — same path as goto)
# ---------------------------------------------------------------------------

def _connect_robot(port: str, robot_id: str) -> SOFollower:
    """Connect SOFollower with given calibration profile."""
    config = SOFollowerRobotConfig(port=port, id=robot_id)
    robot = SOFollower(config)
    robot.connect(calibrate=False)
    # Write calibration from file to EEPROM
    if robot.calibration:
        robot.bus.write_calibration(robot.calibration)
    robot.configure()
    return robot


def _goto_zeros(robot: SOFollower, steps: int = 100, duration: float = 3.0) -> None:
    """Smoothly interpolate all joints to 0° over duration seconds."""
    target = {f"{m}.pos": 0.0 for m in robot.bus.motors}
    current = robot.get_observation()
    current_pos = {f"{m}.pos": current[f"{m}.pos"] for m in robot.bus.motors}
    dt = duration / steps
    for step in range(1, steps + 1):
        t = step / steps
        interp = {}
        for key in target:
            interp[key] = current_pos[key] + t * (target[key] - current_pos[key])
        robot.send_action(interp)
        time.sleep(dt)


def _goto_zeros_hold(robot: SOFollower, stdscr) -> None:
    """Command all zeros and hold until keypress."""
    _goto_zeros(robot)
    stdscr.clear()
    stdscr.addstr(0, 0, "── All Zeros ──", curses.A_BOLD)
    row = 2
    assert robot.calibration is not None
    for j in JOINT_NAMES:
        cal = robot.calibration[j]
        stdscr.addstr(row, 2, f"{j:18s} → 0°  (offset={cal.homing_offset})")
        row += 1
    stdscr.addstr(row + 1, 2, "Press any key to release")
    stdscr.refresh()
    stdscr.getch()
    robot.bus.disable_torque()


def _apply_nudge(robot: SOFollower, calib_dict: dict, joint: str, step: int) -> None:
    """Nudge offset by step, shift range to match, update robot + EEPROM.

    When offset changes by +step, reported space shifts by -step.
    range_min/max shift by -step to keep same physical limits.
    """
    assert robot.calibration is not None
    c = calib_dict[joint]
    new_offset = c["homing_offset"] + step
    new_min = c["range_min"] - step
    new_max = c["range_max"] - step

    # Update JSON dict
    c["homing_offset"] = new_offset
    c["range_min"] = new_min
    c["range_max"] = new_max

    # Update robot calibration
    old_cal = robot.calibration[joint]
    robot.calibration[joint] = MotorCalibration(
        id=old_cal.id, drive_mode=old_cal.drive_mode,
        homing_offset=new_offset, range_min=new_min, range_max=new_max,
    )

    # Write to servo EEPROM (needs torque off)
    robot.bus.disable_torque([joint])
    robot.bus.write("Homing_Offset", joint, new_offset)
    robot.bus.write("Min_Position_Limit", joint, new_min)
    robot.bus.write("Max_Position_Limit", joint, new_max)
    time.sleep(0.05)

    # Re-command 0°
    robot.bus.enable_torque([joint])
    robot.bus.write("Goal_Position", joint, 0.0)


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


def _tune_joint(stdscr, robot: SOFollower, calib: dict, joint: str, name: str, tuned: dict[str, bool]) -> None:
    """Nudge offset until 0° matches sim zero."""
    # Command this joint to 0°
    robot.bus.enable_torque([joint])
    robot.bus.write("Goal_Position", joint, 0.0)

    stdscr.nodelay(True)
    try:
        while True:
            raw = int(robot.bus.sync_read("Present_Position", [joint], normalize=False)[joint])
            c = calib[joint]
            mid = (c["range_min"] + c["range_max"]) / 2

            stdscr.clear()
            stdscr.addstr(0, 0, f"── Zero Tuning: {joint} ──", curses.A_BOLD)
            stdscr.addstr(2, 2, f"Homing offset:  {c['homing_offset']}")
            stdscr.addstr(3, 2, f"Range:          [{c['range_min']}, {c['range_max']}]  mid={mid:.0f}")
            stdscr.addstr(4, 2, f"Raw encoder:    {raw}")
            stdscr.addstr(6, 2, "Motor holding at 0°. Nudge until it matches sim zero.")
            stdscr.addstr(8, 2, "[+/-] ±1    [>/<] ±10    []/[] ±100")
            stdscr.addstr(9, 2, "[z] All zeros    [a] Accept    [b] Back")
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
                _apply_nudge(robot, calib, joint, step)
            elif key == ord("z"):
                robot.bus.disable_torque([joint])
                stdscr.nodelay(False)
                # Reload all calibration to servos and go to zeros
                assert robot.calibration is not None
                with robot.bus.torque_disabled():
                    robot.bus.write_calibration(robot.calibration)
                _goto_zeros_hold(robot, stdscr)
                # Re-enable just this joint
                robot.bus.enable_torque([joint])
                robot.bus.write("Goal_Position", joint, 0.0)
                stdscr.nodelay(True)
            elif key == ord("a"):
                robot.bus.disable_torque([joint])
                tuned[joint] = True
                # Autosave
                out = _calib_path(name)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(calib, indent=4) + "\n")
                return
            elif key == ord("b"):
                robot.bus.disable_torque([joint])
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

    # If target profile already exists, resume from it
    target_path = _calib_path(name)
    if target_path.exists() and name != base_profile:
        calib = json.loads(target_path.read_text())

    # Save as target so SOFollower can load it
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(calib, indent=4) + "\n")

    # Connect using SOFollower (same as goto)
    robot = _connect_robot(port, name)

    tuned: dict[str, bool] = {j: False for j in JOINT_NAMES}
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
                _tune_joint(stdscr, robot, calib, JOINT_NAMES[selected], name, tuned)
            elif key == ord("z"):
                _goto_zeros(robot)
                stdscr.clear()
                stdscr.addstr(0, 0, "── All Zeros ──", curses.A_BOLD)
                row = 2
                for j in JOINT_NAMES:
                    stdscr.addstr(row, 2, f"{j:18s} → 0°  (offset={calib[j]['homing_offset']})")
                    row += 1
                stdscr.addstr(row + 1, 2, "Press any key to release")
                stdscr.refresh()
                stdscr.getch()
                robot.bus.disable_torque()
            elif key == ord("s"):
                out = _calib_path(name)
                out.write_text(json.dumps(calib, indent=4) + "\n")
                register(name)
                curses.endwin()
                print(f"Saved to {out}")
                # Write final calibration to EEPROM
                assert robot.calibration is not None
                with robot.bus.torque_disabled():
                    robot.bus.write_calibration(robot.calibration)
                print("Written to motors. Done.")
                return
            elif key == ord("q"):
                return
    finally:
        robot.disconnect()


def main(port: str = "/dev/ttyACM1", name: str = "sim_accurate", base: str = "frodeo-test") -> None:
    """Launch the interactive calibration TUI.

    Args:
        port: Serial port for the robot arm.
        name: Profile name to create.
        base: Existing profile to start from.
    """
    try:
        curses.wrapper(lambda stdscr: _tui_main(stdscr, port, name, base))
    except _curses.error:
        pass


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
