#!/usr/bin/env python3
"""Live ALL-joints readout for manual freedrive back to a plannable pose.

Six lines, ~10 Hz: per joint — angle now, delta to the survey pose, a bar,
and a verdict color: green = fine, yellow = off, red = FIX (this joint is
what makes the survey plan refuse). The elbow row also flags a sign flip
(the 1009-cup4 abort: elbow +134 deg vs survey's -116 — planner finds no
path from the flipped branch). Freedrive until every row is green-ish,
then start collect.

Target = inspection/run/survey_pose.json (the pose `collect` plans to).
wrist_2 keeps the close-pass arc warning (2026-08-31 posture — the arc
moves if the big joints move; treat as a hint, not gospel).

Run: p inspection/motion/wrist_watch.py [ip]     Ctrl-C to stop.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from rtde_receive import RTDEReceiveInterface

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.2.50"
SURVEY_FILE = Path(__file__).resolve().parent.parent / "run" / "survey_pose.json"
NAMES = ("base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3")
ELBOW = 2
WRIST2 = 4
DANGER_W2 = (-210.0, -145.0)   # deg — cable loop + gripper pass the forearm
LIMIT = 360.0                  # deg — UR hard stop
OK, WARN = 45.0, 90.0          # deg — |delta to survey| thresholds

RED, GREEN, YELLOW, DIM, END = ("\033[91m", "\033[92m", "\033[93m",
                                "\033[2m", "\033[0m")


def _row(i: int, deg: float, target: float) -> str:
    d = deg - target
    if LIMIT - abs(deg) < 20.0:
        state = f"{RED}NEAR HARD STOP{END}"
    elif i == ELBOW and np.sign(deg) != np.sign(target) and abs(d) > OK:
        state = f"{RED}FIX — elbow flipped (sign must be "
        state += f"{'+' if target > 0 else '-'}){END}"
    elif abs(d) > WARN:
        state = f"{RED}FIX{END}"
    elif abs(d) > OK:
        state = f"{YELLOW}off{END}"
    else:
        state = f"{GREEN}ok{END}"
    extra = ""
    if i == WRIST2 and DANGER_W2[0] <= deg <= DANGER_W2[1]:
        extra = f"  {RED}CLOSE PASS — slow, watch cable{END}"
    bar_pos = int(np.clip((deg + LIMIT) / (2 * LIMIT) * 40, 0, 39))
    tgt_pos = int(np.clip((target + LIMIT) / (2 * LIMIT) * 40, 0, 39))
    bar = "".join("|" if k == bar_pos else ("+" if k == tgt_pos else "-")
                  for k in range(40))
    return (f"  {NAMES[i]:9s} {deg:+8.1f}  [{bar}]  "
            f"to survey {-d:+7.1f}  {state}{extra}")


def main():
    target = np.degrees(json.loads(SURVEY_FILE.read_text())["q_rad"])
    rtde = RTDEReceiveInterface(IP)
    print(f"freedrive to the survey pose ({SURVEY_FILE.name}); "
          f"| = now, + = target. Ctrl-C to quit.")
    first = True
    while True:
        q = np.degrees(rtde.getActualQ())
        if not first:
            sys.stdout.write(f"\033[{len(NAMES) + 1}A")
        first = False
        worst = float(np.max(np.abs(q - target)))
        flipped = np.sign(q[ELBOW]) != np.sign(target[ELBOW])
        if worst < OK and not flipped:
            verdict = f"{GREEN}ALL OK — survey should plan; start collect{END}"
        else:
            verdict = (f"{RED}not plannable yet{END}  "
                       f"{DIM}worst delta {worst:.0f} deg{END}")
        for i in range(len(NAMES)):
            sys.stdout.write("\033[2K" + _row(i, q[i], target[i]) + "\n")
        sys.stdout.write("\033[2K  " + verdict + "\n")
        sys.stdout.flush()
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
