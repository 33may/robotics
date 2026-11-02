#!/usr/bin/env python3
"""Freedrive tap logger — press "s", get the TCP point, copy-paste anywhere.

Usage (robo env):  p inspection/calib/tap_probe.py [--ip 192.168.2.50]

Put the pendant in freedrive with TCP_normal active, kiss the surface with
the closed tip, press "s" here. Prints:

    point 1
    [0.2598, -0.6884, -0.0052]

"q" or Ctrl-C quits. Coordinates are meters, UR controller `base` frame,
straight from getActualTCPPose() — no Feature-frame trap possible.
"""

import argparse
import sys
import termios
import time
import tty

ROBOT_IP = "192.168.2.50"


def fresh_tcp(recv):
    """ur_rtde recv freezes SILENTLY (ur_rtde #307) — trust a read only if
    the controller clock advanced between two samples."""
    t0 = recv.getTimestamp()
    time.sleep(0.05)
    if recv.getTimestamp() <= t0:
        raise RuntimeError("RTDE receive is STALE — restart the script")
    return recv.getActualTCPPose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=ROBOT_IP)
    args = ap.parse_args()

    from rtde_receive import RTDEReceiveInterface
    recv = RTDEReceiveInterface(args.ip)
    print("connected — freedrive to the point, [s] sample, [q] quit\n")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    n = 0
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "q":
                break
            if ch != "s":
                continue
            x, y, z = fresh_tcp(recv)[:3]
            n += 1
            print(f"point {n}\n[{x:.4f}, {y:.4f}, {z:.4f}]\n", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        recv.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
