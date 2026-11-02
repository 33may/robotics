#!/usr/bin/env python3
"""Does the model agree with the robot about where things are? (read-only)

Closes the one loop nothing else in this repo closes. Every other calibration
here compares a measurement to another measurement — the table probe fits
pendant readings against a tape measure, hand-eye fits camera against flange.
None of them ever asks whether the MODEL's idea of a joint configuration lands
where the CONTROLLER says it does. That gap hid a 180 deg base-frame error for
three days: `cell.yaml` holds pendant readings (UR controller `base` frame)
while FK ran in the URDF `base_link` frame, which the vendor's own URDF
declares is `base` rotated pi about Z. Every viewer agreed, because they all
drew the same self-consistent, wrong model.

The test is offset-free on purpose. We do not know the pendant's configured
TCP, and assuming one would just move the guesswork: instead, express the
flange -> TCP vector in the FLANGE frame at several configurations. That vector
IS the TCP offset, so it must be the SAME at every pose. If the model's base
frame is rotated relative to the controller's, the vector swings wildly.

Commands nothing. Never moves the arm — it only reads joint angles and the
controller's own Cartesian readout, so it is safe to run at any time.

    p inspection/calib/frame_check.py            # one sample, at the current pose
    p inspection/calib/frame_check.py --n=3      # freedrive between samples
"""

import numpy as np

ROBOT_IP = "192.168.2.50"
TOLERANCE_MM = 2.0


def _sample(ip):
    from rtde_receive import RTDEReceiveInterface

    from inspection.motion.ik import UR5eIK

    receive = RTDEReceiveInterface(ip)
    try:
        q = np.array(receive.getActualQ())
        tcp = np.array(receive.getActualTCPPose())
    finally:
        receive.disconnect()

    T = UR5eIK().fk(q)
    # The flange -> TCP vector, written in the flange frame. Rigid, so it is
    # the same number at every configuration — unless our base frame is wrong.
    offset = T[:3, :3].T @ (tcp[:3] - T[:3, 3])
    return q, tcp, offset


def check(ip: str = ROBOT_IP, n: int = 1) -> dict:
    """Sample n configurations and report whether the offset stays constant."""
    offsets = []
    for i in range(int(n)):
        if i:
            input(f"\nfreedrive to a DIFFERENT pose, then press ENTER "
                  f"({i + 1}/{n}) ")
        q, tcp, offset = _sample(ip)
        offsets.append(offset)
        print(f"[{i + 1}] q_deg = {np.round(np.degrees(q), 1).tolist()}")
        print(f"    pendant TCP = {np.round(tcp[:3], 4).tolist()}")
        print(f"    flange->TCP in flange frame = "
              f"{np.round(offset * 1000, 1).tolist()} mm")

    offsets = np.array(offsets)
    spread_mm = float(np.abs(offsets - offsets.mean(axis=0)).max() * 1000) \
        if len(offsets) > 1 else 0.0
    report = {
        "n": len(offsets),
        "tcp_offset_mm": np.round(offsets.mean(axis=0) * 1000, 2).tolist(),
        "spread_mm": round(spread_mm, 2),
        "ok": spread_mm <= TOLERANCE_MM,
    }
    print(f"\nTCP offset (mean) = {report['tcp_offset_mm']} mm")
    if len(offsets) > 1:
        print(f"spread across poses = {report['spread_mm']} mm "
              f"(tolerance {TOLERANCE_MM} mm)")
        print("MODEL AGREES WITH THE ROBOT" if report["ok"] else
              "MISMATCH: the model's base frame disagrees with the controller")
    else:
        print("single sample — compare against the TCP configured on the "
              "pendant (Installation > TCP). Run with --n=3 to test properly.")
    return report


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(check)
