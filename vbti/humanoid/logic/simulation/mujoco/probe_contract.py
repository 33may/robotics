#!/usr/bin/env python3
"""
One-shot wire-contract probe for the LimX MROS bus.

Run this *while* the launcher is up (sim + deploy + a stand/walk/mimic ability
actively publishing). It subscribes to RobotState, ImuData, and RobotCmd; waits
for `--duration` seconds collecting samples; then prints the first sample shape
+ observed publish rate per topic.

================================================================================
Why two roles? (read this before you complain about 0 samples)
================================================================================

The LimX SDK is **role-gated** in both directions:

                    publishes              subscribes
    sim peer    →   RobotState, ImuData    RobotCmd  (via subscribeRobotCmdForSim)
    policy peer →   RobotCmd               RobotState, ImuData  (via subscribeRobotState/ImuData)

A *policy* peer subscribing to RobotCmd hits a self-loopback — it only sees
cmds it published itself, so an external probe sees 0.

A *sim* peer subscribing to RobotState/ImuData gets nothing — those topics
are produced by the sim peer, not delivered to it.

A single probe run therefore only captures half the contract. Default mode
of this script is `--role both`: spawns two sequential subprocesses (one per
role) that share stdout, so one command captures the full contract while
keeping the SDK state cleanly isolated per role.

================================================================================
Run
================================================================================

    conda activate limx
    python humanoid/logic/simulation/mujoco/probe_contract.py \\
        [--role both|sim|policy] [--duration 3.0] [robot_ip]

Defaults: --role both, --duration 3.0, robot_ip 127.0.0.1.

`both` runs `policy` first (state + imu) then `sim` (cmd), back-to-back,
~6.5 s wall time end-to-end. The launcher must be up the whole time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from typing import Deque

# limxsdk is a Py3.8-ABI-locked binary wheel; Pyright cannot resolve it
# statically. The imports work at runtime in the `limx` conda env.
import limxsdk.datatypes as dt  # type: ignore[import-not-found]
from limxsdk.robot import Robot, RobotType  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="probe_contract.py",
        description="One-shot LimX MROS wire-contract probe (sim and/or policy peer).",
    )
    p.add_argument(
        "--role",
        choices=["both", "sim", "policy"],
        default="both",
        help=(
            "Bus peer role. 'both' (default) runs policy then sim back-to-back as "
            "subprocesses. 'sim' sees RobotCmd via subscribeRobotCmdForSim. "
            "'policy' sees RobotState + ImuData."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Seconds to collect samples per role before printing (default: 3.0).",
    )
    p.add_argument(
        "robot_ip",
        nargs="?",
        default="127.0.0.1",
        help="MROS peer IP (default: 127.0.0.1 = local sim loopback).",
    )
    return p.parse_args()


def run_both(duration: float, robot_ip: str) -> int:
    """Spawn two child processes, one per role, sequentially.

    Subprocess isolation guarantees the SDK threads + bus handles for one
    role are fully torn down before the next role's process starts. Avoids
    any in-process state-collision risk we'd take by reconstructing Robot()
    inside a single interpreter.
    """
    script = os.path.abspath(__file__)
    for role in ("policy", "sim"):
        print(f"\n[{role.upper()} PASS] launching child …", flush=True)
        rc = subprocess.run(
            [
                sys.executable,
                script,
                "--role", role,
                "--duration", str(duration),
                robot_ip,
            ],
            check=False,
        ).returncode
        if rc != 0:
            print(f"[{role.upper()} PASS] child exited with code {rc}", file=sys.stderr)
            return rc
    return 0


def run_capture(role: str, duration: float, robot_ip: str) -> int:
    is_sim = role == "sim"

    # Mirror simulator.py's MROS_IP_LIST derivation so the SDK joins the right
    # subnet. Required for the local loopback path; without it `robot.init()`
    # can hang.
    parts = robot_ip.split(".")
    os.environ.setdefault("MROS_IP_LIST", f"{parts[0]}.{parts[1]}.{parts[2]}.x")

    # Second positional arg = sim-peer flag. False (default) = policy peer.
    robot = Robot(RobotType.Humanoid, is_sim)
    if not robot.init(robot_ip):
        print("ERROR: robot.init failed — is the launcher running?", file=sys.stderr)
        return 1

    states: Deque[tuple[float, dt.RobotState]] = deque()
    imus: Deque[tuple[float, dt.ImuData]] = deque()
    cmds: Deque[tuple[float, dt.RobotCmd]] = deque()

    def on_state(s: dt.RobotState) -> None:
        states.append((time.time(), s))

    def on_imu(i: dt.ImuData) -> None:
        imus.append((time.time(), i))

    def on_cmd(c: dt.RobotCmd) -> None:
        cmds.append((time.time(), c))

    # State + IMU subscribe is kept for both roles so the empty case stays
    # empirically visible (annotated in the print below). The bus silently
    # delivers zero in sim role.
    robot.subscribeRobotState(on_state)
    robot.subscribeImuData(on_imu)

    if is_sim:
        # ForSim variant: delivers cmd packets the deploy stack sent to the
        # sim peer. Same variant used by simulator.py itself.
        robot.subscribeRobotCmdForSim(on_cmd)
    else:
        # Plain variant: self-loopback. Sees only cmds *we* published, which
        # this probe never does → 0 samples (expected; informational).
        robot.subscribeRobotCmd(on_cmd)

    print(f"role={role}  ip={robot_ip}  duration={duration:.1f}s")
    print(f"collecting …")
    time.sleep(duration)

    def hz(samples: Deque) -> float:
        if len(samples) < 2:
            return 0.0
        return (len(samples) - 1) / (samples[-1][0] - samples[0][0])

    print()
    print("=" * 72)
    print(f"LimX MROS wire contract — observed (role={role})")
    print("=" * 72)

    # ---- RobotState ---------------------------------------------------------
    if states:
        _, s = states[0]
        print("\n[RobotState]")
        print(f"  samples              : {len(states)}  (observed rate ≈ {hz(states):.1f} Hz)")
        print(f"  q length             : {len(s.q)}")
        print(f"  dq length            : {len(s.dq)}")
        print(f"  tau length           : {len(s.tau)}")
        print(f"  motor_names length   : {len(s.motor_names)}")
        print(f"  motor_names          :")
        for i, name in enumerate(s.motor_names):
            print(f"    [{i:2d}] {name}")
    else:
        print("\n[RobotState] no samples received")
        if is_sim:
            print("  → expected for role=sim. Sim peers publish state; they do not")
            print("    subscribe. Run --role policy (or --role both) to capture it.")

    # ---- ImuData ------------------------------------------------------------
    if imus:
        _, i = imus[0]
        print("\n[ImuData]")
        print(f"  samples              : {len(imus)}  (observed rate ≈ {hz(imus):.1f} Hz)")
        print(f"  quat                 : {list(i.quat)}")
        print(f"  gyro                 : {list(i.gyro)}")
        print(f"  acc                  : {list(i.acc)}")
    else:
        print("\n[ImuData] no samples received")
        if is_sim:
            print("  → expected for role=sim. Same reason as RobotState above.")

    # ---- RobotCmd -----------------------------------------------------------
    if cmds:
        _, c = cmds[0]
        print("\n[RobotCmd]")
        print(f"  samples              : {len(cmds)}  (observed rate ≈ {hz(cmds):.1f} Hz)")
        print(f"  q length             : {len(c.q)}")
        print(f"  Kp length / sample   : {len(c.Kp)} / {list(c.Kp)[:6]} …")
        print(f"  Kd length / sample   : {len(c.Kd)} / {list(c.Kd)[:6]} …")
        print(f"  mode length / sample : {len(c.mode)} / {list(c.mode)[:6]} …")
        # motor_names on cmd is only set by some controllers; guard the access.
        names = list(getattr(c, "motor_names", []) or [])
        print(f"  motor_names length   : {len(names)}")
        if names:
            print(f"  motor_names sample   : {names[:6]} …")
    else:
        print("\n[RobotCmd] no samples received")
        if not is_sim:
            print("  → expected for role=policy. `subscribeRobotCmd` is a self-loopback —")
            print("    it only delivers cmds *we* published, never cmds from other peers.")
            print("    Run --role sim (or --role both) to capture it.")

    return 0


def main() -> int:
    args = parse_args()
    if args.role == "both":
        return run_both(args.duration, args.robot_ip)
    return run_capture(args.role, args.duration, args.robot_ip)


if __name__ == "__main__":
    sys.exit(main())
