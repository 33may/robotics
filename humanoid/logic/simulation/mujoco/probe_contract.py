#!/usr/bin/env python3
"""
One-shot wire-contract probe for the LimX MROS bus.

Run this *while* the launcher is up (sim + deploy + a stand/walk/mimic ability
actively publishing). It subscribes to RobotState, ImuData, and RobotCmd; waits
~3 seconds collecting samples; then prints:

  - first RobotState (motor_names + lengths of q/dq/tau)
  - first ImuData (quat + gyro + acc shape)
  - first RobotCmd (mode/Kp/Kd lengths + Kp/Kd sample values to infer mode)
  - observed publish rate per topic (Hz)

Paste the output into `humanoid/docs/vendor/humanoid-rl-deploy-python.md § 11`.

Run:
  conda activate limx
  python humanoid/logic/simulation/mujoco/probe_contract.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Deque

import limxsdk.datatypes as dt
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType


def main() -> int:
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    # mirror simulator.py's MROS_IP_LIST derivation
    parts = robot_ip.split(".")
    os.environ.setdefault("MROS_IP_LIST", f"{parts[0]}.{parts[1]}.{parts[2]}.x")

    # IMPORTANT: pass `True` for the second constructor arg → sim-peer role.
    # `subscribeRobotCmdForSim` is role-gated and only delivers to sim peers.
    # `Robot(RobotType.Humanoid)` (policy peer) would silently receive zero cmds.
    # See simulator.py — it constructs Robot(RobotType.Humanoid, True).
    robot = Robot(RobotType.Humanoid, True)
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

    robot.subscribeRobotState(on_state)
    robot.subscribeImuData(on_imu)
    # SDK separates roles: subscribeRobotCmd is the policy-side subscription
    # (you'd see cmds you yourself sent). To observe cmds from the deploy stack
    # as the *sim* would, use the ForSim variant — same as simulator.py does.
    robot.subscribeRobotCmdForSim(on_cmd)

    print("collecting for 3.0 s …")
    time.sleep(3.0)

    def hz(samples: Deque) -> float:
        if len(samples) < 2:
            return 0.0
        return (len(samples) - 1) / (samples[-1][0] - samples[0][0])

    print()
    print("=" * 72)
    print("LimX MROS wire contract — observed")
    print("=" * 72)

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

    if imus:
        _, i = imus[0]
        print("\n[ImuData]")
        print(f"  samples              : {len(imus)}  (observed rate ≈ {hz(imus):.1f} Hz)")
        print(f"  quat                 : {list(i.quat)}")
        print(f"  gyro                 : {list(i.gyro)}")
        print(f"  acc                  : {list(i.acc)}")
    else:
        print("\n[ImuData] no samples received")

    if cmds:
        _, c = cmds[0]
        print("\n[RobotCmd]")
        print(f"  samples              : {len(cmds)}  (observed rate ≈ {hz(cmds):.1f} Hz)")
        print(f"  q length             : {len(c.q)}")
        print(f"  Kp length / sample   : {len(c.Kp)} / {list(c.Kp)[:6]} …")
        print(f"  Kd length / sample   : {len(c.Kd)} / {list(c.Kd)[:6]} …")
        print(f"  mode length / sample : {len(c.mode)} / {list(c.mode)[:6]} …")
        print(f"  motor_names sample   : {list(c.motor_names)[:6]} …")
    else:
        print("\n[RobotCmd] no samples received")

    return 0


if __name__ == "__main__":
    sys.exit(main())
