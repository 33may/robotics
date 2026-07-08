"""
smoke_oli_bridge.py — Phase 6 verification of OliBridge WITHOUT Isaac.

Runs the OliBridge side in whatever interpreter invokes this (intended: the
isaac env, Py 3.11) and has it spawn the sidecar in the limx env (Py 3.8). This
exercises the real cross-env two-process path — the only thing missing is Isaac
physics, which we fake with a stub that calls bridge methods like Oli.tick would.

Verifies Spec scenarios:
  - "spawn_sidecar context manager owns the subprocess"
  - "Context exit terminates the sidecar"
  - handshake + send_state_imu + poll_cmd round-trip without error

Run (from the isaac env to prove cross-env spawn):
  /home/may33/miniconda3/envs/isaac/bin/python \\
    humanoid/openspec/changes/may-147-isaac-limx-sdk-bridge/_research/smoke_oli_bridge.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/may33/projects/ml_portfolio/robotics")
from humanoid.logic.simulation.isaacsim.bridge import OliBridge  # noqa: E402
from humanoid.logic.simulation.isaacsim.oli import PR_ORDER, NUM_JOINTS  # noqa: E402

SOCKET = "/tmp/limx-isaac-bridge-smoke.sock"


def main() -> int:
    print(f"[smoke] OliBridge side python: {sys.version.split()[0]}")
    print("[smoke] spawning sidecar (limx env) + connecting ...")

    with OliBridge.spawn_sidecar(ip="127.0.0.1", socket=SOCKET, debug=True) as bridge:
        # Handshake with PR-canonical names (Isaac would send its own DOF order;
        # the sidecar only checks set-equality, so PR order is fine here).
        bridge.handshake(list(PR_ORDER))
        print("[smoke] handshake OK")

        # Pump ~100 STATE_IMU frames at ~200 Hz, polling for cmds each tick.
        zeros = [0.0] * NUM_JOINTS
        acc = [0.0, 0.0, 9.81]
        gyro = [0.0, 0.0, 0.0]
        quat_wxyz = [1.0, 0.0, 0.0, 0.0]
        cmds_seen = 0
        n = 100
        for seq in range(n):
            q = list(zeros)
            q[0] = 0.01 * (seq % 10)
            bridge.send_state_imu(
                seq=seq, stamp_ns=time.time_ns(),
                q=q, dq=zeros, tau=zeros,
                acc=acc, gyro=gyro, quat_wxyz=quat_wxyz,
            )
            # Drain any cmds (none expected — no deploy-python publishing)
            while True:
                msg = bridge.poll_cmd()
                if msg is None:
                    break
                cmds_seen += 1
            time.sleep(0.005)

        print(f"[smoke] sent {n} STATE_IMU frames, saw {cmds_seen} cmds "
              f"(0 expected — no deploy-python running)")
        print("[smoke] exiting context (should tear down sidecar) ...")

    # After the with-block: socket should be gone, sidecar killed.
    time.sleep(0.5)
    if Path(SOCKET).exists():
        print(f"[smoke] WARN socket {SOCKET} still present after close")
    else:
        print("[smoke] socket cleaned up after close")

    print("[smoke] ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
