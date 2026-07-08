"""
fake_driver_smoke.py — minimal stand-in for the real Isaac driver.

Connects to the sidecar's UDS socket, completes HELLO, sends a few STATE_IMU
frames, then closes. Used in the Phase 4 smoke test to prove that:
  - the sidecar's UDS accept + HELLO recv work
  - the HELLO ack is delivered correctly
  - STATE_IMU frames travel sidecar-bound without IPC errors
  - clean EOF shutdown works on both ends

The fake driver does NOT need limxsdk or Isaac; it's pure stdlib + protocol.py.
Runs in any Python 3.8+ interpreter.

Usage:
  # In terminal A:
  /home/may33/miniconda3/envs/limx/bin/python <path>/bridge/sidecar.py --debug
  # In terminal B (after sidecar prints "waiting for driver to connect"):
  /home/may33/miniconda3/envs/limx/bin/python <path>/fake_driver_smoke.py
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

BRIDGE_DIR = (
    Path("/home/may33/projects/ml_portfolio/robotics/humanoid/logic/simulation/"
         "isaacsim/bridge")
)
sys.path.insert(0, str(BRIDGE_DIR))

import protocol as p  # noqa: E402
import sidecar         # noqa: E402  — we only need its PR_ORDER constant


SOCKET_PATH = "/tmp/limx-isaac-bridge.sock"
N_TICKS = 50  # send this many STATE_IMU frames before disconnecting


def main() -> int:
    # Use Isaac's expected order (PR canonical for the smoke — the sidecar
    # only checks set-equality, so any permutation of these 31 names works).
    isaac_dof_names = list(sidecar.PR_ORDER)

    print(f"[fake_driver] connecting to {SOCKET_PATH}")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.connect(SOCKET_PATH)
    print("[fake_driver] connected")

    # Send HELLO
    hello = p.pack_hello(seq=1, dof_names=isaac_dof_names)
    sock.send(hello)
    print(f"[fake_driver] sent HELLO ({len(hello)} bytes)")

    # Wait for ack (header-only, 8 bytes)
    ack = sock.recv(p.HEADER_SIZE)
    msg_type, version, payload_len, seq = p.unpack_header(ack)
    print(f"[fake_driver] ack: type={msg_type.name} version={version} "
          f"payload_len={payload_len} seq={seq}")
    assert msg_type is p.MsgType.HELLO and payload_len == 0, ack

    # Send a handful of STATE_IMU frames at ~50 Hz
    zeros31 = [0.0] * p.NUM_JOINTS
    acc = [0.0, 0.0, 9.81]
    gyro = [0.0, 0.0, 0.0]
    quat_wxyz = [1.0, 0.0, 0.0, 0.0]

    t_start = time.monotonic()
    for tick in range(N_TICKS):
        stamp_ns = time.time_ns()
        # Wiggle q[0] so we can verify the values arrive on the bus side.
        q = list(zeros31)
        q[0] = 0.01 * (tick % 10)
        buf = p.pack_state_imu(
            seq=tick,
            stamp_ns=stamp_ns,
            q=q, dq=zeros31, tau=zeros31,
            acc=acc, gyro=gyro, quat_wxyz=quat_wxyz,
        )
        sock.send(buf)
        time.sleep(0.02)  # ~50 Hz

    dt_ms = (time.monotonic() - t_start) * 1000
    print(f"[fake_driver] sent {N_TICKS} STATE_IMU frames in {dt_ms:.1f} ms")

    # Clean shutdown — sidecar should see EOF.
    sock.close()
    print("[fake_driver] socket closed — sidecar should now exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
