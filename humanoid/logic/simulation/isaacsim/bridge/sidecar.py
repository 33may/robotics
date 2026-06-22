"""
sidecar.py — Py 3.8 process owning the limxsdk Robot in sim-peer role.

Forwards RobotCmd from the MROS bus → UDS as CMD frames, and STATE_IMU frames
from UDS → MROS as ``publishRobotStateForSim`` + ``publishImuDataForSim`` calls.

This module MUST run in the ``limx`` conda env (CPython 3.8.18 + limxsdk-4.0.1).
It MUST NOT import anything from Isaac Sim.

See:
  - design.md D1 (two-process architecture), D6 (handshake), D10 (parallel_solve_required)
  - spec.md "Sidecar runs only in the limx env" + "Bridge presents Isaac Sim as a sim peer"
  - wire reality at humanoid/docs/vendor/humanoid-rl-deploy-python.md § 11

Usage (standalone for smoke testing):

    /home/may33/miniconda3/envs/limx/bin/python \\
        /home/may33/projects/ml_portfolio/robotics/humanoid/logic/simulation/isaacsim/bridge/sidecar.py \\
        --ip 127.0.0.1 --socket /tmp/limx-isaac-bridge.sock [--debug]

Normally launched indirectly by ``OliBridge.spawn_sidecar(...)`` (Phase 6).
"""

from __future__ import annotations

import argparse
import logging
import os
import os.path
import selectors
import signal
import socket as _sock
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Bootstrap: when this file is run as a script, `import protocol` must work.
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
import protocol as p  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

# PR canonical joint order (HU_D04_01) — MAY-145 § 11. Used as `RobotState.motor_names`
# on every publish, so the deploy-side never sees an Isaac-flavored ordering.
PR_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]
assert len(PR_ORDER) == p.NUM_JOINTS == 31


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LimX MROS sim-peer sidecar")
    ap.add_argument("--ip", default="127.0.0.1",
                    help="MROS peer IP (default: 127.0.0.1)")
    ap.add_argument("--socket", default="/tmp/limx-isaac-bridge.sock",
                    help="AF_UNIX SEQPACKET socket path")
    ap.add_argument("--debug", action="store_true",
                    help="Verbose per-packet logging")
    return ap.parse_args()


def set_mros_ip_list(ip: str) -> str:
    """Set MROS_IP_LIST env var, matching humanoid-mujoco-sim/simulator.py:226."""
    parts = ip.split(".")
    if len(parts) != 4:
        raise ValueError(f"invalid IPv4 {ip!r}")
    val = f"{parts[0]}.{parts[1]}.{parts[2]}.x"
    os.environ["MROS_IP_LIST"] = val
    return val


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:  # noqa: C901 — single linear flow, easier kept whole
    args = parse_args()
    logging.basicConfig(
        format="[sidecar] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
        stream=sys.stderr,
    )
    log = logging.getLogger("sidecar")

    # MUST set MROS_IP_LIST BEFORE constructing Robot (the SDK reads it at ctor time)
    mros_val = set_mros_ip_list(args.ip)
    log.info("MROS_IP_LIST=%s", mros_val)

    # Lazy limxsdk import — keeps the module importable for offline tests
    try:
        from limxsdk.robot import Robot, RobotType  # type: ignore[import-not-found]
        from limxsdk import datatypes as dt  # type: ignore[import-not-found]
    except ImportError as e:
        log.error("limxsdk import failed: %s — sidecar must run in the limx conda env", e)
        return 1

    # ── Construct sim-peer Robot ────────────────────────────────────────────
    robot = Robot(RobotType.Humanoid, True)
    log.info("Robot(Humanoid, True) constructed")
    init_ret = robot.init(args.ip)
    log.info("robot.init(%s) returned %r", args.ip, init_ret)

    # Pre-allocate the working datatypes (reused on every publish)
    state = dt.RobotState()
    state.q = [0.0] * p.NUM_JOINTS
    state.dq = [0.0] * p.NUM_JOINTS
    state.tau = [0.0] * p.NUM_JOINTS
    state.motor_names = list(PR_ORDER)
    imu = dt.ImuData()

    # ── Open UDS server ─────────────────────────────────────────────────────
    sock_path = args.socket
    if os.path.exists(sock_path):
        log.warning("unlinking stale socket at %s", sock_path)
        os.unlink(sock_path)
    server = _sock.socket(_sock.AF_UNIX, _sock.SOCK_SEQPACKET)
    server.bind(sock_path)
    server.listen(1)
    log.info("UDS server listening at %s", sock_path)
    log.info("waiting for driver to connect ...")

    client, _peer = server.accept()
    log.info("driver connected")

    # ── HELLO handshake ─────────────────────────────────────────────────────
    hello_buf = client.recv(p.HELLO_MSG_SIZE)
    if len(hello_buf) != p.HELLO_MSG_SIZE:
        log.error("HELLO short read: %d bytes, expected %d",
                  len(hello_buf), p.HELLO_MSG_SIZE)
        client.close()
        server.close()
        os.unlink(sock_path)
        return 2
    try:
        _seq, dof_count, isaac_dof_names = p.unpack_hello(hello_buf)
    except p.ProtocolError as e:
        log.error("HELLO unpack failed: %s", e)
        client.close()
        server.close()
        os.unlink(sock_path)
        return 2

    if dof_count != p.NUM_JOINTS:
        log.error("HELLO dof_count mismatch: got %d, expected %d",
                  dof_count, p.NUM_JOINTS)
        client.close()
        server.close()
        os.unlink(sock_path)
        return 2

    pr_set = set(PR_ORDER)
    isaac_set = set(isaac_dof_names)
    if pr_set != isaac_set:
        log.error(
            "HELLO joint-name set mismatch. In Isaac but not PR: %s; "
            "in PR but not Isaac: %s",
            sorted(isaac_set - pr_set), sorted(pr_set - isaac_set),
        )
        client.close()
        server.close()
        os.unlink(sock_path)
        return 2

    log.info(
        "HELLO ok: dof_count=%d, joint names match PR canonical (driver handles permutation)",
        dof_count,
    )

    # Ack: 8-byte HELLO header, payload_len=0, seq=0
    client.send(p.pack_header(p.MsgType.HELLO, 0, 0))
    log.debug("HELLO ack sent")

    # ── Set client non-blocking; register cmd callback ──────────────────────
    client.setblocking(False)
    send_lock = threading.Lock()

    # State shared between callback thread and main thread.
    # Single-writer for each (callback writes seq/recv/drop; main writes state_pub).
    # GIL makes int writes atomic on CPython, so no torn reads — fine for stats.
    counters = {"cmd_seq": 0, "cmd_recv": 0, "cmd_drop": 0, "state_pub": 0}
    motor_names_verified = [False]  # mutable cell for nonlocal write in closure

    def on_cmd(cmd: Any) -> None:
        # NB: runs in an SDK-owned thread, NOT the main thread.
        if not motor_names_verified[0]:
            try:
                cmd_names = list(cmd.motor_names)
            except Exception:  # pragma: no cover
                cmd_names = []
            if cmd_names == PR_ORDER:
                log.info("first cmd motor_names exactly match PR canonical")
            elif set(cmd_names) == pr_set:
                log.info(
                    "first cmd motor_names set matches PR canonical "
                    "(order differs — that's fine, driver permutes)"
                )
            else:
                log.warning(
                    "first cmd motor_names set DIFFERS from PR canonical — "
                    "proceeding but joint permutation may be wrong"
                )
            motor_names_verified[0] = True

        try:
            buf = p.pack_cmd(
                seq=counters["cmd_seq"],
                stamp_ns=int(cmd.stamp),
                mode=list(cmd.mode),
                q=list(cmd.q),
                dq=list(cmd.dq),
                tau=list(cmd.tau),
                kp=list(cmd.Kp),
                kd=list(cmd.Kd),
                parallel_solve_required=list(cmd.parallel_solve_required),
            )
        except p.ProtocolError as e:
            log.error("cmd pack failed: %s", e)
            return

        try:
            with send_lock:
                client.send(buf)
            counters["cmd_seq"] += 1
            counters["cmd_recv"] += 1
            if args.debug:
                log.debug(
                    "→ CMD seq=%d stamp_ns=%d q[0]=%.4f kp[0]=%.2f",
                    counters["cmd_seq"] - 1, int(cmd.stamp),
                    cmd.q[0] if cmd.q else 0.0, cmd.Kp[0] if cmd.Kp else 0.0,
                )
        except BlockingIOError:
            counters["cmd_drop"] += 1
            if counters["cmd_drop"] % 100 == 1:
                log.warning("cmd drop count=%d (socket buffer full)",
                            counters["cmd_drop"])

    robot.subscribeRobotCmdForSim(on_cmd)
    log.info("subscribed to RobotCmdForSim; entering relay loop")

    # ── Signal handling ─────────────────────────────────────────────────────
    running = [True]  # mutable so the signal handler can flip it

    def _on_signal(signum: int, _frame: Any) -> None:
        log.info("signal %d received, shutting down", signum)
        running[0] = False

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # ── Main loop: select on client, decode STATE_IMU, publish ──────────────
    sel = selectors.DefaultSelector()
    sel.register(client, selectors.EVENT_READ)
    last_stats_t = time.monotonic()

    try:
        while running[0]:
            events = sel.select(timeout=1.0)
            for _key, _mask in events:
                try:
                    buf = client.recv(p.STATE_IMU_MSG_SIZE)
                except BlockingIOError:
                    continue
                if not buf:  # EOF
                    log.info("driver EOF — shutting down")
                    running[0] = False
                    break
                if len(buf) != p.STATE_IMU_MSG_SIZE:
                    log.warning("short STATE_IMU read: %d bytes", len(buf))
                    continue
                try:
                    (seq, stamp_ns, q, dq, tau,
                     acc, gyro, quat_wxyz) = p.unpack_state_imu(buf)
                except p.ProtocolError as e:
                    log.error("STATE_IMU unpack failed: %s", e)
                    continue

                # RobotState q/dq/tau are std::vector<float> — mutate IN PLACE
                # (element-wise), NOT wholesale `state.q = list`. Wholesale
                # rebinds the Python attr without updating the C++ vector, so the
                # published state carries empty vectors and subscribers drop it.
                # IMU fields are fixed C arrays, so wholesale assignment is fine.
                # (matches humanoid-mujoco-sim/simulator.py element-wise pattern)
                for i in range(p.NUM_JOINTS):
                    state.q[i] = q[i]
                    state.dq[i] = dq[i]
                    state.tau[i] = tau[i]
                state.stamp = stamp_ns

                imu.acc = acc
                imu.gyro = gyro
                imu.quat = quat_wxyz
                imu.stamp = stamp_ns

                # MuJoCo reference publishes IMU first, then state — keep order
                robot.publishImuDataForSim(imu)
                robot.publishRobotStateForSim(state)
                counters["state_pub"] += 1

                if args.debug and counters["state_pub"] % 500 == 1:
                    log.debug(
                        "← STATE_IMU seq=%d q[0]=%.4f acc=%s quat_wxyz=%s",
                        seq, q[0], acc, quat_wxyz,
                    )

            now = time.monotonic()
            if now - last_stats_t > 5.0:
                log.info(
                    "stats: cmd_recv=%d cmd_drop=%d state_pub=%d",
                    counters["cmd_recv"], counters["cmd_drop"],
                    counters["state_pub"],
                )
                last_stats_t = now
    finally:
        log.info(
            "final stats: cmd_recv=%d cmd_drop=%d state_pub=%d",
            counters["cmd_recv"], counters["cmd_drop"], counters["state_pub"],
        )
        try:
            client.close()
        except Exception:  # pragma: no cover
            pass
        try:
            server.close()
        except Exception:  # pragma: no cover
            pass
        try:
            if os.path.exists(sock_path):
                os.unlink(sock_path)
                log.debug("unlinked %s", sock_path)
        except Exception:  # pragma: no cover
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
