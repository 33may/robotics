"""limx_body.py — `LimxBody`, the bus-backed World body (a limxsdk policy peer).

`LimxBody` is the World's body for the MuJoCo (and later the real-robot) path. It
implements the same duck-typed body protocol the Isaac `Oli` does, so `WorldComm`
drives it identically — the only world-aware code is still `WorldComm`'s name-based
PR↔native permutation. The PR↔AB parallel-mechanism math is NOT here: it lives in the
unchanged sim process (the `kinematic_projection` ELF), behind the bus. This body sees
only PR-space `RobotState`/`RobotCmd`.

Body protocol (native = limxsdk motor order):
    body.dof_names                                  -> list[str]
    body.read_joints_isaac()                        -> (q, dq, tau)   each (num_motor,)
    body.read_imu()                                 -> (acc, gyro, quat_wxyz)
    body.apply_isaac(q_des, dq_des, tau_ff, kp, kd)                  each (num_motor,)
    body.latest_stamp_ns()                          -> int (bus/sim time, for D8 pacing)
    body.ready()                                    -> bool (first state+imu received)

Role (memory `limx-sdk-role-gating`): this is the POLICY peer (`is_sim=False`), a
drop-in for the deploy `walk_controller` minus the ONNX. `limxsdk` is imported lazily
in `connect()` so the module stays importable (and unit-testable with a fake) in the
brain env; only the live edge process (py3.8 `limx`) actually pulls the wheel.

References: design.md D4 (Comm owns the permutation), D8 (stamp = sim/bus time);
spec `docs/superpowers/specs/2026-06-25-mujoco-limx-world-design.md` (LD1, LD2).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

_DEFAULT_ROBOT_IP = "127.0.0.1"


class LimxBodyError(RuntimeError):
    """Raised on a missing first sample or a failed limxsdk init."""


class LimxBody:
    """World body backed by a limxsdk `Robot` policy peer (duck-typed, injected)."""

    def __init__(self, robot, *, datatypes) -> None:
        """`robot`: a limxsdk `Robot` (policy peer); `datatypes`: the module exposing
        `RobotCmd` (real `limxsdk.datatypes`, or a fake in tests). Subscriptions are
        registered here; the SDK delivers samples on its own thread (store-latest)."""
        self._robot = robot
        self._dt = datatypes
        self._motor_names: List[str] = list(robot.getMotorNames())
        self._n = len(self._motor_names)
        self._rs = None  # latest RobotState (overwritten by the SDK thread)
        self._imu = None  # latest ImuData
        robot.subscribeRobotState(self._on_state)
        robot.subscribeImuData(self._on_imu)

    # ── SDK-thread callbacks: store the latest sample only ───────────────────
    def _on_state(self, rs) -> None:
        self._rs = rs

    def _on_imu(self, imu) -> None:
        self._imu = imu

    # ── Body protocol ────────────────────────────────────────────────────────
    @property
    def dof_names(self) -> List[str]:
        return list(self._motor_names)

    def ready(self) -> bool:
        return self._rs is not None and self._imu is not None

    def latest_stamp_ns(self) -> int:
        rs = self._rs
        if rs is None:
            raise LimxBodyError("no RobotState received yet")
        return int(rs.stamp)

    def read_joints_isaac(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rs = self._rs
        if rs is None:
            raise LimxBodyError("no RobotState received yet")
        q = np.asarray(rs.q, dtype=np.float32)
        dq = np.asarray(rs.dq, dtype=np.float32)
        tau = np.asarray(rs.tau, dtype=np.float32)
        return q, dq, tau

    def read_imu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        imu = self._imu
        if imu is None:
            raise LimxBodyError("no ImuData received yet")
        acc = np.asarray(imu.acc, dtype=np.float32)
        gyro = np.asarray(imu.gyro, dtype=np.float32)
        quat_wxyz = np.asarray(imu.quat, dtype=np.float32)  # bus is already (w,x,y,z)
        return acc, gyro, quat_wxyz

    def apply_isaac(
        self,
        q_des: Sequence[float],
        dq_des: Sequence[float],
        tau_ff: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
    ) -> None:
        """Build a PR-mode `RobotCmd` (native order) and publish it to the bus.

        Inputs arrive in NATIVE (motor) order — `WorldComm` already permuted PR→native.
        `mode=0` is the per-joint torque-position hybrid the deploy uses; the ELF on the
        far side projects PR→AB. Must be called every edge tick (cmd is a streaming
        setpoint, not latched) — the 1 kHz republish is owned by the edge loop (LD3).
        """
        cmd = self._dt.RobotCmd()
        cmd.stamp = self._now_ns()
        cmd.mode = [0] * self._n
        cmd.q = _as_list(q_des, self._n, "q_des")
        cmd.dq = _as_list(dq_des, self._n, "dq_des")
        cmd.tau = _as_list(tau_ff, self._n, "tau_ff")
        cmd.Kp = _as_list(kp, self._n, "kp")
        cmd.Kd = _as_list(kd, self._n, "kd")
        cmd.motor_names = list(self._motor_names)
        cmd.parallel_solve_required = [True] * self._n
        self._robot.publishRobotCmd(cmd)

    @staticmethod
    def _now_ns() -> int:
        import time

        return time.time_ns()


def _as_list(x: Sequence[float], n: int, name: str) -> List[float]:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.shape != (n,):
        raise LimxBodyError(f"{name} must be length {n}, got {a.shape}")
    return [float(v) for v in a]


def connect(robot_ip: str = _DEFAULT_ROBOT_IP) -> LimxBody:
    """Build a live `LimxBody`: construct a policy-role limxsdk `Robot`, init the bus,
    and return the body. Imports `limxsdk` lazily (py3.8 `limx` env only)."""
    import limxsdk.datatypes as datatypes
    import limxsdk.robot.Robot as Robot
    import limxsdk.robot.RobotType as RobotType

    robot = Robot(RobotType.Humanoid)  # is_sim=False → POLICY role
    if not robot.init(robot_ip):
        raise LimxBodyError(f"limxsdk robot.init({robot_ip!r}) failed — is the sim up?")
    return LimxBody(robot, datatypes=datatypes)
