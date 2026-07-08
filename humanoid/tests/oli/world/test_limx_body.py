"""TDD for LimxBody — the bus-backed World body (a limxsdk policy peer) behind WorldComm.

`LimxBody` implements the duck-typed World body protocol (`dof_names`,
`read_joints_isaac`, `read_imu`, `apply_isaac`) over a limxsdk `Robot` policy peer, so
`WorldComm` drives the MuJoCo (and later real) World exactly like the Isaac `Oli`. We
inject a `FakeRobot` (no limxsdk) so this is a pure brain-env test.

Mirrors the loopback's `FakeBody` trick: SDK motor order = reversed `PR_ORDER`, a
non-trivial permutation, so the body must pass arrays through in NATIVE (motor) order
and leave PR↔native to `WorldComm`'s name-based permutation.
"""

import numpy as np
import pytest

from humanoid.logic.oli.contracts import NUM_JOINTS, PR_ORDER

pytestmark = pytest.mark.brain

N = NUM_JOINTS


class FakeRobotState:
    __slots__ = ("stamp", "q", "dq", "tau", "motor_names")

    def __init__(self, stamp, q, dq, tau, motor_names):
        self.stamp, self.q, self.dq, self.tau = stamp, q, dq, tau
        self.motor_names = motor_names


class FakeImuData:
    __slots__ = ("stamp", "acc", "gyro", "quat")

    def __init__(self, acc, gyro, quat, stamp=0):
        self.acc, self.gyro, self.quat, self.stamp = acc, gyro, quat, stamp


class FakeRobotCmd:
    """Mirrors limxsdk datatypes.RobotCmd's public slots."""

    def __init__(self):
        self.stamp = None
        self.mode = None
        self.q = self.dq = self.tau = None
        self.Kp = self.Kd = None
        self.motor_names = None
        self.parallel_solve_required = None


class FakeDatatypes:
    RobotCmd = FakeRobotCmd


class FakeRobot:
    """Stand-in for a limxsdk policy-role `Robot`. SDK order = reversed PR."""

    def __init__(self):
        self._names = list(reversed(PR_ORDER))
        self._state_cb = None
        self._imu_cb = None
        self.published = []

    def getMotorNames(self):
        return list(self._names)

    def getMotorNumber(self):
        return len(self._names)

    def subscribeRobotState(self, cb):
        self._state_cb = cb

    def subscribeImuData(self, cb):
        self._imu_cb = cb

    def publishRobotCmd(self, cmd):
        self.published.append(cmd)

    # — test helpers: emulate the SDK thread delivering a sample —
    def push_state(self, rs):
        self._state_cb(rs)

    def push_imu(self, imu):
        self._imu_cb(imu)


def _body():
    from humanoid.logic.simulation.mujoco.limx_body import LimxBody

    robot = FakeRobot()
    body = LimxBody(robot, datatypes=FakeDatatypes)
    return robot, body


def test_dof_names_are_sdk_motor_order():
    _robot, body = _body()
    assert list(body.dof_names) == list(reversed(PR_ORDER))


def test_read_joints_returns_latest_state_native_order():
    robot, body = _body()
    q = list(np.arange(N, dtype=float))
    dq = list(np.arange(N, dtype=float) + 100.0)
    tau = list(np.arange(N, dtype=float) + 200.0)
    robot.push_state(FakeRobotState(42, q, dq, tau, robot.getMotorNames()))
    gq, gdq, gtau = body.read_joints_isaac()
    np.testing.assert_allclose(gq, q)
    np.testing.assert_allclose(gdq, dq)
    np.testing.assert_allclose(gtau, tau)
    assert body.latest_stamp_ns() == 42


def test_read_imu_passthrough_wxyz():
    robot, body = _body()
    robot.push_imu(FakeImuData([0.0, 0.1, -9.81], [0.3, -0.2, 0.1], [1.0, 0.0, 0.0, 0.0]))
    acc, gyro, quat = body.read_imu()
    np.testing.assert_allclose(acc, [0.0, 0.1, -9.81])
    np.testing.assert_allclose(gyro, [0.3, -0.2, 0.1])
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_apply_builds_native_order_robotcmd():
    robot, body = _body()
    q_des = np.arange(N, dtype=np.float32)
    body.apply_isaac(
        q_des=q_des,
        dq_des=np.zeros(N, dtype=np.float32),
        tau_ff=np.zeros(N, dtype=np.float32),
        kp=np.full(N, 50.0, dtype=np.float32),
        kd=np.full(N, 2.0, dtype=np.float32),
    )
    assert len(robot.published) == 1
    cmd = robot.published[0]
    assert list(cmd.mode) == [0] * N
    assert list(cmd.parallel_solve_required) == [True] * N
    assert list(cmd.motor_names) == robot.getMotorNames()
    np.testing.assert_allclose(cmd.q, q_des)
    np.testing.assert_allclose(cmd.dq, np.zeros(N))
    np.testing.assert_allclose(cmd.Kp, np.full(N, 50.0))
    np.testing.assert_allclose(cmd.Kd, np.full(N, 2.0))
    assert len(cmd.q) == N and len(cmd.Kp) == N and len(cmd.Kd) == N


def test_not_ready_until_first_state_and_imu():
    robot, body = _body()
    assert not body.ready()
    robot.push_state(FakeRobotState(1, [0.0] * N, [0.0] * N, [0.0] * N, robot.getMotorNames()))
    assert not body.ready()  # imu still missing
    robot.push_imu(FakeImuData([0.0] * 3, [0.0] * 3, [1.0, 0.0, 0.0, 0.0]))
    assert body.ready()
