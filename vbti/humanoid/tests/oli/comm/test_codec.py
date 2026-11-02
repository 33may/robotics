"""TDD for the contract↔wire codec (humanoid.logic.oli.comm.codec).

The codec is the pure seam that maps the dataclass contracts onto the frozen wire:
Observation ↔ STATE_IMU, PolicyOut ↔ CMD. Round-tripping a contract through bytes
must preserve every field (float32 tolerance). Pure: runs in the `brain` env.
"""

import numpy as np
import pytest

from humanoid.logic.oli import NUM_JOINTS, Observation, PolicyOut
from humanoid.logic.oli.comm import codec
from humanoid.logic.oli.comm import protocol as p

pytestmark = pytest.mark.brain

N = NUM_JOINTS


def _obs():
    rng = np.arange(N, dtype=np.float32)
    return Observation(
        stamp_ns=987_654_321,
        q=rng * 0.01, dq=rng * -0.02, tau=rng * 0.03,
        acc=np.array([0.0, 0.1, -9.81], dtype=np.float32),
        gyro=np.array([0.3, -0.2, 0.1], dtype=np.float32),
        quat_wxyz=np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float32),
    )


def _pout():
    rng = np.arange(N, dtype=np.float32)
    return PolicyOut(
        stamp_ns=123_456,
        q_des=rng * 0.05, dq_des=np.zeros(N), tau_ff=np.zeros(N),
        kp=100.0 + rng, kd=1.0 + rng,
        mode=np.zeros(N, dtype=np.int32),
    )


def test_observation_round_trips_through_wire():
    obs = _obs()
    got = codec.decode_observation(codec.encode_observation(obs, seq=5))
    assert isinstance(got, Observation)
    assert got.stamp_ns == obs.stamp_ns
    for name in ("q", "dq", "tau", "acc", "gyro", "quat_wxyz"):
        np.testing.assert_allclose(getattr(got, name), getattr(obs, name), atol=1e-4)


def test_policy_out_round_trips_through_wire():
    po = _pout()
    got = codec.decode_policy_out(codec.encode_policy_out(po, seq=2))
    assert isinstance(got, PolicyOut)
    assert got.stamp_ns == po.stamp_ns
    for name in ("q_des", "dq_des", "tau_ff", "kp", "kd"):
        np.testing.assert_allclose(getattr(got, name), getattr(po, name), atol=1e-4)
    np.testing.assert_array_equal(got.mode, po.mode)


def test_encode_observation_is_a_state_imu_frame():
    buf = codec.encode_observation(_obs(), seq=1)
    assert len(buf) == p.STATE_IMU_MSG_SIZE
    # decodable by the raw wire as STATE_IMU
    _, _, q, *_ = p.unpack_state_imu(buf)
    assert len(q) == N


def test_encode_policy_out_is_a_cmd_frame():
    buf = codec.encode_policy_out(_pout(), seq=1)
    assert len(buf) == p.CMD_MSG_SIZE
    _, _, mode, *_ = p.unpack_cmd(buf)
    assert len(mode) == N
