"""TDD for WalkPolicy — param load, history ring, torque-clamp resolution, ONNX step.

Loads the real `walk_param.yaml` + `policy.onnx` (no mocks). Pins the params against
the YAML, the history ring against the deploy's newest-first replicate-×5 behavior,
the torque-clamp against its |τ|≤0.95·τlim bound, and a full step against the live
ONNX. Pure: runs in the `brain` env.
"""

import numpy as np
import pytest

from humanoid.logic.oli import Intent, Mode, Observation, PolicyIn, PolicyOut
from humanoid.logic.oli.action.policy_runner import WalkPolicy

pytestmark = pytest.mark.brain

N = 31


def _obs(stamp=1, q=None, dq=None):
    q = np.zeros(N, dtype=np.float32) if q is None else np.asarray(q, dtype=np.float32)
    dq = np.zeros(N, dtype=np.float32) if dq is None else np.asarray(dq, dtype=np.float32)
    return Observation(
        stamp_ns=stamp, q=q, dq=dq, tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.zeros(3, dtype=np.float32),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


@pytest.fixture(scope="module")
def walk():
    return WalkPolicy()  # default walk_param.yaml + policy.onnx


def test_walk_params_loaded(walk):
    assert walk.kp.shape == (N,) and walk.default_angle.shape == (N,)
    assert walk.default_angle[3] == pytest.approx(0.3)      # left_knee
    assert walk.kp[3] == pytest.approx(139.41)
    assert walk.kd[3] == pytest.approx(17.75)
    assert walk.action_scale[4] == pytest.approx(0.1121)    # left_ankle_pitch
    assert walk.user_torque_limit[4] == pytest.approx(80.0)
    assert walk.decimation == 10


def test_history_first_obs_replicated_5x(walk):
    walk.reset()
    obs102 = np.arange(102, dtype=np.float32)
    h = walk._advance_history(obs102)
    assert h.shape == (510,)
    for k in range(5):
        np.testing.assert_allclose(h[k * 102:(k + 1) * 102], obs102)


def test_history_shifts_newest_first(walk):
    walk.reset()
    a = np.full(102, 1.0, dtype=np.float32)
    b = np.full(102, 2.0, dtype=np.float32)
    walk._advance_history(a)
    h = walk._advance_history(b)
    np.testing.assert_allclose(h[0:102], b)       # newest at front
    np.testing.assert_allclose(h[102:204], a)     # previous behind it


def test_resolve_torque_clamps_q_des(walk):
    obs = _obs(q=np.zeros(N), dq=np.zeros(N))
    huge = np.full(N, 1e3, dtype=np.float32)       # absurd raw actions
    clamped, q_des = walk._resolve(huge, obs)
    tau = walk.kp * q_des                           # q=dq=0 → τ = kp·q_des
    assert np.all(np.abs(tau) <= 0.95 * walk.user_torque_limit + 1e-2)


def test_step_returns_resolved_policyout(walk):
    walk.reset()
    obs = _obs(stamp=12345, q=walk.default_angle)
    out = walk.step(PolicyIn(observation=obs, intent=Intent(mode=Mode.WALK, v_x=0.3)))
    assert isinstance(out, PolicyOut)
    assert out.stamp_ns == 12345
    assert out.q_des.shape == (N,)
    np.testing.assert_allclose(out.kp, walk.kp, atol=1e-3)
    np.testing.assert_allclose(out.kd, walk.kd, atol=1e-3)
    assert np.isfinite(out.q_des).all()


def test_step_updates_last_actions_to_clamped(walk):
    walk.reset()
    obs = _obs(stamp=1, q=walk.default_angle)
    out = walk.step(PolicyIn(observation=obs, intent=Intent(mode=Mode.WALK)))
    # aliasing fidelity: last_actions == the clamped action == (q_des − default)/scale
    expected = (out.q_des - walk.default_angle) / walk.action_scale
    np.testing.assert_allclose(walk._last_actions, expected, atol=1e-4)
