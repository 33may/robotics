"""TDD for the walk observation encoder (the crux).

`encode_walk_obs` must reproduce `walk_controller.compute_observation` exactly: the
102-dim vector `[ang_vel·0.25 | projected_gravity | commands | (q−default)·1 |
dq·0.05 | last_actions]`, gait terms omitted. The reference here is an INDEPENDENT
reimplementation of the deploy math (incl. its quat→euler(zyx)→quat projected-gravity
detour) so the test pins behavior to the authoritative controller, not to my code.
Pure (numpy + scipy): runs in the `brain` env.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from humanoid.logic.oli import Intent, Mode, Observation
from humanoid.logic.oli.action.policy_runner import encode_walk_obs

pytestmark = pytest.mark.brain

N = 31

# walk_param.yaml default_angle, in PR-positional order (head 15/16 symmetric).
DEFAULT_ANGLE = np.array([
    -0.15, -0.00, -0.05, 0.3, -0.16, 0.0,
    -0.15, 0.00, 0.05, 0.3, -0.16, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0,
    0.1, 0.1, -0.2, -0.2, 0.0, 0.0, 0.0,
    0.1, -0.1, 0.2, -0.2, 0.0, 0.0, 0.0,
], dtype=np.float32)
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05


def _obs(stamp=1):
    rng = np.linspace(-0.4, 0.4, N).astype(np.float32)
    th = 0.3  # 0.3 rad pitch tilt so projected_gravity is non-degenerate
    quat_wxyz = np.array([np.cos(th / 2), 0.0, np.sin(th / 2), 0.0], dtype=np.float32)
    return Observation(
        stamp_ns=stamp, q=rng, dq=rng * 0.5, tau=np.zeros(N),
        acc=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        gyro=np.array([0.05, -0.1, 0.02], dtype=np.float32),
        quat_wxyz=quat_wxyz,
    )


def _reference_obs102(obs, intent, last_actions):
    """walk_controller.compute_observation, reimplemented verbatim (incl. detour)."""
    quat_xyzw = np.array(
        [obs.quat_wxyz[1], obs.quat_wxyz[2], obs.quat_wxyz[3], obs.quat_wxyz[0]]
    )
    q_wi = R.from_quat(quat_xyzw).as_euler("zyx")
    inv = R.from_euler("zyx", q_wi).inv().as_matrix()
    proj_grav = inv @ np.array([0.0, 0.0, -1.0])
    base_ang_vel = np.asarray(obs.gyro) * ANG_VEL_SCALE
    jpos = (np.asarray(obs.q) - DEFAULT_ANGLE) * 1.0
    jvel = np.asarray(obs.dq) * DOF_VEL_SCALE
    cmds = np.array([intent.v_x, intent.v_y, intent.w_z])
    return np.concatenate([base_ang_vel, proj_grav, cmds, jpos, jvel, last_actions])


def test_encode_walk_obs_is_102_dims():
    o = encode_walk_obs(_obs(), Intent(mode=Mode.WALK, v_x=0.5), np.zeros(N), DEFAULT_ANGLE)
    assert o.shape == (102,)  # 3+3+3+31+31+31, gait omitted


def test_encode_walk_obs_matches_walk_controller():
    obs = _obs()
    intent = Intent(mode=Mode.WALK, v_x=0.5, v_y=-0.2, w_z=0.1)
    last = np.linspace(-1, 1, N).astype(np.float32)
    got = encode_walk_obs(obs, intent, last, DEFAULT_ANGLE)
    exp = _reference_obs102(obs, intent, last)
    np.testing.assert_allclose(got, exp, atol=1e-5)


def test_encode_walk_obs_term_layout():
    obs = _obs()
    intent = Intent(mode=Mode.WALK, v_x=0.37, v_y=-0.11, w_z=0.05)
    o = encode_walk_obs(obs, intent, np.zeros(N), DEFAULT_ANGLE)
    # commands at [6:9], UNSCALED
    np.testing.assert_allclose(o[6:9], [0.37, -0.11, 0.05], atol=1e-6)
    # ang_vel at [0:3], scaled by 0.25
    np.testing.assert_allclose(o[0:3], np.array([0.05, -0.1, 0.02]) * 0.25, atol=1e-6)
    # dof_vel slice [40:71] scaled by 0.05 (after ang_vel 0:3, grav 3:6, cmd 6:9, q 9:40)
    np.testing.assert_allclose(o[40:71], np.asarray(obs.dq) * 0.05, atol=1e-5)


def test_encode_walk_obs_last_actions_tail():
    last = np.arange(N, dtype=np.float32)
    o = encode_walk_obs(_obs(), Intent(mode=Mode.WALK), last, DEFAULT_ANGLE)
    np.testing.assert_allclose(o[71:102], last, atol=1e-6)
