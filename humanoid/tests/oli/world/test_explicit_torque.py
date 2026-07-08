"""Unit TDD for the explicit per-substep torque law (legged_gym / TRON1 reproduction).

`pd_torque` is the World's actuator math, pulled out pure so it can be checked without
isaacsim (oli.py defers all isaacsim imports). This pins the exact legged_gym form
`τ = kp(q_des−q) + kd(dq_des−dq) + tau_ff` that the Isaac World now recomputes every
physics substep under effort mode — the structural fix vs the implicit PhysX drive.
"""

import numpy as np
import pytest

from humanoid.logic.oli.contracts import NUM_JOINTS
from humanoid.logic.simulation.isaacsim.oli import pd_torque

pytestmark = pytest.mark.brain

N = NUM_JOINTS


def test_matches_legged_gym_compute_torques():
    q_des = np.full(N, 0.5, np.float32)
    q = np.full(N, 0.2, np.float32)
    dq = np.full(N, 0.1, np.float32)
    kp = np.full(N, 100.0, np.float32)
    kd = np.full(N, 2.0, np.float32)
    tau = pd_torque(q_des, q, np.zeros(N), dq, kp, kd, np.zeros(N))
    # τ = 100*(0.5-0.2) - 2*(0.1) = 30 - 0.2 = 29.8
    np.testing.assert_allclose(tau, 29.8, rtol=1e-5)


def test_at_target_with_zero_velocity_torque_is_feedforward_only():
    q = np.full(N, 0.3, np.float32)
    tau = pd_torque(q, q, np.zeros(N), np.zeros(N),
                    np.full(N, 139.0, np.float32), np.full(N, 17.0, np.float32),
                    np.full(N, 1.5, np.float32))
    np.testing.assert_allclose(tau, 1.5, atol=1e-5)  # only tau_ff survives


def test_velocity_term_damps_motion():
    # moving toward target with no position error → pure damping, opposes velocity
    q = np.zeros(N, np.float32)
    tau = pd_torque(q, q, np.zeros(N), np.full(N, 2.0, np.float32),
                    np.full(N, 50.0, np.float32), np.full(N, 5.0, np.float32), np.zeros(N))
    assert np.all(tau < 0)  # τ = -kd*dq = -10
    np.testing.assert_allclose(tau, -10.0, atol=1e-5)


def test_returns_float32_per_joint_vector():
    tau = pd_torque(np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N),
                    np.ones(N), np.ones(N), np.zeros(N))
    assert tau.shape == (N,)
    assert tau.dtype == np.float32
