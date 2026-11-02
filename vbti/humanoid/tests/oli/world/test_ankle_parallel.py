"""Unit TDD for the faithful dual-motor achilles ankle emulation (`--ankle-parallel`).

The walk policy trains/deploys against the PARALLEL achilles ankle: two motors A/B per
foot, PD in MOTOR space at kp=93.65/kd=11.92, each clipped to ±42 N·m, coupled to the
serial (pitch,roll) joint by the linkage Jacobian J = ∂(A,B)/∂(pitch,roll). Our Isaac
model has a single SERIAL ankle. `ankle_parallel.ankle_joint_torque` reproduces the real
per-motor law on the serial joint:

    e_m = J (q_des − q);  tau_m = clip(kp·e_m + kd·J(dq̇_des−dq̇), ±effort);  tau_q = Jᵀ tau_m

so the per-motor clip (before Jᵀ) is applied faithfully — the shared-authority nonlinearity
a scalar --ankle-kp-scale cannot express. J measured in MuJoCo (walkmatch/ankle_jacobian.py):
coupling ≈0, so the small-signal stiffness is ~diagonal ×2.30 pitch / ×2.04 roll.

Pure (numpy only; oli.py defers isaacsim), so runs in `brain`.
"""
import numpy as np
import pytest

from humanoid.logic.simulation.isaacsim.ankle_parallel import (
    J_LEFT, J_RIGHT, KP_MOTOR, KD_MOTOR, EFFORT_MOTOR, I_MOTOR,
    reflected_inertia, ankle_joint_torque, apply_to_torque_vector,
)

pytestmark = pytest.mark.brain


def test_jacobians_are_mirror_pairs():
    # right ankle mirrors left: the pitch (differential) column flips sign, roll (additive) same
    np.testing.assert_allclose(J_RIGHT[:, 0], -J_LEFT[:, 0], atol=1e-3)
    np.testing.assert_allclose(J_RIGHT[:, 1], J_LEFT[:, 1], atol=1e-3)


def test_small_signal_stiffness_is_diagonal_x230_x204():
    # K/kp = J^T J should be ~diag(2.30, 2.04) with ~zero coupling (measured F11)
    JTJ = J_LEFT.T @ J_LEFT
    assert JTJ[0, 0] == pytest.approx(2.30, abs=0.05)
    assert JTJ[1, 1] == pytest.approx(2.04, abs=0.05)
    assert abs(JTJ[0, 1]) < 0.02  # coupling ~0


def test_reflected_inertia_is_two_motors_through_the_linkage():
    # diag(J^T diag(I,I) J) = I * diag(J^T J) ≈ 0.1845504 * [2.30, 2.04]
    Iq = reflected_inertia(J_LEFT)
    assert Iq[0] == pytest.approx(I_MOTOR * 2.297, abs=0.01)  # pitch ≈ 0.424
    assert Iq[1] == pytest.approx(I_MOTOR * 2.038, abs=0.01)  # roll  ≈ 0.376


def test_at_target_zero_velocity_gives_zero_torque():
    q = np.array([-0.16, 0.0])
    tau = ankle_joint_torque(q, np.zeros(2), q, np.zeros(2), J_LEFT)
    np.testing.assert_allclose(tau, 0.0, atol=1e-5)


def test_small_pitch_error_gives_mostly_pitch_torque():
    # unsaturated small error → tau ≈ kp * (J^T J) @ err ; coupling ~0 so roll torque ~0
    err = np.array([0.01, 0.0])  # +0.01 rad pitch error, 0 roll
    tau = ankle_joint_torque(np.zeros(2), np.zeros(2), err, np.zeros(2), J_LEFT)
    expect_pitch = KP_MOTOR * (J_LEFT.T @ J_LEFT)[0, 0] * 0.01  # ≈ 93.65*2.297*0.01 ≈ 2.15
    assert tau[0] == pytest.approx(expect_pitch, rel=0.02)
    assert abs(tau[1]) < 0.05  # negligible roll torque from a pitch error


def test_pure_pitch_saturation_uses_both_motors_differentially():
    # huge pitch error → motors saturate ±EFFORT (differential) → tau_pitch ≈ 2*lever*42, roll~0
    tau = ankle_joint_torque(np.zeros(2), np.zeros(2), np.array([5.0, 0.0]), np.zeros(2), J_LEFT)
    assert tau[0] == pytest.approx(2 * 1.072 * EFFORT_MOTOR, rel=0.02)  # ≈ 90 N·m
    assert abs(tau[1]) < 1.0


def test_pure_roll_saturation_uses_both_motors_additively():
    # huge roll error → both motors saturate +EFFORT (additive) → tau_roll ≈ 2*lever*42, pitch~0
    tau = ankle_joint_torque(np.zeros(2), np.zeros(2), np.array([0.0, 5.0]), np.zeros(2), J_LEFT)
    assert tau[1] == pytest.approx(2 * 1.0095 * EFFORT_MOTOR, rel=0.02)  # ≈ 84.8 N·m
    assert abs(tau[0]) < 1.0


def test_velocity_term_damps_through_the_linkage():
    # no position error, positive joint velocity → damping torque opposes motion
    tau = ankle_joint_torque(np.zeros(2), np.array([1.0, 0.0]), np.zeros(2), np.zeros(2), J_LEFT)
    assert tau[0] < 0  # pitch damping opposes +pitch velocity


def test_returns_float32_pair():
    tau = ankle_joint_torque(np.zeros(2), np.zeros(2), np.array([0.1, 0.1]), np.zeros(2), J_RIGHT)
    assert tau.shape == (2,)
    assert tau.dtype == np.float32


def test_apply_to_torque_vector_overwrites_only_ankles():
    # 8-joint toy vector: legs at indices 0-3,6-7; ankle pitch/roll at (4,5) L and (10,11) R would
    # exceed len — use a compact layout: pitch_L=2, roll_L=3, pitch_R=6, roll_R=7
    N = 8
    tau = np.arange(N, dtype=np.float32) * 10.0  # 0,10,20,...,70 — sentinel body torques
    q = np.zeros(N); dq = np.zeros(N)
    q_des = np.zeros(N); q_des[2] = 0.01  # small pitch_L error only
    ankles = [(2, 3, J_LEFT), (6, 7, J_RIGHT)]
    out = apply_to_torque_vector(tau.copy(), q, dq, q_des, np.zeros(N), ankles)
    # non-ankle joints untouched
    for i in (0, 1, 4, 5):
        assert out[i] == tau[i]
    # left ankle pitch got the parallel torque (≈ kp*2.297*0.01 ≈ 2.15), roll ~0
    assert out[2] == pytest.approx(KP_MOTOR * (J_LEFT.T @ J_LEFT)[0, 0] * 0.01, rel=0.02)
    assert abs(out[3]) < 0.05
    # right ankle at target → zero torque (overwrites its sentinel 60/70)
    assert out[6] == pytest.approx(0.0, abs=1e-4)
    assert out[7] == pytest.approx(0.0, abs=1e-4)
