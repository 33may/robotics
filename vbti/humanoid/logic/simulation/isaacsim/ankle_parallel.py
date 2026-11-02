"""ankle_parallel.py — faithful dual-motor achilles ankle emulation on the serial USD.

The HU_D04_01 walk policy trains/deploys against the PARALLEL achilles ankle: two motors
A/B per foot, PD in MOTOR space (kp=93.65, kd=11.92, each clipped to ±42 N·m), coupled to
the serial (pitch,roll) joint by the linkage Jacobian J = ∂(A,B)/∂(pitch,roll). Our shipped
Isaac USD has a single SERIAL pitch/roll ankle (PhysX can't hold the real closed loop —
importing it diverges; see NOTEBOOK). This module emulates the real per-motor mechanism in
SOFTWARE — the "free A/B + kinematic constraint" path — so no physical rods are needed:

    A,B are slaved to (pitch,roll) via J (software constraint; no rod bodies → no PhysX blowup)
    e_m   = J (q_des − q)                          # per-motor position error
    tau_m = clip(kp·e_m + kd·J(dq̇_des − dq̇), ±eff)  # per-motor PD, clipped BEFORE mapping
    tau_q = Jᵀ tau_m                                # equivalent serial-joint torque

The per-motor clip (before Jᵀ) is the faithful shared-authority nonlinearity a scalar
--ankle-kp-scale cannot express: pure pitch drives the motors DIFFERENTIALLY (±42 → ~90 N·m
pitch, 0 roll), pure roll drives them ADDITIVELY (both +42 → ~85 N·m roll, 0 pitch), and a
mixed pitch+roll demand makes the two motors trade off exactly as the real linkage does.

The reflected rotor inertia the two motors add at the joint, diag(Jᵀ diag(I,I) J), is set as
the serial ankle's armature so an explicit ankle PD is numerically stable (kd·dt/I ≪ 2).

Pure (numpy only) so it unit-tests in the `brain` env without isaacsim. J values measured in
MuJoCo (the real closed loop) by `logic/simulation/walkmatch/ankle_jacobian.py`; coupling ≈0
and J ~constant across the ankle range, so the constant linearization is faithful.
"""
from __future__ import annotations

import numpy as np

# Measured achilles Jacobian J = ∂(A,B)/∂(pitch,roll) (rows A,B; cols pitch,roll).
# Left: pitch is the differential mode (A,B opposite signs), roll the additive mode (same sign).
# Right mirrors the left (the "twist" only flips the pitch/differential sign — coupling still ≈0).
J_LEFT = np.array([[1.0718, 1.0095],
                   [-1.0717, 1.0095]], dtype=np.float64)
J_RIGHT = np.array([[-1.0718, 1.0095],
                    [1.0717, 1.0095]], dtype=np.float64)

KP_MOTOR = 93.65          # per-motor deploy stiffness (walk_param.yaml ankle)
KD_MOTOR = 11.92          # per-motor deploy damping
EFFORT_MOTOR = 42.0       # per-motor torque limit (training URDF HU_D04_01_rl ankle effort)
I_MOTOR = 0.1845504       # per-motor rotor inertia (MJCF achilles A/B armature)


def reflected_inertia(J: np.ndarray) -> np.ndarray:
    """Diagonal reflected rotor inertia the two motors add at the serial (pitch,roll) joint:
    diag(Jᵀ diag(I_MOTOR, I_MOTOR) J). Returns (2,) [I_pitch, I_roll] ≈ [0.424, 0.376]."""
    M = I_MOTOR * (np.asarray(J, dtype=np.float64).T @ np.asarray(J, dtype=np.float64))
    return np.array([M[0, 0], M[1, 1]], dtype=np.float64)


def ankle_joint_torque(
    q, dq, q_des, dq_des, J,
    kp: float = KP_MOTOR, kd: float = KD_MOTOR, effort: float = EFFORT_MOTOR,
) -> np.ndarray:
    """Faithful dual-motor achilles PD for one serial ankle (pitch, roll).

    q, dq, q_des, dq_des: (2,) arrays [pitch, roll] (rad, rad/s). J: 2×2 linkage Jacobian
    (J_LEFT or J_RIGHT). Returns (2,) float32 [tau_pitch, tau_roll] — the serial-joint torque
    the two per-motor-clipped achilles motors would produce. Pure / array-broadcast so it is
    unit-testable without isaacsim; the World calls it every substep with the LATEST q/dq.
    """
    J = np.asarray(J, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64); dq = np.asarray(dq, dtype=np.float64)
    q_des = np.asarray(q_des, dtype=np.float64); dq_des = np.asarray(dq_des, dtype=np.float64)
    e_m = J @ (q_des - q)                      # per-motor position error
    edot_m = J @ (dq_des - dq)                 # per-motor velocity error
    tau_m = kp * e_m + kd * edot_m             # per-motor PD
    np.clip(tau_m, -effort, effort, out=tau_m)  # clip EACH motor before mapping (faithful)
    return (J.T @ tau_m).astype(np.float32)     # equivalent serial-joint torque


def apply_to_torque_vector(tau, q, dq, q_des, dq_des, ankles) -> np.ndarray:
    """Overwrite the ankle (pitch,roll) entries of a full joint-torque vector `tau` with the
    faithful parallel-ankle torque, leaving every other joint untouched. `ankles` is a list of
    (pitch_idx, roll_idx, J) in the SAME DOF order as `tau/q/dq`. Mutates and returns `tau`.

    This is exactly what the World's explicit substep does: compute joint-space PD for the whole
    body, then replace the ankle joints with the motor-space parallel law. Pure so it unit-tests
    the wiring without isaacsim.
    """
    tau = np.asarray(tau, dtype=np.float32)
    q = np.asarray(q, dtype=np.float64); dq = np.asarray(dq, dtype=np.float64)
    q_des = np.asarray(q_des, dtype=np.float64); dq_des = np.asarray(dq_des, dtype=np.float64)
    for pi, ri, J in ankles:
        tp = ankle_joint_torque(
            [q[pi], q[ri]], [dq[pi], dq[ri]],
            [q_des[pi], q_des[ri]], [dq_des[pi], dq_des[ri]], J)
        tau[pi] = tp[0]
        tau[ri] = tp[1]
    return tau
