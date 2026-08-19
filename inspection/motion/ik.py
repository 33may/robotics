"""Analytic IK for the UR5e — all branches, flange frame (MAY-184).

EAIK subproblem decomposition on NOMINAL DH. The closed form does not survive
the arm's factory calibration (calibrated alpha2=0.00139 breaks the
three-parallel-axes condition) — so: enumerate branches on nominal DH now,
Newton-refine on the calibrated model later (hook, once the calibration is
extracted from the arm; error until then is 1-2 mm, inside the 20 mm padding).

Frame adapters (verified vs pinocchio ur5e_description, residual < 1e-9 over
200 random configs — see MAY-184):
    T_pinocchio_flange = A_BASE @ T_dh @ B_TOOL
    A_BASE: URDF base_link = Rz(pi) @ DH base   (the classic UR convention)
    B_TOOL: DH frame-6 axes permuted onto the flange frame (+X out of tool face)

GOTCHA: EAIK marks approximate solutions with is_LS — ALWAYS filter them.
Unfiltered they carry centimeter-level FK error and look like real branches.
"""

import numpy as np
from eaik.IK_DH import DhRobot

# official UR5e nominal DH
_ALPHA = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
_A = np.array([0, -0.425, -0.3922, 0, 0, 0])
_D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])

A_BASE = np.diag([-1.0, -1.0, 1.0, 1.0])
B_TOOL = np.array([[0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
_A_INV = np.linalg.inv(A_BASE)
_B_INV = np.linalg.inv(B_TOOL)


class UR5eIK:
    """branches(T) -> every exact analytic solution for a flange pose."""

    def __init__(self):
        self._bot = DhRobot(_ALPHA, _A, _D)

    def fk(self, q):
        """Flange pose (4x4, base_link frame) for joint config q."""
        return A_BASE @ self._bot.fwdKin(np.asarray(q, dtype=float)) @ B_TOOL

    def branches(self, T_flange):
        """All exact IK branches for a flange pose in the base_link frame.

        Returns (N, 6) array, N in 0..8, angles wrapped to [-pi, pi].
        Least-squares pseudo-solutions are filtered out (is_LS gotcha).
        """
        sol = self._bot.IK(_A_INV @ np.asarray(T_flange, dtype=float) @ _B_INV)
        Q, ls = np.array(sol.Q), np.array(sol.is_LS)
        if Q.size == 0:
            return np.empty((0, 6))
        Q = Q[~ls]
        return (Q + np.pi) % (2 * np.pi) - np.pi
