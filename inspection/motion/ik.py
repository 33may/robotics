"""Analytic IK for the UR5e — all branches, flange frame (MAY-184).

EAIK subproblem decomposition on NOMINAL DH. The closed form does not survive
the arm's factory calibration (calibrated alpha2=0.00139 breaks the
three-parallel-axes condition) — so: enumerate branches on nominal DH now,
Newton-refine on the calibrated model later (hook, once the calibration is
extracted from the arm; error until then is 1-2 mm, inside the 20 mm padding).

Frame adapters (verified vs pinocchio ur5e_description, residual < 1e-9 over
200 random configs — see MAY-184):
    T_flange = T_dh @ B_TOOL
    B_TOOL: DH frame-6 axes permuted onto the flange frame (+X out of tool face)

WE WORK IN THE CONTROLLER'S `base` FRAME — the one the teach pendant shows —
which is what UR's nominal DH is already written in. There used to be an
`A_BASE = Rz(pi)` here converting DH into the URDF's REP-103 `base_link`
(ur5e_description declares `base` = `base_link` rotated pi about Z, and says so
in a comment on `base_link-base_fixed_joint`). That convention was consistent
across both kinematics engines and still wrong for us, because every number we
MEASURE — the table probe in `cell.yaml`, anything read off the pendant —
lives in `base`. Loading those as `base_link` put the whole cell 180 deg from
the arm; nothing caught it, because the planner, the viewers and the viewsphere
all shared the same rotated world. `cell/world.py` mounts the pinocchio model
with the matching rotation, so both engines still agree — see MAY-184 and
`calib/frame_check.py`, which measures the agreement against the robot itself.

GOTCHA: EAIK marks approximate solutions with is_LS — ALWAYS filter them.
Unfiltered they carry centimeter-level FK error and look like real branches.
"""

import numpy as np
from eaik.IK_DH import DhRobot

# official UR5e nominal DH
_ALPHA = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
_A = np.array([0, -0.425, -0.3922, 0, 0, 0])
_D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])

B_TOOL = np.array([[0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
_B_INV = np.linalg.inv(B_TOOL)


class UR5eIK:
    """branches(T) -> every exact analytic solution for a flange pose."""

    def __init__(self):
        self._bot = DhRobot(_ALPHA, _A, _D)

    def fk(self, q):
        """Flange pose (4x4) for joint config q, in the controller `base` frame."""
        return self._bot.fwdKin(np.asarray(q, dtype=float)) @ B_TOOL

    def branches(self, T_flange):
        """All exact IK branches for a flange pose in the controller `base` frame.

        Returns (N, 6) array, N in 0..8, angles wrapped to [-pi, pi].
        Least-squares pseudo-solutions are filtered out (is_LS gotcha).
        """
        sol = self._bot.IK(np.asarray(T_flange, dtype=float) @ _B_INV)
        Q, ls = np.array(sol.Q), np.array(sol.is_LS)
        if Q.size == 0:
            return np.empty((0, 6))
        Q = Q[~ls]
        return (Q + np.pi) % (2 * np.pi) - np.pi
