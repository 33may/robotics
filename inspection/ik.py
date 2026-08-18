#!/usr/bin/env python3
"""Look-at IK for the SO-ARM101 wrist camera.

Task: place the CAMERA optical center at a target point with its optical
axis (+z) pointing along a desired direction. That is 5 constraints
(3 position + 2 pointing; roll about the optical axis is deliberately
free) for the arm's 5 usable joints — square, no redundancy.

Damped least squares on the camera-frame error twist, joint limits
clamped every step, multiple restarts. Angles in/out are the driver's
EEPROM-centered degrees (identity to URDF frame, see fk_so101).
"""

import numpy as np

from vbti.logic.servos.fk_so101 import _load, JOINT_SIGN, JOINT_OFFSET_DEG
from vbti.logic.servos.so101 import JOINT_NAMES
from inspection.cell.geometry import load_T_flange_cam

ARM_JOINTS = JOINT_NAMES[:5]          # gripper does not move the camera

POS_TOL_M = 0.005
ANG_TOL_RAD = np.radians(2.0)
MAX_ITERS = 200
DAMPING = 1e-4


def _q_from_deg(joints_deg: dict) -> np.ndarray:
    model, _, _ = _load()
    q = np.zeros(model.nq)
    for i in range(1, model.njoints):
        n = model.names[i]
        d = joints_deg.get(n, 0.0)
        q[model.joints[i].idx_q] = JOINT_SIGN[n] * (d - JOINT_OFFSET_DEG[n]) * np.pi / 180
    return q


def _deg_from_q(q: np.ndarray) -> dict:
    model, _, _ = _load()
    out = {}
    for i in range(1, model.njoints):
        n = model.names[i]
        out[n] = float(np.degrees(q[model.joints[i].idx_q]) / JOINT_SIGN[n]
                       + JOINT_OFFSET_DEG[n])
    return out


def solve_lookat(cam_pos: np.ndarray, look_dir: np.ndarray,
                 q0_deg: dict | None = None, restarts: int = 4,
                 seed: int = 0) -> dict | None:
    """Find joint degrees placing the camera at cam_pos looking along look_dir.

    Returns joint dict (5 arm joints) or None if no restart converges
    within joint limits.
    """
    import pinocchio as pin
    model, data, flange_id = _load()
    T_fc = pin.SE3(load_T_flange_cam())
    look_dir = np.asarray(look_dir, dtype=float)
    look_dir = look_dir / np.linalg.norm(look_dir)
    rng = np.random.default_rng(seed)

    lo, hi = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()

    starts = []
    if q0_deg is not None:
        starts.append(_q_from_deg(q0_deg))
    while len(starts) < restarts + (q0_deg is not None):
        starts.append(lo + (hi - lo) * rng.uniform(0.25, 0.75, size=model.nq))

    for q in starts:
        q = q.copy()
        for _ in range(MAX_ITERS):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, flange_id)
            T_cam = data.oMf[flange_id] * T_fc          # base <- cam

            # position error, expressed in camera frame
            e_pos_w = cam_pos - T_cam.translation
            e_pos = T_cam.rotation.T @ e_pos_w

            # pointing error: rotate camera +z onto look_dir (camera frame)
            z_des = T_cam.rotation.T @ look_dir          # desired axis, cam frame
            z_cur = np.array([0.0, 0.0, 1.0])
            w = np.cross(z_cur, z_des)                   # sin * axis
            e_ang = w[:2]                                # roll (z) stays free

            if np.linalg.norm(e_pos) < POS_TOL_M and \
               np.arcsin(min(1.0, np.linalg.norm(w))) < ANG_TOL_RAD and \
               z_des[2] > 0:
                sol = _deg_from_q(q)
                return {k: sol[k] for k in ARM_JOINTS}

            # camera-frame jacobian: flange local jacobian shifted by T_fc
            J_fl = pin.computeFrameJacobian(model, data, q, flange_id,
                                            pin.ReferenceFrame.LOCAL)
            J_cam = T_fc.toActionMatrixInverse() @ J_fl   # twist in cam frame
            J = np.vstack([J_cam[:3], J_cam[3:5]])        # 5 x nq
            J = J[:, :5]                                  # gripper column out

            e = np.concatenate([e_pos, e_ang])
            JJt = J @ J.T + DAMPING * np.eye(5)
            dq = J.T @ np.linalg.solve(JJt, e)
            step = np.zeros(model.nq)
            step[:5] = dq
            q = np.clip(q + step, lo, hi)
    return None
