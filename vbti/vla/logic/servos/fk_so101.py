#!/usr/bin/env python3
"""Forward kinematics for the SO-ARM101 via Pinocchio.

Maps the driver's EEPROM-centered degrees (see ``so101.py``) onto the
``so101_new_calib.urdf`` joint frame. Both conventions are zero-at-center,
so the mapping is per-joint sign/offset only — tuned by physical check
(``jog_check``), not trusted from documentation.

Usage:
    python -m vbti.logic.servos.fk_so101 pose                 # live flange pose
    python -m vbti.logic.servos.fk_so101 jog_check --joint=shoulder_pan
"""

from pathlib import Path

import numpy as np

from vbti.logic.servos.so101 import JOINT_NAMES, connect

URDF_PATH = (Path(__file__).resolve().parents[3] / "third_party" / "SO-ARM100"
             / "Simulation" / "SO101" / "so101_new_calib.urdf")
FLANGE_FRAME = "gripper_frame_link"

# EEPROM-degrees -> URDF-radians mapping: q = sign * (deg - offset_deg) * pi/180.
# Both frames are zero-at-center; signs/offsets verified physically via
# jog_check, NOT assumed. Adjust here if a jog moves opposite to prediction.
JOINT_SIGN = {n: 1.0 for n in JOINT_NAMES}
JOINT_OFFSET_DEG = {n: 0.0 for n in JOINT_NAMES}

_model = _data = _frame_id = None


def _load():
    global _model, _data, _frame_id
    if _model is None:
        import pinocchio as pin
        _model = pin.buildModelFromUrdf(str(URDF_PATH))
        _data = _model.createData()
        _frame_id = _model.getFrameId(FLANGE_FRAME)
    return _model, _data, _frame_id


def deg_to_q(joints_deg: dict) -> np.ndarray:
    """Driver degrees -> Pinocchio configuration vector (URDF joint order).

    Missing joints default to 0 deg — IK solutions carry only the 5 arm
    joints (the gripper joint moves the jaw, not the camera).
    """
    model, _, _ = _load()
    q = np.zeros(model.nq)
    for i in range(1, model.njoints):
        name = model.names[i]
        q[model.joints[i].idx_q] = (
            JOINT_SIGN[name] * (joints_deg.get(name, 0.0) - JOINT_OFFSET_DEG[name])
            * np.pi / 180.0)
    return q


def fk(joints_deg: dict) -> np.ndarray:
    """Joint degrees -> 4x4 T_base_flange (gripper_frame_link in base frame)."""
    import pinocchio as pin
    model, data, frame_id = _load()
    q = deg_to_q(joints_deg)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, frame_id)
    return data.oMf[frame_id].homogeneous.copy()


def _print_pose(T: np.ndarray) -> None:
    import pinocchio as pin
    xyz = T[:3, 3]
    rpy = pin.rpy.matrixToRpy(T[:3, :3]) * 180 / np.pi
    print(f"flange xyz [m]:  x={xyz[0]:+.4f}  y={xyz[1]:+.4f}  z={xyz[2]:+.4f}")
    print(f"flange rpy [deg]: r={rpy[0]:+.1f}  p={rpy[1]:+.1f}  y={rpy[2]:+.1f}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def pose(port: str = "/dev/ttyACM0") -> None:
    """Read the arm and print the FK flange pose."""
    with connect(port) as arm:
        joints = arm.read_degrees()
    print({k: round(v, 1) for k, v in joints.items()})
    _print_pose(fk(joints))


def jog_check(joint: str, delta: float = 15.0, port: str = "/dev/ttyACM0") -> None:
    """Predict, then physically jog one joint and compare flange motion.

    Prints the predicted flange displacement, moves the joint +delta and back,
    and reports the FK displacement from real encoder readings. Watch the arm:
    if it moves opposite the prediction, flip JOINT_SIGN[joint].
    """
    with connect(port) as arm:
        start = arm.read_degrees()
        T0 = fk(start)

        target = dict(start)
        target[joint] = start[joint] + delta
        T1_pred = fk(target)
        dxyz = (T1_pred[:3, 3] - T0[:3, 3]) * 1000
        print(f"predicted flange move for {joint} +{delta} deg: "
              f"dx={dxyz[0]:+.1f} dy={dxyz[1]:+.1f} dz={dxyz[2]:+.1f} mm")

        arm.move_to({joint: start[joint] + delta})
        actual = arm.read_degrees()
        T1 = fk(actual)
        dxyz_a = (T1[:3, 3] - T0[:3, 3]) * 1000
        print(f"FK from real encoders:                    "
              f"dx={dxyz_a[0]:+.1f} dy={dxyz_a[1]:+.1f} dz={dxyz_a[2]:+.1f} mm")

        arm.move_to({joint: start[joint]})
        print("returned. Did the arm physically move in the predicted direction?")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"pose": pose, "jog_check": jog_check})
