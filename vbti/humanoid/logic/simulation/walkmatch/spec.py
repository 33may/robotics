"""walkmatch/spec.py — shared sim-to-sim actuator-ID step-response spec.

Pure (numpy only) so it imports in BOTH the `isaac` (py3.11) and `limx` (py3.8) envs.
Defines the walk policy's PD gains + default pose (PR order, from
`controllers/HU_D04_01/walk_controller/walk_param.yaml`) and the open-loop step-reference
protocol both sims replay, so their per-joint step responses are directly comparable.

NO isaacsim / mujoco / limxsdk imports here — only numpy. This is the contract that makes
the Isaac and MuJoCo actuator-ID harnesses measure the *same* experiment.
"""
from __future__ import annotations

import numpy as np

# ── PR joint order (must match logic/oli/contracts.PR_ORDER) ─────────────────────
PR_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]
NUM_JOINTS = 31
assert len(PR_ORDER) == NUM_JOINTS

# ── Policy PD gains + default pose (walk_param.yaml, PR order) ────────────────────
KP = np.array([
    139.41, 139.41, 139.41, 139.41, 93.65, 93.65,        # left leg
    139.41, 139.41, 139.41, 139.41, 93.65, 93.65,        # right leg
    93.65, 93.65, 93.65,                                  # waist
    15.12, 15.12,                                         # head
    87.51, 87.51, 87.51, 87.51, 15.12, 15.12, 15.12,     # left arm
    87.51, 87.51, 87.51, 87.51, 15.12, 15.12, 15.12,     # right arm
], dtype=np.float32)
KD = np.array([
    17.75, 17.75, 17.75, 17.75, 11.92, 11.92,
    17.75, 17.75, 17.75, 17.75, 11.92, 11.92,
    11.92, 11.92, 11.92,
    1.93, 1.93,
    11.14, 11.14, 11.14, 11.14, 1.93, 1.93, 1.93,
    11.14, 11.14, 11.14, 11.14, 1.93, 1.93, 1.93,
], dtype=np.float32)
DEFAULT = np.array([
    -0.15, -0.00, -0.05, 0.30, -0.16, 0.0,
    -0.15,  0.00,  0.05, 0.30, -0.16, 0.0,
     0.0,   0.0,   0.0,
     0.0,   0.0,
     0.1,   0.1,  -0.2, -0.2, 0.0, 0.0, 0.0,
     0.1,  -0.1,   0.2, -0.2, 0.0, 0.0, 0.0,
], dtype=np.float32)

# ── Per-joint rotor inertia (armature), PR order, HU_D04_01 MJCF ─────────────────
# NOTE: MJCF serial ankle_pitch/roll have NO armature attribute → 0 (the 0.1845504 is on
# the achilles A/B joints, which the serial USD ankle does not have). This corrects the
# earlier ARMATURE_PR that mis-assigned 0.1845504 to the serial ankle.
_ARM = {"hip": 0.14125, "knee": 0.14125, "ankle": 0.0, "waist": 0.1845504,
        "shoulder": 0.0886706, "elbow": 0.0886706, "head": 0.0153218, "wrist": 0.0153218}


def _armature_for(joint: str) -> float:
    for key, val in _ARM.items():
        if key in joint:
            return val
    raise KeyError(joint)


ARMATURE = np.array([_armature_for(n) for n in PR_ORDER], dtype=np.float32)

# ── MuJoCo actuator name per LEG joint (gear-1 direct torque; HU_D04_01.xml) ─────
# Only the leg joints are direct serial motors in BOTH sims. The ankle is the parallel
# achilles A/B in MuJoCo (deploy-only) → excluded from the apples-to-apples comparison.
MJ_MOTOR = {
    "left_hip_pitch_joint": "hip_pitch_left", "left_hip_roll_joint": "hip_roll_left",
    "left_hip_yaw_joint": "hip_yaw_left", "left_knee_joint": "knee_left",
    "right_hip_pitch_joint": "hip_pitch_right", "right_hip_roll_joint": "hip_roll_right",
    "right_hip_yaw_joint": "hip_yaw_right", "right_knee_joint": "knee_right",
}
LEG_JOINTS = list(MJ_MOTOR.keys())

# Drive ONLY the big-inertia leg joints in the actuator-ID. The bare serial ankle (armature
# 0, tiny foot inertia) is numerically unstable under the deploy's explicit ankle gains
# (kd·dt/I > 2) — driving it corrupts the whole chain. With gravity off, undriven joints
# stay at default, so a leg-only gain mask gives a clean, structurally-matched comparison.
LEG_MASK = np.array([n in MJ_MOTOR for n in PR_ORDER], dtype=bool)
KP_LEG = KP * LEG_MASK
KD_LEG = KD * LEG_MASK

# ── Step-reference protocol (both sims replay this open-loop) ─────────────────────
DT = 1.0 / 1000.0       # 1 kHz, matches MJCF timestep + Isaac physics_dt
T_TOTAL = 1.0           # seconds per step test
T_STEP = 0.1            # apply the step at 100 ms (after a brief hold at default)
N_STEPS = int(round(T_TOTAL / DT))
N_HOLD = int(round(T_STEP / DT))


def q_des_at(tick: int, target_idx: int, amp: float) -> np.ndarray:
    """PR-order position target at a given 1 kHz tick: hold DEFAULT, then step the target
    joint by `amp` rad once tick >= N_HOLD. dq target is 0 throughout."""
    q = DEFAULT.copy()
    if tick >= N_HOLD:
        q[target_idx] += amp
    return q


def step_metrics(t: np.ndarray, q: np.ndarray, q0: float, q_final_des: float) -> dict:
    """Characterize a step response: onset lag (time to reach 10% of the commanded step),
    rise time (10%→90%), overshoot %, and steady-state error at the end. `t` from step
    onset. All measured relative to the commanded step (q0 → q_final_des)."""
    step = q_final_des - q0
    if abs(step) < 1e-6:
        return {}
    frac = (q - q0) / step  # 0 at start, 1 at target
    def _t_at(level: float):
        idx = np.argmax(frac >= level)
        return float(t[idx]) if np.any(frac >= level) else float("nan")
    t10, t90 = _t_at(0.10), _t_at(0.90)
    peak = float(np.max(frac)) if step > 0 else float(np.min(-frac) * -1)
    return {
        "onset_lag_ms": t10 * 1e3,
        "rise_time_ms": (t90 - t10) * 1e3,
        "overshoot_pct": max(0.0, (peak - 1.0)) * 100.0,
        "ss_err_rad": float(q[-1] - q_final_des),
    }
