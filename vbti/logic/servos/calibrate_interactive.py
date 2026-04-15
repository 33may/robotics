"""Bus helpers, data model, fitting logic, and save for interactive servo calibration.

No TUI / curses code here — that lives in the caller (Task 3).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from vbti.logic.servos.profiles import (
    JOINT_NAMES,
    LEROBOT_CALIB_DIR,
    _calib_path,
    _load_registry,
    _save_registry,
    register,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODER_RESOLUTION = 4096
ENCODER_MAX = ENCODER_RESOLUTION - 1  # 4095
DEGREES_PER_TICK = 360.0 / ENCODER_MAX

# Motor ID mapping for SO-ARM101
MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    raw_encoder: int
    target_degrees: float


@dataclass
class JointCalibState:
    name: str
    motor_id: int
    range_min: int | None = None
    range_max: int | None = None
    keyframes: list[Keyframe] = field(default_factory=list)
    homing_offset: int | None = None
    accepted: bool = False


# ---------------------------------------------------------------------------
# Bus helpers
# ---------------------------------------------------------------------------

def _make_bus(port: str) -> FeetechMotorsBus:
    """Create a FeetechMotorsBus for all 6 SO-ARM101 joints (no calibration)."""
    motors = {}
    for joint in JOINT_NAMES:
        norm = MotorNormMode.RANGE_0_100 if joint == "gripper" else MotorNormMode.DEGREES
        motors[joint] = Motor(id=MOTOR_IDS[joint], model="sts3215", norm_mode=norm)
    return FeetechMotorsBus(port=port, motors=motors)


def read_raw_position(bus: FeetechMotorsBus, joint: str) -> int:
    """Read raw encoder value for a single joint."""
    result = bus.sync_read("Present_Position", [joint], normalize=False)
    return int(result[joint])


def read_all_raw(bus: FeetechMotorsBus) -> dict[str, int]:
    """Read raw encoder values for all joints."""
    result = bus.sync_read("Present_Position", normalize=False)
    return {name: int(result[name]) for name in JOINT_NAMES}


# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def fit_homing_offset(state: JointCalibState) -> int:
    """Compute homing_offset from keyframes using least-squares averaging.

    LeRobot DEGREES normalization:
        degrees = (raw - homing_offset - midpoint) * 360.0 / 4095
    Solving for homing_offset per keyframe:
        homing_offset = raw - midpoint - (target_degrees / DEGREES_PER_TICK)
    With multiple keyframes we average (least-squares optimal for a single parameter).
    """
    if state.range_min is None or state.range_max is None:
        raise ValueError(f"range_min/range_max not set for joint '{state.name}'")
    if not state.keyframes:
        raise ValueError(f"No keyframes for joint '{state.name}'")

    midpoint = (state.range_min + state.range_max) / 2.0
    offsets = []
    for kf in state.keyframes:
        offset = kf.raw_encoder - midpoint - (kf.target_degrees / DEGREES_PER_TICK)
        offsets.append(offset)
    return int(round(np.mean(offsets)))


def compute_residuals(state: JointCalibState) -> list[float]:
    """Compute degree error per keyframe given current calibration values."""
    if state.homing_offset is None or state.range_min is None or state.range_max is None:
        raise ValueError("Calibration incomplete — need homing_offset, range_min, range_max")

    midpoint = (state.range_min + state.range_max) / 2.0
    residuals = []
    for kf in state.keyframes:
        predicted = (kf.raw_encoder - state.homing_offset - midpoint) * DEGREES_PER_TICK
        residuals.append(predicted - kf.target_degrees)
    return residuals


def raw_to_degrees(raw: int, state: JointCalibState) -> float | None:
    """Convert raw encoder to degrees using current calibration state.

    Returns None if calibration is incomplete.
    """
    if state.homing_offset is None or state.range_min is None or state.range_max is None:
        return None
    midpoint = (state.range_min + state.range_max) / 2.0
    return (raw - state.homing_offset - midpoint) * DEGREES_PER_TICK


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_profile(
    name: str,
    states: dict[str, JointCalibState],
    description: str = "",
) -> Path:
    """Build calibration JSON from states and save to LeRobot cache + registry."""
    calib_data: dict[str, dict] = {}
    for joint, st in states.items():
        if st.homing_offset is None or st.range_min is None or st.range_max is None:
            raise ValueError(f"Joint '{joint}' calibration incomplete")
        calib_data[joint] = {
            "id": st.motor_id,
            "drive_mode": 0,
            "homing_offset": st.homing_offset,
            "range_min": st.range_min,
            "range_max": st.range_max,
        }

    out_path = _calib_path(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calib_data, indent=4) + "\n")

    register(name, description)
    return out_path
