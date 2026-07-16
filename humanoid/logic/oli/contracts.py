"""
contracts.py — the deployment-invariant brain↔world contracts (canonical PR space).

Three dataclasses cross the two seams of the Oli runtime:

    Observation   World →[Comm]→ Reason     ≅ RobotState + ImuData   (policy-agnostic)
    PolicyIn      Reason → Action           = Observation + Intent
    PolicyOut     Action →[Comm]→ World      ≅ RobotCmd               (policy-agnostic)

`Observation` and `PolicyOut` are the ONLY things that cross to the World, and are
identical for every policy — a manipulation skill still emits a full 31-joint
`PolicyOut` (holding the legs). `Intent` selects which policy is active (`mode`)
and carries that policy's expected input (for WALK: a body-frame velocity command).
Adding a skill later = a new `Mode` member + the runner branch that consumes it.
The richer "policy bank" abstraction is deferred by design (kept simple on purpose).

All joint arrays are PR-ordered (`PR_ORDER`), unscaled, float32. This module is
PURE: it imports only numpy + the stdlib — never `isaacsim`, never `limxsdk`. That
import-cleanliness is the enforceable invariant that lets one brain drive sim and
real; the World-native joint order lives only in Communication, never here.

References: design.md D3 (three contracts), D5 (single Reason→Policy seam),
D6 (PolicyOut is a resolved RobotCmd), D8 (stamp = sim time).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence, Union

import numpy as np

FloatArray = Union[Sequence[float], np.ndarray]

# ── Canonical joint order (HU_D04_01, MAY-145 §11) — the contract's ordering ──
# Single source of truth for the wire ordering, shared by the brain and (via
# import) by SimComm, which builds the PR↔Isaac permutation against it at runtime.
PR_ORDER: list[str] = [
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
NUM_JOINTS: int = 31
assert len(PR_ORDER) == NUM_JOINTS, "PR_ORDER must list exactly 31 joints"


class Mode(IntEnum):
    """Active-policy selector. An int so it serializes over the wire untouched.

    Extend with new members as policies are added (e.g. ``TABLE_CLEAN = 2``); the
    runner gains a matching branch. This is the seam where 'walk is one of many'
    lives without yet building a full policy-bank abstraction.
    """

    STAND = 0
    WALK = 1


def _vec(x: FloatArray, n: int, name: str) -> np.ndarray:
    """Coerce to a (n,) float32 array and validate length (seam safety check)."""
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.shape != (n,):
        raise ValueError(f"{name} must be length {n}, got shape {a.shape}")
    return a


@dataclass(frozen=True)
class Observation:
    """Raw body snapshot, World→brain. PR order, unscaled. Policy-agnostic.

    `stamp_ns` is SIM-time (not wall-clock): it is the clock the brain paces the
    policy by (D8). The World/SimComm owns stamping it with advancing sim time.
    """

    stamp_ns: int          # SIM-time nanoseconds — the D8 pacing clock
    q: np.ndarray          # (31,) joint positions     [rad]
    dq: np.ndarray         # (31,) joint velocities     [rad/s]
    tau: np.ndarray        # (31,) measured torques     [N·m]
    acc: np.ndarray        # (3,)  base linear accel     [m/s²]
    gyro: np.ndarray       # (3,)  base angular velocity [rad/s]
    quat_wxyz: np.ndarray  # (4,)  base orientation, (w, x, y, z)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "q", _vec(self.q, NUM_JOINTS, "q"))
        object.__setattr__(self, "dq", _vec(self.dq, NUM_JOINTS, "dq"))
        object.__setattr__(self, "tau", _vec(self.tau, NUM_JOINTS, "tau"))
        object.__setattr__(self, "acc", _vec(self.acc, 3, "acc"))
        object.__setattr__(self, "gyro", _vec(self.gyro, 3, "gyro"))
        object.__setattr__(self, "quat_wxyz", _vec(self.quat_wxyz, 4, "quat_wxyz"))


@dataclass(frozen=True)
class Intent:
    """Operator/reasoning intent: which policy is active + its expected input.

    `mode` selects the policy. The velocity fields are the WALK policy's expected
    input (body-frame command); STAND ignores them. Future policies read whichever
    of these (or future) fields they declare as their input.
    """

    mode: Mode
    v_x: float = 0.0  # forward  [m/s]
    v_y: float = 0.0  # lateral  [m/s]
    w_z: float = 0.0  # yaw rate [rad/s]


@dataclass(frozen=True)
class PolicyIn:
    """Reason → Action. The single seam: an Observation plus an Intent (D5)."""

    observation: Observation
    intent: Intent


@dataclass(frozen=True)
class PolicyOut:
    """Action →[Comm]→ World. A fully resolved RobotCmd (D6). PR order. Policy-agnostic.

    The brain does ALL policy-specific resolution (action·scale + default, gains,
    torque clamp) before this crosses the seam, so the World/Comm apply it without
    knowing any policy scale, default angle, or gain. `mode` here is the per-joint
    *motor* mode of `RobotCmd` (distinct from `Intent.mode`); 0 throughout in sim.
    """

    stamp_ns: int          # the Observation stamp this command was computed from
    q_des: np.ndarray      # (31,) desired joint positions [rad]
    dq_des: np.ndarray     # (31,) desired joint velocities [rad/s]
    tau_ff: np.ndarray     # (31,) feedforward torque       [N·m]
    kp: np.ndarray         # (31,) position gains
    kd: np.ndarray         # (31,) velocity gains
    mode: np.ndarray       # (31,) per-joint motor mode (RobotCmd.mode), int

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "q_des", _vec(self.q_des, NUM_JOINTS, "q_des"))
        object.__setattr__(self, "dq_des", _vec(self.dq_des, NUM_JOINTS, "dq_des"))
        object.__setattr__(self, "tau_ff", _vec(self.tau_ff, NUM_JOINTS, "tau_ff"))
        object.__setattr__(self, "kp", _vec(self.kp, NUM_JOINTS, "kp"))
        object.__setattr__(self, "kd", _vec(self.kd, NUM_JOINTS, "kd"))
        object.__setattr__(
            self, "mode",
            np.asarray(self.mode, dtype=np.int32).reshape(-1),
        )


# ── Vision contract (oli-perception, MAY-149) ─────────────────────────────────
# A 4th invariant that crosses World→brain, on its OWN transport (not the 1 kHz
# control channel): images are ~10⁴× bigger and arrive at ~30 Hz. See design.md
# (oli-perception) D5/D6.


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics (pixels); principal point typically at the image center.

    A contract type (lives here, not in the mount table): `camera_mounts.rgb_intrinsics`
    produces these, `CameraFrame` carries them, `codec` serializes them.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraFrame:
    """One camera's RGBD snapshot, World→brain. Intrinsics only — NO extrinsics.

    The camera's world pose is derivable brain-side by forward kinematics from an
    `Observation` (joint states) plus the static mount table (`camera_mounts`), a
    derivation identical in sim and real — so shipping the pose would both break the
    invariance (the real robot has no ground-truth pose) and bloat the payload (D5).
    RGB is uint8 (H, W, 3); depth is float32 (H, W) planar distance-to-image-plane in
    meters; both must match the declared intrinsics resolution. Depth is OPTIONAL
    (None): RGB-only streams — the head stereo pair (MAY-173) — travel the same
    channel with an empty depth payload on the wire.
    """

    stamp_ns: int              # SIM-time nanoseconds — same clock as Observation (D8)
    name: str                  # camera name, e.g. "chest" / "head" / "head_left"
    rgb: np.ndarray            # (H, W, 3) uint8
    depth: Optional[np.ndarray]  # (H, W) float32, meters (planar Z) — None = RGB-only
    intrinsics: CameraIntrinsics

    def __post_init__(self) -> None:
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))
        object.__setattr__(self, "name", str(self.name))
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
        depth = None
        if self.depth is not None:
            depth = np.asarray(self.depth, dtype=np.float32)
            if depth.ndim != 2:
                raise ValueError(f"depth must be (H, W), got {depth.shape}")
            if rgb.shape[:2] != depth.shape:
                raise ValueError(
                    f"rgb {rgb.shape[:2]} and depth {depth.shape} resolutions disagree")
        i = self.intrinsics
        if (i.height, i.width) != rgb.shape[:2]:
            raise ValueError(
                f"intrinsics ({i.width}x{i.height}) != frame ({rgb.shape[1]}x{rgb.shape[0]})"
            )
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth", depth)
