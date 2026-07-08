"""camera_mounts.py — the shared, World-agnostic camera mount table (design.md D10).

The single source of truth for where Oli's two RealSense **D435i** cameras sit and
what they see. It is consumed in TWO places that must never disagree:

  - the Isaac `build_camera_usd.py`, which bakes the camera prims into the robot USD;
  - the brain's FK-based extrinsics derivation, which computes each camera's world
    pose from an `Observation` (joint states) + these static mounts — identically in
    sim and (future) real. Keeping the mounts here, not re-typed per World, is what
    makes that pose match ground truth everywhere.

Brain-pure: numpy + stdlib only, never `isaacsim`/`limxsdk`.

Sources:
  - mounts: `oli-corpus://user-manual#1.4.1` (chest [0.092, 0.0175, 0.4336] @ 35° down;
    head [0.0615, 0.0175, 0.652] @ 0°), all in the `base_link` frame;
  - kinematic chain: HU_D04_01 URDF joint origins — every joint has rpy=0, so the
    base_link→link chain is pure translation (verified 2026-07-01);
  - D435i RGB horizontal FOV 69° (Intel spec: 69°×42° color).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .contracts import CameraIntrinsics  # the intrinsics type lives with the contracts

# ── Optics ───────────────────────────────────────────────────────────────────
D435I_RGB_HFOV_DEG: float = 69.0
DEFAULT_WIDTH: int = 1280
DEFAULT_HEIGHT: int = 720

# ── Kinematic chain (base_link → link), from the HU_D04_01 URDF ────────────────
# child_link -> (parent_link, translation in the parent frame). All joint rpy=0,
# so nominal link poses are the running sum of these translations.
# oli-corpus://oli-main-2.2.12#install/etc/HU_D04_description/urdf/HU_D04_01.urdf
_JOINT_ORIGINS: dict[str, Tuple[str, np.ndarray]] = {
    "waist_yaw_link": ("base_link", np.array([0.0, 0.0, 0.10239])),
    "waist_roll_link": ("waist_yaw_link", np.array([0.0, 0.0, 0.057])),
    "waist_pitch_link": ("waist_roll_link", np.array([0.0, 0.0, 0.0])),
    "head_yaw_link": ("waist_pitch_link", np.array([-0.013, 0.0, 0.3882])),
    "head_pitch_link": ("head_yaw_link", np.array([0.0, 0.0, 0.0395])),
}


def link_origin_base(link: str) -> np.ndarray:
    """Nominal (zero-joint) origin of `link` in the base_link frame.

    Sums the URDF joint translations from `link` up to base_link. Valid because the
    whole chain has rpy=0 (pure translation).
    """
    origin = np.zeros(3, dtype=float)
    cur = link
    while cur != "base_link":
        parent, offset = _JOINT_ORIGINS[cur]
        origin = origin + offset
        cur = parent
    return origin


def to_parent_local(pos_base, parent_link: str) -> np.ndarray:
    """Convert a base_link-frame position into `parent_link`'s local frame at nominal
    pose. The chain is pure translation, so this is a subtraction of the parent's
    nominal origin; the camera's pitch is unchanged (parent frame is axis-aligned
    with base_link at nominal pose).
    """
    return np.asarray(pos_base, dtype=float) - link_origin_base(parent_link)


def rgb_intrinsics(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    hfov_deg: float = D435I_RGB_HFOV_DEG,
) -> CameraIntrinsics:
    """D435i-like RGB intrinsics at a given resolution. Square pixels (fx=fy),
    focal length derived from the horizontal FOV; principal point at center."""
    fx = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return CameraIntrinsics(
        width=width, height=height, fx=fx, fy=fx, cx=width / 2, cy=height / 2
    )


@dataclass(frozen=True)
class CameraMount:
    """One camera's fixed placement on Oli's body (in the base_link frame)."""

    name: str
    parent_link: str
    pos_base: np.ndarray  # (3,) position in base_link, meters
    pitch_down_deg: float  # optical axis tilt below horizontal (about +Y)
    hfov_deg: float = D435I_RGB_HFOV_DEG


CHEST_CAM = CameraMount(
    name="chest",
    parent_link="waist_pitch_link",
    pos_base=np.array([0.092, 0.0175, 0.4336]),
    pitch_down_deg=35.0,
)
HEAD_CAM = CameraMount(
    name="head",
    parent_link="head_pitch_link",
    pos_base=np.array([0.0615, 0.0175, 0.652]),
    pitch_down_deg=0.0,
)
CAMERAS: Tuple[CameraMount, ...] = (CHEST_CAM, HEAD_CAM)
