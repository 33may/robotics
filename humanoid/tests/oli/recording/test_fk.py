"""TDD for recording/fk.py — brain-side camera FK (MAY-173 slam-demo-loop 1.4).

The Robot-side recorder has no camera prims to read, so it derives each frame's
`T_world_cam` (USD convention, column-vector) by FK: GT base pose (debug-pose
channel) ∘ static mount (camera_mounts table — the SAME numbers the USD bake
authored, `build_camera_usd._local_transform` math). Constant offsets (glide
height, settled pitch) are absorbed by `rosbag_synth.recover_static_mount`,
which only needs base↔cam CONSISTENCY, not absolute truth.

Brain-pure: numpy + camera_mounts only.
"""

import math
import sys

import numpy as np
import pytest

from humanoid.logic.oli.camera_mounts import HEAD_CAM, STEREO_CAMERAS
from humanoid.logic.oli.recording.fk import (
    T_base_cam_usd,
    T_world_base,
    cam_world,
    rig_dict,
)

pytestmark = pytest.mark.brain


# ── T_world_base ─────────────────────────────────────────────────────────────
def test_world_base_identity_at_origin():
    np.testing.assert_allclose(T_world_base(0.0, 0.0, 0.0), np.eye(4), atol=1e-12)


def test_world_base_translation_and_yaw():
    T = T_world_base(2.0, -1.0, math.pi / 2)
    np.testing.assert_allclose(T[:3, 3], [2.0, -1.0, 0.0], atol=1e-12)
    # +X base axis maps to +Y world under a 90° yaw
    np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


# ── T_base_cam_usd (mirrors build_camera_usd._local_transform, base-relative) ──
def test_head_cam_axes_pitch_down():
    T = T_base_cam_usd(HEAD_CAM)
    th = math.radians(HEAD_CAM.pitch_down_deg)
    view = np.array([math.cos(th), 0.0, -math.sin(th)])
    # USD camera looks down local -Z → column Z is -view
    np.testing.assert_allclose(T[:3, 2], -view, atol=1e-12)
    # right axis (+X_local) horizontal, no roll
    np.testing.assert_allclose(T[:3, 0], [0.0, -1.0, 0.0], atol=1e-12)
    # rotation is proper orthonormal
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    # translation = the mount's base-frame position
    np.testing.assert_allclose(T[:3, 3], HEAD_CAM.pos_base, atol=1e-12)


def test_stereo_pair_differs_only_by_baseline():
    left, right = STEREO_CAMERAS
    Tl, Tr = T_base_cam_usd(left), T_base_cam_usd(right)
    np.testing.assert_allclose(Tl[:3, :3], Tr[:3, :3], atol=1e-12)  # same orientation
    base_vec = Tr[:3, 3] - Tl[:3, 3]
    # separation along the camera right axis, |b| = the configured baseline
    np.testing.assert_allclose(
        np.linalg.norm(base_vec), np.linalg.norm(right.pos_base - left.pos_base),
        atol=1e-12)


# ── composition + mount-recovery consistency ──────────────────────────────────
def test_cam_world_composes():
    left = STEREO_CAMERAS[0]
    T = cam_world(1.0, 2.0, math.pi / 4, left)
    np.testing.assert_allclose(
        T, T_world_base(1.0, 2.0, math.pi / 4) @ T_base_cam_usd(left), atol=1e-12)


def test_static_mount_is_constant_across_poses():
    """The bake recovers base→cam as a median over stamps — FK must make it EXACTLY
    constant: inv(T_world_base) @ T_world_cam identical for any base pose."""
    left = STEREO_CAMERAS[0]
    mounts = []
    for (x, y, yaw) in [(0, 0, 0), (3.2, -1.1, 0.7), (-8.0, 19.0, -2.9)]:
        Twb = T_world_base(x, y, yaw)
        mounts.append(np.linalg.inv(Twb) @ cam_world(x, y, yaw, left))
    np.testing.assert_allclose(mounts[0], mounts[1], atol=1e-10)
    np.testing.assert_allclose(mounts[0], mounts[2], atol=1e-10)


# ── rig_dict ─────────────────────────────────────────────────────────────────
def test_rig_dict_shape_matches_bake_contract():
    rig = rig_dict((1280, 720))
    assert rig["stereo_pair"] == ["head_left", "head_right"]
    assert rig["rgbd"] == ["head"]
    assert set(rig["cameras"]) == {"head", "head_left", "head_right"}
    head = rig["cameras"]["head"]
    assert head["intrinsics"]["width"] == 1280
    assert {"parent_link", "pos_base", "pitch_down_deg", "intrinsics"} <= set(head)
    assert rig["baseline_m"] > 0


def test_module_is_brain_pure():
    import humanoid.logic.oli.recording.fk  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules
