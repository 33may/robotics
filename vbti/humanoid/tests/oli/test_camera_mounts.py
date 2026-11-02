"""Tests for the shared, World-agnostic camera mount table (logic.oli.camera_mounts).

Brain-pure (no isaacsim/limxsdk): this table is the single source of truth for where
Oli's two RealSense D435i cameras sit, consumed BOTH by the brain's FK-based pose
derivation and by the Isaac `build_camera_usd.py` (design.md D10). Mounts from
`oli-corpus://user-manual#1.4.1`; kinematic chain from the HU_D04_01 URDF (rpy=0
throughout → pure-translation, verified 2026-07-01). Runs in the `brain` env.
"""

import sys

import numpy as np

from humanoid.logic.oli.camera_mounts import (
    CAMERAS,
    D435I_STEREO_BASELINE_M,
    HEAD_CAM,
    STEREO_CAMERAS,
    CameraIntrinsics,
    CameraMount,
    link_origin_base,
    rgb_intrinsics,
    stereo_pair,
    to_parent_local,
)


def _cam(name: str) -> CameraMount:
    return next(c for c in CAMERAS if c.name == name)


def test_two_cameras_chest_and_head():
    assert {c.name for c in CAMERAS} == {"chest", "head"}


def test_chest_mount_matches_manual():
    chest = _cam("chest")
    assert chest.parent_link == "waist_pitch_link"
    np.testing.assert_allclose(chest.pos_base, [0.092, 0.0175, 0.4336])
    assert chest.pitch_down_deg == 35.0
    assert chest.hfov_deg == 69.0  # D435i RGB horizontal FOV


def test_head_mount_matches_manual():
    head = _cam("head")
    assert head.parent_link == "head_pitch_link"
    np.testing.assert_allclose(head.pos_base, [0.0615, 0.0175, 0.652])
    assert head.pitch_down_deg == 0.0


def test_link_origin_base_at_nominal_pose():
    # Summed URDF joint origins (all rpy=0): waist_yaw 0.10239 + waist_roll 0.057.
    np.testing.assert_allclose(
        link_origin_base("waist_pitch_link"), [0.0, 0.0, 0.15939], atol=1e-9
    )
    # + head_yaw (-0.013, 0, 0.3882) + head_pitch (0, 0, 0.0395).
    np.testing.assert_allclose(
        link_origin_base("head_pitch_link"), [-0.013, 0.0, 0.58709], atol=1e-9
    )


def test_chest_to_parent_local():
    chest = _cam("chest")
    local = to_parent_local(chest.pos_base, chest.parent_link)
    np.testing.assert_allclose(local, [0.092, 0.0175, 0.27421], atol=1e-6)


def test_head_to_parent_local():
    head = _cam("head")
    local = to_parent_local(head.pos_base, head.parent_link)
    np.testing.assert_allclose(local, [0.0745, 0.0175, 0.06491], atol=1e-6)


def test_d435i_rgb_intrinsics_default():
    intr = rgb_intrinsics()
    assert isinstance(intr, CameraIntrinsics)
    assert (intr.width, intr.height) == (1280, 720)
    assert (intr.cx, intr.cy) == (640.0, 360.0)
    expected_fx = (1280 / 2) / np.tan(np.radians(69.0) / 2)
    np.testing.assert_allclose(intr.fx, expected_fx, rtol=1e-9)
    np.testing.assert_allclose(intr.fy, expected_fx, rtol=1e-9)  # square pixels


def test_intrinsics_scale_with_resolution():
    intr = rgb_intrinsics(width=640, height=360)
    assert (intr.width, intr.height) == (640, 360)
    assert (intr.cx, intr.cy) == (320.0, 180.0)
    # fx scales linearly with width for a fixed FOV.
    full = rgb_intrinsics(width=1280, height=720)
    np.testing.assert_allclose(intr.fx, full.fx / 2, rtol=1e-9)


def test_module_is_brain_pure():
    import humanoid.logic.oli.camera_mounts  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules


# ── Stereo pair (MAY-173 locdev T1: cuVGL/map-bake input) ────────────────────


def _stereo(name: str) -> CameraMount:
    return next(c for c in STEREO_CAMERAS if c.name == name)


def test_stereo_baseline_is_d435i_faithful():
    # The D435i's physical left/right imager separation: 50 mm.
    assert D435I_STEREO_BASELINE_M == 0.050


def test_head_stereo_pair_derived_from_head_mount():
    assert {c.name for c in STEREO_CAMERAS} == {"head_left", "head_right"}
    for cam in STEREO_CAMERAS:
        assert cam.parent_link == HEAD_CAM.parent_link
        assert cam.pitch_down_deg == HEAD_CAM.pitch_down_deg
        assert cam.hfov_deg == HEAD_CAM.hfov_deg


def test_head_stereo_offsets_straddle_head_mount():
    # Head cam has 0 pitch: camera-right = base -Y (image-sense left = robot +Y,
    # matching RealSense: infra1 IS the left imager).
    left, right = _stereo("head_left"), _stereo("head_right")
    half = D435I_STEREO_BASELINE_M / 2
    np.testing.assert_allclose(
        left.pos_base, HEAD_CAM.pos_base + [0.0, half, 0.0], atol=1e-9
    )
    np.testing.assert_allclose(
        right.pos_base, HEAD_CAM.pos_base + [0.0, -half, 0.0], atol=1e-9
    )


def test_stereo_pair_preserves_baseline_for_pitched_mounts():
    # stereo_pair() must offset along the camera's LOCAL right axis: for any
    # pitch-about-Y mount (no roll) the pair separation stays exactly the baseline
    # and is orthogonal to the optical axis.
    pitched = CameraMount(
        name="probe",
        parent_link="waist_pitch_link",
        pos_base=np.array([0.1, 0.0, 0.4]),
        pitch_down_deg=35.0,
    )
    left, right = stereo_pair(pitched)
    sep = left.pos_base - right.pos_base
    np.testing.assert_allclose(np.linalg.norm(sep), D435I_STEREO_BASELINE_M, atol=1e-9)
    th = np.radians(pitched.pitch_down_deg)
    view = np.array([np.cos(th), 0.0, -np.sin(th)])
    np.testing.assert_allclose(np.dot(sep, view), 0.0, atol=1e-9)


def test_stereo_cameras_do_not_leak_into_rgbd_table():
    # CAMERAS is the RGBD streaming table (cuvslam candidate, glide dev-app) —
    # the stereo pair is a separate, opt-in rig.
    assert {c.name for c in CAMERAS} == {"chest", "head"}
