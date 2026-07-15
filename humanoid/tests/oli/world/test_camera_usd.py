"""Tests for baking the D435i cameras into the robot USD sensor layer (§3).

Runs in the `isaac` env (needs `pxr`). Operates on a TEMP COPY of the asset USD tree
so the committed asset is never mutated. Verifies the composed stage: two Camera
prims parented to the moving links, at the manual mounts, looking the right way, and
that re-baking is idempotent (design.md D1/D3, spec: cameras baked into the USD).
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom  # noqa: E402

from humanoid.logic.oli.camera_mounts import (  # noqa: E402
    CAMERAS,
    D435I_STEREO_BASELINE_M,
    HEAD_CAM,
    STEREO_CAMERAS,
)
from humanoid.logic.simulation.isaacsim.build_camera_usd import (  # noqa: E402
    bake_cameras,
    camera_prim_path,
)

pytestmark = pytest.mark.isaac

_ASSET_USD = (
    Path(__file__).resolve().parents[3]
    / "assets" / "oli" / "usd"
)


@pytest.fixture
def baked_stage(tmp_path):
    """Copy the asset USD tree to tmp, bake cameras into its sensor layer, return the
    composed top stage + the sensor layer path (for idempotency re-runs)."""
    dst = tmp_path / "usd"
    shutil.copytree(_ASSET_USD, dst)
    sensor = dst / "configuration" / "HU_D04_01_sensor.usd"
    bake_cameras(sensor)
    stage = Usd.Stage.Open(str(dst / "HU_D04_01.usd"))
    return stage, sensor


def _world_pose(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid() and prim.GetTypeName() == "Camera", f"{prim_path} not a Camera"
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = np.array(m.ExtractTranslation())
    R = np.array(m.ExtractRotationMatrix()).T  # Gf row-vector convention → column basis
    view = R @ np.array([0.0, 0.0, -1.0])  # USD camera looks down local -Z
    return pos, view


def test_both_cameras_baked_as_camera_prims(baked_stage):
    stage, _ = baked_stage
    for m in CAMERAS:
        prim = stage.GetPrimAtPath(camera_prim_path(m))
        assert prim.IsValid(), f"{m.name} camera prim missing"
        assert prim.GetTypeName() == "Camera"


def test_chest_camera_world_pose(baked_stage):
    stage, _ = baked_stage
    pos, view = _world_pose(stage, camera_prim_path(CAMERAS[0]))  # chest
    np.testing.assert_allclose(pos, [0.092, 0.0175, 0.4336], atol=1e-4)
    # 35° down, looking forward (+X): (cos35, 0, -sin35)
    expected = [np.cos(np.radians(35)), 0.0, -np.sin(np.radians(35))]
    np.testing.assert_allclose(view, expected, atol=1e-3)


def test_head_camera_world_pose(baked_stage):
    stage, _ = baked_stage
    pos, view = _world_pose(stage, camera_prim_path(CAMERAS[1]))  # head
    np.testing.assert_allclose(pos, [0.0615, 0.0175, 0.652], atol=1e-4)
    np.testing.assert_allclose(view, [1.0, 0.0, 0.0], atol=1e-3)  # horizontal forward


def test_camera_has_d435i_fov(baked_stage):
    stage, _ = baked_stage
    prim = stage.GetPrimAtPath(camera_prim_path(CAMERAS[0]))
    cam = UsdGeom.Camera(prim)
    focal = cam.GetFocalLengthAttr().Get()
    h_ap = cam.GetHorizontalApertureAttr().Get()
    hfov_deg = np.degrees(2 * np.arctan(h_ap / (2 * focal)))
    np.testing.assert_allclose(hfov_deg, 69.0, atol=0.5)


def test_head_stereo_pair_baked_at_baseline(baked_stage):
    # MAY-173 locdev T1: the head stereo pair (D435i-faithful 50 mm) straddles the
    # head mount, both looking horizontal forward like the head cam.
    stage, _ = baked_stage
    half = D435I_STEREO_BASELINE_M / 2
    left, right = STEREO_CAMERAS
    pos_l, view_l = _world_pose(stage, camera_prim_path(left))
    pos_r, view_r = _world_pose(stage, camera_prim_path(right))
    np.testing.assert_allclose(pos_l, HEAD_CAM.pos_base + [0.0, half, 0.0], atol=1e-4)
    np.testing.assert_allclose(pos_r, HEAD_CAM.pos_base + [0.0, -half, 0.0], atol=1e-4)
    np.testing.assert_allclose(view_l, [1.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(view_r, [1.0, 0.0, 0.0], atol=1e-3)
    baseline = np.linalg.norm(pos_l - pos_r)
    np.testing.assert_allclose(baseline, D435I_STEREO_BASELINE_M, atol=1e-6)


def test_bake_is_idempotent(baked_stage):
    stage, sensor = baked_stage
    bake_cameras(sensor)  # second run
    stage2 = Usd.Stage.Open(str(sensor.parent.parent / "HU_D04_01.usd"))
    cams = [p for p in stage2.Traverse() if p.GetTypeName() == "Camera"]
    assert len(cams) == 4, f"expected 4 cameras after re-bake, got {len(cams)}"
