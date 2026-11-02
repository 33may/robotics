"""Tests for the CameraFrame contract (§5) — the 4th invariant, brain-facing frame.

Brain-pure. Locks: RGB/depth shape+dtype coercion, resolution-matches-intrinsics
invariant, and the deliberate ABSENCE of extrinsics (camera pose is derived brain-side
by FK from an Observation + the static mount table, never shipped — design.md D5).
"""

import sys

import numpy as np
import pytest

from humanoid.logic.oli import CameraFrame, CameraIntrinsics


def _intr(w: int = 8, h: int = 4) -> CameraIntrinsics:
    return CameraIntrinsics(width=w, height=h, fx=5.0, fy=5.0, cx=w / 2, cy=h / 2)


def _frame(w: int = 8, h: int = 4, stamp: int = 123) -> CameraFrame:
    return CameraFrame(
        stamp_ns=stamp,
        name="chest",
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth=np.ones((h, w), dtype=np.float32),
        intrinsics=_intr(w, h),
    )


def test_construct_valid_frame():
    f = _frame()
    assert f.name == "chest"
    assert f.rgb.shape == (4, 8, 3) and f.rgb.dtype == np.uint8
    assert f.depth.shape == (4, 8) and f.depth.dtype == np.float32
    assert f.stamp_ns == 123


def test_rgb_and_depth_coerced_to_canonical_dtypes():
    f = CameraFrame(
        stamp_ns=0,
        name="head",
        rgb=np.zeros((4, 8, 3), dtype=np.int64),
        depth=np.ones((4, 8), dtype=np.float64),
        intrinsics=_intr(),
    )
    assert f.rgb.dtype == np.uint8
    assert f.depth.dtype == np.float32


def test_rgb_must_be_hw3():
    with pytest.raises(ValueError):
        CameraFrame(
            stamp_ns=0, name="chest",
            rgb=np.zeros((4, 8), dtype=np.uint8),
            depth=np.ones((4, 8), dtype=np.float32),
            intrinsics=_intr(),
        )


def test_rgb_depth_hw_must_agree():
    with pytest.raises(ValueError):
        CameraFrame(
            stamp_ns=0, name="chest",
            rgb=np.zeros((4, 8, 3), dtype=np.uint8),
            depth=np.ones((2, 8), dtype=np.float32),
            intrinsics=_intr(),
        )


def test_resolution_must_match_intrinsics():
    with pytest.raises(ValueError):
        CameraFrame(
            stamp_ns=0, name="chest",
            rgb=np.zeros((4, 8, 3), dtype=np.uint8),
            depth=np.ones((4, 8), dtype=np.float32),
            intrinsics=_intr(w=16, h=8),
        )


def test_frame_carries_no_extrinsics():
    f = _frame()
    for forbidden in ("pose", "extrinsics", "T_base_cam", "position", "orientation"):
        assert not hasattr(f, forbidden)


def test_intrinsics_has_no_pose():
    i = _intr(8, 4)
    assert (i.width, i.height, i.cx, i.cy) == (8, 4, 4.0, 2.0)
    for forbidden in ("pose", "extrinsics"):
        assert not hasattr(i, forbidden)


def test_camera_frame_is_brain_pure():
    import humanoid.logic.oli.contracts  # noqa: F401

    assert "isaacsim" not in sys.modules
    assert "limxsdk" not in sys.modules
