"""Tests for pure-numpy imaging helpers (no GUI)."""

import numpy as np
import pytest

from humanoid.logic.oli.devapp.imaging import colorize_depth, fit_within

pytestmark = pytest.mark.brain


def test_fit_within_height_limited():
    # 2:1 image in a box that's relatively wide → height is the binding constraint
    assert fit_within(100, 50, box_w=400, box_h=100) == (200, 100)


def test_fit_within_width_limited():
    # same image in a narrow-tall box → width binds
    assert fit_within(100, 50, box_w=100, box_h=100) == (100, 50)


def test_fit_within_upscales_to_fill():
    # small image, big box → scale UP so the panel space is used (still fully visible)
    assert fit_within(50, 50, box_w=200, box_h=200) == (200, 200)


def test_fit_within_preserves_aspect():
    w, h = fit_within(659, 1100, box_w=800, box_h=600)  # the baked nav map shape
    assert abs((w / h) - (659 / 1100)) < 1e-2
    assert w <= 800 and h <= 600


def test_fit_within_degenerate_box_is_clamped():
    assert fit_within(100, 50, box_w=0, box_h=-5) == (1, 1)


def test_colorize_depth_shape_and_dtype():
    depth = np.linspace(0.0, 6.0, 64 * 48, dtype=np.float32).reshape(48, 64)
    out = colorize_depth(depth, near=0.2, far=5.0)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_colorize_depth_near_far_ends_differ():
    depth = np.array([[0.2, 5.0]], dtype=np.float32)
    out = colorize_depth(depth, near=0.2, far=5.0)
    # near end (red-ish) and far end (blue-ish) must not be the same colour
    assert not np.array_equal(out[0, 0], out[0, 1])


def test_colorize_depth_invalid_is_black():
    depth = np.array([[np.nan, 1.0]], dtype=np.float32)
    out = colorize_depth(depth)
    assert np.array_equal(out[0, 0], [0, 0, 0])
