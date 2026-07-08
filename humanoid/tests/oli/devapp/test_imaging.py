"""Tests for pure-numpy imaging helpers (no GUI)."""

import numpy as np
import pytest

from humanoid.logic.oli.devapp.imaging import colorize_depth

pytestmark = pytest.mark.brain


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
