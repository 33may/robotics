"""Tests for the camera source protocol + synthetic source (pure numpy, no GUI)."""

import numpy as np
import pytest

from humanoid.logic.oli.devapp.sources.camera_source import CameraFrame, CameraSource
from humanoid.logic.oli.devapp.sources.synthetic_camera_source import SyntheticCameraSource

pytestmark = pytest.mark.brain


def test_stream_names_default():
    src = SyntheticCameraSource()
    assert src.stream_names() == ["chest", "head"]


def test_conforms_to_protocol():
    # runtime_checkable Protocol: the synthetic source is a valid CameraSource.
    assert isinstance(SyntheticCameraSource(), CameraSource)


def test_read_returns_valid_frame():
    src = SyntheticCameraSource(width=64, height=48)
    f = src.read("chest")
    assert isinstance(f, CameraFrame)
    assert f.name == "chest"
    assert f.rgb.shape == (48, 64, 3)
    assert f.rgb.dtype == np.uint8
    assert f.depth.shape == (48, 64)
    assert f.depth.dtype == np.float32
    assert np.all(np.isfinite(f.depth))
    assert f.intrinsics is not None and f.intrinsics.width == 64


def test_unknown_stream_returns_none():
    assert SyntheticCameraSource().read("nose") is None


def test_frames_animate():
    src = SyntheticCameraSource(width=64, height=48)
    a = src.read("chest")
    b = src.read("chest")
    assert a.stamp_ns != b.stamp_ns
    assert not np.array_equal(a.rgb, b.rgb)  # the moving marker changes the image


def test_streams_are_independent():
    src = SyntheticCameraSource(width=64, height=48)
    src.read("chest")  # advance chest only
    assert src.read("head").stamp_ns == 0  # head still on its first frame
