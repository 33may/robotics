"""Tests for nvblox_inject — feeding recorded sensor depth into the NVIDIA
offline occupancy step (MAY-173 Phase 2.2, vendor-pipeline-first).

The container's `occupancy` step (nvblox fuse_cusfm) eats `frames_meta.json`
keyframes + per-keyframe depth PNGs; FoundationStereo inference only ever
existed to FILL that depth dir. We have real sensor depth (uint16 mm, the exact
format), so the injector maps dump streams onto keyframes:

  - head depth  → depth/head_left/<stamp>.png  (2.5 cm offset, sub-cell)
  - chest depth → new `chest` keyframes: camera_to_world composed from the
    bake's OWN head_left poses ∘ static FK offset (robot description, no GT)

The convention bridge (USD camera → the meta's optical frame) is validated
against the meta's own head_left `sensor_to_vehicle` entry before anything is
written — a bake with different conventions fails loudly, never silently.
"""

import math

import numpy as np
import pytest

from humanoid.logic.oli.camera_mounts import CHEST_CAM, STEREO_CAMERAS
from humanoid.logic.oli.recording.fk import T_base_cam_usd
from humanoid.logic.simulation.mapping.nvblox_inject import (
    T_from_meta,
    axis_angle_from_R,
    meta_from_T,
    optical_from_usd,
    t_headleft_chest_optical,
)

pytestmark = pytest.mark.brain

# the meta's own head_left sensor_to_vehicle (level camera, optical convention)
_HL_META = {
    "translation": {"x": 0.0615, "y": 0.0425, "z": 0.652},
    "axis_angle": {"x": -0.5773502691896257, "y": 0.5773502691896257,
                   "z": -0.5773502691896257, "angle_degrees": 120.0},
}


def test_axis_angle_roundtrip():
    for axis, ang in [((0, 0, 1), 0.5), ((1, -1, 1), 2.0), ((0, 1, 0), math.pi - 0.01)]:
        a = np.asarray(axis, float)
        a /= np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        got_a, got_ang = axis_angle_from_R(R)
        # axis/angle sign pair is only defined up to simultaneous negation
        if np.dot(got_a, a) < 0:
            got_a, got_ang = -got_a, -got_ang
        np.testing.assert_allclose(got_a, a, atol=1e-9)
        assert got_ang % (2 * math.pi) == pytest.approx(ang, abs=1e-9)


def test_optical_convention_matches_bake_meta():
    """THE pin: our USD→optical bridge reproduces the bake's own head_left
    sensor_to_vehicle entry exactly. If a future bake changes conventions,
    this (and the injector's runtime self-check) goes red."""
    hl = next(m for m in STEREO_CAMERAS if m.name == "head_left")
    T = optical_from_usd(T_base_cam_usd(hl))
    expect = T_from_meta(_HL_META)
    np.testing.assert_allclose(T, expect, atol=1e-9)


def test_meta_serialization_roundtrip():
    hl = next(m for m in STEREO_CAMERAS if m.name == "head_left")
    T = optical_from_usd(T_base_cam_usd(hl))
    np.testing.assert_allclose(T_from_meta(meta_from_T(T)), T, atol=1e-9)


def test_chest_offset_geometry():
    """Chest sits 3.05 cm ahead / 2.5 cm right / 21.8 cm below head_left in the
    base frame; the relative OPTICAL transform must reproduce that lever arm
    and the 35° pitch split."""
    rel = t_headleft_chest_optical()
    lever = np.linalg.norm(rel[:3, 3])
    expect = np.linalg.norm(np.asarray(CHEST_CAM.pos_base) - [0.0615, 0.0425, 0.652])
    assert lever == pytest.approx(expect, abs=1e-9)
    # relative rotation angle = chest pitch (head_left is level)
    _, ang = axis_angle_from_R(rel[:3, :3])
    assert abs(math.degrees(ang)) == pytest.approx(35.0, abs=1e-6)


def test_chest_pose_composition_at_identity():
    """With head_left at its start pose (vehicle at map origin), the composed
    chest camera_to_world's translation must equal the chest mount position."""
    hl = next(m for m in STEREO_CAMERAS if m.name == "head_left")
    T_map_hl = optical_from_usd(T_base_cam_usd(hl))   # vehicle at origin
    T_map_chest = T_map_hl @ t_headleft_chest_optical()
    np.testing.assert_allclose(T_map_chest[:3, 3], CHEST_CAM.pos_base, atol=1e-9)
    # and its view axis (+Z optical) pitches 35° below horizontal
    view = T_map_chest[:3, :3] @ [0, 0, 1]
    assert math.degrees(math.asin(-view[2])) == pytest.approx(35.0, abs=1e-6)
