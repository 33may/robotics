#!/usr/bin/env python3
"""Face-evidence geometry: frames, poses, framing, stability delta.
Run: p inspection/tests/test_faceview.py"""
import numpy as np

from inspection.fit.contract import Primitive
from inspection.view.faceview import (FACES, face_frame, face_cam_pose,
                                      face_radius, normals_delta_deg)


def _box_prim(l=0.20, w=0.10, h=0.15, yaw=0.0, center=(0.1, -0.35, 0.075)):
    c, s = np.cos(yaw), np.sin(yaw)
    pose = [[c, -s, 0.0, center[0]], [s, c, 0.0, center[1]],
            [0.0, 0.0, 1.0, center[2]], [0.0, 0.0, 0.0, 1.0]]
    return Primitive(kind="box", pose=pose,
                     params={"extents": [l, w, h]}, label="box")


def test_face_frame_axis_aligned():
    p = _box_prim()
    c, n, ext = face_frame(p, "x+")
    assert np.allclose(n, [1, 0, 0])
    assert np.allclose(c, [0.1 + 0.10, -0.35, 0.075])   # center + l/2 along +x
    assert ext == (0.10, 0.15)                          # (w, h) of that wall
    c, n, ext = face_frame(p, "top")
    assert np.allclose(n, [0, 0, 1])
    assert np.allclose(c, [0.1, -0.35, 0.075 + 0.075])
    assert ext == (0.20, 0.10)                          # (l, w) of the lid


def test_face_frame_follows_yaw():
    p = _box_prim(yaw=np.pi / 2)
    _, n, _ = face_frame(p, "x+")
    assert np.allclose(n, [0, 1, 0], atol=1e-9)         # +x face rotated to +y


def test_face_cam_pose_is_orthogonal_lookat():
    c, n, _ = face_frame(_box_prim(), "x+")
    T = face_cam_pose(c, n, r=0.30)
    assert np.allclose(T[:3, 3], c + 0.30 * n)          # on the normal ray
    assert np.allclose(T[:3, 0], -n)                    # boresight back at face
    assert T[2, 2] > 0.5                                # image-up is up-ish


def test_face_radius_clamps_to_rmin():
    from inspection.view.grid import R_MIN
    assert face_radius((0.01, 0.01), fy=430.0, height_px=480) == R_MIN


def test_normals_delta_yaw():
    a, b = _box_prim(yaw=0.0), _box_prim(yaw=np.deg2rad(10))
    assert abs(normals_delta_deg(a, b) - 10.0) < 0.1
    assert normals_delta_deg(a, a) < 1e-9


def main():
    test_face_frame_axis_aligned()
    test_face_frame_follows_yaw()
    test_face_cam_pose_is_orthogonal_lookat()
    test_face_radius_clamps_to_rmin()
    test_normals_delta_yaw()
    print("OK test_faceview")


if __name__ == "__main__":
    main()
