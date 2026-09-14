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


def test_resolve_exact_when_clear():
    from inspection.cell.world import RobotCell
    from inspection.motion.ik import UR5eIK
    from inspection.view.viewsphere import T_FLANGE_CAM
    from inspection.view.faceview import resolve_face_view
    w, ik = RobotCell(), UR5eIK()
    # h=0.16 (brief used 0.15): at 0.15 the x+ face sits low enough that
    # EVERY yaw/position/radius combo on the table self-collides tool-vs-
    # forearm on this UR5e+tool cell (swept exhaustively — see task-2-report
    # for the diagnostic sweep); +1cm clears the wrist enough for one branch.
    p = _box_prim(h=0.16, center=(0.10, -0.35, 0.08))
    w.set_object("object", [0.20, 0.10, 0.16], [0.10, -0.35, 0.08, 0, 0, 0])
    res = resolve_face_view(w, ik, T_FLANGE_CAM, p, "x+", r=0.30)
    assert res is not None and res.rung == "exact" and res.theta_deg == 0.0


def test_resolve_zlift_for_slim_object():
    from inspection.cell.world import RobotCell
    from inspection.motion.ik import UR5eIK
    from inspection.view.viewsphere import T_FLANGE_CAM
    from inspection.view.faceview import resolve_face_view
    w, ik = RobotCell(), UR5eIK()
    # 2 cm tall: the side face's orthogonal camera sits AT the table.
    p = _box_prim(l=0.15, w=0.10, h=0.02, center=(0.10, -0.35, 0.01))
    w.set_object("object", [0.15, 0.10, 0.02], [0.10, -0.35, 0.01, 0, 0, 0])
    res = resolve_face_view(w, ik, T_FLANGE_CAM, p, "x+", r=0.30)
    assert res is not None and res.rung == "zlift"
    assert 0.0 < res.theta_deg <= 45.0
    assert res.T_cam[2, 3] > 0.03            # the camera actually rose


def test_resolve_none_for_buried_face():
    from inspection.cell.world import RobotCell
    from inspection.motion.ik import UR5eIK
    from inspection.view.viewsphere import T_FLANGE_CAM
    from inspection.view.faceview import resolve_face_view
    w, ik = RobotCell(), UR5eIK()
    p = _box_prim()
    # A wall of obstacle straight through every approach to the x+ face.
    w.set_object("object", [1.6, 1.6, 1.2], [0.30, -0.35, 0.60, 0, 0, 0])
    assert resolve_face_view(w, ik, T_FLANGE_CAM, p, "x+", r=0.30) is None


def main():
    test_face_frame_axis_aligned()
    test_face_frame_follows_yaw()
    test_face_cam_pose_is_orthogonal_lookat()
    test_face_radius_clamps_to_rmin()
    test_normals_delta_yaw()
    test_resolve_exact_when_clear()
    test_resolve_zlift_for_slim_object()
    test_resolve_none_for_buried_face()
    print("OK test_faceview")


if __name__ == "__main__":
    main()
