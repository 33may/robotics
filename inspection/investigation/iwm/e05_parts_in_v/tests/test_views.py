import numpy as np


def test_pose_looks_along_minus_normal_and_is_right_handed():
    from inspection.investigation.iwm.e05_parts_in_v import views as Vw
    T = Vw.pose_for(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), 0.1)
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.4]) and np.allclose(T[:3, 2], [0, 0, -1])
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)
    assert np.allclose(T[:3, 0], [1, 0, 0])                 # the up axis becomes the camera x


def test_standoff_clamps():
    from inspection.investigation.iwm.e05_parts_in_v import views as Vw
    assert Vw.standoff(np.array([5.0, 5.0, 1.0])) == Vw.D_MIN                   # mm in, a 5 mm part -> the clamp
    assert Vw.D_MIN < Vw.standoff(np.array([200.0, 100.0, 10.0])) < Vw.D_MAX
    assert Vw.standoff(np.array([2000.0, 2000.0, 100.0])) == Vw.D_MAX


def test_infeasible_poses_rank_last_whatever_they_see():
    from inspection.investigation.iwm.e05_parts_in_v import views as Vw
    inside = {"visible": 1.0, "incidence_med_deg": 0.0, "centring_x": 0.0, "centring_y": 0.0, "feasible": False}
    free = {"visible": 0.5, "incidence_med_deg": 40.0, "centring_x": 0.3, "centring_y": 0.1, "feasible": True}
    assert Vw.rank_key(free) < Vw.rank_key(inside)
    assert Vw.feasible(0.019) is False and Vw.feasible(0.05) is True       # 2 cm of free space around the camera


def test_shift_moves_the_camera_in_the_face_plane_and_keeps_the_look_direction():
    from inspection.investigation.iwm.e05_parts_in_v import views as Vw
    c, n, up = np.array([0.0, 0.0, 0.0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    T0 = Vw.pose_for(c, n, up, 0.105)
    T1 = Vw.shifted(T0, up, 0.03)
    assert np.allclose(T1[:3, :3], T0[:3, :3])                                  # same orientation: still face-on
    assert np.allclose(T1[:3, 3], [0.03, 0.0, 0.105])                           # moved 3 cm along the face's up axis
    names = [name for name, _ in Vw.candidates(c, n, up, 0.105, [])]
    assert "shift+2up" in names and "shift-4side" in names                      # the family is on offer


def test_axis_angle_is_zero_for_shifts_and_the_tilt_for_tilts():
    from inspection.investigation.iwm.e05_parts_in_v import views as Vw
    c, n, up = np.array([0.0, 0.0, 0.0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    T0 = Vw.pose_for(c, n, up, 0.105)
    assert np.isclose(Vw.axis_deg(T0, n), 0.0, atol=1e-2)
    assert np.isclose(Vw.axis_deg(Vw.shifted(T0, up, 0.04), n), 0.0, atol=1e-2)   # a shift keeps the image plane parallel
    assert np.isclose(Vw.axis_deg(Vw.tilted(T0, c, up, 30), n), 30.0, atol=1e-2)  # a tilt does not


def test_tilt_keeps_the_centre_in_view_and_the_distance():
    from inspection.investigation.iwm.e05_parts_in_v import views as Vw
    c, n, up = np.array([0.0, 0.0, 0.0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    T0 = Vw.pose_for(c, n, up, 0.2)
    T1 = Vw.tilted(T0, c, axis=up, deg=30)
    assert np.isclose(np.linalg.norm(T1[:3, 3] - c), 0.2)                       # same distance
    look = -T1[:3, 2]
    assert np.isclose(np.degrees(np.arccos(look @ n)), 30, atol=1e-6)           # 30 deg off the normal
    assert np.allclose(T1[:3, 3] + 0.2 * T1[:3, 2], c)                          # still looking at the centre
