import numpy as np
from inspection.cell.world import RobotCell
from inspection.fit.contract import Primitive
from inspection.motion.ik import UR5eIK
from inspection.view.viewsphere import T_FLANGE_CAM


def _box_world():
    w = RobotCell()
    w.set_object("object", [0.15, 0.15, 0.075],
                 [0.05, -0.38, 0.0375, 0, 0, 0.17])
    return w, UR5eIK()


def _box_prim(l=0.20, w=0.10, h=0.15, yaw=0.0, center=(0.1, -0.35, 0.075)):
    c, s = np.cos(yaw), np.sin(yaw)
    pose = [[c, -s, 0.0, center[0]], [s, c, 0.0, center[1]],
            [0.0, 0.0, 1.0, center[2]], [0.0, 0.0, 0.0, 1.0]]
    return Primitive(kind="box", pose=pose,
                     params={"extents": [l, w, h]}, label="box")


def test_samples_ascend_and_respect_sensor_floor():
    from inspection.view.rays import Ray, sample_ray, MIN_RANGE_M
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, -0.9701, 0.2425))
    out = sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=0.075)
    assert out, "an open oblique ray must have samples"
    depths = [s.depth for s in out]
    assert depths == sorted(depths)
    assert depths[0] - 0.075 >= MIN_RANGE_M - 1e-9


def test_clear_m_raises_the_floor():
    """The shallow oblique ray above never has a geometrically-free sample
    under ~0.20-0.22 m regardless of clear_m, so depths[0] - clear_m >=
    MIN_RANGE_M alone cannot tell a live sensor floor from a dead one
    (reviewer finding on 9d6dc1d). Isolate it: same ray, two clearances,
    and prove the floor — not geometry — is what moved the answer.

    clear_m=0.25 here, not 0.30: clear_m + MIN_RANGE_M must stay <= R_MAX
    (0.40 m) for ANY floor-bound sample to exist — 0.30 + MIN_RANGE_M
    (0.405) exceeds R_MAX unconditionally, on every ray, so that clearance
    can never come back non-empty. The ray direction is also steeper here
    (45 deg vs the ~14 deg ray above): at 0.25 m clearance the admissible
    floor (0.355 m) sits inside a band the arm can actually still reach —
    verified empirically against this world model, not assumed.
    """
    from inspection.view.rays import Ray, sample_ray, MIN_RANGE_M
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, -0.7071, 0.7071))
    near = sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=0.075)
    far = sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=0.25)
    assert far, "0.25 m clearance must still leave a reachable, in-range sample"
    assert far[0].depth >= 0.25 + MIN_RANGE_M - 1e-9
    # the two runs disagree BECAUSE of clear_m: near's first depth would be
    # rejected as too-close by far's floor, proving the floor is binding.
    assert near[0].depth < 0.25 + MIN_RANGE_M


def test_no_sample_collides():
    from inspection.view.rays import Ray, sample_ray
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, -0.9701, 0.2425))
    for s in sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=0.075):
        assert not w.is_colliding(s.q)


def test_buried_ray_is_empty():
    from inspection.view.rays import Ray, sample_ray
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, 0.0, -1.0))   # into the table
    assert sample_ray(w, ik, T_FLANGE_CAM, ray) == []


def test_sample_pose_is_on_the_ray_looking_at_anchor():
    from inspection.view.rays import Ray, sample_ray
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, -0.9701, 0.2425))
    s = sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=0.075)[0]
    pos = s.T_cam[:3, 3]
    d = pos - np.asarray(ray.anchor)
    assert np.allclose(d / np.linalg.norm(d), ray.direction, atol=1e-6)
    assert np.allclose(s.T_cam[:3, 0], -np.asarray(ray.direction), atol=1e-6)


def test_cell_ray_matches_grid_direction():
    from inspection.view.rays import cell_ray
    r = cell_ray((0.05, -0.38, 0.0375), h=0, v=0)
    # h=0 faces the base: direction's azimuth == azimuth of center->origin
    az = np.arctan2(0.38, -0.05)
    assert np.isclose(np.arctan2(r.direction[1], r.direction[0]), az,
                      atol=1e-6)


def test_cell_clear_is_box_surface_run():
    from inspection.view.rays import Ray, cell_clear_m
    aabb = ((-0.025, -0.455, 0.0), (0.125, -0.305, 0.075))
    ray = Ray((0.05, -0.38, 0.0375), (1.0, 0.0, 0.0))
    assert np.isclose(cell_clear_m(aabb, ray), 0.075, atol=1e-6)


def test_face_family_shifts_before_tilts():
    from inspection.view.rays import face_family
    p = _box_prim()                      # copied from test_faceview.py
    fam = face_family(p, "y-", fy=647.0, height_px=720)
    shifts = [(s, t) for _, s, t, _ in fam if t == 0.0]
    tilts = [(s, t) for _, s, t, _ in fam if t > 0.0]
    assert shifts and tilts
    # the family is ordered: ALL shift entries strictly before ANY tilt
    assert [t for _, _, t, _ in fam[:len(shifts)]] == [0.0] * len(shifts)
    assert [s for s, _ in shifts] == sorted(s for s, _ in shifts)
    assert all(s == 0.0 for s, _ in tilts)


def test_face_family_dmin_is_containment():
    """Controller ruling (task-2-report.md): FOV containment is a per-depth
    constraint, so it cannot cap a shift at a fixed MIN_RANGE_M depth — it
    becomes each family member's own minimum admissible depth instead."""
    from inspection.view.rays import (face_family, _d_contain, MIN_RANGE_M,
                                      R_MAX)
    p = _box_prim()
    fam = face_family(p, "y-", fy=647.0, height_px=720)
    ext_b = p.params["extents"][2]
    shifts = [(s, d_min) for _, s, t, d_min in fam if t == 0.0]
    for s, d_min in shifts:
        assert d_min >= _d_contain(647.0, 720, s + ext_b / 2.0) - 1e-9
        assert d_min <= R_MAX + 1e-9
    assert [d for _, d in shifts] == sorted(d for _, d in shifts)
    assert all(d_min >= MIN_RANGE_M - 1e-9 for _, _, _, d_min in fam)
    # the 15cm _box_prim face: s=0 is now a member with a *floor* depth
    # (not a rejection) — this is the exact number that broke the old
    # fixed-MIN_RANGE_M cap.
    s0, d0 = shifts[0]
    assert s0 == 0.0
    assert np.isclose(d0, (2 * 647.0 / 720) * (ext_b / 2.0), atol=1e-9)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK test_rays")


if __name__ == "__main__":
    main()
