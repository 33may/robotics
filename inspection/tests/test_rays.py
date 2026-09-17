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
    assert d0 >= _d_contain(647.0, 720, ext_b / 2.0) - 1e-9


def test_resolve_cell_picks_closest_sample():
    """The resolver returns the closest ADMISSIBLE sample — admissible now
    includes the whole-object containment floor (2026-09-17), so the raw
    comparison carries the same effective clearance resolve_cell uses."""
    from inspection.view.rays import (MIN_RANGE_M, cell_clear_m,
                                      cell_d_contain, cell_ray, resolve_cell,
                                      sample_ray)
    w, ik = _box_world()
    aabb = ((-0.025, -0.455, 0.0), (0.125, -0.305, 0.075))
    res = resolve_cell(w, ik, T_FLANGE_CAM, (0.05, -0.38, 0.0375), aabb,
                       h=3, v=1, fy=647.0, height_px=720)
    assert res is not None
    ray = cell_ray((0.05, -0.38, 0.0375), 3, 1)
    clear = max(cell_clear_m(aabb, ray),
                cell_d_contain(aabb, ray, 647.0, 720) - MIN_RANGE_M)
    raw = sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=clear)
    assert np.isclose(res.depth, raw[0].depth)


def test_resolve_face_side_uses_shift_family():
    # AMENDED by ruling 2026-09-17: execution proved no fixture lands in
    # the "shift>0 wins" gap (tall box: exact clears; real-height box: the
    # whole shift family is blocked and the cone wins). The unit contract
    # is the FAMILY: theta==0 proves a fronto-parallel member won, and
    # Task 2's ordering test already pins shifts-before-tilts.
    from inspection.view.rays import MIN_RANGE_M, resolve_face
    w, ik = _box_world()
    p = _box_prim()
    res = resolve_face(w, ik, T_FLANGE_CAM, p, "y-", fy=647.0,
                       height_px=720)
    assert res is not None
    assert res.theta_deg == 0.0, "shift family must win before any tilt"
    assert res.shift >= 0.0
    assert res.depth >= MIN_RANGE_M - 1e-9


def test_resolve_face_buried_is_none():
    from inspection.view.rays import resolve_face
    w, ik = _box_world()
    p = _box_prim(center=(0.05, -0.38, -0.5))     # face underground
    assert resolve_face(w, ik, T_FLANGE_CAM, p, "y-", fy=647.0,
                        height_px=720) is None


def test_face_family_contains_whole_rotated_face():
    """Anton 2026-09-17: 'check the whole face is actually in the frame' —
    every SHIFT/TOP family member (the rungs that promise containment; the
    cone is the declared degraded rung) must fit all four face corners
    inside the frame at its own d_min, with the real sensor's asymmetric
    principal point (D405 ppy≈356.7, not H/2) and the box yaw rotating the
    top face's image footprint. The old bound used the unrotated extent and
    cut demo1's top-face corners by 41 px."""
    from inspection.view.faceview import _FACE_DEF, face_cam_pose
    from inspection.view.rays import face_family
    fy, H, W = 647.0, 720, 1280
    K = np.array([[fy, 0.0, 633.0], [0.0, fy, 356.7], [0.0, 0.0, 1.0]])
    p = _box_prim(l=0.1485, w=0.149, h=0.0753, yaw=np.deg2rad(10.0),
                  center=(0.05, -0.376, 0.0465))      # demo1's belief box
    T_p = np.array(p.pose)
    he = np.asarray(p.params["extents"]) / 2.0
    sgn = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1)
                    for c in (-1, 1)])
    corners8 = T_p[:3, 3] + (sgn * he) @ T_p[:3, :3].T
    for face in ("top", "y-", "x+"):
        ax, sg, _ = _FACE_DEF[face]
        four = corners8[[i for i in range(8) if sgn[i][ax] == sg]]
        for ray, s, theta, d_min in face_family(p, face, fy, H):
            if theta != 0.0:
                continue                              # cone: exempt by design
            T = face_cam_pose(np.array(ray.anchor), np.array(ray.direction),
                              float(d_min), 0.0)
            R = np.column_stack([-T[:3, 1], -T[:3, 2], T[:3, 0]])
            pc = (four - T[:3, 3]) @ R
            assert np.all(pc[:, 2] > 0)
            uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2]
            assert uv[:, 0].min() >= 0 and uv[:, 0].max() <= W, \
                (face, s, d_min, uv[:, 0].min(), uv[:, 0].max())
            assert uv[:, 1].min() >= 0 and uv[:, 1].max() <= H, \
                (face, s, d_min, uv[:, 1].min(), uv[:, 1].max())


def test_resolve_cell_contains_whole_aabb():
    """Anton 2026-09-17 (extending the face ruling to cells): a cell view's
    promise is the whole OBJECT in frame — the resolved sample must fit
    every belief-AABB corner inside the image with the real sensor's
    asymmetric principal point. Without a containment floor the closest
    safe sample on a tall object crops it."""
    from inspection.view.faceview import face_cam_pose
    from inspection.view.rays import cell_ray, resolve_cell
    fy, H, W = 647.0, 720, 1280
    K = np.array([[fy, 0.0, 633.0], [0.0, fy, 356.7], [0.0, 0.0, 1.0]])
    w, ik = _box_world()
    center = (0.05, -0.38, 0.1)                  # tall object: AABB to 0.2 m
    aabb = ((-0.05, -0.48, 0.0), (0.15, -0.28, 0.20))
    mn, mx = (np.asarray(v) for v in aabb)
    corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    for h, v in ((3, 1), (9, 1), (3, 2)):
        res = resolve_cell(w, ik, T_FLANGE_CAM, center, aabb, h, v,
                           fy=fy, height_px=H)
        if res is None:
            continue
        T = np.asarray(res.T_cam)
        R = np.column_stack([-T[:3, 1], -T[:3, 2], T[:3, 0]])
        pc = (corners - T[:3, 3]) @ R
        assert np.all(pc[:, 2] > 0), (h, v)
        uv = (K @ (pc / pc[:, 2:3]).T).T[:, :2]
        assert uv[:, 0].min() >= 0 and uv[:, 0].max() <= W, \
            (h, v, res.depth, uv[:, 0].min(), uv[:, 0].max())
        assert uv[:, 1].min() >= 0 and uv[:, 1].max() <= H, \
            (h, v, res.depth, uv[:, 1].min(), uv[:, 1].max())


def test_cell_map_covers_grid_and_matches_resolve_cell():
    """The map is the WHOLE grid, every time (Stage B task 1): a cell with
    no provably-safe sample is an explicit `None` entry — never a missing
    key, because the machine's request gate reads `_reach.get(cell) is
    None` and a missing key would silently mean the same as inaccessible
    without ever having been asked. Each entry is exactly what
    `resolve_cell` returns for that cell, corners and all."""
    from inspection.view.grid import H_BINS, V_ELEVATIONS
    from inspection.view.rays import box_corners, cell_map, resolve_cell
    w, ik = _box_world()
    center, fy, H = (0.05, -0.38, 0.0375), 647.0, 720
    aabb = ((-0.025, -0.455, 0.0), (0.125, -0.305, 0.075))
    corners = box_corners(_box_prim(l=0.15, w=0.15, h=0.075, yaw=0.17,
                                    center=center))
    m = cell_map(w, ik, T_FLANGE_CAM, center, aabb, fy, H, corners=corners)
    assert set(m) == {(h, v) for v in range(len(V_ELEVATIONS))
                      for h in range(H_BINS)}
    live = [c for c, r in m.items() if r is not None]
    dead = [c for c, r in m.items() if r is None]
    assert live and dead, "this fixture must exercise both outcomes"
    for cell in (live[0], dead[0]):
        ref = resolve_cell(w, ik, T_FLANGE_CAM, center, aabb, *cell, fy=fy,
                           height_px=H, corners=corners)
        got = m[cell]
        assert (ref is None) == (got is None), cell
        if ref is not None:
            assert np.isclose(got.depth, ref.depth), cell
            assert np.allclose(got.T_flange, ref.T_flange), cell
            assert np.allclose(got.q, ref.q), cell


def test_box_corners_are_the_oriented_box():
    """`box_corners` is the ONE belief-box corner helper (it used to be a
    private copy in investigation/ray_views/ray_replay.py): eight corners of
    the ORIENTED box, so a yawed box's containment bound is not paid at its
    inflated AABB."""
    from inspection.view.rays import box_corners
    p = _box_prim(l=0.2, w=0.1, h=0.15, yaw=np.deg2rad(30.0),
                  center=(0.1, -0.35, 0.075))
    c = box_corners(p)
    assert c.shape == (8, 3)
    local = (c - np.asarray(p.center)) @ p.axes          # into the box frame
    assert np.allclose(np.abs(local).max(0),
                       np.asarray(p.params["extents"]) / 2.0)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK test_rays")


if __name__ == "__main__":
    main()
