import numpy as np
from inspection.cell.world import RobotCell
from inspection.motion.ik import UR5eIK
from inspection.view.viewsphere import T_FLANGE_CAM


def _box_world():
    w = RobotCell()
    w.set_object("object", [0.15, 0.15, 0.075],
                 [0.05, -0.38, 0.0375, 0, 0, 0.17])
    return w, UR5eIK()


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


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK test_rays")


if __name__ == "__main__":
    main()
