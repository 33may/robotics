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
