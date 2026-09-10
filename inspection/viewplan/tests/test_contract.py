"""The plan on disk must be the plan that was computed — every bit of it.

Rerun's workspace, the renders and any future executor all read the JSON,
not the objects. If `StepPlan -> JSON -> StepPlan` is lossy, the file is a
summary of a plan rather than the plan.
"""
import json

import numpy as np
import pytest

from inspection.viewplan.contract import (PlannedView, StepPlan, abstained,
                                          fingerprint, look_at_cv,
                                          plan_from_json, plan_to_json)


def _demo_plan(n=3):
    views = []
    for k in range(n):
        a = k * 2.0 * np.pi / n
        pos = np.array([0.10 + 0.14 * np.cos(a), -0.38 + 0.14 * np.sin(a), 0.08])
        T = look_at_cv(pos, np.array([0.10, -0.38, 0.06]) - pos)
        views.append(PlannedView(
            view_id=k, T_base_cam=T, standoff_m=0.10, gain=0.008 / (k + 1),
            covered=list(np.arange(3 * k, 3 * k + 7, dtype=np.int32)),
            area_m2=0.0086, mean_incidence_deg=22.4, p95_incidence_deg=51.0,
            cum_area_frac=0.3 * (k + 1)))
    return StepPlan(run="2408-cup3", step=10, variant="base",
                    version=fingerprint("base", {"standoff_m": 0.1}),
                    seed=0, table_plane=np.array([0, 0, 1, -0.0053]),
                    mesh={"path": "meshes/step_010.ply", "method": "poisson",
                          "n_vertices": np.int64(4210), "area_m2": np.float64(0.0312),
                          "watertight": False},
                    samples={"path": "samples/step_010.npz", "count": 1500},
                    params={"standoff_m": 0.10, "weight": "cosine",
                            "standoff_window_m": [0.07, 0.20]},
                    views=views,
                    metrics={"n_views": n, "coverage_area_frac": 0.968})


def test_plan_round_trips_through_json():
    p = _demo_plan()
    blob = json.dumps(plan_to_json(p))          # must need no custom encoder
    back = plan_from_json(json.loads(blob))
    assert back == p


def test_round_trip_is_byte_stable():
    p = _demo_plan()
    a = json.dumps(plan_to_json(p), sort_keys=True)
    b = json.dumps(plan_to_json(plan_from_json(json.loads(a))), sort_keys=True)
    assert a == b


def test_numpy_leaks_nowhere():
    d = plan_to_json(_demo_plan())

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            assert type(o) in (int, float, str, bool, type(None)), type(o)

    walk(d)


def test_abstain_carries_no_views():
    a = abstained("2408-cup3", 0, "base", {"standoff_m": 0.1}, 0, "cloud not ripe")
    assert a.abstain and a.n_views == 0 and a.reason
    assert plan_from_json(json.loads(json.dumps(plan_to_json(a)))) == a
    with pytest.raises(ValueError):
        StepPlan(run="r", step=0, variant="base", version="v", abstain=True,
                 views=_demo_plan(1).views)


def test_from_json_tolerates_missing_optional_keys():
    d = {"run": "r", "step": 3, "variant": "base", "version": "base/0.1+p:0",
         "views": [{"view_id": 0, "T_base_cam": np.eye(4).tolist()}]}
    p = plan_from_json(d)
    assert p.n_views == 1 and p.views[0].standoff_m == 0.0
    assert p.views[0].render == "" and p.metrics == {}


def test_n_covered_must_agree_with_covered():
    with pytest.raises(ValueError):
        PlannedView(view_id=0, T_base_cam=np.eye(4), covered=[1, 2], n_covered=5)
    v = PlannedView(view_id=0, T_base_cam=np.eye(4), covered=[1, 2, 3])
    assert v.n_covered == 3


def test_pose_gate_rejects_non_rigid():
    with pytest.raises(ValueError):
        PlannedView(view_id=0, T_base_cam=np.eye(3))
    bad = np.eye(4)
    bad[:3, :3] *= 1.01                                  # scaled -> not rigid
    with pytest.raises(ValueError):
        PlannedView(view_id=0, T_base_cam=bad)
    mirror = np.eye(4)
    mirror[0, 0] = -1.0                                  # det = -1
    with pytest.raises(ValueError):
        PlannedView(view_id=0, T_base_cam=mirror)


def test_look_at_cv_is_opencv_and_passes_the_gate():
    pos = np.array([0.22, -0.30, 0.09])
    target = np.array([0.10, -0.38, 0.06])
    T = look_at_cv(pos, target - pos)
    v = PlannedView(view_id=0, T_base_cam=T)             # gate must accept it
    b = target - pos
    b /= np.linalg.norm(b)
    assert np.allclose(v.boresight, b, atol=1e-12)       # +Z is the boresight
    assert np.allclose(v.position, pos)
    R = v.T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert R[2, 1] < 0                                   # +Y is image-DOWN
    assert abs(R[2, 0]) < 1e-9                           # +X stays horizontal


def test_look_at_cv_top_down_is_not_nan():
    T = look_at_cv([0.10, -0.38, 0.20], [0.0, 0.0, -1.0])
    assert np.isfinite(T).all()
    PlannedView(view_id=0, T_base_cam=T)


def test_look_at_cv_rejects_zero_boresight():
    with pytest.raises(ValueError):
        look_at_cv([0, 0, 0], [0, 0, 0])


def test_fingerprint_is_order_stable_and_param_sensitive():
    a = fingerprint("base", {"standoff_m": 0.10, "weight": "cosine"})
    b = fingerprint("base", {"weight": "cosine", "standoff_m": 0.10})
    c = fingerprint("base", {"standoff_m": 0.11, "weight": "cosine"})
    d = fingerprint("binary", {"standoff_m": 0.10, "weight": "cosine"})
    assert a == b and a != c and a != d
    assert a.startswith("base/0.1+p:") and len(a.split(":")[-1]) == 8


def test_poses_property_is_the_executable_stack():
    p = _demo_plan(3)
    assert p.poses.shape == (3, 4, 4)
    assert np.allclose(p.poses[1], p.views[1].T)
