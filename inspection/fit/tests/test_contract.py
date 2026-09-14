"""The contract must survive a round trip through JSON without losing a bit.

Everything downstream — the Rerun workspace, the best-view derivation — reads
these files, not the objects. If `Decomposition -> JSON -> Decomposition` is
lossy, the thing on disk is not the thing that was fitted.
"""
import json

import numpy as np
import pytest

from inspection.fit.contract import (KINDS, Decomposition, Primitive,
                                     abstained, decomposition_from_json,
                                     decomposition_to_json, frame_from_axis,
                                     pose_from)


def _one_of_each():
    T = pose_from(np.eye(3), [0.1, -0.35, 0.06])
    return [
        Primitive("box", T, {"extents": [0.06, 0.05, 0.09]}, "face_px",
                  inliers=[1, 2, 3], residual_mm=1.5, support=3),
        Primitive("cylinder", frame_from_axis([0, 0, 1], [0.1, -0.35, 0.06]),
                  {"radius": 0.035, "length": 0.09}, "body",
                  inliers=[4, 5], residual_mm=2.4, support=2),
        Primitive("plane_patch", T, {"extents": [0.04, 0.03]}, "face_pz",
                  inliers=[6], residual_mm=0.8, support=1),
        Primitive("superquadric", T, {"scale": [0.04, 0.03, 0.05],
                                      "eps": [0.25, 0.9]}, "hull"),
        Primitive("mesh", T, {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                              "faces": [[0, 1, 2]]}, "shell"),
    ]


def test_every_kind_round_trips_through_json():
    d = Decomposition("demo", "demo/1.0+p:abcdef01", primitives=_one_of_each(),
                      coverage=0.83, residual_mm=2.1, reason="ok")
    blob = json.dumps(decomposition_to_json(d))       # must need no encoder
    back = decomposition_from_json(json.loads(blob))
    assert back == d
    assert {p.kind for p in back.primitives} == set(KINDS)
    assert back.n_primitives == 5


def test_numpy_params_are_plain_json():
    p = Primitive("cylinder", np.eye(4),
                  {"radius": np.float64(0.035), "length": np.float32(0.09)},
                  "body", inliers=np.array([1, 2]))
    blob = json.dumps(p.to_json())
    assert "radius" in blob
    assert Primitive.from_json(json.loads(blob)).inliers == [1, 2]


def test_abstain_round_trips_and_carries_its_reason():
    a = abstained("cyl", "cyl/1.0", "cloud not ripe: 50 points < 250")
    back = decomposition_from_json(json.loads(json.dumps(a.to_json())))
    assert back.abstain and back.primitives == [] and "not ripe" in back.reason


def test_closed_kind_vocabulary_is_enforced():
    with pytest.raises(ValueError):
        Primitive("cone", np.eye(4), {}, "nope")


def test_missing_params_are_rejected_at_construction():
    with pytest.raises(ValueError):
        Primitive("cylinder", np.eye(4), {"radius": 0.03}, "body")


def test_abstaining_decomposition_cannot_carry_primitives():
    with pytest.raises(ValueError):
        Decomposition("x", "x/1", abstain=True, primitives=_one_of_each()[:1])


def test_pose_is_orthonormal_and_right_handed():
    T = frame_from_axis([0.3, 0.4, 0.86], [0, 0, 0])
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], np.array([0.3, 0.4, 0.86])
                       / np.linalg.norm([0.3, 0.4, 0.86]))


def test_frame_from_axis_is_stable_for_the_same_axis():
    a = frame_from_axis([0, 0, 1], [1, 2, 3])
    b = frame_from_axis([0, 0, 1], [1, 2, 3])
    assert np.array_equal(a, b)
