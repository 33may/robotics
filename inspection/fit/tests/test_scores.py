"""Scoring is computed outside the fitters, so it must be right on its own.

The tests below check the distance functions against shapes whose answers
can be worked out by hand, and check the one property the whole comparison
rests on: two methods scored on the same cloud are scored by the same rule.
"""
import numpy as np

from inspection.fit import bank
from inspection.fit.contract import (Decomposition, Primitive, frame_from_axis,
                                     pose_from)
from inspection.fit.scores import COVER_TOL_M, score, surface_distance
from inspection.fit.tests import synth


def test_cylinder_distance_is_the_lateral_tube_not_a_capped_solid():
    p = Primitive("cylinder", frame_from_axis([0, 0, 1], [0, 0, 0]),
                  {"radius": 0.03, "length": 0.10}, "body")
    P = np.array([[0.03, 0, 0],        # on the wall
                  [0.04, 0, 0],        # 10 mm outside the wall
                  [0.0, 0, 0],         # on the axis, mid-height: 30 mm away
                  [0.03, 0, 0.08]])    # 30 mm above the top rim
    d = surface_distance(P, p)
    assert np.allclose(d, [0.0, 0.01, 0.03, 0.03], atol=1e-9)


def test_plane_patch_distance_is_bounded_by_its_extents():
    p = Primitive("plane_patch", pose_from(np.eye(3), [0, 0, 0]),
                  {"extents": [0.04, 0.02]}, "face_pz")
    P = np.array([[0, 0, 0.001],       # 1 mm above the middle
                  [0.10, 0, 0]])       # 80 mm off the edge, in-plane
    d = surface_distance(P, p)
    assert np.isclose(d[0], 0.001)
    assert np.isclose(d[1], 0.10 - 0.02)


def test_box_distance_handles_inside_and_outside():
    p = Primitive("box", pose_from(np.eye(3), [0, 0, 0]),
                  {"extents": [0.02, 0.02, 0.02]}, "b")
    d = surface_distance(np.array([[0.0, 0, 0], [0.02, 0, 0]]), p)
    assert np.isclose(d[0], 0.01)      # centre: 10 mm to the nearest face
    assert np.isclose(d[1], 0.01)      # 10 mm outside


def test_superquadric_distance_vanishes_on_an_ellipsoid_surface():
    semi = np.array([0.045, 0.035, 0.030])
    p = Primitive("superquadric", pose_from(np.eye(3), [0, 0, 0]),
                  {"scale": semi.tolist(), "eps": [1.0, 1.0]}, "hull")
    v = np.random.default_rng(0).normal(size=(500, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    assert surface_distance(v * semi, p).max() < 1e-6


def test_score_overwrites_whatever_the_fitter_claimed():
    P = synth.cylinder(n=2000, radius=0.035, length=0.09, seed=0)
    d = Decomposition("bogus", "bogus/1",
                      primitives=[Primitive(
                          "cylinder", frame_from_axis([0, 0, 1], [0.1, -0.35, 0.06]),
                          {"radius": 0.035, "length": 0.09}, "body",
                          inliers=[0, 1, 2], residual_mm=999.0, support=3)])
    score(P, d)
    assert d.primitives[0].support > 1500        # not the 3 it claimed
    assert d.primitives[0].residual_mm < 5.0     # not the 999 it claimed
    assert d.coverage > 0.9


def test_no_double_counting_between_overlapping_primitives():
    P = synth.cylinder(n=2000, seed=1)
    twin = [Primitive("cylinder", frame_from_axis([0, 0, 1], [0.1, -0.35, 0.06]),
                      {"radius": 0.035, "length": 0.09}, f"c{i}") for i in range(2)]
    d = Decomposition("bogus", "bogus/1", primitives=twin)
    score(P, d)
    assert d.coverage <= 1.0
    assert sum(p.support for p in d.primitives) <= len(P)


def test_abstain_scores_to_zero_coverage():
    P = synth.cylinder(n=1000, seed=2)
    d = bank.fit(synth.stub(50), ["cyl"])["cyl"]
    assert d.coverage == 0.0 and d.residual_mm == 0.0


def test_methods_are_scored_by_the_same_rule():
    """The point of scoring outside the fitters: one cloud, one yardstick."""
    P = synth.cylinder(n=5000, radius=0.035, length=0.09, seed=3)
    out = bank.fit(P, ["cyl", "ems"])
    for m, d in out.items():
        if d.abstain:
            continue
        # coverage is a fraction of the SAME cloud, by the SAME tolerance
        claimed = np.zeros(len(P), bool)
        for p in d.primitives:
            claimed[p.inliers] = True
        assert np.isclose(d.coverage, claimed.mean())
        assert all(surface_distance(P[p.inliers], p).max() <= COVER_TOL_M + 1e-12
                   for p in d.primitives if p.inliers)
