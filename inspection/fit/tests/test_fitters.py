"""Each fitter, on the shape it claims to fit, must recover it.

Tolerances are stated in millimetres against a 0.1 m object with 1.5 mm
noise — loose enough that a correct fitter never flakes, tight enough that a
wrong axis or a doubled radius fails.
"""
import numpy as np
import pytest

from inspection.fit import bank, common
from inspection.fit.tests import synth


# --- planar ---------------------------------------------------------------
def test_planar_recovers_the_faces_of_a_box():
    P = synth.box(n=7200, extents=(0.06, 0.05, 0.09), seed=1)
    d = bank.fit(P, ["planar"])["planar"]
    assert not d.abstain, d.reason
    assert all(p.kind == "plane_patch" for p in d.primitives)
    # the six face normals, up to sign
    want = np.eye(3)
    got = np.array([p.axes[:, 2] for p in d.primitives])
    for ax in want:
        assert np.abs(got @ ax).max() > np.cos(np.deg2rad(10)), \
            f"no patch normal within 10 deg of {ax}"
    assert 3 <= d.n_primitives <= 8          # 6 faces, at most one split
    assert d.residual_mm < 3.0
    assert d.coverage > 0.5


def test_planar_refuses_to_tile_a_cylinder():
    """The whole reason the guard exists."""
    P = synth.cylinder(n=6000, radius=0.035, length=0.09, seed=1)
    d = bank.fit(P, ["planar"])["planar"]
    assert d.abstain, f"planar tiled a cylinder with {d.n_primitives} patches"
    assert "curvature guard" in d.reason


def test_planar_labels_are_content_derived_not_extraction_order():
    P = synth.box(n=7200, seed=2)
    a = bank.fit(P, ["planar"])["planar"]
    b = bank.fit(P[::-1].copy(), ["planar"])["planar"]   # same cloud, reordered
    assert not a.abstain and not b.abstain
    # a label names a direction; reversing the point order must not rename it
    assert {p.label for p in a.primitives} & {p.label for p in b.primitives}
    for p in a.primitives:
        assert p.label.startswith("face_")


# --- cyl ------------------------------------------------------------------
@pytest.mark.parametrize("axis", [(0, 0, 1), (0.3, 0.1, 0.95), (1, 0, 0)])
def test_cyl_recovers_radius_length_and_axis(axis):
    P = synth.cylinder(n=6000, radius=0.035, length=0.09, axis=axis, seed=3)
    d = bank.fit(P, ["cyl"])["cyl"]
    assert not d.abstain, d.reason
    p = max(d.primitives, key=lambda q: q.support)
    assert p.kind == "cylinder"
    assert abs(p.params["radius"] - 0.035) < 0.003, p.params
    assert abs(p.params["length"] - 0.09) < 0.006, p.params
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    assert abs(float(p.axes[:, 2] @ a)) > np.cos(np.deg2rad(5))
    assert np.linalg.norm(p.center - np.array([0.1, -0.35, 0.06])) < 0.005
    assert p.residual_mm < 3.0


def test_cyl_is_deterministic_under_a_fixed_seed():
    P = synth.cylinder(n=4000, seed=4)
    a = bank.fit(P, ["cyl"], seed=7)["cyl"]
    b = bank.fit(P, ["cyl"], seed=7)["cyl"]
    assert a.to_json() == b.to_json()


def test_cyl_names_an_upright_cylinder_body():
    P = synth.cylinder(n=5000, axis=(0, 0, 1), seed=5)
    d = bank.fit(P, ["cyl"])["cyl"]
    assert "body" in [p.label for p in d.primitives]


def test_cyl_refuses_a_flat_wall():
    """A single box face is a cylinder of infinite radius; do not fit it."""
    P = synth.box(n=6000, seed=6)
    P = P[np.abs(P[:, 2] - P[:, 2].max()) < 0.004]      # one face only
    d = bank.fit(P, ["cyl"])["cyl"]
    assert d.abstain, [(p.label, p.params) for p in d.primitives]


def test_cyl_refuses_a_narrow_arc():
    """60 deg of a cylinder is a curved strip; the azimuth gate must reject."""
    P = synth.cylinder(n=3000, arc_deg=60.0, seed=7)
    d = bank.fit(P, ["cyl"])["cyl"]
    assert d.abstain and "azimuth" in d.reason


# --- ems ------------------------------------------------------------------
def test_ems_recovers_an_ellipsoid():
    semi = (0.045, 0.035, 0.030)
    P = synth.ellipsoid(n=6000, semi=semi, seed=8)
    d = bank.fit(P, ["ems"])["ems"]
    assert not d.abstain, d.reason
    p = d.primitives[0]
    assert p.kind == "superquadric" and p.label == "hull"
    eps = np.asarray(p.params["eps"], float)
    assert np.all(np.abs(eps - 1.0) < 0.25), eps        # an ellipsoid is (1,1)
    # scale is recovered up to a permutation of the axes
    assert np.allclose(np.sort(p.params["scale"]), np.sort(semi), atol=0.004)
    assert np.linalg.norm(p.center - np.array([0.1, -0.35, 0.06])) < 0.004


def test_ems_recovers_a_box_as_a_sharp_superquadric():
    P = synth.box(n=6000, extents=(0.06, 0.05, 0.09), seed=9)
    d = bank.fit(P, ["ems"])["ems"]
    assert not d.abstain, d.reason
    p = d.primitives[0]
    assert np.asarray(p.params["eps"], float).max() < 0.6      # sharp, not round
    assert np.allclose(np.sort(p.params["scale"]),
                       np.sort([0.03, 0.025, 0.045]), atol=0.006)


def test_ems_on_an_open_tube_gets_the_cross_section_and_not_the_length():
    """Documents a real limit, so nobody trusts an EMS height on a mug.

    A cup body is an open tube: no caps, so nothing pins the axial extent.
    Measured on synthetic tubes of radius 35 mm and length 90 mm, EMS
    recovers the radius to better than 0.4 mm every time, while the axial
    semi-axis lands anywhere from 49 mm to 156 mm depending on N and on
    `adaptive_upper_bound`. With the bound on (our default) it can also
    raise `ValueError: Initial guess is outside of provided bounds` — an
    upstream bug, since the bound is recomputed from the current estimate
    inside the EM loop and the estimate can end up outside its own new box.
    We abstain rather than pretend.
    """
    P = synth.cylinder(n=6000, radius=0.035, length=0.09, seed=12)
    d = bank.fit(P, ["ems"])["ems"]
    if d.abstain:
        assert "solver failed" in d.reason
        return
    scale = np.sort(np.asarray(d.primitives[0].params["scale"], float))
    assert abs(scale[0] - 0.035) < 0.002 and abs(scale[1] - 0.035) < 0.002


def test_ems_is_deterministic():
    P = synth.ellipsoid(n=3000, seed=10)
    a = bank.fit(P, ["ems"], seed=1)["ems"]
    b = bank.fit(P, ["ems"], seed=999)["ems"]     # EMS has no RNG at all
    assert a.to_json() == b.to_json()


# --- the maturity gate ----------------------------------------------------
@pytest.mark.parametrize("method", ["planar", "cyl", "ems"])
def test_every_fitter_abstains_on_a_50_point_cloud(method):
    d = bank.fit(synth.stub(50), [method])[method]
    assert d.abstain
    assert "not ripe" in d.reason and "50 points" in d.reason
    assert d.primitives == [] and d.coverage == 0.0


@pytest.mark.parametrize("method", ["planar", "cyl", "ems"])
def test_every_fitter_abstains_on_a_degenerate_line(method):
    P = np.zeros((900, 3))
    P[:, 0] = np.linspace(0, 0.1, 900)
    P += np.random.default_rng(0).normal(0, 0.0005, P.shape)
    d = bank.fit(P, [method])[method]
    assert d.abstain, d.reason


def test_maturity_gate_passes_a_real_sized_cloud():
    assert common.maturity_reason(synth.cylinder(n=900, seed=11)) == ""
