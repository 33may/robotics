"""The mesh is the coverage denominator — it may not invent or lose surface.

A mesher that closes a bubble over the unseen side makes a bad plan look
good (phantom area the planner never has to cover); one that shreds the
surface makes a good plan look bad. Both failures are silent in the plan
JSON, so they get caught here.
"""
import numpy as np
import pytest

from inspection.viewplan import mesh as meshmod
from inspection.viewplan.tests import synth


def test_sphere_area_is_recovered():
    r = 0.04
    m, info = meshmod.reconstruct(synth.sphere_cloud(r=r, n=4000))
    truth = 4 * np.pi * r ** 2
    assert abs(info["area_m2"] - truth) / truth < 0.15, info
    assert info["method"] == "poisson"
    assert info["n_triangles"] > 1000


def test_cylinder_area_is_recovered():
    r, h = 0.03, 0.10
    m, info = meshmod.reconstruct(synth.cylinder_cloud(r=r, h=h, n=6000))
    truth = 2 * np.pi * r * h + 2 * np.pi * r ** 2
    assert abs(info["area_m2"] - truth) / truth < 0.15, info


def test_poisson_does_not_balloon_past_the_cloud():
    """The failure this recipe exists to prevent, as a test.

    Untrimmed Poisson on a partial cloud grows a closed skin well outside the
    data (measured [95,132,132] mm for a [93,120,90] mm cloud). The 6 mm
    distance trim must keep the mesh inside the cloud's own box plus a small
    margin.
    """
    P = synth.sphere_cloud(r=0.04, n=3000)
    P = P[P[:, 0] < np.median(P[:, 0])]              # a half-seen sphere
    m, info = meshmod.reconstruct(P)
    V = np.asarray(m.vertices)
    assert (V.min(0) >= P.min(0) - 0.010).all(), (V.min(0), P.min(0))
    assert (V.max(0) <= P.max(0) + 0.010).all(), (V.max(0), P.max(0))


def test_reconstruct_is_deterministic():
    P = synth.sphere_cloud()
    a, ia = meshmod.reconstruct(P, seed=0)
    b, ib = meshmod.reconstruct(P, seed=0)
    assert ia == ib
    assert np.array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
    assert np.array_equal(np.asarray(a.triangles), np.asarray(b.triangles))


def test_alpha_mesher_is_a_real_alternative():
    """`altmesh` must be a different mechanism, not a different spelling."""
    P = synth.sphere_cloud()
    pm, pi = meshmod.reconstruct(P, method="poisson")
    am, ai = meshmod.reconstruct(P, method="alpha")
    assert ai["method"] == "alpha" and pi["method"] == "poisson"
    assert ai["n_triangles"] != pi["n_triangles"]
    assert abs(ai["area_m2"] - pi["area_m2"]) / pi["area_m2"] < 0.35


def test_maturity_gate_refuses_a_stub():
    assert meshmod.maturity_reason(synth.stub(150))
    assert not meshmod.maturity_reason(synth.sphere_cloud())


def test_pathological_cloud_never_crashes():
    """A 60-point line has no surface. The ladder must end in a mesh or a
    one-sentence ValueError — never a segfault, never a numpy exception."""
    try:
        m, info = meshmod.reconstruct(synth.line(60))
    except ValueError as exc:
        assert str(exc)
    else:
        assert info["method"] in ("poisson", "alpha", "hull")
        assert len(m.triangles) >= 4


def test_reconstruct_refuses_garbage_shapes():
    with pytest.raises(ValueError):
        meshmod.reconstruct(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        meshmod.reconstruct(np.zeros((100, 2)))


def test_table_plane_finds_the_table_under_a_dome():
    for truth in (0.0, 0.02, -0.01):
        P = synth.half_sphere_on_table(table_z=truth)
        a, b, c, d = meshmod.table_plane(P)
        assert (a, b, c) == (0.0, 0.0, 1.0)
        assert abs(-d - truth) < 0.002, (-d, truth)


def test_table_plane_signs_height_upward():
    P = synth.half_sphere_on_table(table_z=0.0)
    plane = meshmod.table_plane(P)
    h = meshmod.height_above(plane, np.array([[0.1, -0.38, 0.05],
                                              [0.1, -0.38, -0.05]]))
    assert h[0] > 0 and h[1] < 0
    assert np.isclose(h[0] - h[1], 0.10)


def test_normals_face_the_camera_that_saw_them():
    """The orientation rule, on a case where the answer is not arguable."""
    P = synth.sphere_cloud(r=0.04, n=3000)
    c = np.asarray(synth.CENTER)
    cams = c + np.array([[0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0],
                         [0, -0.3, 0], [0, 0, 0.3], [0, 0, -0.3]])
    N, rule = meshmod.oriented_normals(P, cams)
    assert rule == "nearest_camera"
    radial = (P - c) / np.linalg.norm(P - c, axis=1, keepdims=True)
    # cameras ring the sphere, so "toward the nearest camera" == outward
    assert (np.einsum("ij,ij->i", N, radial) > 0).mean() > 0.97


def test_normals_fall_back_to_the_centroid_rule_without_cameras():
    P = synth.sphere_cloud(r=0.04, n=2000)
    N, rule = meshmod.oriented_normals(P, None)
    assert rule == "centroid_outward"
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0)


def test_save_load_round_trip(tmp_path):
    m, info = meshmod.reconstruct(synth.sphere_cloud())
    p = tmp_path / "sub" / "m.ply"
    meshmod.save(m, p)
    assert p.exists()
    back = meshmod.load(p)
    assert len(back.triangles) == len(m.triangles)
    assert np.allclose(meshmod.mesh_area(back), meshmod.mesh_area(m), rtol=1e-6)
