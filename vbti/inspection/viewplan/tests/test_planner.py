"""The planner, checked where the right answer is known analytically.

A sphere's ideal views sit on a concentric shell looking at the centre; a
standing cylinder wants a ring around the side and one view down the cap;
a camera cannot see through a plate; a camera may not stand under the table.
Each of those is a test, so "orthogonal views" is a measured property rather
than a description of the intent.
"""
import numpy as np
import pytest

from inspection.viewplan import mesh as meshmod
from inspection.viewplan import planner
from inspection.viewplan.contract import look_at_cv, stable_json
from inspection.viewplan.tests import synth

FLOATING = [0.0, 0.0, 1.0, 0.10]        # table well below a mid-air object


def _plan(P, variant="base", **kw):
    kw.setdefault("intr", synth.INTR)
    kw.setdefault("cams", np.empty((0, 3)))
    return planner.plan_step(P, "synth", 0, variant, seed=0, **kw)


# --- variants --------------------------------------------------------------
def test_variants_are_registered_and_versioned():
    vs = planner.variants()
    assert vs == sorted(vs) and "base" in vs and "binary" in vs
    assert set(vs) == {"altmesh", "base", "binary", "continuity", "cos2",
                       "cos75"}
    versions = {v: planner.version_of(v) for v in vs}
    assert len(set(versions.values())) == len(vs), versions
    assert planner.VARIANTS["binary"]["weight"] == "binary"
    assert planner.VARIANTS["cos75"]["incidence_cutoff_deg"] == 75.0
    assert planner.VARIANTS["altmesh"]["mesher"] == "alpha"
    assert planner.VARIANTS["continuity"]["continuity_bonus"] > 0
    # only `continuity` is chained; the rest must stay order-independent
    assert [v for v in vs if planner.VARIANTS[v]["continuity_bonus"] > 0] \
        == ["continuity"]


def test_params_for_hands_out_copies():
    a = planner.params_for("base")
    a["standoff_m"] = 99.0
    assert planner.params_for("base")["standoff_m"] != 99.0
    with pytest.raises(KeyError):
        planner.params_for("nope")


# --- the sphere: the analytic case ----------------------------------------
def test_sphere_views_are_orthogonal_and_on_the_standoff_shell():
    r, standoff = 0.04, planner.VARIANTS["base"]["standoff_m"]
    plan, m, arr = _plan(synth.sphere_cloud(r=r, n=4000), table_plane=FLOATING)
    assert not plan.abstain, plan.reason
    c = np.asarray(synth.CENTER)
    for v in plan.views:
        d = np.linalg.norm(v.position - c)
        assert abs(d - (r + standoff)) < 0.004, (v.view_id, d)
        to_c = (c - v.position) / d
        err = np.degrees(np.arccos(np.clip(float(to_c @ v.boresight), -1, 1)))
        assert err < 8.0, (v.view_id, err)


def test_sphere_is_covered_by_at_most_twelve_views():
    """The success bar, proven where the answer is known."""
    plan, m, arr = _plan(synth.sphere_cloud(n=4000), table_plane=FLOATING)
    assert plan.metrics["n_views"] <= 12
    assert plan.metrics["coverage_area_frac"] >= 0.95, plan.metrics
    assert plan.metrics["mean_best_incidence_deg"] < 40.0


def test_sphere_views_spread_over_the_shell():
    """Near-uniform, not twelve cameras clustered on one hemisphere."""
    plan, m, arr = _plan(synth.sphere_cloud(n=4000), table_plane=FLOATING)
    b = np.stack([v.boresight for v in plan.views])
    assert np.linalg.norm(b.mean(0)) < 0.30, b.mean(0)   # no net direction
    cos = b @ b.T
    np.fill_diagonal(cos, -1.0)
    assert cos.max() < 0.95                              # no duplicate views


# --- the cylinder ----------------------------------------------------------
def test_cylinder_side_views_are_horizontal_and_the_cap_earns_one():
    plan, m, arr = _plan(synth.cylinder_cloud(r=0.03, h=0.10, n=6000),
                         table_plane=FLOATING)
    bz = np.array([v.boresight[2] for v in plan.views])
    # the side wall (most of the area) wants boresights perpendicular to +Z
    assert (np.abs(bz) < 0.2).sum() >= 4, bz
    # and the top cap wants one looking down
    assert bz.max() > 0.7 or bz.min() < -0.7, bz


# --- the table guard -------------------------------------------------------
def test_no_view_stands_under_the_table():
    P = synth.half_sphere_on_table(table_z=0.0)
    plan, m, arr = _plan(P)
    assert not plan.abstain, plan.reason
    plane = plan.table_plane
    for v in plan.views:
        h = float(meshmod.height_above(plane, v.position[None, :])[0])
        assert h > planner.TABLE_CLEARANCE_M - 1e-9, (v.view_id, h)
    assert plan.metrics["n_culled_table"] > 0


def test_the_sight_line_never_dips_under_the_table():
    """The reason the boresight test is redundant, verified rather than argued."""
    P = synth.half_sphere_on_table(table_z=0.0)
    plan, m, arr = _plan(P)
    plane = plan.table_plane
    pts = arr["points"]
    assert (meshmod.height_above(plane, pts) > 0).all()
    for v in plan.views:
        tgt = pts[v.covered]
        t = np.linspace(0.0, 1.0, 16)[None, :, None]
        seg = v.position[None, None, :] * (1 - t) + tgt[:, None, :] * t
        assert (meshmod.height_above(plane, seg.reshape(-1, 3)) > -1e-9).all()


def test_candidate_cull_counts_land_in_the_metrics():
    P = synth.half_sphere_on_table(table_z=0.0)
    plan, m, arr = _plan(P)
    mt = plan.metrics
    assert mt["n_candidates"] <= planner.VARIANTS["base"]["n_candidates"]
    assert (mt["n_culled_table"] + mt["n_culled_clearance"]
            + mt["n_anchors_legal"]) == mt["n_samples"]
    assert "n_up_after_table" in mt


# --- the clearance guard ---------------------------------------------------
def test_a_camera_may_not_stand_inside_the_object():
    """The cavity trap: the ideal view of an inner wall is inside the cup.

    Measured on the shipped grid before this guard existed: 2408-cup3 step
    011 view 9 sat 0.6 mm from the accumulated cloud, inside its bounding
    box, and claimed 237 samples through the open side of the mesh.
    """
    P, N = synth.inner_wall_patches(r=0.05, n=64)
    params = planner.params_for("base")
    T, anchors, info = planner.candidate_views(P, N, params, FLOATING, cloud=P)
    assert info["n_culled_clearance"] == 64 and len(T) == 0

    # and the guard is what culls them: without a cloud to measure against,
    # every one of these poses is still offered
    T0, _a0, info0 = planner.candidate_views(P, N, params, FLOATING)
    assert info0["n_culled_clearance"] == 0 and len(T0) == 64


def test_no_selected_camera_is_inside_the_cup():
    P = synth.cup_cloud()
    plan, m, arr = _plan(P)
    assert not plan.abstain, plan.reason
    clear = plan.params["clearance_m"]
    d = planner.clearance_to(P, np.stack([v.position for v in plan.views]))
    assert (d >= clear).all(), d.min()
    assert plan.metrics["min_cam_clearance_m"] == pytest.approx(d.min())


def test_the_clearance_guard_culls_the_inside_and_keeps_the_outside():
    """It must be a scalpel, not a hammer: same wall, both facing directions."""
    Pi, Ni = synth.inner_wall_patches(r=0.05, n=32)
    Po, No = Pi.copy(), -Ni                       # same ring, looking outward
    P = np.vstack([Pi, Po])
    N = np.vstack([Ni, No])
    params = planner.params_for("base")
    T, anchors, info = planner.candidate_views(P, N, params, FLOATING, cloud=Pi)
    assert info["n_culled_clearance"] == 32       # exactly the inward half
    assert info["n_anchors_legal"] == 32
    d = planner.clearance_to(Pi, T[:, :3, 3])
    assert (d >= params["clearance_m"]).all()


# --- occlusion -------------------------------------------------------------
def _plate_rig(gap=0.03, side=0.08):
    """Two explicit quads `gap` apart, BOTH facing -X, and samples on each.

    Built by hand rather than meshed: a reconstruction would fuse or drop one
    plate (the ladder keeps the largest connected component), and the point
    here is a clean occluder. Both normals face the camera, so the incidence
    gate passes on both plates and only the raycast can reject the far one —
    which is what makes this a test of occlusion and not of incidence.
    """
    import open3d as o3d
    c = np.asarray(synth.CENTER)
    V, F, S, N = [], [], [], []
    for k, s in enumerate((-0.5, 0.5)):
        x = c[0] + s * gap
        q = np.array([[x, c[1] - side / 2, c[2] - side / 2],
                      [x, c[1] + side / 2, c[2] - side / 2],
                      [x, c[1] + side / 2, c[2] + side / 2],
                      [x, c[1] - side / 2, c[2] + side / 2]])
        V.append(q)
        F += [[4 * k, 4 * k + 1, 4 * k + 2], [4 * k, 4 * k + 2, 4 * k + 3]]
        g = np.linspace(-side / 2 * 0.8, side / 2 * 0.8, 12)
        yy, zz = np.meshgrid(g, g)
        S.append(np.stack([np.full(yy.size, x), c[1] + yy.ravel(),
                           c[2] + zz.ravel()], 1))
        N.append(np.tile([-1.0, 0.0, 0.0], (yy.size, 1)))
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.vstack(V)),
        o3d.utility.Vector3iVector(np.asarray(F, dtype=np.int32)))
    m.compute_vertex_normals()
    return m, np.vstack(S), np.vstack(N), c


def test_a_plate_hides_the_plate_behind_it():
    """Raycast proof: from one side, the far plate is not measurable."""
    m, S, N, c = _plate_rig()
    pos = c - np.array([0.115, 0.0, 0.0])                 # camera on -X side
    T = look_at_cv(pos, np.array([1.0, 0.0, 0.0]))[None, ...]
    cover, cosv, gate = planner.visibility(m, T, S, N, synth.INTR,
                                           planner.params_for("base"))
    near = S[:, 0] < c[0]
    far = ~near
    assert near.sum() and far.sum()
    assert cover[0][near].mean() > 0.95, "the near plate must be visible"
    assert cover[0][far].mean() == 0.0, cover[0][far].mean()
    # and the reason recorded must be occlusion, not some other gate
    assert (gate[0][far] == planner.GATE_OCCLUSION).all()


def test_the_far_plate_is_visible_once_the_near_one_is_gone():
    """The control for the test above: same geometry minus the occluder."""
    m, S, N, c = _plate_rig()
    m.remove_triangles_by_mask(np.array([True, True, False, False]))
    m.remove_unreferenced_vertices()
    pos = c - np.array([0.115, 0.0, 0.0])
    T = look_at_cv(pos, np.array([1.0, 0.0, 0.0]))[None, ...]
    cover, cosv, gate = planner.visibility(m, T, S, N, synth.INTR,
                                           planner.params_for("base"))
    assert cover[0][S[:, 0] > c[0]].mean() > 0.95


# --- the greedy ------------------------------------------------------------
def _matrix():
    """Hand-built 4 views x 4 patches whose two answers are known.

    View 3 sees all four patches obliquely (cos 0.6 ~ 53 deg — inside the
    gate, so plain set cover is *done* after picking it). Views 1 and 2 see
    two patches each head-on; view 0 is a strictly worse version of view 1.

    This is the whole Q2 argument in four rows: binary greedy stops at one
    view and calls the surface covered at 53 deg; the cos-weighted greedy
    takes the same view first and then keeps going, because 53 deg is not
    as good as 8 deg and the objective can see the difference.
    """
    cover = np.array([[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [1, 1, 1, 1]], dtype=bool)
    cosv = np.array([[0.60, 0.60, 0.00, 0.00],
                     [0.99, 0.99, 0.00, 0.00],
                     [0.00, 0.00, 0.99, 0.99],
                     [0.60, 0.60, 0.60, 0.60]], dtype=float)
    return cover, cosv


def test_weighted_greedy_ranks_by_orthogonality():
    cover, cosv = _matrix()
    p = dict(planner.params_for("base"), max_views=4, min_gain_frac=0.0)
    picks, term = planner.greedy(cover, cosv, 1.0, p)
    assert [x["view_idx"] for x in picks] == [3, 1, 2], picks
    assert picks[0]["gain"] == pytest.approx(2.4)
    assert picks[1]["gain"] == pytest.approx(0.78)
    assert np.allclose(picks[-1]["best_after"], 0.99)     # everything head-on


def test_binary_greedy_ranks_by_plain_coverage():
    cover, cosv = _matrix()
    p = dict(planner.params_for("binary"), max_views=4, min_gain_frac=0.0)
    picks, term = planner.greedy(cover, cosv, 1.0, p)
    assert [x["view_idx"] for x in picks] == [3], picks   # covered, and done
    assert picks[0]["gain"] == pytest.approx(4.0)


def test_the_two_objectives_disagree_about_quality_not_coverage():
    """Same patches covered, worse angles — the ablation in one assertion."""
    cover, cosv = _matrix()
    out = {}
    for name in ("base", "binary"):
        p = dict(planner.params_for(name), max_views=4, min_gain_frac=0.0)
        picks, _ = planner.greedy(cover, cosv, 1.0, p)
        sel = [x["view_idx"] for x in picks]
        out[name] = (cover[sel].any(0).mean(), cosv[sel].max(0).mean())
    assert out["base"][0] == out["binary"][0] == 1.0      # equal coverage
    assert out["base"][1] > out["binary"][1]              # better orthogonality


def test_greedy_ties_break_on_the_lowest_index():
    cover = np.ones((3, 2), dtype=bool)
    cosv = np.full((3, 2), 0.8)
    p = dict(planner.params_for("base"), max_views=1, min_gain_frac=0.0)
    picks, _ = planner.greedy(cover, cosv, 1.0, p)
    assert picks[0]["view_idx"] == 0


def test_greedy_reports_which_rule_stopped_it():
    cover, cosv = _matrix()
    p = dict(planner.params_for("base"), max_views=2, min_gain_frac=0.0)
    _, term = planner.greedy(cover, cosv, 1.0, p)
    assert term == "max_views"
    p = dict(planner.params_for("base"), max_views=12, min_gain_frac=0.4)
    picks, term = planner.greedy(cover, cosv, 1.0, p)
    assert term == "gain" and len(picks) < 12


def test_quality_rejects_an_unknown_weight():
    cover, cosv = _matrix()
    assert np.array_equal(planner.quality(cover, cosv, "cosine2"), cosv ** 2)
    with pytest.raises(ValueError):
        planner.quality(cover, cosv, "sqrt")


# --- composition -----------------------------------------------------------
def test_plan_step_is_deterministic():
    P = synth.sphere_cloud(n=3000)
    a, _, aa = _plan(P, table_plane=FLOATING)
    b, _, bb = _plan(P, table_plane=FLOATING)
    assert stable_json(a) == stable_json(b)
    for k in aa:
        assert np.array_equal(aa[k], bb[k]), k


def test_plan_step_abstains_instead_of_raising():
    plan, m, arr = _plan(synth.stub(120))
    assert plan.abstain and "not ripe" in plan.reason and arr == {}
    assert plan.version.startswith("base/")


def test_arrays_match_the_npz_contract():
    plan, m, arr = _plan(synth.sphere_cloud(n=3000), table_plane=FLOATING)
    n = plan.samples["count"]
    assert set(arr) == {"points", "normals", "area", "best_cos", "covered_by"}
    assert arr["points"].shape == (n, 3) and arr["normals"].shape == (n, 3)
    assert arr["area"].shape == (n,) and np.allclose(arr["area"],
                                                     plan.samples["area_per_sample_m2"])
    assert arr["covered_by"].dtype == np.int32
    covered = arr["best_cos"] > 0
    assert (arr["covered_by"][covered] >= 0).all()
    assert (arr["covered_by"][~covered] == -1).all()
    assert arr["covered_by"].max() < plan.n_views
    assert np.allclose(np.linalg.norm(arr["normals"], axis=1), 1.0)


def test_covered_ids_agree_with_the_stored_incidence():
    """Every claimed patch must actually be inside the gate it claims."""
    plan, m, arr = _plan(synth.sphere_cloud(n=3000), table_plane=FLOATING)
    cut = np.deg2rad(plan.params["incidence_cutoff_deg"])
    lo, hi = plan.params["standoff_window_m"]
    P, N = arr["points"], arr["normals"]
    for v in plan.views:
        ids = np.asarray(v.covered)
        vec = v.position[None, :] - P[ids]
        d = np.linalg.norm(vec, axis=1)
        cos = np.einsum("ij,ij->i", vec / d[:, None], N[ids])
        assert (d >= lo - 1e-9).all() and (d <= hi + 1e-9).all()
        assert (cos >= np.cos(cut) - 1e-9).all()
        assert np.isclose(v.mean_incidence_deg,
                          np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
        assert np.isclose(v.area_m2, len(ids) * plan.samples["area_per_sample_m2"])


def test_cumulative_coverage_is_monotone():
    plan, m, arr = _plan(synth.sphere_cloud(n=3000), table_plane=FLOATING)
    cum = [v.cum_area_frac for v in plan.views]
    assert cum == sorted(cum)
    assert np.isclose(cum[-1], plan.metrics["coverage_area_frac"])
    gains = [v.gain for v in plan.views]
    assert gains == sorted(gains, reverse=True) or all(g > 0 for g in gains)


def test_coverage_at_wider_angles_is_monotone():
    plan, m, arr = _plan(synth.sphere_cloud(n=3000), table_plane=FLOATING)
    mt = plan.metrics
    assert mt["coverage_45"] <= mt["coverage_60"] <= mt["coverage_75"]
    assert np.isclose(mt["coverage_60"], mt["coverage_area_frac"], atol=0.02)


def test_binary_variant_selects_a_different_plan_on_real_geometry():
    """The ablation control must actually be a control."""
    P = synth.cylinder_cloud(n=5000)
    a, _, _ = _plan(P, "base", table_plane=FLOATING)
    b, _, _ = _plan(P, "binary", table_plane=FLOATING)
    assert a.params["weight"] != b.params["weight"]
    assert a.version != b.version
    assert a.metrics["mean_best_incidence_deg"] <= \
        b.metrics["mean_best_incidence_deg"] + 1e-9


def test_altmesh_variant_uses_the_other_mesher():
    plan, m, arr = _plan(synth.sphere_cloud(n=3000), "altmesh",
                         table_plane=FLOATING)
    assert plan.mesh["method"] == "alpha"


def test_cos75_admits_more_area_than_cos60():
    P = synth.cylinder_cloud(n=5000)
    a, _, _ = _plan(P, "base", table_plane=FLOATING)
    b, _, _ = _plan(P, "cos75", table_plane=FLOATING)
    assert b.metrics["coverage_area_frac"] >= a.metrics["coverage_area_frac"] - 1e-9


# --- continuity ------------------------------------------------------------
def _churn(A: np.ndarray, B: np.ndarray) -> float:
    """The M5 pose-set distance, restated here so the test does not import
    the metric it is checking from the code that optimises against it."""
    if not len(A) or not len(B):
        return float("nan")
    d = (np.linalg.norm(A[:, None, :3, 3] - B[None, :, :3, 3], axis=2)
         + 0.1 * np.arccos(np.clip(np.einsum("ik,jk->ij", A[:, :3, 2],
                                             B[:, :3, 2]), -1, 1)))
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))


def _growing(n_steps=4):
    """A crude scan: the same sphere, resampled and grown, step after step."""
    return [synth.sphere_cloud(n=1500 + 500 * k, seed=k) for k in range(n_steps)]


def test_continuity_holds_the_plan_still_between_steps():
    """The stability fix, measured against its own control on the same clouds."""
    out = {}
    for variant in ("base", "continuity"):
        prev, churn = None, []
        for P in _growing():
            plan, _m, _a = _plan(P, variant, table_plane=FLOATING, prev_T=prev)
            cur = plan.poses
            if prev is not None:
                churn.append(_churn(cur, prev))
            prev = cur
        out[variant] = float(np.mean(churn))
    assert out["continuity"] < out["base"], out


def test_continuity_reuses_previous_poses_but_only_legal_ones():
    P0, P1 = _growing(2)
    a, _m, _ar = _plan(P0, "continuity", table_plane=FLOATING)
    assert a.metrics["n_warm_start"] == 0            # nothing to warm from
    b, _m, _ar = _plan(P1, "continuity", table_plane=FLOATING, prev_T=a.poses)
    assert b.metrics["n_warm_start"] == a.n_views
    assert b.metrics["n_views_from_warm_start"] > 0  # and some were re-picked

    # a previous pose that is now illegal (buried under the table) is dropped
    bad = a.poses.copy()
    bad[:, 2, 3] = -1.0
    c, _m, _ar = _plan(P1, "continuity", table_plane=FLOATING, prev_T=bad)
    assert c.metrics["n_warm_start"] == 0


def test_continuity_never_buys_a_view_that_earns_nothing():
    """The bonus reorders equals; it may not lower the acceptance floor."""
    cover = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=bool)
    cosv = np.array([[0.9, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.9],
                     [0.9, 0.9, 0.0, 0.0]], dtype=float)
    p = dict(planner.params_for("continuity"), max_views=4, min_gain_frac=0.0)
    # view 2 is a duplicate of view 0 and is the "previous" pick
    picks, _t = planner.greedy(cover, cosv, 1.0, p,
                               affinity=np.array([0.0, 0.0, 1.0]))
    assert [x["view_idx"] for x in picks] == [2, 1]   # continuity chose 2 over 0
    assert len(picks) == 2                            # and stopped: 0 adds zero


def test_pose_affinity_is_local():
    T = np.stack([look_at_cv([0.10, -0.38, 0.16], [0, 0, -1.0]),
                  look_at_cv([0.10, -0.38, 0.17], [0, 0, -1.0]),
                  look_at_cv([0.30, -0.38, 0.16], [0, 0, -1.0])])
    aff = planner.pose_affinity(T, T[:1], radius_m=0.05)
    assert aff[0] == pytest.approx(1.0)
    assert 0.0 < aff[1] < 1.0
    assert aff[2] == 0.0                              # 200 mm away: no bonus
    assert np.array_equal(planner.pose_affinity(T, None, 0.05), np.zeros(3))


# --- cross-evaluation ------------------------------------------------------
def test_score_views_on_its_own_surface_reproduces_the_plan_metrics():
    """The yardstick must agree with the planner when it measures the same
    thing — otherwise the convergence curve is a different quantity."""
    plan, m, arr = _plan(synth.sphere_cloud(n=3000), table_plane=FLOATING)
    s = planner.score_views(m, plan.poses, arr["points"], arr["normals"],
                            synth.INTR, planner.params_for("base"))
    assert s["n_views"] == plan.n_views
    assert s["coverage"] == pytest.approx(plan.metrics["coverage_area_frac"])
    assert s["mean_incidence_deg"] == pytest.approx(
        plan.metrics["mean_best_incidence_deg"])


def test_score_views_sees_less_of_a_surface_the_plan_never_knew():
    """Score a small object's plan against a bigger one: coverage must drop."""
    small, m_s, a_s = _plan(synth.sphere_cloud(r=0.03, n=3000),
                            table_plane=FLOATING)
    big, m_b, a_b = _plan(synth.sphere_cloud(r=0.05, n=3000),
                          table_plane=FLOATING)
    params = planner.params_for("base")
    own = planner.score_views(m_b, big.poses, a_b["points"], a_b["normals"],
                              synth.INTR, params)
    cross = planner.score_views(m_b, small.poses, a_b["points"], a_b["normals"],
                                synth.INTR, params)
    assert cross["coverage"] < own["coverage"]
    assert planner.score_views(m_b, np.empty((0, 4, 4)), a_b["points"],
                               a_b["normals"], synth.INTR,
                               params)["coverage"] == 0.0


# --- helpers ---------------------------------------------------------------
def test_farthest_points_is_deterministic_and_spreads():
    P = synth.sphere_cloud(n=2000)
    a = planner.farthest_points(P, 40)
    b = planner.farthest_points(P, 40)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 40
    d = np.linalg.norm(P[a][:, None, :] - P[a][None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.005


def test_recorded_cams_tolerates_a_missing_run(tmp_path):
    assert planner.recorded_cams("no-such-run", 5, data=tmp_path).shape == (0, 3)


def test_session_intrinsics_falls_back_to_the_d405_defaults(tmp_path):
    intr = planner.session_intrinsics("no-such-run", data=tmp_path)
    assert intr["fx"] == planner.DEFAULT_INTR["fx"]
    assert intr["width"] == 848


# The box sweeps live in data/datasets/ with the NATIVE record layout
# (steps/NNN/step.json). Reading data/runs/ + NNN/meta.json only made the
# planner silently degrade on them: default intrinsics and the centroid
# normal rule. These pin the record layer's `Run` as the one read door.

def _native_dataset_run(tmp_path, run_id="0910-boxT"):
    from inspection.record.schema import SessionRecord
    from inspection.tests.record_fixtures import make_run
    d = make_run(tmp_path / "datasets", run_id=run_id)
    intr = dict(width=848, height=480, fx=100.0, fy=101.0,
                ppx=424.0, ppy=240.0, model="none")
    (d / "session.json").write_text(SessionRecord(
        serial="x", resolution=[848, 480], depth_scale_m_per_unit=1e-4,
        intrinsics={"color": intr}).model_dump_json())
    return d


def test_session_intrinsics_reads_a_native_datasets_run(tmp_path):
    _native_dataset_run(tmp_path)
    intr = planner.session_intrinsics("0910-boxT", data=tmp_path)
    assert intr["fx"] == 100.0 and intr["ppy"] == 240.0  # NOT the D405 default


def test_recorded_cams_reads_native_steps_and_skips_rejected(tmp_path):
    # make_run: step 000 captured+fused at the identity pose, step 001
    # rejected pose-less — exactly one camera position must come back.
    _native_dataset_run(tmp_path)
    cams = planner.recorded_cams("0910-boxT", data=tmp_path)
    assert cams.shape == (1, 3) and np.allclose(cams[0], [0.0, 0.0, 0.0])
    assert planner.recorded_cams("0910-boxT", upto_step=0,
                                 data=tmp_path).shape == (1, 3)


def test_session_intrinsics_screams_on_a_broken_real_run(tmp_path):
    # A named REAL run with a missed lookup must never quietly default —
    # the box sweep ran three days on fixture fallbacks and nothing said so.
    d = _native_dataset_run(tmp_path, run_id="0910-boxU")
    (d / "session.json").unlink()
    with pytest.raises(RuntimeError, match="0910-boxU"):
        planner.session_intrinsics("0910-boxU", data=tmp_path)


def test_session_intrinsics_defaults_on_a_fake_rig_run(tmp_path):
    # rig=fake DECLARES the absent session (schema) — default is legitimate.
    from inspection.tests.record_fixtures import make_run
    make_run(tmp_path / "datasets", run_id="0910-fakeT", rig="fake")
    intr = planner.session_intrinsics("0910-fakeT", data=tmp_path)
    assert intr == planner.DEFAULT_INTR


def test_plan_metrics_stamp_their_input_provenance():
    plan, _, _ = _plan(synth.sphere_cloud(n=2000), table_plane=FLOATING)
    m = plan.metrics
    assert m["intrinsics_source"] == "caller"
    assert m["normals_rule"] == "centroid_outward"  # _plan passes empty cams
    assert m["n_recorded_cams"] == 0
