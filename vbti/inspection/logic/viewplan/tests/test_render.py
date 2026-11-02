"""A render must show the surface its plan claims it shows.

That sentence is the whole point of the render pass, so it is a test rather
than a promise: every planned view's claimed patches are re-projected into
its own render and depth-tested against that render's own buffer. Anything
less and "looks right" is the only quality control we have.
"""
import json

import numpy as np

from inspection.viewplan import mesh as meshmod
from inspection.viewplan import planner
from inspection.viewplan import render
from inspection.viewplan.contract import look_at_cv, plan_to_json
from inspection.viewplan.tests import synth

FLOATING = [0.0, 0.0, 1.0, 0.10]


def _sphere_plan(tmp_path, variant="base", n=3000):
    P = synth.sphere_cloud(n=n)
    plan, m, arr = planner.plan_step(P, "synth", 7, variant, seed=0,
                                     intr=synth.INTR, cams=np.empty((0, 3)),
                                     table_plane=FLOATING)
    assert not plan.abstain, plan.reason
    d = tmp_path / variant
    meshmod.save(m, d / plan.mesh["path"])
    planner.save_arrays(arr, d / plan.samples["path"])
    j = plan_to_json(plan)
    (d / "step_007.json").write_text(json.dumps(j))
    return d, j, m, arr


def test_render_view_frames_the_object():
    P = synth.sphere_cloud()
    m, _ = meshmod.reconstruct(P)
    c = np.asarray(synth.CENTER)
    pos = c + np.array([0.14, 0.0, 0.0])
    img, st = render.render_view(m, look_at_cv(pos, c - pos), synth.INTR)
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert img.shape[:2] == (240, 424)                 # scale 0.5 of 848x480
    assert st["frame_fill"] > 0.05, st
    assert img.max() > 60, "render is black"
    assert 0.09 < st["t_min"] < 0.12, st                # the standoff shell


def test_every_planned_view_frames_what_it_claims(tmp_path):
    """The B3 bar (roadmap M10) as an executable check."""
    d, j, m, arr = _sphere_plan(tmp_path)
    out = render.render_plan(d, j, synth.INTR)
    views = [r for r in out if r["view_id"] >= 0]
    assert len(views) == len(j["views"])
    for r in views:
        assert r["frame_fill"] > 0.05, r
        assert r["n_claimed"] > 0, r
        assert r["framed_frac"] >= 0.9, r


def test_render_files_land_where_the_plan_says(tmp_path):
    d, j, m, arr = _sphere_plan(tmp_path)
    out = render.render_plan(d, j, synth.INTR)
    for r in out:
        p = d / r["render"]
        assert p.exists() and p.stat().st_size > 0
        assert r["render"].startswith("renders/step_007_")
    assert (d / "renders" / "step_007_coverage.png").exists()
    assert any(r["view_id"] == -1 for r in out)         # the coverage panel


def test_render_of_a_bogus_claim_is_caught(tmp_path):
    """The check has teeth: claim patches the view cannot see, watch it fail.

    Without this, `framed_frac == 1.0` might only mean the test never asks a
    hard question.
    """
    d, j, m, arr = _sphere_plan(tmp_path)
    pts, nrm = arr["points"], arr["normals"]
    v = j["views"][0]
    T = np.asarray(v["T_base_cam"], float)
    # patches on the FAR side of the sphere: they project happily inside the
    # silhouette, so only the depth test can tell they are not visible
    back = np.flatnonzero(np.einsum("ij,ij->i", nrm, T[:3, 3] - pts) < -0.02)
    assert len(back) > 20
    img, st = render.render_view(m, T, synth.INTR,
                                 cover_ids=back[:300], samples=pts)
    assert st["framed_frac"] < 0.05, st


def test_coverage_panel_is_a_two_by_two_grid(tmp_path):
    d, j, m, arr = _sphere_plan(tmp_path)
    out = d / "cov.png"
    p = render.coverage_panel(m, {k: arr[k] for k in arr}, out, intr=synth.INTR)
    assert p.exists()
    from PIL import Image
    img = np.asarray(Image.open(p))
    assert img.shape[0] * 2 == img.shape[1] * 1 or img.shape[0] == img.shape[1]
    assert img.ndim == 3
    # four tiles, each with object pixels in it
    h, w = img.shape[0] // 2, img.shape[1] // 2
    for r in (0, 1):
        for cc in (0, 1):
            tile = img[r * h:(r + 1) * h, cc * w:(cc + 1) * w]
            assert (tile.max(2) > 60).mean() > 0.05, (r, cc)


def test_uncovered_area_is_screaming_red(tmp_path):
    """The panel must find the hole for the eye, not only for the metric."""
    d, j, m, arr = _sphere_plan(tmp_path)
    a = {k: arr[k].copy() for k in arr}
    a["best_cos"][:] = 0.0                              # nothing covered
    p = render.coverage_panel(m, a, d / "red.png", intr=synth.INTR)
    from PIL import Image
    img = np.asarray(Image.open(p)).reshape(-1, 3).astype(int)
    obj = img[img.max(1) > 60]
    red = (obj[:, 0] > 150) & (obj[:, 1] < 90) & (obj[:, 2] < 90)
    assert red.mean() > 0.3, red.mean()


def test_render_plan_returns_nothing_for_an_abstain(tmp_path):
    from inspection.viewplan.contract import abstained
    j = plan_to_json(abstained("synth", 0, "base", planner.params_for("base"),
                               0, "cloud not ripe"))
    assert render.render_plan(tmp_path, j, synth.INTR) == []


def test_out_of_window_range_is_tinted():
    """A view standing outside the D405 window must LOOK wrong."""
    P = synth.sphere_cloud()
    m, _ = meshmod.reconstruct(P)
    c = np.asarray(synth.CENTER)

    def redness(dist):
        pos = c + np.array([dist, 0.0, 0.0])
        img, _st = render.render_view(m, look_at_cv(pos, c - pos), synth.INTR)
        o = img.reshape(-1, 3).astype(int)
        o = o[o.max(1) > 40]
        return float((o[:, 0] > o[:, 2] + 40).mean())

    assert redness(0.14) < 0.05                         # inside the window
    assert redness(0.45) > 0.5                          # far outside it


def test_save_png_creates_parents(tmp_path):
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    p = render.save_png(img, tmp_path / "a" / "b" / "c.png")
    assert p.exists()
