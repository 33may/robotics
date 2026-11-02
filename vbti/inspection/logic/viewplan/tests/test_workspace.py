"""Fixture-driven tests for `vbti/inspection/logic/viewplan/workspace.py`.

Deliberately independent of `inspection.viewplan.contract` / `mesh` /
`planner` — per `docs/research/view-planning/poc/structure.md`, the
interface between AGENT-RR and the rest of the package is the JSON/npz
schema, not a Python import. Everything below is a plain dict + a plain
`.npz` + a tiny open3d mesh, built by hand, so this file (and `workspace.py`
itself) is provable before the planner exists and stays provable after,
unaffected by anything CORE changes inside `contract.py`.
"""
from __future__ import annotations

import json

import numpy as np
import open3d as o3d
import pytest

from inspection.viewplan import workspace as ws

RUN = "fx-run"
VARIANT = "fx-variant"

INTR = {"width": 848, "height": 480, "fx": 431.6, "fy": 431.3,
       "ppx": 417.2, "ppy": 237.8}


def _write_ply_cloud(path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pts = rng.uniform([-0.05, -0.4, 0.01], [0.05, -0.3, 0.08], size=(n, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def _write_mesh(path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_box(0.06, 0.06, 0.06)
    mesh.translate([-0.03, -0.35, 0.01])
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)


def _write_samples(path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pts = rng.uniform([-0.03, -0.35, 0.01], [0.03, -0.29, 0.07], size=(n, 3))
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    area = np.full(n, 0.0001 / n)
    best_cos = rng.uniform(0.3, 1.0, size=n)
    covered_by = rng.integers(0, 2, size=n).astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=pts, normals=normals, area=area,
                        best_cos=best_cos, covered_by=covered_by)


def _write_png(path) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :, 1] = 200
    cv2.imwrite(str(path), img)


def _view(view_id: int, render: str = "") -> dict:
    T = np.eye(4)
    T[:3, 3] = [0.02 * view_id, -0.32, 0.10]
    return {
        "view_id": view_id,
        "T_base_cam": T.tolist(),
        "standoff_m": 0.10,
        "gain": 0.005,
        "covered": [0, 1, 2],
        "n_covered": 3,
        "area_m2": 0.005,
        "mean_incidence_deg": 20.0 + view_id,
        "p95_incidence_deg": 40.0,
        "cum_area_frac": 0.3 + 0.1 * view_id,
        "render": render,
        "frame_fill": 0.3 if render else 0.0,
    }


def _plan(pose_id: int, *, abstain: bool = False, reason: str = "",
         views: list | None = None, mesh_rel: str = "",
         samples_rel: str = "") -> dict:
    views = views or []
    return {
        "run": RUN,
        "step": pose_id,
        "variant": VARIANT,
        "version": f"{VARIANT}/0.1+p:deadbeef",
        "seed": 0,
        "abstain": abstain,
        "reason": reason,
        "table_plane": [0.0, 0.0, 1.0, -0.006],
        "mesh": ({} if not mesh_rel else {
            "path": mesh_rel, "method": "poisson", "depth": 6,
            "n_vertices": 8, "n_triangles": 12, "area_m2": 0.0216,
            "watertight": True}),
        "samples": ({} if not samples_rel else {
            "path": samples_rel, "count": 50,
            "area_per_sample_m2": 2e-6}),
        "params": {"standoff_m": 0.10, "incidence_cutoff_deg": 60.0,
                  "max_views": 2, "weight": "cosine"},
        "views": views,
        "metrics": {"n_views": len(views), "coverage_area_frac": 0.6,
                   "weighted_coverage": 0.5,
                   "mean_best_incidence_deg": 25.0, "seconds": 1.2},
    }


@pytest.fixture
def fixture_tree(tmp_path):
    """derived/<run>/{steps,viewplan/<variant>} + runs/<run>/session.json.

    Four steps, exactly the transitions `structure.md` asks this test to
    prove: pose 0 has views {0,1}; pose 1 has {0} only (view 1 must be
    retired); pose 2 abstains (view 0 must then also be retired); pose 3
    has no plan file on disk at all (the honest-gap path).
    """
    derived = tmp_path / "derived"
    runs = tmp_path / "runs"
    steps_dir = derived / RUN / "steps"
    variant_dir = derived / RUN / "viewplan" / VARIANT

    for pid in (0, 1, 2, 3):
        _write_ply_cloud(steps_dir / f"step_{pid:03d}.ply", 40, seed=pid)
    summary = {"run": RUN, "steps": [{"pose_id": pid, "k": i + 1,
                                      "source": "mask"}
                                     for i, pid in enumerate((0, 1, 2, 3))]}
    (steps_dir / "summary.json").write_text(json.dumps(summary))

    (runs / RUN).mkdir(parents=True, exist_ok=True)
    (runs / RUN / "session.json").write_text(
        json.dumps({"intrinsics": {"color": INTR}}))

    _write_mesh(variant_dir / "meshes" / "step_000.ply")
    _write_samples(variant_dir / "samples" / "step_000.npz", 50, seed=100)
    _write_png(variant_dir / "renders" / "step_000_view_00.png")
    (variant_dir / "step_000.json").write_text(json.dumps(_plan(
        0, views=[_view(0, "renders/step_000_view_00.png"), _view(1)],
        mesh_rel="meshes/step_000.ply",
        samples_rel="samples/step_000.npz")))

    _write_mesh(variant_dir / "meshes" / "step_001.ply")
    _write_samples(variant_dir / "samples" / "step_001.npz", 50, seed=101)
    (variant_dir / "step_001.json").write_text(json.dumps(_plan(
        1, views=[_view(0)],
        mesh_rel="meshes/step_001.ply",
        samples_rel="samples/step_001.npz")))

    (variant_dir / "step_002.json").write_text(json.dumps(_plan(
        2, abstain=True, reason="too few points")))

    # pose 3: no step_003.json at all.

    return derived, runs


def test_build_writes_rrd_and_rbl(fixture_tree):
    derived, runs = fixture_tree
    path = ws.build(RUN, VARIANT, derived=derived, runs_dir=runs)
    assert path.exists() and path.stat().st_size > 0
    rbl = path.with_suffix(".rbl")
    assert rbl.exists() and rbl.stat().st_size > 0
    assert path == derived / RUN / "rrd" / f"{RUN}-viewplan.rrd"


def test_retired_views_are_cleared(fixture_tree, monkeypatch):
    """Step 0 has views {0,1}; step 1 has {0}; step 2 abstains.

    So `clear_primitives` must be called once with exactly the view-1
    camera path (after step 1) and once with exactly the view-0 camera
    path (after step 2 abstains it away) — the diff bookkeeping
    `structure.md` asks this test to prove.
    """
    derived, runs = fixture_tree
    calls: list[set[str]] = []
    real_clear = ws.rr_log.clear_primitives

    def spy(paths):
        s = set(paths)
        calls.append(s)
        return real_clear(paths)

    monkeypatch.setattr(ws.rr_log, "clear_primitives", spy)
    ws.build(RUN, VARIANT, derived=derived, runs_dir=runs)

    nonempty = [c for c in calls if c]
    assert {f"{ws.rr_log.VP_CAMS}/01"} in nonempty, (
        f"view 1's camera path was never retired; calls={nonempty}")
    assert {f"{ws.rr_log.VP_CAMS}/00"} in nonempty, (
        f"view 0's camera path was never retired on the abstain step; "
        f"calls={nonempty}")


def test_render_logged_only_for_milestone_view(fixture_tree, monkeypatch):
    derived, runs = fixture_tree
    logged: list[tuple[str, str]] = []
    real_render = ws.rr_log.log_viewplan_render

    def spy(slot, image_path):
        logged.append((slot, str(image_path)))
        return real_render(slot, image_path)

    monkeypatch.setattr(ws.rr_log, "log_viewplan_render", spy)
    ws.build(RUN, VARIANT, derived=derived, runs_dir=runs)

    assert len(logged) == 1
    slot, img = logged[0]
    assert slot == "view_00"
    assert img.endswith("step_000_view_00.png")


def test_blueprint_render_grid_matches_max_views(fixture_tree, monkeypatch):
    derived, runs = fixture_tree
    captured = {}
    real_bp = ws.rr_log.viewplan_blueprint

    def spy(run, variant, render_slots, **kw):
        captured["slots"] = list(render_slots)
        return real_bp(run, variant, render_slots, **kw)

    monkeypatch.setattr(ws.rr_log, "viewplan_blueprint", spy)
    ws.build(RUN, VARIANT, derived=derived, runs_dir=runs)

    # the fixture's params carry max_views=2
    assert captured["slots"] == ["view_00", "view_01"]


def test_missing_variant_dir_raises(tmp_path):
    derived = tmp_path / "derived"
    runs = tmp_path / "runs"
    steps_dir = derived / RUN / "steps"
    _write_ply_cloud(steps_dir / "step_000.ply", 10, seed=0)
    (steps_dir / "summary.json").write_text(json.dumps(
        {"run": RUN, "steps": [{"pose_id": 0, "k": 1, "source": "mask"}]}))
    with pytest.raises(SystemExit):
        ws.build(RUN, "no-such-variant", derived=derived, runs_dir=runs)


def test_all_steps_planned(fixture_tree, capsys):
    """Sanity: the honest-gap and abstain steps do not crash the build,
    and the console summary reports 4 steps walked, 2 actually planned."""
    derived, runs = fixture_tree
    ws.build(RUN, VARIANT, derived=derived, runs_dir=runs)
    out = capsys.readouterr().out
    assert "4 steps (2 planned)" in out
