"""cli.py: step selection, a full `run` pass, and the cross-evaluation.

No real run data is touched — `cli.DERIVED` is monkeypatched to a `tmp_path`
tree for every test, so this suite cannot write into
`vbti/inspection/data/derived/`.

The cross-evaluation tests are the ones that matter most: `plan_step` scores
a plan on its own step's mesh, whose area grows over a run, so that curve
falls as the scan improves. `cross_eval` re-scores every step against ONE
surface, and the tests below pin the two properties that makes it worth
having — the yardstick is the same for every step (`xe_ref`), and a plan made
from a partial cloud scores *worse* on the finished surface than the final
plan does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inspection.viewplan import cli
from inspection.viewplan.tests.synth import sphere_cloud


def _write_ply(path: Path, P: np.ndarray) -> None:
    import open3d as o3d
    path.parent.mkdir(parents=True, exist_ok=True)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    o3d.io.write_point_cloud(str(path), pc, write_ascii=False)


# --------------------------------------------------------------------------
# steps_of / select
def test_steps_of_lists_by_glob_not_range(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    steps_dir = tmp_path / "run1" / "steps"
    for n in (0, 2, 5, 30):                       # a cup5-style gap
        _write_ply(steps_dir / f"step_{n:03d}.ply", sphere_cloud(n=50, seed=n))
    paths = cli.steps_of("run1")
    assert [int(p.stem.split("_")[1]) for p in paths] == [0, 2, 5, 30]


def test_steps_of_missing_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    with pytest.raises(SystemExit):
        cli.steps_of("nope")


def test_select_all_final_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    steps_dir = tmp_path / "run1" / "steps"
    for n in (0, 5, 10):
        _write_ply(steps_dir / f"step_{n:03d}.ply", sphere_cloud(n=50, seed=n))
    paths = cli.steps_of("run1")
    assert cli.select(paths, "all") == paths
    assert [int(p.stem.split("_")[1]) for p in cli.select(paths, "final")] == [10]
    assert [int(p.stem.split("_")[1]) for p in cli.select(paths, "0,10")] == [0, 10]
    with pytest.raises(SystemExit):
        cli.select(paths, "0,7")


def test_milestone_and_anchor_steps_intersect_available():
    # cup5-shaped gap: no 25..29, last step is 30 — the formula must not
    # invent a milestone that is not on disk (the documented quirk).
    ids = list(range(0, 25)) + [30]
    ms = cli.milestone_steps(ids)
    an = cli.anchor_steps(ids)
    assert ms == {0, 2, 5, 8, 12, 16, 20, 30}
    assert an == {2, 12, 30}
    assert 25 not in ms                            # would be wrong if hardcoded


def test_render_cadence_by_variant():
    ms, an = {0, 12, 29}, {2, 12, 29}
    assert cli.wants_render("base", 12, ms, an)
    assert not cli.wants_render("base", 7, ms, an)
    assert cli.wants_render("binary", 2, ms, an)
    assert not cli.wants_render("binary", 12 - 1, ms, an)
    assert not cli.wants_render("cos75", 12, ms, an)   # never renders


# --------------------------------------------------------------------------
# the fire gotcha: argument values arrive literal-evaled, so a comma list can
# be a tuple, a single step an int — `_csv` folds every shape back to strings
def test_csv_accepts_every_shape_fire_produces():
    assert cli._csv("2408-cup3,2408-cup4") == ["2408-cup3", "2408-cup4"]
    assert cli._csv((0, 5, 10)) == ["0", "5", "10"]    # --steps 0,5,10
    assert cli._csv(5) == ["5"]                        # --steps 5
    assert cli._csv("base") == ["base"]
    assert cli._csv([" a ", "", "b"]) == ["a", "b"]


# --------------------------------------------------------------------------
# a full `run` pass, synthetic data, no real run touched
def test_run_step0_writes_plan_mesh_npz_and_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    steps_dir = tmp_path / "sphere-run" / "steps"
    _write_ply(steps_dir / "step_000.ply", sphere_cloud(n=2000, seed=0))

    cli.cli_run(run="sphere-run", steps="0", render=False)

    vdir = tmp_path / "sphere-run" / "viewplan" / "base"
    plan_path = vdir / "step_000.json"
    mesh_path = vdir / "meshes" / "step_000.ply"
    npz_path = vdir / "samples" / "step_000.npz"
    scores_path = vdir / "scores.jsonl"
    for p in (plan_path, mesh_path, npz_path, scores_path):
        assert p.exists() and p.stat().st_size > 0, p

    plan = json.loads(plan_path.read_text())
    assert plan["run"] == "sphere-run" and plan["step"] == 0
    assert plan["abstain"] is False
    assert len(plan["views"]) >= 1

    rows = [json.loads(l) for l in scores_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["step"] == 0 and rows[0]["variant"] == "base"


def test_run_merges_scores_never_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    steps_dir = tmp_path / "sphere-run" / "steps"
    _write_ply(steps_dir / "step_000.ply", sphere_cloud(n=2000, seed=0))
    _write_ply(steps_dir / "step_005.ply", sphere_cloud(n=2000, seed=5))

    cli.cli_run(run="sphere-run", steps="0", render=False)
    # must not delete step 0's row; steps=5 arrives as an int under fire
    cli.cli_run(run="sphere-run", steps=5, render=False)

    scores_path = tmp_path / "sphere-run" / "viewplan" / "base" / "scores.jsonl"
    rows = {json.loads(l)["step"] for l in scores_path.read_text().splitlines()
           if l.strip()}
    assert rows == {0, 5}

    # re-running step 0 again must still leave exactly one row per step
    cli.cli_run(run="sphere-run", steps="0", render=False)
    rows2 = [json.loads(l)["step"] for l in scores_path.read_text().splitlines()
            if l.strip()]
    assert sorted(rows2) == [0, 5]


def test_run_no_render_skips_render_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    steps_dir = tmp_path / "sphere-run" / "steps"
    _write_ply(steps_dir / "step_000.ply", sphere_cloud(n=2000, seed=0))
    cli.cli_run(run="sphere-run", steps="0", render=False)
    renders = tmp_path / "sphere-run" / "viewplan" / "base" / "renders"
    assert not renders.exists() or not any(renders.iterdir())


def test_run_with_render_writes_view_images(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    steps_dir = tmp_path / "sphere-run" / "steps"
    _write_ply(steps_dir / "step_000.ply", sphere_cloud(n=2000, seed=0))
    # step 0 is a milestone for every run (max of an empty-intersection set
    # is always included), so `base` must render it.
    cli.cli_run(run="sphere-run", steps="0", render=True)
    renders = tmp_path / "sphere-run" / "viewplan" / "base" / "renders"
    imgs = list(renders.glob("step_000_view_*.png"))
    assert imgs, "expected at least one rendered view at the milestone step"
    plan = json.loads((tmp_path / "sphere-run" / "viewplan" / "base"
                       / "step_000.json").read_text())
    assert all(v["render"] for v in plan["views"])


def _partial_sphere(frac: float, n: int = 2500, seed: int = 0) -> np.ndarray:
    """A sphere seen from one side only — the shape of an unfinished scan.

    `frac` is the share of the shell that has been observed, cut as a spherical
    cap around +X, so a plan made at `frac=0.5` has literally never seen the
    surface the final plan is judged on.
    """
    P = sphere_cloud(n=int(n / max(frac, 0.05)), seed=seed)
    c = np.asarray([0.10, -0.38, 0.06])
    d = (P - c) / np.linalg.norm(P - c, axis=1, keepdims=True)
    keep = d[:, 0] >= (1.0 - 2.0 * frac)              # cap of area fraction frac
    return P[keep]


def _growing_run(tmp_path, run: str = "grow-run") -> None:
    steps = tmp_path / run / "steps"
    for step, frac in ((0, 0.4), (1, 0.7), (2, 1.0)):
        _write_ply(steps / f"step_{step:03d}.ply", _partial_sphere(frac, seed=step))


# --------------------------------------------------------------------------
# cross-evaluation: every step's plan re-scored on ONE surface
def test_cross_eval_scores_every_step_against_the_same_final_surface(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    _growing_run(tmp_path)
    cli.cli_run(run="grow-run", steps="all", render=False)

    vdir = tmp_path / "grow-run" / "viewplan" / "base"
    rows = cli._read_scores(vdir)
    assert [r["step"] for r in rows] == [0, 1, 2]
    for r in rows:
        for key in ("xe_ref", "xe_coverage", "xe_coverage_coverable",
                    "xe_mean_incidence_deg", "xe_frac_of_final"):
            assert key in r, f"step {r['step']} missing {key}"
    # one yardstick for every step — the whole point
    assert len({r["xe_ref"] for r in rows}) == 1
    assert rows[-1]["xe_ref"] == "base/step_002"
    assert rows[-1]["xe_frac_of_final"] == pytest.approx(1.0)


def test_the_cross_eval_curve_rises_as_the_scan_grows(tmp_path, monkeypatch):
    """The property the self-scored curve does not have.

    A plan made from 40 % of a sphere cannot see the other 60 %, so on the
    finished surface it must score strictly worse than the plan made from the
    whole thing. The self-scored number is free to do the opposite (its mesh
    is 40 % as large), which is exactly why it cannot answer "after which step
    is the geometry good enough".
    """
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    _growing_run(tmp_path)
    cli.cli_run(run="grow-run", steps="all", render=False)

    rows = cli._read_scores(tmp_path / "grow-run" / "viewplan" / "base")
    xe = [r["xe_coverage"] for r in rows]
    assert xe[0] < xe[-1] - 0.05, xe
    assert xe[0] < xe[1] + 1e-9                       # monotone on a growing cap
    assert all(0.0 <= r["xe_frac_of_final"] <= 1.0 + 1e-9 for r in rows)


def test_cross_eval_falls_back_to_the_variants_own_final_geometry(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    _growing_run(tmp_path)
    cli.cli_run(run="grow-run", steps="all", variant="binary", render=False)
    assert cli.reference("grow-run", "base") is None       # never planned
    rows = cli.cross_eval("grow-run", "binary", None, "base")
    assert rows and all(r["xe_ref"] == "binary/step_002" for r in rows)


def test_k_star_is_sustained_not_first_touch():
    rows = [{"step": 0, "v": 0.99}, {"step": 1, "v": 0.10},
            {"step": 2, "v": 0.96}, {"step": 3, "v": 0.97}]
    assert cli.k_star(rows, "v", 0.95) == 2            # not 0: step 1 fell back
    assert cli.k_star(rows, "v", 1.5) is None
    assert cli.k_star([], "v", 0.5) is None


def test_churn_is_nan_at_the_first_step_and_zero_for_a_frozen_plan(tmp_path):
    from inspection.viewplan.contract import look_at_cv
    vdir = tmp_path / "v"
    vdir.mkdir()
    T = [look_at_cv((0.2, -0.38, 0.06), (-1.0, 0.0, 0.0)).tolist(),
         look_at_cv((0.0, -0.38, 0.06), (1.0, 0.0, 0.0)).tolist()]
    for step in (0, 1):
        (vdir / f"step_{step:03d}.json").write_text(json.dumps(
            {"views": [{"T_base_cam": t} for t in T]}))
    moved = look_at_cv((0.25, -0.38, 0.06), (-1.0, 0.0, 0.0)).tolist()
    (vdir / "step_002.json").write_text(json.dumps(
        {"views": [{"T_base_cam": moved}, {"T_base_cam": T[1]}]}))

    rows = [{"step": 0, "abstain": False}, {"step": 1, "abstain": False},
            {"step": 2, "abstain": False}]
    ch = cli.churn(vdir, rows)
    assert np.isnan(ch[0])
    assert ch[1] == pytest.approx(0.0, abs=1e-12)
    assert ch[2] == pytest.approx(0.5 * (0.05 / 2 + 0.05 / 2), abs=1e-6)


# --------------------------------------------------------------------------
# view budget — what 95 % of the achievable surface actually costs
def test_budget_lifts_the_cap_and_reports_the_price(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    steps_dir = tmp_path / "sphere-run" / "steps"
    _write_ply(steps_dir / "step_000.ply", sphere_cloud(n=2500, seed=0))
    cli.cli_run(run="sphere-run", steps="0", render=False)

    b = cli.budget("sphere-run", "base", cap=20)
    plan = json.loads((tmp_path / "sphere-run" / "viewplan" / "base"
                       / "step_000.json").read_text())
    assert b["step"] == 0
    assert b["n_uncapped"] > len(plan["views"]) or b["n_uncapped"] == 20 \
        or b["terminator"] == "gain"
    # the capped plan can never beat the uncapped one on the same geometry
    assert b["final_coverage_coverable"] >= \
        plan["metrics"]["coverage_coverable_frac"] - 1e-6
    # the price of a target is monotone in the target
    prices = [b["views_needed"][k] for k in ("0.90", "0.95", "0.99")]
    seen = [p for p in prices if p is not None]
    assert seen == sorted(seen)
    for n, (raw, able) in b["at_n_views"].items():
        assert 0.0 <= raw <= 1.0 and 0.0 <= able <= 1.0 + 1e-9


def test_budget_without_a_planned_step_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    with pytest.raises(SystemExit):
        cli.budget("nothing-here", "base")


# --------------------------------------------------------------------------
# the continuity variant's chaining, at the CLI level
def test_only_the_continuity_variant_is_chained():
    assert cli.is_chained("continuity")
    assert not any(cli.is_chained(v) for v in cli.planner.variants()
                   if v != "continuity")


def test_prev_poses_skips_abstains_and_gaps(tmp_path):
    vdir = tmp_path / "continuity"
    vdir.mkdir()
    good = np.eye(4)
    good[:3, 3] = (0.2, -0.38, 0.06)
    (vdir / "step_000.json").write_text(json.dumps(
        {"views": [{"T_base_cam": good.tolist()}]}))
    (vdir / "step_001.json").write_text(json.dumps(
        {"abstain": True, "views": []}))
    # step 2 is missing entirely (a --steps gap); step 3 must reach past both
    got = cli.prev_poses(vdir, 3)
    assert got is not None and got.shape == (1, 4, 4)
    assert got[0][:3, 3].tolist() == [0.2, -0.38, 0.06]
    assert cli.prev_poses(vdir, 0) is None             # nothing before step 0


def test_continuity_run_warm_starts_from_the_previous_step(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    _growing_run(tmp_path)
    cli.cli_run(run="grow-run", steps="all", variant="continuity", render=False)
    rows = cli._read_scores(tmp_path / "grow-run" / "viewplan" / "continuity")
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert rows[0]["n_warm_start"] == 0                # nothing to warm from
    assert rows[-1]["n_warm_start"] > 0
    assert rows[-1]["n_views_from_warm_start"] > 0     # and some were re-picked


def test_show_and_variants_do_not_crash(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DERIVED", tmp_path)
    steps_dir = tmp_path / "sphere-run" / "steps"
    _write_ply(steps_dir / "step_000.ply", sphere_cloud(n=2000, seed=0))
    cli.cli_run(run="sphere-run", steps="0", render=False)

    # steps=0 as fire would deliver it: an int, not "0"
    cli.cli_show(run="sphere-run", variant="base", steps=0)
    out = capsys.readouterr().out
    assert "step_000" in out and "view" in out

    cli.cli_variants()
    out = capsys.readouterr().out
    assert "base" in out and "binary" in out
