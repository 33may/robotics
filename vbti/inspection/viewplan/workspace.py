#!/usr/bin/env python3
"""Build the viewplan review workspace: `<run>-viewplan.rrd`, one variant in it.

    p vbti/inspection/viewplan/workspace.py --run 2408-cup3
    p vbti/inspection/viewplan/workspace.py --run 2408-cup3 --variant continuity
    p vbti/inspection/viewplan/workspace.py --run 2408-cup3 --open

Standalone still works, byte-identical flags:

    p vbti/inspection/viewplan/workspace.py --run 2408-cup3 --open

The file name carries the run and NOT the variant (the path is fixed by
`structure.md`), so a second `--variant` for the same run **overwrites** the
first. That is deliberate — the rrd is a review surface for one hypothesis at
a time, and two variants' magenta frusta in one recording are unreadable —
but it does mean the last invocation wins. Use `--out` to keep both.

Reads ONLY: `derived/<run>/steps/` (clouds + `summary.json`),
`derived/<run>/viewplan/<variant>/` (JSON dicts via `json.loads`, `.npz`
sidecars via `np.load`, mesh `.ply` via open3d), and
`runs/<run>/session.json` (colour intrinsics). Never touches
`derived/<run>/rrd/<run>.rrd` — the main record session's file — and
imports nothing from `vbti/inspection/record/` except `record.rr.log`. This
module also never imports `inspection.viewplan` itself: the JSON/npz
schema in `docs/research/view-planning/poc/structure.md` IS the interface,
so it runs against ANY producer that writes that schema — including the
hand-written fixture `tests/test_workspace.py` builds — with no import-time
dependency on the planner package landing first.

Per run step k, walking `steps/summary.json` in order (same order and same
`step` timeline the record session's `rr/workspace.py` uses, so scrubbing
one file and then the other never feels like two different clocks):

  1. `log_cloud` — the accumulated object cloud, every step, no exceptions
     (a plan that failed to build is still worth seeing the geometry for).
  2. `log_viewplan_mesh` — that step's reconstructed mesh, if the plan has
     one (an abstained step does not).
  3. `log_viewplan_coverage` — the plan's surface samples, coloured red
     (uncovered) to green (well-covered), split at the variant's own
     incidence gate. Sibling subtree `plan/patches/view_NN` holds the SAME
     samples split by owning view (frustum colour = territory colour),
     every patch HIDDEN by default in the blueprint — the review gesture
     is revealing them one eye-click at a time.
  4. `log_viewplan_camera` per planned view, magenta, retired with
     `clear_primitives` when a later step's plan is a different size —
     plans CHANGE between steps, recorded frusta never do.
  5. `log_viewplan_render` for any view whose `render` field on disk
     exists this step (the milestone/anchor cadence — most steps log none,
     and that is by design, not a bug in this file).
  6. `log_viewplan_meta` — abstain / reason / n_views / coverage on the
     `/world/plan` entity, so an empty step reads as "refused" or "no plan
     on disk yet", never as silence.

`log_table` is called ONCE, after the loop, with the LAST successfully
planned step's own `table_plane` — the deterministic, non-RANSAC plane
`mesh.table_plane` writes into every `step_NNN.json` — never the unseeded
`fit_table` the main record workspace re-derives for display only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from inspection.record.rr import log as rr_log

DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = DATA / "runs"
DERIVED = DATA / "derived"

DEFAULT_VARIANT = "base"
#: Render-grid width when no step's `params.max_views` is available yet
#: (should not happen once the planner lands; a hand-built fixture step
#: may omit it, and the grid still has to have SOME fixed shape).
DEFAULT_MAX_VIEWS = 12


# ------------------------------------------------------------------- inputs
def _steps_dir(run: str, derived: Path) -> Path:
    d = derived / run / "steps"
    if not (d / "summary.json").exists():
        raise SystemExit(f"no step clouds for {run} in {d} — the record "
                         f"session's derived tree must exist first")
    return d


def _variant_dir(run: str, variant: str, derived: Path) -> Path:
    d = derived / run / "viewplan" / variant
    if not d.is_dir():
        raise SystemExit(f"no plans for {run}/{variant} in {d} — run "
                         f"vbti/inspection/viewplan/cli.py run --run {run} "
                         f"--variant {variant} first")
    return d


def _load_step_cloud(steps_dir: Path, pose_id: int) -> np.ndarray:
    ply = steps_dir / f"step_{pose_id:03d}.ply"
    if not ply.exists():
        raise SystemExit(f"missing step cloud {ply}")
    pcd = o3d.io.read_point_cloud(str(ply))
    return np.asarray(pcd.points)


def _load_mesh(variant_dir: Path, rel_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) from the plan's own `mesh.path`, or empty."""
    if not rel_path:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    p = variant_dir / rel_path
    if not p.exists():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    mesh = o3d.io.read_triangle_mesh(str(p))
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def _load_samples(variant_dir: Path, rel_path: str) -> dict[str, np.ndarray]:
    if not rel_path:
        return {}
    p = variant_dir / rel_path
    if not p.exists():
        return {}
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


# -------------------------------------------------------------------- build
def build(run: str, variant: str = DEFAULT_VARIANT,
         out_dir: Path | None = None, derived: Path | None = None,
         runs_dir: Path | None = None, probe: int = 0) -> Path:
    """Write `<derived>/<run>/rrd/<run>-viewplan.rrd` for one variant.

    `derived` / `runs_dir` are overridable so `tests/test_workspace.py` can
    point this at a `tmp_path` fixture tree without monkeypatching module
    globals.
    """
    derived = Path(derived) if derived else DERIVED
    # An explicit runs_dir (tests) wins; otherwise resolve across BOTH
    # archives — box sweeps live in data/datasets/ (planner.run_dir).
    if runs_dir:
        run_root = Path(runs_dir) / run
    else:
        from inspection.viewplan.planner import run_dir as _resolve
        run_root = _resolve(run) or RUNS / run
    steps_dir = _steps_dir(run, derived)
    variant_dir = _variant_dir(run, variant, derived)
    summary = json.loads((steps_dir / "summary.json").read_text())
    session = json.loads((run_root / "session.json").read_text())
    intr = session["intrinsics"]["color"]

    # Dash, never slash — 0.37 silently hashes an app-id with a `/` in it
    # (record/rr/workspace.py:224); one collision here would rename the
    # window title AND split the app-id's persisted-blueprint cache.
    app_id = f"viewplan-{run}-{variant}"
    rr_log.begin(app_id)
    rr_log.log_world()

    max_views = DEFAULT_MAX_VIEWS
    live_cams: set[str] = set()
    live_renders: set[str] = set()
    live_patches: set[str] = set()
    live_probes: set[str] = set()
    final_cloud: np.ndarray | None = None
    final_plane: list | None = None
    n_steps = 0
    n_planned = 0

    for k, s in enumerate(summary["steps"]):
        pid = s["pose_id"]
        rr_log.at(k)
        cloud = _load_step_cloud(steps_dir, pid)
        rr_log.log_cloud(cloud)
        n_steps += 1

        plan_file = variant_dir / f"step_{pid:03d}.json"
        if not plan_file.exists():
            rr_log.log_viewplan_meta(variant, step=pid, abstain=True,
                                     reason="no plan on disk for this step",
                                     n_views=0, coverage_area_frac=0.0)
            live_cams = _retire(live_cams, set())
            live_renders = _retire(live_renders, set())
            print(f"  step {k:>3}  pose {pid:03d}  no plan on disk")
            continue

        plan = json.loads(plan_file.read_text())
        params = plan.get("params", {}) or {}
        max_views = int(params.get("max_views", max_views))

        if plan.get("abstain"):
            rr_log.log_viewplan_meta(variant, step=pid, abstain=True,
                                     reason=str(plan.get("reason", "")),
                                     n_views=0, coverage_area_frac=0.0)
            live_cams = _retire(live_cams, set())
            live_renders = _retire(live_renders, set())
            print(f"  step {k:>3}  pose {pid:03d}  abstain: "
                 f"{plan.get('reason', '')}")
            continue

        V, F = _load_mesh(variant_dir, (plan.get("mesh") or {}).get("path", ""))
        if len(V) and len(F):
            rr_log.log_viewplan_mesh(V, F)

        npz = _load_samples(variant_dir, (plan.get("samples") or {}).get("path", ""))
        cutoff_deg = float(params.get("incidence_cutoff_deg", 60.0))
        if "points" in npz and "best_cos" in npz and len(npz["points"]):
            rr_log.log_viewplan_coverage(npz["points"], npz["best_cos"],
                                         float(np.cos(np.radians(cutoff_deg))))
        now_patches: set[str] = set()
        if "points" in npz and "covered_by" in npz and len(npz["points"]):
            now_patches = set(
                rr_log.log_viewplan_patches(npz["points"], npz["covered_by"]))
        live_patches = _retire(live_patches, now_patches)

        now_probes: set[str] = set()
        if probe > 0 and "points" in npz and "normals" in npz \
                and len(npz["points"]):
            # An honest draw, not farthest-point: the point of the probe
            # channel is seeing what area-uniform random sampling actually
            # looks like, clusters included. Seeded on the pose id so the
            # same step always shows the same probes.
            rng = np.random.default_rng(pid)
            ids = rng.choice(len(npz["points"]),
                             size=min(probe, len(npz["points"])),
                             replace=False)
            arrow = float(params.get("standoff_m", 0.10))
            now_probes = set(rr_log.log_viewplan_sample_probes(
                npz["points"], npz["normals"], np.sort(ids), arrow_m=arrow,
                cloud=cloud))
        live_probes = _retire(live_probes, now_probes)

        now_cams: set[str] = set()
        now_renders: set[str] = set()
        views = plan.get("views", []) or []
        for v in views:
            view_id = int(v["view_id"])
            slot = f"view_{view_id:02d}"
            T = np.array(v["T_base_cam"], dtype=float)
            label = (f"view {view_id}  n={v.get('n_covered', 0)}  "
                    f"inc {float(v.get('mean_incidence_deg', 0.0)):.0f} deg")
            img_path = None
            render_rel = v.get("render", "")
            if render_rel:
                candidate = variant_dir / render_rel
                if candidate.exists():
                    img_path = candidate
                    rr_log.log_viewplan_render(slot, candidate)
                    now_renders.add(f"{rr_log.RENDERS}/{slot}")
            path = rr_log.log_viewplan_camera(
                view_id, T, intr, label=label, image_path=img_path,
                color=rr_log.viewplan_view_color(view_id))
            now_cams.add(path)

        live_cams = _retire(live_cams, now_cams)
        live_renders = _retire(live_renders, now_renders)

        metrics = plan.get("metrics", {}) or {}
        rr_log.log_viewplan_meta(
            variant, step=pid, abstain=False,
            reason=str(plan.get("reason", "")),
            n_views=int(metrics.get("n_views", len(views))),
            coverage_area_frac=float(metrics.get("coverage_area_frac", 0.0)),
            weighted_coverage=float(metrics.get("weighted_coverage", 0.0)),
            mean_best_incidence_deg=float(
                metrics.get("mean_best_incidence_deg", 0.0)),
            seconds=float(metrics.get("seconds", 0.0)))

        final_cloud = cloud
        final_plane = plan.get("table_plane")
        n_planned += 1
        print(f"  step {k:>3}  pose {pid:03d}  views {len(views):>2}  "
             f"coverage {metrics.get('coverage_area_frac', 0.0):.3f}  "
             f"renders {len(now_renders)}")

    if n_steps == 0:
        raise SystemExit(f"{run}: no steps in {steps_dir}/summary.json")
    if n_planned == 0:
        print(f"{run}/{variant}: WARNING — 0 of {n_steps} steps had a "
             f"usable plan (all abstained or missing); the rrd still "
             f"builds with clouds only")

    if final_plane is not None and final_cloud is not None and len(final_cloud):
        rr_log.log_table(final_plane, [float(final_cloud[:, 0].mean()),
                                       float(final_cloud[:, 1].mean()), 0.0])

    render_slots = [f"view_{i:02d}" for i in range(max_views)]
    patch_paths = ([f"{rr_log.VP_PATCHES}/view_{i:02d}" for i in range(max_views)]
                   + [f"{rr_log.VP_PATCHES}/unowned"])
    bp = rr_log.viewplan_blueprint(run, variant, render_slots,
                                   patch_paths=patch_paths)
    out_dir = Path(out_dir) if out_dir else derived / run / "rrd"
    path = rr_log.save(out_dir / f"{run}-viewplan.rrd", bp, app_id)
    print(f"{run}/{variant}: {n_steps} steps ({n_planned} planned) -> "
         f"{path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def _retire(live: set[str], now: set[str]) -> set[str]:
    """Clear entity paths that were logged last step and are not this step.

    `rr_log.clear_primitives` is generic over any path list (it just issues
    `rr.Clear(recursive=True)`); reused as-is here for planned-camera and
    render-slot retirement — the exact diff pattern `record/rr/workspace.py`
    uses for fit primitives, applied to a schema that has nothing to do
    with fits.
    """
    rr_log.clear_primitives(live - now)
    return now


# --------------------------------------------------------------- the viewer
def _viewer_binary() -> str | None:
    import shutil
    cand = Path(sys.executable).parent / "rerun"
    if cand.exists():
        return str(cand)
    return shutil.which("rerun")


def _open_viewer(path: Path) -> None:
    binary = _viewer_binary()
    if binary is None:
        print("no rerun viewer on PATH — skip --open, or run "
             f"`rerun {path}` yourself")
        return
    files = [str(path)]
    rbl = path.with_suffix(".rbl")
    if rbl.exists():
        files.append(str(rbl))
    print(f"opening {' '.join(files)}")
    subprocess.Popen([binary, *files], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def workspace(run, variant=DEFAULT_VARIANT, out=None, open=False,
              probe=0) -> None:
    """Build `<run>-viewplan.rrd` for one variant; `--open` launches the viewer.

    run:     e.g. 2408-cup3.
    variant: viewplan variant name (default: base).
    out:     override the output DIRECTORY (default:
             vbti/inspection/data/derived/<run>/rrd).
    open:    open the built .rrd in the native viewer.
    probe:   log N raw surface samples per step as individual point+normal
             probe entities (sampler debug view).
    """
    path = build(run, variant, out, probe=int(probe))
    if open:
        _open_viewer(path)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(workspace)
