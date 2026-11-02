#!/usr/bin/env python3
"""Plan inspection views for a run's step clouds, offline.

Wraps `planner.plan_step` and nothing else — the research path and the
loop path are the same call, so a verdict formed here transfers. Mirrors
`inspection/fit/cli.py` flag-for-flag: same `steps_of`/`select` helpers,
same merge-never-truncate `scores.jsonl`, same top-level `_job` for
multiprocessing `spawn`.

Writes one `StepPlan` per (variant, step) to
`data/derived/<run>/viewplan/<variant>/step_NNN.json`, the mesh artifact to
`meshes/step_NNN.ply`, the sample sidecar to `samples/step_NNN.npz`, one
score row per (variant, step) to `data/derived/<run>/viewplan/<variant>/
scores.jsonl`, and — on a per-variant render cadence (roadmap section 3) —
per-view renders + a coverage panel to `renders/`.

Render cadence (never overridable per-step, only wholesale via
`--norender`): `base` (the primary hypothesis) renders at every milestone
step {0,2,5,8,12,16,20,25,last}; `binary`/`cos2`/`altmesh`/`continuity`
render at anchor steps {2,12,last} only; `cos75` never renders (it reuses
`base`'s incidence angles, so a render would just repeat `base`'s pictures
at a different gate).

After planning, every touched (run, variant) is **cross-evaluated**: each
step's plan is re-scored against the run's FINAL geometry (`xe_*` columns in
`scores.jsonl`). That is the curve the POC's core question actually needs —
the self-evaluated one moves in the wrong direction because its denominator,
the mesh, grows 2.8-3.4x over the run. See `planner.score_views`.

Canonical entrypoint (the one operational CLI):

    p inspection/viewplan/cli.py run   --run 2408-cup3 --steps all --variant base,binary --seed 0 --jobs 4
    p inspection/viewplan/cli.py run   --run 2408-cup3,2408-cup4,2408-cup5 --steps all --variant all
    p inspection/viewplan/cli.py show  --run 2408-cup3 --variant base --steps final
    p inspection/viewplan/cli.py variants
    p inspection/viewplan/cli.py crosseval --run 2408-cup3 --variant all
    p inspection/viewplan/cli.py budget --run 2408-cup3 --variant base
    p inspection/viewplan/cli.py curves --run 2408-cup3,2408-cup4,2408-cup5

Standalone still works, same subcommands: `p inspection/viewplan/cli.py run ...`
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from inspection.viewplan import contract, mesh as meshmod, planner, render

DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"

#: Roadmap section 3. `∩ available` is the load-bearing part — cup5 has no
#: step_029 on disk (documented quirk), so a literal `{...,29}` would render
#: nothing for its last step; the formula below always resolves against the
#: steps actually present.
MILESTONE_CANDIDATES = (0, 2, 5, 8, 12, 16, 20, 25)
ANCHOR_CANDIDATES = (2, 12)

#: Which variants render at which cadence. `base` is the roadmap's primary
#: `cos`; `cos75` reuses `base`'s angles under a different gate and earns no
#: pictures of its own (roadmap section 3).
RENDER_MILESTONE_VARIANTS = {"base"}
RENDER_ANCHOR_VARIANTS = {"binary", "cos2", "altmesh", "continuity"}

#: The variant whose final mesh + samples are the common yardstick for the
#: cross-evaluation. One reference for every variant of a run, or the
#: comparison would measure the meshers against each other (Q3) instead of
#: the plans.
REF_VARIANT = "base"

#: The roadmap's M5 stability threshold: 20 mm-equivalent of pose churn.
DELTA_STABLE_M = 0.02


def _csv(x) -> List[str]:
    """A comma-list argument, back to a list of strings, whatever fire made of it.

    Fire literal-evals every argument value: `--steps 0,5,10` arrives as the
    tuple (0, 5, 10), `--steps 5` as the int 5, while `--run 2408-cup3,2408-cup4`
    stays a string (not a valid Python literal). Every ex-comma-list flag goes
    through here before the existing split logic.
    """
    if isinstance(x, (tuple, list)):
        return [str(v).strip() for v in x if str(v).strip()]
    return [s.strip() for s in str(x).split(",") if s.strip()]


# --------------------------------------------------------------------------
# step selection (mirrors fit/cli.py exactly)
def steps_of(run: str) -> List[Path]:
    d = DERIVED / run / "steps"
    if not d.is_dir():
        raise SystemExit(f"no step clouds at {d}")
    return sorted(d.glob("step_*.ply"))


def step_ids(run: str) -> List[int]:
    return [int(p.stem.split("_")[1]) for p in steps_of(run)]


def select(paths: Sequence[Path], spec: str) -> List[Path]:
    if spec == "all":
        return list(paths)
    if spec == "final":
        return list(paths[-1:])
    want = {int(s) for s in spec.split(",") if s.strip()}
    by_n = {int(p.stem.split("_")[1]): p for p in paths}
    missing = sorted(want - set(by_n))
    if missing:
        raise SystemExit(f"steps {missing} not on disk (have 0..{max(by_n)})")
    return [by_n[n] for n in sorted(want)]


def milestone_steps(ids: Sequence[int]) -> set:
    ids = set(int(i) for i in ids)
    out = {s for s in MILESTONE_CANDIDATES if s in ids}
    if ids:
        out.add(max(ids))
    return out


def anchor_steps(ids: Sequence[int]) -> set:
    ids = set(int(i) for i in ids)
    out = {s for s in ANCHOR_CANDIDATES if s in ids}
    if ids:
        out.add(max(ids))
    return out


def wants_render(variant: str, step: int, milestones: set, anchors: set) -> bool:
    if variant in RENDER_MILESTONE_VARIANTS:
        return step in milestones
    if variant in RENDER_ANCHOR_VARIANTS:
        return step in anchors
    return False                                     # cos75 and unknowns


def load_cloud(path: Path) -> np.ndarray:
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=float)


def is_chained(variant: str) -> bool:
    """Does this variant's step k depend on step k-1's answer?

    True for `continuity`, whose whole point is the warm start. Chained
    variants must run in step order inside one worker; everything else stays
    embarrassingly parallel.
    """
    return float(planner.params_for(variant).get("continuity_bonus", 0.0)) > 0


def prev_poses(variant_dir: Path, step: int) -> Optional[np.ndarray]:
    """(k,4,4) poses of the latest planned step before `step`, or None.

    Reads the *latest planned* step rather than literally `step - 1`, so a
    warm start survives an abstain, a gap in `--steps`, and cup5's missing
    029 without inventing a pose.
    """
    prior = sorted(int(p.stem.split("_")[1])
                   for p in variant_dir.glob("step_[0-9][0-9][0-9].json"))
    for s in reversed([s for s in prior if s < int(step)]):
        try:
            d = json.loads((variant_dir / f"step_{s:03d}.json").read_text())
        except Exception:                                     # noqa: BLE001
            continue
        if d.get("abstain") or not d.get("views"):
            continue
        return np.asarray([v["T_base_cam"] for v in d["views"]], dtype=float)
    return None


# --------------------------------------------------------------------------
def _job(args) -> Dict[str, Any]:
    """One (run, step, variant) -> plan + artifacts + a score row.

    Top-level so multiprocessing `spawn` can pickle it. Writes everything to
    disk itself (mesh, npz, JSON, renders) rather than sending open3d
    objects back over IPC — only the small score row crosses the pipe.
    """
    run, path, variant, seed, do_render, milestones, anchors = args
    step = int(path.stem.split("_")[1])
    variant_dir = DERIVED / run / "viewplan" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    prev_T = prev_poses(variant_dir, step) if is_chained(variant) else None
    P = load_cloud(path)
    plan, m, arrays = planner.plan_step(P, run, step, variant, seed=seed,
                                        prev_T=prev_T)

    render_note = ""
    if not plan.abstain:
        meshmod.save(m, variant_dir / plan.mesh["path"])
        planner.save_arrays(arrays, variant_dir / plan.samples["path"])

        want = do_render and wants_render(variant, step, milestones, anchors)
        if want:
            intr = planner.session_intrinsics(run)
            plan_dict = contract.plan_to_json(plan)
            try:
                results = render.render_plan(variant_dir, plan_dict, intr)
            except Exception as exc:                          # noqa: BLE001
                results = []
                render_note = f"render failed: {type(exc).__name__}: {exc}"
            by_id = {v.view_id: v for v in plan.views}
            framed: Dict[str, float] = {}
            for r in results:
                if r["view_id"] == -1:
                    plan.metrics["coverage_render"] = r["render"]
                    continue
                v = by_id.get(r["view_id"])
                if v is not None:
                    v.render = r["render"]
                    v.frame_fill = float(r["frame_fill"])
                framed[str(r["view_id"])] = float(r["framed_frac"])
            if framed:
                vals = list(framed.values())
                plan.metrics["render_framed_frac"] = framed
                plan.metrics["render_framed_frac_min"] = min(vals)
                plan.metrics["render_framed_frac_mean"] = sum(vals) / len(vals)

    out_json = variant_dir / f"step_{step:03d}.json"
    out_json.write_text(json.dumps(contract.plan_to_json(plan), indent=1))

    row = dict(plan.metrics)
    row.update(run=run, step=step, variant=variant, seed=seed,
               abstain=plan.abstain, reason=plan.reason)
    if render_note:
        row["render_note"] = render_note
    return row


def _chain(jobs) -> List[Dict[str, Any]]:
    """A whole (run, variant) in step order, in one worker.

    `_job` reads the warm start off disk, so the only thing a chain has to
    guarantee is that step k-1 was written before step k starts. Top-level
    for the same `spawn` reason as `_job`.
    """
    return [_job(j) for j in sorted(jobs, key=lambda j: j[1].name)]


def _write_scores(run: str, variant: str, rows: List[Dict[str, Any]]) -> Path:
    out = DERIVED / run / "viewplan" / variant
    out.mkdir(parents=True, exist_ok=True)
    log = out / "scores.jsonl"
    # Merge, never truncate: `--steps 0,5,10` must not delete the rows a
    # previous `--steps all` earned. New rows win on their own step.
    merged: Dict[int, Dict[str, Any]] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                old = json.loads(line)
                merged[old["step"]] = old
    for r in rows:
        merged[r["step"]] = r
    ordered = [merged[s] for s in sorted(merged)]
    with log.open("w") as fh:
        for r in ordered:
            fh.write(json.dumps(r) + "\n")
    return log


# --------------------------------------------------------------------------
# cross-evaluation: the step-k plan, scored on the FINAL geometry.
#
# The metric the POC's core question needs and the only one whose curve is
# readable. Everything `plan_step` records is measured on the step's own
# mesh, and that mesh grows 2.8-3.4x over a run (cup3: 169 -> 468 cm2), so
# its coverage@60 *falls* from 0.99 on a 1.1k-point blob to 0.76 on the
# finished cup. Fixing the surface — one mesh, one sample set, one gate, for
# every step of every variant — turns "after which step is the 3D good
# enough" into a step number.
def _read_scores(variant_dir: Path) -> List[Dict[str, Any]]:
    log = variant_dir / "scores.jsonl"
    if not log.exists():
        return []
    return sorted((json.loads(l) for l in log.read_text().splitlines()
                   if l.strip()), key=lambda r: r["step"])


def _plan_poses(variant_dir: Path, step: int) -> np.ndarray:
    f = variant_dir / f"step_{step:03d}.json"
    if not f.exists():
        return np.empty((0, 4, 4))
    d = json.loads(f.read_text())
    return np.asarray([v["T_base_cam"] for v in d.get("views", []) or []],
                      dtype=float).reshape(-1, 4, 4)


def reference(run: str, ref_variant: str = REF_VARIANT) -> Optional[Dict[str, Any]]:
    """The final geometry of `ref_variant`, loaded once and reused.

    Returns the mesh, its sample set, the fraction of it that is measurable
    from *any* legal pose (the honest denominator), and the params that
    define the yardstick.
    """
    vdir = DERIVED / run / "viewplan" / ref_variant
    fitted = [r for r in _read_scores(vdir) if not r.get("abstain")]
    if not fitted:
        return None
    step = int(fitted[-1]["step"])
    mp_ = vdir / "meshes" / f"step_{step:03d}.ply"
    zp = vdir / "samples" / f"step_{step:03d}.npz"
    if not mp_.exists() or not zp.exists():
        return None
    z = np.load(zp)
    return {"mesh": meshmod.load(mp_), "points": z["points"],
            "normals": z["normals"], "step": step, "variant": ref_variant,
            "coverable": float(fitted[-1].get("coverable_area_frac", 1.0)),
            "params": planner.params_for(ref_variant)}


def k_star(rows: Sequence[Dict[str, Any]], key: str, thr: float) -> Optional[int]:
    """First step whose value clears `thr` and never drops below it again.

    Sustained, not first-touch: a curve that grazes a threshold at step 4 and
    falls back has not converged, and the POC is asking when it is safe to
    stop scanning.
    """
    vals = [(int(r["step"]), r.get(key)) for r in rows if r.get(key) is not None]
    for i, (s, _v) in enumerate(vals):
        if all(v >= thr for _s, v in vals[i:]):
            return s
    return None


def cross_eval(run: str, variant: str, ref: Optional[Dict[str, Any]] = None,
               ref_variant: str = REF_VARIANT) -> List[Dict[str, Any]]:
    """Re-score every stored step of (run, variant) on the final geometry.

    Writes the `xe_*` columns back into `scores.jsonl` and returns the rows.
    Falls back to the variant's own final geometry when the reference variant
    has not been planned; `xe_ref` always says which surface was used.
    """
    vdir = DERIVED / run / "viewplan" / variant
    rows = _read_scores(vdir)
    if not rows:
        return []
    ref = ref or reference(run, ref_variant) or reference(run, variant)
    if ref is None:
        return rows
    intr = planner.session_intrinsics(run)
    tag = f"{ref['variant']}/step_{ref['step']:03d}"
    for r in rows:
        T = np.empty((0, 4, 4)) if r.get("abstain") else \
            _plan_poses(vdir, int(r["step"]))
        s = planner.score_views(ref["mesh"], T, ref["points"], ref["normals"],
                                intr, ref["params"])
        r["xe_ref"] = tag
        r["xe_coverage"] = s["coverage"]
        r["xe_coverage_coverable"] = s["coverage"] / max(ref["coverable"], 1e-12)
        r["xe_weighted_coverage"] = s["weighted_coverage"]
        r["xe_mean_incidence_deg"] = s["mean_incidence_deg"]
        r["xe_p95_incidence_deg"] = s["p95_incidence_deg"]
    fitted = [r for r in rows if not r.get("abstain")]
    final = fitted[-1]["xe_coverage"] if fitted else 0.0
    for r in rows:
        r["xe_frac_of_final"] = (r["xe_coverage"] / final if final > 0 else
                                 float("nan"))
    _write_scores(run, variant, rows)
    return rows


def churn(variant_dir: Path, rows: Sequence[Dict[str, Any]]) -> List[float]:
    """M5 pose churn between consecutive planned steps (NaN where undefined).

    Distance-to-final says how far a plan is from the answer; this says
    whether the plan has *stopped moving*, which is the question a live
    "stop scanning now" rule would ask, and the one the shipped curves
    never asked.
    """
    out: List[float] = []
    prev = None
    for r in rows:
        if r.get("abstain"):
            out.append(np.nan)
            continue
        cur = _pose_rows(_plan_poses(variant_dir, int(r["step"])))
        out.append(np.nan if prev is None else _pose_set_distance(cur, prev))
        if len(cur):
            prev = cur
    return out


def cli_crosseval(run, variant="all", ref=REF_VARIANT) -> None:
    """Re-score stored plans against the final geometry.

    run:     run id, or a comma list.
    variant: comma list, or 'all'.
    ref:     variant whose final mesh is the common yardstick.
    """
    variants = planner.variants() if variant == "all" else _csv(variant)
    for r in _csv(run):
        geom = reference(r, ref)
        if geom is None:
            print(f"{r}: no reference geometry from variant {ref!r} "
                  f"— falling back to each variant's own final step")
        for v in variants:
            rows = cross_eval(r, v, geom, ref)
            if not rows:
                continue
            report_convergence(r, v, rows)


def report_convergence(run: str, variant: str,
                       rows: Sequence[Dict[str, Any]]) -> None:
    """The headline sentence for one (run, variant), printed from the rows.

    `self` and `mratio` ride along with the cross-evaluated number on purpose.
    They are the Q3 caveat, and the caveat has teeth: 2408-cup5's alpha mesh
    is 0.70x the cloud's own area estimate — it caps the cavity — and that
    missing surface makes `altmesh` self-report coverage@60 = 0.872 against
    `base`'s 0.631. Scored on ONE surface the same two plans are 0.627 and
    0.631, i.e. indistinguishable. A coverage figure printed without its
    denominator invites exactly that misreading.
    """
    fitted = [r for r in rows if not r.get("abstain")]
    if not fitted:
        print(f"{run}/{variant}: nothing planned")
        return
    fin = fitted[-1]
    k95 = k_star(fitted, "xe_frac_of_final", 0.95)
    k90 = k_star(fitted, "xe_frac_of_final", 0.90)
    vdir = DERIVED / run / "viewplan" / variant
    chur = churn(vdir, fitted)
    ch = [c for c in chur[-10:] if c == c]
    kstab = k_star([{"step": r["step"], "neg": -c}
                    for r, c in zip(fitted, chur) if c == c],
                   "neg", -DELTA_STABLE_M)
    print(f"{run}/{variant}: ref={fin.get('xe_ref', '-')} "
          f"final cov-on-final={fin.get('xe_coverage', float('nan')):.3f} "
          f"(of achievable {fin.get('xe_coverage_coverable', float('nan')):.3f}) "
          f"[self={fin.get('coverage_60', float('nan')):.3f} on its OWN mesh, "
          f"area/cloud={fin.get('mesh_area_ratio_cloud', float('nan')):.2f}] "
          f"| k*(90% of final plan)={k90} k*(95%)={k95} "
          f"| churn last10 mean={np.mean(ch) if ch else float('nan'):.3f}m "
          f"k*(churn<={DELTA_STABLE_M})={kstab}")


# --------------------------------------------------------------------------
def cli_run(run, steps="final", variant="base", seed=0, jobs=1,
            render=True, crosseval=True) -> None:
    """Plan and persist one StepPlan per (run, step, variant).

    run:       run id, or a comma list.
    steps:     all | final | 0,5,10.
    variant:   comma list, or 'all'.
    jobs:      worker processes.
    render:    `--norender` skips the render pass (fast iteration).
    crosseval: `--nocrosseval` skips scoring every step against the
               final geometry.
    """
    variants = planner.variants() if variant == "all" else _csv(variant)
    for v in variants:
        if v not in planner.VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; have {planner.variants()}")
    runs = _csv(run)
    steps = steps if steps in ("all", "final") else ",".join(_csv(steps))
    seed, jobs = int(seed), int(jobs)

    tasks = []
    per_run_ms: Dict[str, Tuple[set, set]] = {}
    for r in runs:
        ids = step_ids(r)
        per_run_ms[r] = (milestone_steps(ids), anchor_steps(ids))
        paths = select(steps_of(r), steps)
        ms, an = per_run_ms[r]
        for v in variants:
            for p in paths:
                tasks.append((r, p, v, seed, bool(render), ms, an))

    print(f"{len(tasks)} (run, step, variant) jobs, variants "
          f"{', '.join(variants)}, seed {seed}, {jobs} worker(s), "
          f"render={'on' if render else 'off'}")

    # `continuity` warm-starts from step k-1, so its steps cannot race each
    # other; every other variant stays one job per step.
    free = [j for j in tasks if not is_chained(j[2])]
    chains: Dict[Tuple[str, str], List] = {}
    for j in tasks:
        if is_chained(j[2]):
            chains.setdefault((j[0], j[2]), []).append(j)
    if chains:
        print(f"  ({len(chains)} chained (run, variant) group(s) run in step "
              f"order: {', '.join(sorted({k[1] for k in chains}))})")

    t0 = time.perf_counter()
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs) as pool:
            rows = pool.map(_job, free) if free else []
            for part in (pool.map(_chain, list(chains.values()))
                         if chains else []):
                rows.extend(part)
    else:
        rows = [_job(j) for j in free]
        for group in chains.values():
            rows.extend(_chain(group))
    wall = time.perf_counter() - t0

    per_run_variant: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        per_run_variant.setdefault((row["run"], row["variant"]), []).append(row)

    # `mratio` is mesh area / crude cloud area (M7). It rides next to every
    # coverage number on purpose: a mesher that deletes 30 % of the cavity
    # (cup5 alpha, ratio 0.70) reports a *better* coverage for a worse plan,
    # and a coverage column read without its denominator invites exactly that
    # misreading. `clr` is the closest any selected camera came to the cloud.
    print(f"\n{'run':>11} {'step':>5} {'variant':>10} {'nv':>3} {'term':>9} "
          f"{'cov60':>6} {'cov/able':>8} {'able':>6} {'mratio':>6} "
          f"{'clr_mm':>6} {'inc':>5} {'p95':>5} {'s':>6}  reason")
    for row in sorted(rows, key=lambda r: (r["run"], r["variant"], r["step"])):
        reason = row.get("reason", "")
        print(f"{row['run']:>11} {row['step']:>5} {row['variant']:>10} "
              f"{row.get('n_views', 0):>3} {row.get('terminator', '-'):>9} "
              f"{row.get('coverage_60', row.get('coverage_area_frac', 0.0)):>6.3f} "
              f"{row.get('coverage_coverable_frac', 0.0):>8.3f} "
              f"{row.get('coverable_area_frac', 0.0):>6.3f} "
              f"{row.get('mesh_area_ratio_cloud', float('nan')):>6.2f} "
              f"{1e3 * row.get('min_cam_clearance_m', 0.0):>6.1f} "
              f"{row.get('mean_best_incidence_deg', 0.0):>5.1f} "
              f"{row.get('p95_best_incidence_deg', 0.0):>5.1f} "
              f"{row.get('seconds', 0.0):>6.2f}  {reason[:60]}")

    print(f"\n{len(tasks)} jobs in {wall:.1f}s wall "
          f"({wall / max(len(tasks), 1):.3f}s/job avg across {jobs} worker(s))")

    for (r, v), rvrows in sorted(per_run_variant.items()):
        log = _write_scores(r, v, rvrows)
        ordered = sorted(rvrows, key=lambda r: r["step"])
        fitted = [r for r in ordered if not r["abstain"]]
        first = fitted[0]["step"] if fitted else None
        # the roadmap's honesty rule (research.md §E4): raw coverage@60 of
        # TOTAL mesh area rarely reaches 95% on these objects; report both
        # the raw first-95% step and the achievable-normalized one, no
        # picking a winner.
        first95_raw = next((r["step"] for r in fitted
                            if r.get("coverage_60", r.get("coverage_area_frac", 0))
                            >= 0.95), None)
        first95_able = next((r["step"] for r in fitted
                             if r.get("coverage_coverable_frac", 0) >= 0.95), None)
        final = ordered[-1] if ordered else None
        print(f"\n{r}/{v}: {len(fitted)}/{len(ordered)} steps planned, "
              f"first at step {first}, "
              f"self-scored first step >=95% raw: {first95_raw}, "
              f">=95% of achievable: {first95_able} "
              f"(both self-scored on the step's OWN mesh — see the xe_ columns), "
              f"final n_views={final.get('n_views') if final else '-'} "
              f"cov60={final.get('coverage_60', final.get('coverage_area_frac', 0)) if final else float('nan'):.3f} "
              f"-> {log}")

    if crosseval:
        print("\ncross-evaluation — every step's plan scored on the FINAL "
              "geometry (the question the self-scored curve cannot answer):")
        for r in sorted({rn for rn, _v in per_run_variant}):
            geom = reference(r, REF_VARIANT)
            if geom is None:
                print(f"  {r}: no {REF_VARIANT!r} reference on disk — "
                      f"each variant is scored on its own final geometry")
            for (r2, v2) in sorted(per_run_variant):
                if r2 != r:
                    continue
                rows2 = cross_eval(r, v2, geom, REF_VARIANT)
                if rows2:
                    report_convergence(r, v2, rows2)


def cli_show(run, variant="base", steps="final") -> None:
    """Print a stored plan, one block per selected step.

    steps: all | final | 0,5,10.
    """
    steps = steps if steps in ("all", "final") else ",".join(_csv(steps))
    variant_dir = DERIVED / run / "viewplan" / variant
    for path in select(steps_of(run), steps):
        f = variant_dir / f"{path.stem}.json"
        if not f.exists():
            print(f"{path.stem}: no plan on disk")
            continue
        plan = contract.plan_from_json(json.loads(f.read_text()))
        print(f"{path.stem} {plan.variant} {plan.version}")
        print(f"  abstain={plan.abstain} reason={plan.reason!r}")
        m = plan.metrics
        print(f"  n_views={plan.n_views} coverage_60="
              f"{m.get('coverage_60', m.get('coverage_area_frac', 0.0)):.3f} "
              f"coverage_coverable={m.get('coverage_coverable_frac', 0.0):.3f} "
              f"mean_incidence={m.get('mean_best_incidence_deg', 0.0):.1f}deg "
              f"seconds={m.get('seconds', 0.0):.2f}")
        # the two numbers that decide whether the line above means anything:
        # which mesh it was measured on (M7), and how close the nearest
        # selected camera came to the object.
        print(f"  mesh={plan.mesh.get('method', '?')} "
              f"area={1e4 * m.get('mesh_area_m2', 0.0):.1f}cm2 "
              f"area/cloud={m.get('mesh_area_ratio_cloud', float('nan')):.2f} "
              f"coverable={m.get('coverable_area_frac', 0.0):.3f} "
              f"min_cam_clearance={1e3 * m.get('min_cam_clearance_m', 0.0):.0f}mm")
        for v in plan.views:
            pos = np.round(v.position, 4).tolist()
            print(f"  - view {v.view_id:>2}  pos={pos}  standoff={v.standoff_m:.3f}m "
                  f"n_covered={v.n_covered:<4} area={v.area_m2 * 1e4:6.1f}cm2 "
                  f"inc(mean/p95)={v.mean_incidence_deg:5.1f}/"
                  f"{v.p95_incidence_deg:5.1f}deg cum={v.cum_area_frac:.3f} "
                  f"render={v.render or '-'}")


def cli_variants() -> None:
    """Registered variants, their versions, and their parameters."""
    for v in planner.variants():
        print(f"{v:<8} {planner.version_of(v)}")
        for k, val in planner.params_for(v).items():
            print(f"    {k}: {val}")


# --------------------------------------------------------------------------
# comparison curves — n_views / coverage / mean incidence / stability vs step,
# one line per variant. Mirrors `fit/render.py`'s `curves()` shape (matplotlib
# Agg, one PNG per run) but adds the plan-stability curve (roadmap M5): the
# symmetric Chamfer distance of each step's view *set* to the final plan's,
# d(v,w) = ||pos_v - pos_w|| + 0.1 * arccos(boresight_v . boresight_w).
LAMBDA_ROT_M_PER_RAD = 0.1


def _pose_rows(T: np.ndarray) -> np.ndarray:
    """(N, 6) rows of [position(3), boresight(3)] from a (N,4,4) pose stack."""
    T = np.asarray(T, dtype=float).reshape(-1, 4, 4)
    if not len(T):
        return np.empty((0, 6))
    return np.concatenate([T[:, :3, 3], T[:, :3, 2]], axis=1)


def _view_set(plan_dict: Dict[str, Any]) -> np.ndarray:
    """(N, 6) rows of [position(3), boresight(3)] for a stored plan's views."""
    return _pose_rows(np.asarray(
        [v["T_base_cam"] for v in plan_dict.get("views", []) or []],
        dtype=float).reshape(-1, 4, 4))


def _pose_set_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Chamfer M5 distance between two view sets; 0.0 if either empty
    matches the other only when both are empty (an honest "no plan" state)."""
    if len(A) == 0 and len(B) == 0:
        return 0.0
    if len(A) == 0 or len(B) == 0:
        return float("nan")

    def pair(X, Y):
        d_pos = np.linalg.norm(X[:, None, :3] - Y[None, :, :3], axis=2)
        cos = np.clip(np.einsum("ik,jk->ij", X[:, 3:], Y[:, 3:]), -1.0, 1.0)
        d_rot = np.arccos(cos)
        return d_pos + LAMBDA_ROT_M_PER_RAD * d_rot

    D = pair(A, B)
    return float(0.5 * (D.min(1).mean() + D.min(0).mean()))


def curves(run: str, variants: Sequence[str]) -> Path:
    """Top row: the convergence answer. Bottom row: the caveats.

    The top row is measured on ONE surface (the reference run's final mesh),
    so it rises and a threshold crossing means something. The bottom row is
    self-scored and is there precisely because it must not be read alone:
    `coverage@60` falls over a run, and the panel next to it — mesh area —
    is why. `area/cloud` is the M7 mesher sanity ratio; an `altmesh` line
    sitting at 0.70 is a mesh missing a third of the cavity, not a plan
    covering more of it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(19, 7.6))
    ref_tag = ""
    for v in variants:
        vdir = DERIVED / run / "viewplan" / v
        rows = _read_scores(vdir)
        if not rows:
            continue

        def col(key, default=np.nan):
            return [r.get(key, default) if not r["abstain"] else np.nan
                    for r in rows]

        s = [r["step"] for r in rows]
        ref_tag = next((r.get("xe_ref", "") for r in reversed(rows)
                        if r.get("xe_ref")), ref_tag)
        fitted = [r for r in rows if not r["abstain"]]
        k95 = k_star(fitted, "xe_frac_of_final", 0.95)

        ln, = axes[0, 0].plot(s, col("xe_coverage"), marker="o", ms=3, label=v)
        c = ln.get_color()
        axes[0, 1].plot(s, col("xe_frac_of_final"), marker="o", ms=3, color=c,
                        label=f"{v} (k*={k95})")
        if k95 is not None:
            axes[0, 1].axvline(k95, color=c, ls=":", lw=1)
        axes[0, 2].plot(s, col("xe_mean_incidence_deg"), marker="o", ms=3,
                        color=c, label=v)
        axes[0, 3].plot([r["step"] for r in fitted], churn(vdir, fitted),
                        marker="o", ms=3, color=c, label=v)

        self_cov = [r.get("coverage_60", r.get("coverage_area_frac", np.nan))
                    if not r["abstain"] else np.nan for r in rows]
        axes[1, 0].plot(s, self_cov, marker="o", ms=3, color=c, label=v)
        axes[1, 1].plot(s, [1e4 * x if x == x else np.nan
                            for x in col("mesh_area_m2")],
                        marker="o", ms=3, color=c, label=v)
        axes[1, 2].plot(s, col("mesh_area_ratio_cloud"), marker="o", ms=3,
                        color=c, label=v)
        axes[1, 3].plot(s, col("n_views"), marker="o", ms=3, color=c, label=v)

    axes[0, 1].axhline(0.95, color="k", ls="--", lw=1)
    axes[0, 3].axhline(DELTA_STABLE_M, color="k", ls="--", lw=1)
    axes[1, 2].axhline(1.0, color="k", ls="--", lw=1)
    titles = [(f"coverage of FINAL surface  [ref {ref_tag or '?'}]",
               "fraction of final-plan coverage (k* = sustained >=0.95)",
               "mean incidence on final surface (deg)",
               f"consecutive plan churn (M5, m) — target {DELTA_STABLE_M}"),
              ("SELF-scored coverage@60 (denominator grows!)",
               "mesh area (cm2) — the growing denominator",
               "mesh area / cloud area (M7 sanity)", "n_views")]
    for i in range(2):
        for j in range(4):
            axes[i, j].set_xlabel("step")
            axes[i, j].set_title(titles[i][j], fontsize=9)
            axes[i, j].grid(alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].set_ylim(0, 1.05)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 3].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"{run} — viewplan: convergence on a fixed surface (top) "
                 f"vs self-scored (bottom)", fontsize=12)
    out_dir = DERIVED / run / "viewplan"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "curves.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(dst, dpi=130)
    plt.close(fig)
    return dst


# --------------------------------------------------------------------------
# view budget — how many views 95 % of the achievable surface actually costs.
# The success bar asks for >=95 % at <=12 views; the greedy terminates on the
# cap in every (run, variant), so "12" is an assumption the data never got to
# argue with. This runs the SAME greedy with the cap lifted, on the same
# stored geometry, and reports the price of each target.
def budget(run: str, variant: str, targets: Sequence[float] = (0.90, 0.95, 0.99),
           cap: int = 60) -> Dict[str, Any]:
    vdir = DERIVED / run / "viewplan" / variant
    fitted = [r for r in _read_scores(vdir) if not r.get("abstain")]
    if not fitted:
        raise SystemExit(f"no planned step for {run}/{variant}")
    step = int(fitted[-1]["step"])
    m = meshmod.load(vdir / "meshes" / f"step_{step:03d}.ply")
    z = np.load(vdir / "samples" / f"step_{step:03d}.npz")
    P, N, area = z["points"], z["normals"], float(z["area"].mean())
    plan = json.loads((vdir / f"step_{step:03d}.json").read_text())
    params = planner.params_for(variant)
    intr = planner.session_intrinsics(run)
    cloud = load_cloud(DERIVED / run / "steps" / f"step_{step:03d}.ply")
    T, _anch, _info = planner.candidate_views(P, N, params, plan["table_plane"],
                                              cloud=cloud)
    cover, cosv, _gate = planner.visibility(m, T, P, N, intr, params)
    coverable = float(cover.any(0).mean())
    picks, term = planner.greedy(cover, cosv, area,
                                 dict(params, max_views=int(cap),
                                      min_gain_frac=0.0))
    got = np.zeros(len(P), dtype=bool)
    need: Dict[str, Optional[int]] = {f"{t:.2f}": None for t in targets}
    at_cap = {}
    for i, pk in enumerate(picks, 1):
        got |= cover[pk["view_idx"]]
        frac = got.mean() / max(coverable, 1e-12)
        for t in targets:
            if need[f"{t:.2f}"] is None and frac >= t:
                need[f"{t:.2f}"] = i
        if i in (6, 12, 18, 24):
            at_cap[i] = float(got.mean()), float(frac)
    return {"run": run, "variant": variant, "step": step,
            "coverable_area_frac": coverable, "n_uncapped": len(picks),
            "terminator": term, "views_needed": need, "at_n_views": at_cap,
            "final_coverage_raw": float(got.mean()),
            "final_coverage_coverable": float(got.mean() / max(coverable, 1e-12))}


def cli_budget(run, variant="base", cap=60) -> None:
    """Views needed for 90/95/99% of the achievable surface.

    run:     run id, or a comma list.
    variant: comma list, or 'all'.
    cap:     uncapped-greedy ceiling.
    """
    variants = planner.variants() if variant == "all" else _csv(variant)
    for r in _csv(run):
        for v in variants:
            b = budget(r, v, cap=int(cap))
            print(f"\n{r}/{v} step {b['step']}: coverable="
                  f"{b['coverable_area_frac']:.3f}, uncapped greedy took "
                  f"{b['n_uncapped']} views (stopped on {b['terminator']}), "
                  f"reaching {b['final_coverage_coverable']:.3f} of achievable")
            for t, n in b["views_needed"].items():
                print(f"    {float(t) * 100:.0f}% of achievable: "
                      f"{n if n is not None else '> ' + str(b['n_uncapped'])} views")
            for n, (raw, able) in sorted(b["at_n_views"].items()):
                print(f"    at {n:>2} views: raw={raw:.3f} of-achievable={able:.3f}")


def cli_curves(run, variant="all") -> None:
    """Comparison curves vs step, one PNG per run.

    run:     run id, or a comma list.
    variant: comma list, or 'all'.
    """
    variants = planner.variants() if variant == "all" else _csv(variant)
    for r in _csv(run):
        print(curves(r, variants))


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"run": cli_run, "show": cli_show, "variants": cli_variants,
               "crosseval": cli_crosseval, "budget": cli_budget,
               "curves": cli_curves})
