#!/usr/bin/env python3
"""Replay bench for ray views — the Stage A gate (design:
inspection/docs/2026-09-17-ray-views-design.md).

What it answers, offline, before any of this touches the machine:

  1. For every view a recorded run actually executed, would the ray sampler
     have put the camera CLOSER to the object surface than the formula did?
  2. Do the side faces come out fronto-parallel (theta = 0, small +z shift)
     instead of the 35-42 deg keystone the 1709 demo produced?
  3. Do the paper timings in the design's feasibility table survive contact
     with the real IK + coal stack (~6 ms/ray, ~230 ms/36-cell refresh,
     <1 ms slide resolve)?

Method. The run's LAST boxguard `decomposition.json` is the belief the
comparison is judged under: its box is both the ray geometry (cell rays
anchor at its centre, face rays on its walls) and — inflated by `--margin`
— the hard obstacle in a fresh `RobotCell` (cell.yaml keepouts come free).
That single belief is deliberately used for every step, including the early
ones recorded while the cloud was still growing: the question is not "what
did the machine know at step 3" but "given what it ended up believing, where
should the camera have gone".

Old endpoint = the recorded `T_base_cam` translation, i.e. where the arm
really was. New endpoint = `resolve_cell` / `resolve_face`. Old face theta is
read from the run's own `view_state.json` (`scores.off_angle_deg`), so the
tilt column is the run's claim about itself, not a recomputation.

TWO DISTANCES, never mixed (see the note above `surface_dist`): `d_surf` is
the Euclidean closest-point distance to the oriented box surface — the
comparison metric, the one number cells and faces share. `along` is the
along-ray travel from where the ray leaves the object (`depth - clear_m` for
cells, `depth` for faces) — the quantity the D405's 0.105 m floor actually
constrains. Every run prints the floor as an invariant on `along`, because a
d_surf of 0.09 on an oblique cell ray is a corner clearance, not a violated
sensor minimum.

Timing rows are compared against the workload each estimate was written for:
`one ray validity` is quoted both raw (with its measured depth count) and
normalised to the design's ~18 depths, and the face-family worst case is
split side (~68 members) vs top (~145) instead of reporting the larger one
as if the design had estimated it. There is no slide-in row: the leg's target
is `samples[0]`, and timing a list index would be evidence for nothing.

`--margin` exists because the live machine's obstacle is NOT this box:
`machine._recenter` installs `geometry.object_box`, the pct-trimmed cloud
AABB inflated 2 cm per side, and coal then pads 2 cm more. The brief's world
is the raw decomposition box (margin 0), which is 2 cm per side more
optimistic than the run's own world; `--margin 0.04` reproduces the live
inflation, and the two numbers bracket the truth.

Stage B, `--world guard` (Anton's ruling 2026-09-17): live sampling must
validate against the world the motion planner actually collision-checks
against — `geometry.object_box(fused cloud)`, installed exactly as
`machine._recenter` does, not the boxguard belief box above. `--world
boxguard` (default) is the unchanged Stage A world; `--margin` only applies
there. Ray geometry (addresses, aabb, `box_corners`, the `d_surf` surface)
stays the boxguard prim in BOTH modes — only the collision world differs.

    p inspection/investigation/ray_views/ray_replay.py --run 1709-cam-ai-demo1
    p inspection/investigation/ray_views/ray_replay.py --run 1509-cam-ai_bench5 --margin 0.04
    p inspection/investigation/ray_views/ray_replay.py --run all
    p inspection/investigation/ray_views/ray_replay.py --run all --world guard

Writes /tmp/ray_views/<run>/comparison.png (+ .../timings.txt) and prints
the per-view table plus the measured-vs-estimated timing table. It never
writes a byte into inspection/data/runs/.
"""
import time
from pathlib import Path

import numpy as np

from inspection.cell.geometry import object_box
from inspection.cell.world import RobotCell
from inspection.fit.contract import Primitive
from inspection.motion.ik import UR5eIK
from inspection.record.catalog import DATASETS_ROOT, RUNS_ROOT
from inspection.record.run import Run
from inspection.view.faceview import FACES
from inspection.view.grid import H_BINS, R_MAX, R_MIN, V_ELEVATIONS
from inspection.view.rays import (MIN_RANGE_M, RAY_STEP_M, belief_bounds,
                                  box_corners, cell_clear_m, cell_ray,
                                  face_family, resolve_cell, resolve_face,
                                  sample_ray)
from inspection.view.viewsphere import T_FLANGE_CAM

OUT = Path("/tmp/ray_views")
#: the run this design came out of — always benched first.
PINNED_RUN = "1709-cam-ai-demo1"
#: explicit extra: the flat LAN block, the object the design names as the
#: shift rung's motivating case. Not a "most recent bench" pick.
EXTRA_RUNS = ("1509-lan-ai_bench5",)
#: design doc, "Feasibility estimates" — the numbers Stage A is measuring.
#: `full cone` is quoted for ~8 + 144 = 152 rays, so a family with a
#: different member count gets the estimate pro-rated by member count
#: (see `timing_table`) rather than compared to a workload it is not.
ESTIMATES = {"one ray validity": 6.0, "36-cell refresh": 230.0,
             "face family, shift rung": 20.0, "face family, full cone": 1000.0,
             "full cone rays": 152.0}
#: the design's ~18 depths per ray — the workload its 6 ms/ray assumed.
EST_DEPTHS = 18


def newest_bench_runs(n=2):
    """The `n` most recently written `*bench*` runs, newest first.

    The brief says "the two most recent bench runs that exist", so they are
    discovered by mtime at call time rather than frozen into a constant that
    silently rots the next time a bench run lands.
    """
    cands = [d for d in RUNS_ROOT.iterdir()
             if d.is_dir() and "bench" in d.name and (d / "run.json").exists()]
    cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in cands[:n]]


def default_runs():
    """Pinned demo, the two newest bench runs, then the explicit extra."""
    out = [PINNED_RUN] + [r for r in newest_bench_runs(2) if r != PINNED_RUN]
    return out + [r for r in EXTRA_RUNS if r not in out]


def run_dir_for(run_id: str) -> Path:
    """Resolve an ID across both archive roots, runs/ first — the same
    resolution `p -m inspection.record` uses."""
    for root in (RUNS_ROOT, DATASETS_ROOT):
        d = root / run_id
        if (d / "run.json").exists():
            return d
    raise SystemExit(f"run {run_id!r} not found under {RUNS_ROOT} "
                     f"or {DATASETS_ROOT}")


# ----------------------------------------------------------------- the belief
def last_box(run: Run):
    """(step_id, Primitive) of the latest boxguard box the run recorded.

    Walks back from the last step: a run whose final step crashed before
    fitting still has an earlier belief, and that is the honest one to use.
    None when the run never fitted a box — the caller skips such a run
    rather than inventing geometry for it.
    """
    for step in reversed(run.steps):
        dec = step.decomposition
        if dec is None or dec.abstain or not dec.primitives:
            continue
        if dec.method != "boxguard":
            continue
        p = dec.primitives[0]
        if p.kind != "box":
            continue
        return step.id, Primitive(kind="box", pose=p.pose,
                                  params={"extents": p.params["extents"]},
                                  label=p.label)
    return None


def box_world(prim: Primitive, margin: float, cell_yaml=None) -> RobotCell:
    """Fresh cell with the object installed the way `machine._recenter` does
    — dims + a yaw-only pose, `margin` added to every dimension (half per
    side). The ray geometry keeps the UNPADDED prim: padding is the safety
    number, not a change of belief.

    `cell_yaml` overrides the keepout world. It exists because cell.yaml's
    corner2 apex was moved by hand on 2026-09-17 12:20 (commit c2f2e77),
    AFTER all four benched runs were recorded — so the default world is not
    the world they executed under, and `--cell_yaml <pre-move copy>` is how
    that difference gets measured instead of assumed.
    """
    world = RobotCell() if cell_yaml is None else RobotCell(yaml_path=cell_yaml)
    dims = (np.asarray(prim.params["extents"], float) + float(margin)).tolist()
    yaw = float(np.arctan2(prim.axes[1, 0], prim.axes[0, 0]))
    world.set_object("object", dims,
                     list(map(float, prim.center)) + [0.0, 0.0, yaw])
    return world


def guard_world(points: np.ndarray, cell_yaml=None):
    """(RobotCell, dims, pose6) with the object installed EXACTLY the way
    `machine._recenter` installs it — `geometry.object_box(points)` ->
    `world.set_object("object", dims, pose6)` — the world the live planner
    actually collision-checks the arm against.

    Anton's ruling 2026-09-17: live sampling must validate against that
    world, not the boxguard belief box `box_world` uses. `object_box`
    carries its OWN inflation (pct-trimmed cloud AABB + 2 cm/side, 5 cm
    floor) — there is no separate `margin` knob here, unlike `box_world`;
    passing one would double-pad relative to what `_recenter` actually
    installs. Ray geometry (addresses, aabb, `box_corners`) stays on the
    UNCHANGED boxguard prim — only the collision world changes; see the
    brief (task-5, ray-views Stage B).
    """
    world = RobotCell() if cell_yaml is None else RobotCell(yaml_path=cell_yaml)
    dims, pose = object_box(np.asarray(points, dtype=float))
    world.set_object("object", dims.tolist(), pose, parent="base")
    return world, dims, pose


def box_aabb(prim: Primitive):
    """Axis-aligned bounds of the ORIENTED box, padded 0 — `resolve_cell`'s
    slab test only needs the anchor->surface run, and over-padding it would
    push every cell ray's sensor floor outward.

    Delegates to `rays.belief_bounds`, the one reduction the live machine
    and the sweep also call: a bench that hulls the prim its own way is a
    bench measuring a belief the machine does not hold.
    """
    return belief_bounds(prim)[0]


#: Two distances appear in every table and they are NOT the same number:
#:
#:   d_surf  Euclidean closest-point distance to the oriented box surface.
#:           The comparison metric — the only one cells and faces share, and
#:           the one "get as much object as possible in the frame" is about.
#:   along   distance along the ray from where the ray leaves the object:
#:           `depth - clear_m` for a cell ray (whose anchor is the centroid,
#:           inside the box), `depth` for a face ray (anchor on the wall).
#:           This is the quantity the D405 floor applies to, so
#:           `along >= MIN_RANGE_M` is the invariant, NOT `d_surf >= 0.105`.
#:
#: A cell endpoint with d_surf 0.09 and along 0.11 is not a violated sensor
#: floor: it is an oblique ray whose 0.11 m of travel clears the box corner
#: by 0.09 m. `invariant_line` states the check on the right quantity.
def surface_dist(prim: Primitive, point) -> float:
    """Distance from `point` to the oriented box's SURFACE [m]; 0 inside.

    Exact exterior distance: in the box frame the closest-point offset per
    axis is max(|p| - half, 0), and its norm is the distance.
    """
    local = prim.axes.T @ (np.asarray(point, float) - prim.center)
    half = np.asarray(prim.params["extents"], float) / 2.0
    return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))


def n_depths(clear_m: float) -> int:
    """How many depths `sample_ray` iterates for this clear_m — the real
    workload behind a "one ray validity" measurement. The design's 6 ms
    assumed ~18; a cell ray whose anchor sits inside the box spends most of
    its budget before MIN_RANGE_M and carries far fewer."""
    d0 = float(np.ceil((clear_m + MIN_RANGE_M) / RAY_STEP_M)) * RAY_STEP_M
    return int(len(np.arange(d0, R_MAX + 1e-9, RAY_STEP_M)))


def recorded_theta(step):
    """The run's own off-axis angle for the address this step executed [deg],
    or None — `view_state.candidates[*].scores.off_angle_deg`, written by the
    face resolution ladder that produced the pose."""
    vs = step.view_state
    if vs is None or step.record.view.address is None:
        return None
    addr = list(step.record.view.address)
    for c in vs.candidates:
        if c.address is not None and list(c.address) == addr:
            v = (c.scores or {}).get("off_angle_deg")
            return None if v is None else float(v)
    return None


# ------------------------------------------------------------------ resolution

def resolve_step(world, ik, prim, aabb, step, fy, height_px):
    """(res, ms) for one executed step's address, or (None, ms) when the ray
    (family) has no provably-safe sample — an honestly inaccessible address."""
    view = step.record.view
    addr = view.address
    t0 = time.perf_counter()
    if view.method == "face":
        res = resolve_face(world, ik, T_FLANGE_CAM, prim, addr[1], fy,
                           height_px)
    else:
        h, v = int(addr[0]), int(addr[1])
        res = resolve_cell(world, ik, T_FLANGE_CAM, prim.center, aabb,
                           h, v, fy, height_px, corners=box_corners(prim))
    return res, (time.perf_counter() - t0) * 1e3


def addr_label(view) -> str:
    a = view.address
    if a is None:
        return f"{view.method}/none"
    if view.method == "face":
        return f"face {a[1]}"
    return f"cell {int(a[0])},{int(a[1])}"


def walk(run: Run, world, ik, prim, aabb, fy, height_px):
    """One row per executed, addressed step: old vs new endpoint."""
    rows = []
    for step in run.steps:
        if step.record.outcome != "captured" or step.T_base_cam is None:
            continue
        if step.record.view.address is None:
            continue                      # survey: a fixed pose, not an address
        old = step.T_base_cam[:3, 3]
        res, ms = resolve_step(world, ik, prim, aabb, step, fy, height_px)
        d_old = surface_dist(prim, old)
        d_new = None if res is None else surface_dist(prim, res.T_cam[:3, 3])
        # along-ray travel from where the ray leaves the object: for a cell
        # the centroid anchor is inside the box, so clear_m of the depth is
        # spent getting out of it; a face anchor already sits on the wall.
        clear = 0.0
        if res is not None and step.record.view.method != "face":
            a = step.record.view.address
            clear = cell_clear_m(aabb, cell_ray(prim.center, int(a[0]),
                                                int(a[1])))
        rows.append({
            "step": step.id, "label": addr_label(step.record.view),
            "kind": "face" if step.record.view.method == "face" else "cell",
            "clear": clear,
            "along": None if res is None else res.depth - clear,
            "old": old, "new": None if res is None else res.T_cam[:3, 3],
            "d_old": d_old, "d_new": d_new,
            "theta_old": recorded_theta(step),
            "theta_new": None if res is None else res.theta_deg,
            "shift": None if res is None else res.shift,
            "depth": None if res is None else res.depth,
            "roll": None if res is None else np.degrees(res.roll),
            "n_samples": 0 if res is None else len(res.samples),
            "ms": ms,
            "verdict": ("unreachable" if res is None else
                        "closer" if d_new < d_old - 1e-4 else
                        "farther" if d_new > d_old + 1e-4 else "same"),
        })
    return rows


def family_scan(world, ik, prim, face, fy, height_px):
    """(ms, flat, best) with EVERY family member sampled, not just up to the
    first hit. Two things only a full scan can say:

      * the honest worst case for the design table's "face family, worst
        (full cone)" row — `resolve_face` stops early, so its timing never
        exercises all ~68 members;
      * the closest safe sample the family contains ANYWHERE, which is the
        price of the settled ladder order ("shift ascending before depth"):
        when member k>0 holds a much closer sample than member 0, the
        predictable ladder is paying real centimetres for predictability.

    Two bests, because they answer different questions: `flat` is the closest
    FRONTO-PARALLEL member (theta = 0 — the rung the design wants to serve
    side faces with), `any` is the closest member of any kind, which on a
    tilt-friendly face is always a 60 deg cone ray and therefore not a real
    alternative. Each is (surface distance, shift, theta, depth) or None.
    """
    t0 = time.perf_counter()
    flat = best = None
    for ray, shift, theta, d_min in face_family(prim, face, fy, height_px):
        got = sample_ray(world, ik, T_FLANGE_CAM, ray,
                         clear_m=max(0.0, d_min - MIN_RANGE_M))
        if not got:
            continue
        cand = (surface_dist(prim, got[0].T_cam[:3, 3]), float(shift),
                float(theta), float(got[0].depth))
        if best is None or cand[0] < best[0]:
            best = cand
        if theta == 0.0 and (flat is None or cand[0] < flat[0]):
            flat = cand
    return (time.perf_counter() - t0) * 1e3, flat, best


# --------------------------------------------------------------------- timings
def measure(world, ik, prim, aabb, fy, height_px, repeats=3):
    """The design's feasibility table, measured. Warmed up first: the very
    first IK/coal call carries import-time and data-alloc cost that would
    otherwise be charged to one unlucky ray."""
    resolve_cell(world, ik, T_FLANGE_CAM, prim.center, aabb, 0, 1,
                 fy, height_px, corners=box_corners(prim))   # warmup
    refreshes, ray_ms, depths = [], [], []
    for rep in range(repeats):
        t0 = time.perf_counter()
        for h in range(H_BINS):
            for v in range(len(V_ELEVATIONS)):
                t1 = time.perf_counter()
                resolve_cell(world, ik, T_FLANGE_CAM, prim.center,
                             aabb, h, v, fy, height_px,
                             corners=box_corners(prim))
                ray_ms.append((time.perf_counter() - t1) * 1e3)
                if rep == 0:
                    depths.append(n_depths(
                        cell_clear_m(aabb, cell_ray(prim.center, h, v))))
        refreshes.append((time.perf_counter() - t0) * 1e3)
    faces, scans, members = {}, {}, {}
    for face in FACES:
        members[face] = len(face_family(prim, face, fy, height_px))
        t0 = time.perf_counter()
        res = resolve_face(world, ik, T_FLANGE_CAM, prim, face, fy, height_px)
        faces[face] = ((time.perf_counter() - t0) * 1e3, res)
        scans[face] = family_scan(world, ik, prim, face, fy, height_px)
    # No slide-in row: the leg's target is `samples[0]`, a list index, and
    # timing an index would dress up a 13 ns lookup as evidence for the
    # design's "<1 ms segment validation". The thing that claim is about —
    # validating the swept segment — does not exist until Stage B.
    return {"ray_ms": ray_ms, "depths": depths, "refresh_ms": refreshes,
            "faces": faces, "scans": scans, "members": members}


def timing_table(m):
    """Measured vs estimated, as printable lines — each row compared against
    the workload the estimate was written for, not a different one:

      * `one ray validity` is quoted twice. A real cell ray carries far
        fewer than the design's ~18 depths (its anchor is the centroid, so
        `clear_m` eats the near half of the range), so the raw ms/ray is
        reported WITH its measured depth count, and then normalised
        per-depth up to 18 depths — the second line is the number the 6 ms
        estimate should actually be judged against.
      * the face-family worst case is split side vs top. The design quoted
        ~8 + 144 = 152 rays; a side family here is ~68 members and the top
        family ~145, so the side row's estimate is pro-rated by member
        count instead of being measured against a 2x larger workload.
      * `face family, shift rung` averages the SIDE faces only — `top` has
        no shift rung (exact member only), so including it halved the mean.
    """
    ray = float(np.mean(m["ray_ms"]))
    mean_d = float(np.mean(m["depths"]))
    per_depth = ray / mean_d
    refresh = float(np.mean(m["refresh_ms"]))
    sides = [f for f in FACES if f != "top"]
    shift_f = [m["faces"][f][0] for f in sides
               if m["faces"][f][1] is not None
               and m["faces"][f][1].theta_deg == 0.0]
    n_side = int(np.max([m["members"][f] for f in sides]))
    side_ms = float(np.max([m["scans"][f][0] for f in sides]))
    top_ms, n_top = m["scans"]["top"][0], m["members"]["top"]
    est_cone = ESTIMATES["face family, full cone"]
    rows = [
        (f"one ray validity ({mean_d:.1f} depths)", ray,
         ESTIMATES["one ray validity"]),
        (f"  same, per-depth x {EST_DEPTHS} depths", per_depth * EST_DEPTHS,
         ESTIMATES["one ray validity"]),
        ("36-cell refresh", refresh, ESTIMATES["36-cell refresh"]),
        (f"face family, shift rung ({len(shift_f)}/{len(sides)} side)",
         float(np.mean(shift_f)) if shift_f else float("nan"),
         ESTIMATES["face family, shift rung"]),
        (f"face family, side ({n_side} members)", side_ms,
         est_cone * n_side / ESTIMATES["full cone rays"]),
        (f"face family, top ({n_top} members)", top_ms,
         est_cone * n_top / ESTIMATES["full cone rays"]),
    ]
    lines = [f"{'computation':<34} {'measured':>12} {'estimated':>11} "
             f"{'ratio':>7}  verdict",
             "-" * 82]
    for name, got, est in rows:
        ratio = got / est if est else float("nan")
        if not np.isfinite(got):
            verdict = "not exercised (no member on this rung)"
        elif ratio <= 1.0:
            verdict = "under estimate"
        elif ratio <= 2.0:
            verdict = "within 2x"
        else:
            verdict = "OVER 2x"
        lines.append(f"{name:<34} {f'{got:.4g} ms':>12} "
                     f"{f'{est:.0f} ms':>11} {f'{ratio:.3g}':>7}  {verdict}")
    lines.append(f"{'slide-in resolve':<34} "
                 f"{'not measured (Stage B: segment validation)':>12}")
    lines.append(f"per-ray spread: min {np.min(m['ray_ms']):.2f} ms  "
                 f"median {np.median(m['ray_ms']):.2f} ms  "
                 f"max {np.max(m['ray_ms']):.2f} ms  "
                 f"(n={len(m['ray_ms'])}); depths per ray "
                 f"{np.min(m['depths'])}-{np.max(m['depths'])}, "
                 f"mean {mean_d:.1f} vs the design's {EST_DEPTHS}")
    lines.append("refresh repeats: " +
                 ", ".join(f"{v:.1f} ms" for v in m["refresh_ms"]))
    return lines


def face_table(m, prim):
    """All five faces: what the ladder returns vs what the family contains.

    The `price` column was the diagnosis that motivated the 2026-09-17
    bench amendment: under the old "first shift with a sample wins" it read
    +0.12 to +0.20 m on `x+` in every run. Now that `resolve_face` picks the
    shallowest fronto-parallel sample it reads **+0.000 by construction**,
    and that is exactly its value — a standing check that the amendment is
    in force. `closest of any member` still shows what a tilt would buy,
    which is deliberately NOT taken (tilt is a strictly later rung).
    """
    lines = [f"{'face':<5} {'ladder — first member with a sample':<45} "
             f"{'closest FRONTO-PARALLEL member':<32} {'price':>7}  "
             f"{'closest of any member':<30} scan",
             "-" * 128]
    for face in FACES:
        ms, res = m["faces"][face]
        scan_ms, flat, best = m["scans"][face]
        if res is None:
            got, lad = f"INACCESSIBLE ({ms:6.1f} ms)", None
        else:
            lad = surface_dist(prim, res.T_cam[:3, 3])
            got = (f"d_surf={lad:.3f} theta={res.theta_deg:4.1f}d "
                   f"shift={res.shift:.2f} depth={res.depth:.2f} "
                   f"({ms:6.1f} ms)")
        def fmt(b):
            return ("-" if b is None else
                    f"d_surf={b[0]:.3f} shift={b[1]:.2f} "
                    f"theta={b[2]:4.1f}d depth={b[3]:.2f}")
        price = ("-" if (flat is None or lad is None)
                 else f"{lad - flat[0]:+.3f}")
        lines.append(f"{face:<5} {got:<45} {fmt(flat):<32} {price:>7}  "
                     f"{fmt(best):<30} {scan_ms:.0f} ms")
    return lines


# --------------------------------------------------------------------- renders
def _box_edges(prim):
    e = np.asarray(prim.params["extents"], float)
    signs = np.array([[a, b, c] for a in (-.5, .5) for b in (-.5, .5)
                      for c in (-.5, .5)])
    c = (prim.axes @ (signs * e).T).T + prim.center
    pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 6), (5, 7), (6, 7)]
    return [(c[i], c[j]) for i, j in pairs]


def _ring(ax, center, r=R_MIN, **kw):
    """Faint R_MIN sphere ring — the old frozen shell, drawn as one
    horizontal and two vertical great circles purely for scale."""
    t = np.linspace(0, 2 * np.pi, 121)
    z = np.zeros_like(t)
    for a, b, c in ((np.cos(t), np.sin(t), z), (np.cos(t), z, np.sin(t)),
                    (z, np.cos(t), np.sin(t))):
        ax.plot(center[0] + r * a, center[1] + r * b, center[2] + r * c, **kw)


def render(out: Path, run_id: str, prim, rows, margin: float,
          world: str = "boxguard"):
    import matplotlib
    matplotlib.use("Agg")                      # headless: never open a window
    import matplotlib.pyplot as plt

    pts = [r["old"] for r in rows] + [r["new"] for r in rows if r["new"] is not None]
    P = np.vstack(pts + [prim.center + R_MIN, prim.center - R_MIN])
    mid, half = (P.max(0) + P.min(0)) / 2, float((P.max(0) - P.min(0)).max()) / 2 + 0.02
    fig = plt.figure(figsize=(15, 7.2))
    for k, (elev, azim) in enumerate(((24, -62), (88, -90))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for a, b in _box_edges(prim):
            ax.plot(*zip(a, b), color="#333333", lw=1.6)
        _ring(ax, prim.center, color="#7f7f7f", lw=0.6, alpha=0.30)
        seen = set()
        for r in rows:
            o = r["old"]
            ax.plot([o[0]], [o[1]], [o[2]], "o", ms=5.5, color="#9e9e9e",
                    mec="#555555", mew=0.4,
                    label=None if "old" in seen else "old executed camera")
            seen.add("old")
            ax.text(o[0], o[1], o[2], "  " + r["label"].replace("cell ", "")
                    .replace("face ", ""), fontsize=6, color="#555555")
            if r["new"] is None:
                ax.plot([o[0]], [o[1]], [o[2]], "x", ms=11, mew=2.0,
                        color="#d62728",
                        label=None if "bad" in seen else "address inaccessible")
                seen.add("bad")
                continue
            n = r["new"]
            cone = r["theta_new"] > 0.0
            col = "#ff7f0e" if cone else "#2ca02c"
            tag = "ray endpoint (cone rung)" if cone else \
                "ray endpoint (shift/exact, theta=0)"
            key = "cone" if cone else "exact"
            ax.plot([n[0]], [n[1]], [n[2]], "o", ms=6, color=col,
                    label=None if key in seen else tag)
            seen.add(key)
            ax.plot(*zip(o, n), color=col, lw=0.8, alpha=0.55, ls="--")
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(labelsize=6)
        ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.set_title("oblique" if k == 0 else "top-down", fontsize=9)
        if k == 0:
            ax.legend(loc="upper left", fontsize=7.5)
        else:
            ax.set_zticklabels([])       # edge-on axis, pure clutter
            ax.set_zlabel("")
    closer = sum(r["verdict"] == "closer" for r in rows)
    obstacle_label = (f"boxguard box +{margin * 100:.0f} cm" if world != "guard"
                      else "guard world (geometry.object_box, fused cloud)")
    fig.suptitle(f"{run_id} — old endpoint vs ray endpoint  "
                 f"({closer}/{len(rows)} closer to the surface; "
                 f"obstacle = {obstacle_label}, "
                 f"faint ring = old {R_MIN:.2f} m shell)", fontsize=11)
    # subplots_adjust, not tight_layout: `set_box_aspect` + tight_layout
    # collapses 3D panels to a fraction of the canvas for some data extents.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01,
                        wspace=0.02)
    p = out / "comparison.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------ main
def _table(rows):
    """d_surf = Euclidean closest-point to the box surface (the comparison
    metric); along = along-ray travel from the surface, `depth - clear_m`
    (the metric the 0.105 m sensor floor applies to). See the module note."""
    lines = ["d_surf = Euclidean closest-point to box surface   |   "
             "along = along-ray from surface (depth - clear_m), "
             "sensor floor applies HERE",
             f"{'address':<12} {'old d_surf':>10} {'ray d_surf':>10} "
             f"{'along':>7} {'old th':>7} {'ray theta/shift/depth':>24} "
             f"{'roll':>5} {'smp':>4}  verdict",
             "-" * 104]
    for r in rows:
        th_old = "-" if r["theta_old"] is None else f"{r['theta_old']:5.1f}d"
        if r["new"] is None:
            lines.append(f"{r['label']:<12} {r['d_old']:>10.3f} "
                         f"{'--':>10} {'--':>7} {th_old:>7} {'--':>24} "
                         f"{'--':>5} {0:>4}  UNREACHABLE")
            continue
        new = (f"th={r['theta_new']:4.1f}d s={r['shift']:.2f} "
               f"d={r['depth']:.2f}")
        delta = r["d_new"] - r["d_old"]
        lines.append(f"{r['label']:<12} {r['d_old']:>10.3f} "
                     f"{r['d_new']:>10.3f} {r['along']:>7.3f} {th_old:>7} "
                     f"{new:>24} {r['roll']:>5.0f} {r['n_samples']:>4}  "
                     f"{r['verdict']} ({delta:+.3f} m)")
    return lines


def invariant_line(rows):
    """The sensor floor stated on the quantity it constrains."""
    got = [r for r in rows if r["along"] is not None]
    if not got:
        return "   sensor floor: no resolved rows to check"
    worst = min(r["along"] for r in got)
    ok = worst >= MIN_RANGE_M - 1e-9
    return (f"   sensor floor: all {len(got)} executed rows along-ray "
            f">= {MIN_RANGE_M:.3f} m {'OK' if ok else 'VIOLATED'} "
            f"(min {worst:.3f} m); d_surf can legitimately read lower — "
            f"oblique rays clear the box by less than they travel")


def bench(run: str = PINNED_RUN, out: str = str(OUT), margin: float = 0.0,
          repeats: int = 3, render_png: bool = True, cell_yaml: str = None,
          world: str = "boxguard"):
    """Replay one run: old endpoints vs ray endpoints, plus measured timings.

    margin: metres added to every box dimension for the OBSTACLE only
    (0.04 = the live `geometry.object_box` inflation, 2 cm per side).
    Ignored when `world='guard'` — `object_box` carries its own inflation.
    repeats: how many full 36-cell refreshes to time.
    cell_yaml: alternative cell.yaml (e.g. a pre-corner2-move copy).
    world: 'boxguard' (default, unchanged) — obstacle = the boxguard belief
    box + `margin`, the Stage A bench's world. 'guard' — obstacle =
    `geometry.object_box(fused cloud)`, built exactly as `machine._recenter`
    installs it: the world live sampling actually validates against (Anton's
    ruling 2026-09-17). Ray geometry (addresses, aabb, `box_corners`, the
    `d_surf` surface it is measured against) stays the boxguard prim in
    BOTH modes — only the collision world changes.
    """
    if world not in ("boxguard", "guard"):
        raise SystemExit(f"--world must be 'boxguard' or 'guard' (got {world!r})")
    run_id = str(run)                 # fire literal-evals bare-int-ish ids
    rd = run_dir_for(run_id)
    r = Run.load(rd)
    box = last_box(r)
    if box is None:
        print(f"== {run_id}: no boxguard decomposition on any step — SKIPPED")
        return None
    step_id, prim = box
    ses = r.session
    if ses is None or "color" not in (ses.intrinsics or {}):
        raise SystemExit(f"{run_id}: no colour intrinsics in session.json")
    # The run's OWN colour stream — old runs are 848x480, 1709 is 720p, and
    # the containment floor is fy/height, so this must never be a constant.
    color = ses.intrinsics["color"]
    fy, height_px = float(color.fy), int(color.height)

    guard_dims = None
    if world == "guard":
        fused = r.fused()
        if fused is None:
            raise SystemExit(f"{run_id}: --world guard needs a fused cloud "
                             f"(fused/cloud.npy) and this run has none")
        w, guard_dims, _ = guard_world(fused[0], cell_yaml)
    else:
        w = box_world(prim, margin, cell_yaml)
    ik = UR5eIK()
    aabb = box_aabb(prim)
    ext = np.asarray(prim.params["extents"], float)
    print(f"== {run_id}  ({len(r.steps)} steps, provenance {r.provenance})")
    print(f"   belief: step {step_id:03d} boxguard box  extents "
          f"{np.round(ext * 1000, 1)} mm  centre {np.round(prim.center, 3)}  "
          f"yaw {np.degrees(np.arctan2(prim.axes[1, 0], prim.axes[0, 0])):.1f} deg")
    obstacle_desc = (f"box + {margin * 100:.0f} cm/dim" if world != "guard" else
                     "geometry.object_box(fused cloud) "
                     f"{np.round(guard_dims * 1000, 1).tolist()} mm/dim")
    print(f"   world:  obstacle = {obstacle_desc}, coal "
          f"padding {w.padding * 1000:.0f} mm; intrinsics fy={fy:.1f} "
          f"h={height_px}px; RAY_STEP_M={RAY_STEP_M}")
    print(f"   keepouts: {'cell.yaml (current, post corner2 move c2f2e77)' if cell_yaml is None else cell_yaml}")

    # Warm up before the walk: without this the first address in the table
    # carries the cold-start cost of IK/coal data allocation and its ms
    # column is a lie about that address rather than a measurement.
    resolve_cell(w, ik, T_FLANGE_CAM, prim.center, aabb, 0, 1,
                 fy, height_px, corners=box_corners(prim))
    rows = walk(r, w, ik, prim, aabb, fy, height_px)
    if not rows:
        print("   no executed addressed steps — nothing to compare")
        return None
    print()
    for line in _table(rows):
        print(line)

    got = [x for x in rows if x["d_new"] is not None]
    closer = [x for x in got if x["verdict"] == "closer"]
    faces = [x for x in got if x["kind"] == "face"]
    flat = [x for x in faces if x["theta_new"] == 0.0]
    inside = [x for x in got if x["d_new"] <= 0.0]
    print(f"\n   {len(closer)}/{len(rows)} views closer to the surface, "
          f"{len(rows) - len(got)} inaccessible; mean surface distance "
          f"{np.mean([x['d_old'] for x in got]):.3f} -> "
          f"{np.mean([x['d_new'] for x in got]):.3f} m")
    if faces:
        olds = ", ".join("{}={:.0f}d".format(x["label"][5:], x["theta_old"])
                         for x in faces if x["theta_old"] is not None)
        print(f"   faces: {len(flat)}/{len(faces)} fronto-parallel "
              f"(theta=0); old face theta {olds or '-'}")
    print(f"   endpoints inside the box: {len(inside)} "
          f"(min surface distance {min(x['d_new'] for x in got):.3f} m)")
    print(invariant_line(rows))

    m = measure(w, ik, prim, aabb, fy, height_px, repeats=repeats)
    tl = timing_table(m)
    fl = face_table(m, prim)
    print()
    for line in fl:
        print(line)
    print()
    for line in tl:
        print(line)

    odir = Path(out) / run_id
    odir.mkdir(parents=True, exist_ok=True)
    header = (f"run {run_id}   belief step {step_id:03d}   "
             f"obstacle margin {margin:.3f} m   RAY_STEP_M {RAY_STEP_M}"
             if world != "guard" else
             f"run {run_id}   belief step {step_id:03d}   "
             f"world=guard obstacle=geometry.object_box(fused cloud) "
             f"{np.round(guard_dims * 1000, 1).tolist()} mm/dim   "
             f"RAY_STEP_M {RAY_STEP_M}")
    body = [header,
            f"box extents {np.round(ext, 4).tolist()} m at "
            f"{np.round(prim.center, 4).tolist()}",
            f"keepouts: {'cell.yaml current (post corner2 move, c2f2e77)' if cell_yaml is None else cell_yaml}",
            # Provenance the gate needs: the default keepout world is NOT the
            # one these runs executed under (corner2's apex was moved by hand
            # on 2026-09-17 12:20, after the last of them). Re-run with
            # --cell_yaml <pre-move copy> to measure the difference instead
            # of assuming it. Verified scope, 1709-cam-ai-demo1 and
            # 1509-lan-ai_bench5: every EXECUTED row is identical
            # (d_old/d_new/along/theta/shift/depth/roll/samples), so no
            # per-view conclusion moves. One NON-executed family does move —
            # lan's `y-` resolves 35 deg / d_surf 0.125 pre-move vs 20 deg /
            # 0.135 now — which is consistent with the wedge that was
            # "biting the front workspace" sitting right in front of y-.
            "corner2 note: all EXECUTED rows identical under the pre-move "
            "cell.yaml (verified on 1709-cam-ai-demo1 + 1509-lan-ai_bench5); "
            "one non-executed family differs (lan y-: 35deg/0.125 pre-move vs "
            "20deg/0.135 now) — no per-view conclusion in this bench moves",
            "",
            "metrics: d_surf = Euclidean closest-point to the box surface; "
            "along = along-ray from the surface (depth - clear_m), which is "
            "what the 0.105 m sensor floor constrains",
            invariant_line(rows).strip(),
            "", "per-view resolve (warmed up; still the first call per "
            "address, so cache-cold in the IK branch tables):"]
    for x in rows:
        what = ("INACCESSIBLE" if x["new"] is None else
                "theta={:4.1f} shift={:.2f} depth={:.2f} along={:.3f} "
                "d_surf={:.3f}".format(x["theta_new"], x["shift"], x["depth"],
                                       x["along"], x["d_new"]))
        body.append(f"  {x['label']:<12} {x['ms']:>8.2f} ms   {what}")
    body += ["", "per-face family (all five faces, not only the executed):"]
    for face, (ms, res) in m["faces"].items():
        scan_ms, flat, best = m["scans"][face]
        body.append(f"  {face:<4} ladder {ms:>8.2f} ms   full scan "
                    f"{scan_ms:>8.2f} ms ({m['members'][face]:>3} members)   "
                    + ("no member has a safe sample" if res is None else
                       f"shift={res.shift:.2f} theta={res.theta_deg:.1f} "
                       f"depth={res.depth:.2f}")
                    + ("" if flat is None else
                       f"   closest fronto-parallel d_surf={flat[0]:.3f} "
                       f"(shift={flat[1]:.2f} depth={flat[3]:.2f})")
                    + ("" if best is None else
                       f"   closest any d_surf={best[0]:.3f} "
                       f"(theta={best[2]:.1f} depth={best[3]:.2f})"))
    body += ["", f"36-cell refresh x{repeats}: "
                 + ", ".join(f"{v:.1f} ms" for v in m["refresh_ms"]),
             "", "face ladder vs family-closest:"] + fl + \
           ["", "design table (Feasibility estimates) vs measured:"] + tl
    (odir / "timings.txt").write_text("\n".join(body) + "\n")
    print(f"\n   timings: {odir / 'timings.txt'}")
    if render_png:
        print(f"   render:  {render(odir, run_id, prim, rows, margin, world=world)}")
    return {"run": run_id, "rows": rows, "timings": m}


def main(run: str = PINNED_RUN, out: str = str(OUT), margin: float = 0.0,
         repeats: int = 3, render_png: bool = True, cell_yaml: str = None,
         world: str = "boxguard"):
    """`--run all` benches the pinned 1709 demo, the two most recent
    `*bench*` runs (by mtime, discovered now) and the explicit flat-object
    extra; `--run a,b` benches a list; anything else is one run id.

    world: 'boxguard' (default) or 'guard' — see `bench`'s docstring.
    """
    ids = (default_runs() if str(run) == "all"
           else [s for s in str(run).split(",") if s])
    if str(run) == "all":
        print(f"runs: pinned {PINNED_RUN}; newest bench "
              f"{newest_bench_runs(2)}; extra {list(EXTRA_RUNS)}\n")
    done = []
    for run_id in ids:
        res = bench(run=run_id, out=out, margin=margin, repeats=repeats,
                    render_png=render_png, cell_yaml=cell_yaml, world=world)
        if res is not None:
            done.append(res)
        print()
    if len(ids) > 1:
        print(f"benched {len(done)}/{len(ids)} runs: "
              f"{', '.join(d['run'] for d in done)}")
    return None


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
