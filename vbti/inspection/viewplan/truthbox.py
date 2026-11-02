#!/usr/bin/env python3
"""Ground-truth cuboid annotator + completion benchmark (box runs).

Anton fits ONE oriented cuboid per collection run over the final
accumulated cloud — the GT for the surface-completion benchmark. The box
is seeded from `min_yaw_aabb` (or a previously saved truth), its bottom
face pinned to the probed table plane; he only micro-adjusts.

    p inspection/viewplan/truthbox.py edit --run 0809-box2
    p inspection/viewplan/truthbox.py edit --all
    p inspection/viewplan/truthbox.py seed --run 0809-box2   # print, no UI

Viewer: meshcat (URL printed — open in browser). Terminal keys:

    w/s  +/- along the box's yaw axis      (5 mm; W/S = 1 mm)
    a/d  +/- perpendicular                 (5 mm; A/D = 1 mm)
    q/e  yaw -/+ 1 deg                     (Q/E = 0.2 deg)
    j/l  length -/+ 5 mm                   (J/L = 1 mm)
    i/k  width  +/- 5 mm                   (I/K = 1 mm)
    u/o  height +/- 5 mm                   (U/O = 1 mm)
    g    re-seed from min_yaw_aabb
    p    print the current numbers
    x    SAVE -> derived/<run>/truth/box.json
    n    next run / quit (does NOT save)

Truth format (metres, base frame): {run, center_xy, z_table, yaw_deg,
dims_lwh} — l along the yaw direction, bottom face at z_table.
"""
from __future__ import annotations

import json
import sys
import termios
import tty
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from inspection.cell.geometry import min_yaw_aabb
from inspection.viewplan import mesh as meshmod
from inspection.viewplan import planner

DERIVED = planner.DATA / "derived"
RUNS_ALL = ["0809-box2", "0809-box3", "0809-box4-3", "0809-box5",
            "0809-box6", "0809-box7", "0809-box8", "0809-box9",
            "0809-box10"]


# ---------------------------------------------------------------- state
def _final_cloud(run: str):
    import open3d as o3d
    plys = sorted((DERIVED / run / "steps").glob("step_*.ply"))
    if not plys:
        raise SystemExit(f"no derived step clouds for {run}")
    pcd = o3d.io.read_point_cloud(str(plys[-1]))
    P = np.asarray(pcd.points)
    C = (np.asarray(pcd.colors) if pcd.has_colors()
         else np.full((len(P), 3), 0.7))
    return P, C


def _table_z(P: np.ndarray, cx: float, cy: float) -> float:
    a, b, c, d = meshmod.table_plane(P)
    return float(-(a * cx + b * cy + d) / c)


def seed_state(run: str, P: np.ndarray) -> Dict:
    """min_yaw_aabb footprint + table-pinned height, as the truth dict."""
    center, dims, yaw = min_yaw_aabb(P, pct=1.0)
    z_t = _table_z(P, float(center[0]), float(center[1]))
    top = float(np.percentile(P[:, 2], 99.5))
    return {"run": run, "center_xy": [float(center[0]), float(center[1])],
            "z_table": z_t, "yaw_deg": float(np.degrees(yaw)),
            "dims_lwh": [float(dims[0]), float(dims[1]),
                         max(top - z_t, 0.01)],
            "seed": "min_yaw_aabb"}


def truth_path(run: str) -> Path:
    return DERIVED / run / "truth" / "box.json"


def load_truth(run: str) -> Optional[Dict]:
    p = truth_path(run)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- meshcat
def _draw(vis, P: np.ndarray, C: np.ndarray, st: Dict) -> None:
    import meshcat.geometry as g
    import meshcat.transformations as tf
    vis["cloud"].set_object(g.PointCloud(P.T, C.T, size=0.0025))
    l, w, h = st["dims_lwh"]
    box = g.Box([l, w, h])
    vis["truth"].set_object(
        box, g.MeshBasicMaterial(color=0xFF40FF, wireframe=True,
                                 linewidth=2))
    yaw = np.radians(st["yaw_deg"])
    T = tf.rotation_matrix(yaw, [0, 0, 1])
    T[:3, 3] = [st["center_xy"][0], st["center_xy"][1],
                st["z_table"] + h / 2]
    vis["truth"].set_transform(T)
    # table patch under the box, for the pinned-bottom sanity
    vis["table"].set_object(
        g.Box([0.5, 0.5, 0.001]),
        g.MeshBasicMaterial(color=0x445544, opacity=0.3, transparent=True))
    Tt = np.eye(4)
    Tt[:3, 3] = [st["center_xy"][0], st["center_xy"][1],
                 st["z_table"] - 0.0005]
    vis["table"].set_transform(Tt)


def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _fmt(st: Dict) -> str:
    l, w, h = st["dims_lwh"]
    return (f"c=({st['center_xy'][0]:+.4f},{st['center_xy'][1]:+.4f}) "
            f"yaw={st['yaw_deg']:+7.2f}deg "
            f"lwh=({l * 1e3:.0f},{w * 1e3:.0f},{h * 1e3:.0f})mm "
            f"z_table={st['z_table']:+.4f}")


def _edit_one(vis, run: str) -> bool:
    """Edit loop for one run. Returns False if the user quit the session."""
    P, C = _final_cloud(run)
    st = load_truth(run) or seed_state(run, P)
    saved = "saved truth" if "annotated" in st else "min_yaw seed"
    print(f"\n=== {run} ({len(P)} pts, start = {saved}) ===")
    print("keys go in THIS terminal (browser = orbit only) — "
          "x saves, n next, ? help")
    _draw(vis, P, C, st)
    print(_fmt(st), end="\r", flush=True)

    step = {"t": 0.005, "T": 0.001, "yaw": 1.0, "YAW": 0.2}
    dirty = False
    warned = False
    while True:
        k = _getch()
        if k.lower() in "wasdqejlikuo":
            dirty, warned = True, False
        yaw = np.radians(st["yaw_deg"])
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        d_t = step["T"] if k.isupper() else step["t"]
        kl = k.lower()
        if kl == "w":
            st["center_xy"] = (np.array(st["center_xy"]) + d_t * fwd).tolist()
        elif kl == "s":
            st["center_xy"] = (np.array(st["center_xy"]) - d_t * fwd).tolist()
        elif kl == "a":
            st["center_xy"] = (np.array(st["center_xy"]) + d_t * left).tolist()
        elif kl == "d":
            st["center_xy"] = (np.array(st["center_xy"]) - d_t * left).tolist()
        elif kl == "q":
            st["yaw_deg"] -= step["YAW"] if k.isupper() else step["yaw"]
        elif kl == "e":
            st["yaw_deg"] += step["YAW"] if k.isupper() else step["yaw"]
        elif kl == "j":
            st["dims_lwh"][0] = max(st["dims_lwh"][0] - d_t, 0.01)
        elif kl == "l":
            st["dims_lwh"][0] += d_t
        elif kl == "i":
            st["dims_lwh"][1] += d_t
        elif kl == "k":
            st["dims_lwh"][1] = max(st["dims_lwh"][1] - d_t, 0.01)
        elif kl == "u":
            st["dims_lwh"][2] += d_t
        elif kl == "o":
            st["dims_lwh"][2] = max(st["dims_lwh"][2] - d_t, 0.01)
        elif kl == "g":
            st = seed_state(run, P)
            print("re-seeded:", _fmt(st))
        elif kl == "p":
            print(_fmt(st))
        elif kl == "x":
            st["annotated"] = "anton"
            st["z_table"] = _table_z(P, *st["center_xy"])
            out = truth_path(run)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(st, indent=1))
            dirty = False
            print(f"\nSAVED {out}")
        elif kl == "n":
            if dirty and not warned:
                warned = True
                print("\nUNSAVED CHANGES — x to save, n again to discard")
                continue
            return True
        elif k in ("\x03", "\x04"):          # ctrl-c / ctrl-d
            return False
        elif kl == "?":
            print("\n" + __doc__.split("Terminal keys:")[1])
        _draw(vis, P, C, st)
        print(_fmt(st) + "   ", end="\r", flush=True)


# ---------------------------------------------------------------- CLI
def edit(run: Optional[str] = None, all: bool = False):   # noqa: A002
    """Interactive GT annotation (meshcat + terminal keys)."""
    import meshcat
    vis = meshcat.Visualizer()
    print(f"open the viewer: {vis.url()}")
    runs = RUNS_ALL if all or run is None else [str(run)]
    for r in runs:
        if not _edit_one(vis, r):
            break
    print("done")


def seed(run: str):
    """Print the min_yaw seed (and any saved truth) for one run. No UI."""
    P, _C = _final_cloud(str(run))
    print("seed :", _fmt(seed_state(str(run), P)))
    t = load_truth(str(run))
    if t:
        print("truth:", _fmt(t))


# ---------------------------------------------------------------- bench
def _to_box_frame(pts: np.ndarray, t: Dict) -> np.ndarray:
    yaw = np.radians(t["yaw_deg"])
    c, s = np.cos(yaw), np.sin(yaw)
    ctr = np.array([t["center_xy"][0], t["center_xy"][1],
                    t["z_table"] + t["dims_lwh"][2] / 2])
    d = pts - ctr
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return d @ R.T


def _box_surface_dist(pts_box: np.ndarray, half: np.ndarray) -> np.ndarray:
    """|SDF| of an axis-aligned box, vectorised — exact surface distance."""
    q = np.abs(pts_box) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return np.abs(outside + inside)


def _face_grids(t: Dict, step_m: float = 0.005):
    """(name, (K,3) box-frame samples) for 4 side faces + top."""
    l, w, h = t["dims_lwh"]
    hx, hy, hz = l / 2, w / 2, h / 2
    out = []

    def grid(a_lim, b_lim, sa, sb):
        a = np.arange(-a_lim, a_lim + 1e-9, step_m)
        b = np.arange(-b_lim, b_lim + 1e-9, step_m)
        return np.stack(np.meshgrid(a, b), -1).reshape(-1, 2)

    g = grid(hy, hz, step_m, step_m)
    out.append(("x+", np.column_stack([np.full(len(g), hx), g])))
    out.append(("x-", np.column_stack([np.full(len(g), -hx), g])))
    g = grid(hx, hz, step_m, step_m)
    out.append(("y+", np.column_stack([g[:, :1], np.full(len(g), hy),
                                       g[:, 1:]])))
    out.append(("y-", np.column_stack([g[:, :1], np.full(len(g), -hy),
                                       g[:, 1:]])))
    g = grid(hx, hy, step_m, step_m)
    out.append(("top", np.column_stack([g, np.full(len(g), hz)])))
    return out


def _eval_mesh(m, t: Dict, tau: float) -> Dict:
    """Face coverage / surface error / hallucination of one recon mesh."""
    import open3d as o3d
    yaw = np.radians(t["yaw_deg"])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ctr = np.array([t["center_xy"][0], t["center_xy"][1],
                    t["z_table"] + t["dims_lwh"][2] / 2])
    half = np.asarray(t["dims_lwh"]) / 2

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))

    faces = {}
    for name, g_box in _face_grids(t):
        g_base = g_box @ R.T + ctr
        d = scene.compute_distance(
            o3d.core.Tensor(g_base.astype(np.float32))).numpy()
        faces[name] = float((d <= tau).mean())
        if name == "top":
            gt_to_recon = float(d.mean())

    n_samp = 2000
    if len(m.triangles):
        # Seeded per call: unseeded sampling made halluc jitter by
        # ~0.001 across processes (caught by the GUIDE §4 reproduction
        # check, 15:5x) — every bench row must be exactly reproducible.
        o3d.utility.random.seed(0)
        samp = np.asarray(m.sample_points_uniformly(n_samp).points)
        d_box = _box_surface_dist(_to_box_frame(samp, t), half)
        recon_to_gt = float(d_box.mean())
        halluc = float((d_box > tau).mean())
    else:
        recon_to_gt, halluc = float("inf"), 1.0
    return {"faces": faces, "worst_face": min(faces.values()),
            "surface_err_mm": 1e3 * recon_to_gt,
            "hallucination_frac": halluc}


def bench(algo: str = "all", runs: str = "all", steps: str = "all",
          tau: float = 0.01, render_worst: bool = False):
    """Score completion algorithms against the saved GT cuboids.

    Appends one row per (run, step, algo) to
    `derived/completion-bench/<algo>.jsonl` and prints the aggregate:
    per-run mean worst-face coverage, the overall min, error and
    hallucination. Runs without a saved truth are skipped and counted.
    """
    import open3d as o3d
    from inspection.viewplan.complete import ALGOS
    from inspection.viewplan.planner import recorded_cams

    def _csv(x):                      # fire literal-evals "a,b" -> tuple
        if isinstance(x, (tuple, list)):
            return [str(i).strip() for i in x]
        return [s.strip() for s in str(x).split(",") if s.strip()]

    algos = list(ALGOS) if algo == "all" else _csv(algo)
    run_list = RUNS_ALL if runs == "all" else _csv(runs)
    with_truth = [r for r in run_list if load_truth(r)]
    print(f"truth available: {len(with_truth)}/{len(run_list)} runs")
    if not with_truth:
        raise SystemExit("no GT annotated yet — run `edit` first")

    out_dir = DERIVED / "completion-bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in algos:
        fn = ALGOS[a]
        rows = []
        for r in with_truth:
            t = load_truth(r)
            cams = recorded_cams(r)
            plys = sorted((DERIVED / r / "steps").glob("step_*.ply"))
            if steps == "final":
                plys = plys[-1:]
            elif steps == "half":
                plys = plys[::2] + ([plys[-1]] if plys else [])
            for p in plys:
                k = int(p.stem.split("_")[1])
                P = np.asarray(o3d.io.read_point_cloud(str(p)).points)
                try:
                    m = fn(P, cams)
                    row = _eval_mesh(m, t, tau)
                except Exception as exc:              # noqa: BLE001
                    row = {"faces": {}, "worst_face": 0.0,
                           "surface_err_mm": float("inf"),
                           "hallucination_frac": 1.0,
                           "error": f"{type(exc).__name__}: {exc}"}
                row.update(run=r, step=k, algo=a, tau=tau)
                rows.append(row)
        log = out_dir / f"{a}.jsonl"
        with log.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        by_run = {}
        for row in rows:
            by_run.setdefault(row["run"], []).append(row)
        print(f"\n[{a}] per-run mean worst-face / err_mm / halluc "
              f"(n steps):")
        worst_overall = 1.0
        for r, rr in sorted(by_run.items()):
            wf = float(np.mean([x["worst_face"] for x in rr]))
            worst_overall = min(worst_overall,
                                min(x["worst_face"] for x in rr))
            err = float(np.median([x["surface_err_mm"] for x in rr]))
            hal = float(np.mean([x["hallucination_frac"] for x in rr]))
            print(f"  {r:12s} {wf:.3f} / {err:5.1f} / {hal:.3f}  "
                  f"({len(rr)})")
        print(f"  min worst-face over all (run,step): {worst_overall:.3f} "
              f"-> {log}")


def board(algo: str = "all"):
    """Aggregate completion-bench jsonl into leaderboard columns.

    Per algo: headline (min-over-runs worst-face at each run's FINAL
    step), mean worst-face final, mean worst-face mid (mid = benched
    step nearest floor(0.5*(n_steps-1)), tie -> lower), mean
    surface_err_mm / hallucination_frac over final steps. Rows are
    deduped by (run, step), last wins (odd-length runs bench the final
    step twice under `half`).
    """
    out_dir = DERIVED / "completion-bench"
    files = sorted(out_dir.glob("*.jsonl")) if algo == "all" else \
        [out_dir / f"{a.strip()}.jsonl" for a in str(algo).split(",")]
    print(f"{'algo':10s} {'headline':>8s} {'wf_final':>8s} {'wf_mid':>7s} "
          f"{'min_mid':>7s} {'err_mm':>7s} {'halluc':>7s}  runs")
    for f in files:
        if not f.exists():
            print(f"{f.stem:10s} (no jsonl)")
            continue
        rows = {}
        for line in f.read_text().splitlines():
            row = json.loads(line)
            rows[(row["run"], row["step"])] = row
        by_run: Dict[str, dict] = {}
        for (r, s), row in rows.items():
            by_run.setdefault(r, {})[s] = row
        wf_fin, wf_mid, errs, hals = [], [], [], []
        for r, sr in sorted(by_run.items()):
            steps_b = sorted(sr)
            fin = steps_b[-1]
            mid_t = fin // 2                      # floor(0.5*(n-1))
            mid = min(steps_b, key=lambda s: (abs(s - mid_t), s))
            wf_fin.append(sr[fin]["worst_face"])
            wf_mid.append(sr[mid]["worst_face"])
            errs.append(sr[fin]["surface_err_mm"])
            hals.append(sr[fin]["hallucination_frac"])
        print(f"{f.stem:10s} {min(wf_fin):8.3f} "
              f"{float(np.mean(wf_fin)):8.3f} "
              f"{float(np.mean(wf_mid)):7.3f} "
              f"{min(wf_mid):7.3f} "
              f"{float(np.mean(errs)):7.1f} "
              f"{float(np.mean(hals)):7.3f}  {len(by_run)}/9")
        worst = min(by_run, key=lambda r: by_run[r][max(by_run[r])]
                    ["worst_face"])
        print(f"{'':10s} worst run by final worst-face: {worst} "
              f"({by_run[worst][max(by_run[worst])]['worst_face']:.3f})")


def sheet(algo: str, run: str, steps: str, out: Optional[str] = None):
    """Evidence sheet (GUIDE 3.4): rows = steps, cols = h0/h3/h6/h9 at
    ring v=0, worst-face from the bench jsonl burned into each row."""
    import cv2
    import open3d as o3d
    from inspection.viewplan import regions
    from inspection.viewplan import render as rendermod
    from inspection.viewplan.complete import ALGOS
    from inspection.viewplan.planner import recorded_cams, session_intrinsics
    run = str(run)
    if isinstance(steps, (tuple, list)):          # fire literal-eval
        step_list = [int(s) for s in steps]
    else:
        step_list = [int(s) for s in str(steps).split(",")]
    wf = {}
    log = DERIVED / "completion-bench" / f"{algo}.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            if row["run"] == run:
                wf[row["step"]] = row["worst_face"]
    shell = regions.shell_of(run)
    intr = session_intrinsics(run)
    cams = recorded_cams(run)
    rows_img = []
    for s in step_list:
        P = np.asarray(o3d.io.read_point_cloud(str(
            DERIVED / run / "steps" / f"step_{s:03d}.ply")).points)
        m = ALGOS[algo](P, cams)
        scene = rendermod.scene_of(m)
        row = []
        for h in (0, 3, 6, 9):
            img, _ = rendermod.render_view(
                scene, regions.orig_pose(shell, h, 0), intr, scale=0.5,
                window=(0.07, shell["r"] + 0.05))
            img = np.ascontiguousarray(img)
            tag = f"{algo} s{s} h{h}"
            if s in wf:
                tag += f" wf={wf[s]:.3f}"
            cv2.putText(img, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(img)
        rows_img.append(np.concatenate(row, axis=1))
    dst = Path(out or f"/tmp/completion_bench/{algo}/"
               f"{run}_{'_'.join(f's{s}' for s in step_list)}.png")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), np.concatenate(rows_img, axis=0)[:, :, ::-1])
    print(dst)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({"edit": edit, "seed": seed, "bench": bench, "board": board,
               "sheet": sheet})
