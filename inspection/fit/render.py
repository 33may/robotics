#!/usr/bin/env python3
"""Evidence renders — a decomposition drawn over the cloud it came from.

These exist for one purpose: so a human can look at a fit and say plausible
or garbage. Nothing here feeds the loop. Matplotlib only (no seaborn), one
figure per (run, method) with four panels — early, quarter, half, final —
and a camera and axis box fixed across all panels of a run, so the panels
are comparable to each other and to the other methods' figures.

    p inspection/fit/render.py evidence --run 2408-cup3 --method cyl
    p inspection/fit/render.py evidence --run 2408-cup3,2408-cup4 --method all
    p inspection/fit/render.py curves   --run 2408-cup3,2408-cup4,2408-cup5

Output: `data/derived/<run>/fits/renders/<run>_<method>.png` and
`<run>_curves.png`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from inspection.fit.contract import Decomposition, Primitive, decomposition_from_json  # noqa: E402

DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"

KIND_COLOR = {"plane_patch": "#d62728", "cylinder": "#1f77b4",
              "superquadric": "#2ca02c", "box": "#ff7f0e", "mesh": "#9467bd"}


# --- primitive -> line segments / surface ---------------------------------
def _sq_surface(scale, eps, nu=28, nv=56):
    a = np.asarray(scale, float)
    e1, e2 = max(float(eps[0]), 1e-2), max(float(eps[1]), 1e-2)
    u = np.linspace(-np.pi / 2, np.pi / 2, nu)
    v = np.linspace(-np.pi, np.pi, nv)
    U, V = np.meshgrid(u, v, indexing="ij")

    def c(t, e):
        return np.sign(np.cos(t)) * np.abs(np.cos(t)) ** e

    def s(t, e):
        return np.sign(np.sin(t)) * np.abs(np.sin(t)) ** e

    return (a[0] * c(U, e1) * c(V, e2),
            a[1] * c(U, e1) * s(V, e2),
            a[2] * s(U, e1) * np.ones_like(V))


def draw(ax, prim: Primitive):
    T = prim.T
    R, t = T[:3, :3], T[:3, 3]
    col = KIND_COLOR.get(prim.kind, "k")

    def W(X, Y, Z):
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) @ R.T + t
        return [P[:, i].reshape(X.shape) for i in range(3)]

    if prim.kind == "plane_patch":
        hx, hy = np.asarray(prim.params["extents"], float) / 2
        c = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0],
                      [-hx, -hy, 0]]) @ R.T + t
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=col, lw=1.8)
        n = R[:, 2] * 0.012
        ax.plot(*zip(t, t + n), color=col, lw=1.0, alpha=0.7)

    elif prim.kind == "cylinder":
        r, L = float(prim.params["radius"]), float(prim.params["length"])
        th = np.linspace(0, 2 * np.pi, 64)
        for z in (-L / 2, L / 2):
            X, Y, Z = W(r * np.cos(th), r * np.sin(th), np.full_like(th, z))
            ax.plot(X, Y, Z, color=col, lw=1.6)
        for a0 in np.linspace(0, 2 * np.pi, 9)[:-1]:
            X, Y, Z = W(np.full(2, r * np.cos(a0)), np.full(2, r * np.sin(a0)),
                        np.array([-L / 2, L / 2]))
            ax.plot(X, Y, Z, color=col, lw=0.8, alpha=0.6)

    elif prim.kind == "box":
        h = np.asarray(prim.params["extents"], float) / 2
        s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * h
        V = s @ R.T + t
        for i, j in [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
                     (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]:
            ax.plot(*zip(V[i], V[j]), color=col, lw=1.4)

    elif prim.kind == "superquadric":
        X, Y, Z = W(*_sq_surface(prim.params["scale"], prim.params["eps"]))
        ax.plot_surface(X, Y, Z, color=col, alpha=0.28, linewidth=0,
                        antialiased=True, shade=True)
        ax.plot_wireframe(X, Y, Z, color=col, lw=0.35, alpha=0.55,
                          rstride=4, cstride=6)

    elif prim.kind == "mesh":
        V = np.asarray(prim.params["vertices"], float)
        F = np.asarray(prim.params["faces"], int)
        if len(F):
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                            color=col, alpha=0.3, linewidth=0.2)


def _wrap(text, width=44, lines=3):
    """Panel titles must stay inside their panel or the figure is unreadable."""
    import textwrap
    out = textwrap.wrap(str(text), width=width)[:lines]
    if not out:
        return ""
    if len("\n".join(out)) < len(str(text)):
        out[-1] = out[-1][:width - 1] + "…"
    return "\n".join(out)


def _steps(run):
    return sorted((DERIVED / run / "steps").glob("step_*.ply"))


def _pick(paths):
    """early / quarter / half / final — the four the eye needs."""
    n = len(paths)
    return sorted({min(1, n - 1), n // 4, n // 2, n - 1})


def _load(path):
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, float)


def _box(run):
    P = _load(_steps(run)[-1])
    c = (P.max(0) + P.min(0)) / 2
    h = float((P.max(0) - P.min(0)).max()) / 2 * 1.35
    return c, h


def evidence(run: str, method: str, elev=22, azim=-60):
    paths = _steps(run)
    picks = _pick(paths)
    fitdir = DERIVED / run / "fits" / method
    c, h = _box(run)

    fig = plt.figure(figsize=(4.1 * len(picks), 4.6))
    fig.suptitle(f"{run} — {method}", fontsize=13, y=0.99)
    for k, i in enumerate(picks):
        p = paths[i]
        P = _load(p)
        ax = fig.add_subplot(1, len(picks), k + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.7, c="0.55", alpha=0.55,
                   linewidths=0, depthshade=False)
        f = fitdir / f"{p.stem}.json"
        sub = "no fit on disk"
        if f.exists():
            d: Decomposition = decomposition_from_json(json.loads(f.read_text()))
            for prim in d.primitives:
                draw(ax, prim)
            sub = (f"ABSTAIN\n{_wrap(d.reason)}" if d.abstain else
                   f"{d.n_primitives} prim · cov {d.coverage:.2f} · "
                   f"res {d.residual_mm:.1f} mm\n"
                   + _wrap(", ".join(f"{q.label}({q.support})"
                                     for q in d.primitives)))
        ax.set_title(f"step {i}  N={len(P)}\n{sub}", fontsize=7.5)
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)
        ax.set_zlim(c[2] - h, c[2] + h)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
        ax.tick_params(length=0)
        ax.grid(True, alpha=0.25)

    out = DERIVED / run / "fits" / "renders"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{run}_{method}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    return dst


def curves(run: str, methods):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for m in methods:
        f = DERIVED / run / "fits" / m / "scores.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: r["step"])
        s = [r["step"] for r in rows]
        axes[0].plot(s, [r["coverage"] for r in rows], marker="o", ms=3, label=m)
        axes[1].plot(s, [r["residual_mm"] if not r["abstain"] else np.nan
                         for r in rows], marker="o", ms=3, label=m)
        axes[2].plot(s, [r["n_primitives"] for r in rows], marker="o", ms=3, label=m)
    for a, t in zip(axes, ("coverage", "residual (mm)", "n primitives")):
        a.set_xlabel("step")
        a.set_title(t, fontsize=10)
        a.grid(alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{run} — score vs step", fontsize=12)
    out = DERIVED / run / "fits" / "renders"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{run}_curves.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    return dst


def main():
    from inspection.fit import bank
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evidence")
    e.add_argument("--run", required=True)
    e.add_argument("--method", default="all")
    e.add_argument("--elev", type=float, default=22)
    e.add_argument("--azim", type=float, default=-60)
    c = sub.add_parser("curves")
    c.add_argument("--run", required=True)
    c.add_argument("--methods", default=",".join(bank.DEFAULT_METHODS))
    args = ap.parse_args()

    runs = [r.strip() for r in args.run.split(",") if r.strip()]
    if args.cmd == "evidence":
        ms = bank.methods() if args.method == "all" else \
            [m.strip() for m in args.method.split(",")]
        for r in runs:
            for m in ms:
                print(evidence(r, m, args.elev, args.azim))
    else:
        ms = [m.strip() for m in args.methods.split(",")]
        for r in runs:
            print(curves(r, ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
