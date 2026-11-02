"""map_audit.py — layer-1 map validation: bake trajectory vs GT (MAY-173 locdev T3).

Joins a bake output TUM trajectory (`poses/*.tum` from create_map_offline.py) to
the coverage-dump GT base rows by sim-time stamp, rigid-aligns (Umeyama, NO
scale — stereo is metric, scale drift must stay visible), and reports ATE stats
against the P1 gate (full-drive backbone mean ≤ 0.15 m).

GT is the eval ORACLE only — it never feeds the map pipeline (rule:
no-gt-in-localization-pipeline). Offline tooling, `hum` env; numpy + stdlib.

Usage:
    p logic/simulation/mapping/map_audit.py \
        --tum data/maps/<bake>/poses/keyframe_pose_optimized.tum \
        --dump data/coverage_drives/warehouse_coverage_v1 [--min-t-rel 6.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

#: stamp join precision in seconds-decimals (TUM carries 9; 4 ≈ 0.1 ms slack)
_JOIN_DECIMALS = 4


def umeyama_align(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Rigid-align estimated points P onto GT points G (rotation+translation,
    no scale). Returns the aligned copy of P."""
    P, G = np.asarray(P, float), np.asarray(G, float)
    muP, muG = P.mean(0), G.mean(0)
    U, _, Vt = np.linalg.svd((G - muG).T @ (P - muP))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return (R @ (P - muP).T).T + muG


def ate_stats(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, float)
    return {
        "n": int(err.size),
        "mean": float(err.mean()),
        "p50": float(np.percentile(err, 50)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
    }


def _load_gt(dump_dir: Path) -> Dict[float, np.ndarray]:
    gt = {}
    with open(dump_dir / "poses.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["cam"] == "base":
                key = round(r["stamp_ns"] / 1e9, _JOIN_DECIMALS)
                gt[key] = np.array([r["x"], r["y"], 0.0])
    return gt


def audit_tum_vs_gt(
    tum_path: Path | str,
    dump_dir: Path | str,
    *,
    min_t_rel: float = 0.0,
) -> Dict[str, float]:
    """ATE of a TUM trajectory vs dump GT. `min_t_rel` drops the first N seconds
    (relative to the trajectory start) BEFORE alignment — for scoring past a
    known cold-start segment without letting it drag the alignment."""
    tum = np.loadtxt(Path(tum_path))
    if tum.ndim == 1:
        tum = tum[None, :]
    gt = _load_gt(Path(dump_dir))
    t0 = tum[0, 0]
    P, G = [], []
    for row in tum:
        if row[0] - t0 < min_t_rel:
            continue
        key = round(row[0], _JOIN_DECIMALS)
        if key in gt:
            P.append(row[1:4])
            G.append(gt[key])
    if len(P) < 3:
        raise ValueError(f"only {len(P)} stamps joined — stamp mismatch?")
    P, G = np.array(P), np.array(G)
    err = np.linalg.norm(umeyama_align(P, G) - G, axis=1)
    return ate_stats(err)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bake trajectory vs GT (ATE).")
    ap.add_argument("--tum", required=True, help="TUM trajectory from the bake")
    ap.add_argument("--dump", required=True, help="coverage dump dir (GT rows)")
    ap.add_argument("--min-t-rel", type=float, default=0.0,
                    help="score only t >= start + N seconds")
    args = ap.parse_args(list(argv) if argv is not None else None)
    rep = audit_tum_vs_gt(args.tum, args.dump, min_t_rel=args.min_t_rel)
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
