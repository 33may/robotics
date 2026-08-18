"""Fit the base->table transform from probed touch points (MAY-183 Phase B).

Usage (from repo root, robo env active):
    p inspection/calib/probe_fit.py                 # newest table_probe_*.yaml
    p inspection/calib/probe_fit.py --data <yaml>   # specific session

Data format (see table_probe_*.yaml): a list of points, each with the location
in the cell.yaml `table` frame [m] and the pendant TCP reading in the robot
Base frame [m]. Points marked `holdout: true` are excluded from the fit and
used as an independent accuracy test (predicted vs measured).

Output: per-point residuals, holdout error, and the `pose:` line to paste into
cell.yaml `frames.table` — same [x, y, z, roll, pitch, yaw] convention the
viewer uses (meshcat.transformations 'sxyz'), verified by reconstruction.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from meshcat import transformations as tf


def fit_rigid(A, B):
    """Least-squares rigid transform T (4x4) with B ~ R @ A + t  (Kabsch).

    A, B: (N, 3) matched points. No scaling — the world is rigid; scale
    mismatch shows up honestly in the residuals instead of being absorbed.
    """
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))  # guard against reflection
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    args = ap.parse_args()

    data_file = args.data
    if data_file is None:
        sessions = sorted(Path(__file__).parent.glob("table_probe_*.yaml"))
        if not sessions:
            sys.exit("no table_probe_*.yaml found in calib/")
        data_file = sessions[-1]
    print(f"data: {data_file.name}")

    pts = yaml.safe_load(data_file.read_text())["points"]
    fit_pts = [p for p in pts if not p.get("holdout")]
    holdouts = [p for p in pts if p.get("holdout")]

    A = np.array([p["table"] for p in fit_pts])  # table frame
    B = np.array([p["base"] for p in fit_pts])   # base frame
    T = fit_rigid(A, B)                          # T_base_table

    print(f"\nfit on {len(fit_pts)} points — residuals (mm):")
    res = apply(T, A) - B
    for p, r in zip(fit_pts, res):
        print(f"  {p['name']}: [{r[0]*1e3:+6.2f} {r[1]*1e3:+6.2f} {r[2]*1e3:+6.2f}]"
              f"  |{np.linalg.norm(r)*1e3:5.2f}|")
    rms = np.sqrt((res ** 2).sum(axis=1).mean())
    print(f"  RMS {rms*1e3:.2f} mm   worst {np.linalg.norm(res, axis=1).max()*1e3:.2f} mm")

    for p in holdouts:
        pred = apply(T, np.array([p["table"]]))[0]
        err = pred - np.array(p["base"])
        print(f"\nholdout {p['name']}: predicted vs measured (mm): "
              f"[{err[0]*1e3:+6.2f} {err[1]*1e3:+6.2f} {err[2]*1e3:+6.2f}]"
              f"  |{np.linalg.norm(err)*1e3:5.2f}|")

    # pose in the exact convention cell/viewer.py uses (tf 'sxyz' euler)
    roll, pitch, yaw = tf.euler_from_matrix(T)
    x, y, z = T[:3, 3]
    T_check = tf.euler_matrix(roll, pitch, yaw)
    T_check[:3, 3] = [x, y, z]
    assert np.abs(T_check - T).max() < 1e-9, "euler convention self-check failed"

    print("\ncell.yaml frames.table:")
    print(f"    pose: [{x:.4f}, {y:.4f}, {z:.4f}, {roll:.5f}, {pitch:.5f}, {yaw:.5f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
