"""compare.py — overlay + score the Isaac vs MuJoCo actuator step responses.

Loads the JSON dumps from actuator_id_isaac.py and actuator_id_mujoco.py (same target),
prints a metrics table with each Isaac config's divergence from the MuJoCo reference, and
saves a q(t) overlay PNG. Pure numpy for the table; matplotlib optional for the plot.

    python logic/simulation/walkmatch/compare.py \
        --mujoco /tmp/actid_mujoco_knee.json --isaac /tmp/actid_isaac_knee.json \
        --png /tmp/actid_knee.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _resample(t, q, grid):
    return np.interp(grid, np.asarray(t), np.asarray(q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco", required=True)
    ap.add_argument("--isaac", required=True)
    ap.add_argument("--png", default="/tmp/actid_compare.png")
    args = ap.parse_args()

    mj = json.loads(Path(args.mujoco).read_text())
    isa = json.loads(Path(args.isaac).read_text())
    target = mj["meta"]["target"]
    t_step = mj["meta"]["t_step"]
    ref = mj["configs"]["mujoco"]
    grid = np.linspace(t_step, mj["meta"]["dt"] * (len(ref["t"]) - 1), 400)
    ref_q = _resample(ref["t"], ref["q"], grid)

    print(f"\n=== actuator step ID — target {target}, amp {mj['meta']['amp']} rad ===")
    print(f"{'config':18s} {'onset_ms':>9s} {'rise_ms':>8s} {'over_%':>7s} "
          f"{'ss_err':>8s} {'RMS_vs_MJ':>10s}")
    rm = ref["metrics"]
    print(f"{'MUJOCO (ref)':18s} {rm.get('onset_lag_ms', 0):9.1f} "
          f"{rm.get('rise_time_ms', 0):8.1f} {rm.get('overshoot_pct', 0):7.1f} "
          f"{rm.get('ss_err_rad', 0):+8.4f} {'—':>10s}")

    scored = []
    for name, cfg in isa["configs"].items():
        cq = _resample(cfg["t"], cfg["q"], grid)
        rms = float(np.sqrt(np.mean((cq - ref_q) ** 2)))
        m = cfg.get("metrics", {})
        scored.append((rms, name, m))
        print(f"{name:18s} {m.get('onset_lag_ms', float('nan')):9.1f} "
              f"{m.get('rise_time_ms', float('nan')):8.1f} "
              f"{m.get('overshoot_pct', float('nan')):7.1f} "
              f"{m.get('ss_err_rad', float('nan')):+8.4f} {rms:10.5f}")

    scored.sort()
    print(f"\nBEST MATCH to MuJoCo: {scored[0][1]} (RMS {scored[0][0]:.5f} rad)\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(grid, ref_q, "k-", lw=3, label="MuJoCo (reference)", zorder=10)
        ax.plot(grid, np.full_like(grid, mj["meta"]["q_final"]), "k:", lw=1, label="q_des")
        for _, name, _ in scored:
            cfg = isa["configs"][name]
            ax.plot(cfg["t"], cfg["q"], lw=1.5, alpha=0.85, label=f"isaac:{name}")
        ax.axvline(t_step, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"{target} q (rad)")
        ax.set_title(f"Actuator step response — Isaac configs vs MuJoCo ({target})")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.png, dpi=110)
        print(f"wrote {args.png}")
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
