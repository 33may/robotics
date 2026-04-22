"""Plot inference speed benchmark results for a distillation run.

Usage:
    python -u scripts/distill_plot_speed.py --run m1_baseline
    python -u scripts/distill_plot_speed.py --run m1_baseline --compare m2_aug
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAINING_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/training"
)
CAMERAS = ["left", "right", "top", "gripper"]
TARGET_MS = 5.0   # sweep spec: <5 ms/frame single cam
DINO_MS   = 100.0 # teacher reference latency


def load_bench(run: str) -> dict:
    p = TRAINING_ROOT / run / "inference_bench.json"
    if not p.exists():
        raise FileNotFoundError(f"No inference_bench.json for run '{run}'. Run distill_bench_speed.py first.")
    return json.loads(p.read_text())


def plot_speed(runs: list[str]) -> Path:
    """Bar chart: per-cam latency for each run, plus 4-cam total and reference lines."""
    data = {r: load_bench(r) for r in runs}

    n_runs = len(runs)
    n_cams = len(CAMERAS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Student Inference Latency — RTX 4070 Ti SUPER", fontsize=13, y=1.01)

    # ── Left: per-cam b1 grouped bar chart ──
    ax = axes[0]
    x = np.arange(n_cams)
    bar_w = 0.7 / max(n_runs, 1)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, run in enumerate(runs):
        b1 = data[run]["per_cam_ms_b1"]
        vals = [b1[c] for c in CAMERAS]
        offset = (i - (n_runs - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w * 0.9, label=run, color=colors[i], zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color="black"
            )

    ax.axhline(TARGET_MS, color="red", linestyle="--", linewidth=1.2, label=f"target {TARGET_MS} ms", zorder=2)
    ax.axhline(DINO_MS, color="gray", linestyle=":", linewidth=1.0, label=f"G-DINO ~{DINO_MS:.0f} ms", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(CAMERAS)
    ax.set_ylabel("Latency (ms/frame)")
    ax.set_title("Single-cam latency (batch=1, 200 iters)")
    ax.set_ylim(0, min(TARGET_MS * 1.8, max(
        max(data[r]["per_cam_ms_b1"][c] for r in runs for c in CAMERAS) * 1.4, TARGET_MS * 1.2
    )))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # ── Right: 4-cam total + speedup vs DINO ──
    ax2 = axes[1]
    four_cam_vals = [data[r]["four_cam_sequential_ms"] for r in runs]
    bar_colors = [colors[i] for i in range(n_runs)]
    bars2 = ax2.bar(runs, four_cam_vals, color=bar_colors, width=0.4, zorder=3)
    for bar, v in zip(bars2, four_cam_vals):
        speedup = DINO_MS / v
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{v:.2f} ms\n{speedup:.0f}× faster\nthan G-DINO",
            ha="center", va="bottom", fontsize=9
        )

    ax2.axhline(TARGET_MS * 4, color="red", linestyle="--", linewidth=1.2,
                label=f"4× target = {TARGET_MS*4:.0f} ms", zorder=2)
    ax2.set_ylabel("Total latency all 4 cams (ms)")
    ax2.set_title("4-cam sequential (1 inference cycle)")
    ax2.set_ylim(0, max(four_cam_vals) * 2.2)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    suffix = "_vs_".join(runs)
    out_path = TRAINING_ROOT / runs[0] / f"inference_speed_{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[speed] saved {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="primary run name")
    p.add_argument("--compare", nargs="*", default=[], help="additional runs to overlay")
    args = p.parse_args()

    runs = [args.run] + (args.compare or [])
    plot_speed(runs)
    print("done.")


if __name__ == "__main__":
    main()
