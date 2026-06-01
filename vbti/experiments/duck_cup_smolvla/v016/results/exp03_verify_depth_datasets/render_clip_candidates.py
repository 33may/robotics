"""Render a few canonical frames at multiple candidate clip ranges.

Goal: pick a clip whose LUT span is concentrated on the duck/cup grasp region
(~0.05-0.20 m) rather than the table/floor (>0.30 m).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vbti.logic.depth.colorize import colorize_fixed_clip


REPO = "eternalmay33/04_05_06_07_merged_may-sim_depth"
KEY = "observation.images.gripper_depth"
GRIPPER_KEY = "observation.images.gripper"
DEPTH_SCALE_M = 1e-4

# Mixed phases across 3 episodes drawn from different source cohorts:
# approach (25% of episode), grasp (55%), lift (80%).
# ep=0 (cohort 04), ep=97 (cohort 06), ep=108 (cohort 06)
FRAMES_LABELED = [
    (135,   "ep0  approach"),
    (298,   "ep0  grasp"),
    (433,   "ep0  lift"),
    (37852, "ep97 approach"),
    (37960, "ep97 grasp"),
    (38050, "ep97 lift"),
    (42213, "ep108 approach"),
    (42372, "ep108 grasp"),
    (42505, "ep108 lift"),
]
FRAMES = [f for f, _ in FRAMES_LABELED]
LABELS = {f: lbl for f, lbl in FRAMES_LABELED}

# Candidate clips: (label, clip_min_m, clip_max_m)
CLIPS = [
    ("[0.05, 0.20] m  (tight — duck-only)",   0.05, 0.20),
    ("[0.05, 0.30] m  (focused)",              0.05, 0.30),
    ("[0.05, 0.50] m  (moderate)",             0.05, 0.50),
    ("[0.07, 0.95] m  (current canonical)",    0.07, 0.95),
]

OUT_PATH = Path(__file__).parent / "plots" / "clip_candidates.png"


def _depth_to_uint16_meters(t: torch.Tensor) -> np.ndarray:
    """t: (3, H, W) float in [0,1] from LeRobotDataset → metric depth (H, W) float32."""
    arr = (t.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    high = arr[0].astype(np.uint16)
    low = arr[1].astype(np.uint16)
    d_u16 = (high << 8) | low
    return d_u16.astype(np.float32) * DEPTH_SCALE_M


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[load] {REPO}")
    ds = LeRobotDataset(REPO)
    print(f"[load] {len(ds)} frames; rendering {len(FRAMES)} × {len(CLIPS)} grid")

    n_rows = len(FRAMES)
    n_cols = len(CLIPS) + 1  # +1 for the gripper RGB context column
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for r, idx in enumerate(FRAMES):
        item = ds[idx]
        rgb_t = item[GRIPPER_KEY]                          # (3, H, W) float [0,1]
        depth_t = item[KEY]                                # (3, H, W) float [0,1]
        rgb = rgb_t.permute(1, 2, 0).numpy()
        depth_m = _depth_to_uint16_meters(depth_t)

        ax = axes[r, 0]
        ax.imshow(rgb)
        ax.set_title(f"{LABELS.get(idx, '')} (f={idx})\nRGB gripper", fontsize=8)
        ax.axis("off")

        for c, (label, lo, hi) in enumerate(CLIPS):
            colored = colorize_fixed_clip(depth_m, lo, hi)  # (H, W, 3) uint8
            ax = axes[r, c + 1]
            ax.imshow(colored)
            if r == 0:
                ax.set_title(label, fontsize=8)
            ax.axis("off")

    fig.suptitle(
        "Candidate depth clips on merged 04+05+06+07 — pick whichever puts color span on the duck/cup",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[done] {OUT_PATH}")


if __name__ == "__main__":
    main()
