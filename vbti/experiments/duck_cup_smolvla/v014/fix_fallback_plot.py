"""Regenerate fallback_ratio.png with correct interpretation labels.

Formula: fb = mean(||pred - mean||) / mean(||GT - mean||)
- 0   = pred sits AT the action mean (full collapse to mean)
- 1   = pred is as far from mean as GT is (could be tracking GT, OR wandering)
- >1  = pred wanders even further from mean than GT
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
sys.path.insert(0, "/home/may33/projects/ml_portfolio/robotics")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

V014 = Path("/home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v014")
RAW = V014 / "sensitivity_raw.npz"
PLOTS = V014 / "sensitivity_plots"

data = np.load(RAW, allow_pickle=False)
gt = data["gt"]
action_mean = data["action_mean"]
gt_dist = np.linalg.norm(gt - action_mean[None], axis=1).mean()

# rebuild ordered condition list (same order as run)
CONDITIONS = [
    "control", "no_detection", "no_joints", "no_state",
    "no_images", "no_images_no_detection",
    "drop_top", "drop_left", "drop_right", "drop_gripper",
    "det_small_noise", "det_big_noise", "det_shuffled", "wrong_task",
]

fbs = []
for n in CONDITIONS:
    pred = data[n]
    fb = np.linalg.norm(pred - action_mean[None], axis=1).mean() / max(gt_dist, 1e-9)
    fbs.append(fb)
fbs = np.array(fbs)
names = CONDITIONS

# sort ascending — most-collapsed (low) at the top? Or bottom?
# matplotlib horizontal bars list bottom->top, so put the most collapsed at the BOTTOM
# i.e. ascending order by fb, lowest at bottom of plot.
order = np.argsort(fbs)[::-1]   # high fb at top, low fb at bottom

fig, ax = plt.subplots(figsize=(8.5, 7.5))
y = np.arange(len(names))
# colour: red for fb < 0.7 (clear collapse toward mean), grey otherwise
colors = ["tab:red" if fbs[i] < 0.7 else "tab:blue" for i in order]
ax.barh(y, fbs[order], color=colors)
ax.set_yticks(y)
ax.set_yticklabels([names[i] for i in order])

ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0,
           label="collapsed to action mean (=0)")
ax.axvline(1.0, color="black", linestyle=":", linewidth=1.0,
           label="GT-scale distance from mean (=1)")

ax.set_xlabel("fallback ratio  =  mean(||pred-mean||) / mean(||GT-mean||)\n"
              "0 = collapsed onto action mean    1 = same distance from mean as GT")
ax.set_title("v014 fall-back-to-action-mean ratio per condition\n"
             "(red bars: pred collapsed clearly toward action mean)")
for yi, vi in zip(y, fbs[order]):
    ax.text(vi + 0.01, yi, f"{vi:.2f}", va="center", fontsize=9)
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(0, max(1.05, fbs.max() * 1.05))
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "fallback_ratio.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("rewrote", PLOTS / "fallback_ratio.png")
print("fbs:", dict(zip(names, fbs.round(3))))
