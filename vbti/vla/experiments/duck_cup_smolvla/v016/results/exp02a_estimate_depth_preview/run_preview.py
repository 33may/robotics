"""exp02a: depth-anything-v2 preview on gripper-cam frames for colormap selection.

Pure visualization study. No training. Picks colormap+clip range for v016 dataset
backfill (frozen depth as 4th camera). Outputs per-frame comparison grids and a
distribution histogram.
"""
from __future__ import annotations
import os, sys, json, traceback
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

DATASET_ROOT = Path("/home/may33/.cache/huggingface/lerobot/eternalmay33/01_02_03_merged_may-sim_detection")
OUT_DIR = Path("/home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v016/results/exp02a_estimate_depth_preview")
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_KEY = "observation.images.gripper"
EPISODE_INDICES = [10, 120, 230]  # early / middle / late
FRAMES_PER_EPISODE = 5            # 15 total samples
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Load LeRobot dataset and sample frames ----------
print("[1] Loading LeRobotDataset", flush=True)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="eternalmay33/01_02_03_merged_may-sim_detection",
    root=str(DATASET_ROOT),
)
print(f"    total episodes={ds.num_episodes}, total frames={ds.num_frames}", flush=True)

# Map episode -> list of (global_frame_idx, episode_local_frame_idx)
def sample_frames(ds, ep_idx, n):
    ep_len = int(ds.meta.episodes["length"][ep_idx])
    # evenly spaced, skipping very first/last frame
    locs = np.linspace(int(0.05 * ep_len), int(0.95 * ep_len) - 1, n).astype(int)
    # global indices: cumulative sum of lengths up to ep_idx
    starts = np.cumsum([0] + list(ds.meta.episodes["length"][:ep_idx]))
    g = int(starts[-1] if ep_idx > 0 else 0)
    return [(g + int(l), int(l)) for l in locs]

samples = []  # list of (ep_idx, local_idx, rgb_np_uint8)
for ep in EPISODE_INDICES:
    for g_idx, l_idx in sample_frames(ds, ep, FRAMES_PER_EPISODE):
        item = ds[g_idx]
        img = item[CAMERA_KEY]  # tensor C,H,W in [0,1]
        if isinstance(img, torch.Tensor):
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            arr = np.asarray(img)
        samples.append((ep, l_idx, arr))
        print(f"    sampled ep={ep} local={l_idx} shape={arr.shape}", flush=True)

print(f"[1] total samples: {len(samples)}", flush=True)

# ---------- 2. Load Depth Anything V2 (metric indoor small) ----------
print("[2] Loading depth model", flush=True)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_CANDIDATES = [
    ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", True),   # metric (m)
    ("depth-anything/Depth-Anything-V2-Small-hf", False),                # relative
]

model = None
processor = None
model_id = None
is_metric = False
for mid, metric in MODEL_CANDIDATES:
    try:
        print(f"    trying {mid}", flush=True)
        processor = AutoImageProcessor.from_pretrained(mid)
        model = AutoModelForDepthEstimation.from_pretrained(mid, torch_dtype=torch.float16)
        model = model.to(DEVICE).eval()
        model_id = mid
        is_metric = metric
        print(f"    loaded {mid} (metric={metric})", flush=True)
        break
    except Exception as e:
        print(f"    failed: {e}", flush=True)

if model is None:
    raise RuntimeError("could not load any depth model")

# ---------- 3. Run inference ----------
print("[3] Running depth inference", flush=True)
depth_maps = []  # list of np.float32 H,W matched to samples
with torch.no_grad():
    for ep, l_idx, rgb in samples:
        pil = Image.fromarray(rgb)
        inputs = processor(images=pil, return_tensors="pt").to(DEVICE)
        # cast pixel_values to fp16 to match model dtype
        inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        out = model(**inputs)
        pred = out.predicted_depth  # (1, h, w)
        # resize to original
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )[0, 0].cpu().numpy().astype(np.float32)
        depth_maps.append(pred)
        print(f"    ep={ep} local={l_idx} depth min={pred.min():.3f} max={pred.max():.3f} mean={pred.mean():.3f}", flush=True)

# ---------- 4. Render comparison grids ----------
print("[4] Rendering grids", flush=True)

def colorize(depth, vmin, vmax, cmap_name):
    cmap = mpl.colormaps[cmap_name]
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    rgba = cmap(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)

CONFIGS = [
    ("Turbo, clip [0.05, 0.5]m (spec)", "turbo", 0.05, 0.5),
    ("Turbo, clip [0.10, 1.0]m (spec)", "turbo", 0.10, 1.0),
    ("Turbo, clip [0.30, 2.0]m (data-fit)", "turbo", 0.30, 2.0),
    ("Viridis, clip [0.30, 2.0]m", "viridis", 0.30, 2.0),
    ("Gray, clip [0.30, 2.0]m", "gray", 0.30, 2.0),
    ("Inferno, clip [0.30, 2.0]m", "inferno", 0.30, 2.0),
]

for (ep, l_idx, rgb), depth in zip(samples, depth_maps):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    axes[0].imshow(rgb); axes[0].set_title("RGB"); axes[0].axis("off")

    # raw auto min-max
    dmin, dmax = float(depth.min()), float(depth.max())
    raw_norm = (depth - dmin) / (dmax - dmin + 1e-8)
    axes[1].imshow(raw_norm, cmap="gray")
    axes[1].set_title(f"Raw auto [{dmin:.2f}, {dmax:.2f}]" + (" m" if is_metric else " (rel)"))
    axes[1].axis("off")

    for i, (label, cmap, vmin, vmax) in enumerate(CONFIGS, start=2):
        img = colorize(depth, vmin, vmax, cmap)
        axes[i].imshow(img); axes[i].set_title(label); axes[i].axis("off")

    fig.suptitle(f"ep={ep} frame_local={l_idx}  model={model_id}  metric={is_metric}", fontsize=10)
    fig.tight_layout()
    fname = PLOTS_DIR / f"sample_ep{ep:03d}_f{l_idx:04d}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {fname.name}", flush=True)

# ---------- 5. Aggregate stats + histogram ----------
print("[5] Aggregating stats", flush=True)
all_depth = np.concatenate([d.flatten() for d in depth_maps])
stats = {
    "model_id": model_id,
    "is_metric": is_metric,
    "n_samples": len(samples),
    "n_pixels_total": int(all_depth.size),
    "min": float(all_depth.min()),
    "max": float(all_depth.max()),
    "mean": float(all_depth.mean()),
    "std": float(all_depth.std()),
    "p1": float(np.percentile(all_depth, 1)),
    "p5": float(np.percentile(all_depth, 5)),
    "p25": float(np.percentile(all_depth, 25)),
    "p50": float(np.percentile(all_depth, 50)),
    "p75": float(np.percentile(all_depth, 75)),
    "p95": float(np.percentile(all_depth, 95)),
    "p99": float(np.percentile(all_depth, 99)),
}
with open(OUT_DIR / "depth_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2), flush=True)

# Histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(all_depth, bins=100, log=True, color="steelblue", alpha=0.85)
unit = "m" if is_metric else "(relative)"
ax.set_xlabel(f"depth {unit}")
ax.set_ylabel("pixel count (log)")
ax.set_title(f"Depth distribution across {len(samples)} samples ({model_id})")
for q, label in [(stats["p5"], "p5"), (stats["p50"], "p50"), (stats["p95"], "p95")]:
    ax.axvline(q, color="red", ls="--", alpha=0.6)
    ax.text(q, ax.get_ylim()[1] * 0.5, label, color="red", fontsize=8, rotation=90)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "depth_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("    saved depth_histogram.png", flush=True)

print("[done]", flush=True)
