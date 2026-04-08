"""Plotting for CLIP Surgery results. Reads similarity_maps.npz.

Separated from clip_surgery.py so you can tweak appearance without re-running
the model. Just edit and re-run this file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path

# ── Config — tweak these ────────────────────────────────────────────────────

DATA_DIR = Path("/home/may33/projects/ml_portfolio/robotics/vbti/research/how_to_mask_right/exp02_clip_surgery")
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Heatmap appearance
CMAP = "viridis"              # colormap for heatmaps (no red/blue — avoids confusion with objects)
HEATMAP_ALPHA = 0.6          # overlay transparency (0=invisible, 1=opaque)
IMAGE_ALPHA = 0.7             # base image brightness under overlay
DPI = 300
COLORBAR_SHRINK = 0.6

# ── Load data ───────────────────────────────────────────────────────────────

data = np.load(DATA_DIR / "similarity_maps.npz", allow_pickle=False)
frame_indices = data["frame_indices"]
prompts = list(data["prompts"])
cameras = list(data["cameras"])

PHASE_NAMES = ["start", "early", "mid", "late", "end"]


def get_surgery(frame_idx, cam):
    return data[f"surgery_{frame_idx}_{cam}"]

def get_raw(frame_idx, cam):
    return data[f"raw_{frame_idx}_{cam}"]

def get_image(frame_idx, cam):
    return data[f"image_{frame_idx}_{cam}"]


def _upscale_heatmap(heatmap_27x27, target_h, target_w):
    """Upscale 27x27 heatmap to image resolution via PIL bilinear."""
    pil = Image.fromarray(heatmap_27x27.astype(np.float32))
    return np.array(pil.resize((target_w, target_h), Image.Resampling.BILINEAR))


# ── Per-frame grid: 3 rows × 2 cols, each cell = 4 cameras ─────────────────

def plot_frame(frame_idx, phase_name, norm):
    """
    Layout: 3 rows × 2 cols = 6 cells.
      (0,0) = original   (0,1) = prompt 0
      (1,0) = prompt 1   (1,1) = prompt 2
      (2,0) = prompt 3   (2,1) = prompt 4

    Each cell is a horizontal strip of 4 camera images.
    """
    n_prompts = len(prompts)

    # Load all images for this frame
    imgs = {cam: get_image(frame_idx, cam) for cam in cameras}
    h, w = imgs[cameras[0]].shape[:2]

    # Build cells: list of (label, render_fn)
    cells = []

    # Cell 0: original images (no overlay)
    cells.append(("Original", None))

    # Cells 1-5: one per prompt
    for pi in range(n_prompts):
        cells.append((prompts[pi], pi))

    fig, axes = plt.subplots(3, 2, figsize=(30, 10),
                             gridspec_kw={"hspace": 0.25, "wspace": 0.08})

    for cell_idx, (label, prompt_idx) in enumerate(cells):
        row, col = divmod(cell_idx, 2)
        ax = axes[row, col]

        # Build horizontal strip of 4 cameras
        strips = []
        for cam in cameras:
            img = imgs[cam]
            if prompt_idx is not None:
                hmap = get_surgery(frame_idx, cam)[prompt_idx]
                hmap_up = _upscale_heatmap(hmap, h, w)

                # Blend: darken image, overlay heatmap with colormap
                img_float = img.astype(np.float32) / 255.0 * IMAGE_ALPHA
                cmap_fn = plt.cm.get_cmap(CMAP)
                hmap_normed = norm(hmap_up)
                hmap_rgba = cmap_fn(hmap_normed)[:, :, :3]
                blended = (1 - HEATMAP_ALPHA) * img_float + HEATMAP_ALPHA * hmap_rgba
                blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
                strips.append(blended)
            else:
                strips.append(img)

        # Add camera labels on the original row
        if prompt_idx is None:
            # Add small text labels at top of each camera
            for i, cam in enumerate(cameras):
                x_start = i * w + w // 2
                ax.text(x_start, -8, cam, ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="0.2")

        composite = np.concatenate(strips, axis=1)
        ax.imshow(composite)
        ax.set_title(label, fontsize=11, fontweight="bold", loc="left", pad=8)
        ax.axis("off")

    fig.suptitle(f"CLIP Surgery — Frame {frame_idx} ({phase_name})",
                 fontsize=15, fontweight="bold", y=0.98)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=COLORBAR_SHRINK, aspect=30,
                        pad=0.02, location="right")
    cbar.set_label("Surgery similarity", fontsize=10)

    path = PLOT_DIR / f"frame_{frame_idx}_{phase_name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Summary bar chart ───────────────────────────────────────────────────────

def plot_summary():
    mean_attn = np.zeros((len(prompts), len(cameras)))
    for pi in range(len(prompts)):
        for ci, cam in enumerate(cameras):
            vals = []
            for fidx in frame_indices:
                hmap = get_surgery(fidx, cam)[pi]
                vals.append(hmap[hmap > 0].mean() if (hmap > 0).any() else 0.0)
            mean_attn[pi, ci] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cameras))
    width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(prompts)))

    for pi, prompt in enumerate(prompts):
        offset = (pi - len(prompts) / 2 + 0.5) * width
        ax.bar(x + offset, mean_attn[pi], width, label=prompt, color=colors[pi])

    ax.set_xlabel("Camera", fontsize=12)
    ax.set_ylabel("Mean positive surgery similarity", fontsize=12)
    ax.set_title("Average CLIP Surgery Attribution by Prompt and Camera",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cameras)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    path = PLOT_DIR / "summary_mean_attention.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Temporal evolution ──────────────────────────────────────────────────────

def plot_temporal():
    fig, axes = plt.subplots(1, len(cameras), figsize=(4 * len(cameras), 4.5),
                             sharey=True)
    for ci, cam in enumerate(cameras):
        ax = axes[ci]
        for pi, prompt in enumerate(prompts):
            mean_vals = [get_surgery(fidx, cam)[pi].mean()
                         for fidx in frame_indices]
            ax.plot(range(len(frame_indices)), mean_vals,
                    marker="o", markersize=5, label=prompt, linewidth=1.5)
        ax.set_title(f"{cam} camera", fontweight="bold")
        ax.set_xlabel("Trajectory step")
        ax.set_xticks(range(len(frame_indices)))
        ax.set_xticklabels([f"{f}\n({p})" for f, p in
                            zip(frame_indices, PHASE_NAMES)], fontsize=7)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        if ci == 0:
            ax.set_ylabel("Mean surgery similarity")
        if ci == len(cameras) - 1:
            ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.2)

    fig.suptitle("CLIP Surgery: Temporal Evolution of Prompt Attribution",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    path = PLOT_DIR / "temporal_evolution.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Duck vs Cup overlap ────────────────────────────────────────────────────

def plot_duck_cup():
    duck_idx = prompts.index("a rubber duck")
    cup_idx = prompts.index("a red cup")
    cam = "top"

    fig, axes = plt.subplots(2, len(frame_indices),
                             figsize=(4 * len(frame_indices), 8))

    for fi_idx, fidx in enumerate(frame_indices):
        duck_map = get_surgery(fidx, cam)[duck_idx]
        cup_map = get_surgery(fidx, cam)[cup_idx]
        img = get_image(fidx, cam)
        h, w = img.shape[:2]

        # Row 0: duck - cup difference
        ax = axes[0, fi_idx]
        ax.imshow(img, alpha=0.5)
        diff = duck_map - cup_map
        d_up = _upscale_heatmap(diff, h, w)
        vabs = max(np.abs(diff).max(), 1e-6)
        ax.imshow(d_up, cmap="PiYG", vmin=-vabs, vmax=vabs, alpha=0.6)
        phase = PHASE_NAMES[fi_idx]
        ax.set_title(f"Frame {fidx}\n({phase})", fontsize=10)
        ax.axis("off")
        if fi_idx == 0:
            ax.set_ylabel("duck − cup\n(green=duck, pink=cup)",
                          fontsize=10, fontweight="bold")

        # Row 1: correlation scatter
        ax2 = axes[1, fi_idx]
        ax2.scatter(cup_map.ravel(), duck_map.ravel(), alpha=0.3, s=5,
                    c="steelblue")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.axvline(0, color="gray", lw=0.5)
        corr = np.corrcoef(duck_map.ravel(), cup_map.ravel())[0, 1]
        ax2.set_title(f"r = {corr:.3f}", fontsize=10)
        ax2.set_xlabel("cup sim", fontsize=9)
        if fi_idx == 0:
            ax2.set_ylabel("duck sim", fontsize=9)

    fig.suptitle("Duck vs Cup Spatial Attribution — Top Camera",
                 fontsize=14, fontweight="bold")
    path = PLOT_DIR / "duck_cup_overlap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Compute consistent color scale across all frames
    all_vals = np.concatenate([
        get_surgery(fidx, cam).ravel()
        for fidx in frame_indices for cam in cameras
    ])
    vmax = float(np.percentile(np.abs(all_vals), 97))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    print("Generating per-frame heatmap grids...")
    for fi, fidx in enumerate(frame_indices):
        phase = PHASE_NAMES[fi] if fi < len(PHASE_NAMES) else f"t{fi}"
        plot_frame(fidx, phase, norm)

    print("\nGenerating summary bar chart...")
    plot_summary()

    print("\nGenerating temporal evolution...")
    plot_temporal()

    print("\nGenerating duck-cup overlap...")
    plot_duck_cup()

    print("\nDone.")
