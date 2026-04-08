"""Stage 0: Visualize raw SigLIP patch token variance + UMAP baseline.

No text, no VLM — pure vision encoder output. This establishes:
1. Per-patch variance maps: which spatial regions change across captures?
2. UMAP: do the 3 datasets form separate clusters in SigLIP space?

Usage:
    python plot_siglip_baseline.py
    python plot_siglip_baseline.py --camera left
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────

EXP_DIR = Path("/home/may33/projects/ml_portfolio/robotics/vbti/research/how_to_mask_right/exp03_prompt_stability")
PLOT_DIR = EXP_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "fixed_duck_free_red": EXP_DIR / "captures_fixed_duck_free_red",
    "free_duck_fixed_red": EXP_DIR / "captures_free_duck_fixed_red",
    "free_duck_free_red":  EXP_DIR / "captures_free_duck_free_red",
}

DATASET_COLORS = {
    "fixed_duck_free_red": "#e74c3c",  # red — cup moves
    "free_duck_fixed_red": "#2ecc71",  # green — duck moves
    "free_duck_free_red":  "#3498db",  # blue — both move
}

DATASET_LABELS = {
    "fixed_duck_free_red": "duck fixed, cup moves",
    "free_duck_fixed_red": "duck moves, cup fixed",
    "free_duck_free_red":  "both move",
}

GRID_SIZE = 27  # SigLIP: 384/14 = 27x27 = 729 patches
DPI = 300


# ── Load data ───────────────────────────────────────────────────────────────

def load_tokens(camera):
    """Load SigLIP tokens for one camera, grouped by dataset.
    Returns {dataset_name: np.array (N_images, 729, 1152)}
    """
    data = np.load(EXP_DIR / "siglip_tokens.npz", allow_pickle=False)
    result = {}
    for ds_name in DATASETS:
        indices = data[f"indices_{ds_name}"]
        tokens_list = []
        for idx in indices:
            key = f"tokens_{ds_name}_{idx}_{camera}"
            tokens_list.append(data[key])
        result[ds_name] = np.stack(tokens_list)  # (N, 729, 1152)
    data.close()
    return result


def load_sample_image(ds_name, index, camera):
    """Load a sample capture image for overlay."""
    img_path = DATASETS[ds_name] / f"{index:03d}_{camera}.png"
    return np.array(Image.open(img_path).convert("RGB"))


# ── Plot 1: Per-patch variance heatmaps ─────────────────────────────────────

def plot_variance_maps(tokens_by_dataset, camera):
    """Per-patch variance across images within each dataset.
    High variance = this spatial region changes across captures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Compute variance maps
    var_maps = {}
    for ds_name, tokens in tokens_by_dataset.items():
        # tokens: (N, 729, 1152)
        # Per-patch: variance of the mean embedding magnitude, or L2 norm of std
        # Use: std across images per patch per dim, then L2 norm across dims
        std_per_dim = tokens.std(axis=0)        # (729, 1152)
        var_map = np.linalg.norm(std_per_dim, axis=1)  # (729,)
        var_maps[ds_name] = var_map.reshape(GRID_SIZE, GRID_SIZE)

    # Shared colorscale
    vmax = max(v.max() for v in var_maps.values())

    for ax, (ds_name, var_map) in zip(axes, var_maps.items()):
        im = ax.imshow(var_map, cmap="magma", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(DATASET_LABELS[ds_name], fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.8, label="Patch embedding std (L2 norm)")
    fig.suptitle(f"SigLIP Patch Variance — {camera} camera (no text, raw encoder)",
                 fontsize=13, fontweight="bold")

    path = PLOT_DIR / f"siglip_variance_{camera}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Plot 2: Variance overlaid on sample image ──────────────────────────────

def plot_variance_overlay(tokens_by_dataset, camera):
    """Variance heatmap overlaid on a sample image from each dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.05})

    for col, (ds_name, tokens) in enumerate(tokens_by_dataset.items()):
        # Load first capture as reference image
        data = np.load(EXP_DIR / "siglip_tokens.npz", allow_pickle=False)
        first_idx = int(data[f"indices_{ds_name}"][0])
        data.close()
        img = load_sample_image(ds_name, first_idx, camera)
        h, w = img.shape[:2]

        # Variance map
        std_per_dim = tokens.std(axis=0)
        var_map = np.linalg.norm(std_per_dim, axis=1).reshape(GRID_SIZE, GRID_SIZE)

        # Row 0: original image
        axes[0, col].imshow(img)
        axes[0, col].set_title(DATASET_LABELS[ds_name], fontsize=11, fontweight="bold")
        axes[0, col].axis("off")

        # Row 1: overlay
        axes[1, col].imshow(img, alpha=0.6)
        var_up = np.array(Image.fromarray(var_map.astype(np.float32)).resize(
            (w, h), Image.Resampling.BILINEAR))
        im = axes[1, col].imshow(var_up, cmap="magma", alpha=0.55,
                                  vmin=0, vmax=var_map.max())
        axes[1, col].axis("off")

    fig.colorbar(im, ax=axes, shrink=0.4, aspect=20, label="Patch std (L2)")
    fig.suptitle(f"SigLIP Patch Variance Overlay — {camera} camera",
                 fontsize=13, fontweight="bold")

    path = PLOT_DIR / f"siglip_variance_overlay_{camera}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Plot 3: UMAP colored by dataset ────────────────────────────────────────

def plot_umap(tokens_by_dataset, camera):
    """UMAP of mean-pooled SigLIP tokens, colored by dataset."""
    import umap

    # Mean-pool vision tokens per image: (N, 729, 1152) → (N, 1152)
    all_embeddings = []
    all_labels = []
    all_colors = []

    for ds_name, tokens in tokens_by_dataset.items():
        pooled = tokens.mean(axis=1)  # (N, 1152)
        all_embeddings.append(pooled)
        all_labels.extend([DATASET_LABELS[ds_name]] * len(pooled))
        all_colors.extend([DATASET_COLORS[ds_name]] * len(pooled))

    all_embeddings = np.concatenate(all_embeddings)  # (55, 1152)
    print(f"  UMAP on {all_embeddings.shape[0]} points ({all_embeddings.shape[1]}-d)...")

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot per dataset for legend
    offset = 0
    for ds_name, tokens in tokens_by_dataset.items():
        n = len(tokens)
        ax.scatter(coords[offset:offset + n, 0], coords[offset:offset + n, 1],
                   c=DATASET_COLORS[ds_name], label=DATASET_LABELS[ds_name],
                   s=40, alpha=0.7, edgecolors="white", linewidths=0.5)
        offset += n

    ax.set_title(f"UMAP — Raw SigLIP Embeddings (mean-pooled) — {camera} camera",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    path = PLOT_DIR / f"siglip_umap_{camera}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Plot 4: Pairwise cosine similarity matrices ────────────────────────────

def plot_similarity_matrices(tokens_by_dataset, camera):
    """Per-dataset cosine similarity matrix of mean-pooled tokens."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (ds_name, tokens) in zip(axes, tokens_by_dataset.items()):
        pooled = tokens.mean(axis=1)  # (N, 1152)
        # Normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled_normed = pooled / (norms + 1e-8)
        sim = pooled_normed @ pooled_normed.T  # (N, N)

        im = ax.imshow(sim, cmap="RdYlGn", vmin=0.85, vmax=1.0)
        ax.set_title(f"{DATASET_LABELS[ds_name]}\nmean cos={sim[np.triu_indices(len(sim), k=1)].mean():.4f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Capture index")
        ax.set_ylabel("Capture index")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Cosine similarity")
    fig.suptitle(f"SigLIP Pairwise Similarity — {camera} camera (raw, no text)",
                 fontsize=13, fontweight="bold")

    path = PLOT_DIR / f"siglip_similarity_{camera}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="top",
                        choices=["top", "left", "right", "gripper"])
    args = parser.parse_args()

    print(f"Loading SigLIP tokens for {args.camera} camera...")
    tokens_by_dataset = load_tokens(args.camera)
    for ds_name, tokens in tokens_by_dataset.items():
        print(f"  {ds_name}: {tokens.shape}")

    print("\nPlot 1: Patch variance maps...")
    plot_variance_maps(tokens_by_dataset, args.camera)

    print("Plot 2: Variance overlay on sample images...")
    plot_variance_overlay(tokens_by_dataset, args.camera)

    print("Plot 3: UMAP...")
    plot_umap(tokens_by_dataset, args.camera)

    print("Plot 4: Pairwise cosine similarity...")
    plot_similarity_matrices(tokens_by_dataset, args.camera)

    print("\nDone. All plots in:", PLOT_DIR)
