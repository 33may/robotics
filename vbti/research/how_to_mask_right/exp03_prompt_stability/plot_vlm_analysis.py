"""
VLM token analysis: variance maps, prompt effects, UMAP, similarity, sensitivity.
Reads vlm_tokens.npz and produces 5 plot types.
"""

import argparse
import os
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics.pairwise import cosine_similarity
import umap

# ── Constants ──────────────────────────────────────────────────────────────────

GRID_SIZE = 8
DPI = 300

DATASET_LABELS = {
    "fixed_duck_free_red": "duck fixed, cup moves",
    "free_duck_fixed_red": "duck moves, cup fixed",
    "free_duck_free_red":  "both move",
}

DATASET_COLORS = {
    "fixed_duck_free_red": "#e74c3c",
    "free_duck_fixed_red": "#2ecc71",
    "free_duck_free_red":  "#3498db",
}

PROMPT_MARKERS = ["o", "s", "^"]

SAVEKW = dict(dpi=DPI, facecolor="white", bbox_inches="tight")


def load_data(npz_path: str, camera: str):
    """Load vlm_tokens.npz and organize into nested dict[prompt_idx][dataset] = (N, 64, 960)."""
    data = np.load(npz_path, allow_pickle=True)
    datasets = list(data["datasets"])
    prompts = list(data["prompts"])
    cameras = list(data["cameras"])

    assert camera in cameras, f"Camera '{camera}' not in {cameras}"

    indices = {ds: data[f"indices_{ds}"] for ds in datasets}

    # tokens[prompt_idx][dataset] = list of arrays
    tokens = {pi: {ds: [] for ds in datasets} for pi in range(len(prompts))}

    for pi in range(len(prompts)):
        for ds in datasets:
            for ci in indices[ds]:
                key = f"vlm_tokens_{pi}_{ds}_{ci}_{camera}"
                tokens[pi][ds].append(data[key])
            tokens[pi][ds] = np.stack(tokens[pi][ds])  # (N, 64, 960)

    return tokens, datasets, prompts, indices


def plot_variance_grid(tokens, datasets, prompts, camera, out_dir):
    """Plot 1: 3x3 variance heatmaps (datasets x prompts)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Compute all variance maps first to find shared colorscale
    var_maps = {}
    for ri, ds in enumerate(datasets):
        for ci, pi in enumerate(range(len(prompts))):
            arr = tokens[pi][ds]  # (N, 64, 960)
            std_per_dim = np.std(arr, axis=0)  # (64, 960)
            var_map = np.linalg.norm(std_per_dim, axis=1)  # (64,)
            var_maps[(ri, ci)] = var_map.reshape(GRID_SIZE, GRID_SIZE)

    vmin = min(v.min() for v in var_maps.values())
    vmax = max(v.max() for v in var_maps.values())

    for ri, ds in enumerate(datasets):
        for ci in range(len(prompts)):
            ax = axes[ri, ci]
            im = ax.imshow(var_maps[(ri, ci)], cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"{DATASET_LABELS[ds]}\n\"{prompts[ci]}\"", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.6, label="L2 norm of per-dim std")
    fig.suptitle(f"VLM patch variance across captures — {camera} camera", fontsize=13)

    path = out_dir / f"vlm_variance_grid_{camera}.png"
    fig.savefig(path, **SAVEKW)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_prompt_effect(tokens, datasets, prompts, camera, out_dir):
    """Plot 2: variance difference between duck prompt and cup prompt."""
    # prompt indices: 0=full, 1=duck, 2=cup
    duck_pi, cup_pi = 1, 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    diffs = []
    for ds in datasets:
        for pi in [duck_pi, cup_pi]:
            arr = tokens[pi][ds]
            std_per_dim = np.std(arr, axis=0)
            var_map = np.linalg.norm(std_per_dim, axis=1).reshape(GRID_SIZE, GRID_SIZE)
            if pi == duck_pi:
                duck_var = var_map
            else:
                cup_var = var_map
        diffs.append(duck_var - cup_var)

    absmax = max(np.abs(d).max() for d in diffs)

    for i, (ds, diff) in enumerate(zip(datasets, diffs)):
        ax = axes[i]
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-absmax, vmax=absmax)
        ax.set_title(f"{DATASET_LABELS[ds]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.8, label="duck_prompt_var - cup_prompt_var")
    fig.suptitle(
        f"Prompt effect: \"pick up the duck\" vs \"place it in the cup\" — {camera}",
        fontsize=11,
    )

    path = out_dir / f"vlm_prompt_effect_{camera}.png"
    fig.savefig(path, **SAVEKW)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_umap(tokens, datasets, prompts, camera, out_dir):
    """Plot 3: UMAP of mean-pooled tokens colored by dataset, shaped by prompt."""
    embeddings = []
    ds_labels = []
    prompt_labels = []

    for pi in range(len(prompts)):
        for ds in datasets:
            arr = tokens[pi][ds]  # (N, 64, 960)
            pooled = arr.mean(axis=1)  # (N, 960)
            embeddings.append(pooled)
            ds_labels.extend([ds] * len(pooled))
            prompt_labels.extend([pi] * len(pooled))

    embeddings = np.concatenate(embeddings, axis=0)
    ds_labels = np.array(ds_labels)
    prompt_labels = np.array(prompt_labels)

    reducer = umap.UMAP(
        n_neighbors=10, min_dist=0.1, metric="cosine", random_state=42
    )
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))

    for pi in range(len(prompts)):
        for ds in datasets:
            mask = (ds_labels == ds) & (prompt_labels == pi)
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=DATASET_COLORS[ds],
                marker=PROMPT_MARKERS[pi],
                s=50, alpha=0.8, edgecolors="k", linewidths=0.3,
            )

    # Legend: datasets (color) + prompts (marker)
    ds_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DATASET_COLORS[ds],
               markersize=8, label=DATASET_LABELS[ds])
        for ds in datasets
    ]
    prompt_handles = [
        Line2D([0], [0], marker=PROMPT_MARKERS[pi], color="w", markerfacecolor="gray",
               markersize=8, label=prompts[pi])
        for pi in range(len(prompts))
    ]
    ax.legend(handles=ds_handles + prompt_handles, fontsize=7, loc="best")
    ax.set_title(f"UMAP of mean-pooled VLM tokens — {camera}", fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    path = out_dir / f"vlm_umap_{camera}.png"
    fig.savefig(path, **SAVEKW)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_similarity_by_prompt(tokens, datasets, prompts, camera, out_dir):
    """Plot 4: cosine similarity matrices within each prompt."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for pi, prompt in enumerate(prompts):
        # Gather all captures for this prompt, mean-pooled
        vecs = []
        tick_labels = []
        for ds in datasets:
            arr = tokens[pi][ds]  # (N, 64, 960)
            pooled = arr.mean(axis=1)  # (N, 960)
            vecs.append(pooled)
            for j in range(len(pooled)):
                tick_labels.append(f"{DATASET_LABELS[ds][:8]}_{j}")

        vecs = np.concatenate(vecs, axis=0)
        sim = cosine_similarity(vecs)

        ax = axes[pi]
        im = ax.imshow(sim, cmap="viridis", vmin=0.8, vmax=1.0)
        ax.set_title(f"\"{prompt}\"", fontsize=9)

        # Draw dataset boundaries
        boundaries = []
        offset = 0
        for ds in datasets:
            n = len(tokens[pi][ds])
            boundaries.append(offset + n)
            offset += n
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="white", linewidth=0.8)
            ax.axvline(b - 0.5, color="white", linewidth=0.8)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.7, label="cosine similarity")
    fig.suptitle(f"Within-prompt cosine similarity — {camera}", fontsize=12)

    path = out_dir / f"vlm_similarity_by_prompt_{camera}.png"
    fig.savefig(path, **SAVEKW)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_prompt_sensitivity(tokens, datasets, prompts, camera, out_dir):
    """Plot 5: per-patch prompt sensitivity (avg cosine distance between prompts)."""
    # For each capture (dataset, capture_idx), compute pairwise cosine distance
    # between prompts at each patch, then average across all pairs and captures.

    all_sensitivity = []  # list of (64,) arrays

    for ds in datasets:
        n_captures = tokens[0][ds].shape[0]
        for ci in range(n_captures):
            # Get tokens for each prompt: (64, 960)
            patch_vecs = [tokens[pi][ds][ci] for pi in range(len(prompts))]
            # Pairwise cosine distance at each patch
            patch_dists = np.zeros(64)
            n_pairs = 0
            for a, b in combinations(range(len(prompts)), 2):
                for patch_idx in range(64):
                    patch_dists[patch_idx] += cosine_dist(
                        patch_vecs[a][patch_idx], patch_vecs[b][patch_idx]
                    )
                n_pairs += 1
            patch_dists /= n_pairs
            all_sensitivity.append(patch_dists)

    sensitivity = np.mean(all_sensitivity, axis=0).reshape(GRID_SIZE, GRID_SIZE)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sensitivity, cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean cosine distance between prompts")
    ax.set_title(f"Prompt sensitivity per patch — {camera}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    path = out_dir / f"vlm_prompt_sensitivity_{camera}.png"
    fig.savefig(path, **SAVEKW)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="VLM token analysis plots")
    parser.add_argument("--camera", default="top", help="Camera to analyze")
    args = parser.parse_args()

    base = Path(__file__).parent
    npz_path = base / "vlm_tokens.npz"
    out_dir = base / "plots"
    out_dir.mkdir(exist_ok=True)

    tokens, datasets, prompts, indices = load_data(str(npz_path), args.camera)
    print(f"Loaded VLM tokens for camera={args.camera}")
    for ds in datasets:
        print(f"  {ds}: {tokens[0][ds].shape[0]} captures, shape={tokens[0][ds].shape}")

    plot_variance_grid(tokens, datasets, prompts, args.camera, out_dir)
    plot_prompt_effect(tokens, datasets, prompts, args.camera, out_dir)
    plot_umap(tokens, datasets, prompts, args.camera, out_dir)
    plot_similarity_by_prompt(tokens, datasets, prompts, args.camera, out_dir)
    plot_prompt_sensitivity(tokens, datasets, prompts, args.camera, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
