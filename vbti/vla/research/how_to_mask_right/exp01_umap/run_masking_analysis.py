"""Masking embedding analysis — loads shards one at a time, embeds, plots.

Reads sharded frames from extract_frames.py output. Only one shard of images
is in RAM at a time; embeddings (4.5KB each) accumulate freely.

Random states are deterministic: seed = MASK_SEED + dataset_idx + cam_idx.
Any script can reproduce the exact mask from these values.

Usage:
    python vbti/research/how_to_mask_right/run_masking_analysis.py \
        --n_masked 500 --embed_batch 64
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from masking_lib import (
    FILL_MODES,
    SHAPES,
    get_embeddings_batch,
    load_siglip,
)

BLUR_K = 51
MASK_SEED = 42

CACHE_DIR = Path(__file__).parent / "cached_frames"
OUT_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Load shards → embed clean + masked
# ═══════════════════════════════════════════════════════════════════════════

def collect_embeddings(n_masked: int, embed_batch: int) -> list[dict]:
    # Load manifest
    with open(CACHE_DIR / "manifest.json") as f:
        manifest = json.load(f)

    cameras = manifest["cameras"]
    all_indices = manifest["frame_indices"]
    n_shards = manifest["n_shards"]

    # Pick which frames get masked (evenly spaced subset)
    masked_step = max(len(all_indices) // n_masked, 1)
    masked_indices_set = set(all_indices[::masked_step][:n_masked])

    n_conds = len(SHAPES) * len(FILL_MODES)
    n_clean_imgs = len(all_indices) * len(cameras)
    n_masked_imgs = len(masked_indices_set) * len(cameras) * n_conds
    print(f"Clean frames: {len(all_indices)}, Masked frames: {len(masked_indices_set)}")
    print(f"Conditions: {len(SHAPES)} shapes x {len(FILL_MODES)} fills = {n_conds}")
    print(f"Total images to embed: {n_clean_imgs + n_masked_imgs}")

    # Load SigLIP once
    model, processor, device = load_siglip()

    all_records = []

    for shard_i in tqdm(range(n_shards), desc="Processing shards"):
        shard = torch.load(CACHE_DIR / f"shard_{shard_i:04d}.pt", weights_only=False)

        imgs_to_embed = []
        meta = []

        for dataset_idx, cam_dict in shard.items():
            should_mask = dataset_idx in masked_indices_set

            for cam_idx, cam in enumerate(cameras):
                img = cam_dict[cam]

                # Clean embedding
                imgs_to_embed.append(img)
                meta.append({"frame_idx": dataset_idx, "camera": cam,
                             "condition": "clean"})

                # Masked variants (deterministic seed)
                if should_mask:
                    for shape_name, shape_fn in SHAPES.items():
                        for fill in FILL_MODES:
                            torch.manual_seed(MASK_SEED + dataset_idx + cam_idx)
                            masked_img = shape_fn(img, fill, BLUR_K)
                            imgs_to_embed.append(masked_img)
                            meta.append({
                                "frame_idx": dataset_idx, "camera": cam,
                                "condition": f"{shape_name}_{fill}",
                            })

        # Embed this shard's images, then free them
        embs = get_embeddings_batch(imgs_to_embed, model, processor, device,
                                    batch_size=embed_batch)
        del imgs_to_embed, shard
        gc.collect()

        for i, m in enumerate(meta):
            all_records.append({**m, "embedding": embs[i]})

    # Free model
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    # Cosine sims
    clean_lookup = {}
    for i, r in enumerate(all_records):
        if r["condition"] == "clean":
            clean_lookup[(r["frame_idx"], r["camera"])] = i

    for r in all_records:
        if r["condition"] == "clean":
            r["cos_sim"] = 1.0
        else:
            ci = clean_lookup[(r["frame_idx"], r["camera"])]
            clean_emb = all_records[ci]["embedding"]
            emb = r["embedding"]
            r["cos_sim"] = float(np.dot(clean_emb, emb) /
                                 (np.linalg.norm(clean_emb) * np.linalg.norm(emb) + 1e-8))

    # Summary
    print(f"\nTotal records: {len(all_records)}")
    print("Mean cosine similarity per condition:")
    cond_sims = defaultdict(list)
    for r in all_records:
        if r["condition"] != "clean":
            cond_sims[r["condition"]].append(r["cos_sim"])
    for cond in sorted(cond_sims):
        sims = cond_sims[cond]
        print(f"  {cond:35s}  mean={np.mean(sims):.4f}  "
              f"std={np.std(sims):.4f}  n={len(sims)}")

    return all_records


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: UMAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_umap(records: list[dict], cameras: list[str]):
    import umap

    all_embs = np.array([r["embedding"] for r in records])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=42)
    coords = reducer.fit_transform(all_embs)

    for i, r in enumerate(records):
        r["umap_x"] = coords[i, 0]
        r["umap_y"] = coords[i, 1]

    all_conditions = sorted(set(r["condition"] for r in records
                                if r["condition"] != "clean"))
    palette = px.colors.qualitative.T10
    color_map = {c: palette[i % len(palette)]
                 for i, c in enumerate(all_conditions)}

    n_clean = sum(1 for r in records if r["condition"] == "clean")
    n_masked = sum(1 for r in records if r["condition"] != "clean")

    fig = make_subplots(rows=1, cols=4, subplot_titles=cameras,
                        horizontal_spacing=0.04)

    for col_idx, cam in enumerate(cameras, 1):
        cam_recs = [r for r in records if r["camera"] == cam]
        clean_recs = [r for r in cam_recs if r["condition"] == "clean"]
        fig.add_trace(go.Scatter(
            x=[r["umap_x"] for r in clean_recs],
            y=[r["umap_y"] for r in clean_recs],
            mode="markers",
            marker=dict(size=3, color="grey", opacity=0.3),
            name="clean", legendgroup="clean",
            text=[f"frame={r['frame_idx']}" for r in clean_recs],
            hovertemplate="%{text}<extra>clean</extra>",
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

        for cond in all_conditions:
            cond_recs = [r for r in cam_recs if r["condition"] == cond]
            if not cond_recs:
                continue
            fig.add_trace(go.Scatter(
                x=[r["umap_x"] for r in cond_recs],
                y=[r["umap_y"] for r in cond_recs],
                mode="markers",
                marker=dict(size=4, color=color_map[cond], opacity=0.7),
                name=cond, legendgroup=cond,
                text=[f"frame={r['frame_idx']}<br>cos={r['cos_sim']:.3f}"
                      for r in cond_recs],
                hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(
        title=f"UMAP — {len(all_conditions)} masked conditions vs "
              f"{n_clean} clean (N_MASKED={n_masked})",
        height=600, width=1800,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.write_html(OUT_DIR / "umap.html")
    print(f"  → {OUT_DIR / 'umap.html'}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Cosine violin + summary table
# ═══════════════════════════════════════════════════════════════════════════

def plot_cosine_violin(records: list[dict], cameras: list[str]):
    masked_records = [r for r in records if r["condition"] != "clean"]
    all_conditions = sorted(set(r["condition"] for r in masked_records))
    palette = px.colors.qualitative.T10
    color_map = {c: palette[i % len(palette)]
                 for i, c in enumerate(all_conditions)}

    fig = make_subplots(rows=1, cols=4, subplot_titles=cameras,
                        horizontal_spacing=0.06)

    for col_idx, cam in enumerate(cameras, 1):
        cam_masked = [r for r in masked_records if r["camera"] == cam]
        for cond in all_conditions:
            cond_recs = [r for r in cam_masked if r["condition"] == cond]
            if not cond_recs:
                continue
            fig.add_trace(go.Violin(
                x=[cond] * len(cond_recs),
                y=[r["cos_sim"] for r in cond_recs],
                name=cond, legendgroup=cond,
                showlegend=(col_idx == 1),
                line_color=color_map[cond],
                spanmode="hard", meanline_visible=True,
            ), row=1, col=col_idx)

    fig.update_layout(
        title="Cosine Similarity Distribution per Condition per Camera",
        height=600, width=1800,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        violinmode="overlay",
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    fig.update_yaxes(title_text="cos_sim", col=1)
    fig.write_html(OUT_DIR / "cosine_violin.html")
    print(f"  → {OUT_DIR / 'cosine_violin.html'}")

    # Summary table
    print(f"\n{'CONDITION':<35s}", end="")
    for cam in cameras:
        print(f"  {cam:>12s} (med/IQR/min)", end="")
    print("\n" + "=" * 100)
    for cond in all_conditions:
        print(f"{cond:<35s}", end="")
        for cam in cameras:
            sims = np.array([r["cos_sim"] for r in masked_records
                             if r["condition"] == cond and r["camera"] == cam])
            if len(sims) == 0:
                print(f"  {'n/a':>30s}", end="")
            else:
                med = np.median(sims)
                q25, q75 = np.percentile(sims, [25, 75])
                print(f"  {med:6.4f} / {q75-q25:5.4f} / {sims.min():6.4f}", end="")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: k-NN intrusion + MMD heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def plot_knn_mmd(records: list[dict], cameras: list[str]):
    all_conditions = sorted(set(r["condition"] for r in records
                                if r["condition"] != "clean"))
    K = 10

    knn_results = {}
    for cam in cameras:
        cam_recs = [r for r in records if r["camera"] == cam]
        cam_embs = np.array([r["embedding"] for r in cam_recs])
        cam_conds = [r["condition"] for r in cam_recs]
        is_clean = np.array([c == "clean" for c in cam_conds])

        nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine",
                              algorithm="brute")
        nn.fit(cam_embs)

        masked_indices = [i for i, c in enumerate(cam_conds) if c != "clean"]
        if not masked_indices:
            continue

        _, neighbors = nn.kneighbors(cam_embs[masked_indices])
        for local_idx, global_idx in enumerate(masked_indices):
            cond = cam_conds[global_idx]
            neigh_idx = neighbors[local_idx]
            neigh_idx = neigh_idx[neigh_idx != global_idx][:K]
            frac_clean = is_clean[neigh_idx].mean()
            knn_results.setdefault((cond, cam), []).append(frac_clean)

    knn_matrix = np.zeros((len(all_conditions), len(cameras)))
    for i, cond in enumerate(all_conditions):
        for j, cam in enumerate(cameras):
            vals = knn_results.get((cond, cam), [])
            knn_matrix[i, j] = np.mean(vals) if vals else 0.0

    # MMD
    sample_embs = np.array([r["embedding"] for r in records])
    subsample = sample_embs[np.random.choice(
        len(sample_embs), min(500, len(sample_embs)), replace=False)]
    median_dist = np.median(pdist(subsample, metric="euclidean"))
    gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0

    mmd_matrix = np.zeros((len(all_conditions), len(cameras)))
    for j, cam in enumerate(cameras):
        clean_embs = np.array([r["embedding"] for r in records
                               if r["camera"] == cam and r["condition"] == "clean"])
        for i, cond in enumerate(all_conditions):
            masked_embs = np.array([r["embedding"] for r in records
                                    if r["camera"] == cam
                                    and r["condition"] == cond])
            if len(masked_embs) == 0 or len(clean_embs) == 0:
                continue
            Kxx = rbf_kernel(clean_embs, clean_embs, gamma=gamma)
            Kyy = rbf_kernel(masked_embs, masked_embs, gamma=gamma)
            Kxy = rbf_kernel(clean_embs, masked_embs, gamma=gamma)
            mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
            mmd_matrix[i, j] = max(mmd2, 0.0) ** 0.5

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["k-NN Clean Intrusion Rate (k=10)", "MMD (RBF kernel)"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Heatmap(
        z=knn_matrix, x=cameras, y=all_conditions,
        colorscale="Greens", zmin=0, zmax=1,
        text=np.round(knn_matrix, 3).astype(str), texttemplate="%{text}",
        colorbar=dict(x=0.45, len=0.9),
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=mmd_matrix, x=cameras, y=all_conditions,
        colorscale="Reds",
        text=np.round(mmd_matrix, 4).astype(str), texttemplate="%{text}",
        colorbar=dict(x=1.0, len=0.9),
    ), row=1, col=2)
    fig.update_layout(height=500, width=1200,
                      title="Distribution Metrics: k-NN Intrusion + MMD")
    fig.write_html(OUT_DIR / "knn_mmd.html")
    print(f"  → {OUT_DIR / 'knn_mmd.html'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Masking embedding analysis")
    parser.add_argument("--n_masked", type=int, default=500,
                        help="Frames to apply masks to (subset of cached clean)")
    parser.add_argument("--embed_batch", type=int, default=64,
                        help="SigLIP embedding batch size")
    parser.add_argument("--skip_umap", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Read cameras from manifest
    with open(CACHE_DIR / "manifest.json") as f:
        cameras = json.load(f)["cameras"]

    print("── Phase 1: Embedding from shards ──")
    records = collect_embeddings(args.n_masked, args.embed_batch)

    if not args.skip_umap:
        print("\n── Phase 2: UMAP ──")
        plot_umap(records, cameras)

    print("\n── Phase 3: Cosine similarity ──")
    plot_cosine_violin(records, cameras)

    print("\n── Phase 4: k-NN intrusion + MMD ──")
    plot_knn_mmd(records, cameras)

    # Save for reuse
    np.savez_compressed(
        OUT_DIR / "records.npz",
        embeddings=np.array([r["embedding"] for r in records]),
        cos_sims=np.array([r["cos_sim"] for r in records]),
        frame_idx=np.array([r["frame_idx"] for r in records]),
        cameras=np.array([r["camera"] for r in records]),
        conditions=np.array([r["condition"] for r in records]),
    )
    print(f"\n  → Saved to {OUT_DIR / 'records.npz'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
