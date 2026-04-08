"""Build and launch FiftyOne dataset from saved images + precomputed embeddings.

Memory-conscious: no duplicate embedding copies, PCA before UMAP,
batched sample ingestion.

Usage:
    python vbti/research/how_to_mask_right/explore_fiftyone.py [--port 5151]
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path

import fiftyone as fo
import numpy as np
import psutil
from tqdm import tqdm


def log_ram(label: str):
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1e9
    avail = psutil.virtual_memory().available / 1e9
    print(f"  [RAM] {label}: process={rss:.1f}GB, available={avail:.1f}GB")

IMG_DIR = Path(__file__).parent / "images"
RESULTS_DIR = Path(__file__).parent / "results"

DATASET_NAME = "masking_analysis"


def load_embeddings_ordered(dataset) -> np.ndarray | None:
    """Load npz embeddings and return array aligned to dataset sample order."""
    records_path = RESULTS_DIR / "records.npz"
    if not records_path.exists():
        print(f"Warning: {records_path} not found — skipping embeddings")
        return None

    print("Loading precomputed embeddings...")
    data = np.load(records_path, allow_pickle=False)
    # Materialize arrays into RAM (npz is lazy — repeated access re-decompresses)
    emb_array = data["embeddings"].copy()
    frame_idx_arr = data["frame_idx"].copy()
    cameras_arr = data["cameras"].copy()
    conditions_arr = data["conditions"].copy()
    data.close()

    n_records = len(conditions_arr)
    emb_dim = emb_array.shape[1]
    print(f"  NPZ: {n_records} records, embeddings shape={emb_array.shape}")
    log_ram("after npz load (materialized)")

    # Build lookup: (frame_idx, camera, condition) → index into arrays
    print("Building embedding lookup...")
    idx_lookup = {}
    for i in range(n_records):
        key = (int(frame_idx_arr[i]), str(cameras_arr[i]), str(conditions_arr[i]))
        idx_lookup[key] = i
    del frame_idx_arr, cameras_arr, conditions_arr
    log_ram("after lookup build")

    n_samples = len(dataset)
    print(f"Matching {n_samples} samples to {n_records} embeddings...")

    # Vectorized: pull all fields at once (no per-sample DB round-trips)
    frame_ids = dataset.values("frame_idx")
    cams = dataset.values("camera")
    conds = dataset.values("condition")
    log_ram("after values() fetch")

    embeddings = np.zeros((n_samples, emb_dim), dtype=np.float32)
    matched = 0
    for i in tqdm(range(n_samples), desc="Matching embeddings"):
        key = (frame_ids[i], cams[i], conds[i])
        npz_idx = idx_lookup.get(key)
        if npz_idx is not None:
            embeddings[i] = emb_array[npz_idx]
            matched += 1

    del emb_array, idx_lookup, frame_ids, cams, conds
    gc.collect()

    print(f"Matched {matched}/{n_samples} embeddings")
    log_ram("after embedding match + cleanup")
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument("--skip_umap", action="store_true",
                        help="Skip UMAP computation (use Embeddings panel in app instead)")
    args = parser.parse_args()

    # Delete existing dataset if it exists
    if fo.dataset_exists(DATASET_NAME):
        fo.delete_dataset(DATASET_NAME)

    dataset = fo.Dataset(DATASET_NAME, persistent=True)
    log_ram("after dataset init")

    # --- Load CSV manifest in batches ---
    csv_path = IMG_DIR / "manifest.csv"
    print(f"Loading manifest from {csv_path}...")

    BATCH = 5000
    batch = []
    total = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = fo.Sample(filepath=row["filepath"])
            sample["frame_idx"] = int(row["frame_idx"])
            sample["camera"] = row["camera"]
            sample["condition"] = row["condition"]
            sample["mask_shape"] = row["mask_shape"]
            sample["fill_mode"] = row["fill_mode"]
            sample["cos_sim"] = float(row["cos_sim"])
            sample["dataset_source"] = row.get("dataset_source", "01_02_black")
            batch.append(sample)

            if len(batch) >= BATCH:
                dataset.add_samples(batch)
                total += len(batch)
                print(f"  Added {total} samples...")
                batch.clear()

    if batch:
        dataset.add_samples(batch)
        total += len(batch)
        batch.clear()

    print(f"Added {total} samples total")
    log_ram("after sample ingestion")

    # --- Attach precomputed embeddings ---
    if not args.skip_umap:
        embeddings = load_embeddings_ordered(dataset)

        if embeddings is not None:
            import fiftyone.brain as fob

            print("Computing UMAP visualization...")
            log_ram("before UMAP")
            fob.compute_visualization(
                dataset,
                embeddings=embeddings,
                brain_key="siglip_umap",
                method="umap",
                seed=42,
            )
            del embeddings
            gc.collect()
            print("UMAP brain key registered: 'siglip_umap'")
            log_ram("after UMAP")

    # --- Print summary ---
    print(f"\nDataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")
    print(f"\nConditions: {dataset.distinct('condition')}")
    print(f"Cameras: {dataset.distinct('camera')}")
    print(f"Mask shapes: {dataset.distinct('mask_shape')}")
    print(f"Fill modes: {dataset.distinct('fill_mode')}")
    print(f"Dataset sources: {dataset.distinct('dataset_source')}")

    # --- Launch app ---
    session = fo.launch_app(dataset, port=args.port)
    print(f"\nFiftyOne app running at http://localhost:{args.port}")
    print("Press Ctrl+C to exit")
    session.wait()


if __name__ == "__main__":
    main()
