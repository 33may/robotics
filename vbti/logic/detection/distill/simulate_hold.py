"""Simulate hold-strategy detection for training data.

Takes the dense (stride-3 interpolated) detection_results.parquet and
produces a version where each episode samples detections at random intervals
(stride 5–20 frames), forward-filling between samples.  This mimics what
inference sees with async TRT detection + hold strategy.

Usage:
    python -m vbti.logic.detection.simulate_hold \
        eternalmay33/01_02_03_merged_may-sim \
        --stride-min 5 --stride-max 20 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from vbti.logic.dataset import resolve_dataset_path


COORD_COLS = [
    c for cam in ["left", "right", "top", "gripper"]
    for obj in ["duck", "cup"]
    for c in [f"{cam}_{obj}_cx", f"{cam}_{obj}_cy"]
]

CONF_COLS = [
    f"{cam}_{obj}_conf"
    for cam in ["left", "right", "top", "gripper"]
    for obj in ["duck", "cup"]
]


def simulate_hold_episode(
    ep_df: pd.DataFrame,
    stride_min: int,
    stride_max: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate hold for a single episode with variable stride.

    At each "arrival" frame, copy the real detection value.
    Between arrivals, forward-fill (hold last known position).
    The stride is sampled uniformly from [stride_min, stride_max] for each gap.
    """
    n = len(ep_df)
    if n == 0:
        return ep_df

    # Determine which frames are "arrivals"
    arrivals = set()
    pos = 0
    while pos < n:
        arrivals.add(pos)
        gap = rng.integers(stride_min, stride_max + 1)
        pos += gap

    result = ep_df.copy()
    cols = [c for c in COORD_COLS + CONF_COLS if c in result.columns]

    # Mask non-arrival frames to NaN, then forward-fill
    mask = np.ones(n, dtype=bool)
    for i in arrivals:
        mask[i] = False
    # mask[i] = True means "not an arrival" → set to NaN
    result.loc[result.index[mask], cols] = np.nan
    result[cols] = result[cols].ffill()
    # First frames before any arrival: backfill
    result[cols] = result[cols].bfill()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Simulate hold-strategy detection with variable stride",
    )
    parser.add_argument("dataset", help="Dataset repo_id")
    parser.add_argument("--stride-min", type=int, default=5,
                        help="Min frames between detection arrivals (default: 5)")
    parser.add_argument("--stride-max", type=int, default=20,
                        help="Max frames between detection arrivals (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: detection_results_hold.parquet)")
    args = parser.parse_args()

    ds_path = resolve_dataset_path(args.dataset)
    src = ds_path / "detection_results.parquet"
    if not src.exists():
        raise FileNotFoundError(f"No detection_results.parquet in {ds_path}")

    out_name = args.output or "detection_results_hold.parquet"
    dst = ds_path / out_name

    print(f"Source:     {src}")
    print(f"Output:     {dst}")
    print(f"Stride:     {args.stride_min}–{args.stride_max} frames (uniform)")
    print(f"Seed:       {args.seed}")

    df = pd.read_parquet(src)
    print(f"Loaded:     {len(df)} rows")

    rng = np.random.default_rng(args.seed)
    episodes = sorted(df["episode_index"].unique())
    print(f"Episodes:   {len(episodes)}")

    results = []
    for ep_idx in episodes:
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        held = simulate_hold_episode(ep_df, args.stride_min, args.stride_max, rng)
        results.append(held)

    out_df = pd.concat(results, ignore_index=True)

    # Stats: how different is this from original?
    coord_cols = [c for c in COORD_COLS if c in df.columns]
    orig_vals = df[coord_cols].values
    hold_vals = out_df[coord_cols].values
    diff = np.abs(orig_vals - hold_vals)
    valid = ~np.isnan(diff)
    if valid.any():
        print(f"\nDifference from dense (px at 560px):")
        print(f"  Mean:   {np.nanmean(diff) * 560:.1f} px")
        print(f"  Median: {np.nanmedian(diff) * 560:.1f} px")
        print(f"  P95:    {np.nanpercentile(diff, 95) * 560:.1f} px")

    out_df.to_parquet(dst, index=False)
    print(f"\nSaved: {dst} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
