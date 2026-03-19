"""
Dataset loading and splitting utilities.

Functions for loading LeRobot datasets from HuggingFace and creating
train/validation splits with proper shuffling and episode filtering.

Supports single-dataset and multi-dataset (weighted) loading.
"""
import random
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def load_and_split_dataset(
    repo_id: str,
    root: Optional[Path] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    delta_timestamps: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[LeRobotDataset, LeRobotDataset, LeRobotDataset]:
    """
    Load dataset and split into train/validation sets.

    Splits by episodes (not frames) to avoid data leakage.

    Args:
        repo_id: HuggingFace dataset repository ID
        root: Root directory for datasets (defaults to ~/.cache/huggingface/lerobot)
        train_ratio: Fraction of episodes for training (0.8 = 80/20 split)
        random_seed: Random seed for reproducible splits
        delta_timestamps: Optional temporal configuration for observations/actions
        verbose: Print detailed information

    Returns:
        Tuple of (full_dataset, train_dataset, val_dataset)
    """
    if root is None:
        # Check if dataset exists in local cache (possibly symlinked)
        cache_path = Path.home() / ".cache/huggingface/lerobot" / repo_id
        if cache_path.exists():
            root = cache_path.resolve()

    # Load dataset metadata for episode count
    kwargs = {"repo_id": repo_id}
    if root:
        kwargs["root"] = root
    dataset_meta = LeRobotDatasetMetadata(**kwargs)

    print(dataset_meta)
    
    total_episodes = dataset_meta.total_episodes

    if verbose:
        print(f"Dataset: {repo_id}")
        print(f"Total episodes: {total_episodes}")
        print(f"Total frames: {dataset_meta.total_frames}")
        print(f"FPS: {dataset_meta.fps}")

    # Split episodes into train/val (80/20 by default)
    random.seed(random_seed)
    all_episodes = list(range(total_episodes))
    random.shuffle(all_episodes)
    split_idx = int(total_episodes * train_ratio)
    train_episode_ids = all_episodes[:split_idx]
    val_episode_ids = all_episodes[split_idx:]

    if verbose:
        print(f"Train episodes: {len(train_episode_ids)} ({train_ratio*100:.0f}%)")
        print(f"Val episodes: {len(val_episode_ids)} ({(1-train_ratio)*100:.0f}%)")

    # Load full dataset (for metadata)
    full_dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )

    # Load train dataset
    train_dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=train_episode_ids,
        delta_timestamps=delta_timestamps,
    )

    # Load validation dataset (only if we have validation episodes)
    if val_episode_ids:
        val_dataset = LeRobotDataset(
            repo_id,
            root=root,
            episodes=val_episode_ids,
            delta_timestamps=delta_timestamps,
        )
    else:
        val_dataset = None

    if verbose:
        print(f"Train dataset: {len(train_dataset)} frames")
        print(f"Val dataset: {len(val_dataset) if val_dataset else 0} frames")
        print()

    return full_dataset, train_dataset, val_dataset


def create_dataloaders(
    train_dataset: LeRobotDataset,
    val_dataset: LeRobotDataset,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes (0 for single-threaded)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    return train_loader, val_loader


# ── Multi-dataset aggregation ────────────────────────────────────────────────

LEROBOT_CACHE = Path.home() / ".cache/huggingface/lerobot"


def _resolve_root(repo_id: str, root: Optional[str] = None) -> Optional[Path]:
    """Resolve dataset root — check explicit path, then HF cache."""
    if root:
        p = Path(root).expanduser()
        if p.exists():
            return p.resolve()
    cache = LEROBOT_CACHE / repo_id
    if cache.exists():
        return cache.resolve()
    return None


def resolve_dataset_source(sources: list, experiment: str = None, version: str = None) -> tuple:
    """Resolve dataset config sources to a single (repo_id, root) for training.

    If single source: returns it directly.
    If multiple sources: aggregates into one dataset, returns the merged result.

    Args:
        sources: list of DatasetSource objects from config
        experiment: experiment name (for naming the aggregated dataset)
        version: version id (for naming)

    Returns:
        (repo_id, root) pointing to the dataset ready for training
    """
    if len(sources) == 1:
        src = sources[0]
        root = _resolve_root(src.repo_id, src.root)
        return src.repo_id, str(root) if root else None

    # Multiple sources — aggregate
    if experiment and version:
        output_repo_id = f"eternalmay33/{experiment}_{version}_mix"
    else:
        names = [s.repo_id.split("/")[-1][:15] for s in sources]
        output_repo_id = f"eternalmay33/mix_{'_'.join(names)}"

    # Check if already aggregated
    output_root = LEROBOT_CACHE / output_repo_id
    if output_root.exists() and (output_root / "meta/info.json").exists():
        meta = LeRobotDatasetMetadata(output_repo_id, root=str(output_root))
        print(f"Using existing aggregated dataset: {output_repo_id} "
              f"({meta.total_episodes} eps, {meta.total_frames} frames)")
        return output_repo_id, str(output_root)

    # Aggregate
    aggregate_sources(sources, output_repo_id)
    return output_repo_id, str(output_root)


def aggregate_sources(
    sources: list,
    output_repo_id: str,
    output_root: Optional[str] = None,
) -> str:
    """Aggregate multiple datasets into one using LeRobot's aggregate_datasets.

    Produces a single merged dataset with combined stats, reindexed episodes,
    and concatenated videos. The result is linked into the HF cache.

    Args:
        sources: list of DatasetSource objects or dicts with repo_id + root
        output_repo_id: repo_id for the aggregated dataset
        output_root: output directory (default: HF lerobot cache)

    Returns:
        Path to the aggregated dataset root.
    """
    from lerobot.datasets.aggregate import aggregate_datasets

    repo_ids = []
    roots = []

    for src in sources:
        repo_id = src.repo_id if hasattr(src, "repo_id") else src["repo_id"]
        root = src.root if hasattr(src, "root") else src.get("root")
        resolved = _resolve_root(repo_id, root)
        repo_ids.append(repo_id)
        roots.append(resolved)

    if output_root:
        aggr_root = Path(output_root).expanduser().resolve()
    else:
        aggr_root = LEROBOT_CACHE / output_repo_id

    print(f"\nAggregating {len(sources)} datasets → {output_repo_id}")
    for rid, root in zip(repo_ids, roots):
        print(f"  {rid} ({root})")
    print(f"  Output: {aggr_root}")
    print()

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=aggr_root,
    )

    # Verify
    meta = LeRobotDatasetMetadata(output_repo_id, root=str(aggr_root))
    print(f"\nAggregated dataset:")
    print(f"  Episodes: {meta.total_episodes}")
    print(f"  Frames:   {meta.total_frames}")
    print(f"  FPS:      {meta.fps}")
    print(f"  Path:     {aggr_root}")

    return str(aggr_root)


def aggregate_from_config(config, experiment: str = None, version: str = None) -> str:
    """Aggregate datasets listed in a TrainConfig's sources.

    Names the output dataset based on experiment/version for traceability.

    Args:
        config: TrainConfig with multiple dataset.sources
        experiment: experiment name (for output naming)
        version: version id (for output naming)

    Returns:
        repo_id of the aggregated dataset
    """
    sources = config.dataset.sources

    if len(sources) < 2:
        raise ValueError("Need at least 2 sources to aggregate")

    # Build output name from experiment/version
    if experiment and version:
        output_repo_id = f"eternalmay33/{experiment}_{version}_mix"
    else:
        names = [s.repo_id.split("/")[-1][:15] for s in sources]
        output_repo_id = f"eternalmay33/mix_{'_'.join(names)}"

    aggregate_sources(sources, output_repo_id)
    return output_repo_id


# ── CLI ──────────────────────────────────────────────────────────────────────

def _aggregate_cli(
    datasets: str,
    output: str,
    output_root: str = None,
):
    """Aggregate multiple datasets into one.

    Args:
        datasets: comma-separated repo_ids (e.g. "eternalmay33/ds1,eternalmay33/ds2")
        output: output repo_id
        output_root: output directory (default: HF cache)

    Example:
        python -m vbti.logic.dataset.loading_utils aggregate \\
            --datasets="eternalmay33/08-merged_trimmed,eternalmay33/so101_real_pick_place_50eps_trimmed,eternalmay33/so101_sim_pick_place_130eps" \\
            --output="eternalmay33/duck_cup_v005_mix"
    """
    from types import SimpleNamespace

    sources = []
    for repo_id in datasets.split(","):
        repo_id = repo_id.strip()
        root = _resolve_root(repo_id)
        sources.append(SimpleNamespace(repo_id=repo_id, root=str(root) if root else None))

    aggregate_sources(sources, output, output_root)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "aggregate": _aggregate_cli,
    })
