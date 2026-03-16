"""
Dataset loading and splitting utilities.

Functions for loading LeRobot datasets from HuggingFace and creating
train/validation splits with proper shuffling and episode filtering.
"""
import random
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
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
        root = Path.home() / ".cache/huggingface/lerobot"

    # Load dataset metadata for episode count
    dataset_meta = LeRobotDatasetMetadata(repo_id=repo_id)

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
