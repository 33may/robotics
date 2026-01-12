"""
Shared training utilities for ACT, SmolVLA, and other policies.

This module provides reusable functions for:
- Dataset loading and splitting
- DataLoader creation
- WandB setup
- Validation loops
- Checkpoint management
"""
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.utils.data
import wandb
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# DATASET LOADING AND SPLITTING
# =============================================================================

def load_and_split_dataset(
    repo_id: str,
    root: Optional[Path] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    delta_timestamps: Optional[Dict] = None,
    video_backend: str = "pyav",
    verbose: bool = True,
) -> Tuple[LeRobotDataset, LeRobotDataset, LeRobotDataset]:
    """
    Load dataset and split into train/validation sets.

    Args:
        repo_id: HuggingFace dataset repository ID
        root: Root directory for datasets (defaults to ~/.cache/huggingface/lerobot)
        train_ratio: Ratio of training data (e.g., 0.8 for 80/20 split)
        random_seed: Random seed for reproducible splits
        delta_timestamps: Delta timestamps configuration for temporal data
        video_backend: Video decoding backend ("pyav" or "torchvision")
        verbose: Print loading information

    Returns:
        Tuple of (full_dataset, train_dataset, val_dataset)
    """
    if verbose:
        print(f"Loading dataset: {repo_id}")

    # Load full dataset first to get metadata
    full_dataset = LeRobotDataset(repo_id, root=root)
    total_episodes = full_dataset.num_episodes

    if verbose:
        print(f"Total episodes: {total_episodes}")

    # Split episodes
    all_episodes = list(range(total_episodes))
    random.seed(random_seed)
    random.shuffle(all_episodes)

    split_idx = int(total_episodes * train_ratio)
    train_ids = all_episodes[:split_idx]
    val_ids = all_episodes[split_idx:]

    if verbose:
        print(f"Train episodes: {len(train_ids)} ({train_ratio*100:.0f}%)")
        print(f"Val episodes: {len(val_ids)} ({(1-train_ratio)*100:.0f}%)")

    # Create train dataset
    train_dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=train_ids,
        video_backend=video_backend,
        delta_timestamps=delta_timestamps,
    )

    # Create validation dataset
    val_dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=val_ids,
        video_backend=video_backend,
        delta_timestamps=delta_timestamps,
    )

    if verbose:
        print(f"Train dataset: {len(train_dataset)} frames")
        print(f"Val dataset: {len(val_dataset)} frames")
        print()

    return full_dataset, train_dataset, val_dataset


def create_dataloaders(
    train_dataset: LeRobotDataset,
    val_dataset: LeRobotDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def move_batch_to_device(batch, device="cuda"):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

# =============================================================================
# DELTA TIMESTAMPS HELPERS
# =============================================================================

def create_act_delta_timestamps(chunk_size: int = 100, fps: float = 30.0) -> Dict:
    """
    Create delta_timestamps configuration for ACT policy.

    Args:
        chunk_size: Number of action steps to predict
        fps: Frames per second of the dataset

    Returns:
        Dictionary of delta timestamps
    """
    return {
        "observation.images.front": [0],
        "observation.images.third_person": [0],
        "observation.images.gripper": [0],
        "observation.state": [0],
        "action": [i / fps for i in range(chunk_size)],
    }


def create_smolvla_delta_timestamps(
    chunk_size: int = 50,
    fps: float = 30.0,
    image_features: Optional[list] = None,
) -> Dict:
    """
    Create delta_timestamps configuration for SmolVLA policy.

    Args:
        chunk_size: Number of action steps to predict
        fps: Frames per second of the dataset
        image_features: List of image feature keys (optional)

    Returns:
        Dictionary of delta timestamps
    """
    delta_timestamps = {
        "action": [i / fps for i in range(chunk_size)],
    }

    # Add image features (either from provided list or default)
    if image_features:
        for key in image_features:
            delta_timestamps[key] = [0]
    else:
        # Default image features
        delta_timestamps.update({
            "observation.images.front": [0],
            "observation.images.third_person": [0],
            "observation.images.gripper": [0],
            "observation.state": [0],
        })

    return delta_timestamps


# =============================================================================
# WANDB SETUP
# =============================================================================

def setup_wandb(
    project: str,
    name: str,
    config: Dict[str, Any],
    enabled: bool = True,
) -> Optional[Any]:
    """
    Initialize Weights & Biases logging.

    Args:
        project: WandB project name
        name: Run name
        config: Configuration dictionary to log
        enabled: Enable WandB logging

    Returns:
        WandB run object or None if disabled
    """
    if not enabled:
        print("WandB logging disabled")
        return None

    run = wandb.init(
        project=project,
        name=name,
        config=config,
    )
    print(f"WandB initialized: {project}/{name}")
    return run


# =============================================================================
# VALIDATION
# =============================================================================

def validate_policy(
    policy: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    preprocessor: Optional[Any] = None,
    squeeze_observation: bool = True,
) -> float:
    """
    Evaluate policy on validation set.

    Args:
        policy: Policy model
        val_loader: Validation data loader
        device: Device to run validation on
        preprocessor: Optional preprocessor for batch (used by SmolVLA)
        squeeze_observation: Squeeze observation temporal dimension (for ACT)

    Returns:
        Average validation loss
    """
    val_losses = []

    # policy.eval()

    policy.train()

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Apply preprocessor if provided (SmolVLA)
            if preprocessor is not None:
                batch = preprocessor(batch)

            # Squeeze observation dimension if needed (ACT)
            if squeeze_observation:
                for key in batch:
                    if "observation" in key:
                        # If it is an image (5D) or state (3D) with time=1, squeeze it
                        if batch[key].ndim == 5 and batch[key].shape[1] == 1:
                            batch[key] = batch[key].squeeze(1)
                        elif batch[key].ndim == 3 and batch[key].shape[1] == 1:
                            batch[key] = batch[key].squeeze(1)

            # Forward pass
            loss, _ = policy.forward(batch)
            val_losses.append(loss.item())

    policy.train()
    return np.mean(val_losses)


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    global_step: int,
    epoch: int,
    best_val_loss: float,
    is_best: bool = False,
) -> Path:
    """
    Save model checkpoint with training state.

    Args:
        policy: Policy model
        optimizer: Optimizer
        checkpoint_dir: Directory to save checkpoint
        global_step: Current training step
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        is_best: Whether this is the best checkpoint

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save policy
    policy.save_pretrained(str(checkpoint_dir / "pretrained_model"))

    # Save training state
    training_state = {
        "global_step": global_step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    with open(checkpoint_dir / "training_state.json", 'w') as f:
        json.dump(training_state, f, indent=2)

    # Save optimizer state
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    if is_best:
        print(f"  → New best checkpoint saved to {checkpoint_dir}")
    else:
        print(f"Checkpoint saved at step {global_step}")

    return checkpoint_dir


def load_checkpoint(
    checkpoint_path: Path,
    policy_class: type,
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[Dict], Optional[Dict]]:
    """
    Load policy checkpoint and training state.

    Args:
        checkpoint_path: Path to checkpoint directory or pretrained_model
        policy_class: Policy class (e.g., ACTPolicy, SmolVLAPolicy)
        device: Device to load policy to

    Returns:
        Tuple of (policy, training_state, optimizer_state)
    """
    print(f"Loading policy from checkpoint: {checkpoint_path}")

    # Load policy
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        policy = policy_class.from_pretrained(str(checkpoint_path.resolve()))
    else:
        # Assume it's a HuggingFace repo ID
        policy = policy_class.from_pretrained(str(checkpoint_path))

    policy.to(device)
    policy.train()
    print(f"Policy loaded: {type(policy).__name__}")

    # Load training state if exists
    checkpoint_dir = checkpoint_path.parent
    state_file = checkpoint_dir / "training_state.json"
    training_state = None
    if state_file.exists():
        with open(state_file, 'r') as f:
            training_state = json.load(f)
        print(f"Training state loaded: step {training_state.get('global_step', 0)}")
    else:
        print("Warning: No training_state.json found")

    # Load optimizer state if exists
    optimizer_file = checkpoint_dir / "optimizer.pt"
    optimizer_state = None
    if optimizer_file.exists():
        optimizer_state = torch.load(optimizer_file)
        print(f"Optimizer state loaded")
    else:
        print("Warning: No optimizer.pt found")

    return policy, training_state, optimizer_state


# =============================================================================
# TRAINING LOOP HELPERS
# =============================================================================

def log_training_metrics(
    wandb_run: Optional[Any],
    metrics: Dict[str, float],
    step: int,
    print_every: int = 200,
) -> None:
    """
    Log training metrics to WandB and console.

    Args:
        wandb_run: WandB run object (None if disabled)
        metrics: Dictionary of metrics to log
        step: Current training step
        print_every: Print to console every N steps
    """
    if wandb_run is not None:
        wandb.log(metrics, step=step)

    if step % print_every == 0:
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {step} | {metric_str}")


def should_validate(step: int, eval_freq: int) -> bool:
    """Check if validation should be run at this step."""
    return step % eval_freq == 0


def should_save(step: int, save_freq: int) -> bool:
    """Check if checkpoint should be saved at this step."""
    return step % save_freq == 0


# =============================================================================
# GRADIENT CLIPPING
# =============================================================================

def clip_gradients(
    policy: torch.nn.Module,
    max_norm: float = 10.0,
) -> float:
    """
    Clip gradients by norm.

    Args:
        policy: Policy model
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_norm)
