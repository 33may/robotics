"""
Train ACT policy with validation.

REFACTORED VERSION - Uses shared utilities from smolvla_in_isaac.training

This script trains an Action Chunking Transformer (ACT) policy on the pick_place_test dataset
with 80/20 train/validation split, WandB logging, and periodic checkpointing.
"""
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import shared utilities
from smolvla_in_isaac.training.utils import (
    load_and_split_dataset,
    create_dataloaders,
    move_batch_to_device,
    create_act_delta_timestamps,
    setup_wandb,
    validate_policy,
    save_checkpoint,
    load_checkpoint,
    log_training_metrics,
    should_validate,
    should_save,
    clip_gradients,
)

from common import (
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_DATASET_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)

# LeRobot imports
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ACT policy with validation")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/train/act_refactored"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--total_steps", type=int, default=25000)
    parser.add_argument("--eval_freq", type=int, default=400)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--log_freq", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_backbone", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # ACT config
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--n_action_steps", type=int, default=100)
    # WandB
    parser.add_argument("--wandb_project", type=str, default="lerobot_pick_place")
    parser.add_argument("--wandb_name", type=str, default="act_refactored")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def train(args):
    """Main training function."""

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("ACT TRAINING WITH VALIDATION (REFACTORED)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Total steps: {args.total_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("=" * 80)
    print()

    # Initialize WandB
    wandb_run = setup_wandb(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        enabled=not args.no_wandb,
    )

    # Load and split dataset
    delta_timestamps = create_act_delta_timestamps(
        chunk_size=args.chunk_size,
        fps=30.0,
    )

    full_dataset, train_dataset, val_dataset = load_and_split_dataset(
        repo_id=args.dataset,
        root=DEFAULT_DATASET_ROOT,
        train_ratio=0.8,
        random_seed=42,
        delta_timestamps=delta_timestamps,
        verbose=True,
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize or load policy
    if args.resume:
        policy, training_state, optimizer_state = load_checkpoint(
            checkpoint_path=Path(args.resume),
            policy_class=ACTPolicy,
            device=device,
        )
        resume_step = training_state.get("global_step", 0) if training_state else 0
        resume_epoch = training_state.get("epoch", 0) if training_state else 0
        best_val_loss = training_state.get("best_val_loss", float('inf')) if training_state else float('inf')
    else:
        print("Initializing ACT policy from scratch...")
        config = ACTConfig(
            n_obs_steps=args.n_obs_steps,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            device=str(device),
        )
        policy = make_policy(config, ds_meta=full_dataset.meta)
        policy.train()
        optimizer_state = None
        resume_step = 0
        resume_epoch = 0
        best_val_loss = float('inf')
        print(f"Policy initialized: {type(policy).__name__}")

    # Initialize optimizer
    backbone_params = list(policy.model.backbone.parameters())
    other_params = [p for n, p in policy.named_parameters() if not n.startswith('model.backbone')]

    params = [
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": other_params, "lr": args.learning_rate},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    # Load optimizer state if resuming
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("Optimizer state loaded")

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    global_step = resume_step
    epoch = resume_epoch

    while global_step < args.total_steps:
        epoch += 1
        epoch_losses = []

        for batch in train_loader:
            if global_step >= args.total_steps:
                break

            # Move batch to device
            batch = move_batch_to_device(batch, device)

            # Squeeze observation temporal dimension (ACT uses n_obs_steps=1)
            for key in batch:
                if "observation" in key:
                    if batch[key].ndim == 5 and batch[key].shape[1] == 1:
                        batch[key] = batch[key].squeeze(1)
                    elif batch[key].ndim == 3 and batch[key].shape[1] == 1:
                        batch[key] = batch[key].squeeze(1)

            # Forward pass
            loss, output_dict = policy.forward(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_gradients(policy, max_norm=10.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            # Log training metrics
            if global_step % args.log_freq == 0:
                train_loss = np.mean(epoch_losses[-args.log_freq:])
                log_training_metrics(
                    wandb_run,
                    {
                        "train/loss": train_loss,
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                    print_every=args.log_freq,
                )

            # Validation
            if should_validate(global_step, args.eval_freq):
                print(f"\nRunning validation at step {global_step}...")
                val_loss = validate_policy(
                    policy,
                    val_loader,
                    device,
                    preprocessor=None,  # ACT doesn't use preprocessor
                    squeeze_observation=True,
                )

                log_training_metrics(
                    wandb_run,
                    {"val/loss": val_loss},
                    step=global_step,
                    print_every=1,  # Always print validation
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        policy,
                        optimizer,
                        args.output_dir / "checkpoints" / "best",
                        global_step,
                        epoch,
                        best_val_loss,
                        is_best=True,
                    )

                    if wandb_run:
                        wandb_run.log({"val/best_loss": best_val_loss}, step=global_step)

            # Save periodic checkpoint
            if should_save(global_step, args.save_freq):
                save_checkpoint(
                    policy,
                    optimizer,
                    args.output_dir / "checkpoints" / f"{global_step:06d}",
                    global_step,
                    epoch,
                    best_val_loss,
                    is_best=False,
                )

    # Final save
    save_checkpoint(
        policy,
        optimizer,
        args.output_dir / "checkpoints" / "final",
        global_step,
        epoch,
        best_val_loss,
        is_best=False,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {args.output_dir}/checkpoints/best/pretrained_model")
    print(f"All checkpoints: {args.output_dir}/checkpoints/")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
