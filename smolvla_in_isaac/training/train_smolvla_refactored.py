"""
Fine-tune SmolVLA on pick_place dataset with validation.

REFACTORED VERSION - Uses shared utilities from smolvla_in_isaac.training

This script fine-tunes a SmolVLA policy with 80/20 train/validation split,
WandB logging, and periodic checkpointing.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import shared utilities
from training import (
    load_and_split_dataset,
    create_dataloaders,
    create_smolvla_delta_timestamps,
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
)

# LeRobot imports
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA with validation")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/finetune/smolvla_refactored"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)  # SmolVLA tokenizer issue
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=200)
    parser.add_argument("--log_freq", type=int, default=10)
    # SmolVLA config
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--freeze_vision_encoder", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    # WandB
    parser.add_argument("--wandb_project", type=str, default="smolvla_pick_place")
    parser.add_argument("--wandb_name", type=str, default="smolvla_refactored")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def train(args):
    """Main training function."""

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("SMOLVLA FINE-TUNING WITH VALIDATION (REFACTORED)")
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

    # Load dataset metadata for feature configuration
    print("Loading dataset metadata...")
    dataset_meta = LeRobotDatasetMetadata(repo_id=args.dataset)
    print(f"Total episodes: {dataset_meta.total_episodes}")
    print(f"Total frames: {dataset_meta.total_frames}")
    print(f"FPS: {dataset_meta.fps}")
    print()

    # Convert features for SmolVLA configuration
    features = dataset_to_policy_features(dataset_meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Configure SmolVLA
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=args.n_obs_steps,
        chunk_size=args.chunk_size,
        freeze_vision_encoder=args.freeze_vision_encoder,
        train_expert_only=True,
        train_state_proj=True,
        optimizer_lr=args.learning_rate,
        optimizer_weight_decay=args.weight_decay,
        optimizer_grad_clip_norm=args.grad_clip_norm,
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=30000,
        device=str(device),
    )

    # Create delta timestamps
    delta_timestamps = create_smolvla_delta_timestamps(
        chunk_size=args.chunk_size,
        fps=dataset_meta.fps,
        image_features=cfg.image_features,
    )

    # Load and split dataset
    full_dataset, train_dataset, val_dataset = load_and_split_dataset(
        repo_id=args.dataset,
        root=DEFAULT_DATASET_ROOT,
        train_ratio=0.8,
        random_seed=42,
        delta_timestamps=delta_timestamps,
        verbose=True,
    )

    # Create dataloaders (num_workers=0 due to tokenizer issues)
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
            policy_class=SmolVLAPolicy,
            device=device,
        )
        resume_step = training_state.get("global_step", 0) if training_state else 0
        resume_epoch = training_state.get("epoch", 0) if training_state else 0
        best_val_loss = training_state.get("best_val_loss", float('inf')) if training_state else float('inf')
    else:
        print("Initializing SmolVLA from base model...")
        model_id = "lerobot/smolvla_base"
        policy = SmolVLAPolicy.from_pretrained(model_id, config=cfg)
        policy.train()
        policy.to(device)
        optimizer_state = None
        resume_step = 0
        resume_epoch = 0
        best_val_loss = float('inf')
        print(f"Policy initialized: {type(policy).__name__}")

    # Create preprocessor/postprocessor
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        cfg,
        dataset_stats=dataset_meta.stats
    )
    print("Preprocessor and postprocessor created")

    # Initialize optimizer (SmolVLA uses config-based optimizer)
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

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

            # Preprocess batch (SmolVLA preprocessor handles normalization)
            batch = preprocessor(batch)

            # Forward pass
            loss, loss_dict = policy.forward(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_gradients(policy, max_norm=cfg.optimizer_grad_clip_norm)
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
                    preprocessor=preprocessor,  # SmolVLA uses preprocessor
                    squeeze_observation=False,  # SmolVLA doesn't squeeze
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
