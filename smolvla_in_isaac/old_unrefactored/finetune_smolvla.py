"""
Fine-tune SmolVLA on pick_place dataset with validation.

This script:
1. Splits dataset 80/20 train/val
2. Evaluates on validation set every N steps
3. Logs train and val loss to WandB
4. Saves best checkpoint based on validation loss
5. Supports resume from checkpoint

Usage:
    # Start new training
    python finetune_smolvla.py

    # Resume from checkpoint
    python finetune_smolvla.py --resume outputs/finetune/smolvla_pick_place/checkpoints/best/pretrained_model
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
import numpy as np
import random
import json

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """Convert frame indices to timestamps."""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def evaluate_on_validation(policy, preprocessor, val_loader, max_batches=38):
    """Compute validation loss on limited number of batches."""
    val_losses = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            val_losses.append(loss.item())

    return np.mean(val_losses)


def train_with_validation(resume_from=None):
    # Configuration
    dataset_id = "eternalmay33/pick_place_test"
    dataset_root = Path.home() / ".cache/huggingface/lerobot"
    output_dir = Path("outputs/finetune/smolvla_pick_place")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training hyperparameters
    batch_size = 8
    num_workers = 4
    total_steps = 10000
    eval_freq = 50
    save_freq = 200
    log_freq = 10

    print("="*80)
    print("SMOLVLA FINE-TUNING WITH VALIDATION")
    print("="*80)
    print(f"Dataset: {dataset_id}")
    print(f"Total steps: {total_steps}")
    print(f"Validation frequency: every {eval_freq} steps")
    print(f"Save frequency: every {save_freq} steps")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("="*80)
    print()

    # Initialize WandB
    wandb.init(
        project="smolvla_pick_place",
        name="smolvla_with_validation",
        config={
            "dataset": dataset_id,
            "batch_size": batch_size,
            "total_steps": total_steps,
            "eval_freq": eval_freq,
        }
    )

    # Load dataset metadata
    print("Loading dataset metadata...")
    dataset_meta = LeRobotDatasetMetadata(repo_id=dataset_id)
    print(f"Total episodes: {dataset_meta.total_episodes}")
    print(f"Total frames: {dataset_meta.total_frames}")
    print(f"FPS: {dataset_meta.fps}")
    print()

    # Convert features
    features = dataset_to_policy_features(dataset_meta.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Configure SmolVLA
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        chunk_size=50,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        optimizer_lr=1e-5,
        optimizer_weight_decay=1e-10,
        optimizer_grad_clip_norm=10,
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=30000,
        device=device_str,
    )

    # Configure delta timestamps
    delta_timestamps = {
        "action": make_delta_timestamps(list(range(cfg.chunk_size)), dataset_meta.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps([0], dataset_meta.fps)
        for k in cfg.image_features
    }

    # Train/val split (80/20)
    total_episodes = dataset_meta.total_episodes
    val_ratio = 0.2
    random.seed(42)
    all_episodes = list(range(total_episodes))
    random.shuffle(all_episodes)
    split_idx = int(total_episodes * (1 - val_ratio))
    train_ids = all_episodes[:split_idx]
    val_ids = all_episodes[split_idx:]

    print(f"Train episodes: {len(train_ids)}")
    print(f"Val episodes: {len(val_ids)}")
    print()

    # Load datasets
    train_dataset = LeRobotDataset(
        dataset_id,
        root=dataset_root,
        episodes=train_ids,
        delta_timestamps=delta_timestamps,
    )
    val_dataset = LeRobotDataset(
        dataset_id,
        root=dataset_root,
        episodes=val_ids,
        delta_timestamps=delta_timestamps,
    )

    print(f"Train dataset: {len(train_dataset)} frames")
    print(f"Val dataset: {len(val_dataset)} frames")
    print()

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing - torchcodec has issues with workers
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for validation to avoid tokenizer deadlock
        pin_memory=True,
    )

    # Initialize or load policy
    if resume_from:
        print(f"Loading policy from checkpoint: {resume_from}")
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path.resolve()), config=cfg)
        else:
            policy = SmolVLAPolicy.from_pretrained(resume_from, config=cfg)

        policy.to(device)
        policy.train()

        # Load training state if exists
        checkpoint_dir = Path(resume_from).parent
        state_file = checkpoint_dir / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                training_state = json.load(f)
            resume_step = training_state.get("global_step", 0)
            best_val_loss = training_state.get("best_val_loss", float('inf'))
            print(f"Resuming from step {resume_step}")
            print(f"Best validation loss so far: {best_val_loss:.4f}")
        else:
            print("Warning: No training_state.json found, starting from step 0")
            resume_step = 0
            best_val_loss = float('inf')
    else:
        print("Initializing SmolVLA from base model...")
        model_id = "lerobot/smolvla_base"
        policy = SmolVLAPolicy.from_pretrained(model_id, config=cfg)
        policy.train()
        policy.to(device)
        resume_step = 0
        best_val_loss = float('inf')

    print(f"Policy loaded: {type(policy).__name__}")
    print()

    # Create preprocessor/postprocessor
    preprocessor, postprocessor = make_smolvla_pre_post_processors(cfg, dataset_stats=dataset_meta.stats)

    # Setup optimizer
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # Load optimizer state if resuming
    if resume_from:
        checkpoint_dir = Path(resume_from).parent
        optimizer_file = checkpoint_dir / "optimizer.pt"
        if optimizer_file.exists():
            optimizer.load_state_dict(torch.load(optimizer_file))
            print(f"Optimizer state loaded from {optimizer_file}")
        else:
            print("Warning: No optimizer.pt found, starting with fresh optimizer")

    # Training loop
    print("\nStarting training...")
    print("="*80)

    global_step = resume_step

    while global_step < total_steps:
        epoch_losses = []

        for batch in tqdm(train_loader, desc="Training", leave=False):
            if global_step >= total_steps:
                break

            # Preprocess batch
            batch = preprocessor(batch)

            # Forward pass
            loss, loss_dict = policy.forward(batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer_grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            global_step += 1

            # Log training loss
            if global_step % log_freq == 0:
                train_loss = np.mean(epoch_losses[-log_freq:])
                wandb.log({
                    "train/loss": train_loss,
                    "train/step": global_step,
                }, step=global_step)

                print(f"Step {global_step}/{total_steps} | Train loss: {train_loss:.4f}")

            # Validation
            if global_step % eval_freq == 0:
                print(f"\nRunning validation at step {global_step}...")
                val_loss = evaluate_on_validation(policy, preprocessor, val_loader, max_batches=38)

                wandb.log({
                    "val/loss": val_loss,
                    "val/step": global_step,
                }, step=global_step)

                print(f"Validation loss: {val_loss:.4f}")
                print(f"[DEBUG] Logged to WandB: val/loss={val_loss:.4f} at step {global_step}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_dir = output_dir / "checkpoints" / "best"
                    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

                    policy.save_pretrained(str(best_checkpoint_dir / "pretrained_model"))

                    # Save training state
                    training_state = {
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                    }
                    with open(best_checkpoint_dir / "training_state.json", 'w') as f:
                        json.dump(training_state, f, indent=2)

                    # Save optimizer state
                    torch.save(optimizer.state_dict(), best_checkpoint_dir / "optimizer.pt")

                    print(f"  → New best val loss! Saved to {best_checkpoint_dir}")
                    wandb.log({"val/best_loss": best_val_loss}, step=global_step)

            # Save periodic checkpoint
            if global_step % save_freq == 0:
                checkpoint_dir = output_dir / "checkpoints" / f"{global_step:06d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                policy.save_pretrained(str(checkpoint_dir / "pretrained_model"))

                # Save training state
                training_state = {
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                }
                with open(checkpoint_dir / "training_state.json", 'w') as f:
                    json.dump(training_state, f, indent=2)

                # Save optimizer state
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

                print(f"Saved checkpoint at step {global_step}")

    # Final save
    final_checkpoint_dir = output_dir / "checkpoints" / "final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(final_checkpoint_dir / "pretrained_model"))

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {output_dir}/checkpoints/best/pretrained_model")
    print(f"All checkpoints: {output_dir}/checkpoints/")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA with validation")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    train_with_validation(resume_from=args.resume)
