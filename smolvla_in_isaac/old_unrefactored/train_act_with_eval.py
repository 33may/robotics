import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import wandb
import numpy as np
import random
import json

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy


def evaluate_on_validation(policy, val_loader, device):
    """Compute validation loss."""
    
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            for key in batch:
                if "observation" in key:
                    if batch[key].ndim == 5 and batch[key].shape[1] == 1:
                        batch[key] = batch[key].squeeze(1)
                    elif batch[key].ndim == 3 and batch[key].shape[1] == 1:
                        batch[key] = batch[key].squeeze(1)

            loss, _ = policy.forward(batch)
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def train_with_validation(resume_from=None):
    # Configuration
    dataset_repo_id = "eternalmay33/pick_place_test"
    dataset_root = Path.home() / ".cache/huggingface/lerobot"
    output_dir = Path("outputs/train/act_pick_place_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training hyperparameters
    batch_size = 8
    num_workers = 4
    total_steps = 25000
    eval_freq = 400
    save_freq = 5000
    log_freq = 200
    learning_rate = 1e-6
    lr_backbone = 1e-6
    weight_decay = 1e-4

    print("="*80)
    print("ACT TRAINING WITH VALIDATION")
    print("="*80)
    print(f"Dataset: {dataset_repo_id}")
    print(f"Train episodes: 0-30 (31 episodes)")
    print(f"Val episodes: 31-38 (8 episodes)")
    print(f"Total steps: {total_steps}")
    print(f"Validation frequency: every {eval_freq} steps")
    print(f"Save frequency: every {save_freq} steps")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("="*80)
    print()

    # Initialize WandB
    wandb.init(
        project="lerobot_pick_place",
        name="act_with_validation",
        config={
            "batch_size": batch_size,
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "lr_backbone": lr_backbone,
            "weight_decay": weight_decay,
            "eval_freq": eval_freq,
            "train_episodes": "0-30",
            "val_episodes": "31-38",
        }
    )

    # Load datasets
    print("Loading datasets...")

    # Configure delta_timestamps for ACT
    # n_obs_steps=1: current observation only
    # chunk_size=100: predict next 100 actions
    fps = 30  # Dataset was recorded at ~30 fps
    delta_timestamps = {
        "observation.images.front": [0],
        "observation.images.third_person": [0],
        "observation.images.gripper": [0],
        "observation.state": [0],
        "action": [i / fps for i in range(100)],
    }

    # Full dataset for getting metadata
    full_dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)

    total_episodes = 39 
    val_ratio = 0.2

    
    all_episodes = list(range(total_episodes))
    
    
    random.seed(42) 
    random.shuffle(all_episodes)

    split_idx = int(total_episodes * (1 - val_ratio))

    train_ids = all_episodes[:split_idx]
    val_ids = all_episodes[split_idx:]

    train_dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=train_ids,
        video_backend="pyav",
        delta_timestamps=delta_timestamps,
    )

    val_dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        episodes=val_ids,
        video_backend="pyav",
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
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize or load policy
    if resume_from:
        print(f"Loading policy from checkpoint: {resume_from}")

        # Check if it's a local path or HuggingFace repo
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            # Local checkpoint - use local loading with pretrained_model_name_or_path
            policy = ACTPolicy.from_pretrained(str(checkpoint_path.resolve()))
        else:
            # Assume it's a HuggingFace repo ID
            policy = ACTPolicy.from_pretrained(resume_from)

        policy.to(device)
        policy.train()
        print(f"Policy loaded from checkpoint: {type(policy).__name__}")

        # Load training state if exists
        checkpoint_dir = Path(resume_from).parent
        state_file = checkpoint_dir / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                training_state = json.load(f)
            resume_step = training_state.get("global_step", 0)
            resume_epoch = training_state.get("epoch", 0)
            best_val_loss = training_state.get("best_val_loss", float('inf'))
            print(f"Resuming from step {resume_step}, epoch {resume_epoch}")
            print(f"Best validation loss so far: {best_val_loss:.4f}")
        else:
            print("Warning: No training_state.json found, starting from step 0")
            resume_step = 0
            resume_epoch = 0
            best_val_loss = float('inf')
    else:
        print("Initializing ACT policy from scratch...")
        config = ACTConfig(
            n_obs_steps=1,
            chunk_size=100,
            n_action_steps=100,
            device=str(device),
        )
        policy = make_policy(config, ds_meta=full_dataset.meta)
        policy.train()
        resume_step = 0
        resume_epoch = 0
        best_val_loss = float('inf')
        print(f"Policy initialized: {type(policy).__name__}")

    print(f"Policy device: {device}")
    print()

    # Initialize optimizer with different learning rates for backbone vs rest
    backbone_params = list(policy.model.backbone.parameters())
    other_params = [p for n, p in policy.named_parameters() if not n.startswith('model.backbone')]

    params = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": other_params, "lr": learning_rate},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)

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
    epoch = resume_epoch

    while global_step < total_steps:
        epoch += 1
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            if global_step >= total_steps:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            for key in batch:
                if "observation" in key:
                    # If it is an image (5D) or state (3D) with time=1, squeeze it
                    if batch[key].ndim == 5 and batch[key].shape[1] == 1:
                        batch[key] = batch[key].squeeze(1)
                    elif batch[key].ndim == 3 and batch[key].shape[1] == 1:
                        batch[key] = batch[key].squeeze(1)

            # Forward pass
            loss, output_dict = policy.forward(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            # Log training loss
            if global_step % log_freq == 0:
                train_loss = np.mean(epoch_losses[-log_freq:])
                wandb.log({
                    "train/loss": train_loss,
                    "train/step": global_step,
                    "train/epoch": epoch,
                }, step=global_step)

                print(f"Step {global_step}/{total_steps} | Train loss: {train_loss:.4f}")

            # Validation
            if global_step % eval_freq == 0:
                print(f"\nRunning validation at step {global_step}...")
                val_loss = evaluate_on_validation(policy, val_loader, device)

                wandb.log({
                    "val/loss": val_loss,
                    "val/step": global_step,
                }, step=global_step)

                print(f"Validation loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_dir = output_dir / "checkpoints" / "best"
                    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

                    policy.save_pretrained(str(best_checkpoint_dir / "pretrained_model"))

                    # Save training state
                    training_state = {
                        "global_step": global_step,
                        "epoch": epoch,
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
                    "epoch": epoch,
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
    parser = argparse.ArgumentParser(description="Train ACT policy with validation")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., outputs/train/act_pick_place_eval/checkpoints/005000/pretrained_model)",
    )
    args = parser.parse_args()

    train_with_validation(resume_from=args.resume)
