"""
Test SmolVLA predictions on validation dataset.

This script:
1. Loads validation dataset
2. Loads trained SmolVLA model
3. Compares predictions vs ground truth
4. Computes error statistics
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


# Configuration
dataset_id = "eternalmay33/pick_place_test"
dataset_root = Path.home() / ".cache/huggingface/lerobot"
checkpoint_path = "outputs/finetune/smolvla_pick_place/checkpoints/best/pretrained_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("TESTING SMOLVLA ON VALIDATION DATASET")
print("="*80)
print(f"Checkpoint: {checkpoint_path}")
print(f"Device: {device}")
print()

# Load policy
print("Loading SmolVLA policy...")
policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
policy.to(device)
policy.eval()
print(f"Policy loaded: {type(policy).__name__}")
print()

# Load dataset metadata
dataset_meta = LeRobotDatasetMetadata(repo_id=dataset_id)
print(f"Dataset: {dataset_id}")
print(f"Total episodes: {dataset_meta.total_episodes}")
print(f"FPS: {dataset_meta.fps}")
print()

# Create preprocessor/postprocessor
preprocessor, postprocessor = make_smolvla_pre_post_processors(
    policy.config,
    dataset_stats=dataset_meta.stats
)

# Configure delta timestamps
fps = dataset_meta.fps
delta_timestamps = {
    "observation.images.front": [0],
    "observation.images.third_person": [0],
    "observation.images.gripper": [0],
    "observation.state": [0],
    "action": [i / fps for i in range(policy.config.chunk_size)],
}

# Train/val split (80/20) - SAME as training
total_episodes = dataset_meta.total_episodes
val_ratio = 0.2
random.seed(42)
all_episodes = list(range(total_episodes))
random.shuffle(all_episodes)
split_idx = int(total_episodes * (1 - val_ratio))
train_ids = all_episodes[:split_idx]
val_ids = all_episodes[split_idx:]

print(f"Train episodes: {len(train_ids)} - {train_ids}")
print(f"Val episodes: {len(val_ids)} - {val_ids}")
print()

# Load validation dataset
val_dataset = LeRobotDataset(
    dataset_id,
    root=dataset_root,
    episodes=val_ids,
    delta_timestamps=delta_timestamps,
)
print(f"Validation dataset: {len(val_dataset)} frames")
print()

# Also test on train dataset to check for overfitting
train_dataset = LeRobotDataset(
    dataset_id,
    root=dataset_root,
    episodes=train_ids,
    delta_timestamps=delta_timestamps,
)
print(f"Train dataset: {len(train_dataset)} frames")
print()

# Test samples
print("="*80)
print("TESTING ON VALIDATION SET")
print("="*80)

val_errors = []
val_predictions = []
val_ground_truth = []

# Sample evenly from validation set
num_samples = min(50, len(val_dataset))
sample_indices = np.linspace(0, len(val_dataset)-1, num_samples, dtype=int)

with torch.no_grad():
    for idx in sample_indices:
        batch = val_dataset[idx]

        # Prepare batch for model
        batch_dict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.unsqueeze(0).to(device)

                # Squeeze time dimension for observations if present
                if "observation" in key:
                    if batch_dict[key].ndim == 5 and batch_dict[key].shape[1] == 1:
                        batch_dict[key] = batch_dict[key].squeeze(1)
                    elif batch_dict[key].ndim == 3 and batch_dict[key].shape[1] == 1:
                        batch_dict[key] = batch_dict[key].squeeze(1)
            else:
                batch_dict[key] = value

        # Apply preprocessor
        batch_preprocessed = preprocessor(batch_dict)

        # Get prediction (normalized)
        predicted_action_normalized = policy.select_action(batch_preprocessed)

        # Check raw normalized predictions
        predicted_action_raw = predicted_action_normalized.cpu().numpy()[0]

        # Apply postprocessor (denormalize)
        predicted_action = postprocessor(predicted_action_normalized)
        predicted_action = predicted_action.cpu().numpy()[0]  # First action in chunk

        # Get ground truth (first action in chunk)
        gt_action = batch["action"].cpu().numpy()[0]

        # Print raw vs denormalized for first sample
        if len(val_errors) == 0:
            print(f"\n[DEBUG] First sample:")
            print(f"  Raw normalized prediction: {predicted_action_raw}")
            print(f"  Denormalized prediction:   {predicted_action}")
            print(f"  Ground truth:              {gt_action}")

        # Compute error
        error = np.abs(predicted_action - gt_action)
        val_errors.append(error)
        val_predictions.append(predicted_action)
        val_ground_truth.append(gt_action)

        # Print first 5 samples
        if len(val_errors) <= 5:
            obs_state = batch["observation.state"].cpu().numpy()
            print(f"\nSample {idx}:")
            print(f"  Observation state: {obs_state}")
            print(f"  Ground truth:      {gt_action}")
            print(f"  Predicted:         {predicted_action}")
            print(f"  Error:             {error}")
            print(f"  Max error:         {error.max():.3f}")

val_errors = np.array(val_errors)
print(f"\n{'='*80}")
print("VALIDATION SET STATISTICS")
print(f"{'='*80}")
print(f"Mean absolute error per joint: {val_errors.mean(axis=0)}")
print(f"Max error per joint: {val_errors.max(axis=0)}")
print(f"Overall mean error: {val_errors.mean():.3f}")
print(f"Overall max error: {val_errors.max():.3f}")
print()

# Test on TRAIN set to check overfitting
print("="*80)
print("TESTING ON TRAIN SET (Check for overfitting)")
print("="*80)

train_errors = []
num_train_samples = min(50, len(train_dataset))
train_sample_indices = np.linspace(0, len(train_dataset)-1, num_train_samples, dtype=int)

with torch.no_grad():
    for idx in train_sample_indices:
        batch = train_dataset[idx]

        # Prepare batch
        batch_dict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.unsqueeze(0).to(device)
                if "observation" in key:
                    if batch_dict[key].ndim == 5 and batch_dict[key].shape[1] == 1:
                        batch_dict[key] = batch_dict[key].squeeze(1)
                    elif batch_dict[key].ndim == 3 and batch_dict[key].shape[1] == 1:
                        batch_dict[key] = batch_dict[key].squeeze(1)
            else:
                batch_dict[key] = value

        batch_preprocessed = preprocessor(batch_dict)
        predicted_action = policy.select_action(batch_preprocessed)
        predicted_action = postprocessor(predicted_action)
        predicted_action = predicted_action.cpu().numpy()[0]

        gt_action = batch["action"].cpu().numpy()[0]
        error = np.abs(predicted_action - gt_action)
        train_errors.append(error)

train_errors = np.array(train_errors)
print(f"Train mean error: {train_errors.mean():.3f}")
print(f"Train max error: {train_errors.max():.3f}")
print()

# Check for overfitting
print("="*80)
print("OVERFITTING CHECK")
print("="*80)
print(f"Val error:   {val_errors.mean():.3f}")
print(f"Train error: {train_errors.mean():.3f}")
if train_errors.mean() < val_errors.mean() * 0.5:
    print("⚠️  WARNING: Possible overfitting (train error << val error)")
elif val_errors.mean() > 10.0:
    print("⚠️  WARNING: High validation error - model not learning well")
else:
    print("✓ Error levels seem reasonable")
print()

# Visualize predictions vs ground truth
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
joint_names = ['Joint 0', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5']

val_predictions = np.array(val_predictions)
val_ground_truth = np.array(val_ground_truth)

for i in range(6):
    row = i // 3
    col = i % 3

    axes[row, col].scatter(val_ground_truth[:, i], val_predictions[:, i], alpha=0.5)
    axes[row, col].plot([val_ground_truth[:, i].min(), val_ground_truth[:, i].max()],
                        [val_ground_truth[:, i].min(), val_ground_truth[:, i].max()],
                        'r--', label='Perfect prediction')
    axes[row, col].set_xlabel('Ground Truth')
    axes[row, col].set_ylabel('Predicted')
    axes[row, col].set_title(joint_names[i])
    axes[row, col].legend()
    axes[row, col].grid(True)

plt.tight_layout()
plt.savefig('smolvla_predictions_vs_gt.png', dpi=120, bbox_inches='tight')
print(f"Saved prediction plot to: smolvla_predictions_vs_gt.png")
print()

print("="*80)
print("DONE")
print("="*80)
