"""
Comprehensive analysis of SmolVLA predictions on train and validation sets.

This script generates:
1. Time-series plots (GT vs Predicted) for all joints
2. Scatter plots (Predicted vs GT) for all joints
3. Error histograms
4. Summary statistics

All outputs saved to research/plots/
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random
import json

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


# Configuration
DATASET_ID = "eternalmay33/pick_place_test"
DATASET_ROOT = Path.home() / ".cache/huggingface/lerobot"
CHECKPOINT_PATH = "outputs/finetune/smolvla_pick_place/checkpoints/best/pretrained_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("smolvla_in_isaac/research")

print("="*80)
print("SMOLVLA PREDICTION ANALYSIS")
print("="*80)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

# Load policy
print("Loading SmolVLA policy...")
policy = SmolVLAPolicy.from_pretrained(CHECKPOINT_PATH)
policy.to(DEVICE)
policy.eval()
print(f"✓ Policy loaded")

# Load dataset metadata
dataset_meta = LeRobotDatasetMetadata(repo_id=DATASET_ID)
fps = dataset_meta.fps

# Create preprocessor/postprocessor
preprocessor, postprocessor = make_smolvla_pre_post_processors(
    policy.config,
    dataset_stats=dataset_meta.stats
)

# Configure delta timestamps
delta_timestamps = {
    "observation.images.front": [0],
    "observation.images.third_person": [0],
    "observation.images.gripper": [0],
    "observation.state": [0],
    "action": [i / fps for i in range(policy.config.chunk_size)],
}

# Train/val split (same as training)
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

# Load datasets
val_dataset = LeRobotDataset(
    DATASET_ID,
    root=DATASET_ROOT,
    episodes=val_ids,
    delta_timestamps=delta_timestamps,
)

train_dataset = LeRobotDataset(
    DATASET_ID,
    root=DATASET_ROOT,
    episodes=train_ids,
    delta_timestamps=delta_timestamps,
)

print(f"✓ Train dataset: {len(train_dataset)} frames")
print(f"✓ Val dataset: {len(val_dataset)} frames")
print()


def get_predictions(dataset, num_samples=None, name="", sequential=False):
    """Get predictions for a dataset.

    Args:
        dataset: LeRobotDataset
        num_samples: Number of samples to process
        name: Name for logging
        sequential: If True, take sequential frames from first episode
    """
    if num_samples is None:
        num_samples = len(dataset)

    num_samples = min(num_samples, len(dataset))

    if sequential:
        # Take sequential frames from the dataset
        sample_indices = list(range(num_samples))
        print(f"Processing {name} - SEQUENTIAL {num_samples} frames from start")
    else:
        # Take evenly distributed samples across entire dataset
        sample_indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
        print(f"Processing {name} - RANDOM SAMPLING {num_samples} frames")

    predictions = []
    ground_truth = []
    observations = []

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{num_samples} samples processed")

            batch = dataset[idx]

            # Prepare batch
            batch_dict = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.unsqueeze(0).to(DEVICE)
                    if "observation" in key:
                        if batch_dict[key].ndim == 5 and batch_dict[key].shape[1] == 1:
                            batch_dict[key] = batch_dict[key].squeeze(1)
                        elif batch_dict[key].ndim == 3 and batch_dict[key].shape[1] == 1:
                            batch_dict[key] = batch_dict[key].squeeze(1)
                else:
                    batch_dict[key] = value

            # Get prediction
            batch_preprocessed = preprocessor(batch_dict)
            predicted_action = policy.select_action(batch_preprocessed)
            predicted_action = postprocessor(predicted_action)
            predicted_action = predicted_action.cpu().numpy()[0]

            # Get ground truth
            gt_action = batch["action"].cpu().numpy()[0]
            obs_state = batch["observation.state"].cpu().numpy()

            predictions.append(predicted_action)
            ground_truth.append(gt_action)
            observations.append(obs_state)

    return np.array(predictions), np.array(ground_truth), np.array(observations)


# Get predictions for validation set (random sampling)
print("\n" + "="*80)
print("VALIDATION SET PREDICTIONS (Random Sampling)")
print("="*80)
val_pred, val_gt, val_obs = get_predictions(val_dataset, num_samples=600, name="Validation", sequential=False)

# Get predictions for train set (random sampling)
print("\n" + "="*80)
print("TRAIN SET PREDICTIONS (Random Sampling)")
print("="*80)
train_pred, train_gt, train_obs = get_predictions(train_dataset, num_samples=157, name="Train", sequential=False)

# Get predictions for ONE EPISODE SEQUENTIAL (validation)
print("\n" + "="*80)
print("ONE EPISODE SEQUENTIAL (Validation)")
print("="*80)
val_seq_pred, val_seq_gt, val_seq_obs = get_predictions(val_dataset, num_samples=min(300, len(val_dataset)), name="Val Sequential", sequential=True)

# Get predictions for ONE EPISODE SEQUENTIAL (train)
print("\n" + "="*80)
print("ONE EPISODE SEQUENTIAL (Train)")
print("="*80)
train_seq_pred, train_seq_gt, train_seq_obs = get_predictions(train_dataset, num_samples=min(300, len(train_dataset)), name="Train Sequential", sequential=True)

# Save data
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "data").mkdir(exist_ok=True)
np.savez(
    OUTPUT_DIR / "data" / "predictions.npz",
    val_pred=val_pred,
    val_gt=val_gt,
    val_obs=val_obs,
    train_pred=train_pred,
    train_gt=train_gt,
    train_obs=train_obs,
    val_seq_pred=val_seq_pred,
    val_seq_gt=val_seq_gt,
    val_seq_obs=val_seq_obs,
    train_seq_pred=train_seq_pred,
    train_seq_gt=train_seq_gt,
    train_seq_obs=train_seq_obs,
)
print(f"\n✓ Saved predictions to {OUTPUT_DIR / 'data' / 'predictions.npz'}")


# ==============================================================================
# PLOT 1: Time-series comparison (GT vs Predicted)
# ==============================================================================
print("\nGenerating time-series plots...")

joint_names = ['Joint 0', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5']

# Validation time-series
fig, axes = plt.subplots(6, 1, figsize=(16, 12))
fig.suptitle('Validation Set: Ground Truth vs Predictions', fontsize=16, fontweight='bold')

for i in range(6):
    axes[i].plot(val_gt[:, i], label='GT', color='blue', alpha=0.7, linewidth=2)
    axes[i].plot(val_pred[:, i], label='Pred', color='orange', alpha=0.7, linewidth=1.5)
    axes[i].set_ylabel(f'q[{i}]', fontsize=12, fontweight='bold')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

    # Compute error for this joint
    error = np.abs(val_pred[:, i] - val_gt[:, i])
    axes[i].set_title(f'{joint_names[i]} - Mean Error: {error.mean():.2f}°, Max: {error.max():.2f}°')

axes[-1].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'validation_timeseries.png', dpi=120, bbox_inches='tight')
print(f"✓ Saved: validation_timeseries.png")

# Train time-series
fig, axes = plt.subplots(6, 1, figsize=(16, 12))
fig.suptitle('Train Set: Ground Truth vs Predictions', fontsize=16, fontweight='bold')

for i in range(6):
    axes[i].plot(train_gt[:, i], label='GT', color='blue', alpha=0.7, linewidth=2)
    axes[i].plot(train_pred[:, i], label='Pred', color='orange', alpha=0.7, linewidth=1.5)
    axes[i].set_ylabel(f'q[{i}]', fontsize=12, fontweight='bold')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

    error = np.abs(train_pred[:, i] - train_gt[:, i])
    axes[i].set_title(f'{joint_names[i]} - Mean Error: {error.mean():.2f}°, Max: {error.max():.2f}°')

axes[-1].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'train_timeseries.png', dpi=120, bbox_inches='tight')
print(f"✓ Saved: train_timeseries.png")


# ==============================================================================
# PLOT 1b: SEQUENTIAL Time-series (One Episode)
# ==============================================================================
print("\nGenerating SEQUENTIAL time-series plots (one episode)...")

# Validation sequential
fig, axes = plt.subplots(6, 1, figsize=(16, 12))
fig.suptitle('Validation Set: SEQUENTIAL Episode - Ground Truth vs Predictions', fontsize=16, fontweight='bold')

for i in range(6):
    axes[i].plot(val_seq_gt[:, i], label='GT', color='blue', alpha=0.7, linewidth=2)
    axes[i].plot(val_seq_pred[:, i], label='Pred', color='orange', alpha=0.7, linewidth=1.5)
    axes[i].set_ylabel(f'q[{i}]', fontsize=12, fontweight='bold')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

    error = np.abs(val_seq_pred[:, i] - val_seq_gt[:, i])
    axes[i].set_title(f'{joint_names[i]} - Mean Error: {error.mean():.2f}°, Max: {error.max():.2f}°')

axes[-1].set_xlabel('Frame Index (Sequential)', fontsize=12, fontweight='bold')
plt.tight_layout()
(OUTPUT_DIR / 'plots').mkdir(exist_ok=True)
plt.savefig(OUTPUT_DIR / 'plots' / 'validation_sequential_timeseries.png', dpi=120, bbox_inches='tight')
print(f"✓ Saved: validation_sequential_timeseries.png")

# Train sequential
fig, axes = plt.subplots(6, 1, figsize=(16, 12))
fig.suptitle('Train Set: SEQUENTIAL Episode - Ground Truth vs Predictions', fontsize=16, fontweight='bold')

for i in range(6):
    axes[i].plot(train_seq_gt[:, i], label='GT', color='blue', alpha=0.7, linewidth=2)
    axes[i].plot(train_seq_pred[:, i], label='Pred', color='orange', alpha=0.7, linewidth=1.5)
    axes[i].set_ylabel(f'q[{i}]', fontsize=12, fontweight='bold')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

    error = np.abs(train_seq_pred[:, i] - train_seq_gt[:, i])
    axes[i].set_title(f'{joint_names[i]} - Mean Error: {error.mean():.2f}°, Max: {error.max():.2f}°')

axes[-1].set_xlabel('Frame Index (Sequential)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'train_sequential_timeseries.png', dpi=120, bbox_inches='tight')
print(f"✓ Saved: train_sequential_timeseries.png")


# ==============================================================================
# PLOT 2: Scatter plots (Predicted vs GT) with marginal histograms
# ==============================================================================
print("\nGenerating scatter plots with marginal histograms...")

from matplotlib.gridspec import GridSpec

def create_scatter_with_histograms(gt_data, pred_data, joint_name):
    """Create scatter plot with marginal histograms on top and right."""
    # Create grid: 3x3 with different sizes
    gs = GridSpec(3, 3, width_ratios=[1, 0.2, 0.05], height_ratios=[0.2, 1, 0.05],
                  hspace=0.05, wspace=0.05)

    # Main scatter plot
    ax_scatter = plt.subplot(gs[1, 0])

    # Top histogram (GT distribution)
    ax_histx = plt.subplot(gs[0, 0], sharex=ax_scatter)

    # Right histogram (Prediction distribution)
    ax_histy = plt.subplot(gs[1, 1], sharey=ax_scatter)

    # Scatter plot
    ax_scatter.scatter(gt_data, pred_data, alpha=0.5, s=10, color='steelblue')

    # Perfect prediction line
    min_val = min(gt_data.min(), pred_data.min())
    max_val = max(gt_data.max(), pred_data.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val],
                    'r--', label='Perfect', linewidth=2)

    ax_scatter.set_xlabel('Ground Truth (deg)', fontsize=10)
    ax_scatter.set_ylabel('Predicted (deg)', fontsize=10)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc='lower right', fontsize=8)

    # Compute R² correlation
    correlation = np.corrcoef(gt_data, pred_data)[0, 1]
    ax_scatter.text(0.05, 0.95, f'R²={correlation**2:.3f}',
                   transform=ax_scatter.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Top histogram (GT)
    ax_histx.hist(gt_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax_histx.set_ylabel('Count', fontsize=8)
    ax_histx.tick_params(labelbottom=False)
    ax_histx.set_title(joint_name, fontweight='bold', fontsize=11)

    # Right histogram (Predictions)
    ax_histy.hist(pred_data, bins=30, orientation='horizontal',
                  alpha=0.7, color='orange', edgecolor='black')
    ax_histy.set_xlabel('Count', fontsize=8)
    ax_histy.tick_params(labelleft=False)

    return ax_scatter, ax_histx, ax_histy


# Validation scatter with marginal histograms
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Validation Set: Predicted vs Ground Truth with Marginal Distributions',
             fontsize=16, fontweight='bold', y=0.995)

for i in range(6):
    row = i // 3
    col = i % 3

    # Create subplot grid for this joint
    plt.subplot(2, 3, i + 1)
    gs_inner = GridSpec(3, 3,
                       width_ratios=[1, 0.2, 0.05],
                       height_ratios=[0.2, 1, 0.05],
                       left=0.05 + col * 0.32,
                       right=0.35 + col * 0.32,
                       bottom=0.52 - row * 0.48,
                       top=0.95 - row * 0.48,
                       hspace=0.05,
                       wspace=0.05)

    # Main scatter plot
    ax_scatter = fig.add_subplot(gs_inner[1, 0])
    ax_histx = fig.add_subplot(gs_inner[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs_inner[1, 1], sharey=ax_scatter)

    # Scatter plot
    ax_scatter.scatter(val_gt[:, i], val_pred[:, i], alpha=0.5, s=10, color='steelblue')

    # Perfect prediction line
    min_val = min(val_gt[:, i].min(), val_pred[:, i].min())
    max_val = max(val_gt[:, i].max(), val_pred[:, i].max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val],
                    'r--', label='Perfect', linewidth=2)

    ax_scatter.set_xlabel('Ground Truth (deg)', fontsize=9)
    ax_scatter.set_ylabel('Predicted (deg)', fontsize=9)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc='lower right', fontsize=7)

    # R² correlation
    correlation = np.corrcoef(val_gt[:, i], val_pred[:, i])[0, 1]
    ax_scatter.text(0.05, 0.95, f'R²={correlation**2:.3f}',
                   transform=ax_scatter.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Top histogram (GT) - 1 degree bin width
    bin_width = 1.0
    gt_bins = np.arange(np.floor(val_gt[:, i].min()), np.ceil(val_gt[:, i].max()) + bin_width, bin_width)
    ax_histx.hist(val_gt[:, i], bins=gt_bins, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax_histx.set_ylabel('Count', fontsize=8)
    ax_histx.tick_params(labelbottom=False, labelsize=7)
    ax_histx.set_title(joint_names[i], fontweight='bold', fontsize=10, pad=3)

    # Right histogram (Predictions) - 1 degree bin width
    pred_bins = np.arange(np.floor(val_pred[:, i].min()), np.ceil(val_pred[:, i].max()) + bin_width, bin_width)
    ax_histy.hist(val_pred[:, i], bins=pred_bins, orientation='horizontal',
                  alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
    ax_histy.set_xlabel('Count', fontsize=8)
    ax_histy.tick_params(labelleft=False, labelsize=7)

plt.savefig(OUTPUT_DIR / 'plots' / 'validation_scatter.png', dpi=120, bbox_inches='tight')
print(f"✓ Saved: validation_scatter.png")
plt.close()


# Train scatter with marginal histograms
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Train Set: Predicted vs Ground Truth with Marginal Distributions',
             fontsize=16, fontweight='bold', y=0.995)

for i in range(6):
    row = i // 3
    col = i % 3

    # Create subplot grid for this joint
    gs_inner = GridSpec(3, 3,
                       width_ratios=[1, 0.2, 0.05],
                       height_ratios=[0.2, 1, 0.05],
                       left=0.05 + col * 0.32,
                       right=0.35 + col * 0.32,
                       bottom=0.52 - row * 0.48,
                       top=0.95 - row * 0.48,
                       hspace=0.05,
                       wspace=0.05)

    # Main scatter plot
    ax_scatter = fig.add_subplot(gs_inner[1, 0])
    ax_histx = fig.add_subplot(gs_inner[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs_inner[1, 1], sharey=ax_scatter)

    # Scatter plot
    ax_scatter.scatter(train_gt[:, i], train_pred[:, i], alpha=0.5, s=10, color='steelblue')

    # Perfect prediction line
    min_val = min(train_gt[:, i].min(), train_pred[:, i].min())
    max_val = max(train_gt[:, i].max(), train_pred[:, i].max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val],
                    'r--', label='Perfect', linewidth=2)

    ax_scatter.set_xlabel('Ground Truth (deg)', fontsize=9)
    ax_scatter.set_ylabel('Predicted (deg)', fontsize=9)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc='lower right', fontsize=7)

    # R² correlation
    correlation = np.corrcoef(train_gt[:, i], train_pred[:, i])[0, 1]
    ax_scatter.text(0.05, 0.95, f'R²={correlation**2:.3f}',
                   transform=ax_scatter.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Top histogram (GT) - 1 degree bin width
    bin_width = 1.0
    gt_bins = np.arange(np.floor(train_gt[:, i].min()), np.ceil(train_gt[:, i].max()) + bin_width, bin_width)
    ax_histx.hist(train_gt[:, i], bins=gt_bins, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax_histx.set_ylabel('Count', fontsize=8)
    ax_histx.tick_params(labelbottom=False, labelsize=7)
    ax_histx.set_title(joint_names[i], fontweight='bold', fontsize=10, pad=3)

    # Right histogram (Predictions) - 1 degree bin width
    pred_bins = np.arange(np.floor(train_pred[:, i].min()), np.ceil(train_pred[:, i].max()) + bin_width, bin_width)
    ax_histy.hist(train_pred[:, i], bins=pred_bins, orientation='horizontal',
                  alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
    ax_histy.set_xlabel('Count', fontsize=8)
    ax_histy.tick_params(labelleft=False, labelsize=7)

plt.savefig(OUTPUT_DIR / 'plots' / 'train_scatter.png', dpi=120, bbox_inches='tight')
print(f"✓ Saved: train_scatter.png")
plt.close()


# ==============================================================================
# PLOT 3: Error histograms
# ==============================================================================
print("\nGenerating error histograms...")

val_errors = np.abs(val_pred - val_gt)
train_errors = np.abs(train_pred - train_gt)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')

for i in range(6):
    row = i // 3
    col = i % 3

    axes[row, col].hist(val_errors[:, i], bins=50, alpha=0.5, label='Val', color='blue')
    axes[row, col].hist(train_errors[:, i], bins=50, alpha=0.5, label='Train', color='orange')

    axes[row, col].axvline(val_errors[:, i].mean(), color='blue', linestyle='--', linewidth=2, label=f'Val Mean={val_errors[:, i].mean():.2f}°')
    axes[row, col].axvline(train_errors[:, i].mean(), color='orange', linestyle='--', linewidth=2, label=f'Train Mean={train_errors[:, i].mean():.2f}°')

    axes[row, col].set_xlabel('Absolute Error (deg)', fontsize=10)
    axes[row, col].set_ylabel('Frequency', fontsize=10)
    axes[row, col].set_title(joint_names[i], fontweight='bold')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'error_histograms.png', dpi=120, bbox_inches='tight')
print(f"✓ Saved: error_histograms.png")


# ==============================================================================
# STATISTICS SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

stats = {
    "validation": {
        "mean_error_per_joint": val_errors.mean(axis=0).tolist(),
        "max_error_per_joint": val_errors.max(axis=0).tolist(),
        "overall_mean_error": float(val_errors.mean()),
        "overall_max_error": float(val_errors.max()),
    },
    "train": {
        "mean_error_per_joint": train_errors.mean(axis=0).tolist(),
        "max_error_per_joint": train_errors.max(axis=0).tolist(),
        "overall_mean_error": float(train_errors.mean()),
        "overall_max_error": float(train_errors.max()),
    }
}

# Save statistics
with open(OUTPUT_DIR / 'data' / 'statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\nVALIDATION SET:")
print(f"  Overall mean error: {stats['validation']['overall_mean_error']:.3f}°")
print(f"  Overall max error:  {stats['validation']['overall_max_error']:.3f}°")
print(f"  Per-joint mean errors:")
for i, err in enumerate(stats['validation']['mean_error_per_joint']):
    print(f"    Joint {i}: {err:.3f}°")

print("\nTRAIN SET:")
print(f"  Overall mean error: {stats['train']['overall_mean_error']:.3f}°")
print(f"  Overall max error:  {stats['train']['overall_max_error']:.3f}°")
print(f"  Per-joint mean errors:")
for i, err in enumerate(stats['train']['mean_error_per_joint']):
    print(f"    Joint {i}: {err:.3f}°")

print(f"\n✓ Saved statistics to {OUTPUT_DIR / 'data' / 'statistics.json'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"  - Plots: {OUTPUT_DIR / 'plots'}/")
print(f"  - Data: {OUTPUT_DIR / 'data'}/")
