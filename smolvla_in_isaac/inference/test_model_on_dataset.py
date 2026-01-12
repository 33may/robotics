"""
Test trained ACT model predictions on validation dataset.

This script:
1. Loads validation dataset
2. Loads trained model
3. Runs inference on validation samples
4. Shows predicted vs ground truth actions
"""
import torch
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

# Configuration
dataset_repo_id = "eternalmay33/pick_place_test"
dataset_root = Path.home() / ".cache/huggingface/lerobot"
checkpoint_path = "outputs/train/act_pick_place_eval/checkpoints/best/pretrained_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("TESTING MODEL PREDICTIONS ON VALIDATION DATASET")
print("="*80)
print(f"Checkpoint: {checkpoint_path}")
print(f"Device: {device}")
print()

# Load policy
print("Loading policy...")
policy = ACTPolicy.from_pretrained(checkpoint_path)
policy.to(device)
policy.eval()
print(f"Policy loaded: {type(policy).__name__}")
print()

# Configure delta_timestamps for ACT
fps = 30
delta_timestamps = {
    "observation.images.front": [0],
    "observation.images.third_person": [0],
    "observation.images.gripper": [0],
    "observation.state": [0],
    "action": [i / fps for i in range(100)],
}

# Load validation dataset
print("Loading validation dataset...")
total_episodes = 39
val_ratio = 0.2
import random
random.seed(42)
all_episodes = list(range(total_episodes))
random.shuffle(all_episodes)
split_idx = int(total_episodes * (1 - val_ratio))
val_ids = all_episodes[split_idx:]

val_dataset = LeRobotDataset(
    dataset_repo_id,
    root=dataset_root,
    episodes=val_ids,
    video_backend="pyav",
    delta_timestamps=delta_timestamps,
)
print(f"Validation dataset: {len(val_dataset)} frames from episodes {val_ids}")
print()

# Test on samples from different parts of dataset
test_indices = [0, 100, 200, 500, 1000, 1500, 1600, 1700, 1780, 1790]
test_indices = [i for i in test_indices if i < len(val_dataset)]

print(f"Testing predictions on {len(test_indices)} samples from different parts:")
print("-"*80)

with torch.no_grad():
    for idx, i in enumerate(test_indices):
        batch = val_dataset[i]

        # Prepare batch for model
        batch_dict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Add batch dimension and move to device
                batch_dict[key] = value.unsqueeze(0).to(device)

                # Squeeze time dimension for observations if present
                if "observation" in key:
                    if batch_dict[key].ndim == 5 and batch_dict[key].shape[1] == 1:
                        batch_dict[key] = batch_dict[key].squeeze(1)
                    elif batch_dict[key].ndim == 3 and batch_dict[key].shape[1] == 1:
                        batch_dict[key] = batch_dict[key].squeeze(1)
            else:
                batch_dict[key] = value

        # Get prediction
        predicted_action = policy.select_action(batch_dict)
        predicted_action = predicted_action.cpu().numpy()[0]  # First action in chunk

        # Get ground truth
        gt_action = batch["action"].cpu().numpy()[0]  # First action in chunk

        # Get observation state
        obs_state = batch["observation.state"].cpu().numpy()

        print(f"\nSample index {i} (test #{idx}):")
        print(f"  observation.state: {obs_state}")
        print(f"  GT action:         {gt_action}")
        print(f"  Predicted action:  {predicted_action}")
        print(f"  Difference:        {np.abs(gt_action - predicted_action)}")
        print(f"  Max error:         {np.abs(gt_action - predicted_action).max():.3f}")

print("\n" + "="*80)
print("Done!")
