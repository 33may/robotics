"""
Data transformation utilities for Isaac Lab <-> LeRobot conversions.

This module provides functions to convert between Isaac Lab and LeRobot data formats,
including normalization/denormalization and observation preparation.
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict

from .constants import (
    ISAACLAB_JOINT_POS_LIMIT_RANGE,
    LEROBOT_JOINT_POS_LIMIT_RANGE,
    ISAAC_CAMERA_KEYS,
    get_stats_path,
)


# =============================================================================
# NORMALIZATION STATISTICS LOADING
# =============================================================================

def load_normalization_stats(dataset_repo_id: str, root: Path | None = None) -> Dict:
    """
    Load normalization statistics from dataset metadata.

    Args:
        dataset_repo_id: HuggingFace dataset repository ID
        root: Root directory for datasets (defaults to ~/.cache/huggingface/lerobot)

    Returns:
        Dictionary containing mean/std statistics for observations and actions
    """
    stats_path = get_stats_path(dataset_repo_id, root)

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization statistics not found at {stats_path}. "
            f"Make sure the dataset '{dataset_repo_id}' is downloaded."
        )

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return {
        'obs_state_mean': np.array(stats['observation.state']['mean']),
        'obs_state_std': np.array(stats['observation.state']['std']),
        'action_mean': np.array(stats['action']['mean']),
        'action_std': np.array(stats['action']['std']),
    }


# =============================================================================
# JOINT POSITION TRANSFORMATIONS
# =============================================================================

def preprocess_isaac_to_lerobot(
    joint_pos: np.ndarray,
    obs_state_mean: np.ndarray,
    obs_state_std: np.ndarray,
    limit_range=False
) -> np.ndarray:
    """
    Convert joint positions from Isaac Lab to LeRobot format with MEAN_STD normalization.

    Pipeline:
    1. Convert radians to degrees
    2. Map from Isaac range to LeRobot range
    3. Apply MEAN_STD normalization: (x - mean) / std

    Args:
        joint_pos: Joint positions in radians [6]
        obs_state_mean: Mean values for normalization [6]
        obs_state_std: Std values for normalization [6]

    Returns:
        Normalized joint positions ready for policy input [6]
    """
    # Step 1: Convert radians to degrees
    joint_pos = joint_pos / np.pi * 180

    # Step 2: Map from Isaac range to LeRobot range
    if limit_range:
        for i in range(6):
            isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
            lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
            isaac_range = isaaclab_max - isaaclab_min
            lerobot_range = lerobot_max - lerobot_min
            joint_pos[i] = (joint_pos[i] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min

    # Step 3: Apply MEAN_STD normalization
    joint_pos = (joint_pos - obs_state_mean) / obs_state_std

    return joint_pos


def postprocess_lerobot_to_isaac(
    joint_pos: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    limit_range=False
) -> np.ndarray:
    """
    Convert joint positions from LeRobot to Isaac Lab format.

    Pipeline:
    0. MEAN_STD denormalize: action * std + mean
    1. Map from LeRobot range to Isaac range
    2. Convert degrees to radians

    Args:
        joint_pos: Normalized joint positions from policy [6]
        action_mean: Mean values for denormalization [6]
        action_std: Std values for denormalization [6]

    Returns:
        Joint positions in radians ready for Isaac Lab [6]
    """
    # Step 0: MEAN_STD denormalize
    joint_pos = joint_pos * action_std + action_mean

    # Step 1: Map from LeRobot range to Isaac range
    if limit_range:
        for i in range(6):
            isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
            lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
            isaac_range = isaaclab_max - isaaclab_min
            lerobot_range = lerobot_max - lerobot_min
            joint_pos[i] = (joint_pos[i] - lerobot_min) / lerobot_range * isaac_range + isaaclab_min

    # Step 2: Convert degrees to radians
    joint_pos = joint_pos / 180 * np.pi

    return joint_pos


# =============================================================================
# OBSERVATION PREPARATION
# =============================================================================

def prepare_observation(
    obs_dict: dict,
    device: torch.device,
    obs_state_mean: np.ndarray,
    obs_state_std: np.ndarray
) -> dict:
    """
    Prepare observation from Isaac Lab format to LeRobot format.

    Args:
        obs_dict: Observation dictionary from Isaac Lab environment
        device: Device to move tensors to
        obs_state_mean: Mean values for state normalization
        obs_state_std: Std values for state normalization

    Returns:
        Dictionary with LeRobot-formatted observations ready for policy input
    """
    # Extract observations from policy group
    policy_obs = obs_dict["policy"]

    # Get joint positions (shape: [num_envs, 6])
    joint_pos = policy_obs["joint_pos"].cpu().numpy()[0]  # Take first environment

    # Convert to LeRobot format
    joint_pos_lerobot = preprocess_isaac_to_lerobot(joint_pos, obs_state_mean, obs_state_std)

    # Get camera images (shape: [num_envs, height, width, channels])
    # Isaac Lab returns uint8 0-255, but policies expect float32 0-1
    front_img = policy_obs["front"].cpu().numpy()[0].astype(np.float32) / 255.0
    third_person_img = policy_obs["front_cam_cfg"].cpu().numpy()[0].astype(np.float32) / 255.0
    gripper_img = policy_obs["gripper_cam_cfg"].cpu().numpy()[0].astype(np.float32) / 255.0

    # Prepare LeRobot observation batch
    observation = {
        "observation.state": torch.from_numpy(joint_pos_lerobot).float().unsqueeze(0).to(device),
        "observation.images.front": torch.from_numpy(front_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
        "observation.images.third_person": torch.from_numpy(third_person_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
        "observation.images.gripper": torch.from_numpy(gripper_img).permute(2, 0, 1).float().unsqueeze(0).to(device),
    }

    return observation


# =============================================================================
# DATASET CONVERSION UTILITIES (for isaaclab2lerobot script)
# =============================================================================

def preprocess_joint_pos_batch(joint_pos: np.ndarray) -> np.ndarray:
    """
    Preprocess batch of joint positions for dataset conversion.

    Used in isaaclab2lerobot conversion script.

    Args:
        joint_pos: Joint positions in radians [T, 6]

    Returns:
        Converted joint positions in LeRobot range [T, 6]
    """
    # Convert radians to degrees
    joint_pos = joint_pos / np.pi * 180

    # Map from Isaac range to LeRobot range
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        isaac_range = isaaclab_max - isaaclab_min
        lerobot_range = lerobot_max - lerobot_min
        joint_pos[:, i] = (joint_pos[:, i] - isaaclab_min) / isaac_range * lerobot_range + lerobot_min

    return joint_pos
