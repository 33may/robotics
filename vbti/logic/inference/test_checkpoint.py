"""
Quick offline test for a trained SmolVLA checkpoint.

Loads the model, feeds it real batches from the training dataset,
compares predicted actions vs ground truth. No robot or sim needed.

Usage:
    python vbti/logic/inference/test_checkpoint.py test \
        --checkpoint=vbti/experiments/duck_cup_smolvla/v001/checkpoints/best

    python vbti/logic/inference/test_checkpoint.py test \
        --checkpoint=vbti/experiments/duck_cup_smolvla/v001/checkpoints/step_001000 \
        --n_samples=20 --plot
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features


def _resolve_checkpoint(checkpoint: str) -> Path:
    """Resolve checkpoint path, handling relative paths."""
    p = Path(checkpoint).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _load_policy(checkpoint_path: Path, device: torch.device):
    """Load policy + preprocessor + postprocessor from checkpoint."""
    print(f"Loading policy from {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_preprocessor.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_postprocessor.json",
    )

    return policy, preprocessor, postprocessor


def _load_dataset_sample(repo_id: str, root: str | None, n_samples: int,
                         policy_config, dataset_meta):
    """Load a few samples from the dataset with proper delta_timestamps."""
    features = dataset_to_policy_features(dataset_meta.features)
    input_features = {k: f for k, f in features.items() if f.type is not FeatureType.ACTION}
    fps = dataset_meta.fps

    # Build delta timestamps from policy config
    delta_timestamps = {
        "observation.state": [i / fps for i in policy_config.observation_delta_indices],
    }
    for key in input_features:
        if input_features[key].type is FeatureType.VISUAL:
            delta_timestamps[key] = [i / fps for i in policy_config.observation_delta_indices]
    delta_timestamps["action"] = [i / fps for i in policy_config.action_delta_indices]

    # Resolve root from cache if needed
    if root is None:
        cache_path = Path.home() / ".cache/huggingface/lerobot" / repo_id
        if cache_path.exists():
            root = str(cache_path.resolve())

    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)

    # Sample random indices
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]
    return samples


def test(checkpoint: str, repo_id: str = None, root: str = None,
         n_samples: int = 10, plot: bool = False, device: str = "auto"):
    """Test a checkpoint against dataset samples.

    Args:
        checkpoint: path to checkpoint directory
        repo_id: dataset repo_id (auto-detected from checkpoint config if not given)
        root: dataset root path (auto-resolved from cache)
        n_samples: number of samples to test
        plot: save comparison plots
        device: "auto", "cuda", "cpu"
    """
    # Resolve device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load checkpoint
    checkpoint_path = _resolve_checkpoint(checkpoint)
    policy, preprocessor, postprocessor = _load_policy(checkpoint_path, device)

    # Auto-detect dataset from checkpoint config
    if repo_id is None:
        import yaml
        # Try to find config.yaml in parent directories
        for parent in [checkpoint_path.parent, checkpoint_path.parent.parent,
                       checkpoint_path.parent.parent.parent]:
            config_file = parent / "config.yaml"
            if config_file.exists():
                with open(config_file) as f:
                    cfg = yaml.safe_load(f)
                repo_id = cfg.get("dataset", {}).get("sources", [{}])[0].get("repo_id")
                root = root or cfg.get("dataset", {}).get("sources", [{}])[0].get("root")
                print(f"Auto-detected dataset: {repo_id}")
                break
        if not repo_id:
            raise ValueError("Could not auto-detect dataset. Pass --repo_id explicitly.")

    if root:
        root = str(Path(root).expanduser())

    # Load dataset metadata
    meta_kwargs = {"repo_id": repo_id}
    if root:
        meta_kwargs["root"] = root
    dataset_meta = LeRobotDatasetMetadata(**meta_kwargs)

    # Load samples
    print(f"Loading {n_samples} samples from {repo_id}...")
    samples = _load_dataset_sample(repo_id, root, n_samples, policy.config, dataset_meta)

    # Run inference on each sample
    print(f"\nTesting {len(samples)} samples...")
    print("=" * 70)

    all_errors = []
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]

    for i, sample in enumerate(samples):
        # Ground truth actions (raw, from dataset — already in dataset units)
        gt_actions = sample["action"]  # (chunk_size, action_dim)

        # Preprocess and run policy
        batch = preprocessor(sample)
        with torch.inference_mode():
            pred_normalized = policy.select_action(batch)

        # Denormalize predictions back to dataset units
        pred_denorm = postprocessor({"action": pred_normalized})["action"]

        # Compare first action (most important)
        gt_first = gt_actions[0].cpu().numpy()
        pred_first = pred_denorm[0].cpu().numpy()
        error = np.abs(gt_first - pred_first)
        all_errors.append(error)

        if i < 5:  # Print details for first 5 samples
            print(f"\nSample {i}:")
            print(f"  GT actions[0]:   {np.round(gt_first, 2)}")
            print(f"  Pred actions[0]: {np.round(pred_first, 2)}")
            print(f"  Abs error:       {np.round(error, 2)}")
            print(f"  Mean error:      {error.mean():.4f}")

    # Summary
    errors = np.stack(all_errors)
    print("\n" + "=" * 70)
    print(f"RESULTS ({len(samples)} samples)")
    print("=" * 70)
    print(f"\nPer-joint mean absolute error:")
    for j, name in enumerate(joint_names[:errors.shape[1]]):
        print(f"  {name:15s}  MAE={errors[:, j].mean():.4f}  std={errors[:, j].std():.4f}")
    print(f"\nOverall MAE: {errors.mean():.4f}")
    print(f"Overall max error: {errors.max():.4f}")

    # Sanity checks
    print(f"\nSanity checks:")
    if errors.mean() < 1.0:
        print(f"  [OK] Mean error < 1.0 — model is learning")
    elif errors.mean() < 5.0:
        print(f"  [WARN] Mean error 1-5 — model partially learned, needs more training")
    else:
        print(f"  [BAD] Mean error > 5 — model may not be learning correctly")

    if errors.max() > 50:
        print(f"  [WARN] Max error > 50 — some predictions are way off")

    # Plot
    if plot:
        _plot_comparison(samples, policy, preprocessor, postprocessor,
                         joint_names, checkpoint_path)

    return {"mae": float(errors.mean()), "max_error": float(errors.max()),
            "per_joint_mae": {joint_names[j]: float(errors[:, j].mean())
                              for j in range(min(len(joint_names), errors.shape[1]))}}


def _plot_comparison(samples, policy, preprocessor, postprocessor,
                     joint_names, checkpoint_path):
    """Plot predicted vs ground truth actions for a few samples."""
    import matplotlib.pyplot as plt

    n_plot = min(3, len(samples))
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 4 * n_plot))
    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        sample = samples[i]
        gt = sample["action"].cpu().numpy()

        batch = preprocessor(sample)
        with torch.inference_mode():
            pred_norm = policy.select_action(batch)
        pred = postprocessor({"action": pred_norm})["action"].cpu().numpy()

        ax = axes[i]
        n_joints = min(gt.shape[1], len(joint_names))
        x = range(gt.shape[0])

        for j in range(n_joints):
            color = f"C{j}"
            ax.plot(x, gt[:, j], color=color, alpha=0.7, linewidth=1.5,
                    label=f"{joint_names[j]} (GT)" if i == 0 else None)
            ax.plot(x, pred[:, j], color=color, alpha=0.7, linewidth=1.5,
                    linestyle="--", label=f"{joint_names[j]} (pred)" if i == 0 else None)

        ax.set_ylabel(f"Sample {i}")
        ax.set_xlabel("Action step")
        ax.grid(True, alpha=0.3)

    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.suptitle(f"Checkpoint: {checkpoint_path.name}", fontsize=12)
    plt.tight_layout()

    save_path = checkpoint_path / "test_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "test": test,
    })
