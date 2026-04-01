"""Diagnose which dataset/episodes produce high-loss batches.

Loads a checkpoint, runs forward passes over each dataset separately,
and reports per-dataset and per-episode loss distributions.

Usage:
    python vbti/scripts/diagnose_loss_spikes.py \
        --checkpoint vbti/experiments/duck_cup_smolvla/v005/checkpoints/step_006000 \
        --config vbti/experiments/duck_cup_smolvla/v005/config.yaml \
        --n_batches 50
"""

import argparse
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

from vbti.logic.train.config_utils import TrainConfig
from vbti.logic.dataset.loading_utils import _resolve_root


def load_model_and_preprocessor(config, checkpoint_path):
    model_cfg = config.model
    dataset_cfg = config.dataset
    training_cfg = config.training

    # Use first source for metadata
    repo_id = dataset_cfg.sources[0].repo_id
    root = _resolve_root(repo_id, dataset_cfg.sources[0].root)
    meta = LeRobotDatasetMetadata(repo_id, root=str(root))

    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    camera_names = dataset_cfg.cameras.names
    if camera_names:
        allowed_keys = {f"observation.images.{n}" for n in camera_names}
        input_features = {
            k: f for k, f in input_features.items()
            if f.type is not FeatureType.VISUAL or k in allowed_keys
        }

    smolvla_cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=model_cfg.n_obs_steps,
        chunk_size=model_cfg.chunk_size,
        n_action_steps=model_cfg.chunk_size,
        freeze_vision_encoder=model_cfg.freeze_vision_encoder,
        train_expert_only=model_cfg.train_expert_only,
        train_state_proj=model_cfg.train_state_proj,
        empty_cameras=model_cfg.empty_cameras,
        tokenizer_max_length=model_cfg.tokenizer_max_length,
        num_steps=model_cfg.num_denoising_steps,
        optimizer_lr=training_cfg.lr,
        scheduler_decay_steps=training_cfg.steps,
        scheduler_decay_lr=training_cfg.decay_lr,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path, config=smolvla_cfg)
    policy.eval()
    policy.to(device)

    preprocessor, _ = make_pre_post_processors(smolvla_cfg, dataset_stats=meta.stats)
    return policy, preprocessor, meta, features


def make_dataset(repo_id, root, config, features):
    fps = 30  # default
    chunk_size = config.model.chunk_size
    n_obs_steps = config.model.n_obs_steps
    camera_names = config.dataset.cameras.names

    input_features = {k: f for k, f in features.items() if f.type is not FeatureType.ACTION}
    if camera_names:
        allowed_keys = {f"observation.images.{n}" for n in camera_names}
        input_features = {
            k: f for k, f in input_features.items()
            if f.type is not FeatureType.VISUAL or k in allowed_keys
        }

    obs_indices = list(range(1 - n_obs_steps, 1))
    action_indices = list(range(chunk_size))

    delta_timestamps = {"observation.state": [i / fps for i in obs_indices]}
    for key in input_features:
        if input_features[key].type is FeatureType.VISUAL:
            delta_timestamps[key] = [i / fps for i in obs_indices]
    delta_timestamps["action"] = [i / fps for i in action_indices]

    resolved_root = _resolve_root(repo_id, root)
    ds = LeRobotDataset(repo_id, root=str(resolved_root) if resolved_root else None,
                        delta_timestamps=delta_timestamps)
    return ds


def diagnose_dataset(policy, preprocessor, dataset, repo_id, n_batches, batch_size=16):
    device = next(policy.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    batch_losses = []
    sample_max_losses = []
    episode_losses = defaultdict(list)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            processed = preprocessor(batch)
            loss, loss_dict = policy.forward(processed)

            batch_loss = loss.item()
            batch_losses.append(batch_loss)

            # Per-sample max from the loss_dict debug tensors
            for key in ["losses_after_forward", "losses_after_rm_padding"]:
                if key in loss_dict and hasattr(loss_dict[key], 'shape'):
                    per_sample = loss_dict[key]
                    if per_sample.dim() >= 2:
                        # (batch, chunk, action_dim) → per-sample mean and max
                        sample_means = per_sample.mean(dim=tuple(range(1, per_sample.dim())))
                        sample_maxes = per_sample.amax(dim=tuple(range(1, per_sample.dim())))
                        sample_max_losses.extend(sample_maxes.cpu().tolist())

                        # Track by episode if available
                        if "episode_index" in batch:
                            eps = batch["episode_index"].cpu().tolist()
                            for ep, sm, sx in zip(eps, sample_means.cpu().tolist(), sample_maxes.cpu().tolist()):
                                episode_losses[int(ep)].append({"mean": sm, "max": sx})
                    break

    return {
        "repo_id": repo_id,
        "n_batches": len(batch_losses),
        "batch_loss_mean": float(np.mean(batch_losses)),
        "batch_loss_std": float(np.std(batch_losses)),
        "batch_loss_max": float(np.max(batch_losses)),
        "sample_max_mean": float(np.mean(sample_max_losses)) if sample_max_losses else None,
        "sample_max_p95": float(np.percentile(sample_max_losses, 95)) if sample_max_losses else None,
        "sample_max_p99": float(np.percentile(sample_max_losses, 99)) if sample_max_losses else None,
        "sample_max_max": float(np.max(sample_max_losses)) if sample_max_losses else None,
        "episode_losses": {
            int(ep): {
                "mean": float(np.mean([x["mean"] for x in losses])),
                "max_of_max": float(np.max([x["max"] for x in losses])),
                "n_samples": len(losses),
            }
            for ep, losses in episode_losses.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_batches", type=int, default=50)
    args = parser.parse_args()

    config = TrainConfig.load(args.config)
    print(f"Loading model from {args.checkpoint}...")
    policy, preprocessor, meta, features = load_model_and_preprocessor(config, args.checkpoint)

    results = []
    for src in config.dataset.sources:
        print(f"\n{'='*60}")
        print(f"Dataset: {src.repo_id} (source={src.source}, weight={src.weight})")
        print(f"{'='*60}")

        ds = make_dataset(src.repo_id, src.root, config, features)
        print(f"  {len(ds)} samples, {ds.meta.total_episodes} episodes")

        result = diagnose_dataset(policy, preprocessor, ds, src.repo_id, args.n_batches)
        results.append(result)

        print(f"  batch_loss:  mean={result['batch_loss_mean']:.5f}  std={result['batch_loss_std']:.5f}  max={result['batch_loss_max']:.5f}")
        if result["sample_max_mean"] is not None:
            print(f"  sample_max:  mean={result['sample_max_mean']:.2f}  p95={result['sample_max_p95']:.2f}  p99={result['sample_max_p99']:.2f}  max={result['sample_max_max']:.2f}")

        # Top offending episodes
        if result["episode_losses"]:
            sorted_eps = sorted(result["episode_losses"].items(), key=lambda x: x[1]["max_of_max"], reverse=True)
            print(f"\n  Top 10 worst episodes (by max loss):")
            for ep, stats in sorted_eps[:10]:
                print(f"    ep {ep:4d}: mean={stats['mean']:.5f}  max={stats['max_of_max']:.2f}  (n={stats['n_samples']})")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<50s} {'mean':>8s} {'std':>8s} {'max':>8s} {'p99_max':>8s}")
    for r in results:
        name = r["repo_id"].split("/")[-1][:45]
        p99 = f"{r['sample_max_p99']:.2f}" if r["sample_max_p99"] is not None else "n/a"
        print(f"{name:<50s} {r['batch_loss_mean']:8.5f} {r['batch_loss_std']:8.5f} {r['batch_loss_max']:8.5f} {p99:>8s}")

    # Save full results
    out_path = Path(args.checkpoint).parent.parent / "metrics" / "loss_spike_diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
