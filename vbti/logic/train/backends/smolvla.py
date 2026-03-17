"""
SmolVLA training backend.

Wraps LeRobot's SmolVLA policy into the unified TrainingBackend interface.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from pathlib import Path

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

from vbti.logic.dataset import load_and_split_dataset, create_dataloaders
from vbti.logic.train.backends.base import TrainingBackend, ModelBundle


class SmolVLABackend(TrainingBackend):

    def load_model(self, config) -> ModelBundle:
        """Load SmolVLA from pretrained + configure for finetuning."""
        model_cfg = config.model
        dataset_cfg = config.dataset
        training_cfg = config.training

        # Use first source's repo_id for metadata
        repo_id = dataset_cfg.sources[0].repo_id
        root = dataset_cfg.sources[0].root
        meta = LeRobotDatasetMetadata(repo_id, root=root) if root else LeRobotDatasetMetadata(repo_id)

        # Build feature maps from dataset
        features = dataset_to_policy_features(meta.features)
        output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
        input_features = {k: f for k, f in features.items() if k not in output_features}

        # Filter to configured cameras if specified
        camera_names = dataset_cfg.cameras.names
        if camera_names:
            allowed_keys = {f"observation.images.{n}" for n in camera_names}
            input_features = {
                k: f for k, f in input_features.items()
                if f.type is not FeatureType.VISUAL or k in allowed_keys
            }

        # Build SmolVLA native config
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
            optimizer_weight_decay=training_cfg.weight_decay,
            optimizer_grad_clip_norm=training_cfg.grad_clip_norm,
            scheduler_warmup_steps=training_cfg.warmup_steps,
            scheduler_decay_steps=training_cfg.steps,
            scheduler_decay_lr=training_cfg.decay_lr,
        )

        # Load pretrained weights with our config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = SmolVLAPolicy.from_pretrained(model_cfg.pretrained, config=smolvla_cfg)
        policy.train()
        policy.to(device)

        # Create pre/post processors
        preprocessor, postprocessor = make_pre_post_processors(
            smolvla_cfg, dataset_stats=meta.stats
        )

        total_params = sum(p.numel() for p in policy.parameters())
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"SmolVLA loaded: {total_params:,} params, {trainable:,} trainable ({100*trainable/total_params:.1f}%)")

        return ModelBundle(
            model=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_meta=meta,
        )

    def make_dataloaders(self, config, dataset_meta) -> tuple:
        """Create dataloaders with proper delta_timestamps for SmolVLA."""
        dataset_cfg = config.dataset
        training_cfg = config.training

        # Get the SmolVLA config from the model to compute delta timestamps
        # We need the policy config that was used during load_model
        repo_id = dataset_cfg.sources[0].repo_id
        root = dataset_cfg.sources[0].root
        meta = dataset_meta
        fps = meta.fps

        features = dataset_to_policy_features(meta.features)
        input_features = {k: f for k, f in features.items() if f.type is not FeatureType.ACTION}

        # Filter cameras
        camera_names = dataset_cfg.cameras.names
        if camera_names:
            allowed_keys = {f"observation.images.{n}" for n in camera_names}
            input_features = {
                k: f for k, f in input_features.items()
                if f.type is not FeatureType.VISUAL or k in allowed_keys
            }

        # Build delta_timestamps from model config
        chunk_size = config.model.chunk_size
        n_obs_steps = config.model.n_obs_steps
        obs_indices = list(range(1 - n_obs_steps, 1))  # e.g. [0] for n_obs_steps=1
        action_indices = list(range(chunk_size))         # e.g. [0..49] for chunk_size=50

        delta_timestamps = {
            "observation.state": [i / fps for i in obs_indices],
        }
        for key in input_features:
            if input_features[key].type is FeatureType.VISUAL:
                delta_timestamps[key] = [i / fps for i in obs_indices]
        delta_timestamps["action"] = [i / fps for i in action_indices]

        # Load and split
        _, train_dataset, val_dataset = load_and_split_dataset(
            repo_id=repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            train_ratio=dataset_cfg.train_ratio,
        )

        # Create dataloaders
        if val_dataset is not None:
            train_loader, val_loader = create_dataloaders(
                train_dataset, val_dataset,
                batch_size=training_cfg.batch_size,
                num_workers=training_cfg.num_workers,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=training_cfg.batch_size,
                shuffle=True,
                num_workers=training_cfg.num_workers,
            )
            val_loader = None

        return train_loader, val_loader

    def train_step(self, bundle: ModelBundle, batch, optimizer) -> tuple[torch.Tensor, dict]:
        """One SmolVLA training step: preprocess → forward → loss."""
        batch = bundle.preprocessor(batch)
        loss, loss_dict = bundle.model.forward(batch)
        return loss, loss_dict

    def validate(self, bundle: ModelBundle, val_loader, n_batches: int) -> dict:
        """Run validation on SmolVLA."""
        bundle.model.eval()
        val_losses = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break
                batch = bundle.preprocessor(batch)
                loss, _ = bundle.model.forward(batch)
                val_losses.append(loss.item())

        bundle.model.train()

        if val_losses:
            return {"val_loss": sum(val_losses) / len(val_losses)}
        return {"val_loss": float("inf")}

    def save_checkpoint(self, bundle: ModelBundle, path: Path, is_best: bool = False):
        """Save SmolVLA checkpoint: model + preprocessor + postprocessor."""
        path.mkdir(parents=True, exist_ok=True)
        bundle.model.save_pretrained(path)
        bundle.preprocessor.save_pretrained(path)
        bundle.postprocessor.save_pretrained(path)
        label = "best" if is_best else str(path.name)
        print(f"  Saved checkpoint: {label} → {path}")

    def get_optimizer(self, bundle: ModelBundle, config) -> tuple:
        """Create optimizer and scheduler using SmolVLA's presets."""
        policy = bundle.model
        smolvla_cfg = policy.config

        optimizer_preset = smolvla_cfg.get_optimizer_preset()
        optimizer = optimizer_preset.build(policy.parameters())

        scheduler_preset = smolvla_cfg.get_scheduler_preset()
        scheduler = scheduler_preset.build(optimizer, config.training.steps)

        return optimizer, scheduler
