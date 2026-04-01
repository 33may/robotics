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
from vbti.logic.dataset.loading_utils import _resolve_root
from vbti.logic.train.backends.base import TrainingBackend, ModelBundle


class SmolVLABackend(TrainingBackend):

    def load_model(self, config) -> ModelBundle:
        """Load SmolVLA from pretrained + configure for finetuning."""
        model_cfg = config.model
        dataset_cfg = config.dataset
        training_cfg = config.training

        # Use first source — engine handles aggregation before calling this
        repo_id = dataset_cfg.sources[0].repo_id
        root = _resolve_root(repo_id, dataset_cfg.sources[0].root)
        meta = LeRobotDatasetMetadata(repo_id, root=str(root)) if root else LeRobotDatasetMetadata(repo_id)

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

        # Align camera names to pretrained slots (camera1, camera2, camera3, ...)
        # so that pretrained vision weights are properly utilized.
        # Uses config camera order (which the user controls) rather than alphabetical.
        pretrained_slots = ["camera1", "camera2", "camera3"]
        if camera_names:
            dataset_cam_keys = [f"observation.images.{n}" for n in camera_names
                                if f"observation.images.{n}" in input_features]
        else:
            dataset_cam_keys = sorted(k for k, f in input_features.items() if f.type is FeatureType.VISUAL)
        rename_map = {}
        n_empty = 0
        for i, cam_key in enumerate(dataset_cam_keys):
            cam_name = cam_key.split(".")[-1]  # e.g. "top" from "observation.images.top"
            if i < len(pretrained_slots):
                new_name = pretrained_slots[i]
            else:
                new_name = f"empty_camera_{n_empty}"
                n_empty += 1
            new_key = f"observation.images.{new_name}"
            if cam_key != new_key:
                rename_map[cam_key] = new_key

        # Apply remap to input_features
        if rename_map:
            remapped_features = {}
            for k, f in input_features.items():
                remapped_features[rename_map.get(k, k)] = f
            input_features = remapped_features
            print(f"Camera alignment: {rename_map}")

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
            empty_cameras=max(model_cfg.empty_cameras, n_empty),
            tokenizer_max_length=model_cfg.tokenizer_max_length,
            num_steps=model_cfg.num_denoising_steps,
            optimizer_lr=training_cfg.lr,
            optimizer_weight_decay=training_cfg.weight_decay,
            optimizer_grad_clip_norm=training_cfg.grad_clip_norm,
            scheduler_warmup_steps=training_cfg.warmup_steps,
            scheduler_decay_steps=training_cfg.steps,
            scheduler_decay_lr=training_cfg.decay_lr,
        )

        # Remap dataset stats to match aligned camera names
        stats = meta.stats
        if rename_map:
            stats = {rename_map.get(k, k): v for k, v in stats.items()}

        # Load pretrained weights with our config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = SmolVLAPolicy.from_pretrained(model_cfg.pretrained, config=smolvla_cfg)
        policy.train()
        policy.to(device)

        # Create pre/post processors
        preprocessor, postprocessor = make_pre_post_processors(
            smolvla_cfg, dataset_stats=stats
        )

        total_params = sum(p.numel() for p in policy.parameters())
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"SmolVLA loaded: {total_params:,} params, {trainable:,} trainable ({100*trainable/total_params:.1f}%)")

        bundle = ModelBundle(
            model=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_meta=meta,
        )
        bundle.camera_rename_map = rename_map
        return bundle

    def make_dataloaders(self, config, dataset_meta) -> tuple:
        """Create dataloaders with proper delta_timestamps for SmolVLA.

        Supports single and multi-source datasets. Multi-source uses
        WeightedRandomSampler to respect per-source weights.
        """
        dataset_cfg = config.dataset
        training_cfg = config.training
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
        obs_indices = list(range(1 - n_obs_steps, 1))
        action_indices = list(range(chunk_size))

        delta_timestamps = {
            "observation.state": [i / fps for i in obs_indices],
        }
        for key in input_features:
            if input_features[key].type is FeatureType.VISUAL:
                delta_timestamps[key] = [i / fps for i in obs_indices]
        delta_timestamps["action"] = [i / fps for i in action_indices]

        # Use first source — engine handles aggregation before calling this
        repo_id = dataset_cfg.sources[0].repo_id
        root = _resolve_root(repo_id, dataset_cfg.sources[0].root)

        _, train_dataset, val_dataset = load_and_split_dataset(
            repo_id=repo_id,
            root=str(root) if root else None,
            delta_timestamps=delta_timestamps,
            train_ratio=dataset_cfg.train_ratio,
        )

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
        """One SmolVLA training step: rename cameras → preprocess → forward → loss."""
        rename_map = getattr(bundle, 'camera_rename_map', None)
        if rename_map:
            batch = {rename_map.get(k, k): v for k, v in batch.items()}
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
                rename_map = getattr(bundle, 'camera_rename_map', None)
                if rename_map:
                    batch = {rename_map.get(k, k): v for k, v in batch.items()}
                batch = bundle.preprocessor(batch)
                loss, _ = bundle.model.forward(batch)
                val_losses.append(loss.item())

        bundle.model.train()

        if val_losses:
            return {"val_loss": sum(val_losses) / len(val_losses)}
        return {"val_loss": float("inf")}

    def save_checkpoint(self, bundle: ModelBundle, path: Path, is_best: bool = False,
                        optimizer=None, scheduler=None, step: int = None):
        """Save SmolVLA checkpoint: model + preprocessor + postprocessor + training state."""
        path.mkdir(parents=True, exist_ok=True)
        bundle.model.save_pretrained(path)
        bundle.preprocessor.save_pretrained(path)
        bundle.postprocessor.save_pretrained(path)

        # Save training state for resume
        if optimizer is not None:
            state = {"optimizer": optimizer.state_dict()}
            if scheduler is not None:
                state["scheduler"] = scheduler.state_dict()
            if step is not None:
                state["step"] = step
            torch.save(state, path / "training_state.pt")

        label = "best" if is_best else str(path.name)
        print(f"  Saved checkpoint: {label} → {path}")

    def load_checkpoint_for_resume(self, bundle: ModelBundle, path: Path,
                                   optimizer, scheduler, reset_lr: bool = False) -> int:
        """Load model weights + training state from checkpoint. Returns the step to resume from.

        Args:
            reset_lr: if True, skip scheduler state restoration so a fresh schedule can be applied.
        """
        # Load model weights
        bundle.model = SmolVLAPolicy.from_pretrained(path, config=bundle.model.config)
        bundle.model.train()
        bundle.model.to(next(iter(optimizer.param_groups[0]['params'])).device)

        # Load training state
        state_path = path / "training_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"No training_state.pt in {path} — checkpoint was saved without resume support")

        state = torch.load(state_path, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        if not reset_lr and "scheduler" in state and scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])

        step = state.get("step", 0)
        print(f"  Resumed from checkpoint: {path.name} (step {step})")
        if reset_lr:
            print(f"  LR schedule reset — scheduler state NOT restored")
        return step

    def get_optimizer(self, bundle: ModelBundle, config) -> tuple:
        """Create optimizer and scheduler using SmolVLA's presets."""
        policy = bundle.model
        smolvla_cfg = policy.config

        optimizer_preset = smolvla_cfg.get_optimizer_preset()
        optimizer = optimizer_preset.build(policy.parameters())

        if config.training.lr_schedule == "wsd":
            scheduler = self._build_wsd_scheduler(optimizer, config)
        else:
            scheduler_preset = smolvla_cfg.get_scheduler_preset()
            scheduler = scheduler_preset.build(optimizer, config.training.steps)

        return optimizer, scheduler

    @staticmethod
    def _build_wsd_scheduler(optimizer, config):
        """Warmup-Stable-Decay: hold peak LR, then cosine decay at the end."""
        import math
        from torch.optim.lr_scheduler import LambdaLR

        cfg_t = config.training
        warmup = cfg_t.warmup_steps
        total = cfg_t.steps
        decay_steps = int(total * cfg_t.decay_ratio)
        stable_end = total - decay_steps
        alpha = cfg_t.decay_lr / cfg_t.lr

        def lr_lambda(step):
            if step < warmup:
                return max(1 / (warmup + 1), step / warmup)
            if step < stable_end:
                return 1.0
            progress = (step - stable_end) / max(1, decay_steps)
            return alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * progress))

        print(f"  WSD schedule: warmup={warmup}, stable={warmup}–{stable_end}, "
              f"decay={stable_end}–{total} ({decay_steps} steps)")
        return LambdaLR(optimizer, lr_lambda)
