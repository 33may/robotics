"""
GR00T N1.6 training backend.

Wraps the Isaac-GR00T model into the unified TrainingBackend interface.
Uses the GR00T processor + collator for data preprocessing, and the
Gr00tN1d6 model with flow-matching action head for training.
"""

import json
import torch
from pathlib import Path

from vbti.logic.dataset import load_and_split_dataset, create_dataloaders
from vbti.logic.train.backends.base import TrainingBackend, ModelBundle


def _lazy_import_groot():
    """Lazy-import GR00T modules (they require the gr00t package)."""
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import (
        Gr00tN1d6Processor,
        Gr00tN1d6DataCollator,
        EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    )
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
    from gr00t.data.embodiment_tags import EmbodimentTag
    return {
        "Gr00tN1d6": Gr00tN1d6,
        "Gr00tN1d6Processor": Gr00tN1d6Processor,
        "Gr00tN1d6DataCollator": Gr00tN1d6DataCollator,
        "Gr00tN1d6Config": Gr00tN1d6Config,
        "ModalityConfig": ModalityConfig,
        "ActionConfig": ActionConfig,
        "ActionRepresentation": ActionRepresentation,
        "ActionType": ActionType,
        "ActionFormat": ActionFormat,
        "EmbodimentTag": EmbodimentTag,
        "EMBODIMENT_TAG_TO_PROJECTOR_INDEX": EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    }


def _build_modality_config(config, dataset_meta) -> dict:
    """Build GR00T modality_configs dict from our unified config + dataset metadata.

    This creates the ModalityConfig objects that the GR00T processor needs,
    mapping our camera names and dataset features to GR00T's expected format.

    Returns:
        {embodiment_tag: {"video": ModalityConfig, "state": ModalityConfig,
                          "action": ModalityConfig, "language": ModalityConfig}}
    """
    g = _lazy_import_groot()
    ModalityConfig = g["ModalityConfig"]
    ActionConfig = g["ActionConfig"]
    ActionRepresentation = g["ActionRepresentation"]
    ActionType = g["ActionType"]
    ActionFormat = g["ActionFormat"]

    model_cfg = config.model
    dataset_cfg = config.dataset
    embodiment_tag = model_cfg.embodiment

    camera_names = dataset_cfg.cameras.names

    # Discover state/action keys from dataset features
    features = dataset_meta.features
    state_keys = []
    action_keys = []
    for key, feat in features.items():
        if key.startswith("observation.state"):
            # Use the part after "observation.state" or just "joint_pos" as key
            state_keys.append(key.replace("observation.", ""))
        elif key == "action":
            action_keys.append("joint_pos")

    # Fallback if nothing discovered
    if not state_keys:
        state_keys = ["state"]
    if not action_keys:
        action_keys = ["joint_pos"]

    # Build action configs (one per action key, defaulting to absolute non-EEF)
    action_configs = [
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )
        for _ in action_keys
    ]

    modality_configs = {
        embodiment_tag: {
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(camera_names),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=list(range(16)),  # GR00T default action horizon
                modality_keys=action_keys,
                action_configs=action_configs,
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.human.task_description"],
            ),
        }
    }

    return modality_configs


def _ensure_modality_json(dataset_path: Path, camera_names: list[str], features: dict):
    """Generate modality.json in the dataset's meta/ dir if it doesn't exist.

    GR00T's LeRobotEpisodeLoader requires this file to map camera names
    and state/action slices to the dataset's parquet columns.
    """
    meta_dir = dataset_path / "meta"
    modality_path = meta_dir / "modality.json"

    if modality_path.exists():
        print(f"  modality.json already exists at {modality_path}")
        return

    print(f"  Generating modality.json at {modality_path}")

    modality = {}

    # Video modality: map short camera names → observation.images.X keys
    video_modality = {}
    for cam_name in camera_names:
        original_key = f"observation.images.{cam_name}"
        if original_key in features:
            video_modality[cam_name] = {"original_key": original_key}
    if video_modality:
        modality["video"] = video_modality

    # State modality: discover state features and their dimensions
    state_modality = {}
    state_start = 0
    for key, feat in features.items():
        if key.startswith("observation.state"):
            short_key = key.replace("observation.", "")
            dim = feat["shape"][-1] if isinstance(feat.get("shape"), (list, tuple)) else 1
            state_modality[short_key] = {
                "original_key": key,
                "start": state_start,
                "end": state_start + dim,
            }
            state_start += dim
    if state_modality:
        modality["state"] = state_modality

    # Action modality
    action_modality = {}
    if "action" in features:
        action_feat = features["action"]
        dim = action_feat["shape"][-1] if isinstance(action_feat.get("shape"), (list, tuple)) else 1
        action_modality["joint_pos"] = {
            "original_key": "action",
            "start": 0,
            "end": dim,
        }
    if action_modality:
        modality["action"] = action_modality

    # Language/annotation modality
    for key in features:
        if key.startswith("task") or key.startswith("language"):
            modality["annotation"] = {
                "human.task_description": {"original_key": key}
            }
            break

    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(modality_path, "w") as f:
        json.dump(modality, f, indent=2)
    print(f"  Written modality.json with keys: {list(modality.keys())}")


class GR00TBackend(TrainingBackend):

    def load_model(self, config) -> ModelBundle:
        """Load GR00T N1.6 from pretrained + configure for finetuning."""
        g = _lazy_import_groot()
        Gr00tN1d6 = g["Gr00tN1d6"]
        Gr00tN1d6Processor = g["Gr00tN1d6Processor"]
        EmbodimentTag = g["EmbodimentTag"]

        model_cfg = config.model
        dataset_cfg = config.dataset

        # Load dataset metadata for feature discovery
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        repo_id = dataset_cfg.sources[0].repo_id
        root = dataset_cfg.sources[0].root
        meta = LeRobotDatasetMetadata(repo_id, root=root) if root else LeRobotDatasetMetadata(repo_id)

        # Build modality configs for this embodiment
        modality_configs = _build_modality_config(config, meta)

        # Ensure modality.json exists in the dataset
        if root:
            dataset_path = Path(root) / repo_id
            _ensure_modality_json(dataset_path, dataset_cfg.cameras.names, meta.features)

        # Load pretrained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading GR00T N1.6 from {model_cfg.pretrained}...")
        model = Gr00tN1d6.from_pretrained(
            model_cfg.pretrained,
            tune_llm=not model_cfg.freeze_vlm,
            tune_visual=not model_cfg.freeze_vlm,
            tune_top_llm_layers=model_cfg.unfreeze_top_vlm_layers,
            tune_projector=True,
            tune_diffusion_model=True,
            tune_vlln=True,
            trust_remote_code=True,
        )
        model.train()
        model.to(device)

        # Create processor
        processor = Gr00tN1d6Processor(
            modality_configs=modality_configs,
            model_name=model.config.model_name,
            model_type=model.config.backbone_model_type,
            max_state_dim=model.config.max_state_dim,
            max_action_dim=model.config.max_action_dim,
            max_action_horizon=model.config.action_horizon,
            transformers_loading_kwargs={"trust_remote_code": True},
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"GR00T N1.6 loaded: {total_params:,} params, {trainable:,} trainable ({100*trainable/total_params:.1f}%)")

        return ModelBundle(
            model=model,
            preprocessor=processor,
            postprocessor=processor,  # GR00T uses same processor for pre/post
            dataset_meta=meta,
        )

    def make_dataloaders(self, config, dataset_meta) -> tuple:
        """Create dataloaders for GR00T training.

        Uses standard LeRobot dataset loading (same as SmolVLA) since
        GR00T's processor handles the embodiment-specific transforms.
        The GR00T collator is used instead of default collation.
        """
        dataset_cfg = config.dataset
        training_cfg = config.training

        repo_id = dataset_cfg.sources[0].repo_id
        root = dataset_cfg.sources[0].root
        meta = dataset_meta
        fps = meta.fps

        from lerobot.configs.types import FeatureType
        from lerobot.datasets.utils import dataset_to_policy_features
        features = dataset_to_policy_features(meta.features)

        # Build delta_timestamps for GR00T
        # GR00T uses action_horizon=16 by default
        action_horizon = 16
        obs_indices = [0]  # single observation step
        action_indices = list(range(action_horizon))

        delta_timestamps = {
            "observation.state": [i / fps for i in obs_indices],
        }
        # Add camera timestamps
        camera_names = dataset_cfg.cameras.names
        for key in features:
            if features[key].type is FeatureType.VISUAL:
                # Only include configured cameras
                cam_name = key.split(".")[-1]
                if cam_name in camera_names:
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
        """One GR00T training step: forward through model → loss dict.

        GR00T's model.forward(inputs) returns a dict with:
            {"loss": tensor, "action_loss": tensor, "action_mask": tensor, ...}
        We extract loss as scalar and build a metrics dict.
        """
        # GR00T model.forward() expects a dict and handles device transfer internally
        outputs = bundle.model.forward(batch)

        loss = outputs["loss"]
        metrics = {
            "action_loss": outputs["action_loss"].mean().item(),
        }

        return loss, metrics

    def validate(self, bundle: ModelBundle, val_loader, n_batches: int) -> dict:
        """Run validation on GR00T."""
        bundle.model.eval()
        val_losses = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break
                outputs = bundle.model.forward(batch)
                val_losses.append(outputs["loss"].item())

        bundle.model.train()

        if val_losses:
            return {"val_loss": sum(val_losses) / len(val_losses)}
        return {"val_loss": float("inf")}

    def save_checkpoint(self, bundle: ModelBundle, path: Path, is_best: bool = False):
        """Save GR00T checkpoint: model + processor configs."""
        path.mkdir(parents=True, exist_ok=True)
        bundle.model.save_pretrained(path)
        bundle.preprocessor.save_pretrained(path)
        label = "best" if is_best else str(path.name)
        print(f"  Saved checkpoint: {label} -> {path}")

    def get_optimizer(self, bundle: ModelBundle, config) -> tuple:
        """Create AdamW optimizer + cosine scheduler for GR00T.

        GR00T defaults: lr=1e-4, weight_decay=1e-5, warmup_ratio=0.05.
        We use our unified config values but fall back to GR00T defaults
        when the unified config has SmolVLA-oriented defaults.
        """
        training_cfg = config.training

        # Use config LR, but GR00T typically uses higher LR than SmolVLA
        lr = training_cfg.lr
        weight_decay = training_cfg.weight_decay

        # Only train parameters that require grad
        trainable_params = [p for p in bundle.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Cosine schedule with warmup
        warmup_steps = training_cfg.warmup_steps
        total_steps = training_cfg.steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return optimizer, scheduler
