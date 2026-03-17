"""
Unified training configs for all model backends.

Draccus dataclasses — typed, serializable to YAML/JSON, CLI-overridable.
Each model backend translates these into its native config format.

Usage:
    from vbti.logic.train.configs import TrainConfig, SmolVLAModelConfig

    config = TrainConfig(
        model=SmolVLAModelConfig(chunk_size=50),
        dataset=DatasetConfig(repo_id="eternalmay33/lift_cube_3cams"),
        training=TrainingConfig(steps=10000, lr=1e-5),
    )

    # Serialize
    config.save("path/to/config.yaml")
    loaded = TrainConfig.load("path/to/config.yaml")
"""

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import yaml


# ── Enums ─────────────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    SMOLVLA = "smolvla"
    GROOT = "groot"


# ── Dataset ───────────────────────────────────────────────────────────────────

class DataSource(str, Enum):
    SIM = "sim"
    REAL = "real"
    SYNTHETIC = "synthetic"   # cosmos-transferred or augmented
    MIXED = "mixed"


@dataclass
class DatasetSource:
    """A single dataset source with sampling weight and role."""
    repo_id: str = ""
    root: str | None = None
    episodes: list[int] | None = None
    weight: float = 1.0
    source: DataSource = DataSource.SIM
    # Per-source role: train, val, or both
    role: str = "both"  # "train" | "val" | "both"

    def __post_init__(self):
        if self.root:
            self.root = str(Path(self.root).expanduser())


@dataclass
class CameraConfig:
    """Camera setup — explicit ordering and naming."""
    # Ordered list of camera names as they appear in the dataset.
    # Order matters — both SmolVLA and GR00T are order-sensitive.
    # e.g. ["front", "wrist", "gripper"]
    names: list[str] = field(default_factory=lambda: ["front"])
    # Optional remap: dataset key → model key. If None, names used as-is.
    # e.g. {"cam_top": "front", "cam_wrist": "wrist"}
    remap: dict[str, str] | None = None


@dataclass
class DatasetConfig:
    """Multi-source dataset config with flexible train/val routing."""
    sources: list[DatasetSource] = field(default_factory=lambda: [DatasetSource()])
    cameras: CameraConfig = field(default_factory=CameraConfig)
    # Global train/val split ratio (applied to sources with role="both")
    train_ratio: float = 0.95
    # Validation filtering — which sources to include in val loss
    val_sources: list[str] | None = None  # None = all, or filter by source type: ["real", "sim"]
    use_imagenet_stats: bool = True


# ── Model configs (one per backend) ──────────────────────────────────────────

@dataclass
class SmolVLAModelConfig:
    type: ModelType = ModelType.SMOLVLA
    pretrained: str = "lerobot/smolvla_base"
    chunk_size: int = 50
    n_obs_steps: int = 1
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
    empty_cameras: int = 0           # pad with fake cameras when training with fewer than base expects
    tokenizer_max_length: int = 48   # increase for longer task descriptions
    num_denoising_steps: int = 10    # diffusion steps at inference


@dataclass
class GR00TModelConfig:
    type: ModelType = ModelType.GROOT
    pretrained: str = "nvidia/GR00T-N1.6-3B"
    embodiment: str = "new_embodiment"
    freeze_vlm: bool = True
    unfreeze_top_vlm_layers: int = 4
    deepspeed_config: str = "zero2"


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    steps: int = 10_000
    batch_size: int = 4
    lr: float = 1e-5
    weight_decay: float = 1e-10
    grad_clip_norm: float = 10.0
    warmup_steps: int = 500
    decay_lr: float = 2.5e-6         # final LR after cosine decay
    num_workers: int = 0
    device: str = "auto"        # "auto", "cuda", "cpu", "cuda:0", etc.
    seed: int = 42
    fp16: bool = False
    bf16: bool = True


# ── Logging ───────────────────────────────────────────────────────────────────

@dataclass
class LoggingConfig:
    log_freq: int = 100
    save_freq: int = 1000
    val_freq: int = 500
    val_size: int = 50
    wandb_enabled: bool = False
    wandb_project: str = "vbti-training"
    wandb_entity: str | None = None
    wandb_mode: str = "online"   # "online", "offline", "disabled"


# ── Eval ──────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    n_episodes: int = 50
    sim_env: str | None = None
    record_video: bool = True


# ── Top-level config ─────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    model: SmolVLAModelConfig | GR00TModelConfig = field(default_factory=SmolVLAModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def model_type(self) -> ModelType:
        return self.model.type

    def save(self, path: str | Path):
        """Serialize config to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        """Load config from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert to plain dict for serialization."""
        from dataclasses import asdict
        d = asdict(self)
        # Convert enums to their string values
        d["model"]["type"] = d["model"]["type"].value
        for src in d.get("dataset", {}).get("sources", []):
            if isinstance(src.get("source"), DataSource):
                src["source"] = src["source"].value
            # already a string from asdict
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """Reconstruct config from dict, picking correct model/dataset classes."""
        # Model
        model_data = data.get("model", {})
        model_type = model_data.get("type", "smolvla")
        if model_type == "smolvla":
            model_data["type"] = ModelType.SMOLVLA
            model = SmolVLAModelConfig(**model_data)
        elif model_type == "groot":
            model_data["type"] = ModelType.GROOT
            model = GR00TModelConfig(**model_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Dataset with nested sources + cameras
        ds_data = data.get("dataset", {})
        sources_raw = ds_data.pop("sources", [])
        sources = []
        for s in sources_raw:
            if isinstance(s.get("source"), str):
                s["source"] = DataSource(s["source"])
            sources.append(DatasetSource(**s))
        cameras_raw = ds_data.pop("cameras", {})
        cameras = CameraConfig(**cameras_raw) if cameras_raw else CameraConfig()
        dataset = DatasetConfig(sources=sources, cameras=cameras, **ds_data)

        return cls(
            model=model,
            dataset=dataset,
            training=TrainingConfig(**data.get("training", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            eval=EvalConfig(**data.get("eval", {})),
        )

    def diff(self, other: "TrainConfig") -> dict:
        """Return fields that differ between two configs."""
        d1, d2 = self.to_dict(), other.to_dict()
        diffs = {}
        for section in d1:
            if d1[section] != d2.get(section):
                diffs[section] = {
                    k: {"old": d1[section][k], "new": d2[section][k]}
                    for k in d1[section]
                    if d1[section].get(k) != d2.get(section, {}).get(k)
                }
        return diffs


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _schema(section: str = "all"):
    """Print config schema with defaults. Sections: all, model, smolvla, groot, dataset, training, logging, eval, source."""
    from dataclasses import fields as dc_fields
    targets = {
        "smolvla": SmolVLAModelConfig,
        "groot": GR00TModelConfig,
        "dataset": DatasetConfig,
        "camera": CameraConfig,
        "source": DatasetSource,
        "training": TrainingConfig,
        "logging": LoggingConfig,
        "eval": EvalConfig,
    }
    if section == "all":
        show = targets
    elif section in targets:
        show = {section: targets[section]}
    elif section == "model":
        show = {"smolvla": SmolVLAModelConfig, "groot": GR00TModelConfig}
    else:
        print(f"Unknown section: {section}. Available: {', '.join(['all', 'model'] + list(targets.keys()))}")
        return

    for name, cls in show.items():
        print(f"\n[{name}]")
        for f in dc_fields(cls):
            default = f.default if f.default is not f.default_factory else f.default_factory()  # type: ignore
            print(f"  {f.name}: {f.type}  = {default}")


def _default(model: str = "smolvla"):
    """Print a full default config as YAML."""
    if model == "smolvla":
        cfg = TrainConfig(model=SmolVLAModelConfig())
    elif model == "groot":
        cfg = TrainConfig(model=GR00TModelConfig())
    else:
        print(f"Unknown model: {model}. Available: smolvla, groot")
        return
    import sys
    yaml.dump(cfg.to_dict(), sys.stdout, default_flow_style=False, sort_keys=False)


def _show(path: str):
    """Load and print a config file."""
    cfg = TrainConfig.load(path)
    import sys
    yaml.dump(cfg.to_dict(), sys.stdout, default_flow_style=False, sort_keys=False)


def _create(model: str = "smolvla", output: str | None = None, **overrides):
    """Create a typed config with dotted overrides.

    Usage:
        python -m vbti.logic.train.config_utils create smolvla \\
            --dataset.repo_id=eternalmay33/lift_cube_3cams \\
            --training.lr=3e-5 \\
            --training.steps=20000 \\
            -o path/to/config.yaml

        # With dataset sources:
        python -m vbti.logic.train.config_utils create smolvla \\
            --dataset.repo_id=eternalmay33/lift_cube_3cams \\
            --dataset.source=sim

    Dotted keys: model.X, dataset.X, training.X, logging.X, eval.X
    For dataset sources, use dataset.repo_id / dataset.source / dataset.weight as
    shorthand for a single source.
    """
    # Start with defaults
    if model == "smolvla":
        base = TrainConfig(model=SmolVLAModelConfig()).to_dict()
    elif model == "groot":
        base = TrainConfig(model=GR00TModelConfig()).to_dict()
    else:
        print(f"Unknown model: {model}. Available: smolvla, groot")
        return

    # Apply dotted overrides
    sources_override = None
    source_shorthand = {}
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) == 2:
            section, field = parts
            # dataset.sources as JSON array of objects
            if section == "dataset" and field == "sources":
                if isinstance(value, str):
                    import json as _json
                    sources_override = _json.loads(value)
                elif isinstance(value, list):
                    sources_override = value
                continue
            # Single-source shorthand
            if section == "dataset" and field in ("repo_id", "source", "weight", "role", "root", "episodes"):
                source_shorthand[field] = value
                continue
            if section in base and field in base[section]:
                orig = base[section][field]
                if isinstance(orig, bool):
                    value = str(value).lower() in ("true", "1", "yes")
                elif isinstance(orig, int):
                    value = int(value)
                elif isinstance(orig, float):
                    value = float(value)
                base[section][field] = value
            else:
                print(f"Warning: unknown field {section}.{field}, skipping")
        else:
            print(f"Warning: key '{key}' is not dotted (section.field), skipping")

    # Apply sources — full array takes priority over shorthand
    if sources_override is not None:
        base["dataset"]["sources"] = sources_override
    elif source_shorthand:
        src = base["dataset"]["sources"][0]
        for k, v in source_shorthand.items():
            if k == "weight":
                v = float(v)
            src[k] = v

    # Validate by constructing typed config
    try:
        cfg = TrainConfig.from_dict(base)
    except (TypeError, ValueError) as e:
        print(f"Config validation failed: {e}")
        return

    if output:
        cfg.save(output)
        print(f"Config saved to {output}")
    else:
        import sys
        yaml.dump(cfg.to_dict(), sys.stdout, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "schema":  _schema,
        "default": _default,
        "show":    _show,
        "create":  _create,
    })
