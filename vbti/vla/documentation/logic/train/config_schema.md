# Training Config Schema

Configs are stored per experiment version:

```text
experiments/duck_cup_smolvla/vNNN/config.yaml
```

## Config CLI

```bash
python -m vbti.logic.train.config_utils schema
python -m vbti.logic.train.config_utils schema training
python -m vbti.logic.train.config_utils default smolvla
python -m vbti.logic.train.config_utils default groot
python -m vbti.logic.train.config_utils show experiments/duck_cup_smolvla/v020/config.yaml
```

Create with overrides:

```bash
python -m vbti.logic.train.config_utils create smolvla \
  --dataset.repo_id=eternalmay33/duck_cup_v020_all \
  --training.lr=1e-4 \
  --training.steps=50000 \
  --output=/tmp/config.yaml
```

## Top-Level Sections

```yaml
model: {}
dataset: {}
training: {}
logging: {}
eval: {}
output: {}
```

## Model: SmolVLA

Common fields:

```yaml
model:
  type: smolvla
  pretrained: lerobot/smolvla_base
  chunk_size: 50
  n_obs_steps: 1
  freeze_vision_encoder: true
  train_expert_only: true
  train_state_proj: true
  empty_cameras: 0
  tokenizer_max_length: 48
  num_denoising_steps: 10
  aux_weight: null
```

`freeze_vision_encoder: false` was important for v020. `aux_weight` is relevant for UVA variants.

## Model: GR00T

```yaml
model:
  type: groot
  pretrained: nvidia/GR00T-N1.6-3B
  embodiment: new_embodiment
  freeze_vlm: true
  unfreeze_top_vlm_layers: 4
  deepspeed_config: zero2
```

## Dataset

Common shape:

```yaml
dataset:
  sources:
    - repo_id: eternalmay33/duck_cup_v020_all
      root: null
      episodes: null
      weight: 1.0
      source: real
      role: both
  cameras:
    names: [top, left, right, gripper, gripper_depth]
    remap: null
  train_ratio: 0.95
  val_sources: null
  use_imagenet_stats: true
  image_transforms: null
```

Important fields:

- `repo_id`: dataset repo/path.
- `root`: explicit root if not default cache.
- `episodes`: optional episode list/filter.
- `cameras.names`: camera schema contract.
- `image_transforms`: LeRobot image augmentation config.

## Training

```yaml
training:
  steps: 10000
  epochs: null
  batch_size: 4
  lr: 1e-5
  weight_decay: 1e-10
  grad_clip_norm: 10.0
  warmup_steps: 500
  decay_lr: 2.5e-6
  lr_schedule: cosine
  decay_ratio: 0.3
  num_workers: 0
  device: auto
  seed: 42
  fp16: false
  bf16: true
  vision_lr_scale: 1.0
  gradient_checkpoint: true
```

If `epochs` is set, `engine._build_lerobot_command()` derives steps from frame count and batch size.

## Logging

```yaml
logging:
  log_freq: 100
  save_freq: 1000
  val_freq: 500
  val_size: 50
  wandb_enabled: false
  wandb_project: vbti-training
  wandb_entity: null
  wandb_mode: online
```

## Eval

```yaml
eval:
  n_episodes: 50
  sim_env: null
  record_video: true
```

Real robot evaluation is not driven only by this section. Use `logic/inference/eval_engine.py` protocols for physical validation.
