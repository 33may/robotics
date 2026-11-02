# Training CLI reference

All commands: `python -m vla.logic.train.<module> <command>`. Without `--experiment/--version`, commands use the active version from `experiments/state.json` (tracked; `use` rewrites it).

## experiment_utils

| Command | Args | Does |
|---|---|---|
| `create` | `NAME [--hypothesis] [--goal]` | New experiment dir: `experiment.md`, `base_config.yaml`, `compare.md` |
| `use` | `EXPERIMENT [VERSION]` | Set active experiment/version |
| `active` | | Active experiment/version |
| `lse` / `lsv [--experiment]` | | List experiments / versions |
| `status` | `[--experiment]` | All versions with run status |
| `version` | `--config=<vNNN\|last\|base\|path.yaml> [--notes] [--experiment]` | Next `vNNN` with frozen `config.yaml`, `notes.md`, `run.md`; becomes active |
| `config` / `dir` | `[--experiment] [--version]` | Frozen config / version dir path |
| `ckpts` | `[--experiment] [--version]` | Pulled checkpoints with paths |
| `complete` | `[--status] [--duration_str]` | Mark a version finished in `run.md` |

## config_utils

| Command | Args | Does |
|---|---|---|
| `schema` | `[--section=all\|model\|smolvla\|groot\|dataset\|training\|logging\|eval\|source]` | Fields and defaults |
| `default` | `[--model=smolvla\|groot]` | Full default config YAML |
| `create` | `[MODEL] [--output=O] [--<section>.<key>=<value> ...]` | Typed config from defaults + dotted overrides; prints if no `--output` |
| `show` | `PATH` | Validated print of a config |

Overrides: `--training.lr=3e-5 --model.chunk_size=50`. Single source shorthand: `--dataset.repo_id=<repo> --dataset.source=real`.

## engine

| Command | Args | Does |
|---|---|---|
| `train-lerobot` | `[--config] [--experiment] [--version] [--notes] [--dry_run]` | Build (and run) the `lerobot-train` command: `epochs` to `steps`, camera rename map to `camera1..3` / `empty_camera_N`, `smolvla-adamw` when `vision_lr_scale` < 1, cosine schedule |
| `train` | `[--config] [--experiment] [--version] [--resume] [--reset_lr]` | Local training loop (SmolVLA/GR00T backends); smoke tests |
| `status` | | Active experiment/version |

## remote (GPU box from `remote.yaml`)

| Command | Args | Does |
|---|---|---|
| `push-data` | `REPO_ID [--remote_cfg]` | Rsync a cached dataset to the box |
| `train` | `--run_name [--resume_from=<run>/checkpoints/<N>] [--stream] [--dry_run] [--experiment] [--version]` | Write `train_<run>.sh`, launch tmux `train_<exp>_<vNNN>_<run>`, log to `<remote_version_dir>/train_<run>.log`, save `remote_session.json` |
| `status` | `[--experiment] [--version]` | tmux session alive or not |
| `logs` | `[--lines]` | Reads `<remote_run_dir>/train.log`, which the launcher does not write. Use W&B or `ssh tail` |
| `pull` | `[--run_name] [--checkpoint=step_<N>\|all] [--pretrained_only]` | Rsync run or one checkpoint into `<vNNN>/<run_name>/` |

## chain / chain_remote

| Command | Args | Does |
|---|---|---|
| `chain` | `--versions a,b,c [--run_name] [--poll_interval]` | Local supervisor: launch each, poll tmux, pull, stop if the final checkpoint is missing or < 800 MB. Experiment hardcoded to `duck_cup_smolvla` |
| `chain_remote` | `--versions a,b [--run_name] [--wait_for_session <tmux>]` | One script on the box runs all versions; continues past failures; outputs stay remote until `remote pull --experiment=<exp>` |

## monitor (local engine only)

`snapshot`, `logs [--last] [--metric]`, `trend [--metric] [--window] [--threshold]`, `compare V1 V2 [--metric]`, `history [--metric] [--points]`, `metrics`. All take `--experiment --version`. They read `metrics/training_log.jsonl`; remote runs log to W&B.

## Experiment layout

```
experiments/<exp>/
├── experiment.md  base_config.yaml  compare.md
└── vNNN/
    ├── config.yaml            # frozen
    ├── notes.md  run.md
    ├── remote_session.json    # remote launch receipt
    ├── <run_name>/checkpoints/<NNNNNN>/pretrained_model/
    │       config.json  model.safetensors  policy_{pre,post}processor.json
    └── eval_sessions/chkpt_<step>_ah_<h>_pr_<protocol>_<ts>/session.{md,json}, videos/
```

## Config (SmolVLA defaults, `config_utils default`)

```yaml
model:
  type: smolvla            # smolvla | smolvla_uva | groot
  pretrained: lerobot/smolvla_base
  chunk_size: 50
  freeze_vision_encoder: true
  train_expert_only: true
  empty_cameras: 0
  aux_weight: null         # smolvla_uva auxiliary loss weight
dataset:
  sources: [{repo_id: "", root: null, episodes: null, weight: 1.0, source: sim, role: both}]
  cameras: {names: [front], remap: null}   # order matters
  train_ratio: 0.95
  image_transforms: null
training:
  steps: 10000             # or epochs
  batch_size: 4
  lr: 1.0e-05
  decay_lr: 2.5e-06
  lr_schedule: cosine
  warmup_steps: 500
  vision_lr_scale: 1.0     # < 1 with an unfrozen vision encoder (v020: 0.1)
  gradient_checkpoint: true
  bf16: true
logging: {log_freq: 100, save_freq: 1000, val_freq: 500, wandb_enabled: false}
```

v020 reference (the anchor model): batch 64, peak LR 1e-4, vision LR 1e-5, unfrozen SigLIP with gradient checkpointing, brightness/contrast/sharpness/affine augmentation, saturation/hue off. Full config: `experiments/duck_cup_smolvla/v020/config.yaml`.
