# Training Module

## Scope

Training code lives in `logic/train/` and experiment state lives under `experiments/duck_cup_smolvla/`.

Main paths:

- `logic/train/config_utils.py`
- `logic/train/engine.py`
- `logic/train/remote.py`
- `logic/train/chain.py`
- `logic/train/chain_remote.py`
- `logic/train/experiment_utils.py`
- `logic/train/backends/`
- `experiments/duck_cup_smolvla/`
- `remote.yaml`

## Architecture

There are two training paths:

| Path | Use | Notes |
|---|---|---|
| Local/custom engine | Local experiments and backend development | Runs `logic/train/engine.py` and backend classes. |
| Remote LeRobot path | Main SmolVLA training path | Builds a `lerobot-train` command and launches it remotely in `tmux`. |

Important: `python -m vbti.logic.train.remote train` does not run `SmolVLABackend`. It builds a `lerobot-train` command from the VBTI config. If a change must affect remote training, bake it into the dataset or patch the LeRobot fork.

## Config Schema

Configs are YAML files, usually stored as:

```text
experiments/duck_cup_smolvla/vNNN/config.yaml
```

Top-level sections:

- `model`
- `dataset`
- `training`
- `logging`
- `eval`
- `output`

SmolVLA defaults include:

```yaml
model:
  type: smolvla
  pretrained: lerobot/smolvla_base
  chunk_size: 50
  n_obs_steps: 1
  freeze_vision_encoder: true
  train_expert_only: true
  train_state_proj: true
  tokenizer_max_length: 48
  num_denoising_steps: 10
```

Training defaults include:

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
  num_workers: 0
  device: auto
  seed: 42
  bf16: true
  gradient_checkpoint: true
```

If `training.epochs` is set, `engine._build_lerobot_command()` derives steps from dataset frame count and batch size.

## Config Commands

```bash
python -m vbti.logic.train.config_utils schema
python -m vbti.logic.train.config_utils schema training
python -m vbti.logic.train.config_utils default smolvla
python -m vbti.logic.train.config_utils default groot
python -m vbti.logic.train.config_utils show /home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v020/config.yaml
```

Create config with overrides:

```bash
python -m vbti.logic.train.config_utils create smolvla \
  --dataset.repo_id=eternalmay33/duck_cup_v020_all \
  --training.lr=1e-4 \
  --training.steps=50000 \
  --output=/tmp/config.yaml
```

## Local Engine Commands

```bash
python -m vbti.logic.train.engine status
python -m vbti.logic.train.engine train --config=/path/to/config.yaml --experiment=duck_cup_smolvla
python -m vbti.logic.train.engine train --experiment=duck_cup_smolvla --version=v020
python -m vbti.logic.train.engine train --resume
python -m vbti.logic.train.engine train --resume --reset_lr
```

Generate/print LeRobot CLI:

```bash
python -m vbti.logic.train.engine train-lerobot --experiment=duck_cup_smolvla --version=v020 --dry_run
python -m vbti.logic.train.engine train-lerobot
```

## Remote Training Commands

Push dataset:

```bash
python -m vbti.logic.train.remote push-data --repo_id=eternalmay33/duck_cup_v020_all
```

Launch training:

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --run_name=lerobot_output_r1
```

Resume from checkpoint:

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --run_name=lerobot_output_r2 \
  --resume_from=lerobot_output_r1/checkpoints/030000
```

Dry run:

```bash
python -m vbti.logic.train.remote train \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --run_name=lerobot_output_r1 \
  --dry_run
```

Monitor/pull:

```bash
python -m vbti.logic.train.remote status --experiment=duck_cup_smolvla --version=v020
python -m vbti.logic.train.remote logs --experiment=duck_cup_smolvla --version=v020 --lines=100
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=v020
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=v020 --checkpoint=step_080000
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=v020 --checkpoint=step_080000 --pretrained_only=true
```

Remote `train()` defaults:

- `run_name="lerobot_output"`
- `expandable_segments=True`
- `stream=True`
- `dry_run=False`

## Sequential Chains

Local polling chain:

```bash
python -m vbti.logic.train.chain --versions v021,v022,v023,v024
python -m vbti.logic.train.chain --versions v021,v022,v023,v024 --run_name=lerobot_output_r1 --poll_interval=300
```

Remote chain script:

```bash
python -m vbti.logic.train.chain_remote --versions v021,v022,v023
python -m vbti.logic.train.chain_remote --versions v025,v026,v027,v028,v029 --run_name=lerobot_output_r4_aux0003 --wait_for_session uva_bake
```

The remote chain creates one remote script and launches one `tmux` chain. Use this for long sequential sweeps when mid-run intervention is unlikely.

## LeRobot CLI Mapping

`engine._build_lerobot_command()` maps config to flags like:

- `--policy.path=<model.pretrained>`
- `--dataset.repo_id=<repo_id>`
- `--dataset.root=<root>`
- `--dataset.episodes=<json>`
- `--batch_size`
- `--steps`
- `--num_workers`
- `--use_policy_training_preset=false`
- scheduler/optimizer settings
- SmolVLA chunk/obs/vision/state/tokenizer settings
- camera `rename_map`
- W&B settings

`--use_policy_training_preset=false` is critical; otherwise LeRobot presets can override optimizer/scheduler CLI flags.

## Outputs

Experiment layout:

```text
experiments/{experiment}/
  experiment.md
  base_config.yaml
  compare.md
  vNNN/
    config.yaml
    run.md
    notes.md
    remote_session.json
    lerobot_output_rX/
      checkpoints/
        005000/
          pretrained_model/
            config.json
            model.safetensors
            preprocessor.json
            postprocessor.json
            train_config.json
          training_state/
```

Remote receipts include run name, tmux session, remote paths, repo ID, and host.

## Duck-Cup Version History

| Version | Meaning |
|---|---|
| `v001` | Early simulation-trained model. |
| `v006` | Real-data baseline; showed real behavior matters more than loss. |
| `v013` | Fresh May-Sim baseline, no detection, 6D state. |
| `v017` | RGB-only comparison baseline. |
| `v018` | Added `gripper_depth`; evaluated against v017. |
| `v019` | 671-episode combined corpus, 5 cams, underfit analysis. |
| `v020` | Strong anchor: 765 episodes, 336,940 frames, 5 cams, unfrozen vision, augmentation. |
| `v021-v024` | Data-efficiency sweep: 6.25/12.5/25/50% slices. |
| `v025-v029` | SmolVLA-UVA aux video feature sweep. |

Training strategy memory:

- SmolVLA duck-cup tends to peak around 20-25k steps on some schedules, but v020 long training also produced strong later checkpoints.
- Size schedules by epoch count when dataset size changes.
- Real-robot protocol evaluation is the deciding metric.

## Remote Assumptions

`remote.yaml` stores host/path settings and a plaintext password. Do not copy the password into docs or commits.

Remote scripts assume:

- `sshpass`, `ssh`, `rsync`, `tmux`.
- remote env activation at `/home/vbti/anton/env/bin/activate`.
- remote data under `/home/vbti/anton/data`.
- remote experiments under `/home/vbti/anton/experiments`.
- patched editable LeRobot fork at `/home/vbti/anton/lerobot`.

## Pitfalls

- Remote path bypasses local backend classes.
- `remote.py status/logs` may look at a different log path than the training launch writes for newer runs.
- SSH commands disable strict host key checking.
- `run.md` status can be stale; use notes, receipts, checkpoints, W&B, and eval sessions.
- LeRobot fork patches are required for some project-specific features: optimizer, vision unfreeze, transforms, depth/image handling, UVA policy registration.
- Do not compare training loss as final quality; evaluate on real protocols.
