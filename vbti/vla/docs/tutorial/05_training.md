# 5. Training

**What you have:** a dataset in the LeRobot cache (block 4) and a GPU box described in `remote.yaml` (host, remote data and experiment dirs; auth by ssh key or `VLA_REMOTE_PASSWORD`). Every run is a version `experiments/<exp>/vNNN/` with a frozen `config.yaml`; the GPU box runs `lerobot-train` from its own env (patched LeRobot fork) inside tmux, and the version dir is the local source of truth. Commands without `--experiment/--version` fall back to the active version in `experiments/state.json`.

**What you need to do:** cut a new version from the closest existing config, change only what the run tests, check the generated `lerobot-train` command, launch it on the GPU box, pull the checkpoints back.

```bash
# once per experiment
python -m vla.logic.train.experiment_utils create <exp> --hypothesis="<hypothesis>" --goal="<goal>"
python -m vla.logic.train.config_utils create smolvla --dataset.repo_id=<user>/<dataset> --output=experiments/<exp>/base_config.yaml

# new version: copies the config of <vNNN|last|base> into the next vNNN and makes it active
python -m vla.logic.train.experiment_utils use <exp>
python -m vla.logic.train.experiment_utils version --config=<vNNN|last|base> --notes="<what changes and why>"
# edit experiments/<exp>/<vNEW>/config.yaml: dataset.sources, dataset.cameras.names, training.*, logging.save_freq
python -m vla.logic.train.config_utils show experiments/<exp>/<vNEW>/config.yaml
python -m vla.logic.train.engine train-lerobot --experiment=<exp> --version=<vNEW> --dry_run

# launch: syncs the dataset if incomplete, writes remote_session.json, tails the log (Ctrl+C detaches, training keeps running)
python -m vla.logic.train.remote train --experiment=<exp> --version=<vNEW> --run_name=lerobot_output_r1

# monitor: W&B run <exp>_<vNEW>_<run_name>, or the log on the box (host and dirs in remote_session.json)
python -m vla.logic.train.remote status --experiment=<exp> --version=<vNEW>
ssh <host> tail -f <remote_version_dir>/train_<run_name>.log

# pull: whole run, or one checkpoint's weights only
python -m vla.logic.train.remote pull --experiment=<exp> --version=<vNEW>
python -m vla.logic.train.remote pull --experiment=<exp> --version=<vNEW> --run_name=<run_name> --checkpoint=step_<NNNNNN> --pretrained_only=true

# continue a run from a checkpoint (optimizer + scheduler state restored) into a new run dir
python -m vla.logic.train.remote train --experiment=<exp> --version=<vNEW> --run_name=lerobot_output_r2 --resume_from=lerobot_output_r1/checkpoints/<NNNNNN>

# several versions back-to-back (configs already written, dry-run each first)
python -m vla.logic.train.experiment_utils use <exp>
nohup python -m vla.logic.train.chain --versions <vA>,<vB>,<vC> --run_name=lerobot_output_r1 > chain.log 2>&1 &
python -m vla.logic.train.chain_remote --versions <vD>,<vE> --run_name=lerobot_output_r1 --wait_for_session <tmux_session>
```

**Done when:**
- `experiments/<exp>/<vNEW>/` holds `config.yaml`, `notes.md` (hypothesis filled in), `remote_session.json`
- the dry run shows the intended dataset, episodes, steps, batch size, camera rename map and `--use_policy_training_preset=false`
- `remote status` reports `RUNNING` and the W&B loss is moving
- `experiments/<exp>/<vNEW>/<run_name>/checkpoints/<NNNNNN>/pretrained_model/` contains `model.safetensors`, `config.json`, `policy_preprocessor.json`, `policy_postprocessor.json`
- `python -m vla.logic.train.experiment_utils ckpts --experiment=<exp> --version=<vNEW>` lists the pulled checkpoints

## Training commands

| Command | What it does |
|---|---|
| `python -m vla.logic.train.experiment_utils create <exp> [--hypothesis --goal]` | New experiment dir: `experiment.md`, empty `base_config.yaml`, `compare.md` |
| `python -m vla.logic.train.experiment_utils use <exp> [<vNNN>]` | Set the active experiment/version in `experiments/state.json` |
| `python -m vla.logic.train.experiment_utils version --config=<vNNN\|last\|base\|path.yaml> [--notes]` | Next `vNNN` with frozen `config.yaml`, `notes.md`, `run.md`; becomes active |
| `python -m vla.logic.train.experiment_utils active \| lsv \| lse \| status \| ckpts` | Active version, versions, experiments, version status, checkpoints with paths |
| `python -m vla.logic.train.config_utils create smolvla --<section>.<key>=<value> --output=<yaml>` | Typed config from defaults + dotted overrides (`model.`, `dataset.`, `training.`, `logging.`) |
| `python -m vla.logic.train.config_utils schema \| default \| show <yaml>` | Field reference, full default config, validated print of a config |
| `python -m vla.logic.train.engine train-lerobot [--experiment --version] --dry_run` | Print the `lerobot-train` command built from the config: `epochs` → `steps`, camera → `camera1..3`/`empty_camera_N` rename map, optimizer (`smolvla-adamw` when `vision_lr_scale` < 1), cosine schedule |
| `python -m vla.logic.train.remote push-data --repo_id=<user>/<dataset>` | Rsync a cached dataset to `data_dir` on the GPU box |
| `python -m vla.logic.train.remote train --run_name=<run> [--resume_from=<run>/checkpoints/<N>] [--stream=false] [--dry_run]` | Write `train_<run>.sh` on the box, launch it in tmux `train_<exp>_<vNNN>_<run>`, log to `<remote_version_dir>/train_<run>.log`, save `remote_session.json` |
| `python -m vla.logic.train.remote status` | Whether the run's tmux session is alive |
| `python -m vla.logic.train.remote pull [--run_name] [--checkpoint=step_<N>] [--pretrained_only=true]` | Rsync the run (or one checkpoint) into `<vNNN>/<run_name>/` |
| `python -m vla.logic.train.chain --versions <a,b,…> [--run_name] [--poll_interval]` | Local supervisor: launches each version, polls tmux, pulls checkpoints, halts if the final checkpoint is missing or under 800 MB; status in `chain_status.json` |
| `python -m vla.logic.train.chain_remote --versions <a,b,…> [--run_name] [--wait_for_session <tmux>]` | One script on the box runs the versions in tmux `chain_<exp>_<run>`; continues past a failed version; waits up to 4 h for another session; outputs stay remote until `pull` |
| `python -m vla.logic.train.engine train [--experiment --version] [--resume [--reset_lr]]` | Local training loop (SmolVLA/GR00T backends) into the version dir; smoke tests only |
| `python -m vla.logic.train.monitor snapshot \| logs \| trend \| compare \| history \| metrics` | Read local-engine `metrics/training_log.jsonl`; remote runs log to W&B |

Full reference: `docs/modules/train/`.
