---
name: train-copilot
description: Training copilot for VLA policies in the vla framework (SmolVLA, GR00T). Use when creating experiments and versions, editing configs, dry-running and launching remote lerobot-train runs, pulling checkpoints, running training chains, monitoring, or analysing results across versions. Triggers on train, experiment, version, SmolVLA, GR00T, config, checkpoint, remote, chain, W&B, loss.
---

# Train Copilot

Plan, configure, launch and analyse training runs. The user decides what to run; you prepare configs, show diffs, launch on approval and report. Paths are relative to `vbti/vla`; commands run there in the `vla` conda env.

## Session start

1. `python -m vla.logic.train.experiment_utils active`
2. `python -m vla.logic.train.experiment_utils status`
3. If `$ARGUMENTS` names a topic, go to it. Otherwise ask what is being trained.

## References

- [cli_reference.md](cli_reference.md): commands, config fields, experiment layout.
- [model_knowledge.md](model_knowledge.md): SmolVLA and GR00T internals.
- `docs/tutorial/05_training.md`, `docs/modules/train/`, `docs/runbooks/{remote_training,training_chains,checkpoint_pull}.md`.
- `docs/case-study/README.md`: what worked and what did not on duck/cup (v001-v029).
- `experiments/<exp>/vNNN/notes.md`: reasoning and results per version.

## Workflows

### New version

1. Agree on the hypothesis: one change the run tests.
2. Check the dataset: `python -m vla.logic.dataset.check_utils cameras <repo_id>` and `info`.
3. Cut the version from the closest config:
   ```bash
   python -m vla.logic.train.experiment_utils use <exp>
   python -m vla.logic.train.experiment_utils version --config=<vNNN|last|base> --notes="<change and why>"
   ```
4. Edit `experiments/<exp>/<vNEW>/config.yaml`. Show the diff against the source version.
5. `python -m vla.logic.train.engine train-lerobot --experiment=<exp> --version=<vNEW> --dry_run`; check dataset, episodes, steps, batch size, camera rename map.
6. Launch only after the user approves:
   ```bash
   python -m vla.logic.train.remote train --experiment=<exp> --version=<vNEW> --run_name=lerobot_output_r1
   ```

### Monitor a remote run

- `python -m vla.logic.train.remote status --experiment=<exp> --version=<vNNN>` (tmux alive or not).
- Loss: W&B run `<exp>_<vNNN>_<run_name>`, or `ssh <host> tail -f <remote_version_dir>/train_<run_name>.log` (host and dirs in `remote_session.json`). `remote logs` reads the wrong path (`KNOWN_ISSUES.md`).
- `monitor` only reads local-engine `metrics/training_log.jsonl`.

### Pull and evaluate

```bash
python -m vla.logic.train.remote pull --experiment=<exp> --version=<vNNN> --run_name=<run> --checkpoint=step_<N> --pretrained_only=true
python -m vla.logic.train.experiment_utils ckpts --experiment=<exp> --version=<vNNN>
```

Quality is decided by a real-robot protocol session (`docs/tutorial/06_evaluation.md`), not by loss.

### Analyse

Read `notes.md`, `config.yaml` and `eval_sessions/*/session.md` of the versions involved. Compare only sessions of the same protocol, checkpoint step and inference settings. Report numerator/denominator. Suggest the next run; the user decides.

## Rules

- Never launch training without the user seeing the config diff and the dry-run command.
- One change per version where possible; write the hypothesis into `notes.md`.
- Inspect the dataset distribution before touching hyperparameters. Rest bias, shortcuts and spatial coverage moved results more than hyperparameter changes did.
- Loss is not the metric. Better-loss checkpoints did worse on the robot (v019 at 30k). Evaluate several checkpoints with a protocol.
- No engineered shortcut inputs in the state (v014 detector coordinates became a crutch).
- Fix observed failure cells with targeted data (v019 to v020).
- Camera names and order in `dataset.cameras.names` must match the dataset and the inference preset.
- Local backend edits do not affect remote `lerobot-train`. `smolvla_uva` (v025-v029), per-group vision LR (`vision_lr_scale` < 1, v020) and the depth-transform patch need the LeRobot fork `github.com/33may/lerobot` branch `vbti/main` on the GPU box.
- Do not print the remote password; auth is an ssh key or `VLA_REMOTE_PASSWORD`.
- Use the CLI for anything it covers; no ad-hoc inline Python for those operations.
