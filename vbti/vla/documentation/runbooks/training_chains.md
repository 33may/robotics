# Runbook: Sequential Training Chains

Use when launching multiple experiment versions in sequence.

## Option A: Local Polling Chain

Use when the local machine should supervise each run and pull outputs after each version.

```bash
python -m vbti.logic.train.chain \
  --versions v021,v022,v023,v024 \
  --run_name=lerobot_output_r1 \
  --poll_interval=300
```

Per version, the chain:

1. launches remote training;
2. polls tmux;
3. pulls checkpoints;
4. validates expected final checkpoint;
5. stops on failure.

## Option B: Fully Remote Chain

Use when the remote should continue independently.

```bash
python -m vbti.logic.train.chain_remote \
  --versions v025,v026,v027,v028,v029 \
  --run_name=lerobot_output_r4_aux0003 \
  --wait_for_session uva_bake
```

The remote chain:

1. writes one remote bash script;
2. optionally waits for another tmux session;
3. runs each version sequentially;
4. leaves outputs on remote;
5. writes local receipts.

## Before Launch

For every version:

```bash
python -m vbti.logic.train.config_utils show experiments/duck_cup_smolvla/vNNN/config.yaml
python -m vbti.logic.train.engine train-lerobot --experiment=duck_cup_smolvla --version=vNNN --dry_run
```

Check:

- dataset exists locally or remotely;
- steps/epochs are intended;
- save frequency matches desired checkpoint sweep;
- run name is consistent;
- W&B naming is meaningful.

## Common Chain Scenarios

Data-efficiency sweep:

```bash
python -m vbti.logic.train.chain --versions v021,v022,v023,v024 --run_name=lerobot_output_r1
```

UVA sweep after feature bake:

```bash
python -m vbti.logic.train.chain_remote --versions v025,v026,v027,v028,v029 --run_name=lerobot_output_r4_aux0003 --wait_for_session uva_bake
```

## Failure Handling

If chain stops:

1. inspect the failed version's `remote_session.json`;
2. inspect remote tmux/logs;
3. check dataset availability;
4. pull partial checkpoints if useful;
5. decide whether to resume same version or create a new run name.

Do not blindly restart the whole chain if the failure reveals a config/data bug.
