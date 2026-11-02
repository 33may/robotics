# Sequential Training Chains

## Local Polling Chain

File: `logic/train/chain.py`.

Command:

```bash
python -m vbti.logic.train.chain --versions v021,v022,v023,v024
python -m vbti.logic.train.chain --versions v021,v022,v023,v024 --run_name=lerobot_output_r1 --poll_interval=300
```

Workflow per version:

1. Read config.
2. Derive expected final step.
3. Launch remote train with `stream=false`.
4. Poll tmux/session status.
5. Pull checkpoints.
6. Validate expected checkpoint exists and model file size is sane.
7. Stop on first failure.

Use when local machine should supervise and pull after each run.

## Fully Remote Chain

File: `logic/train/chain_remote.py`.

Command:

```bash
python -m vbti.logic.train.chain_remote --versions v021,v022,v023
python -m vbti.logic.train.chain_remote --versions v025,v026,v027,v028,v029 --run_name=lerobot_output_r4_aux0003 --wait_for_session uva_bake
```

Workflow:

1. Generate one remote bash script.
2. Optionally wait for a remote tmux session to finish.
3. Run versions sequentially on remote.
4. Write local receipts.
5. Leave outputs on remote for later pull.

Use for long sweeps where the remote should keep going without local polling.

## Parameters

Common:

| Arg | Meaning |
|---|---|
| `--versions` | Comma-separated version list. |
| `--run_name` | Output run folder name. |
| `--wait_for_session` | Remote tmux session to wait for before starting. |

Local chain also supports:

| Arg | Default | Meaning |
|---|---|---|
| `--poll_interval` | `300` | Seconds between status polls. |

## Sweep Examples

Data-efficiency sweep:

```bash
python -m vbti.logic.train.chain --versions v021,v022,v023,v024 --run_name=lerobot_output_r1
```

UVA sweep after feature bake:

```bash
python -m vbti.logic.train.chain_remote \
  --versions v025,v026,v027,v028,v029 \
  --run_name=lerobot_output_r4_aux0003 \
  --wait_for_session uva_bake
```

## Failure Criteria

Treat chain run as failed if:

- tmux session exits unexpectedly;
- expected checkpoint is missing;
- `model.safetensors` is much too small;
- remote dataset path is missing;
- logs show LeRobot config/schema error;
- wrong run name/version was launched.
