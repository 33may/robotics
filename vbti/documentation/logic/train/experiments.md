# Experiments And Version History

## Experiment Root

```text
experiments/duck_cup_smolvla/
```

Active state is tracked in:

```text
experiments/state.json
```

## Expected Version Layout

```text
experiments/duck_cup_smolvla/vNNN/
  config.yaml
  notes.md
  run.md
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
  eval_sessions/
    chkpt_step_.../
      session.json
      session.md
      videos/
```

## Version Meaning

| Version | Meaning |
|---|---|
| `v001` | Early sim-data baseline; poor real transfer. |
| `v006` | Real-data baseline; showed loss is not enough to judge policy quality. |
| `v013` | Fresh May-Sim baseline, no detection state, 6D state. |
| `v017` | RGB-only baseline for depth A/B. |
| `v018` | RGB plus `gripper_depth`; depth comparison. |
| `v019` | 671-episode combined corpus, 5 cameras, underfit/epoch analysis. |
| `v020` | Strong anchor: 765 episodes, 336,940 frames, 5 streams, unfrozen vision, augmentation. |
| `v021` | Data-efficiency: about 6.25% / stride 16. |
| `v022` | Data-efficiency: about 12.5% / stride 8. |
| `v023` | Data-efficiency: about 25% / stride 4. |
| `v024` | Data-efficiency: about 50% / stride 2. |
| `v025-v029` | SmolVLA-UVA auxiliary feature sweep. |

## Result Anchors

Known evaluation anchors:

| Version/checkpoint | Protocol | Result |
|---|---|---|
| `v017` step 20k | `dual_cup_30` | 11/30 |
| `v018` step 20k | `dual_cup_30` | 14/30 |
| `v019` step 20k | `dual_cup_30` | 20/30 |
| `v020` step 150k | `dual_cup_30` | 30/30 |
| `v020` step 150k | `dual_cup_60` | 56/60 |
| `v020` step 336940 | `checkpoint_sweep` | 19/20 |
| `v024` step 167940 | `checkpoint_sweep` | 14/20 |

## How To Compare Versions

Only compare quantitative results when:

- same protocol;
- same hardware/camera setup;
- same action horizon unless intentionally testing horizon;
- same success/failure marking rules;
- comparable checkpoint resolution.

Training loss alone is not a policy-quality conclusion.
