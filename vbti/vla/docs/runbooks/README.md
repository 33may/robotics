# Runbooks

Runbooks are task-oriented scenarios: what to check, what to run, what output to expect, and how to recover from common failures.

Use these when you want to operate the pipeline, not study the code.

## Index

| Scenario | Runbook |
|---|---|
| Preflight before any robot work | `preflight.md` |
| Real data collection with SO-101 and cameras | `data_collection.md` |
| Dataset inspection and preparation after recording | `dataset_preparation.md` |
| Create a new experiment version | `create_experiment_version.md` |
| Local model training / dry-run command generation | `local_training.md` |
| Remote training on the workstation | `remote_training.md` |
| Sequential remote training chains | `training_chains.md` |
| Pull checkpoints and prepare for inference/eval | `checkpoint_pull.md` |
| Run real-robot inference manually | `real_inference.md` |
| Run protocol evaluation | `evaluation.md` |
| Analyze evaluation sessions | `evaluation_analysis.md` |
| Common scenario shortcuts | `common_scenarios.md` |

## Rules

- Run preflight before data collection, inference, or evaluation.
- Create new artifacts by default; do not mutate datasets in place unless explicitly intended.
- Use fixed evaluation protocols for quantitative comparison.
- Do not treat training loss as the final robot-learning metric.
- Remote training uses `lerobot-train`; local backend edits do not affect remote runs.
