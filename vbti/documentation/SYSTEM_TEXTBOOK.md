# Robot Learning Pipeline System Textbook

## Purpose

This repository implements a modular robot-learning pipeline around a physical SO-101 setup, LeRobot datasets, SmolVLA training, real-robot inference/evaluation, and a prepared reconstruction/simulation extension path.

The validated loop is the real-world flow:

```text
hardware check -> teleoperation/data collection -> LeRobot dataset inspection/curation -> training -> inference -> protocol evaluation -> analysis -> next dataset/model decision
```

The prepared extension flow is:

```text
video/photos -> COLMAP/MILo reconstruction -> USD/Isaac scene -> simulated/RL demonstrations -> HDF5 -> LeRobot -> same training/evaluation path
```

## Evidence Baseline

Primary baseline document:

`/home/may33/Documents/vbti/vbti/oficial/docs/portfolio/Robot Learning Pipeline Design and Technical Advice.md`

That document establishes the current architecture and design decisions: real-world baseline first, protocol-based real-robot evaluation, versioned experiments, dataset quality as a bottleneck, simulation as a prepared extension path, depth as an image stream, and vision unfreezing for v020.

Additional evidence used here:

- Code under `logic/`.
- Existing docs under `docs/`.
- Project memory under `.august/memory/project/`.
- Experiment folders under `experiments/duck_cup_smolvla/`.
- Shell/session command evidence summarized in `evidence/terminal_and_session_evidence.md`.

## Current Module Map

| Module | Main paths | Responsibility |
|---|---|---|
| Hardware | `logic/cameras`, `logic/servos` | RealSense/OpenCV camera presets, SO-101 servo scan/calibration/recovery/rest. |
| Dataset | `logic/dataset`, `logic/depth`, `logic/detection` | LeRobot/HDF5 conversion, inspection, curation, depth transforms, detection labels, UVA feature baking. |
| Training | `logic/train`, `experiments/duck_cup_smolvla` | Versioned configs, local engine, remote `lerobot-train`, sequential chains, experiment receipts. |
| Evaluation | `logic/inference`, `logic/inference/protocols` | Real policy deployment, protocol trials, session JSON/Markdown, heatmaps and breakdowns. |
| Reconstruction | `logic/reconstruct`, `scripts/3d`, `scripts/sim` | Video-to-mesh reconstruction, USD conversion, Isaac/LeIsaac task export, Cosmos prep. |

## Operating Principle

The pipeline is not one script. It is a set of contracts:

- Hardware contract: stable robot calibration, stable camera names, stable physical scene.
- Dataset contract: LeRobot feature schema with fixed `action`, `observation.state`, `observation.images.*`, and task text.
- Training contract: frozen experiment `config.yaml` plus remote/local outputs tied to a version folder.
- Evaluation contract: every quantitative result comes from a protocol and a saved session.
- Simulation contract: generated data must be converted back into the same LeRobot schema before training.

## Environments

Observed project environments:

| Env | Purpose | Notes |
|---|---|---|
| `lerobot` | Real datasets, training tooling, inference/eval utilities | Python 3.12, LeRobot editable fork, PyTorch 2.7/cu128, torchcodec 0.5. |
| `isaac` | Isaac Sim/Lab and simulation-side tooling | Python 3.11, Isaac/LeIsaac stack. |
| `gsplat-pt25` | COLMAP/MILo/reconstruction | Python 3.11, PyTorch 2.5.1/cu124, CUDA 12.9, GCC 14. |
| `groot` | GR00T experiments | Python 3.10, GR00T/flash-attn stack. |

Package import note: the `vbti` package is expected to be importable from `/home/may33/projects/ml_portfolio/robotics`, often via a `.pth` in the active env.

## End-To-End Real Workflow

### 1. Hardware Preflight

Run servo scan before blaming a model:

```bash
python -m vbti.logic.servos.scan_all
```

Use voltage to identify arms:

- leader arm: about 5V;
- follower arm: about 12V.

Check camera layout:

```bash
python -m vbti.logic.cameras.view_cameras --preset realsense
python -m vbti.logic.cameras.view_cameras --preset opencv
python -m vbti.logic.cameras.check_usb
```

Load known calibration when needed:

```bash
python -m vbti.logic.servos.profiles list
python -m vbti.logic.servos.profiles load frodeo-test --port /dev/ttyACM1
```

Move follower to rest before/after risky runs:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/logic/servos/rest.py --port=/dev/ttyACM1 --speed=3.0
```

### 2. Data Collection

Real demonstrations are normally recorded with `lerobot-record`, using SO-101 follower/leader and four cameras. Older commands used numeric OpenCV indices; current docs prefer stable `/dev/cam_*` symlinks or the shared camera preset code.

Known recording caveat: LeRobot keyboard controls are more reliable under X11 than Wayland.

### 3. Dataset Inspection And Curation

List/inspect datasets:

```bash
python -m vbti.logic.dataset.check_utils ls
python -m vbti.logic.dataset.check_utils report eternalmay33/duck_cup_v020_all
python -m vbti.logic.dataset.check_utils cameras eternalmay33/duck_cup_v020_all
```

Inspect HDF5 simulation data:

```bash
python -m vbti.logic.dataset.hdf5_utils report /path/to/data.hdf5
python -m vbti.logic.dataset.convert_utils discover /path/to/data.hdf5
```

Convert HDF5 to LeRobot:

```bash
python -m vbti.logic.dataset.convert_utils convert /path/to/data.hdf5 eternalmay33/my_dataset "Pick up the duck and place it in the cup"
```

Depth and feature transforms should normally produce a new dataset artifact. Avoid destructive in-place transformations unless explicitly intended.

### 4. Training

Inspect active experiment:

```bash
python -m vbti.logic.train.engine status
python -m vbti.logic.train.config_utils show /home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v020/config.yaml
```

Remote training is the main path for larger SmolVLA runs:

```bash
python -m vbti.logic.train.remote train --experiment=duck_cup_smolvla --version=v020 --run_name=lerobot_output_r1
python -m vbti.logic.train.remote status --experiment=duck_cup_smolvla --version=v020
python -m vbti.logic.train.remote pull --experiment=duck_cup_smolvla --version=v020 --checkpoint=all
```

Important: remote training launches `lerobot-train`. It does not execute local `SmolVLABackend` code. If a transform must affect remote training, bake it into the dataset or patch the LeRobot fork.

### 5. Real Inference

Preview cameras:

```bash
python -m vbti.logic.inference.run_real_inference preview
```

Run policy interactively:

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/path/to/pretrained_model \
  --port=/dev/ttyACM1 \
  --cameras=realsense \
  --task="Pick up the duck and place it in the red cup" \
  --action_horizon=10 \
  --fps=30
```

For depth models, enable depth and ensure the gripper camera is RealSense-backed:

```bash
python -m vbti.logic.inference.run_real_inference run \
  --checkpoint=/path/to/pretrained_model \
  --depth=true \
  --cameras=opencv_depth
```

### 6. Protocol Evaluation

Run structured evaluation:

```bash
python -m vbti.logic.inference.eval_engine \
  --checkpoint=v020-150k \
  --protocol=dual_cup_60 \
  --experiment=duck_cup_smolvla \
  --version=v020 \
  --record=True
```

Operator flow:

- Place objects according to overlay.
- Press `SPACE` to start trial.
- Press `S` for success or `F` for failure.
- Press `V` to save video when applicable.
- Press `Q`/Esc to quit.

Analyze sessions:

```bash
python -m vbti.logic.inference.eval_helpers ls --detailed
python -m vbti.logic.inference.eval_helpers info latest
python -m vbti.logic.inference.eval_helpers play latest 3
python -m vbti.logic.inference.eval_render heatmap /path/to/session --output=/tmp/heatmap.png
```

## Experiment History Anchor

The duck-cup SmolVLA line is the main validated experiment family.

| Version | Meaning |
|---|---|
| `v001` | Early sim-trained baseline; poor real transfer. |
| `v006` | Real-data baseline that showed loss is not enough to judge policy quality. |
| `v013` | Fresh May-Sim baseline, no detection state, 6D state. |
| `v017` | RGB-only A/B baseline for depth comparison. |
| `v018` | RGB plus gripper-depth stream; measured improvement over v017 in one protocol. |
| `v019` | Large combined corpus; underfit/step-count analysis showed epochs matter. |
| `v020` | Strong anchor: 765 episodes, 5 image streams, unfrozen vision, targeted right-side data and augmentation. |
| `v021-v024` | Data-efficiency sweep over v020-style data. |
| `v025-v029` | SmolVLA-UVA auxiliary future-feature sweep. |

Key result anchor: v020 step 150k reached `30/30` on `dual_cup_30` and `56/60` on `dual_cup_60`; later checkpoint-sweep evidence includes v020 step 336940 at `19/20` on `checkpoint_sweep`.

## Design Decisions That Matter Operationally

- Build the real-world baseline before claiming sim-to-real improvement.
- Treat real-robot success rate under a fixed protocol as the main validation metric.
- Use versioned experiment folders as source of truth.
- Inspect datasets before training; rest-frame bias and schema drift can dominate model behavior.
- Do not mix simulation data naively before the real distribution is stable.
- Avoid detector coordinates as the main policy interface unless the experiment explicitly tests that shortcut.
- Add depth as `observation.images.gripper_depth`, not as a custom model branch.
- Unfreeze the vision encoder when visual representation is the suspected bottleneck.

## Module Docs

For command-level usage, use the folder tree that mirrors `logic/`:

- `logic/cameras/`
- `logic/servos/`
- `logic/dataset/`
- `logic/depth/`
- `logic/detection/`
- `logic/train/`
- `logic/inference/`
- `logic/reconstruct/`

The older `modules/*.md` files are broad summaries only. Use `logic/` for readable, split module documentation.

## Runbooks

For operational scenarios, use `runbooks/`:

- `runbooks/preflight.md`
- `runbooks/data_collection.md`
- `runbooks/dataset_preparation.md`
- `runbooks/create_experiment_version.md`
- `runbooks/local_training.md`
- `runbooks/remote_training.md`
- `runbooks/training_chains.md`
- `runbooks/checkpoint_pull.md`
- `runbooks/real_inference.md`
- `runbooks/evaluation.md`
- `runbooks/evaluation_analysis.md`
- `runbooks/common_scenarios.md`
