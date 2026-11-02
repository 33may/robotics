# vla — VLA development framework

A framework for building vision-language-action (VLA) policies for real robot manipulation tasks. It covers the full development loop — collecting teleoperated demonstrations, managing and inspecting the dataset, training with versioned experiments (locally or on a remote GPU box), and evaluating policies on the real robot with predefined protocols. Simulation scene creation from real-world captures is available as an additional data source.

Built on: SO-101 arms (Feetech servos), Intel RealSense D405 cameras, LeRobot datasets, SmolVLA via `lerobot-train`.

## What's inside

| Stage | Module | What it gives you |
|---|---|---|
| Physical Env | `logic/servos`, `logic/cameras` | SO-101 scan, calibration profiles, rest pose, EEPROM recovery; camera presets, live view, USB reset |
| Collection | `lerobot-record` + camera presets | Leader–follower teleoperation recorded straight into a LeRobot dataset |
| Dataset | `logic/dataset`, `logic/depth` | Inspection reports, curation/trimming, conversion (HDF5 ↔ LeRobot), merging, depth as an extra image stream |
| Training | `logic/train`, `experiments/` | Versioned experiment folders with frozen configs, local engine, remote `lerobot-train` over SSH, sequential training chains |
| Evaluation | `logic/inference` | Real-robot inference, protocol-based evaluation sessions (trial list defined up front), session analysis and rendering |
| Simulation | `logic/reconstruct` | Video → COLMAP/MILo mesh → USD → Isaac/LeIsaac task; output converts back into the same LeRobot schema |

All stages meet at one contract: the **LeRobot dataset schema** — cameras define the image keys, servos define the joint meaning of `action`/`observation.state`, and training and inference must match it.

## Install

The framework is split across environments because the stacks don't co-exist (Isaac Sim pins numpy < 2, reconstruction needs torch 2.5 / CUDA 12.4).

| Env | Spec | Needed for |
|---|---|---|
| `vla` | `envs/vla.yml` (py3.12, torch 2.7 cu128, lerobot 0.4.4) | physical env, collection, dataset tools, inference, evaluation |
| `gsplat-pt25` | `envs/gsplat-pt25.yml` (py3.11, torch 2.5.1 cu124) | simulation: reconstruction |
| `isaac` | `envs/isaac.yml` (py3.11, Isaac Sim 5.0, Isaac Lab 0.47.1) | simulation: scene + task generation |

```bash
conda env create -f vbti/vla/envs/vla.yml
conda activate vla
pip install -e vbti/vla
```

## Documentation

```
docs/
  tutorial/     hands-on, in order: 01 cameras · 02 robot · 03 teleoperation · 04 dataset · 05 training · 06 evaluation · 07 inference
  runbooks/     task recipes — preflight, collection, training (local / remote / chains), evaluation, checkpoints
  modules/      reference per code module, mirrors logic/ — commands, arguments, contracts
  blueprints/   how each subsystem is designed and why — dataset, training, evaluation, hardware, reconstruction; C4 model in design/
  agents/       instructions for the specialist agents (dataset, hardware, training, evaluation, reconstruction)
  research/     background: SmolVLA / GR00T training internals, camera handling, Cosmos transfer, domain randomization
  case-study/   duck-cup with SmolVLA: the process we followed and what we got
```

**How to read:** start with `tutorial/` on the real hardware. When you start a concrete task, open its runbook. Go to `modules/` for exact commands and arguments, and to `blueprints/` before changing how a subsystem works. `case-study/` shows the whole loop used end to end.

**Code layout:** `logic/` framework code · `experiments/` versioned experiment configs and eval sessions · `envs/` conda specs · `udev/` camera rules · `research/` past research experiments. Open bugs: `KNOWN_ISSUES.md`.
