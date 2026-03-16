# Simulation-Driven Robotics — VBTI

End-to-end framework for developing and deploying robotic manipulation systems through 3D simulation. Reconstructs real environments into physically accurate digital scenes, enabling scalable data generation, policy training, and sim-to-real transfer.

Built during a full-time research project at [VBTI](https://vbti.nl) (Eindhoven, NL) with the SO-101 6-DOF robot arm, NVIDIA Isaac Sim, and HuggingFace LeRobot.

## Pipeline

```
Phone Video → Frame Extraction → COLMAP SfM → Gaussian Splatting (MILo)
    → Mesh Extraction → USD Scene → Isaac Sim Environment
        → Teleoperation & Data Collection (with Domain Randomization)
            → HDF5 → LeRobot Dataset → SmolVLA Training → Inference
```

Each step is automated through `master.py` — a single CLI orchestrator that chains all modules.

## Key Components

| Module | Purpose |
|--------|---------|
| **3D Reconstruction** | Video → COLMAP → Gaussian Splatting → mesh extraction → USD scene |
| **Scene Generation** | Automated IsaacLab environment generation from composed USDA scenes |
| **Domain Randomization** | Object pose, camera jitter, lighting, physics properties per episode |
| **Data Collection** | Leader-follower teleoperation in simulation, HDF5 recording with 4 cameras |
| **Cosmos Transfer** | NVIDIA Cosmos 2.5 for sim-to-real visual augmentation (depth + edge control) |
| **Training & Inference** | SmolVLA vision-language-action model, sim-to-real co-training |
| **Dataset Tools** | HDF5 ↔ LeRobot conversion, inspection, sim/real format unification |

## Project Structure

```
robotics/
├── vbti/                   ← Core toolkit
│   ├── logic/              ← Pipeline modules (master.py, all utils)
│   ├── libs/               ← MILo, nerfstudio, sharp-frame-extractor
│   ├── data/               ← Scene assets, meshes, HDRI, configs
│   ├── docs/               ← Pipeline docs, knowledge base
│   ├── scripts/            ← Camera, servo, 3D processing utilities
│   └── env/                ← Conda environment specs
├── leisaac/                ← Isaac Sim framework (LeRobot + IsaacLab bridge)
├── lerobot/                ← HuggingFace LeRobot (real robot control)
├── robots/                 ← SO-101 USD assets
├── datasets/               ← Sim + real collected datasets
└── calibration/            ← Servo calibration configs
```

## Quick Start

```bash
# Reconstruct environment from video
python vbti/logic/master.py video_processing --video_path scene.mp4 --output_dir frames
python vbti/logic/master.py gs_reconstruction --frames_dir frames --output_dir gs
python vbti/logic/master.py ply_to_usda --mesh_path gs/milo/mesh_learnable_sdf.ply --output_path scene.usda

# Generate simulation environment (after composing scene in Isaac Sim)
python vbti/logic/master.py scene_composition --scene_usda_path scene.usda --task_name my_task

# Collect data via teleoperation
python teleop_se3_agent.py --task LeIsaac-SO101-MyTask-v0 --teleop_device so101leader --enable_cameras

# Train policy
python vbti/logic/train/train_smolvla_custom.py

# Run inference in simulation
python -m vbti.logic.inference.run_smolvla_inference --checkpoint outputs/best --task LeIsaac-SO101-MyTask-v0
```

## Hardware

- **Robot**: SO-101 6-DOF arm + gripper (Feetech STS3215 servos)
- **Cameras**: 4× Intel RealSense D405 (top, left, right, wrist)
- **Simulation**: NVIDIA Isaac Sim 5.0 + IsaacLab 0.47.1
- **GPU**: RTX 4070 Ti SUPER (local), A100/H100 (cloud for Cosmos Transfer)

## Stack

PyTorch · NVIDIA Isaac Sim/Lab · HuggingFace LeRobot · SmolVLA · PhysX · USD · COLMAP · Gaussian Splatting (MILo) · NVIDIA Cosmos Transfer 2.5

## Documentation

Detailed pipeline docs, process flows, and module reference in [`vbti/docs/`](vbti/docs/).
