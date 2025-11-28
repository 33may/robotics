# LeRobot Study Guide - SO101 Arm with SmolVLA

## Overview
This guide documents the process of setting up LeRobot framework for implementing SmolVLA (Small Vision-Language-Action model) on the SO101 robotic arm. This serves as a learning resource for understanding the LeRobot framework and its integration with vision-language models for robotics tasks.

---

## Table of Contents
1. [Installation](#installation)
2. [LeRobot Architecture](#lerobot-architecture)
3. [SmolVLA Module](#smolvla-module)
4. [SO101 Robot](#so101-robot)
5. [Key Concepts](#key-concepts)
6. [Next Steps](#next-steps)

---

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.2+
- Git
- Conda (recommended for environment management)

### Step 1: Clone LeRobot Repository
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

### Step 2: Install LeRobot with SmolVLA Module
LeRobot uses a modular installation system with optional dependencies. For SmolVLA on SO101 arm, we install:

```bash
pip install -e ".[smolvla]"
```

**What gets installed with `smolvla` module:**
- `transformers>=4.53.0,<5.0.0` - Hugging Face Transformers for model architecture
- `num2words>=0.5.14,<0.6.0` - Text processing utilities
- `accelerate>=1.7.0,<2.0.0` - Training acceleration
- `safetensors>=0.4.3,<1.0.0` - Safe model serialization

---

## LeRobot Architecture

### Core Components

#### 1. Dataset Management
- **LeRobotDataset**: Main dataset class for loading/handling robot data
- Supports temporal frame retrieval with `delta_timestamps`
- Backed by Hugging Face datasets (Arrow/Parquet format)
- Videos stored in MP4 format for efficiency
- Metadata stored in JSON/JSONL

**Dataset Structure:**
```
dataset/
├── hf_dataset/           # Parquet files with observations/actions
├── videos/               # MP4 video files for camera streams
└── meta/                 # JSON metadata files
    ├── info.json
    ├── episodes.json
    └── stats.json
```

#### 2. Policy Models
LeRobot provides several pretrained policies:
- **ACT** (Action Chunking Transformer)
- **Diffusion Policy**
- **TDMPC** (Temporal Difference Model Predictive Control)
- **VQ-BeT** (Vector Quantized Behavior Transformer)
- **SmolVLA** (Small Vision-Language-Action) - Our focus
- **Gr00t** (Vision-language-action model)

#### 3. Command Line Tools
LeRobot provides CLI scripts for common tasks:
```bash
lerobot-calibrate          # Calibrate robot motors
lerobot-find-cameras       # Detect connected cameras
lerobot-find-port          # Find serial port for motors
lerobot-record             # Record demonstration data
lerobot-replay             # Replay recorded episodes
lerobot-setup-motors       # Setup motor configurations
lerobot-teleoperate        # Manual teleoperation
lerobot-train              # Train policies
lerobot-eval               # Evaluate policies
lerobot-dataset-viz        # Visualize datasets with rerun.io
```

---

## SmolVLA Module

### What is SmolVLA?
SmolVLA is a vision-language-action model that combines:
- **Vision**: Processes camera images
- **Language**: Understands natural language task descriptions
- **Action**: Predicts robot actions based on vision + language

### Architecture Details
- Built on top of Hugging Face Transformers
- Uses vision encoders to process camera streams
- Language model processes task instructions
- Action decoder outputs robot control commands

### Use Cases
- Natural language-guided manipulation
- Few-shot learning from demonstrations
- Multi-modal task understanding
- Vision-grounded language following

### Key Papers & Resources
- [SmolVLA Paper](https://arxiv.org/abs/2506.01844)
- [SmolVLA Blog Post](https://huggingface.co/blog/smolvla)

---

## SO101 Robot

### Overview
- **Cost**: ~€114 per arm (affordable entry point)
- **Purpose**: Low-cost robotic arm for learning and experimentation
- **Training**: Can be trained in minutes with few demonstrations
- Evolved from SO-100 design

### Hardware Features
- Follower arm (for task execution)
- Leader arm (for teleoperation/data collection)
- Compatible with LeRobot's teleoperation pipeline

### Resources
- [SO-101 Tutorial](https://huggingface.co/docs/lerobot/so101)
- Community: [huggingface.co/lerobot](https://huggingface.co/lerobot)

---

## Key Concepts

### 1. Imitation Learning
LeRobot focuses on imitation learning - learning robot behaviors from human demonstrations:
1. **Collect Data**: Teleoperate robot to demonstrate task
2. **Train Policy**: Learn mapping from observations to actions
3. **Deploy**: Run trained policy autonomously

### 2. Delta Timestamps
Special feature of LeRobotDataset for temporal context:
```python
delta_timestamps = {
    "observation.image": [-1.0, -0.5, -0.2, 0],  # 4 frames: 1s, 0.5s, 0.2s ago + current
    "action": [0]  # Current action only
}
```

### 3. Episode-Based Organization
Data organized in episodes:
- Each episode = one complete task demonstration
- Frames indexed within episodes
- Metadata tracks episode boundaries

### 4. Modality Types
- **State observations**: Joint positions, velocities
- **Visual observations**: Camera images/videos
- **Actions**: Target joint positions, velocities, or torques
- **Language**: Task descriptions (for SmolVLA)

---

## Available Installation Modules

### Robot Hardware
- `feetech` - Feetech servo motors
- `dynamixel` - Dynamixel servo motors
- `gamepad` - Game controller support
- `hopejr` - HopeJR humanoid arm
- `lekiwi` - Mobile base for SO-101
- `reachy2` - Reachy2 robot support
- `intelrealsense` - Intel RealSense cameras
- `phone` - Phone-based teleoperation

### Policies
- `pi` - OpenPI policy
- `smolvla` - SmolVLA vision-language-action
- `groot` - Gr00t vision-language-action
- `hilserl` - HIL-SERL policy

### Simulation Environments
- `aloha` - ALOHA simulation
- `pusht` - PushT 2D manipulation
- `libero` - LIBERO benchmark
- `metaworld` - Meta-World benchmark

### Development
- `dev` - Development tools
- `test` - Testing utilities
- `async` - Async inference support

---

## Next Steps

### 1. Explore LeRobot Structure
```bash
cd lerobot
ls src/lerobot/
# Key directories:
# - policies/      Policy implementations
# - datasets/      Dataset utilities
# - robots/        Robot configurations
# - scripts/       CLI tools
# - cameras/       Camera interfaces
```

### 2. Understand SmolVLA Implementation
- Study the policy architecture in `src/lerobot/policies/`
- Review SmolVLA configuration files
- Understand image preprocessing pipeline

### 3. Dataset Exploration
- Download example datasets from Hugging Face hub
- Visualize with `lerobot-dataset-viz`
- Understand data format and structure

### 4. SO101 Configuration
- Review SO101 robot configuration
- Understand motor setup
- Configure cameras

### 5. Training Pipeline
- Prepare training configuration
- Set up WandB for experiment tracking
- Run initial training experiments
- Evaluate and iterate

### 6. Deployment
- Test trained policy in simulation
- Deploy to real SO101 robot
- Collect feedback and refine

---

## Important Notes

### File Formats
- **Datasets**: Parquet (via Hugging Face datasets)
- **Videos**: MP4 (H.264 encoding recommended)
- **Metadata**: JSON/JSONL
- **Models**: SafeTensors (secure PyTorch format)

### Best Practices
1. Always use virtual environments
2. Track experiments with WandB
3. Version control your robot configs
4. Document your dataset collection process
5. Use `rerun.io` for visualization during debugging

### Common Pitfalls
- Ensure camera permissions are set correctly
- Check motor serial port access
- Verify ffmpeg installation for video encoding
- Match dataset FPS with robot control frequency

---

## Resources

### Documentation
- [LeRobot Docs](https://huggingface.co/docs/lerobot/index)
- [Hugging Face Hub](https://huggingface.co/lerobot)

### Community
- [Discord](https://discord.gg/s3KuuzsPFb)
- [GitHub Issues](https://github.com/huggingface/lerobot/issues)

### Papers
- SmolVLA: [arxiv.org/abs/2506.01844](https://arxiv.org/abs/2506.01844)
- ACT (ALOHA): [tonyzhaozh.github.io/aloha](https://tonyzhaozh.github.io/aloha)
- Diffusion Policy: [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu)

---

## Session Log

### 2025-11-27: Initial Setup
- Cloned LeRobot repository
- Researched available installation modules
- Installed LeRobot with SmolVLA module from source
- Created this study guide

**Next Actions:**
- Explore LeRobot codebase structure
- Study SmolVLA policy implementation
- Set up SO101 robot configuration
- Prepare dataset collection pipeline
