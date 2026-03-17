# VBTI — Vision-Based Training Infrastructure

Simulation-driven robotics learning pipeline. Reconstructs real-world environments into physically-based simulations for scalable data generation and robot training.

**Pipeline:** Phone video → 3D reconstruction → mesh extraction → Isaac Sim scene → data collection → policy training → inference

---

## Environment Setup

Two conda environments are used — one for 3D reconstruction, one for simulation/training.

### 1. Reconstruction Environment (`gsplat-pt25`)

Used for: COLMAP, MILo GS training, mesh processing, format conversion.

```bash
conda create -n gsplat-pt25 python=3.11
conda activate gsplat-pt25

# PyTorch
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Vendored libraries (see Library Setup below)
bash vbti/libs/setup_libs.sh
pip install -e vbti/libs/nerfstudio
pip install -e vbti/libs/sharp-frame-extractor
pip install viser==1.0.21

# MILo — requires gcc-14 on Fedora 42
sudo dnf install -y gcc14-c++ gcc14
export CUDA_HOME=/usr/local/cuda-12.9
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4070 Ti SUPER
export MAX_JOBS=8
# See docs/project_knowledge_base.md for full MILo build instructions

# Core pipeline deps
pip install open3d trimesh plyfile polyscope fire loguru pyyaml
pip install opencv-python numpy h5py pandas pillow tqdm
pip install usd-core  # USD Python bindings (standalone, no Isaac needed)
```

### 2. Isaac Sim Environment (`isaac`)

Used for: simulation, teleoperation, data collection, training, inference.

```bash
# Isaac Sim 5.0 + Isaac Lab 0.47.1 (follow NVIDIA install docs)
# Key versions:
#   python 3.11, torch 2.7.0+cu128, numpy 1.26.4
#   See vbti/env/isaac_env.txt for full package list
#   See vbti/env/isaac_constraints.txt for pinned versions

# IMPORTANT: numpy>=2.0 breaks omni.syntheticdata, opencv>=4.11 forces numpy>=2
pip install -c vbti/env/isaac_constraints.txt opencv-python numpy

# LeRobot (from submodule)
pip install -e lerobot/

# LeIsaac (from submodule)
pip install -e leisaac/source/leisaac/

# Additional deps
pip install fire loguru h5py pandas
```

### Compilation Notes (Fedora 42)

- System GCC 15 is unsupported by CUDA — always use `gcc-14`
- glibc 2.41 has cospi/sinpi conflicts — patch CUDA `math_functions.h` if MILo build fails
- Missing `#include <cstdint>` in 9 MILo headers — add manually if compilation errors
- See `docs/project_knowledge_base.md` for full build troubleshooting

---

## Library Setup

External libraries live in `vbti/libs/` — they are **not git submodules**, just cloned repos. Run the setup script to clone them:

```bash
bash vbti/libs/setup_libs.sh
```

This clones:

| Library | Repo | Used by |
|---------|------|---------|
| **MILo** | `Anttwo/MILo` | `gs_milo_utils.py` — mesh extraction from GS |
| **nerfstudio** | `nerfstudio-project/nerfstudio` | `colmap_utils.py` — COLMAP SfM wrapper |
| **sharp-frame-extractor** | `cansik/sharp-frame-extractor` | `video_utils.py` — quality-based frame selection |

After cloning, install with pip (see Environment Setup above). MILo requires additional compilation steps documented in the knowledge base.

`vbti/libs/` is gitignored — each developer clones their own copy.

---

## Quick Start

### Full Pipeline (video → runnable simulation task)

```bash
conda activate gsplat-pt25

# 1. Extract frames from video
python -m vbti.logic.reconstruct.master video_processing \
  --video_path data/scene.mp4 --output_dir data/frames --mode count --value 200

# 2. COLMAP + MILo reconstruction → mesh
python -m vbti.logic.reconstruct.master gs_reconstruction \
  --frames_dir data/frames --output_dir data/gs

# 3. Mesh → USD with physics
python -m vbti.logic.reconstruct.master ply_to_usda \
  --mesh_path data/gs/milo/mesh_learnable_sdf.ply --output_path data/scene.usda

# 4. Compose scene in Isaac Sim GUI (place robot, cameras, objects, lights)
# Then generate LeIsaac task:
conda activate isaac
python -m vbti.logic.reconstruct.master scene_composition \
  --scene_usda_path data/scene_composed.usda --task_name my_task
```

### Teleoperation & Data Collection

```bash
conda activate isaac
python teleop_se3_agent.py \
  --task LeIsaac-SO101-VbtiMeshTable-v0 \
  --teleop_device so101leader --enable_cameras
# Controls: B=start, N=success+reset, R=discard+reset
```

### Training

```bash
conda activate isaac
python -m vbti.logic.train.train_smolvla_custom
```

### Inference

```bash
conda activate isaac
python -m vbti.logic.inference.run_smolvla_inference \
  --checkpoint outputs/train/smolvla_lift_cube_3cams/best \
  --task LeIsaac-SO101-LiftCube-v0 --enable_cameras
```

---

## Directory Structure

```
vbti/
├── logic/                      # Core pipeline modules
│   ├── reconstruct/            # 3D reconstruction pipeline
│   │   ├── master.py           # Pipeline orchestrator CLI
│   │   ├── video_utils.py      # Frame extraction, rotation fix
│   │   ├── colmap_utils.py     # COLMAP SfM reconstruction
│   │   ├── gs_milo_utils.py    # MILo GS training + mesh extraction
│   │   ├── format_utils.py     # PLY/GLB → USD conversion
│   │   ├── clean_mesh.py       # Interactive Polyscope mesh cleaner
│   │   ├── robot_utils.py      # Robot USD preparation + LeIsaac pipeline
│   │   ├── isaac_cfg_utils.py  # IsaacLab/LeIsaac code generation
│   │   └── cosmos_transfer.py  # Cosmos Transfer data augmentation (experimental)
│   ├── dataset/                # Dataset inspection, conversion, loading
│   ├── inference/              # SmolVLA inference in Isaac Sim
│   ├── train/                  # Training scripts + experiment management
│   ├── cameras/                # Camera utilities
│   └── servos/                 # Servo utilities
│
├── data/                   # Scene data (not in git)
│   ├── so_v1/              # SO-ARM101 v1 scene (COLMAP, MILo, configs)
│   └── ready_export_sov1/  # Pre-built LeIsaac env configs
│
├── docs/                   # Project documentation
│   ├── project_knowledge_base.md  # Complete project knowledge
│   ├── pipeline_processes.md      # Step-by-step process flows
│   ├── module_reference.md        # Per-file function index
│   ├── domain_randomization.md    # DR parameter reference
│   ├── hardware_setup.md          # Physical hardware config
│   └── sessions/                  # Debug session notes
│
├── libs/                   # Vendored libraries (not in git)
│   ├── MILo/               # Mesh-from-GS extraction
│   ├── nerfstudio/         # GS training
│   └── sharp-frame-extractor/ # Video frame extraction
│
├── isaac/                  # IsaacLab env configs
├── research/               # Research docs & experiments
└── outputs/                # Training outputs (not in git)
```

---

## Core Modules

| Module | What it does | Entry point |
|--------|-------------|-------------|
| `reconstruct/master.py` | Orchestrates full pipeline | `python -m vbti.logic.reconstruct.master <command>` |
| `reconstruct/video_utils.py` | Extract frames, fix phone rotation | Called by master |
| `reconstruct/colmap_utils.py` | COLMAP SfM + model validation + undistortion | Called by master |
| `reconstruct/gs_milo_utils.py` | MILo GS training + learnable SDF mesh extraction | Called by master |
| `reconstruct/format_utils.py` | PLY→USD (sRGB→linear, COLMAP→USD coords, PCA align) | Called by master |
| `reconstruct/clean_mesh.py` | Interactive Polyscope GUI for mesh artifact removal | Standalone |
| `reconstruct/robot_utils.py` | Robot USD prep (drives, kinematic base, joint config) | `python -m vbti.logic.reconstruct.robot_utils <command>` |
| `reconstruct/isaac_cfg_utils.py` | Generate LeIsaac/IsaacLab task code from composed USDA | Called by master or robot_utils |
| `reconstruct/cosmos_transfer.py` | Cosmos Transfer augmentation (experimental, not production) | `python -m vbti.logic.reconstruct.cosmos_transfer <command>` |
| `dataset/inspect_dataset.py` | Dataset inspection reports (LeRobot + HDF5) | Standalone |

---

## Documentation

- `docs/project_knowledge_base.md` — Full project context, decisions, architecture
- `docs/pipeline_processes.md` — Code-to-knowledge map: exact functions, data formats, gotchas per pipeline step
- `docs/module_reference.md` — Function index for every module (use for code review)
- `docs/hardware_setup.md` — SO101 arms, RealSense cameras, servo config
- `docs/domain_randomization.md` — DR parameter tables
