# Container Diagram Prompt

Create a C4 Container diagram for the VBTI Modular Robot Learning Platform. The main design requirement is modularity: the project must not be shown as one fixed script for one demonstration task, but as a reusable platform made of independent modules that can be reused in future robotics projects.

The central software system should be named **VBTI Modular Robot Learning Platform**. Inside it, show two clear product layers/subblocks:

## 1. 3D Reconstruction Pipeline

This layer turns raw visual captures into simulation-ready 3D assets and environments. It should include these modules:

- **Reconstruction Orchestrator** (`logic/reconstruct/master.py`): CLI that chains the reconstruction steps.
- **Media Processing Utilities** (`logic/reconstruct/video_utils.py`): raw video/photo handling, rotation fixes, sharp frame extraction.
- **SfM / Camera Reconstruction** (`logic/reconstruct/colmap_utils.py`): Nerfstudio/COLMAP processing, sparse model validation, undistortion.
- **Gaussian Splat + Mesh Reconstruction** (`logic/reconstruct/gs_milo_utils.py`, `scripts/3d/`): MILo Gaussian splat training, learnable SDF mesh extraction, Poisson/repair scripts.
- **3D Format + Physics Export** (`logic/reconstruct/format_utils.py`, `clean_mesh.py`, `scripts/3d/`): PLY, point cloud, GLB, USD/USDA, collision meshes, deformable assets, coordinate/color conversions.
- **Scene + Simulation Asset Generation** (`logic/reconstruct/isaac_cfg_utils.py`, `robot_utils.py`): scene config extraction, robot prep, LeIsaac/IsaacLab task generation, cameras/lights/assets.
- **Cosmos Augmentation Preparation** (`logic/reconstruct/cosmos_transfer.py`): HDF5 modality extraction, RGB/depth/edge/seg video prep, transfer configs.

Show the main flow as:

`Raw Capture Sources -> Media Processing -> SfM/COLMAP -> Gaussian Splat + Mesh -> Format/Physics Export -> Scene/Simulation Generation -> Isaac Sim / Isaac Lab`

## 2. AI Model Pipeline

This layer contains reusable utilities for robot learning, from hardware setup to dataset preparation, training, inference, and evaluation. It should include these modules:

- **Hardware Utilities** (`logic/cameras/`, `logic/servos/`): RealSense camera setup/view/calibration/reset and Feetech servo scan/calibration/profile/rest/unlock tools.
- **Dataset Utilities** (`logic/dataset/`): LeRobot/HDF5 conversion, trimming, augmentation, subsampling, inspection, replay, validation, feature editing.
- **Perception + Detection Utilities** (`logic/detection/`, `logic/inference/async_detector.py`): object detection, phase detection, dataset processing, detector distillation/export, realtime async detection.
- **Depth Utilities** (`logic/depth/`, dataset depth tools): depth baking, gripper depth capture, realtime preparation, colorization, real-vs-estimated comparison, depth feature insertion.
- **Training Utilities** (`logic/train/`): SmolVLA/GR00T training backends, config utilities, experiment launch/monitoring, chains, remote training.
- **Inference Utilities** (`logic/inference/`): real policy inference, prompt/voice input, async chunk runner, policy action dispatch.
- **Evaluation Utilities** (`logic/inference/eval_*`, `logic/inference/protocols/`): eval engine, helpers, renderers, checkpoint sweeps, protocol/scenario generators.
- **Simulation Utilities** (`scripts/sim/`, generated Isaac/LeIsaac configs): simulation playgrounds, generated environments, synthetic/domain-randomized task execution.
- **Remote Execution Utilities** (`logic/train/remote.py`, `remote.yaml`): remote sync and job launch for training/evaluation.
- **Design Knowledge Base** (`.august/knowledge`, `.august/memory`, `docs/`): codegraph, memory, and docs that ground design decisions.

Show the main AI model flow as:

`Hardware Utilities -> Dataset Utilities -> Training Utilities -> Hugging Face Hub -> Inference/Evaluation Utilities -> Hardware Utilities`

Also show cross-links:

- Dataset utilities can call detection and depth utilities to enrich datasets.
- Inference can call perception/depth utilities for realtime state augmentation.
- Evaluation drives inference and uses hardware setup.
- Simulation uses generated scenes/tasks from the 3D reconstruction layer.
- The design knowledge base documents both layers and should sit as supporting infrastructure, not as runtime logic.

Visual direction: make the two product layers visually obvious as separate horizontal or vertical grouped blocks inside the main system boundary. The 3D layer should read as an asset/environment pipeline. The AI layer should read as a robot-learning lifecycle pipeline. Avoid making it look like a single monolithic script.
