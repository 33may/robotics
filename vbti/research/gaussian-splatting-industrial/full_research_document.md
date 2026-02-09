# Gaussian Splatting for Industrial Robotics Simulation
## Complete Research Document with Reasoning Chain

**Research Date:** January 27, 2026
**Project:** Robotics - Digital Twin Creation
**Hardware:** RTX 4070 Ti SUPER (16GB VRAM), CUDA 12.9

---

# Table of Contents

1. [Research Context & Constraints](#1-research-context--constraints)
2. [Initial Analysis & Tool Landscape](#2-initial-analysis--tool-landscape)
3. [Industry Landscape Research](#3-industry-landscape-research)
4. [Reconstruction Pipelines & Tools](#4-reconstruction-pipelines--tools)
5. [Post-Processing & Cleanup](#5-post-processing--cleanup)
6. [Scene Editing Capabilities](#6-scene-editing-capabilities)
7. [Export Formats & Simulation Integration](#7-export-formats--simulation-integration)
8. [Emerging Methods (2025-2026)](#8-emerging-methods-2025-2026)
9. [Final Decision & Recommendations](#9-final-decision--recommendations)
10. [Knowledge Graph Node Index](#10-knowledge-graph-node-index)

---

# 1. Research Context & Constraints

## 1.1 Project Goal

> **Task: Digital Twin Creation**
> Explore tools and methods to develop a complete pipeline for 3D scene reconstruction. Create and integrate sim-ready assets for components requiring physics interaction or enhanced detail.

The broader robotics project focuses on creating a complete pipeline to train and deploy robots in manufacturing and agriculture domains, using simulation capabilities to train models in 3D reconstructed scenes.

## 1.2 System Environment

**Discovery: System Configuration**
> System check: RTX 4070 Ti SUPER (16GB VRAM), CUDA 12.9, Python 3.13.5 - solid setup for Gaussian Splatting workloads

**Discovery: Existing Environment**
> Environment: Use existing `isaac` conda env for Gaussian Splatting. Has Python 3.11.14, PyTorch 2.7.0+cu128, CUDA 12.8. Path: /home/may33/miniconda3/envs/isaac. Just needs gsplat and nerfstudio installed.

## 1.3 User Constraints

**Constraint 1: NVIDIA Preference**
> User prefers to use NVIDIA tools and stack as much as possible for the pipeline

**Constraint 2: Open Source Preference**
> User prefers gsplat/nerfstudio for Gaussian Splatting pipeline - open source, suitable for custom pipelines

These constraints shaped all subsequent tool evaluations and decisions.

---

# 2. Initial Analysis & Tool Landscape

## 2.1 Initial Tool Survey

**Thought: Tool Landscape Overview**
> Researched Gaussian Splatting tools landscape 2026: Main options are:
> 1. **gsplat/nerfstudio** - open source, CUDA accelerated, NVIDIA 3DGUT integrated, 4x memory reduction vs original
> 2. **Splatfacto** - nerfstudio's implementation, developer friendly
> 3. **Postshot** - commercial GUI app
> 4. **Unity/Unreal plugins** for game/XR
>
> gsplat supports batch rasterization, depth rendering, multi-GPU, anti-aliasing.

## 2.2 Detailed Library Comparison

**Thought: Gaussian Splatting Libraries 2026**

### Option 1: gsplat (nerfstudio-project)
- Open-source, Apache 2.0, PyTorch bindings
- NVIDIA 3DGUT integrated (April 2025)
- 4x memory reduction vs original, 15% faster training
- Batch rasterization, depth rendering, multi-GPU, anti-aliasing
- PPIPS integrated (Jan 2026)
- Powers nerfstudio's Splatfacto

### Option 2: 3DGRUT (nv-tlabs)
- NVIDIA's official implementation
- Ray tracing + rasterization hybrid
- Supports fisheye, rolling shutter, reflections, refractions
- Requires RT cores, 50% longer training, 55-190 fps
- More advanced but slower

### Option 3: Original 3DGS (graphdeco-inria)
- Reference implementation
- 2.7x speedup with sparse_adam optimizer
- Good baseline but gsplat has more features

### For Isaac Sim Robotics Integration:
- Isaac Sim 5.0+ has NuRec for 3DGS/3DGUT rendering
- Exports to USD format (custom schema with Gaussian data)
- GaussGym: 100k+ steps/sec on consumer GPUs
- RoboGSim: Real2Sim2Real pipeline
- World Labs Marble workflow: PLY -> USDZ via NuRec -> Isaac Sim

## 2.3 Preliminary Decision

**Decision: Use gsplat as primary library (tentative)**
> Best fit for NVIDIA stack: 3DGUT integrated, NuRec/USD export path to Isaac Sim, 4x memory reduction suits 16GB VRAM, most active development. May revisit if advanced camera effects (fisheye, reflections) require 3DGRUT.

---

# 3. Industry Landscape Research

## 3.1 Company Tech Stacks

**Discovery: NVIDIA Isaac Sim Dominance**
> NVIDIA Isaac Sim is the dominant industrial robotics simulation platform as of 2025-2026. Key capabilities: GPU-accelerated physics (PhysX), synthetic data generation, RL training, ROS integration, digital twin simulation. Major adopters include Amazon Robotics, TSMC, Wistron, FieldAI. Isaac Sim 4.5.0 is latest version. Also includes Isaac Lab for RL research. Newton physics engine (co-developed with Google DeepMind and Disney Research) is now available for GPU-accelerated simulation.

**Discovery: NVIDIA Cosmos World Foundation Models**
> NVIDIA Cosmos World Foundation Models launched CES 2025 for Physical AI. Purpose-built for robotics and autonomous vehicles. Generates physics-aware synthetic videos from text/image/video/sensor inputs. Cosmos Transfer 2.5 is 3.5x smaller, faster, with improved physics accuracy. Downloaded 2M+ times. Key adopters: 1X, Agility Robotics, Figure AI, Foretellix, Skild AI, Uber. Integrates with Isaac Sim for photorealistic data generation to reduce sim-to-real gap.

**Discovery: Covariant AI (Amazon)**
> Covariant AI (acquired by Amazon August 2024) uses: NVIDIA A100/H100 GPUs for training, RTX A5000 for inference. RFM-1 foundation model trained on warehouse robot data (text, images, videos, actions, sensor readings). Collects 1M+ trajectories every few weeks. RFM-1 can predict physics via AI-generated video - essentially a world model for predicting how objects react to robotic actions. Uses deep RL and self-supervised learning.

**Discovery: Google Intrinsic**
> Google Intrinsic (Alphabet robotics unit) key developments 2025: Intrinsic Vision Model (IVM) - industrial-grade foundation model with sub-mm precision using RGB cameras ($500-1000), reducing hardware costs 5-20x vs depth sensors. AI for Industry Challenge running Feb 2025 - July 2026. RoboBallet multi-robot orchestration research with DeepMind - 25% trajectory improvement, 60% time reduction with 8 robots. Partnership with NVIDIA for Isaac foundation model integration. Google Genie 3 world model for generating interactive environments.

**Discovery: Boston Dynamics & RAI Institute**
> Boston Dynamics & RAI Institute (Feb 2025 partnership): Focus on reinforcement learning for Atlas humanoid. Key areas: sim-to-real transfer, whole-body loco-manipulation, full-body contact strategies. Atlas RL demo showed zero-shot transfer - policy trained in simulation deployed directly without fine-tuning. Prior collaboration produced Spot RL Researcher Kit achieving 11.5 mph (5.2 m/s) running. RAI Institute founded by Marc Raibert, funded by Hyundai.

**Discovery: Agility Robotics**
> Agility Robotics simulation approach: Whole-body control foundation model for Digit humanoid - LSTM neural network with <1M parameters trained in NVIDIA Isaac Sim for decades of simulated time over 3-4 days. Transfers zero-shot to real world. Uses deep RL for control. Powered by NVIDIA Jetson AGX Thor for real-time perception/navigation. Commercial deployment at 350+ sites (April 2026). Named 2024 RBR50 Robot of the Year - first humanoid in commercial warehouse deployments.

**Discovery: Figure AI**
> Figure AI tech stack: NVIDIA partnership for GPU infrastructure training and simulation. Helix VLA (Vision-Language-Action) model - dual-processing with System 2 at 7-9 Hz for planning, System 1 at 200 Hz for control. Figure 03 (Oct 2025) - third-gen humanoid with complete hardware/software redesign. Trains robots via visual and natural language prompts. 2026 prediction from CEO: humanoids performing unsupervised multi-day tasks in unfamiliar homes.

## 3.2 Industry Verticals

**Discovery: Automotive Digital Twins**
> Automotive digital twin simulation 2025: BMW Virtual Factory uses NVIDIA Omniverse for real-time 3D simulation - collision checks reduced from 4 weeks to 3 days. Complete line section with 41 cycles over 1000 sqm simulated. VR training for employees. Mercedes-Benz built high-precision twins of Rastatt, Kecskemét, Beijing plants using Omniverse and OpenUSD - every station, robot, aisle modeled. Tesla uses digital twins for production line optimization, battery testing, crash simulation. Market: $2.1B (2024) → $28.7B (2034) at 30.1% CAGR.

**Discovery: Warehouse Robotics**
> Warehouse robotics simulation 2025: Locus Robotics uses multi-faceted simulation combining DES (discrete event simulation) with robot autonomy models - differentiator for accurate AMR-human collaboration modeling. AI-Powered Simulation & Workflow Optimization unveiled at LogiMAT/ProMat 2025. 350+ sites deployed (April 2026). Amazon Robotics uses Omniverse libraries to reduce dev from years to months - BlueJay multi-arm manipulator from concept to production in ~1 year. 6 River Systems (now Ocado) also major player.

**Discovery: GS Adoption in Industry**
> Industrial adopters of Gaussian Splatting for simulation: Applied Intuition NeuralSim powered by GS, Tesla using GS, NVIDIA announced NuRec library for GS in Omniverse/Isaac Sim at SIGGRAPH 2025. Third Dimension SuperSim - simulator built from reality using GS. Key use cases: synthetic data multiplication, photorealistic training environments, reducing sim-to-real gap. Integration path: 3DGS assets into game engines with physics simulation for natural object arrangements.

## 3.3 GS in Robotics Research

**Discovery: 3D Gaussian Splatting in Robotics**
> 3D Gaussian Splatting in Robotics (2025-2026): Emerging as major sim-to-real bridge technology. Key research:
> - **RL-GSBridge** for manipulation policy training
> - **SplatSim** (CMU) for zero-shot Sim2Real RGB policies
> - **GWM** (Gaussian World Models) for model-based RL
> - NVIDIA approach: Physically Embodied Gaussians for real-time digital twin synchronization
>
> Key advantage: photorealistic rendering addresses domain gap between synthetic and real imagery. Training data generation pipeline using 3DGS for computer vision.

## 3.4 Analysis Thoughts

**Thought: Tech Stack Analysis**
> NVIDIA dominates industrial robotics simulation with Isaac Sim/Omniverse ecosystem. Almost every major robotics company (Agility, Figure AI, Boston Dynamics, Amazon, BMW, Mercedes, Tesla) uses NVIDIA technologies. The ecosystem includes: Isaac Sim (simulation), PhysX/Newton (physics), Cosmos (world foundation models), Jetson (edge deployment). This creates strong vendor lock-in but also ensures compatibility and support.

**Thought: Gaussian Splatting Opportunity**
> GS is at inflection point for robotics simulation. Research papers exploding (1,692 in 2025), key players adopting (NVIDIA NuRec, Applied Intuition NeuralSim, Tesla). Main value prop: photorealistic rendering that bridges sim-to-real gap. Current limitation: deployment on consumer hardware. Integration path is via NVIDIA ecosystem (Omniverse/Isaac Sim). This validates GS as a strong research direction for robotics simulation.

**Thought: Simulation Training Paradigm Shift**
> Industry moving from hand-coded movements to RL policies trained entirely in simulation with zero-shot transfer to real hardware. Key examples: Boston Dynamics Atlas, Agility Digit, Covariant RFM-1. Success requires: high-fidelity physics (MuJoCo, PhysX), photorealistic rendering (GS, ray tracing), massive scale (decades of simulated time in days). Foundation models (VLAs like Helix, world models like Cosmos/Genie) are the new frontier.

**Thought: Alternative Tech Stacks**
> While NVIDIA dominates, alternatives exist: Unity (Robotics Hub for warehouse automation, ROS2 support), Unreal Engine (Chaos Physics, MetaHuman), MuJoCo (leading for RL research, now open-source with GPU acceleration via MuJoCo-Warp), PyBullet (open-source, easier entry). Google Intrinsic building proprietary platform. For research: MuJoCo preferred for physics accuracy; for industrial deployment: Isaac Sim/Omniverse dominates.

**Thought: Industry Vertical Insights**
> (1) Automotive: Digital twins most mature, BMW/Mercedes/Tesla using Omniverse extensively. Focus on production line optimization, collision testing, worker training. (2) Warehouse: Locus, Amazon using simulation for deployment planning and AMR training. DES + robot autonomy models key. (3) Humanoids: Agility and Boston Dynamics leading with RL-based whole-body control. Zero-shot sim-to-real transfer becoming standard. (4) Manufacturing: TSMC, Wistron using Isaac for robotics development.

---

# 4. Reconstruction Pipelines & Tools

## 4.1 End-to-End Pipelines

**Discovery: Video-to-3DGS Pipelines**
> End-to-End Video-to-3DGS Pipeline: video-3d-reconstruction-gsplat (GitHub: Nannigalaxy/video-3d-reconstruction-gsplat) provides a streamlined pipeline for converting videos into 3D Gaussian Splatting models. Uses COLMAP for SfM and gsplat for training. LongSplat (NVIDIA Labs, ICCV 2025) is an unposed 3DGS framework for casual long videos - uses MASt3R for pose estimation, incremental joint optimization, and adaptive octree anchoring. Memory-efficient and scalable for novel view synthesis from free-form video capture.

## 4.2 Camera Pose Estimation

**Discovery: Pose Estimation Comparison**
> Camera Pose Estimation Comparison (2025):
> - **COLMAP**: Incremental SfM, high accuracy, slower baseline, limited scalability for large datasets
> - **GLOMAP**: Global SfM, 10-100x faster than COLMAP, 8% higher recall, on-par or superior quality. Requires COLMAP database as input. Some edge cases where relative pose estimation becomes bottleneck.
> - **hloc**: Modular toolbox for 6-DoF visual localization, state-of-the-art with learning-based features, works well combined with GLOMAP
> - **FASTMAP**: Newer method, 1-2 orders of magnitude faster for dense SfM, handles thousands of images efficiently

**Discovery: Alternative Pose Methods**
> Pose Estimation Methods for Gaussian Splatting (beyond COLMAP):
>
> **ACE-SLAM** (Scene Coordinate Regression):
> - Neural RGB-D SLAM with implicit map learning in real-time
> - Lightweight network maps 2D image features directly to 3D global coordinates
> - Efficient, low-memory, extremely fast relocalization
> - Privacy-preserving
>
> **GSplatLoc**:
> - Uses differentiable rendering of 3DGS for ultra-precise pose estimation
> - Gradient-based optimization minimizing depth map discrepancies
> - Translational errors within centimeters, rotation errors < 1 degree

## 4.3 Training Libraries

**Discovery: gsplat Training Library**
> gsplat Training Library (nerfstudio-project/gsplat):
> - CUDA accelerated rasterization with Python bindings
> - **4x less GPU memory** than original 3DGS implementation
> - **15% faster training** time
> - Exact same performance on PSNR, SSIM, LPIPS benchmarks
> - 2025 updates: PPIPS integration (Jan 2026), arbitrary batching over multiple scenes (May 2025), NVIDIA 3DGUT integration (April 2025)
>
> **Nerfstudio's Splatfacto**:
> - Uses gsplat as backend
> - Two optimization strategies: DefaultStrategy (gradient-based densification) and MCMCStrategy (stochastic birth/death)

**Discovery: 3DGUT Integration**
> 3DGUT (3D Gaussian Unbound Toolkit) - NVIDIA:
> - Integrated into gsplat (April 2025)
> - Enables secondary visual phenomena: reflections, refractions
> - Seamless integration with 3DGRT (NVIDIA ray-tracing)
> - Open-source at nv-tlabs/3dgrut
> - Features: fisheye camera support, advanced rendering
> - Works with PyTorch via gsplat Python bindings

**Discovery: Nerfstudio Workflow**
> Nerfstudio Gaussian Splatting Workflow (Splatfacto):
>
> **Installation:**
> - `pip install git+https://github.com/nerfstudio-project/gsplat.git`
> - Specific versions: `pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118`
> - Prerequisites: NVIDIA GPU + CUDA, FFmpeg
>
> **Training Command:**
> - `ns-train splatfacto --data <data>`
> - Interactive web viewer during training
> - Supports checkpoints, rendering, export
>
> **Data Requirements:**
> - Works MUCH better with pre-existing geometry (SfM points from COLMAP)
> - Supports COLMAP, Record3D, Polycam, nerfstudio formats

## 4.4 Data Capture

**Discovery: Data Capture Best Practices**
> Data Capture Best Practices for Gaussian Splatting:
>
> **Camera Equipment:**
> - Wide-angle lens recommended (e.g., Sigma 14mm f/1.4 DG DN Art)
> - Gimbal (e.g., DJI RS 4) for smoother captures, less vibration
> - Tripod or stabilizer essential
>
> **Camera Path:**
> - 20-30% overlap between adjacent views essential for SfM feature matching
> - Video often provides overlap automatically
> - DJI drones: "Point of Interest" orbiting feature works well, multiple passes at different distances
> - 100-250 images for small objects, 500+ for large scenes
>
> **Lighting:**
> - Even, diffuse lighting best
> - Avoid harsh directional light (creates shadows that confuse reconstruction)
> - Outdoor: overcast days ideal
> - Indoor: multiple soft light sources
>
> **Video Settings:**
> - 60+ FPS recommended
> - 4K resolution if possible
> - Lock exposure and focus
> - Avoid motion blur

## 4.5 Commercial Tools

**Discovery: Commercial GS Tools**
> Commercial Gaussian Splatting Tools (2025-2026):
>
> **Polycam:**
> - Web-based, cloud processing
> - Easy-to-use interface with capture guidance
> - Subscription required for multi-format export
> - Requires stable internet connection
>
> **Luma AI:**
> - Uses NeRF + 3DGS technology
> - Mobile app no longer supports direct GS (exports PLY for desktop rendering)
> - Cloud processing: 20-30 minutes per model
> - Best for developers/graphics professionals (JavaScript workflow)
> - Company focusing on Dream Machine (text/image-to-video)
>
> **Postshot (Jawset):**
> - Local processing, privacy-friendly
> - Live preview during training
> - Exports to Blender, Unreal Engine
> - Best for local/privacy-sensitive workflows

## 4.6 Quality Metrics

**Discovery: Quality Metrics**
> Quality Metrics for Gaussian Splatting Evaluation:
>
> **PSNR (Peak Signal-to-Noise Ratio):**
> - Higher = better (typically >30 dB is good quality)
> - Measures pixel-level errors
> - May fail to capture human visual sensitivity to high-frequency details
>
> **SSIM (Structural Similarity Index):**
> - Range: 0-1 (1 = perfect similarity)
> - Considers luminance, contrast, structure
> - More perceptually aligned than PSNR
> - Good at capturing edges and textures
>
> **LPIPS (Learned Perceptual Image Patch Similarity):**
> - Lower = better
> - Uses deep network features
> - Most aligned with human perception
> - Standard: AlexNet or VGG backbone
>
> **Benchmark Datasets:**
> - Mip-NeRF 360 (7 scenes)
> - Tanks & Temples
> - ScanNet/ScanNet++

## 4.7 Hardware Requirements

**Discovery: Hardware Requirements**
> Hardware Requirements for Gaussian Splatting Training:
>
> **Standard Requirements:**
> - CUDA Compute Capability 7.0+
> - 24 GB VRAM recommended for paper-quality training
> - RTX 3090 or better preferred
>
> **GPU Recommendations by Use Case:**
> - Professional/Research: NVIDIA A6000 (highest VRAM, most CUDA cores)
> - Developers/Large datasets: RTX 4090 or RTX 3090
> - Beginners/Hobbyists: RTX 3080 Ti
> - Budget: RTX 3060 (12GB VRAM)
>
> **Training with Limited VRAM (16GB like yours):**
> - VRAM usage scales with number of points (increases over training)
> - Use gsplat (4x more efficient than original)
> - Increase `--densify_grad_threshold`
> - Reduce `--densify_until_iter`
> - Set source images to CPU instead of CUDA

## 4.8 Tool Comparison Analysis

**Thought: Key Tool Comparison Analysis**
> Key Tool Comparison Analysis for Gaussian Splatting:
>
> **TRAINING TOOLS RANKING (2025):**
>
> 1. **gsplat (nerfstudio-project/gsplat)** - BEST FOR DEVELOPMENT
>    - Pros: 4x less VRAM, 15% faster, Apache 2.0 license, 3DGUT integration, active development
>    - Cons: Requires coding knowledge
>
> 2. **Nerfstudio Splatfacto** - BEST FOR RESEARCHERS
>    - Pros: Full pipeline, web viewer, ns-train CLI, good documentation
>    - Cons: COLMAP dependency, heavier setup
>
> 3. **Postshot (Jawset)** - BEST FOR LOCAL/PRIVACY
>    - Pros: GUI, live preview, Blender/Unreal export
>    - Cons: Commercial, less flexible

## 4.9 Recommended Workflow

**Thought: Practical Pipeline Recommendation**
> Practical Gaussian Splatting Pipeline Recommendation for Robotics:
>
> **RECOMMENDED END-TO-END WORKFLOW:**
>
> 1. **DATA CAPTURE:**
>    - Use gimbal-stabilized camera (DJI RS 4 or similar)
>    - Wide-angle lens (14mm range)
>    - 60+ FPS video or high-quality stills
>    - 20-30% overlap between views
>    - Even, diffuse lighting (avoid harsh shadows)
>    - No movement in scene during capture
>    - DJI drone "Point of Interest" orbiting works well
>
> 2. **CAMERA POSE ESTIMATION:**
>    - Default: COLMAP (most compatible, well-tested)
>    - Speed priority: GLOMAP (10-100x faster, 8% better recall)
>    - Large datasets: FASTMAP
>
> 3. **TRAINING:**
>    - gsplat library (most efficient for 16GB VRAM)
>    - Or Nerfstudio Splatfacto for full pipeline
>
> 4. **CLEANUP:**
>    - Clean-GS for floater removal
>    - FlashSplat for segmentation (~30 seconds)
>
> 5. **MESH EXPORT (if needed):**
>    - SuGaR for Blender/Unity compatibility
>    - OMeGa for best quality
>
> 6. **ISAAC SIM INTEGRATION:**
>    - Export to USDZ via NuRec
>    - Import into Isaac Sim 5.0+
>    - Add physics proxy (ground plane or mesh)

---

# 5. Post-Processing & Cleanup

## 5.1 Floater Removal

**Discovery: Floater/Artifact Removal Methods**
> Floater/Artifact Removal Methods for Gaussian Splatting:
>
> **EFA-GS (Eliminating-Floating-Artifacts GS):**
> - Frequency-domain analysis + depth-aware optimization
> - +3.57 dB PSNR improvement over Mip-splatting
> - Addresses spectral/geometric misalignment, density control failures, over-shrinking
> - https://jcwang-gh.github.io/EFA-GS
>
> **Clean-GS (Semantic Mask-Guided Pruning):**
> - Uses sparse semantic masks (as few as 3 masks, 1% of views)
> - 60-80% model compression while preserving object quality
> - 3 stages: visibility mask → object retrieval → background removal
> - Processing time: 2-5 minutes on CPU
> - https://arxiv.org/abs/2601.00913

## 5.2 Background Removal

**Discovery: Background Removal / Segmentation**
> Background Removal / Segmentation Methods for Gaussian Splatting:
>
> **Object-Centric 2DGS (ICPRAM 2025):**
> - Uses object masks for targeted reconstruction
> - Occlusion-aware pruning strategy
> - Up to 96% smaller models, 71% faster training
> - Uses SAM 2 for semi-automatic mask generation
> - Background loss penalizes Gaussians in non-object regions
>
> **Clean-GS (January 2025):**
> - Semantic mask-guided pruning
> - Only needs 3 segmentation masks (1% of views)
> - 60-80% compression with quality preservation
>
> **FlashSplat (ECCV 2024):**
> - ~30 seconds for segmentation
> - 50x faster than existing methods
> - Robust to noisy/inaccurate masks
>
> **Gaussian Grouping:**
> - SAM integration for "edit anything"
> - Local editing with identity encoding

## 5.3 Mesh Extraction

**Discovery: Mesh Extraction from GS**
> Mesh Extraction from Gaussian Splatting:
>
> **SuGaR (CVPR 2024):**
> - Surface-Aligned Gaussian Splatting
> - Regularization term encourages Gaussians to align with scene surface
> - Uses Poisson reconstruction (fast, scalable, preserves details)
> - Optional refinement: binds Gaussians to mesh surface
> - Enables editing in Blender/Unity/Unreal by manipulating mesh
> - GitHub: Anttwo/SuGaR
>
> **2DGS (2D Gaussian Splatting):**
> - Uses 2D Gaussian disks for view-consistent geometry
> - Depth map-based mesh extraction
> - Struggles with thin structures, high-frequency details
>
> **OMeGa:**
> - Joint mesh-splat optimization
> - +37-80% F-score improvement vs 2DGS baseline
> - Best quality but more complex

## 5.4 Point Cloud Tools

**Discovery: Point Cloud Processing Tools**
> Point Cloud Processing Tools for Gaussian Splatting:
>
> **CloudCompare:**
> - Open-source, free
> - Features: ICP registration, distance calculations, segmentation, statistical analysis
> - **LIMITATION: Does not manage GS natively** - only loads Gaussian centers, ignores rotation/scale
> - Result: exported point clouds more sparse than desired
> - Useful for: SOR filtering, Chamfer distance measurement
> - Version 2.13.2 commonly used
>
> **Open3D:**
> - Used for point cloud to mesh conversion
> - Statistical outlier removal
> - Poisson surface reconstruction
> - Python API
>
> **3DGSConverter:**
> - N-to-N format conversion (PLY, KSplat, SOG, SPZ, etc.)
> - GPU-accelerated filtering
> - GitHub: francescofugazzi/3dgsconverter

---

# 6. Scene Editing Capabilities

## 6.1 Text-Guided Editing

**Discovery: Scene Editing Tools**
> Scene Editing Tools for Gaussian Splatting (2025 State):
>
> 1. **GaussianEditor (CVPR 2024):** First 3D editing algorithm for GS, uses Gaussian semantic tracing and Hierarchical GS. Training in ~20 min. GitHub: buaacyw/GaussianEditor
>
> 2. **Instruct-GS2GS:** Uses InstructPix2Pix for iterative text-based scene editing. Integrated into nerfstudio.
>
> 3. **GaussCtrl (ECCV 2024):** Multi-view consistent editing using ControlNet, faster than iterative methods through depth-conditioned editing.
>
> 4. **3DSceneEditor (Dec 2024):** Fully 3D-based approach (not 2D-to-3D lifting), lowest GPU memory requirements, fastest of the text-guided methods.

## 6.2 Interactive Editors

**Discovery: Interactive GS Editors**
> Interactive GS Editors (Production Ready):
>
> 1. **SuperSplat (PlayCanvas):** Open-source, browser-based, MIT licensed
>    - Current version: 2.14 (Nov 2025) with video export (WebM/MOV/MKV)
>    - Features: Timeline animation, VR/AR mode (Quest 2/3, Vision Pro), selection tools (rect/brush/picker)
>    - URL: https://superspl.at/editor
>
> 2. **KIRI Engine:** Commercial app with GS creation and mesh export
>
> **Key finding:** SuperSplat is the leading free production tool for GS editing - fully featured browser-based editor.

## 6.3 Object Manipulation

**Discovery: Object Insertion/Removal**
> Object Insertion/Removal in Gaussian Splatting (2025):
>
> 1. **D3DR:** Inserts GS objects into GS scenes with diffusion-based lighting correction (DDS objective)
>
> 2. **MVInpainter:** Multi-view diffusion (Stable Video Diffusion) for view-consistent object inpainting
>
> 3. **MCMC Framework:** Robust object insertion via bilateral grids, directly operates on Gaussian attributes
>
> 4. **AG2aussian (ICCV 2025):** Anchor-graph structure for instance-level understanding, enables interactive click/text query and object removal

## 6.4 Relighting & Materials

**Discovery: Relighting and Material Editing**
> Relighting and Material Editing for Gaussian Splatting (2025):
>
> 1. **RTR-GS:** Geometry-light-material decomposition with albedo/metallic/roughness for PBR relighting
>
> 2. **Ref-Gaussian (ICLR 2025):** Real-time reflective object rendering with split-sum approximation, supports relighting/editing
>
> 3. **Large Material Gaussian Model (Sept 2025):** Generates PBR materials (albedo, roughness, metallic) instead of baked RGB. Dynamic relighting via ambient light maps.
>
> 4. **Relightable 3D Gaussian:** Per-Gaussian BRDF with BVH ray-tracing for real-time shadows

## 6.5 Text/Image to 3D

**Discovery: DreamGaussian**
> DreamGaussian - Text/Image to 3D (ICLR 2024 Oral):
>
> **Speed:** 10x faster than previous methods - photorealistic 3D from single image in 2 minutes
>
> **Pipeline:**
> 1. 3DGS initialization via SDS (Score Distillation Sampling)
> 2. Extract textured mesh from Gaussians
> 3. UV-space texture refinement via differentiable rendering
>
> **Related:**
> - **GaussianDreamer (CVPR 2024):** Text-to-3D in 15 minutes using 2D+3D diffusion bridging
> - **DreamGaussian4D:** Extends to 4D generation with spatial transformation

## 6.6 Summary Analysis

**Thought: Summary Analysis - Scene Editing**
> **Summary Analysis: GS Scene Editing, Export & Simulation for Robotics**
>
> **What's Production-Ready Today:**
> 1. **SuperSplat** - Free, browser-based editor for cleaning/cropping GS scenes
> 2. **NVIDIA Isaac Sim 5.0 + NuRec** - Complete smartphone-to-simulation pipeline
> 3. **3DGUT/3DGRUT** - Handles real robot camera distortion (fisheye, rolling shutter)
> 4. **glTF extensions (KHR_gaussian_splatting)** - Standardized interoperability
> 5. **LangSplat** - 199x faster than LERF for semantic queries
>
> **Current Gaps:**
> - Physics simulation requires mesh extraction (PhysGaussian/MILo for complex collision)
> - Dynamic scenes need 4DGS (adds complexity)
> - Material editing still research-stage for most use cases

---

# 7. Export Formats & Simulation Integration

## 7.1 Export Formats

**Discovery: Export Formats**
> Export Formats for Gaussian Splatting (2025):
>
> 1. **PLY:** Traditional format, but files can exceed 250MB for detailed outdoor scenes
>
> 2. **SPZ (Niantic):** New format achieving 90% smaller files vs PLY, MIT licensed, designed as "JPG for splats"
>
> 3. **glTF (Khronos Standard - Aug 2025):**
>    - Extensions: KHR_gaussian_splatting, KHR_gaussian_splatting_compression_spz
>    - Attributes: Position, Rotation (quat), Scale, Opacity, Spherical Harmonics
>    - Major step for interoperability
>
> 4. **USD/USDZ:** NVIDIA custom schema (OmniNuRecVolumeAPI), Apple proposal (PR #3716) in progress

**Discovery: USD Schema for GS**
> USD Schema for Gaussian Splatting (2025):
>
> **Apple's OpenUSD Proposal (PR #3716):**
> - GaussiansAPI and SphericalHarmonicsAPI schemas
> - Applied on top of UsdGeomPoints for ecosystem compatibility
> - Developed with Adobe, NVIDIA feedback via AOUSD Emerging Geometry Interest Group
>
> **NVIDIA NuRec Schema:**
> - OmniNuRecVolumeAPI on UsdVolVolume prim
> - Properties for rendering control
> - USDZ packaging of GS data
>
> **OpenUSD Core Spec 1.0:** Standard data types, file formats, composition behaviors now finalized

## 7.2 NVIDIA Isaac Sim Integration

**Discovery: NVIDIA Isaac Sim + NuRec Integration**
> NVIDIA Isaac Sim + NuRec Integration (2025):
>
> **Workflow:**
> 1. Capture photos with smartphone (iPhone/Android), 60% overlap, lock focus/exposure
> 2. Sparse reconstruction with COLMAP (pinhole camera model)
> 3. Dense reconstruction with 3DGUT (MCMC densification)
> 4. Export to USDZ and import into Isaac Sim 5.0+
>
> **Key Features:**
> - Custom USD schema (OmniNuRecVolumeAPI) for Gaussian data
> - Physics proxy support via ground plane connection
> - Robot insertion and simulation directly in reconstructed environment
> - Pre-made scenes available on Hugging Face (NVIDIA Physical AI)
>
> **Performance:** 3DGRUT achieves 347-846 FPS on RTX 5090

## 7.3 Physics Proxy Generation

**Discovery: Physics Proxy Generation**
> Physics Proxy Generation for Gaussian Splatting (2025):
>
> 1. **PhysGaussian:** Uses Material Point Method (MPM), no meshing required - physics and rendering use same Gaussians
>
> 2. **GS-Verse (VR):** High-quality mesh as "geometry-aware proxy" for physics-aware VR interaction
>
> 3. **DecoupledGaussian (CVPR 2025):** Creates proxy points from depth maps for Newtonian physics simulation
>
> 4. **MaGS (ICCV 2025):** Mesh-adsorbed Gaussians - meshes deform via soft body simulation, Gaussians follow
>
> 5. **MILo (SIGGRAPH Asia 2025):** Mesh-In-the-Loop optimization, cleaner meshes for physics simulation
>
> **Current Gap:** Isaac Sim NuRec uses simple ground plane proxy. Complex collision geometry requires mesh extraction.

## 7.4 Real-Time Rendering

**Discovery: Real-Time GS Rendering Performance**
> Real-Time GS Rendering Performance (2025):
>
> **Web Implementations:**
> - antimatter15/splat: WebGL viewer (CPU sorting bottleneck)
> - kishimisu/Gaussian-Splatting-WebGL: WebGL2 with vertex shader bounding rects
> - GaussianSplats3D: Three.js implementation with .ksplat format for fastest loading
>
> **WebGPU Advantage:** ~2.1ms/frame for 6M points (0.58ms sorting, 1.52ms draw) - eliminates CPU bottleneck
>
> **Optimization Techniques:**
> - GPU radix sort (CUB) vs CPU async sort
> - 16-bit float covariances
> - Tile-based rendering
> - Level-of-detail (FLoD)

---

# 8. Emerging Methods (2025-2026)

## 8.1 3DGUT - Critical for Robotics

**Discovery: 3DGUT (CVPR 2025 Oral)**
> 3DGUT: 3D Gaussian Unscented Transform (CVPR 2025 Oral):
>
> **Problem Solved:** EWA splatting uses Jacobian approximation, causing errors with distorted cameras (fisheye, rolling shutter)
>
> **Solution:** Uses Unscented Transform (from Kalman Filter) - projects sigma points exactly, then re-estimates 2D Gaussian
>
> **Capabilities:**
> - Supports any nonlinear camera projection (fisheye, rolling shutter, time-dependent effects)
> - Secondary rays for reflections/refractions via hybrid rendering
> - Integrated into gsplat library (April 2025)
> - Free and open-source
>
> **Why Critical for Robotics:** Real robot cameras have lens distortion that standard GS cannot handle. 3DGUT solves this.

## 8.2 4D Gaussian Splatting

**Discovery: 4D Gaussian Splatting - Dynamic Scenes**
> 4D Gaussian Splatting - Dynamic Scenes (2025-2026):
>
> 1. **4D-GS (CVPR 2024 foundation):** 3D Gaussians + 4D neural voxels for holistic dynamic representation
>
> 2. **4DGS-1K (NeurIPS 2025):** Achieves **1000+ FPS** on modern GPUs
>
> 3. **MEGA (ICCV 2025):** Memory-efficient - 0.91M Gaussians vs 13M for same scene (only 7% of baseline)
>
> 4. **Native 4D Primitives:** Uses 4D Gaussians (anisotropic ellipses in space+time) + Spherindrical Harmonics for time-varying appearance
>
> 5. **Label-guided 4DGS (2026):** Fuses spatiotemporal features for dynamic object understanding

## 8.3 Language-Guided Editing

**Discovery: Language-Guided Gaussian Splatting**
> Language-Guided Gaussian Splatting (2025):
>
> 1. **LangSplat (CVPR 2024 Highlight):** 3D language fields with GS, **199x faster than LERF**. Uses SAM for hierarchical semantics.
>
> 2. **LangSplat V2 (NeurIPS 2025):** **450+ FPS** rendering
>
> 3. **4D LangSplat (CVPR 2025):** Extends to dynamic scenes using Multimodal LLMs for time-agnostic/sensitive queries
>
> 4. **Gen-LangSplat:** Generalized autoencoder pre-trained on ScanNet, no scene-specific training needed
>
> **Capabilities:** Open-vocabulary 3D object detection, semantic segmentation, natural language scene queries for task specification

## 8.4 Mesh Hybrids

**Discovery: Gaussian + Mesh Hybrids**
> Gaussian + Mesh Hybrids (2025-2026):
>
> 1. **MeGA (CVPR 2025):** Mesh-Gaussian head avatars - mesh for face, Gaussians for hair. Supports hair alteration/texture editing.
>
> 2. **MeshSplats:** Two-stage pipeline - ray tracing creates rough mesh, mesh guides Gaussian placement
>
> 3. **HybridSplat (Dec 2025):** Reflection-baked Gaussian tracing using hybrid splatting
>
> 4. **SDFoam:** SDF-biased Voronoi cells for metric-consistent isosurfaces, hybrid implicit-explicit
>
> 5. **Mani-GS:** Extracts triangular mesh proxy for physics-based manipulation (soft body, rigid body)

---

# 9. Final Decision & Recommendations

## 9.1 Final Decision

**Decision: Use gsplat + 3DGUT + NuRec pipeline for GS robotics simulation**

> **Rationale:** Research confirms this is the optimal stack:
> - gsplat provides 4x VRAM efficiency (critical for 16GB card)
> - 3DGUT handles real camera lens distortion (CVPR 2025 Oral, free in gsplat)
> - NuRec provides complete Isaac Sim integration
> - All major robotics companies (Covariant, Agility, Figure AI) use NVIDIA ecosystem
> - Physics proxy generation via MILo/SuGaR for collision meshes
> - glTF standardization (Aug 2025) and SPZ format ensure interoperability
>
> **Alternatives Considered:**
> 1. 3DGRUT standalone (slower, more features but overkill)
> 2. Commercial tools like Postshot/Polycam (easier but less control)
> 3. Unity/Unreal GS plugins (different ecosystem, harder Isaac Sim integration)

## 9.2 Recommended Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CAPTURE          POSES           TRAINING         CLEANUP      │
│  ───────          ─────           ────────         ───────      │
│  Smartphone   →   GLOMAP      →   gsplat       →   Clean-GS     │
│  or Gimbal        (fast)          (4x VRAM)        FlashSplat   │
│  60% overlap      COLMAP                           (30 sec)     │
│                   (compatible)                                  │
│                                                                 │
│  DISTORTION       MESH            EXPORT           SIMULATE     │
│  ──────────       ────            ──────           ────────     │
│  3DGUT        →   SuGaR/MILo  →   USDZ         →   Isaac Sim    │
│  (fisheye,        (physics        (NuRec)          5.0+         │
│  rolling          collision)                       + robot      │
│  shutter)                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 9.3 Key Decisions to Make

1. **Camera:** Smartphone (simple) vs dedicated camera with gimbal (better quality)?
2. **Physics:** Simple proxy (ground plane) vs full mesh collision (more work)?
3. **Dynamic scenes:** Static GS sufficient or need 4DGS?
4. **Semantics:** Need LangSplat for language queries?

## 9.4 Next Steps

| Priority | Task | Description |
|----------|------|-------------|
| 1 | Environment Setup | Install gsplat in `isaac` conda env |
| 2 | Test Pipeline | Capture test scene with smartphone, run full pipeline |
| 3 | Verify 3DGUT | Test with distorted camera footage |
| 4 | Isaac Sim Import | Export USDZ and import into Isaac Sim |
| 5 | Physics Proxy | Add collision mesh if needed |

---

# 10. Knowledge Graph Node Index

## 10.1 Constraints (User Requirements)

| ID | Content |
|----|---------|
| `e68bde41` | User prefers to use NVIDIA tools and stack as much as possible |
| `d2e3b99b` | User prefers gsplat/nerfstudio - open source, suitable for custom pipelines |

## 10.2 Key Decisions

| ID | Title | Date |
|----|-------|------|
| `2490bf2c` | Use gsplat + 3DGUT + NuRec pipeline for GS robotics simulation | 2026-01-27 |
| `7da322e0` | Use gsplat as primary Gaussian Splatting library (tentative) | 2026-01-27 |

## 10.3 Key Discoveries

| ID | Topic | Summary |
|----|-------|---------|
| `a6538427` | Isaac Sim Pipeline | Complete GS-to-Isaac Sim workflow confirmed |
| `0c87fcf5` | 3DGUT | Critical for robotics - handles camera distortion |
| `adb55295` | glTF Standard | Standardized Aug 2025, SPZ provides 90% compression |
| `24760df1` | Physics Gap | Current limitation - needs mesh extraction |
| `73a033d4` | NVIDIA Dominance | Isaac Sim is the dominant platform |
| `34730db5` | Cosmos WFMs | World Foundation Models for Physical AI |
| `bdffb5d1` | GS in Robotics | SplatSim, RL-GSBridge, GWM research |
| `c654b9f8` | Pose Estimation | GLOMAP 10-100x faster than COLMAP |
| `b0708b82` | gsplat | 4x less VRAM, 15% faster |
| `6d69556a` | Floater Removal | Clean-GS 60-80% compression |
| `6bbae802` | Segmentation | FlashSplat ~30 seconds |
| `5b77e665` | Mesh Extraction | SuGaR, OMeGa methods |
| `36c2f153` | 3DGUT Details | Unscented Transform for distorted cameras |
| `b7333ce2` | 4DGS | 1000+ FPS for dynamic scenes |
| `912cdbca` | LangSplat | 199x faster than LERF, 450+ FPS in V2 |

## 10.4 Analysis Thoughts

| ID | Topic |
|----|-------|
| `a8551640` | Research initialization - 6 areas defined |
| `048e19cc` | Library comparison - gsplat vs 3DGRUT vs original |
| `c881f177` | Tool landscape overview |
| `96d33f92` | Tech stack analysis - NVIDIA dominance |
| `00a7bb99` | GS opportunity at inflection point |
| `0cd5dc2b` | Paradigm shift to RL simulation training |
| `80ac3ba8` | Alternative tech stacks (Unity, MuJoCo, etc.) |
| `60b1b4c7` | Industry vertical insights |
| `02691bf4` | Tool comparison ranking |
| `e520358f` | Practical pipeline recommendation |
| `bbc03820` | Summary - what's production ready |

## 10.5 Project Structure

```
robotics (4537ffeb)
└── Digital Twin Creation (9739a104)
    ├── Tools Research (290994be) ✓ COMPLETED
    ├── Environment Setup (b47cc266)
    ├── Data Preparation (d064487d)
    ├── Pipeline Execution (0564fd5e)
    └── Export & Integration (c1201986)
```

---

*Document generated: January 27, 2026*
*Knowledge Graph Project ID: 4537ffeb-e483-4d91-987f-8bab60900c49*
*Total nodes in research session: 69*
