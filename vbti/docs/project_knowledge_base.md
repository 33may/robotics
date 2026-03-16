# VBTI Project — Complete Knowledge Base

**Author:** Anton Novokhatskyi
**Organization:** VBTI (robotics automation company) + Fontys University of Applied Sciences
**Date:** February 2026
**Status:** Active R&D — Phase 2 complete, Phase 3 (Sprint 3) in progress — real teleop data collection

---

## Table of Contents

1. [Who Is Anton](#1-who-is-anton)
2. [The Vision — Why This Matters](#2-the-vision--why-this-matters)
3. [The Problem We Solve](#3-the-problem-we-solve)
4. [The Solution — One Sentence](#4-the-solution--one-sentence)
5. [Business Context — VBTI](#5-business-context--vbti)
6. [Stakeholders & Collaboration](#6-stakeholders--collaboration)
7. [Technical Architecture — Full Pipeline](#7-technical-architecture--full-pipeline)
8. [Phase Breakdown — What Each Phase Does](#8-phase-breakdown--what-each-phase-does)
9. [Tools & Technology Stack](#9-tools--technology-stack)
10. [Current State — What Works Today](#10-current-state--what-works-today)
11. [Key Technical Decisions & Pivots](#11-key-technical-decisions--pivots)
12. [The Data Flow — End to End](#12-the-data-flow--end-to-end)
13. [Process Notes & Code Health](#13-process-notes--code-health)
14. [Research Direction — CoRL 2026](#14-research-direction--corl-2026)
15. [Literature & Prior Work](#15-literature--prior-work)
16. [Timeline & Milestones](#16-timeline--milestones)
17. [Metrics & Success Criteria](#17-metrics--success-criteria)
18. [Risks & Mitigations](#18-risks--mitigations)
19. [Glossary & Key Concepts](#19-glossary--key-concepts)
20. [Key Files & Entry Points](#20-key-files--entry-points)
21. [Presentation System Prompt](#21-presentation-system-prompt)

---

## 1. Who Is Anton

Anton Novokhatskyi is a Fontys ICT intern at VBTI, a robotics automation company. He has spent two years building in the robotics space — training policies at 3am, debugging sim-to-real transfer, and watching foundation models evolve from brittle tools into something with genuine capability.

### Personal Philosophy

Anton sees robotics through the lens of biology and evolutionary theory. Inspired by Dawkins' concept of memes as a second replicator alongside genes, he draws a direct parallel to robotics:

- **Behavioral Cloning = Genetic Evolution.** Pre-training builds a general-purpose "copying machine" — a model that understands how to move, perceive, and act. But without task-specific learning, it is like a human raised in isolation: genetically complete, memetically empty.
- **Reinforcement Learning = Memetic Evolution.** RL takes the general-purpose substrate and transmits specialized skills into it. Just as memetic evolution operates orders of magnitude faster than genetic evolution, RL in simulation operates orders of magnitude faster than real-world data collection.
- **The Simulation Environment = The Primeval Soup.** The medium in which this second evolution takes place.

### The Barbecue Moment

The conviction for 3D reconstruction came from a personal experience. At a backyard barbecue, Anton captured the scene on his phone — no tripod, no plan. When he ran the Gaussian Splatting reconstruction and shared it, people didn't say "that looks nice." They felt like they were there. A photograph shows you a scene. A spatial reconstruction gives you the scene. That reaction crystallized something: if we can reconstruct real environments with that fidelity, we can give machines something to reason about that actually resembles the world they need to operate in.

### Core Conviction

> "We have spent years building better bodies. We have spent years refining the algorithms. What we lack is the soul. Data is what contains compressed knowledge of how the physical world works."

Anton believes the data problem is the fundamental bottleneck in robotics. There is no "internet of robot demonstrations." Every demonstration requires a human physically moving a robot arm through a trajectory. Compute is cheap. Storage is free. Human time is the constraint that refuses to move. Simulation-generated data is the way to break this bottleneck.

---

## 2. The Vision — Why This Matters

### The Big Picture

Robotics is about to change — not eventually, but soon. Models that understand 3D space, simulation that transfers to real hardware, robots you can actually afford to break while learning. Foundation models are beginning to control physical arms with the kind of fluency that LLMs brought to text in 2022.

### What We Are Building

A pipeline that takes a camera scan of any real environment and produces something a robot can physically interact with — push, grasp, knock over, and learn from. This means:

1. **Extracting collision geometry** from 3D reconstructions
2. **Estimating physics properties** — which surfaces are slippery, which objects are heavy
3. **Segmenting individual objects** so they can be manipulated independently
4. **Generating variations** — different positions, lighting, sizes — so the robot learns the task itself, not one arrangement

### The Economics Argument

Right now, creating a training scene for a new task takes weeks of manual work by simulation engineers. If an automated pipeline can produce something good enough, the economics of the entire field change. You don't need an army of simulation engineers. You need a camera and a few minutes.

And if you can generate one scene, you can generate thousands. Each variation is another training episode. Each episode teaches the policy something slightly different. The real world becomes the final exam, not the classroom.

---

## 3. The Problem We Solve

### Industry Perspective

Deploying robotic solutions at client sites currently requires significant on-site time. Engineers must:
1. Visit the client location to collect data using a physical robot
2. Return to the office to train models
3. Verify results in a simplified in-house setup that doesn't accurately represent the client's environment

This process is time-consuming, costly, and limits iteration speed. Any mismatch between the testing environment and the real deployment scene leads to unpredictable performance.

### Academic Perspective

In robotics learning, data is the fundamental bottleneck. Behavioral cloning depends on real-world demonstrations which are expensive and limited. Scaling model performance beyond what small datasets allow demands disproportionate effort.

### The Specific Limitation

We already have a working solution using classic behavioral cloning that achieves ~80% task success rate. Scaling above this accuracy takes exponentially more effort in data collection. The goal is to break through this ceiling by designing environments where computation can replace human data collection.

---

## 4. The Solution — One Sentence

**Develop a framework that reconstructs real-world environments into physically-based simulations to enable scalable data generation and continuous learning for robotics applications.**

### Expanded

Take a phone video of any real workspace. Reconstruct it as a 3D Gaussian Splat. Extract collision meshes. Estimate physics. Import into NVIDIA Isaac Sim. Train a robot policy with reinforcement learning inside this digital twin. Deploy to the real robot. Measure the improvement over behavioral cloning alone.

---

## 5. Business Context — VBTI

### What Is VBTI

VBTI is a robotics automation company that provides solutions to industrial clients. They build and deploy robot systems for real-world tasks.

### The Business Problem

Every new client site means:
- Weeks of manual setup to create simulation environments
- On-site data collection with expensive engineer time
- Models that don't generalize beyond the exact conditions they were trained in
- Multiple rounds of on-site visits to fix deployment issues

### The Business Value of This Project

1. **Reduce on-site time** — Scan the workspace once, develop in simulation
2. **Faster iteration** — Test ideas in hours instead of weeks
3. **Better models** — RL in simulation can exceed what BC alone achieves
4. **Scalable infrastructure** — Same pipeline works for any new client, any new task
5. **Competitive advantage** — Early adoption of GS-based digital twins before it becomes industry standard

### The PoC Task

Tomato sorting — a real agricultural manipulation task that VBTI works on. The framework will be validated on this task, comparing BC-only baseline against simulation-enhanced models.

## 7. Technical Architecture — Full Pipeline

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    REAL WORLD CAPTURE                            │
├─────────────────────────────────────────────────────────────────┤
│  iPhone video / photos of workspace                             │
│  ↓                                                              │
│  master.py video_processing → sharp-frame-extractor (best       │
│  frames)                                                        │
│  ↓                                                              │
│  COLMAP (Structure from Motion) → camera poses + sparse cloud   │
│  ↓ (exhaustive matching for <500 images, vocab tree for more)   │
│  ↓                                                              │
│  Nerfstudio splatfacto → Gaussian Splat (PLY)                   │
│  ↓                                                              │
│  SuperSplat → clean floating artifacts (optional)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MESH EXTRACTION                               │
├─────────────────────────────────────────────────────────────────┤
│  COLMAP undistort images (OPENCV → PINHOLE)                     │
│  ↓                                                              │
│  MILo (SIGGRAPH Asia 2025) — learnable SDF from GS             │
│  ↓ Train: ~57 min on RTX 4070 Ti SUPER                          │
│  ↓ Quality: PSNR 31.2, SSIM 0.89, 243K Gaussians               │
│  ↓                                                              │
│  mesh_extract_sdf.py → mesh_learnable_sdf.ply                   │
│  ↓                                                              │
│  clean_mesh.py (Polyscope GUI) → crop OBB, remove artifacts     │
│  ↓                                                              │
│  PCA-based alignment + center-at-origin                         │
│  ↓                                                              │
│  Decimate for collision (configurable simplification)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SCENE COMPOSITION                             │
├─────────────────────────────────────────────────────────────────┤
│  create_scene_usd.py:                                           │
│  ├── MILo mesh → USD with vertex colors (sRGB→Linear)           │
│  ├── Collision mesh (decimated) with physics material            │
│  ├── HDRI dome light (polyhaven.com)                             │
│  ├── Distant light with configurable angle                      │
│  └── Interactive objects (GLB→USD via format_utils.py)           │
│                                                                 │
│  Isaac Sim GUI:                                                 │
│  ├── Place SO-101 robot with drives configured                  │
│  ├── Position cameras (side_cam, table_cam, gripper_cam)        │
│  └── Save as scene_v3.usda                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LEISAAC ENV GENERATION                        │
├─────────────────────────────────────────────────────────────────┤
│  robot_utils.py pipeline command (single CLI):                   │
│  ├── Step 1: create_no_robot_scene() → remove robot + lights    │
│  ├── Step 2: generate_scene_asset() → register in leisaac       │
│  ├── Step 3: create_task_boilerplate() → gym.register + mdp     │
│  └── Step 4: generate_env_cfg() → full Python env config        │
│              ├── SceneCfg: scene USD + cameras + lights          │
│              ├── ObservationsCfg: joints + 3 camera images       │
│              ├── TerminationsCfg: timeout                        │
│              └── EnvCfg: robot pos + subasset auto-discovery     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                               │
├─────────────────────────────────────────────────────────────────┤
│  Teleoperation with physical SO-101 leader arm:                  │
│  python teleop_se3_agent.py \                                    │
│    --task LeIsaac-SO101-VbtiMeshTable-v0 \                       │
│    --teleop_device so101leader \                                 │
│    --enable_cameras                                              │
│                                                                 │
│  Controls: B=start, N=success+reset, R=discard+reset            │
│  Output: HDF5 dataset (joints + camera images per timestep)     │
│  Domain randomization: objects, lighting, cameras, physics       │
│  100+ episodes collected, ~85 GB uncompressed                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA AUGMENTATION (Cosmos Transfer 2.5)       │
├─────────────────────────────────────────────────────────────────┤
│  cosmos_transfer.py:                                            │
│  1. extract  → HDF5 episode → PNG frames (per camera)           │
│  2. process  → frames → MP4 (RGB + depth + edge)                │
│  3. config   → generate Cosmos spec JSON                        │
│  4. transfer → run inference (RunPod A40, ~5 min/93-frame chunk)│
│  5. reassemble → augmented frames → HDF5                        │
│                                                                 │
│  Controls: depth weight 0.5, edge weight 0.3-1.0               │
│  NVIDIA benchmark: 54% → 91% (+68.5%) with mixed data          │
│  Deployment: RunPod A40 48GB, ~$0.76/hr, 136h for 109 episodes │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    POLICY TRAINING                               │
├─────────────────────────────────────────────────────────────────┤
│  Behavioral Cloning:                                            │
│  ├── Convert HDF5 → LeRobot dataset format                      │
│  ├── Train SmolVLA (vision-language-action model)                │
│  ├── Preprocessor normalizes (mean/std from dataset)             │
│  └── Saves: policy + preprocessor + postprocessor                │
│                                                                 │
│  Reinforcement Learning (next phase):                           │
│  ├── Isaac Lab parallel environments                            │
│  ├── Reward function engineering                                │
│  └── Continue training from BC checkpoint                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE / DEPLOYMENT                        │
├─────────────────────────────────────────────────────────────────┤
│  run_smolvla_inference.py:                                       │
│  1. Load policy + preprocessor + POSTPROCESSOR                   │
│  2. Get observations from Isaac (radians, uint8 images)          │
│  3. Convert images: uint8→float, NHWC→NCHW                      │
│  4. Apply preprocessor (normalize state)                         │
│  5. policy.select_action() → normalized actions                  │
│  6. Apply postprocessor → degrees                                │
│  7. Convert degrees→radians (* π / 180)                          │
│  8. env.step(radians)                                            │
│                                                                 │
│  Real robot deployment via lerobot inference                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Phase Breakdown — What Each Phase Does

### Phase 1: Behavioral Cloning & Baseline (Feb 3-20)

**Goal:** Establish a working hardware baseline.

- Configure SO-101 robot arm (6-DOF + gripper, 7 joints)
- Collect real-world demonstrations using leader arm
- Train SmolVLA using LeRobot framework
- Evaluate real-world task success rate → establish ~80% baseline
- This is the reference point against which all improvements are measured

**Status:** COMPLETED

### Phase 2: Digital Twin Creation (Feb 9 - Feb 27)

**Goal:** Construct a high-fidelity digital twin of the real workspace.

Sub-tasks completed:
- Scene reconstruction: Video → COLMAP → Gaussian Splatting (nerfstudio splatfacto)
- **Pivot (Feb 13):** NuRec GS incompatible with TiledCamera → switched to MILo mesh extraction
- Mesh extraction via MILo (PSNR 31.2, SSIM 0.89, 243K Gaussians)
- Interactive mesh cleaning tool (Polyscope GUI with OBB cropping)
- sRGB→Linear color space conversion for Isaac Sim rendering
- GLB→USD asset pipeline with materials and physics (SAM3D object extraction)
- Scene composition with custom HDRI (phone panorama → Hugin → EXR)
- USDA→LeIsaac automated pipeline (`robot_utils.py pipeline` single CLI)
- Parallel IsaacLab standalone export (`isaac_cfg_utils.py generate_isaaclab_env`)
- Teleoperation working with physical leader arm
- Domain randomization configured (objects, lighting, cameras, physics)
- 100+ demonstration episodes collected in HDF5 format
- Cosmos Transfer 2.5 data preparation pipeline built (`cosmos_transfer.py`)

**Status:** COMPLETED

### Phase 3: Real Data Collection + Cosmos Augmentation (Mar 1 - ongoing)

**Goal:** Collect real-world teleop data and augment sim data with Cosmos Transfer.

Sprint 3 (current):
- Real hardware setup: SO101 leader (/dev/ttyACM2) + follower (/dev/ttyACM1)
- 4× RealSense D405 cameras (640×480@15fps, USB 2.1 bandwidth limit)
- Calibration debugging (Koen's offsets restored after experimental recalibration)
- Cosmos Transfer running on RunPod (A40 GPU, ~$0.76/hr, 136 hrs for 109 episodes)
- Master pipeline CLI (`master.py`) for end-to-end automation
- Dataset viewer TUI for episode inspection

**Status:** IN PROGRESS

### Phase 4: Model Training + RL (Mar 15 - Mar 30)

**Goal:** Train SmolVLA on augmented dataset, then use RL to improve.

- SmolVLA training on mixed original + Cosmos-augmented data
- Reward engineering for task completion
- Parallel RL training in Isaac Lab
- Metric recording: task success rate in simulation vs BC baseline

**Status:** UPCOMING

### Phase 5: Sim-to-Real Transfer & Validation (Apr 1 - Apr 15)

**Goal:** Deploy simulation-improved model to physical robot.

- Run optimized model on real SO-101
- Validate camera input similarity (sim vs real)
- Record final task success rate
- Calculate pure method gains (Phase 5 - Phase 1)
- Gap analysis if transfer degrades performance

**Status:** PLANNED

### Phase 6: Infrastructure Scaling (Apr 16 - Jun 30)

**Goal:** Solidify pipeline for real production use.

- Reusable scripts and tools from Phases 1-5
- Scale to real agricultural manipulation (tomato sorting)
- Shift from training-from-scratch to utilizing foundation models
- Deliverable: working PoC + complete documented pipeline

**Status:** PLANNED

---

## 9. Tools & Technology Stack

### Simulation & Physics
| Tool | Purpose |
|---|---|
| **NVIDIA Isaac Sim 5.0+** | Physics simulation engine (PhysX), USD-based scene management |
| **NVIDIA Isaac Lab** | RL training framework, parallel environments |
| **LeIsaac** | Custom wrapper for Isaac Lab — SO-101 task registration, teleop, data collection |
| **PhysX** | Physics solver (gravity, collisions, friction, joint dynamics) |

### 3D Reconstruction & Mesh
| Tool | Purpose |
|---|---|
| **Nerfstudio + splatfacto** | Gaussian Splatting reconstruction from video/photos |
| **COLMAP** | Structure from Motion — camera pose estimation |
| **MILo** (SIGGRAPH Asia 2025) | Differentiable mesh extraction from GS (SOTA quality) |
| **SuperSplat** | Browser-based GS editing and cleaning |
| **Polyscope** | Interactive mesh cleaning GUI (custom tool built in project) |
| **Open3D** | Point cloud processing, Poisson reconstruction |

### Format Conversion & Scene Authoring
| Tool | Purpose |
|---|---|
| **3DGRUT** (NVIDIA) | PLY→USDZ conversion for NuRec neural rendering (legacy — NuRec path abandoned) |
| **USD (Universal Scene Description)** | Scene format for Isaac Sim |
| **Custom create_scene_usd.py** | MILo mesh→USD with vertex colors + collision |
| **Custom format_utils.py** | GLB→USD with materials + physics |
| **Custom robot_utils.py** | USDA→LeIsaac pipeline automation |
| **Custom isaac_cfg_utils.py** | IsaacLab standalone env generation (no LeIsaac dependency) |
| **Custom master.py** | Pipeline orchestrator CLI (video→GS→mesh→USD→env) |

### Data Augmentation
| Tool | Purpose |
|---|---|
| **NVIDIA Cosmos Transfer 2.5** | Sim→photorealistic video augmentation (2.36B params, BF16) |
| **Custom cosmos_transfer.py** | HDF5→frames→MP4→spec JSON preparation pipeline |
| **RunPod** | Cloud GPU for Cosmos inference (A40 48GB, ~$0.76/hr) |

### Training & Inference
| Tool | Purpose |
|---|---|
| **LeRobot** (HuggingFace) | Behavioral cloning framework, dataset management |
| **SmolVLA** | Small Vision-Language-Action model (policy network) |
| **Custom run_smolvla_inference.py** | Isaac Sim inference loop with proper unit conversion |
| **Custom dataset_utils.py** | HDF5 dataset inspection, video grid export |

### Hardware
| Component | Specs |
|---|---|
| **Robot** | SO-ARM101 leader + follower (6-DOF arm + gripper, 7 joints) |
| **Cameras** | 4× Intel RealSense D405 (640×480@15fps, USB 2.1) |
| **GPU** | NVIDIA RTX 4070 Ti SUPER (16GB VRAM, SM 89) |
| **OS** | Fedora 42 (Linux 6.18) |
| **CUDA** | 12.9 (patched for glibc 2.41 compatibility) |
| **GCC** | 14 (system GCC 15 unsupported by CUDA) |

---

## 10. Current State — What Works Today

### Completed & Proven

1. **BC baseline with SmolVLA** — ~80% task success rate on real robot
2. **Full GS reconstruction pipeline** — Video → COLMAP → splatfacto → PLY → cleaned splat
3. **MILo mesh extraction** — PSNR 31.2, SSIM 0.89, high-quality textured mesh
4. **Interactive mesh cleaning tool** — Polyscope GUI with OBB cropping, artifact removal, decimation control
5. **Scene composition** — MILo mesh + collision geometry + HDRI lighting + interactive objects in USD
6. **USDA→LeIsaac automated pipeline** — Single CLI command: `python robot_utils.py pipeline scene.usda task_name`
7. **Teleoperation working** — Physical SO-101 leader arm controlling simulated robot
8. **Data collection** — 100+ demonstration episodes collected with 3 cameras (side, table, wrist) in HDF5
9. **SmolVLA inference in Isaac Sim** — Debugged: postprocessor loading, degree→radian conversion, image format conversion
10. **GLB→USD asset pipeline** — SAM3D objects converted with materials, physics, collision
11. **Domain randomization** — Object positions/rotations, lighting, camera jitter, physics properties randomized per reset
12. **Cosmos Transfer data prep** — `cosmos_transfer.py` extracts HDF5 → frames → MP4 + generates spec JSON
13. **Master pipeline CLI** — `master.py` orchestrates video_processing → gs_reconstruction → ply_to_usda → scene_composition
14. **Real hardware setup** — SO101 leader+follower arms, 4× RealSense D405 cameras configured

### Key Results

| Metric | Value | Context |
|---|---|---|
| MILo PSNR | 31.2 dB | High reconstruction fidelity (target >30) |
| MILo SSIM | 0.89 | Close to target of 0.9 |
| MILo LPIPS | 0.17 | Perceptual similarity |
| MILo Gaussians | 243K | Table scene |
| BC success rate | ~80% | Real-world baseline on SO-101 |
| MILo train time | 57 min | On RTX 4070 Ti SUPER |
| Pipeline time | ~1 hour | Video → runnable LeIsaac task |
| Episodes collected | 100+ | With domain randomization, 3 cameras |
| Dataset size | ~85 GB | Uncompressed HDF5, 480×640 RGB+depth+seg |
| Cosmos aug cost | ~$103 | 136 hrs × $0.76/hr for 109 episodes |

---

## 11. Key Technical Decisions & Pivots

### Decision 1: NuRec → MILo Mesh (Feb 13)

**Problem:** Gaussian Splat (NuRec USDZ) + TiledCamera = hang forever. NuRec primitives are not rendered in synthetic data passes. TiledCamera uses Replicator render products = synthetic data passes → NuRec invisible → hangs.

**Viewport worked fine** (RTX renderer supports NuRec). Only the data pipeline was broken.

**Decision:** Pivot to mesh-based approach using MILo. This also fixed shadow catcher issues we had with NuRec volumes.

**Lesson:** Be agile — when a blocker appears (NVIDIA closed-source limitation), pivot to what works rather than trying to bridge unsupported features.

### Decision 2: Never Scale Articulated Robots (Feb 12)

**Problem:** SO-101 robot at 10x scale had broken physics — lower pan joint falling from orbit, slow/strange movements.

**Root cause:** PhysX simulates in meters. Scaling a rigid body 10x grows geometry but mass/inertia stay the same. Joint stiffness can't hold giant links. Solver iterations can't converge.

**Solution:** Fix the scene units instead of scaling the robot. Always work at native scale.

### Decision 3: sRGB→Linear Color Space (Feb 16)

**Problem:** MILo mesh materials showing white in Isaac Sim.

**Root cause:** Vertex colors from MILo are in sRGB space. RTX renderer (PhysX) works in linear space. A midtone sRGB value of 0.5 is actually linear ~0.214. Without conversion, everything looks ~2x too bright, washing out to white.

**Solution:** Added sRGB→Linear conversion in the mesh import pipeline.

### Decision 4: Spawn=None for Scene Cameras (Feb 12-13)

**Problem:** How to load cameras positioned in the Isaac Sim GUI into LeIsaac code.

**Solution:** Keep cameras in scene USD, reference with `spawn=None` — LeIsaac uses existing camera prims instead of creating new ones. Only gripper camera (which moves with robot) needs to be spawned separately.

### Decision 5: NVIDIA Native Stack Throughout (Feb 3)

**Rationale:** NVIDIA is the leading institution for embodied AI tooling. Keeping the entire stack native (Isaac Sim, Isaac Lab, CUDA, USD) prevents integration issues when transitioning between project phases. This matters because the project spans reconstruction → simulation → training → deployment.

### Decision 6: Exhaustive Matching for Small Datasets (Feb 9)

**Problem:** COLMAP vocab tree format changed (flann → faiss). Nerfstudio downloads old version, crashes on newer COLMAP.

**Solution:** Use exhaustive matching for datasets <500 images (no vocab tree needed). For larger sets, replace cached vocab tree with faiss version.

### Decision 7: Instance Proxies Deferred (Feb 12-16)

**Problem:** Instance proxies optimize VRAM for multi-env cloning but make material editing, camera placement, and scene iteration difficult.

**Solution:** Use non-instanced copies for composition and initial development. Switch to instanced versions only for parallel RL training.

### Decision 8: Cosmos Transfer — Depth+Edge Only (Feb 20)

**Problem:** Segmentation renders black in Cosmos Transfer 2.5 (known limitation).

**Decision:** Use only depth (weight 0.5) + edge (weight 1.0) controls. This preserves 3D structure while allowing appearance transformation.

---

## Known Technical Issues

### PhysX Replay Nondeterminism

PhysX replay from in-contact states is nondeterministic. Some episodes (e.g., ep33) show different trajectories on replay vs original recording. State-driven replay didn't fix it. **Impact:** Affects Cosmos augmentation pipeline (needs deterministic replay to capture depth/seg from same trajectory). **Workaround:** Best-effort — some episodes work (ep59, ep75), others drift.

### USB Bandwidth Contention

Multiple RealSense cameras on USB 2.1 contend for bandwidth. 4 cameras @30fps exceeds limits. **Solution:** 4 cameras @15fps or 2 cameras @30fps. May need `usbfs_memory_mb` tuning.

---

## 12. The Data Flow — End to End

### Training Data Format (LeRobot)

```
Dataset (LeRobot HDF5):
├── observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] (DEGREES)
├── observation.images.side_cam: (H, W, 3) uint8
├── observation.images.table_cam: (H, W, 3) uint8
├── observation.images.wrist_cam: (H, W, 3) uint8
└── action: [6 joint positions] (DEGREES)
```

### Raw Collection Format (Isaac Sim HDF5)

```
data/
  demo_000/
    obs/
      side_cam_rgb        (T, H, W, 3) uint8     — 480×640 RGB
      side_cam_depth      (T, H, W) float32       — depth in meters
      side_cam_seg        (T, H, W, 4) uint8      — RGBA segmentation
      table_cam_rgb       (T, H, W, 3) uint8
      table_cam_depth     (T, H, W) float32
      table_cam_seg       (T, H, W, 4) uint8
      wrist_rgb           (T, H, W, 3) uint8
      wrist_depth         (T, H, W) float32
      wrist_seg           (T, H, W, 4) uint8
      joint_pos           (T, 6) float32           — radians
      joint_vel           (T, 6) float32
    actions               (T, 6) float32           — radians
```

**Episode sizes:** ~294–426 frames per camera. **Total:** ~85 GB uncompressed for 100+ episodes.

### Normalization Pipeline

```
Training:
  Raw degrees → Preprocessor (normalize with mean/std) → Model learns NORMALIZED actions
  Saves: policy_preprocessor.json + policy_postprocessor.json

Inference:
  Isaac observations (radians) → Convert to degrees → Preprocessor (normalize) → Model → Postprocessor (denormalize) → Degrees → Convert to radians → env.step()
```

### Unit Conversion (Critical)

```python
# Training data stored in DEGREES
# Isaac Sim operates in RADIANS

# Isaac → Dataset: radians * (180 / π) = degrees
# Dataset → Isaac: degrees * (π / 180) = radians

# Action mean (degrees): [2.55, 13.68, -19.39, 76.21, 4.79, 23.37]
# Action std (degrees):  [8.72, 24.19, 33.97, 18.91, 11.25, 16.16]
```

### Image Processing

```python
# Isaac returns: (N, H, W, C) uint8 [0-255]
# Policy expects: (N, C, H, W) float32 [0-1]

# NHWC → NCHW: img.permute(0, 3, 1, 2)
# uint8 → float: img.float() / 255.0
```

---

## 13. Process Notes & Code Health

### Things That Need Attention

1. **`cosmos_transfer.py transfer()` calls Transfer 1 API**, not 2.5 — endpoint is `nvidia/cosmos-transfer1-7b`. The actual 2.5 deployment on RunPod uses the `cosmos-transfer2.5` repo directly. Script needs updating to match.

2. **`cosmos_transfer.py reassemble()` is NOT IMPLEMENTED** — marked TODO. Currently there's no code path to write Cosmos output frames back into HDF5.

3. **Domain randomization is commented out** in `vbti_so_v1_env_cfg.py` — the imports and wiring exist (`leisaac.utils.domain_randomization`), but the actual randomization events are not active. Need to uncomment/enable per task variant.

4. **Missing HDF5→LeRobot format converter** — Training uses LeRobot datasets but collection produces HDF5. There's no automated converter in the codebase. The existing 51-episode real dataset (`may33/so101_pick_place`) was manually converted.

5. **Robot joint initialization hardcoded to zero** in `generate_isaaclab_env()` — doesn't use the joint_targets from scene_config.json.

6. **`LEISAAC_ROOT` hardcoded** in `isaac_cfg_utils.py` to `/home/may33/projects/ml_portfolio/robotics/leisaac` — not portable.

7. **format_utils.py: no texture support** on vertex-colored meshes — only displayColor primvar, no UV-mapped textures.

### Dataset Discrepancy

The Obsidian notes mention **217 episodes / 154 GB** in `vbti_so_v1_mix_v1.hdf5` (the full sim dataset with DR), while the knowledge base previously said "100+ episodes / 85 GB". Both may be accurate for different dataset versions:
- `vbti_so_v1_mix_v1.hdf5`: 217 episodes, 154 GB (full DR dataset)
- `vbti_table_v2_cosmos/raw.hdf5`: 3 episodes (Cosmos prep subset)

### Cross-Module Data Contracts

| From | To | Format | Units |
|------|-----|--------|-------|
| Isaac Sim collection | HDF5 | uint8 images + float32 joints | **radians** |
| HDF5 | LeRobot dataset | Parquet + video | **degrees** (converted) |
| LeRobot dataset | SmolVLA training | Normalized tensors | **normalized** (mean/std) |
| SmolVLA inference | Postprocessor | Normalized actions | **normalized** → **degrees** |
| Postprocessor output | Isaac Sim env.step | float32 | **degrees** → **radians** |
| HDF5 | Cosmos extract | PNG frames + depth NPY | **uint8 RGB** + **float32 meters** |
| Cosmos output | Reassemble | MP4 video | **uint8 RGB** |

### Key Magic Numbers

| Value | Where | Why |
|-------|-------|-----|
| `stiffness=17.8, damping=0.60` | robot_utils.py | Matches LeIsaac ImplicitActuatorCfg defaults |
| `SH_C0 = 0.28209` | format_utils.py | Spherical harmonics coefficient 1/(2√π) |
| `COLMAP_TO_USD` matrix | format_utils.py | x→-x, y→-z, z→-y coordinate transform |
| `depth_min=0.01, depth_max=2.0` | cosmos_transfer.py | Depth normalization range (meters) |
| `Canny (50, 150)` | cosmos_transfer.py | Edge detection thresholds |
| `opacity > 0.5` | format_utils.py | GS point cloud opacity filter threshold |
| `chunk_size=50` | train_smolvla_custom.py | SmolVLA predicts 50 future actions |
| `CRF 18` | video_utils.py | ffmpeg quality for rotation fix |
| `18.1mm focal length` | vbti_so_v1_env_cfg.py | Camera focal length in sim |
| `25s episode` | env_cfg template | Termination timeout |

### Observation Specification (Isaac Sim → Policy)

```
Isaac Sim env.get_observations() returns:
  "policy": {
    "joint_pos":      (6,)          float32 radians
    "joint_vel":      (6,)          float32 rad/s
    "joint_pos_rel":  (6,)          float32 radians (relative to init)
    "joint_vel_rel":  (6,)          float32 rad/s
    "actions":        (6,)          float32 (last action)
    "cam_top":        (480, 640, 3) uint8
    "cam_right":      (480, 640, 3) uint8
    "cam_left":       (480, 640, 3) uint8
    "wrist":          (480, 640, 3) uint8
  }

SmolVLA policy expects:
  "observation.state":                 float32 DEGREES
  "observation.images.front":          (C, H, W) float32 [0,1]
  "observation.images.third_person":   (C, H, W) float32 [0,1]
  "observation.images.gripper":        (C, H, W) float32 [0,1]
  "task":                              string
```

---

## 14. Research Direction — CoRL 2026

### Target Conference

**CoRL 2026** (Conference on Robot Learning) — deadline May 28, 2026

### Central Research Question

> Can a Gaussian Splatting scan of a real workspace be automatically transformed into a fully interactive simulation environment — with per-object collision meshes, estimated physics properties, and generative scene variations — that produces manipulation policies transferable to the real world?

### The Gap Nobody Has Filled

Every component exists in isolation. Nobody has connected them end-to-end and measured whether the result supports contact-rich robot learning:

| What Exists | What's Missing |
|---|---|
| GS visual augmentation: 86-88% sim2real (SplatSim, RoboSplat) | Visual-only — no collision, grasping, force feedback |
| LangSplatV2 segments GS at 384 FPS (NeurIPS'25) | Never used for per-object mesh extraction in robotics |
| MILo extracts meshes 10x more efficiently (SIGGRAPH Asia'25) | Never evaluated for physics simulation quality |
| PhysSplat estimates materials via VLM zero-shot (ICCV'25) | Uses custom MPM solver, not PhysX |
| PhysX-Anything generates sim-ready URDF (Nov'25) | No grasp success evaluation |
| World Labs Marble exports to Isaac Sim (Jan'26) | No per-object physics, coarse geometry |

### Three Research Branches

**Branch 1: GS Editing as Structured Domain Randomization**
- Does physics-synchronized GS domain randomization outperform visual-only GS augmentation for contact-rich manipulation?

**Branch 2: Semantic-Physical Scene Decomposition** (CORE)
- Can MILo-extracted meshes serve as PhysX-compatible collision proxies?
- First empirical study of GS-extracted mesh quality for contact simulation

**Branch 3: Compositional Object Generation**
- Can text-prompted generative 3D assets be automatically made sim-ready for manipulation?

### The Universal Bottleneck

**Mesh quality** appears in ALL three branches. Whether editing GS scenes, decomposing them into objects, or generating new objects — the collision mesh must be good enough for PhysX contact simulation. This is the single most important technical question.

### Expected Contributions

1. First empirical study of GS-extracted mesh quality for contact simulation in PhysX
2. First automated end-to-end pipeline from GS scan to interactive training environment
3. Quantified comparison of automated GS-based scenes vs hand-authored simulation
4. Scaling analysis of generative scene variation vs policy robustness

---

## 14. Literature & Prior Work

### Landscape (40+ papers reviewed)

| Paper | Venue | Key Finding | Limitation |
|---|---|---|---|
| **SplatSim** | CMU 2024 | GS rendering for sim2real, 86.25% zero-shot | Visual-only, no physics interaction |
| **RoboSplat** | RSS 2025 | GS augmentation, 87.8% one-shot | Admits failure on contact tasks |
| **RoboSimGS** | Oct 2025 | Hybrid GS+mesh with VLM physics | Slow pipeline, no DR framework |
| **PhysSplat** | ICCV 2025 | Zero-shot physics via MLLM | Custom MPM solver, not PhysX |
| **MILo** | SIGGRAPH Asia 2025 | 10x fewer vertices mesh from GS | Never tested for physics sim |
| **PhysX-Anything** | Nov 2025 | Single image → sim-ready URDF | No grasp evaluation |
| **PhysXGen** | NeurIPS 2025 Spotlight | Physics-grounded 3D generation | — |
| **LangSplatV2** | NeurIPS 2025 | 384 FPS semantic GS segmentation | Semantic-only, not instance |
| **World Labs Marble** | Jan 2026 | Generative 3D worlds + Isaac Sim | Coarse geometry |
| **Phys2Real** | 2025 | Physics-informed > DR for pushing | Only 2 trivial tasks tested |

### Our Positioning

We are **not** inventing new algorithms. We are connecting existing tools into a pipeline that nobody has built end-to-end, and measuring whether the result is good enough for real robot learning. The key contribution is empirical validation, not theoretical novelty.

---

## 16. Timeline & Milestones

**Project duration:** February 1 – June 30, 2026

| Phase | Dates | Status | Key Deliverable |
|---|---|---|---|
| Phase 1: BC Baseline | Feb 3-20 | DONE | ~80% success rate baseline |
| Phase 2: Digital Twin | Feb 9 - Feb 27 | DONE | Working sim environment + 100+ episodes + DR |
| Phase 3: Real Data + Cosmos | Mar 1 - ongoing | IN PROGRESS | Real teleop data + Cosmos-augmented dataset |
| Phase 4: Model Training + RL | Mar 15 - Mar 30 | UPCOMING | SmolVLA on augmented data + RL improvement |
| Phase 5: Sim2Real | Apr 1-15 | PLANNED | Real-world validation + method gains |
| Phase 6: Infrastructure | Apr 16 - Jun 30 | PLANNED | Reusable pipeline + tomato sorting PoC |

### Research Timeline (CoRL 2026)

| Phase | Weeks | Dates | Deliverable |
|---|---|---|---|
| Mesh quality threshold study | 1-4 | Feb 6 - Mar 7 | Go/no-go on mesh extraction |
| Full pipeline + DR comparison | 5-10 | Mar 7 - Apr 18 | Pipeline vs hand-authored baseline |
| Generative variation experiment | 11-14 | Apr 18 - May 10 | Parametric diversity ablation |
| Paper writing | 14-16 | May 10 - May 28 | CoRL submission |

---

## 16. Metrics & Success Criteria

### Primary Metric
**Task success rate improvement:** Compare BC-only baseline (~80%) against simulation-enhanced model. The difference = pure method gains.

### Secondary Metrics
| Metric | Target | Purpose |
|---|---|---|
| PSNR | >30 dB | Reconstruction quality |
| SSIM | >0.9 | Structural similarity |
| LPIPS | <0.1 | Perceptual similarity |
| Pipeline time | <2 hours | Video → runnable task |
| Data efficiency | Reduce real-world collection by >50% | Business value |
| Iteration speed | Same-day turnaround | Development velocity |

### How We Measure Success
- All comparisons use the same task, robot, and environment
- Experimental setup remains constant to isolate the effect of the simulation framework
- Results documented for reproducibility

---

## 17. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 3D reconstruction insufficient fidelity | Sim-trained models don't transfer | Iterate on quality; supplement with domain randomization |
| Physics simulation doesn't match real objects | Grasping fails in transfer | Tune mass/friction/softness; use VLM physics estimation |
| Sim-to-real gap too large | Model degrades on real robot | Domain randomization across world parameters; fine-tune with minimal real data |
| Action space too large for RL | Training doesn't converge | Constrain action space; collaborate on reward shaping |
| Results indifferent or worse than BC-only | Hypothesis not validated | Document as negative result; analyze which components failed |
| Mesh quality insufficient for grasping | Pipeline fails at foundation | This is the universal gate — evaluating first |
| Compute constraints limit training scale | Can't run enough parallel envs | Optimize scene complexity; prioritize critical scenarios |
| PhysX replay nondeterminism | Cosmos augmentation pipeline can't capture consistent depth/seg | Best-effort per episode; some work, some drift |
| Cosmos augmentation cost | ~$103 per full dataset pass | Batch processing on RunPod; optimize control weights to minimize re-runs |

---

## 18. Glossary & Key Concepts

| Term | Definition |
|---|---|
| **Behavioral Cloning (BC)** | Imitation learning — train a policy to replay demonstrated trajectories |
| **Gaussian Splatting (GS)** | 3D scene representation using collections of 3D Gaussians — continuous, differentiable, real-time rendering |
| **Digital Twin** | A simulation environment that matches a real-world workspace in geometry, physics, and appearance |
| **Sim-to-Real Transfer** | Deploying a simulation-trained model to a physical robot |
| **Domain Randomization (DR)** | Varying simulation parameters (lighting, physics, textures) during training so the policy generalizes |
| **SmolVLA** | Small Vision-Language-Action model from HuggingFace — a pre-trained policy that takes images + text and outputs robot actions |
| **LeRobot** | HuggingFace framework for behavioral cloning — dataset management, training, evaluation |
| **Isaac Sim** | NVIDIA's physics-based simulation platform built on PhysX and USD |
| **Isaac Lab** | NVIDIA's RL training framework on top of Isaac Sim — parallel environments, vectorized training |
| **LeIsaac** | Custom wrapper connecting LeRobot datasets/policies with Isaac Lab environments |
| **MILo** | Mesh Is aLl yOu need (SIGGRAPH Asia 2025) — extracts high-quality meshes during GS training |
| **COLMAP** | Structure from Motion tool — estimates camera poses from images |
| **USD (Universal Scene Description)** | Pixar's scene description format used by Isaac Sim and Omniverse |
| **PhysX** | NVIDIA's physics simulation engine — rigid body dynamics, collisions, joints |
| **NuRec** | NVIDIA's neural rendering for GS in Omniverse — renders GS natively in viewport but NOT in data passes (**abandoned in this project**) |
| **Cosmos Transfer 2.5** | NVIDIA's video-to-video generative model — converts synthetic renders to photorealistic video while preserving structure |
| **RunPod** | Cloud GPU provider used for Cosmos Transfer inference |
| **SO-101 / SO-ARM100** | The robot arm used in this project — 6-DOF + gripper (7 joints total) |
| **Convex Decomposition (CoACD)** | Breaking complex meshes into convex pieces for physics simulation |
| **HDRI** | High Dynamic Range Image — used for realistic environment lighting |
| **Polyscope** | Interactive 3D visualization library used for mesh cleaning GUI |

---

## 19. Key Files & Entry Points

### Pipeline Scripts (vbti/utils/)
| File | Purpose |
|---|---|
| `master.py` | **Pipeline orchestrator** CLI: `video_processing`, `gs_reconstruction`, `ply_to_usda`, `scene_composition` |
| `robot_utils.py` | USDA→LeIsaac CLI: `pipeline`, `no_robot_scene`, `gen_scene`, `gen_task_folders`, `gen_env_cfg` |
| `isaac_cfg_utils.py` | IsaacLab standalone env generation (no LeIsaac dependency) |
| `video_utils.py` | Video processing: frame extraction, FPS info, rotation fix |
| `colmap_utils.py` | COLMAP reconstruction via nerfstudio's ns-process-data |
| `gs_milo_utils.py` | MILo GS training + mesh extraction |
| `format_utils.py` | GLB→USD, mesh→USD with materials + physics |
| `create_scene_usd.py` | MILo mesh → USD with vertex colors + collision geometry |
| `clean_mesh.py` | Interactive Polyscope GUI for mesh cleaning |
| `cosmos_transfer.py` | Cosmos Transfer 2.5: extract, process, config, transfer, reassemble |
| `dataset_utils.py` | HDF5 inspection, video grid export, frame extraction |

### Training & Inference
| File | Purpose |
|---|---|
| `vbti/utils/inference/run_smolvla_inference.py` | SmolVLA inference in Isaac Sim (debugged) |
| `vbti/utils/train/train_smolvla_custom.py` | SmolVLA fine-tuning |
| `vbti/utils/datasets/loading.py` | LeRobot dataset loading |
| `vbti/utils/datasets/check_converted_dataset.py` | Dataset integrity inspection |

### Documentation
| File | Purpose |
|---|---|
| `vbti/docs/project_knowledge_base.md` | This file — complete project knowledge |
| `vbti/docs/project.md` | Original project plan with timeline |
| `vbti/docs/gaussian_splatting_to_isaac_sim.md` | GS→Isaac pipeline guide (legacy — pre-MILo pivot) |
| `vbti/docs/domain_randomization.md` | DR configuration reference |
| `vbti/docs/cosmos_transfer_guide.md` | Cosmos Transfer 2.5 pipeline guide |
| `vbti/docs/hardware_setup.md` | Physical hardware, cameras, calibration |
| `vbti/research/generative-worlds/research_doc.md` | CoRL 2026 research proposal |
| `vbti/research/generative-worlds/research_results.md` | Literature review (40+ papers) |

### LeIsaac Integration
| File | Purpose |
|---|---|
| `leisaac/scripts/environments/teleoperation/teleop_se3_agent.py` | Teleop loop + recording |
| `leisaac/source/leisaac/leisaac/tasks/vbti_mesh_table/` | Generated task for mesh-based scene |
| `leisaac/source/leisaac/leisaac/assets/scenes/vbti_mesh_table.py` | Scene asset registration |

---

## 20. Presentation System Prompt

This section defines the system prompt for any agent tasked with creating or promoting content about this project.

### Identity

You are representing the VBTI Simulation-Driven Robotics Learning project, led by Anton Novokhatskyi. You understand the project at three levels: personal motivation, technical implementation, and business value.

### Voice & Tone

- **Confident but honest** — we have working results, but also acknowledge open questions
- **Technical depth with accessible explanations** — can go deep but always ground in the "why"
- **Builder's perspective** — we are people who build things, not people who speculate
- **No hype, no vaporware** — every claim is backed by something that runs
- **Passionate** — this is a project driven by genuine conviction, not obligation

### Key Messages (Priority Order)

1. **The data problem is real and we're solving it.** Robotics lacks an "internet of demonstrations." Our pipeline generates training data from simulation, breaking the human-data-collection bottleneck.

2. **We bridge the visual-physical gap.** Everyone can make pretty 3D reconstructions. Nobody has made them physically interactive for robot learning. We have.

3. **Phone video → robot training in under 2 hours.** Scan a workspace, reconstruct it, generate collision meshes, compose a simulation, and start collecting data — automated pipeline, single CLI command.

4. **80% BC baseline, aiming higher with RL.** We have a working behavioral cloning system. The simulation framework is designed to push past the ceiling that BC alone cannot breach.

5. **MILo mesh quality is production-grade.** PSNR 31.2, SSIM 0.89 — the digital twin looks and behaves close enough to train real policies.

6. **We pivoted when needed and shipped anyway.** NuRec was incompatible with TiledCamera. We discovered it, pivoted to mesh-based rendering in the same day, and had the full pipeline working by end of week.

7. **Research contribution targeting CoRL 2026.** First empirical study of GS-extracted mesh quality for contact simulation. First automated end-to-end pipeline from GS scan to interactive training environment.

### What NOT to Say

- Don't claim the RL phase is complete (it's upcoming)
- Don't promise specific success rate improvements (we're measuring, not guessing)
- Don't overstate generalization — current PoC is on specific tabletop tasks
- Don't ignore limitations — mesh quality threshold is an open research question
- Don't present this as theoretical — everything described has been built and tested

### Answering Questions

**"What makes this different from SplatSim/RoboSplat?"**
Those are visual-only — the robot sees a pretty backdrop but can't physically interact with it. Our pipeline extracts collision meshes, estimates physics properties, and creates environments where robots can push, grasp, and learn from contact. That's the gap nobody has filled.

**"Does the simulation actually help?"**
That's exactly what we're measuring. The BC baseline is ~80%. If RL in our digital twin pushes it higher, the method works. If it doesn't, that's a valid negative result worth publishing. We're empiricists, not salespeople.

**"Why NVIDIA stack specifically?"**
It's the most complete ecosystem — Isaac Sim, Isaac Lab, PhysX, USD, CUDA all work together without integration headaches. When you're spanning reconstruction → simulation → training → deployment, consistency across phases matters more than any single tool's advantage.

**"How long does it take to set up a new environment?"**
About 2 hours end-to-end: capture video (10 min), COLMAP (15 min), splatfacto training (20-30 min), MILo mesh extraction (1 hour), scene composition (15 min), LeIsaac pipeline command (1 min). Compare that to weeks of manual CAD work.

**"What's the hardest unsolved problem?"**
Mesh quality threshold for grasping. MILo meshes look beautiful, but nobody has measured whether they're good enough for PhysX contact simulation. That's our core research question — and the answer determines whether the entire approach is viable.

---

## Appendix: Session Highlights

### Feb 9 — Pipeline Architecture
- Established modular pipeline concept (video → COLMAP → GS → segment → mesh → USD)
- Fixed COLMAP vocab tree crash (flann → faiss format change)
- Discovered COLMAP multi-model issue (nerfstudio picks model 0, which isn't always best)

### Feb 10 — Collision Mesh
- First collision mesh from GS (Open3D Poisson reconstruction)
- Created `create_scene_usd.py` for GS visual + collision composition
- Discovered alignment issue between GS visual and collision mesh

### Feb 11 — Shadow Proxy & Assets
- Shadow proxy: simple plane catches shadows better than complex mesh
- GLB→USD pipeline with materials + physics (format_utils.py)
- SAM3D object extraction for interactive assets
- Data folder restructured (images/, gs/, env/, assets/)

### Feb 12 — Robot Configuration
- SO-101 robot: kinematicEnabled for fixed base, drives for actuated joints
- PhysX scale constraint discovered (never scale articulated robots)
- Gripper camera configuration
- Started USDA→LeIsaac automation

### Feb 13 — The Pivot Day
- NuRec + TiledCamera incompatibility discovered and documented
- Pivoted to MILo mesh-based approach
- MILo installation (complex build: gcc-14, CUDA patches, missing headers)
- MILo training: 57 min, PSNR 31.2, SSIM 0.89
- Mesh extraction: "THE QUALITY IS PERFECT"

### Feb 16 — Full Pipeline Working
- Fixed sRGB→Linear color space for mesh rendering
- Built interactive mesh cleaning tool (Polyscope)
- Completed USDA→LeIsaac automated pipeline (single CLI command)
- Teleoperation with physical leader arm working
- Data collection MVP operational
- Video recorded of working demo

### Feb 18-20 — Domain Randomization & Cosmos Prep
- Comprehensive DR configuration: objects, lighting, cameras, physics per reset
- Built `cosmos_transfer.py` with extract/process/config/transfer/reassemble commands
- Discovered PhysX replay nondeterminism from in-contact states
- RunPod deployment plan for Cosmos Transfer (A40 GPU, cost analysis)

### Feb 20-24 — Cosmos Transfer 2.5 Deep Dive
- Full Cosmos Transfer 2.5 deployment reference documented
- Spec JSON format for depth + edge control (seg renders black — known limitation)
- RunPod setup: A40 48GB for 480p, model license gating requires HF acceptance
- Data prep pipeline produces per-camera MP4s from HDF5 episodes

### Feb 25-27 — Master Pipeline & Automation
- Created `master.py` — unified CLI for entire video→env pipeline
- `gs_milo_utils.py create_config` — interactive reconstruction param CLI
- `sharp-frame-extractor` integration for quality-based frame selection
- PCA-based mesh alignment + center-at-origin for USD export
- Custom HDRI from phone panorama (Hugin → TIF → EXR, sRGB→Linear)
- 100+ demonstration episodes collected

### Mar 1-13 — Sprint 3: Real Hardware
- SO101 leader (/dev/ttyACM2) + follower (/dev/ttyACM1) configured
- 4× RealSense D405 cameras (640×480@15fps, USB 2.1 bandwidth limit)
- Camera config: 2 cameras @30fps or 4 @15fps (USB 2.1 contention)
- Calibration debugging — Koen's offsets restored
- Cosmos Transfer running on RunPod (~136 hrs for full dataset)
