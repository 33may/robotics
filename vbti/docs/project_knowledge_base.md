# VBTI Project — Complete Knowledge Base

**Author:** Anton Novokhatskyi
**Organization:** VBTI (robotics automation company) + Fontys University of Applied Sciences
**Date:** February 2026
**Status:** Active R&D — Phase 2 (Digital Twin Creation) nearing completion

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
13. [Research Direction — CoRL 2026](#13-research-direction--corl-2026)
14. [Literature & Prior Work](#14-literature--prior-work)
15. [Timeline & Milestones](#15-timeline--milestones)
16. [Metrics & Success Criteria](#16-metrics--success-criteria)
17. [Risks & Mitigations](#17-risks--mitigations)
18. [Glossary & Key Concepts](#18-glossary--key-concepts)
19. [Key Files & Entry Points](#19-key-files--entry-points)
20. [Presentation System Prompt](#20-presentation-system-prompt)

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

---

## 6. Stakeholders & Collaboration

| Stakeholder | Role | Involvement |
|---|---|---|
| **VBTI** | Robotics company | Hosts the project, defines practical requirements, provides hardware and domain expertise |
| **Fontys University** | Academic institution | Assesses the project as part of the ICT internship programme |
| **Anton Novokhatskyi** | Fontys ICT intern at VBTI | Responsible for the simulation framework — 3D reconstruction, digital twin pipeline, simulation-ready asset creation |
| **TU/e Collaborator** | Master thesis student at VBTI | Develops RL algorithms and training strategies that consume the simulation environments produced by this project |

### The Two-Sided Collaboration

This project sits at the intersection of two parallel efforts:
- **Anton's work:** The simulation infrastructure — environments, digital twins, scene creation
- **TU/e collaborator:** The learning algorithms — RL strategies, reward engineering, training optimization

The simulation framework provides the environments; the RL research provides the training methods. Both depend on each other.

---

## 7. Technical Architecture — Full Pipeline

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    REAL WORLD CAPTURE                            │
├─────────────────────────────────────────────────────────────────┤
│  iPhone video / photos of workspace                             │
│  ↓                                                              │
│  COLMAP (Structure from Motion) → camera poses + sparse cloud   │
│  ↓                                                              │
│  Nerfstudio splatfacto → Gaussian Splat (PLY)                   │
│  ↓                                                              │
│  SuperSplat → clean floating artifacts                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MESH EXTRACTION                               │
├─────────────────────────────────────────────────────────────────┤
│  COLMAP undistort images (OPENCV → PINHOLE)                     │
│  ↓                                                              │
│  MILo (SIGGRAPH Asia 2025) — learnable SDF from GS             │
│  ↓ Train: ~57 min on RTX 4070 Ti SUPER                          │
│  ↓ Quality: PSNR 31.2, SSIM 0.89                                │
│  ↓                                                              │
│  mesh_extract_sdf.py → mesh_learnable_sdf.ply                   │
│  ↓                                                              │
│  clean_mesh.py (Polyscope GUI) → crop OBB, remove artifacts     │
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

### Phase 2: Digital Twin Creation (Feb 11 - Mar 30)

**Goal:** Construct a high-fidelity digital twin of the real workspace.

Sub-tasks completed:
- Scene reconstruction via Gaussian Splatting (nerfstudio splatfacto)
- Mesh extraction via MILo (PSNR 31.2, SSIM 0.89)
- Interactive mesh cleaning tool (Polyscope GUI)
- sRGB→Linear color space conversion for Isaac Sim rendering
- GLB→USD asset pipeline with materials and physics
- Scene composition with HDRI lighting
- USDA→LeIsaac automated pipeline (single CLI command)
- Teleoperation working with physical leader arm
- Data collection MVP operational

**Status:** NEARLY COMPLETE — collecting first simulation datasets

### Phase 3: Simulation Training / RL (Mar 1 - Mar 30)

**Goal:** Use RL to improve the pre-trained model inside the digital twin.

- Reward engineering for task completion
- Parallel training in Isaac Lab
- Metric recording: task success rate in simulation vs BC baseline
- Runs in parallel with Phase 2 refinement

**Status:** UPCOMING

### Phase 4: Sim-to-Real Transfer & Validation (Apr 1 - Apr 15)

**Goal:** Deploy simulation-improved model to physical robot.

- Run optimized model on real SO-101
- Validate camera input similarity (sim vs real)
- Record final task success rate
- Calculate pure method gains (Phase 4 - Phase 1)
- Gap analysis if transfer degrades performance

**Status:** PLANNED

### Phase 5: Infrastructure Scaling (Apr 16 - Jun 30)

**Goal:** Solidify pipeline for real production use.

- Reusable scripts and tools from Phases 1-4
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
| **3DGRUT** (NVIDIA) | PLY→USDZ conversion for NuRec neural rendering |
| **USD (Universal Scene Description)** | Scene format for Isaac Sim |
| **Custom create_scene_usd.py** | MILo mesh→USD with vertex colors + collision |
| **Custom format_utils.py** | GLB→USD with materials + physics |
| **Custom robot_utils.py** | USDA→LeIsaac pipeline automation |

### Training & Inference
| Tool | Purpose |
|---|---|
| **LeRobot** (HuggingFace) | Behavioral cloning framework, dataset management |
| **SmolVLA** | Small Vision-Language-Action model (policy network) |
| **Custom run_smolvla_inference.py** | Isaac Sim inference loop with proper unit conversion |

### Hardware
| Component | Specs |
|---|---|
| **Robot** | SO-ARM100/101 (6-DOF arm + gripper, 7 joints) |
| **GPU** | NVIDIA RTX 4070 Ti SUPER (16GB VRAM, SM 89) |
| **OS** | Fedora 42 (Linux 6.17) |
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
8. **Data collection MVP** — Recording trajectories with multiple cameras in simulation
9. **SmolVLA inference in Isaac Sim** — Debugged: postprocessor loading, degree→radian conversion, image format conversion
10. **GLB→USD asset pipeline** — SAM3D objects converted with materials, physics, collision

### Key Results

| Metric | Value | Context |
|---|---|---|
| MILo PSNR | 31.2 dB | High reconstruction fidelity (target >30) |
| MILo SSIM | 0.89 | Close to target of 0.9 |
| MILo LPIPS | 0.17 | Perceptual similarity |
| BC success rate | ~80% | Real-world baseline on SO-101 |
| MILo train time | 57 min | On RTX 4070 Ti SUPER |
| Pipeline time | ~1 hour | Video → runnable LeIsaac task |

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

---

## 12. The Data Flow — End to End

### Training Data Format

```
Dataset (LeRobot HDF5):
├── observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] (DEGREES)
├── observation.images.side_cam: (H, W, 3) uint8
├── observation.images.table_cam: (H, W, 3) uint8
├── observation.images.wrist_cam: (H, W, 3) uint8
└── action: [6 joint positions] (DEGREES)
```

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

## 13. Research Direction — CoRL 2026

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

## 15. Timeline & Milestones

**Project duration:** February 1 – June 30, 2026

| Phase | Dates | Status | Key Deliverable |
|---|---|---|---|
| Phase 1: BC Baseline | Feb 3-20 | DONE | ~80% success rate baseline |
| Phase 2: Digital Twin | Feb 11 - Mar 30 | 90% DONE | Working sim environment + data collection |
| Phase 3: RL Training | Mar 1-30 | UPCOMING | RL-improved policy in simulation |
| Phase 4: Sim2Real | Apr 1-15 | PLANNED | Real-world validation + method gains |
| Phase 5: Infrastructure | Apr 16 - Jun 30 | PLANNED | Reusable pipeline + tomato sorting PoC |

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
| **NuRec** | NVIDIA's neural rendering for GS in Omniverse — renders GS natively in viewport but NOT in data passes |
| **SO-101 / SO-ARM100** | The robot arm used in this project — 6-DOF + gripper (7 joints total) |
| **Convex Decomposition (CoACD)** | Breaking complex meshes into convex pieces for physics simulation |
| **HDRI** | High Dynamic Range Image — used for realistic environment lighting |
| **Polyscope** | Interactive 3D visualization library used for mesh cleaning GUI |

---

## 19. Key Files & Entry Points

### Pipeline Scripts
| File | Purpose |
|---|---|
| `vbti/utils/robot_utils.py` | Master CLI: `pipeline`, `no_robot_scene`, `gen_scene`, `gen_task_folders`, `gen_env_cfg` |
| `vbti/utils/create_scene_usd.py` | MILo mesh → USD with vertex colors + collision geometry |
| `vbti/utils/format_utils.py` | GLB→USD conversion with materials + physics |
| `vbti/utils/clean_mesh.py` | Interactive Polyscope GUI for mesh cleaning |
| `vbti/utils/video_utils.py` | Video processing utilities |

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
| `vbti/docs/project.md` | Project plan with timeline |
| `vbti/docs/gaussian_splatting_to_isaac_sim.md` | Complete GS pipeline guide |
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
