# BRANCH 1: GS Editing as Structured Domain Randomization for Contact-Rich Tasks

**Date:** 2026-02-06
**Status:** Deep Literature Review Complete

---

## Table of Contents

1. [Literature Landscape](#1-literature-landscape)
2. [The Gap (Refined)](#2-the-gap-refined)
3. [Top 5 Papers](#3-top-5-papers-with-relevance-and-unsolved-problems)
4. [Primary Output: Refined Research Questions](#4-primary-output-refined-research-questions)
5. [Feasibility on RTX 4070 Ti SUPER](#5-feasibility-on-rtx-4070-ti-super-16gb-vram)
6. [Minimal Experiment Design](#6-minimal-experiment-design)
7. [Biggest Risk + Mitigation](#7-biggest-risk--mitigation)

---

## 1. Literature Landscape

### 1.1 Visual-Only GS Augmentation (The Current State)

**RoboSplat** (Yang et al., RSS 2025, [arxiv 2504.13175](https://arxiv.org/abs/2504.13175))
- 6 augmentation types: object pose (equivariant transforms), object type (GPT-4 + 3D generation), camera view (spherical randomization), embodiment type (GS replacement), lighting (diffuse color scaling/offset/noise), and scene appearance
- 87.8% one-shot success across 6 generalization types
- **Explicit limitation from Section VI:** "incapable of handling deformable objects" and "lacks physical constraints, making it unsuitable for contact-rich and dynamic tasks"
- This is purely visual augmentation -- no physics proxy, no collision geometry, no force/torque feedback

**SplatSim** (CMU, 2024, [arxiv 2409.10161](https://arxiv.org/abs/2409.10161))
- GS rendering overlaid on MuJoCo physics simulation
- 86.25% zero-shot sim2real success for RGB manipulation policies
- The GS layer is a visual skin; physics remains in the underlying MuJoCo scene
- No mechanism to edit the GS appearance AND update the physics proxy simultaneously

**RL-GSBridge** (Wu et al., 2024, [arxiv 2409.20291](https://arxiv.org/abs/2409.20291))
- Real2Sim2Real with GS integrated into RL pipeline
- Mesh-based 3D GS with soft binding constraints
- Integrates dynamics signals from simulator to edit GS models for realistic physical interaction
- Tested on grasping and pick-and-place with diverse textures/geometries
- **Closest to visual+physics sync**, but focused on rendering quality rather than systematic domain randomization

### 1.2 Hybrid GS+Physics Approaches (Emerging)

**RoboSimGS** (Zhao et al., Oct 2025, [arxiv 2510.10637](https://arxiv.org/abs/2510.10637))
- Hybrid representation: 3DGS for static environment appearance, explicit mesh for interactive objects
- Uses GPT-4o to analyze 4 orthographic views and estimate density (rho), Young's modulus (E), Poisson's ratio (nu)
- Supports "Holistic Scene Augmentation": object pose randomization, camera perturbations, lighting mods, trajectory variations
- Tested on 8 tasks including deformable pick-and-place, articulated interactions, tool use
- **Key limitation:** "time-consuming and complex scene reconstruction pipeline... significant bottleneck for large-scale deployment"
- **This is the closest existing work to our proposed research**, but it does NOT frame augmentation as structured domain randomization with principled parameter sweeps

**Robo-GS** (Lou et al., ICRA 2025, [arxiv 2408.14873](https://arxiv.org/abs/2408.14873))
- Hybrid representation: mesh geometry + 3D Gaussian kernels + physics attributes
- Gaussian-Mesh-Pixel binding via isomorphic mapping between mesh vertices and Gaussians
- Fully differentiable rendering pipeline
- Focused on digital asset reconstruction of robotic arms, not on scene-level DR or policy training

**Phys2Real** (2025, [OpenReview](https://openreview.net/forum?id=7HQTnl8qao))
- Real-to-sim-to-real pipeline using GS (via SuGaR) + VLM-inferred physics parameters
- Estimates friction and center of mass using VLMs
- Conditions RL policies on known physics params during training, VLM-inferred params at test time
- 100% vs 60% success rate compared to domain-randomization baselines on planar pushing
- **Explicitly argues AGAINST uniform DR** -- proposes physics-informed conditioning instead
- Limited to 2 planar pushing tasks (T-block, hammer)

### 1.3 GS + Physics World Models

**Physically Embodied Gaussian Splatting (PEGS)** (Abou-Chakra et al., CoRL 2024, [arxiv 2406.10788](https://arxiv.org/abs/2406.10788))
- Dual Gaussian-Particle representation
- Particles for geometry/physics, 3D Gaussians attached for rendering
- "Visual forces" correct particle positions while respecting physical constraints
- 30Hz real-time, 3 cameras
- World model for prediction, NOT a training environment for DR

**GWM: Gaussian World Models** (Lu et al., ICCV 2025, [arxiv 2508.17600](https://arxiv.org/abs/2508.17600))
- Action-conditioned 3D video prediction using latent Diffusion Transformer + 3D VAE
- Can serve as neural simulator for model-based RL
- Predicts future GS states from robot actions
- NOT a physics simulator -- learned dynamics, no explicit collision or force modeling

**RoboPearls** (Tao et al., ICCV 2025, [arxiv 2506.22756](https://arxiv.org/abs/2506.22756))
- Editable video simulation from demonstration videos via GS
- LLM agents automate scene editing from natural language
- +17.5% and +10.8% success over RVT/RVT2
- Video simulation, not physics-in-the-loop

### 1.4 Mesh-GS Binding (Enabling Technology)

**SuGaR** (Guedon & Lepetit, CVPR 2024, [arxiv 2311.12775](https://arxiv.org/abs/2311.12775))
- Surface-aligned Gaussians, Poisson mesh extraction
- Optional refinement: bind Gaussians to mesh surface, jointly optimize
- Enables Blender/Unity/Unreal editing of GS via mesh manipulation
- **Foundation for physics proxy extraction from GS scenes**

**MILo** (Guedon et al., SIGGRAPH Asia 2025)
- Mesh-in-the-Loop: differentiable mesh extraction DURING GS optimization
- Gradient flow from mesh to Gaussians -- bidirectional consistency
- Higher quality meshes with fewer vertices
- Explicitly designed for "downstream applications like physics simulations"

**GS-Verse** (Pechko et al., 2025, [arxiv 2510.11878](https://arxiv.org/abs/2510.11878))
- Directly integrates object mesh with GS for VR physics manipulation
- Physics-engine-agnostic
- Validated on stretching, twisting, shaking
- User study: statistically significant improvement over prior GS+VR methods

### 1.5 Domain Randomization Theory

**Tobin et al. (IROS 2017)** ([arxiv 1703.06907](https://arxiv.org/abs/1703.06907))
- Seminal DR paper: randomize rendering -> real world = "just another variation"
- Number of images and unique textures were most important parameters
- 1.5cm avg error in object localization, sufficient for grasping

**Chen et al. (ICLR 2022)** ([arxiv 2110.03239](https://arxiv.org/abs/2110.03239))
- First theoretical framework for DR sim2real gap
- Sim2real gap scales as O(M^3 log(MH)) with M candidate simulators
- **Key insight:** history-dependent policies are important for DR
- More randomization helps transfer but harms simulation performance -- fundamental tension

**Continual Domain Randomization (CDR)** (2024, [arxiv 2403.12193](https://arxiv.org/abs/2403.12193))
- Sequential training on parameter subsets instead of all-at-once
- Can combine with active DR for automatic range finding
- Addresses the "too much randomization kills performance" problem

### 1.6 Agriculture-Specific Sim2Real

**Zero-Shot Sim2Real RL for Fruit Harvesting** (2025, [arxiv 2505.08458](https://arxiv.org/abs/2505.08458))
- MuJoCo simulation for strawberry picking with Franka Panda
- DR over lighting, object placement, sensor noise
- End-to-end from raw pixels + proprioception
- One of first sim2real deep RL applications to agricultural harvesting

**Find the Fruit** (2025, [arxiv 2505.16547](https://arxiv.org/abs/2505.16547))
- Digital twin + PPO for occlusion-aware plant manipulation
- 96% sim success, 87% zero-shot real success
- Multi-modal observations (RGB-D + segmentation + proprioception)

### 1.7 Soft-Body + GS

**Real-to-Sim Policy Evaluation with GS** (Zhang et al., 2025, [arxiv 2511.04665](https://arxiv.org/abs/2511.04665))
- Soft-body digital twins from real videos + GS rendering
- Deformation-aware rendering
- Policy evaluation only (not training)
- r > 0.9 correlation between sim and real success rates

**Material-Informed GS** (2025, [arxiv 2511.20348](https://arxiv.org/abs/2511.20348))
- Camera-only pipeline: GS reconstruction -> semantic material masks -> physics-based material properties
- Assigns density, friction, etc. from visual appearance
- Designed for sensor simulation in graphics engines

---

## 2. The Gap (Refined)

After this deep search, the gap is sharper than initially stated:

### What EXISTS:
1. **Visual-only GS augmentation** (RoboSplat, SplatSim) -- edits appearance, ignores physics
2. **GS+mesh hybrid reconstruction** (RoboSimGS, Robo-GS) -- builds physics-compatible scenes from reality
3. **GS+mesh binding technology** (SuGaR, MILo, GS-Verse) -- links GS edits to mesh geometry
4. **Physics-informed GS** (Phys2Real, Material-Informed GS) -- infers physical properties from visuals
5. **DR theory** (Chen et al. ICLR 2022) -- proves DR works but with diminishing returns

### What DOES NOT EXIST:
**A closed-loop system that:**
1. Takes a GS scene reconstruction
2. Extracts physics-compatible mesh proxies (via SuGaR/MILo)
3. Performs STRUCTURED edits to BOTH the GS visual layer AND the physics proxy layer in lockstep
4. Uses these synchronized edits as principled domain randomization (not just random visual noise)
5. Trains RL/IL policies in the edited environments with physics feedback
6. Measures whether visual+physics-consistent DR outperforms visual-only DR for **contact-rich** tasks

RoboSimGS comes closest but: (a) does not frame its augmentation as structured DR with controlled parameter sweeps, (b) has a prohibitively slow reconstruction pipeline, (c) does not compare visual+physics DR vs visual-only DR, and (d) does not provide scaling curves.

Phys2Real argues against DR entirely, proposing physics-informed conditioning instead -- but tests on only 2 simple planar tasks, not contact-rich manipulation.

---

## 3. Top 5 Papers with Relevance and Unsolved Problems

### Paper 1: RoboSimGS
- **Title:** High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting
- **Authors:** Haoyu Zhao, Cheng Zeng, Linghao Zhuang, et al.
- **Year/Venue:** 2025, arXiv (Oct 2025)
- **Link:** [arxiv 2510.10637](https://arxiv.org/abs/2510.10637)
- **Relevance:** Closest to our proposal -- hybrid GS+mesh with MLLM-inferred physics and holistic scene augmentation. Demonstrates deformable and articulated task training.
- **Unsolved:** Does not systematically study which DR parameters matter for contact-rich transfer. Reconstruction pipeline is too slow for iterative DR experiments. No ablation separating visual DR from physics DR contributions.

### Paper 2: RoboSplat
- **Title:** Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation
- **Authors:** Sizhe Yang, Wenye Yu, Jia Zeng, et al.
- **Year/Venue:** RSS 2025
- **Link:** [arxiv 2504.13175](https://arxiv.org/abs/2504.13175)
- **Relevance:** State-of-the-art in visual-only GS augmentation. Explicitly states the failure mode we target: "lacks physical constraints, making it unsuitable for contact-rich and dynamic tasks."
- **Unsolved:** Cannot handle deformable objects. No physics layer. Unknown whether adding physics consistency to its 6 augmentation types would improve contact-rich transfer.

### Paper 3: Phys2Real
- **Title:** Physically-Informed Gaussian Splatting for Adaptive Sim-to-Real Transfer in Robotic Manipulation
- **Authors:** (Available on OpenReview)
- **Year/Venue:** 2025, Under review
- **Link:** [OpenReview](https://openreview.net/forum?id=7HQTnl8qao)
- **Relevance:** Directly argues that physics-informed GS beats uniform DR. Uses SuGaR for mesh extraction + VLM for physics parameter estimation. Demonstrates 100% vs 60% over DR baselines.
- **Unsolved:** Only tested on 2 planar pushing tasks. Does not address contact-rich grasping. The "DR is bad" claim may not hold when DR is STRUCTURED (informed by physics) rather than uniform.

### Paper 4: MILo (Mesh-in-the-Loop Gaussian Splatting)
- **Title:** Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction
- **Authors:** Antoine Guedon et al.
- **Year/Venue:** SIGGRAPH Asia 2025
- **Link:** [Project page](https://anttwo.github.io/milo/)
- **Relevance:** The key enabling technology -- differentiable mesh extraction during GS optimization means edits to GS propagate to mesh and vice versa. This is the "binding" needed for synchronized visual+physics DR.
- **Unsolved:** Not applied to robotics or DR at all. No physics simulation integration. The quality of extracted collision meshes for contact simulation is unknown.

### Paper 5: Zero-Shot Sim2Real RL for Fruit Harvesting
- **Title:** Zero-Shot Sim-to-Real Reinforcement Learning for Fruit Harvesting
- **Authors:** (See arxiv)
- **Year/Venue:** 2025
- **Link:** [arxiv 2505.08458](https://arxiv.org/abs/2505.08458)
- **Relevance:** Demonstrates that sim2real RL works for agriculture with traditional DR in MuJoCo. Establishes the baseline that GS-based structured DR must beat. End-to-end pixel-to-action policy for strawberry picking.
- **Unsolved:** Uses hand-designed MuJoCo environments, not real-world reconstructions. Visual fidelity is limited. Does not leverage GS for photorealism. Unknown whether GS-quality visuals would improve the policy.

---

## 4. Primary Output: Refined Research Questions

### Research Question 1: Does Physics-Synchronized GS Domain Randomization Outperform Visual-Only GS Augmentation for Contact-Rich Manipulation?

**The Question:** When training manipulation policies in GS-reconstructed environments, does synchronizing visual edits (GS appearance) with physics proxy edits (collision mesh geometry, mass, friction) produce higher sim2real transfer rates for contact-rich tasks (grasping, insertion, peg-in-hole) compared to visual-only augmentation (a la RoboSplat)?

**Prior Work Context:**
- RoboSplat (RSS 2025): 87.8% one-shot with visual-only GS augmentation, explicitly admits failure on contact-rich tasks
- RoboSimGS (2025): Hybrid GS+mesh with holistic augmentation, but no controlled ablation separating visual from physics DR
- Phys2Real (2025): Claims physics-informed beats DR, but only on 2 planar tasks

**Proposed Method:** Reconstruct a tabletop manipulation scene using 3DGS (e.g., gsplat or nerfstudio). Extract physics-compatible meshes using SuGaR/MILo. Define 3 DR axes: (A) visual-only (lighting, texture, camera -- matching RoboSplat's approach), (B) physics-only (mass +/-30%, friction +/-50%, mesh scale +/-10%), (C) synchronized visual+physics (visual edits matched to physics changes -- e.g., scaling an object visually also scales its collision mesh and adjusts mass proportionally). Train identical policies (e.g., Diffusion Policy or ACT) under conditions A, B, C on grasping tasks in Isaac Sim. Evaluate zero-shot transfer to real robot.

**Expected Outcome:** If synchronized DR (condition C) significantly outperforms visual-only (A) on contact-rich tasks while performing comparably on non-contact tasks, this confirms that the physics gap (not just visual gap) is the bottleneck for contact-rich sim2real. We expect >10% absolute improvement on grasping success rate for C over A.

**Why It Matters:** This would redirect the GS-for-robotics community from purely visual augmentation toward physics-aware augmentation, unlocking GS for the harder class of contact-rich tasks that remain the primary industrial need.

---

### Research Question 2: What is the Scaling Curve of GS-Based Structured DR, and Does It Beat the Diminishing Returns of Uniform DR?

**The Question:** How does manipulation policy performance scale with the NUMBER and DIVERSITY of GS-edited training environments, and does structured (physics-consistent) DR achieve better sample efficiency than uniform (random) DR?

**Prior Work Context:**
- Tobin et al. (IROS 2017): Showed texture count and image count are dominant DR parameters, but provided no formal scaling law
- Chen et al. (ICLR 2022): Proved DR gap scales O(M^3 log(MH)) -- diminishing returns with more simulators
- CDR (2024): Showed sequential parameter randomization beats all-at-once

**Proposed Method:** Starting from a single GS scene, generate N = {1, 5, 10, 25, 50, 100} edited environments using (a) uniform random parameter sampling and (b) structured sampling (Latin hypercube or Sobol sequences over a physics-informed parameter space: object mass, friction, scale, position, lighting). Train policies on each set. Plot success rate vs N for both sampling strategies. Measure whether structured sampling reaches saturation later (better scaling) or earlier (more efficient).

**Expected Outcome:** Structured DR should achieve equivalent performance to uniform DR with roughly 3-5x fewer environments (based on the CDR finding that sequential/structured beats all-at-once). The scaling curve should show a log-linear relationship up to ~50 environments before saturating for both approaches, but with structured DR shifted left.

**Why It Matters:** If GS-based structured DR scales better than uniform DR, practitioners can achieve robust sim2real with fewer (expensive) scene reconstructions and edits, making the approach practical for real labs. This directly addresses RoboSimGS's reconstruction bottleneck.

---

### Research Question 3: Can VLM-Inferred Physics Properties from GS Reconstructions Replace Manual Physics Parameter Specification for Contact-Rich Training?

**The Question:** Given a GS reconstruction of a manipulation scene, can a VLM (GPT-4o, Gemini) accurately estimate sufficient physics properties (mass, friction, stiffness, center of mass) from rendered views to enable contact-rich policy training that transfers to the real world, without ANY manual physics specification?

**Prior Work Context:**
- RoboSimGS (2025): Uses GPT-4o to estimate density, Young's modulus, Poisson's ratio from 4 orthographic views -- but doesn't validate estimation accuracy against ground truth
- Phys2Real (2025): Uses VLM for friction and CoM estimation, achieves 100% success on pushing -- but only 2 simple tasks
- Material-Informed GS (2025): Assigns physics material properties from semantic masks -- autonomous driving only

**Proposed Method:** Collect 20 common manipulation objects with KNOWN physics properties (measured with scale, friction tester). Reconstruct each using 3DGS. Feed rendered views to GPT-4o and Gemini, prompt for physics parameter estimation. Compare VLM estimates to ground truth. Then train grasping policies using (a) ground-truth physics, (b) VLM-estimated physics, (c) uniform DR over physics. Measure sim2real transfer for each.

**Expected Outcome:** VLM estimates will be coarse (likely 30-50% error on individual parameters) but sufficient for policy training -- i.e., condition (b) will perform within 5-10% of condition (a) and significantly better than (c). This would validate RoboSimGS's approach while quantifying the error budget.

**Why It Matters:** If VLMs can estimate "good enough" physics from GS renders, the entire Real2Sim pipeline becomes fully automatic: scan scene -> reconstruct GS -> VLM estimates physics -> train policy. No human-in-the-loop. This is critical for scalability.

---

### Research Question 4: Does GS-Based Visual Fidelity Provide Measurable Benefit Over Standard Rendering for Contact-Rich Tasks Specifically?

**The Question:** For contact-rich tasks where force/torque feedback dominates the policy (grasping, insertion), does the photorealistic rendering quality of GS provide ANY benefit over standard simulation rendering (e.g., Isaac Sim RTX renderer), or is physics accuracy alone sufficient?

**Prior Work Context:**
- SplatSim (2024): 86.25% with GS rendering vs lower with standard rendering -- but on non-contact tasks
- Zero-Shot Fruit Harvesting (2025): Good sim2real with standard MuJoCo rendering + DR -- suggesting visual fidelity may not be critical for some contact tasks
- Isaac Lab (2025): RTX rendering + PhysX -- high visual quality without GS

**Proposed Method:** Design a grasping benchmark with 3 conditions: (a) Isaac Sim standard rendering + PhysX physics, (b) Isaac Sim RTX rendering + PhysX, (c) GS rendering (via SplatSim approach) + PhysX. Keep physics identical across all three. Train identical policies. Measure sim2real transfer gap for each. If GS rendering shows negligible benefit for contact tasks, the community should focus on physics fidelity instead.

**Expected Outcome:** For purely contact-rich tasks (force-closure grasping), GS rendering will provide minimal (<5%) improvement over RTX rendering. For visually-guided contact tasks (e.g., grasping a specific object among clutter), GS will provide 10-20% improvement. This clarifies WHEN GS visual quality matters.

**Why It Matters:** Prevents over-investment in GS visual fidelity for tasks where physics is the real bottleneck. Helps the community allocate effort correctly between visual and physics sim2real gaps.

---

### Research Question 5: Can Mesh-In-the-Loop GS Optimization (MILo) Produce Collision Meshes Suitable for RL Training, and at What Computational Cost?

**The Question:** Can the bidirectional mesh-GS consistency from MILo produce collision meshes that are accurate enough for contact simulation in Isaac Sim/MuJoCo, and can this be done within a practical time budget for generating multiple DR environments?

**Prior Work Context:**
- MILo (SIGGRAPH Asia 2025): Differentiable mesh extraction during GS optimization, fewer vertices, higher quality
- SuGaR (CVPR 2024): Post-hoc mesh extraction, used by Phys2Real for physics proxy
- GS-Verse (2025): Direct mesh+GS integration for physics, but in VR context only

**Proposed Method:** Take 10 common manipulation objects. Reconstruct with 3DGS. Extract collision meshes via (a) SuGaR, (b) MILo, (c) manual CAD models (ground truth). Import all meshes into Isaac Sim. Run 1000 grasping trials per mesh type. Measure (i) grasp success rate vs ground-truth CAD, (ii) physics simulation stability (penetration, explosion artifacts), (iii) mesh extraction time. Then edit the GS scene (scale, translate, deform an object) and measure how well the mesh updates track the visual edit.

**Expected Outcome:** MILo meshes will be closer to CAD ground truth than SuGaR (fewer artifacts, better surface normals). Both will have 5-15% lower grasp success than CAD due to mesh quality issues. MILo will take ~2x longer than SuGaR but produce meshes requiring less manual cleanup. The synchronized edit quality will be the key differentiator.

**Why It Matters:** This validates (or invalidates) MILo as the practical tool for our Branch 1 pipeline. If extracted meshes are NOT suitable for contact simulation, the entire approach needs a different mesh extraction strategy.

---

## 5. Feasibility on RTX 4070 Ti SUPER (16GB VRAM)

### What Fits:

| Component | VRAM Estimate | Feasible? |
|-----------|--------------|-----------|
| 3DGS scene reconstruction (gsplat) | 4-8 GB | Yes |
| SuGaR mesh extraction | 6-10 GB | Yes (borderline for large scenes) |
| MILo mesh-in-the-loop | 8-12 GB | Tight -- small scenes only |
| GaussianEditor text-guided editing | 10-14 GB | Marginal -- may need optimization |
| Isaac Sim headless with PhysX | 4-6 GB | Yes (separate from GS) |
| Policy training (Diffusion Policy) | 6-10 GB | Yes |
| Policy training (ACT) | 4-6 GB | Yes |
| GPT-4o VLM inference | API call | Yes (no local VRAM) |

### Practical Strategy:
1. **Sequential pipeline, not concurrent:** Reconstruct GS -> extract mesh -> import to Isaac Sim -> train policy. Each step uses GPU separately.
2. **Use gsplat (not original 3DGS)** for reconstruction -- more memory efficient.
3. **Start with SuGaR** for mesh extraction (proven, lighter). Upgrade to MILo only if mesh quality is insufficient.
4. **Isaac Sim headless mode** for training -- no rendering overhead.
5. **Small scenes (tabletop, ~500K Gaussians)** keep VRAM manageable.

### What Does NOT Fit:
- Running GS rendering AND Isaac Sim physics simultaneously on the same GPU (would need ~20GB+)
- MILo on complex multi-object scenes (>1M Gaussians)
- Full RoboSimGS pipeline (designed for multi-GPU)

### Workaround:
The SplatSim approach (pre-render GS images, use them as textures in physics sim) avoids simultaneous GS+physics GPU load. This is the recommended path for 16GB VRAM.

---

## 6. Minimal Experiment Design

### Phase 1: Proof of Concept (2-3 weeks)
**Goal:** Demonstrate synchronized GS+physics editing on a single object

1. Reconstruct a tabletop scene with 3 graspable objects using gsplat (~1 day)
2. Extract collision meshes using SuGaR (~1 day)
3. Import both GS visual layer and physics meshes into Isaac Sim
4. Implement synchronized edit: scale one object by 1.2x in BOTH GS and physics mesh
5. Verify: visual rendering matches physics collision boundary
6. Train a simple grasping policy on original and edited scene
7. **Success metric:** Policy trained on {original + 1 edit} grasps the scaled object >50% of the time

### Phase 2: Controlled DR Comparison (3-4 weeks)
**Goal:** Compare visual-only vs synchronized DR on grasping

1. Generate 10 edited environments: 5 visual-only (lighting, texture), 5 synchronized (scale + mass, position + collision mesh)
2. Train Diffusion Policy on each set
3. Evaluate on 3 held-out real-world object configurations
4. **Success metric:** Synchronized DR shows >10% improvement over visual-only on grasping tasks

### Phase 3: Scaling Analysis (2-3 weeks)
**Goal:** Plot the scaling curve

1. Generate N = {1, 5, 10, 25, 50} synchronized DR environments
2. Train and evaluate at each N
3. Plot success rate vs N
4. Compare to RoboSplat's reported visual-only augmentation numbers
5. **Success metric:** Clear log-linear scaling with interpretable saturation point

### Total Timeline: ~8-10 weeks
### Hardware: 1x RTX 4070 Ti SUPER (sequential pipeline)

---

## 7. Biggest Risk + Mitigation

### BIGGEST RISK: Extracted meshes are too noisy for stable contact simulation

**Why this is the risk:**
SuGaR and MILo extract meshes from neural representations. These meshes can have:
- Non-manifold geometry (holes, self-intersections)
- Noisy surface normals
- Incorrect scale
- Missing thin features critical for grasping (edges, handles)

If the collision mesh is poor, physics simulation becomes unstable (objects penetrate, explode, or slide unrealistically), and no amount of visual fidelity will save the policy.

**Evidence this is real:**
- GS-Verse explicitly noted that "simplified geometric proxies... compromise the accuracy of physical simulations"
- RoboSimGS uses SEPARATE manually-modeled mesh primitives for interactive objects rather than extracted meshes
- Phys2Real only tested on simple geometric shapes (T-block, hammer) where mesh extraction is trivial

**Mitigation Strategy:**
1. **Convex decomposition:** After mesh extraction, apply V-HACD (Volumetric Hierarchical Approximate Convex Decomposition) to get simulation-friendly collision hulls. This is standard in Isaac Sim/MuJoCo.
2. **Hybrid approach (Plan B):** Use GS for visual rendering but substitute extracted meshes with parameterized primitives (boxes, cylinders, superquadrics) for physics. Edit the primitives to match GS edits approximately. Less elegant but robust.
3. **Mesh quality validation gate:** Before using an extracted mesh for policy training, run 100 stability test grasps. If >20% result in physics instability, fall back to primitive approximation for that object.
4. **Start with geometrically simple objects:** Boxes, cylinders, and bottles first. Only attempt complex geometry (mugs with handles, tools) after validating the pipeline on simple cases.

### SECOND RISK: The contribution delta over RoboSimGS is too small

**Mitigation:** Focus the contribution on the STRUCTURED DR framework and scaling analysis, not just the visual+physics synchronization. RoboSimGS shows it CAN be done; our contribution is showing it SHOULD be done in a principled way and providing the community with scaling curves and ablations they need to adopt it.

---

## Appendix: Additional Papers Reviewed

| Paper | Year | Key Idea | Why Not Top 5 |
|-------|------|----------|---------------|
| RoboGSim | 2024 | Real2Sim2Real GS simulator | Broader scope, less focused on DR |
| GaussianEditor | CVPR 2024 | Text-guided GS editing | Not robotics-focused |
| VR-GS | 2024 | GS + XPBD physics in VR | VR interaction, not policy training |
| CDR | 2024 | Continual DR | No GS component |
| Find the Fruit | 2025 | Digital twin RL for plant manipulation | Standard rendering, no GS |
| Real-to-Sim Soft-Body | 2025 | GS + deformable digital twins | Evaluation only, not training |
| Material-Informed GS | 2025 | Physics material from GS | Autonomous driving, not manipulation |

---

## Sources

- [RoboSplat (RSS 2025)](https://arxiv.org/abs/2504.13175)
- [SplatSim (2024)](https://splatsim.github.io/)
- [RL-GSBridge (2024)](https://arxiv.org/abs/2409.20291)
- [RoboSimGS (2025)](https://arxiv.org/abs/2510.10637)
- [Robo-GS (ICRA 2025)](https://arxiv.org/abs/2408.14873)
- [Phys2Real (2025)](https://openreview.net/forum?id=7HQTnl8qao)
- [PEGS (CoRL 2024)](https://arxiv.org/abs/2406.10788)
- [GWM (ICCV 2025)](https://arxiv.org/abs/2508.17600)
- [RoboPearls (ICCV 2025)](https://arxiv.org/abs/2506.22756)
- [SuGaR (CVPR 2024)](https://arxiv.org/abs/2311.12775)
- [MILo (SIGGRAPH Asia 2025)](https://anttwo.github.io/milo/)
- [GS-Verse (2025)](https://arxiv.org/abs/2510.11878)
- [Tobin et al. (IROS 2017)](https://arxiv.org/abs/1703.06907)
- [Chen et al. (ICLR 2022)](https://arxiv.org/abs/2110.03239)
- [CDR (2024)](https://arxiv.org/abs/2403.12193)
- [Zero-Shot Fruit Harvesting (2025)](https://arxiv.org/abs/2505.08458)
- [Find the Fruit (2025)](https://arxiv.org/abs/2505.16547)
- [Real-to-Sim Soft-Body (2025)](https://arxiv.org/abs/2511.04665)
- [Material-Informed GS (2025)](https://arxiv.org/abs/2511.20348)
- [RoboGSim (2024)](https://arxiv.org/abs/2411.11839)
- [Isaac Lab (2025)](https://d1qx31qr3h6wln.cloudfront.net/publications/Isaac%20Lab,%20A%20GPU-Accelerated%20Simulation%20Framework%20for%20Multi-Modal%20Robot%20Learning.pdf)
- [DR for Grasping QD (2023)](https://arxiv.org/abs/2310.04517)
