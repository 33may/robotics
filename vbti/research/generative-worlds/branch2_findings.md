# BRANCH 2: Semantic-Physical Scene Decomposition from GS Reconstructions

## Refined Research Questions, Literature Survey, and Feasibility Analysis

**Date:** 2026-02-06
**Scope:** Connecting GS scan -> semantic segmentation -> per-object collision mesh -> physics properties -> Isaac Sim as interactive objects

---

## Table of Contents

1. [Refined Research Questions (PRIMARY OUTPUT)](#refined-research-questions)
2. [Top Papers with Full Details](#top-papers-with-full-details)
3. [Complete Literature Map](#complete-literature-map)
4. [Feasibility on RTX 4070 Ti SUPER](#feasibility-on-rtx-4070-ti-super)
5. [Minimal Experiment Design](#minimal-experiment-design)
6. [Biggest Risk + Mitigation](#biggest-risk--mitigation)

---

## Refined Research Questions

### RQ1: Can MILo-extracted meshes serve as PhysX-compatible collision proxies for robotics grasping without manual post-processing?

**The question:** Given a Gaussian Splatting reconstruction of a tabletop scene, can MILo's differentiable mesh extraction produce meshes that, after automated CoACD convex decomposition, yield grasp-quality collision geometry in Isaac Sim (contact normal error < 5 degrees, penetration depth < 1mm)?

**Prior work context:**
- **MILo (SIGGRAPH Asia 2025):** Extracts meshes differentiably during GS training via Delaunay triangulation + Marching Tetrahedra. Produces 10x fewer vertices than prior methods while preserving detail. But never evaluated for physics simulation quality -- only rendering fidelity.
- **SuGaR (CVPR 2024):** Poisson reconstruction from aligned Gaussians. Meshes are smooth but not guaranteed watertight/manifold. GS-Verse (2025) showed SuGaR meshes can be used in Unity physics but required manual cleanup.
- **CoACD (SIGGRAPH 2022):** Collision-aware approximate convex decomposition. Already integrated into Isaac Sim/Isaac Lab's mesh converter. Requires input meshes to be reasonably clean.

**The gap:** MILo produces compact, detailed meshes. CoACD can decompose them for PhysX. Nobody has measured whether MILo meshes are clean enough (watertight, manifold, no self-intersections) to go directly into CoACD -> PhysX without manual intervention. SuGaR meshes are known to have holes and non-manifold edges that break convex decomposition.

**Proposed method:** Reconstruct 10 tabletop objects (varying geometry complexity: sphere, box, mug, tomato, bottle, wrench, banana, teddy bear, drill, bowl) using MILo from 50-view captures. Run automated pipeline: MILo mesh -> manifold repair (ManifoldPlus) -> CoACD decomposition -> Isaac Sim PhysX import. Measure: (a) mesh quality metrics (watertight ratio, self-intersection count, Hausdorff distance to ground-truth scan), (b) physics metrics (stable resting on table, grasp success with parallel-jaw gripper at 10 pre-computed poses), (c) pipeline automation rate (% requiring zero manual intervention).

**Expected outcome:** MILo meshes should achieve >80% automation rate for simple convex objects (sphere, box, bowl) but likely fail for thin/concave geometry (mug handle, wrench). If automation rate >60% across all objects, the pipeline is viable. If <40%, mesh post-processing is a hard blocker.

**Why it matters:** This is the critical bottleneck. Without automated mesh extraction -> collision proxy, every scanned object requires artist intervention, killing scalability for real2sim robotics.

---

### RQ2: Can PhysSplat's MLLM-based material estimation provide physically accurate parameters for Isaac Sim rigid/soft body simulation?

**The question:** When PhysSplat's MLLM-P3 module estimates physical properties (mass, friction, restitution, Young's modulus) from visual appearance, how do these estimates compare to ground-truth values, and do they produce physically plausible simulation outcomes in Isaac Sim's PhysX/Newton engine?

**Prior work context:**
- **PhysSplat (ICCV 2025):** Uses multimodal LLMs to predict mean physical properties in zero-shot. Estimates property distributions via geometry-conditioned sampling. Runs MPM simulation. Never validated against measured ground-truth properties or integrated with PhysX.
- **Phys2Real (2025):** Uses VLMs to estimate friction and center-of-mass for sim2real transfer. Showed 100% vs 60% success rate over domain randomization for pushing tasks. Limited to 2 objects (T-block, hammer), planar only.
- **Physics3D (2024):** Learns material properties via video diffusion priors. Handles elastic and plastic materials. Uses custom viscoelastic model, not standard PhysX material parameters.

**The gap:** PhysSplat and Phys2Real show VLMs can estimate some physics properties, but neither maps to the specific parameter set Isaac Sim needs (static/dynamic friction coefficients, restitution, density for rigid; Young's modulus, Poisson's ratio, damping for soft). Nobody has measured estimation accuracy against measured values for common household objects.

**Proposed method:** Build a lookup table of 20 common materials (wood, ceramic, plastic, rubber, metal, glass, fabric, etc.) with measured PhysX parameters from engineering references. Have PhysSplat's MLLM-P3 estimate properties for 30 objects spanning these materials. Compare MLLM estimates to reference values. Then run Isaac Sim simulations (drop test, slide test, grasp test) with both estimated and reference parameters. Measure: (a) parameter estimation error (% deviation from reference), (b) simulation outcome divergence (object final position error, energy conservation), (c) grasp success rate delta.

**Expected outcome:** MLLM estimates should be within 30-50% for density and friction (VLMs understand "heavy metal" vs "light plastic") but likely poor for precise values like Young's modulus. Simulation outcomes may still be acceptable if errors stay within PhysX's solver tolerance. Hypothesis: for rigid body manipulation (pick-and-place), 50% parameter error is tolerable; for deformable manipulation, it is not.

**Why it matters:** If VLM-estimated parameters are "good enough" for rigid body robotics, the entire label-to-physics pipeline can be automated without material testing equipment. This removes the last manual step in the GS-to-sim pipeline.

---

### RQ3: What is the end-to-end latency and fidelity of a complete GS scan -> segmented interactive objects -> Isaac Sim pipeline?

**The question:** Can we chain LangSplatV2 semantic segmentation -> per-object MILo mesh extraction -> PhysSplat material estimation -> CoACD decomposition -> Isaac Sim import into a single automated pipeline, and what are the cumulative error and time costs?

**Prior work context:**
- **LangSplatV2 (NeurIPS 2025):** 384.6 FPS open-vocabulary querying, 42x faster than LangSplat. Uses sparse coefficient field, eliminating heavyweight decoder. Produces accurate masks but never tested for object-level 3D instance segmentation required for mesh extraction.
- **DecoupledGaussian (CVPR 2025):** Full scene decomposition with physics. Uses joint Poisson fields to repair separated objects. Requires manual bounding box input. Physics uses simplified collision proxies.
- **Material-Informed GS (2025):** Camera-only pipeline: GS reconstruction -> semantic material masks -> mesh extraction -> material label projection -> physics properties. Closest to our target but focused on LiDAR simulation, not robotics manipulation.

**The gap:** Material-Informed GS comes closest but targets autonomous driving (large-scale, static). DecoupledGaussian handles decomposition but needs manual input and uses simplified physics. Nobody has built the full pipeline for tabletop robotics where contact accuracy matters.

**Proposed method:** Implement the pipeline on a standardized tabletop scene (5 objects on a table, captured from 100 views):
1. Train 3DGS + LangSplatV2 features (joint or sequential)
2. Query LangSplatV2 for object labels -> per-object Gaussian clusters
3. Run MILo mesh extraction on each cluster independently
4. Run PhysSplat MLLM-P3 for per-object material estimation
5. CoACD decomposition on each mesh
6. Import into Isaac Sim with estimated physics properties
7. Run grasp evaluation (10 grasps per object)

Measure end-to-end: total wall-clock time, per-stage error accumulation, final grasp success vs hand-authored scene baseline.

**Expected outcome:** Pipeline should complete in <30 min for 5 objects (LangSplatV2 training ~10 min, MILo ~5 min/object, rest trivial). Grasp success likely 15-25% below hand-authored baseline due to cumulative mesh quality and physics parameter errors. The critical question is whether the gap is small enough to be closed by domain randomization during policy training.

**Why it matters:** This is the full integration test. If the pipeline works at all -- even with degraded fidelity -- it proves the thesis that GS reconstructions can replace manual scene authoring for sim2real robotics.

---

### RQ4: Does per-Gaussian constitutive modeling (OmniPhysGS/PhysGaussian) outperform mesh-proxy physics for deformable object manipulation in simulation?

**The question:** For soft/deformable objects (cloth, fruit, stuffed toys), does simulating physics directly on Gaussians via MPM (as in OmniPhysGS) produce more accurate manipulation outcomes than the mesh-proxy approach (extract mesh -> FEM/PhysX soft body)?

**Prior work context:**
- **OmniPhysGS (ICLR 2025):** Extends 3DGS with learnable constitutive models. 12 domain-expert sub-models (rubber, metal, honey, water, etc.). Differentiable MPM. Produces visually realistic dynamics but uses custom MPM solver, not PhysX.
- **PhysGaussian (CVPR 2024 Highlight):** Original MPM-on-Gaussians. Custom CUDA MPM solver. No Isaac Sim integration. Handles continuum mechanics (elasticity, plasticity, viscosity) directly on Gaussian kernels.
- **MaGS (ICCV 2025):** Mesh-adsorbed Gaussians. Uses Blender/Taichi for physics, not PhysX. Supports ARAP editing, cloth simulation, soft body via mesh deformation driving Gaussian motion.
- **Physics-Informed Deformable GS (AAAI 2026):** Lagrangian material point formulation with time-evolving constitutive laws. Unified framework for fluid, elastic, cloth. Uses Cauchy momentum residual as physics constraint.

**The gap:** All Gaussian-native physics methods use custom solvers (MPM/Taichi), none use PhysX/Newton. Isaac Sim only supports PhysX rigid bodies and Newton (formerly Warp) for soft bodies via FEM on tetrahedral meshes. There is a fundamental solver incompatibility. Nobody has compared: (a) PhysGaussian-style MPM simulation quality vs (b) mesh-extracted + FEM simulation quality, for the same objects under manipulation forces.

**Proposed method:** Select 5 deformable objects (sponge, cloth napkin, rubber duck, foam ball, stuffed animal). For each: (A) run PhysGaussian MPM simulation with known forces (gravity drop, gripper squeeze), (B) extract mesh via MILo/SuGaR, convert to tetrahedral mesh (TetGen), run Isaac Sim Newton FEM with equivalent forces. Compare: deformation accuracy vs real-world video ground truth (optical flow + depth), computation time, visual rendering quality.

**Expected outcome:** MPM-on-Gaussians likely wins on visual fidelity (rendering directly from deformed Gaussians). Mesh-proxy FEM likely wins on integration with existing robotics stacks (Isaac Sim, ROS). The practical answer for robotics is probably: use mesh-proxy for rigid/semi-rigid, reserve Gaussian-native MPM for research on truly deformable manipulation.

**Why it matters:** Determines whether the field should push for PhysX/Newton integration of Gaussian-native physics (hard engineering) or accept mesh-proxy as "good enough" for robotics (pragmatic path).

---

### RQ5: Can language-grounded segmentation quality from LangSplatV2 match instance-level precision needed for collision mesh extraction?

**The question:** LangSplatV2 achieves state-of-the-art open-vocabulary querying at 384 FPS, but does its Gaussian-level segmentation precision (boundary accuracy, completeness) suffice for downstream mesh extraction, or do we need tighter methods like Object-Centric 2DGS or Gaussian Grouping?

**Prior work context:**
- **LangSplatV2 (NeurIPS 2025):** Sparse coefficient field splatting. 42x faster than LangSplat. Better mask quality (cleaner boundaries). Evaluated on semantic metrics (mIoU), never on instance segmentation quality for mesh extraction.
- **SceneSplat++ (NeurIPS 2025):** Benchmark with 49K scenes, 3.7x more semantic classes. Shows generalizable models beat per-scene optimization. Does not evaluate mesh extraction downstream.
- **Gaussian Grouping (ECCV 2024):** SAM-integrated, open-world segmentation. Lifts 2D masks to 3D. Supports editing (removal, inpainting). More instance-aware than LangSplat but slower.
- **Object-Centric 2DGS (2025):** Uses SAM2 for per-object masks, background removal, occlusion-aware pruning. Produces compact per-object models. Designed for isolated objects, not full scenes.

**The gap:** LangSplatV2 is fast but semantic (knows "mug" but may group two mugs together). Gaussian Grouping is instance-aware but slower. Object-Centric 2DGS is precise but needs per-object capture. Nobody has measured segmentation quality in terms of its impact on downstream mesh extraction (Hausdorff distance, watertight success rate) rather than just mIoU.

**Proposed method:** Reconstruct 3 scenes of increasing difficulty (isolated objects, cluttered tabletop, stacked/touching objects). Run 4 segmentation methods (LangSplatV2, Gaussian Grouping, Object-Centric 2DGS, FlashSplat) on each. For each segmented object cluster, extract mesh via MILo. Measure: (a) segmentation purity (% of Gaussians belonging to target object), (b) mesh quality (watertight, manifold, Chamfer distance to GT), (c) pipeline compatibility (time, automation, failure modes).

**Expected outcome:** LangSplatV2 likely sufficient for isolated objects but fails on touching/stacked objects where instance boundaries are ambiguous. Gaussian Grouping likely best for cluttered scenes. Object-Centric 2DGS impractical for multi-object scenes. The winning strategy is probably LangSplatV2 for initial semantic grouping + a lightweight instance refinement step (e.g., connected component analysis on Gaussian positions within a semantic cluster).

**Why it matters:** Segmentation quality is the first domino. If boundaries are wrong by even a few Gaussians, the extracted mesh will have artifacts that cascade through the entire pipeline (bad collision geometry -> bad grasps -> bad policy).

---

## Top Papers with Full Details

### 1. PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting

- **Venue:** ICCV 2025
- **Authors:** Zhao, Wang et al.
- **Key contribution:** MLLM-based Physical Property Perception (MLLM-P3) for zero-shot material estimation. Material Property Distribution Prediction (MPDP) via geometry-conditioned probabilistic sampling. Physical-Geometric Adaptive Sampling (PGAS) for efficient particle simulation.
- **Method:** Scene reconstruction -> open-vocabulary 3D segmentation -> multi-view inpainting -> MLLM predicts material properties -> MPDP estimates distributions -> MPM simulation with PGAS sampling.
- **Limitation:** Uses custom MPM, not PhysX. Material estimates not validated against measured ground truth. Single-GPU, ~2 min per scene.
- **Relevance to our gap:** Closest to automated label->physics. The MLLM-P3 module could be extracted and adapted to output Isaac Sim PhysX parameters instead of MPM constitutive parameters.
- **Links:** [arXiv](https://arxiv.org/abs/2411.12789) | [ICCV Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_PhysSplat_Efficient_Physics_Simulation_for_3D_Scenes_via_MLLM-Guided_Gaussian_ICCV_2025_paper.pdf)

### 2. MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction

- **Venue:** SIGGRAPH Asia 2025 (TOG)
- **Authors:** Guedon, Gomez, Maruani, Gong, Drettakis, Ovsjanikov (Ecole polytechnique / INRIA)
- **Key contribution:** Differentiable mesh extraction during GS training via Delaunay triangulation + Marching Tetrahedra. Bidirectional gradient flow between Gaussians and mesh. 10x fewer vertices than prior methods.
- **Method:** During every training iteration, mesh (vertices + connectivity) is differentiably extracted from Gaussian parameters. SDF values computed on tetrahedra edges, sign changes locate mesh vertices. Gradient flows from mesh loss back to Gaussians.
- **Limitation:** Mesh quality evaluated only on rendering metrics (PSNR, SSIM), not physics simulation suitability (watertightness, manifoldness, self-intersections). No convex decomposition evaluation.
- **Relevance to our gap:** Best current candidate for automated mesh extraction from GS. Same author as SuGaR (natural evolution). The in-the-loop approach should produce cleaner meshes than post-hoc extraction.
- **Links:** [Project Page](https://anttwo.github.io/milo/) | [GitHub](https://github.com/Anttwo/MILo) | [arXiv](https://arxiv.org/abs/2506.24096)

### 3. DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction

- **Venue:** CVPR 2025
- **Authors:** Wang et al.
- **Key contribution:** Separates foreground objects from background in wild-captured GS scenes. Joint Poisson fields repair geometry after separation. Multi-carve strategy refines object geometry. Supports collisions, fractures.
- **Method:** Scene GS -> manual bounding box selection -> object-scene separation -> Poisson field repair -> per-object physics properties (mass, collision boundaries) -> simulation (dropping, collisions, fractures).
- **Limitation:** Requires manual bounding box input (not automated). Physics uses simplified collision proxies. Complex geometries have inaccurate physics interactions.
- **Relevance to our gap:** Proves the decomposition-to-physics concept works. The Poisson field repair is critical -- after segmentation, object boundaries need healing. Their simplified collision proxy limitation is exactly what MILo could fix.
- **Links:** [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DecoupledGaussian_Object-Scene_Decoupling_for_Physics-Based_Interaction_CVPR_2025_paper.pdf) | [arXiv](https://arxiv.org/html/2503.05484v1)

### 4. LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS

- **Venue:** NeurIPS 2025
- **Authors:** Qin et al. (IBM Research contribution noted)
- **Key contribution:** Eliminates heavyweight decoder bottleneck. Each Gaussian acts as sparse code in global dictionary. Sparse coefficient splatting with CUDA optimization. 42x speedup, 47x query boost over LangSplat.
- **Method:** Learn 3D sparse coefficient field instead of dense language features. Efficient CUDA-optimized sparse coefficient splatting renders high-dimensional feature maps at cost of ultra-low-dimensional feature splatting.
- **Limitation:** Semantic-level, not instance-level segmentation. Two mugs would be labeled "mug" but not distinguished as "mug_1" and "mug_2". Evaluated on LERF, 3D-OVS, Mip-NeRF360 -- all semantic benchmarks, no instance segmentation metrics.
- **Relevance to our gap:** Speed makes it viable for real-time applications. Semantic labels are sufficient for material estimation (all "ceramic mugs" share properties). Instance disambiguation needed as separate step.
- **Links:** [Project Page](https://langsplat-v2.github.io/) | [arXiv](https://arxiv.org/html/2507.07136) | [NeurIPS Poster](https://neurips.cc/virtual/2025/poster/117503)

### 5. OmniPhysGS: 3D Constitutive Gaussians for General Physics-Based Dynamics Generation

- **Venue:** ICLR 2025
- **Authors:** Wang et al.
- **Key contribution:** Extends 3DGS with learnable per-Gaussian constitutive models. Ensemble of 12 domain-expert material sub-models. Differentiable MPM integration. Handles elastic, viscoelastic, plastic, fluid materials.
- **Method:** Static Gaussians from multi-view -> extend to Constitutive Gaussians with per-particle material as weighted ensemble of 12 sub-models (rubber, metal, honey, water, etc.) -> differentiable MPM simulation -> rendered via GS.
- **Limitation:** Custom MPM solver, not PhysX-compatible. Material models are learned from visual appearance (video diffusion priors), not measured. 12 sub-model ensemble may not cover all materials. Not evaluated for robotics manipulation accuracy.
- **Relevance to our gap:** The per-Gaussian constitutive model is the right abstraction for heterogeneous objects. The 12-model ensemble could be mapped to PhysX material presets if someone builds the bridge.
- **Links:** [Project Page](https://wgsxm.github.io/projects/omniphysgs/) | [GitHub](https://github.com/wgsxm/OmniPhysGS) | [arXiv (ICLR)](https://arxiv.org/pdf/2501.18982)

---

## Complete Literature Map

### Category 1: Semantic Segmentation of Gaussians

| Paper | Venue | Speed | Granularity | Instance-Aware? |
|-------|-------|-------|-------------|-----------------|
| LangSplat | CVPR 2024 | 199x > LERF | Semantic | No |
| LangSplatV2 | NeurIPS 2025 | 384 FPS query | Semantic | No |
| FlashSplat | ECCV 2024 | ~30 sec | Per-object | Semi (user click) |
| Gaussian Grouping | ECCV 2024 | Minutes | Instance | Yes (SAM) |
| Object-Centric 2DGS | VISAPP 2025 | Per-object training | Instance | Yes (SAM2) |
| SceneSplat++ | NeurIPS 2025 | Benchmark only | Semantic | Dataset only |
| ObjectGS | 2025 | Per-scene | Instance | Yes (SAM) |

### Category 2: Mesh Extraction from Gaussians

| Paper | Venue | Method | Vertex Count | Watertight? |
|-------|-------|--------|-------------|-------------|
| SuGaR | CVPR 2024 | Poisson recon | High | Sometimes |
| MILo | SIGGRAPH Asia 2025 | Delaunay+MarchTet | 10x fewer | Likely (SDF-based) |
| GS2Mesh | ECCV 2024 | Stereo depth | Medium | No |
| GS-2M | arXiv Sep 2025 | PGSR-based + shading | Medium | Yes (material-aware) |
| GS-Verse | arXiv Oct 2025 | SuGaR/Trellis + GaMeS | Variable | Depends on source |

### Category 3: Physics on Gaussians

| Paper | Venue | Physics Engine | Material Model | Isaac Sim? |
|-------|-------|---------------|----------------|------------|
| PhysGaussian | CVPR 2024 | Custom MPM (CUDA) | Continuum mechanics | No |
| OmniPhysGS | ICLR 2025 | Differentiable MPM | 12-model ensemble | No |
| PhysSplat | ICCV 2025 | MPM + MLLM props | MLLM-estimated | No |
| Physics3D | 2024 | Video diffusion + MPM | Viscoelastic | No |
| GASP | 2024 | Black-box engine | Flat Gaussians -> triangles | Partial |
| MaGS | ICCV 2025 | Blender/Taichi | Mesh-driven | No |
| PhysInformed DGS | AAAI 2026 | Cauchy momentum | Lagrangian material field | No |
| Embodied Gaussians | CoRL 2024 | Particle-based | Visual forces | No |
| PhysTwin | ICCV 2025 | Spring-mass + inverse | Learned from video | No |

### Category 4: Decomposition + Physics Pipelines

| Paper | Venue | Automation | Physics Quality | Target App |
|-------|-------|-----------|----------------|------------|
| DecoupledGaussian | CVPR 2025 | Semi (bbox) | Simplified proxy | VR/media |
| Material-Informed GS | arXiv Nov 2025 | Automated | Material labels -> reflectivity | Autonomous driving |
| Phys2Real | 2025 | Semi (VLM) | VLM-estimated friction, CoM | Robotic pushing |
| RL-GSBridge | ICRA 2025 | Semi | Soft mesh binding | Robotic grasping |
| GS-Verse | arXiv 2025 | Semi | Unity physics | VR interaction |
| RoboGSim | 2024 | Semi | Simplified | Robot learning |

---

## Feasibility on RTX 4070 Ti SUPER

**GPU Specs:** 16 GB GDDR6X VRAM, 8448 CUDA cores, Ada Lovelace architecture, ~44 TFLOPS FP32

### Per-Stage Memory and Time Estimates

| Pipeline Stage | VRAM Required | Wall Time | Feasible? |
|---------------|--------------|-----------|-----------|
| 3DGS Training (100 views, single scene) | 6-10 GB | 15-30 min | YES |
| LangSplatV2 (add language features) | +2-4 GB on top of 3DGS | +5-10 min | YES |
| MILo mesh extraction (per object) | ~4-6 GB (reuses GS) | 5-10 min/obj | YES |
| PhysSplat MLLM-P3 (material estimation) | ~8-12 GB (LLM inference) | 1-2 min/obj | TIGHT -- need quantized MLLM |
| CoACD convex decomposition | CPU only, <1 GB RAM | <30 sec/obj | YES |
| Isaac Sim scene import + PhysX | 4-8 GB (separate from training) | Seconds | YES |
| OmniPhysGS/PhysGaussian MPM sim | 8-14 GB (MPM grid + Gaussians) | Minutes | TIGHT for large scenes |

### Critical Constraints

1. **Cannot run 3DGS training and Isaac Sim simultaneously** -- both need 6+ GB VRAM. Must be sequential pipeline stages.
2. **MLLM inference for PhysSplat** requires a quantized model (4-bit or 8-bit) to fit in 16 GB alongside the GS representation. Alternatively, offload MLLM to CPU (slower but feasible).
3. **MPM simulation** (PhysGaussian/OmniPhysGS) is VRAM-intensive for the grid. Limit to <500K Gaussians per object. Tabletop objects typically 50K-200K Gaussians, so this is fine.
4. **Scene scale matters:** A 5-object tabletop scene with ~500K total Gaussians is well within budget. A full room scan with 2M+ Gaussians will need host memory offloading (GS-Scale technique).

### Verdict: FEASIBLE with constraints

The full pipeline is feasible on RTX 4070 Ti SUPER 16GB if:
- Stages are run sequentially (not simultaneously)
- MLLM for material estimation uses quantized weights or API calls
- Scene complexity limited to tabletop scale (~500K Gaussians)
- MPM physics simulation limited to single-object deformation (not full-scene)

---

## Minimal Experiment Design

### Phase 1: Mesh Quality Baseline (1-2 weeks)

**Goal:** Answer RQ1 -- can automated mesh extraction produce PhysX-compatible collision geometry?

**Setup:**
1. Select 5 objects from YCB dataset (known CAD ground truth): mug, banana, bowl, drill, foam brick
2. Capture 50 views per object (phone camera, turntable)
3. Train 3DGS for each object (~30 min each)
4. Extract meshes via: (a) SuGaR, (b) MILo (if code released; else use GS2Mesh as proxy)
5. Run ManifoldPlus -> CoACD -> Isaac Sim import
6. Measure: watertight success, Chamfer distance to YCB CAD, resting stability in PhysX

**Deliverable:** Table of mesh quality metrics per method per object. Go/no-go decision on mesh extraction approach.

### Phase 2: Segmentation -> Mesh Pipeline (1-2 weeks)

**Goal:** Answer RQ5 -- does segmentation quality survive through mesh extraction?

**Setup:**
1. Create 2 multi-object scenes (3 objects each, known arrangement)
2. Train 3DGS + LangSplatV2 features
3. Segment objects via language query
4. Extract per-object meshes
5. Import into Isaac Sim alongside ground-truth CAD models
6. Compare: per-object Chamfer distance, grasp success (10 trials)

**Deliverable:** Quantified error from segmentation boundary noise on downstream mesh quality.

### Phase 3: Full Pipeline Integration (2-3 weeks)

**Goal:** Answer RQ3 -- end-to-end pipeline performance

**Setup:**
1. Capture novel scene (not from YCB -- real kitchen objects)
2. Run full pipeline: GS -> LangSplatV2 -> per-object MILo -> PhysSplat material -> CoACD -> Isaac Sim
3. Run identical scene hand-authored in Isaac Sim as baseline
4. Train simple grasp policy (PPO, 1000 episodes) in both scenes
5. Compare: grasp success rate, policy training convergence, sim2real transfer (if robot available)

**Deliverable:** End-to-end comparison demonstrating feasibility and quantifying the fidelity gap.

---

## Biggest Risk + Mitigation

### BIGGEST RISK: Mesh quality from GS extraction is insufficient for contact-rich manipulation

**Why this is the top risk:**

Every downstream step depends on mesh quality. If extracted meshes have:
- Non-manifold edges -> CoACD crashes
- Holes -> PhysX interpenetration
- Over-smoothed geometry -> gripper slides off
- Noisy surfaces -> unstable physics

Then the entire pipeline fails regardless of how good segmentation or material estimation is.

**Evidence this risk is real:**
- SuGaR (CVPR 2024) produces meshes with known holes and non-manifold edges (multiple GitHub issues)
- MILo is newer and SDF-based (theoretically cleaner) but has zero physics evaluation
- DecoupledGaussian (CVPR 2025) explicitly states "simplified collision proxies may not perfectly match complex object geometries"
- No paper in the literature has measured GS-extracted mesh quality in terms of PhysX simulation stability

**Mitigation Strategy (3-tier):**

1. **Tier 1: Automated repair** (low effort, moderate reliability)
   - ManifoldPlus for watertight repair
   - MeshFix for non-manifold edge removal
   - Automatic pipeline: MILo -> ManifoldPlus -> MeshFix -> CoACD
   - Expected to handle 60-70% of objects

2. **Tier 2: Hybrid representation** (medium effort, high reliability)
   - Use GS-extracted mesh for visual rendering
   - Use simplified convex hull (or SDF-sampled mesh) for collision
   - Accept visual-collision mismatch up to 5mm tolerance
   - This is what DecoupledGaussian already does, and what most game engines do

3. **Tier 3: Primitive fitting fallback** (low effort, universal but crude)
   - When mesh extraction fails, fit primitive shapes (box, sphere, cylinder, capsule) to Gaussian cluster bounding geometry
   - PhysX natively supports primitive colliders with zero mesh issues
   - Accuracy limited but guaranteed to work
   - Sufficient for pick-and-place; insufficient for dexterous manipulation

**Decision rule:** Try Tier 1. If watertight success rate > 80%, proceed. If 50-80%, add Tier 2 as hybrid. If < 50%, fall back to Tier 3 and scope research to pick-and-place only.

---

## Summary: The State of the Field in One Paragraph

As of February 2026, every individual component of the GS-scan-to-interactive-simulation pipeline exists: LangSplatV2 segments at 384 FPS, MILo extracts meshes 10x more efficiently, PhysSplat estimates material properties via MLLM, CoACD decomposes for PhysX, and Isaac Sim accepts the result. The fundamental gap remains integration and quality propagation. No paper has connected these stages end-to-end for robotics, and the cumulative error from segmentation boundaries -> mesh extraction artifacts -> approximate physics parameters -> PhysX solver tolerance is completely uncharacterized. The five research questions above systematically probe each interface in this chain. The practical bet is that rigid-body tabletop manipulation will work within 6 months of focused engineering, while deformable manipulation requires a solver bridge (MPM <-> PhysX) that does not yet exist.
