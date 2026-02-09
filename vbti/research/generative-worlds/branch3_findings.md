# BRANCH 3: Compositional Scene Generation with Interactable Objects

## Deep Literature Review & Refined Research Questions

**Date**: 2026-02-06
**Scope**: Text/image-to-3D Gaussian generation, physics-ready asset pipelines, compositional scene building for robotics simulation

---

## TABLE OF CONTENTS

1. [Refined Research Questions (Primary Output)](#refined-research-questions)
2. [Top Papers with Full Details](#top-papers)
3. [Extended Literature Map](#extended-literature-map)
4. [Feasibility on RTX 4070 Ti SUPER](#feasibility-analysis)
5. [Minimal Experiment Design](#minimal-experiment-design)
6. [Risks and Mitigations](#risks-and-mitigations)
7. [Venue/Deadline Analysis](#venue-deadline-analysis)

---

## 1. REFINED RESEARCH QUESTIONS <a name="refined-research-questions"></a>

### RQ1: Can text-prompted generative 3D assets be automatically made sim-ready for contact-rich robotic manipulation?

**The Question**: Given a text prompt (e.g., "ripe red tomato, 8cm diameter"), can we generate a 3D Gaussian Splat, extract a watertight collision mesh, assign physically plausible properties (mass, friction, deformability), and deploy the resulting asset in Isaac Sim such that a manipulation policy can successfully grasp it?

**Prior Work Context**:
- **PhysX-Anything** (Nov 2025): First framework producing simulation-ready assets from a single image with URDF/SDF output for MuJoCo/Isaac Gym. However, it requires an *image* input, not text, and targets articulated objects (doors, drawers) rather than graspable produce-scale objects. No evaluation of grasp success rates.
- **DreamGaussian** (ICLR 2024): Text/image to 3D Gaussian + mesh in ~2 minutes. Mesh extraction exists but produces oversaturated, oversmoothed geometry. No physics properties, no collision mesh quality evaluation. The extracted mesh is for rendering, not simulation.
- **PhysX-3D / PhysXGen** (NeurIPS 2025 Spotlight): First physics-grounded 3D dataset (PhysXNet) with scale, material, affordance, kinematics annotations. Feed-forward image-to-3D with physical property prediction. Closest to the gap, but still image-conditioned and not evaluated for grasp feasibility in simulation.

**The Specific Gap**: Nobody has closed the loop: text prompt -> generate GS -> extract collision-quality mesh -> assign physics -> place in scene -> verify graspable. PhysX-Anything and PhysXGen assign physics but do not verify manipulation success. DreamGaussian generates from text but has no physics. The pipeline is fragmented across 3-4 separate papers.

**Proposed Method**: Build a pipeline that chains: (1) text-to-3D generation via DreamGaussian or GaussianDreamerPro, (2) mesh extraction with explicit collision mesh simplification (convex decomposition via V-HACD or CoACD), (3) physics property assignment using PhysXGen's dual-branch predictor or MLLM-based estimation (a la PhysSplat), (4) URDF/USD export and Isaac Sim integration, (5) grasp success evaluation using a trained policy or analytical grasp planner. Measure grasp success rate as the primary metric, comparing against artist-made SimReady assets as the gold standard.

**Expected Outcome**: If the pipeline achieves >60% grasp success rate (vs. ~90% for artist-made assets), this confirms that generative assets are viable for training diversity, even if imperfect. If <30%, the mesh quality bottleneck is confirmed and the research pivots to mesh refinement.

**Why It Matters**: Sim2Real transfer for manipulation requires massive object diversity. Manually creating SimReady assets costs $50-200 per object. A working text-to-sim-ready pipeline would reduce this to seconds and enable parametric variation (size, color, shape) for domain randomization at scale.

---

### RQ2: Does parametric variation of generated 3D objects (size, color, shape, deformability) improve sim-to-real manipulation policy transfer compared to static asset libraries?

**The Question**: If we generate N parametric variants of a target object class (e.g., tomatoes varying in size 4-12cm, color green-to-red, shape round-to-oblong, deformability soft-to-firm), does training a manipulation policy on these variants improve real-world grasp success rate compared to training on a fixed set of 3D-scanned assets?

**Prior Work Context**:
- **OnePoseviaGen** (CoRL 2025 Oral): Uses text-guided generative domain randomization for 6D pose estimation. Generates diverse textures via text prompts on a single 3D model. Achieves SOTA pose estimation but only varies texture, not geometry/physics.
- **RoboSplat** (RSS 2025): Swaps GS scans of real objects for data augmentation in one-shot manipulation. Achieves 87.8% real-world success vs 57.2% with 2D augmentation. But limited to objects you can physically scan -- no generation of novel shapes.
- **SplatSim** (CMU, 2024): Foreground/background GS composition for sim2real. 86.25% success in zero-shot transfer (vs 97.5% real data). Does not vary object properties.

**The Specific Gap**: OnePoseviaGen varies texture but not shape/physics. RoboSplat replaces objects but only with real scans. Nobody has tested whether *generatively varying* object geometry and physics properties improves downstream manipulation transfer.

**Proposed Method**: (1) Select 3 target object classes (e.g., tomato, pepper, apple for agriculture). (2) For each class, generate 50 parametric variants using conditioned 3D generation (vary size via scale, shape via prompt interpolation, color via texture editing, deformability via material property ranges). (3) Train identical manipulation policies on: (a) 5 fixed 3D scans, (b) 50 generated variants, (c) 50 generated + domain randomization. (4) Evaluate zero-shot sim-to-real transfer on 10 real instances per class. Primary metric: real-world grasp success rate.

**Expected Outcome**: We hypothesize a 15-25% improvement in real-world success rate for the generated-variants condition over fixed assets, based on the magnitude of improvement RoboSplat showed (30% gain from GS augmentation). If the improvement is <5%, parametric generation is not worth the pipeline complexity.

**Why It Matters**: Agriculture robotics needs to handle natural variation (every tomato is different). If generative parametric variation closes the sim-to-real gap as effectively as scanning real objects, it eliminates the need for physical asset collection entirely.

---

### RQ3: Can lighting-consistent compositional insertion of generated GS objects into GS scenes maintain sufficient visual fidelity for RGB-based policy learning?

**The Question**: When a generated GS object is inserted into a reconstructed GS scene, does the composite rendering maintain sufficient photorealism for an RGB-based manipulation policy to transfer to the real world? Specifically, do lighting/shadow inconsistencies from naive insertion degrade policy performance, and does lighting-aware insertion (D3DR, ComGS) recover it?

**Prior Work Context**:
- **D3DR** (March 2025): Diffusion-driven lighting correction for GS object insertion. Achieves +0.5 PSNR, +0.15 SSIM over baselines. But evaluated only on visual metrics, not on downstream task performance.
- **ComGS** (2025): Three-stage composition (reconstruct, edit, render) with Surface Octahedral Probes for real-time shadow casting at ~28 FPS. Efficient but no robotics evaluation.
- **FreeInsert** (May 2025): Text-guided object insertion using MLLM parsing. No spatial priors needed. But focused on visual editing, not simulation.
- **SplatSim** (2024): Demonstrated that GS rendering quality matters for sim2real -- 86.25% success rate. But used real scans, not generated objects, and did not test lighting consistency.

**The Specific Gap**: D3DR, ComGS, and FreeInsert solve the visual consistency problem for GS composition, but none evaluate whether this visual consistency matters for downstream policy learning. SplatSim shows GS rendering helps sim2real, but does not test compositional scenes with inserted objects.

**Proposed Method**: (1) Reconstruct a real workspace as a GS background scene. (2) Insert a target object using three methods: (a) naive insertion (no lighting correction), (b) D3DR lighting-aware insertion, (c) ComGS with shadow casting. (3) Train identical diffusion policies on images rendered from each condition. (4) Evaluate zero-shot sim2real transfer. Secondary analysis: measure FID between rendered composites and real images to correlate visual fidelity with policy performance.

**Expected Outcome**: We expect lighting-aware insertion to improve policy transfer by 10-20% over naive insertion, based on the visual fidelity improvements reported. If the improvement is negligible (<5%), it means policies are robust to lighting artifacts and naive composition suffices (which would be a useful negative result).

**Why It Matters**: If lighting consistency matters for policy learning, the field needs to integrate D3DR/ComGS-style methods into sim2real pipelines. If it doesn't, the simpler naive insertion approach saves significant compute and complexity.

---

### RQ4: Can World Labs Marble + automatic physics annotation replace manual scene authoring for robotic simulation environments?

**The Question**: Using Marble's text-to-3D-world generation with its exported collider meshes, combined with automatic physics property annotation (PhysSplat-style MLLM estimation), can we generate simulation environments that produce policy performance comparable to manually authored Isaac Sim environments?

**Prior Work Context**:
- **World Labs Marble** (Nov 2025, API Jan 2026): Generates explorable 3D worlds from text/image prompts. Exports PLY (Gaussian splats) + GLB (collider meshes). Direct integration with Isaac Sim demonstrated by NVIDIA. Produces both low-fidelity collider meshes and high-quality visual meshes.
- **PhysSplat** (ICCV 2025): MLLM-guided physics property estimation for GS objects. Zero-shot material prediction. Runs on single GPU in ~2 minutes.
- **NVIDIA Edify 3D** (2024, discontinued as NIM preview June 2025): Generated PBR materials and clean topology, but is no longer available as a standalone service.

**The Specific Gap**: Marble generates environments with collider meshes but no physics properties (mass, friction, material type). PhysSplat estimates physics but only for individual objects, not full scenes. Nobody has combined generative scene creation + automatic physics annotation + policy training evaluation.

**Proposed Method**: (1) Use Marble API to generate 20 kitchen/agricultural scene variants from text prompts. (2) Export PLY + GLB. (3) Segment individual objects in the scene. (4) Apply PhysSplat/MLLM-based physics estimation per object. (5) Import into Isaac Sim with NuRec rendering. (6) Train navigation and manipulation policies. (7) Compare against 5 manually authored scenes. Primary metrics: policy success rate, time-to-environment-creation.

**Expected Outcome**: We expect Marble-generated environments to achieve 70-85% of the policy performance of manually authored scenes, but at 100x faster creation speed (hours vs weeks). The physics annotation step is the main risk -- if MLLM-estimated physics are too inaccurate, object interactions will be unrealistic.

**Why It Matters**: If generative scene creation reaches "good enough" quality for policy training, it fundamentally changes the economics of sim2real research. Instead of spending weeks building each environment, researchers can generate hundreds of diverse scenes overnight.

---

### RQ5: What is the minimal mesh quality threshold for generated 3D objects to support reliable grasp planning in simulation?

**The Question**: How does collision mesh quality (measured by Hausdorff distance from ground truth, watertightness, convex decomposition piece count) correlate with grasp success rate in simulation? Is there a quality threshold below which grasping fails catastrophically?

**Prior Work Context**:
- **DreamGaussian** (ICLR 2024): Extracts meshes via local density querying. Known to produce oversaturated, oversmoothed geometry. No quantitative mesh quality evaluation for manipulation.
- **GaussianDreamerPro** (2024): Binds Gaussians to mesh for enhanced quality. Targets animation/simulation integration. No grasp evaluation.
- **MaGS** (ICCV 2025): Mesh-adsorbed Gaussians with physics simulation support (soft body, ARAP, cloth). Best mesh-GS binding, but focused on dynamic reconstruction, not generative assets.

**The Specific Gap**: None of these papers measure how mesh quality affects downstream manipulation. We do not know: Does a 5mm Hausdorff error kill grasping? Does 10 convex pieces suffice or do we need 100? Is watertightness strictly necessary?

**Proposed Method**: (1) Take 20 real objects with ground-truth 3D scans. (2) Generate 3D assets of the same objects via DreamGaussian and GaussianDreamerPro. (3) Extract meshes at multiple quality levels (varying marching cubes resolution, convex decomposition granularity). (4) For each quality level, run 100 grasp attempts in Isaac Sim using an analytical grasp planner. (5) Plot grasp success rate vs mesh quality metrics. (6) Identify the knee point where quality improvements stop mattering.

**Expected Outcome**: We expect a sigmoid-shaped relationship: below some threshold, grasping fails completely; above it, marginal quality improvements don't help. Identifying this threshold directly informs the minimum generation quality needed.

**Why It Matters**: This answers the practical question: "How good does my generated mesh need to be?" If the threshold is low (e.g., 5mm error, 20 convex pieces), current generators suffice. If high (e.g., 0.5mm, 200 pieces), significant mesh refinement research is needed.

---

## 2. TOP 5 PAPERS WITH FULL DETAILS <a name="top-papers"></a>

### Paper 1: PhysX-Anything -- Simulation-Ready Physical 3D Assets from Single Image

- **Authors**: Ziang Cao et al.
- **Venue**: arXiv, November 2025 (preprint)
- **Link**: [arXiv:2511.13648](https://arxiv.org/abs/2511.13648)
- **Code**: [GitHub](https://github.com/ziangcao0312/PhysX-Anything)
- **Key Contribution**: First framework that produces sim-ready 3D assets (explicit shape, articulation, physics) from a single image. Outputs URDF, SDF, glTF, MJCF formats directly. Introduces PhysX-Mobility dataset (2,063 instances, 47 categories).
- **Method**: VLM-based physical 3D generative model with 193x token reduction for efficient geometry tokenization. Predicts articulation structure, joint types, material properties.
- **Limitations**: Image-conditioned only (no text input). Focused on articulated objects (cabinets, laptops), not graspable produce-scale items. Physics properties not verified via manipulation experiments.
- **Relevance to Branch 3**: Closest existing work to the full pipeline. The physics prediction and URDF export modules could be directly reused. Gap: needs text conditioning and grasp verification.

### Paper 2: PhysX-3D / PhysXGen -- Physical-Grounded 3D Asset Generation

- **Authors**: Ziang Cao, Zhaoxi Chen et al.
- **Venue**: NeurIPS 2025 (Spotlight)
- **Link**: [arXiv:2507.12465](https://arxiv.org/abs/2507.12465)
- **Code**: [GitHub](https://github.com/ziangcao0312/PhysX-3D)
- **Key Contribution**: PhysXNet dataset with physics annotations across 5 dimensions (absolute scale, material, affordance, kinematics, function). PhysXGen feed-forward framework with dual-branch architecture for joint geometry+physics prediction.
- **Method**: Human-in-the-loop annotation pipeline using VLMs. Dual-branch architecture injects physical knowledge into pre-trained 3D generation model.
- **Limitations**: Image-conditioned. Physics annotations are predicted, not verified in simulation. Focus on static properties, not deformability.
- **Relevance to Branch 3**: PhysXNet dataset could serve as training data for physics property prediction in our pipeline. The dual-branch architecture is a strong baseline for physics-aware generation.

### Paper 3: PhysSplat -- Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting

- **Authors**: Haoyu Zhao, Hao Wang et al.
- **Venue**: ICCV 2025
- **Link**: [arXiv:2411.12789](https://arxiv.org/abs/2411.12789)
- **Key Contribution**: MLLM-based zero-shot physics property prediction (MLLM-P3) for GS objects. Material Property Distribution Prediction (MPDP) model. Physical-Geometric Adaptive Sampling (PGAS) for efficient simulation.
- **Method**: Uses multi-modal large language models to predict mean physical properties of objects from visual appearance. Reformulates physics estimation as probability distribution sampling. Runs on single GPU in ~2 minutes.
- **Limitations**: Predicts distributions, not exact values. Evaluated on visual realism of simulated dynamics, not manipulation success. Physics estimation accuracy not benchmarked against ground truth materials.
- **Relevance to Branch 3**: MLLM-P3 module could be directly used for automatic physics annotation of generated objects or Marble-generated scenes.

### Paper 4: ComGS -- Efficient 3D Object-Scene Composition via Surface Octahedral Probes

- **Authors**: NJU-3DV group
- **Venue**: arXiv, October 2025
- **Link**: [arXiv:2510.07729](https://arxiv.org/abs/2510.07729)
- **Project**: [NJU-3DV ComGS](https://nju-3dv.github.io/projects/ComGS/)
- **Key Contribution**: Surface Octahedral Probes (SOPs) for efficient lighting estimation and occlusion caching. Real-time shadow computation in Gaussian scenes (~28 FPS). 2x speedup over ray-tracing methods.
- **Method**: Three-stage pipeline: (1) reconstruct GS scene + relightable GS object, (2) estimate lighting + cache occlusion with SOPs, (3) splatting + relighting + shadow casting + depth compositing.
- **Limitations**: Requires multi-view input for both scene and object. No evaluation for robotics/policy learning. Focused on visual quality only.
- **Relevance to Branch 3**: The composition pipeline could be used for RQ3 (lighting-consistent insertion). SOPs provide efficient relighting that could enable real-time training data generation.

### Paper 5: World Labs Marble -- Generative 3D World Model for Robotics Simulation

- **Authors**: World Labs (Fei-Fei Li et al.)
- **Venue**: Commercial product, API released January 2026
- **Link**: [World Labs](https://www.worldlabs.ai/)
- **NVIDIA Integration Blog**: [NVIDIA Developer Blog](https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble)
- **Key Contribution**: Text/image/video to explorable 3D worlds with depth, lighting, geometry, and exportable collider meshes. Direct Isaac Sim integration via PLY + GLB export -> USD conversion -> NuRec rendering.
- **Capabilities**: Generates both low-fidelity collider meshes (for physics) and high-quality visual meshes. Supports quadruped navigation, manipulation, collaborative robotics scenarios.
- **Limitations**: Collider meshes are coarse. No per-object physics properties (mass, friction). Scene segmentation for individual object extraction is non-trivial. API is commercial (pricing tiers).
- **Relevance to Branch 3**: Could be the primary scene generation backbone for RQ4. The collider mesh export + Isaac Sim integration is already demonstrated. Gap: needs per-object physics annotation.

---

## 3. EXTENDED LITERATURE MAP <a name="extended-literature-map"></a>

### 3A. Text/Image to 3D Generation

| Paper | Venue | Input | Output | Time | Mesh? | Physics? |
|-------|-------|-------|--------|------|-------|----------|
| DreamGaussian | ICLR 2024 Oral | Text/Image | 3DGS + Mesh + Texture | ~2 min | Yes (density query) | No |
| GaussianDreamer | CVPR 2024 | Text | 3DGS | ~15 min | No | No |
| GaussianDreamerPro | arXiv 2024 | Text | 3DGS + Mesh | ~15 min | Yes (bound) | No |
| NVIDIA Edify 3D | arXiv 2024 | Text/Image | Mesh + PBR + UV | ~2 min | Yes (clean topo) | PBR only |
| PhysX-3D / PhysXGen | NeurIPS 2025 | Image | 3D + Physics | Feed-forward | Yes | Scale, material, affordance, kinematics |
| PhysX-Anything | arXiv Nov 2025 | Image | URDF/SDF/MJCF | N/A | Yes | Articulation + physics |

### 3B. GS Composition and Insertion

| Paper | Venue | Method | Lighting? | Shadows? | Real-time? |
|-------|-------|--------|-----------|----------|------------|
| D3DR | arXiv Mar 2025 | Diffusion-based DDS optimization | Yes | Partial | No |
| ComGS | arXiv Oct 2025 | Surface Octahedral Probes | Yes | Yes | Yes (~28 FPS) |
| FreeInsert | arXiv May 2025 | MLLM-guided text insertion | Partial | No | No |
| MVInpainter | ScienceDirect 2025 | Multi-view diffusion | Yes | Implicit | No |

### 3C. GS for Robotics / Sim2Real

| Paper | Venue | Application | Success Rate | Key Innovation |
|-------|-------|-------------|-------------|----------------|
| SplatSim | CoRL 2024 WS | Sim2Real manipulation | 86.25% | GS as rendering primitive in sim |
| RoboSplat | RSS 2025 | One-shot manipulation | 87.8% | GS replacement augmentation |
| RoboGSim | arXiv 2024 | Real2Sim2Real | N/A | Full GS simulation pipeline |
| GWM | ICCV 2025 | World models for manipulation | N/A | Action-conditioned 3D prediction |

### 3D. Physics-Aware GS

| Paper | Venue | Physics Type | Method |
|-------|-------|-------------|--------|
| PhysSplat | ICCV 2025 | Material properties | MLLM zero-shot prediction |
| PhysGaussian | CVPR 2024 | Generative dynamics | Physics-integrated 3DGS |
| MaGS | ICCV 2025 | Soft body, ARAP, cloth | Mesh-adsorbed GS + simulation |
| Ref-Gaussian | ICLR 2025 | PBR / Relighting | 2D Gaussian + deferred shading |

### 3E. Domain Randomization with Generative Models

| Paper | Venue | Application | Variation Type |
|-------|-------|-------------|---------------|
| OnePoseviaGen | CoRL 2025 Oral | 6D pose estimation | Texture randomization via text |
| CropCraft | 2024 | Agriculture | Procedural plant morphology |
| Agricultural SDG | arXiv 2024 | Crop detection | Blender procedural generation |

---

## 4. FEASIBILITY ON RTX 4070 Ti SUPER (16GB VRAM) <a name="feasibility-analysis"></a>

### Component-by-Component Analysis

| Component | VRAM Required | Feasible? | Notes |
|-----------|--------------|-----------|-------|
| DreamGaussian generation | ~8-10 GB | Yes | Runs on single GPU, designed for consumer hardware |
| GaussianDreamerPro | ~12-14 GB | Tight | May need gradient checkpointing or reduced batch |
| PhysSplat MLLM-P3 | ~8-10 GB | Yes | Uses quantized MLLM (e.g., LLaVA-7B in 4-bit) |
| ComGS composition | ~6-8 GB | Yes | SOPs are memory-efficient by design |
| D3DR diffusion optimization | ~10-12 GB | Yes | Standard diffusion model + GS optimization |
| Isaac Sim rendering | ~4-6 GB (headless) | Yes | Headless mode for training data generation |
| Policy training (SmolVLA-scale) | ~10-14 GB | Yes | Already running locally per MEMORY.md |
| Marble API | 0 (cloud) | Yes | API call, no local compute |
| Mesh extraction + V-HACD | ~1-2 GB CPU | Yes | CPU-bound, negligible GPU |
| PhysXGen inference | ~10-12 GB | Tight | Dual-branch architecture, may need FP16 |

### Overall Assessment

**Feasible with careful sequencing.** The pipeline cannot run all components simultaneously, but each individual component fits in 16GB. The workflow is naturally sequential (generate -> extract mesh -> assign physics -> compose -> render -> train), so GPU memory can be reused between stages.

**Key constraints**:
- Run generation and training in separate sessions (not simultaneously)
- Use FP16/BF16 for all diffusion-based components
- Quantize MLLM to 4-bit for PhysSplat inference
- Use headless Isaac Sim for rendering
- Marble generation is cloud-based, no local GPU needed

**Estimated time per experiment**: ~4-6 hours for full pipeline (generation + composition + 1000-step policy training). Reasonable for iterative research.

---

## 5. MINIMAL EXPERIMENT DESIGN <a name="minimal-experiment-design"></a>

### Phase 1: Mesh Quality Threshold (RQ5) -- 2 weeks

**Goal**: Establish minimum mesh quality for grasping before building the full pipeline.

1. Select 5 objects from YCB dataset (known ground-truth meshes)
2. Generate each with DreamGaussian (text prompt)
3. Extract meshes at 3 quality levels (low/med/high marching cubes resolution)
4. Run 50 grasp attempts per object per quality level in Isaac Sim
5. Plot success rate vs Hausdorff distance

**Deliverable**: Quality threshold number, go/no-go for full pipeline.

### Phase 2: Text-to-Sim-Ready Pipeline (RQ1) -- 3 weeks

**Goal**: Build and validate the end-to-end pipeline.

1. Implement: DreamGaussian -> mesh extraction -> V-HACD convex decomposition -> URDF export
2. Add physics annotation: use PhysSplat's MLLM-P3 or manual lookup table for common objects
3. Import into Isaac Sim, test with analytical grasp planner
4. Benchmark: grasp success rate on 10 generated objects vs 10 YCB ground truth

**Deliverable**: Working pipeline, comparative grasp success rates.

### Phase 3: Parametric Variation (RQ2) -- 3 weeks

**Goal**: Test whether variation helps policy transfer.

1. Select 3 object classes (tomato, apple, pepper)
2. Generate 20 variants per class (vary text prompt parameters)
3. Train SmolVLA on: (a) 3 fixed assets, (b) 20 generated variants per class
4. Evaluate sim-to-real on real objects (if hardware available) or sim-to-sim generalization

**Deliverable**: Quantitative comparison, paper-ready results.

### Phase 4: Composition Quality (RQ3) -- 2 weeks (optional, time-permitting)

**Goal**: Test if lighting-consistent composition matters.

1. Reconstruct 1 workspace as GS scene
2. Insert 5 objects via: (a) naive, (b) D3DR
3. Train policy on each, compare sim performance

**Deliverable**: Ablation table for composition methods.

### Total timeline: ~10 weeks for Phases 1-3, ~12 weeks including Phase 4

---

## 6. RISKS AND MITIGATIONS <a name="risks-and-mitigations"></a>

### Biggest Risk: Generated Mesh Quality is Too Low for Grasping

**Probability**: Medium-High (60%). DreamGaussian's mesh extraction is known to produce oversaturated, oversmoothed geometry. The resulting collision meshes may be too blobby for finger-level contact simulation.

**Impact**: If meshes cannot support grasping, the entire pipeline premise fails.

**Mitigation Strategy**:
1. **First line**: Use GaussianDreamerPro instead of DreamGaussian -- it binds Gaussians to mesh explicitly, producing better geometry.
2. **Second line**: Apply post-processing mesh refinement (Instant Meshes for remeshing, then V-HACD for convex decomposition). This is computationally cheap.
3. **Third line**: Use PhysX-Anything's mesh generation (designed for simulation) even though it requires image input. Generate an image from the text prompt first (SDXL), then pass to PhysX-Anything.
4. **Escape hatch**: If generation quality is fundamentally insufficient, pivot RQ1 to: "What mesh post-processing pipeline makes generated assets graspable?" This is still a novel and publishable result.

### Secondary Risks

| Risk | Probability | Mitigation |
|------|------------|------------|
| Isaac Sim + GS rendering integration issues | Medium | Use Marble's demonstrated workflow; fallback to mesh-only rendering |
| PhysSplat MLLM gives nonsensical physics values | Medium | Use lookup table for common objects; MLLM only for novel objects |
| Parametric variation doesn't help (RQ2 null result) | Low-Medium | Null result is still publishable and useful for the community |
| VRAM constraints block GaussianDreamerPro | Low | Fallback to DreamGaussian (lower quality but fits in memory) |
| Marble API costs become prohibitive | Low | Only needed for RQ4; RQ1-3 and RQ5 use local generation |

---

## 7. VENUE / DEADLINE ANALYSIS <a name="venue-deadline-analysis"></a>

### RSS 2026

- **Conference**: July 13-17, 2026, Sydney, Australia
- **Workshop proposal deadline**: February 12, 2026, 23:59 AoE (6 DAYS FROM NOW)
- **Paper submission**: January 30, 2026 (PASSED)
- **Supplementary material**: February 6, 2026 (TODAY)
- **Acceptance notification**: April 27, 2026
- **Workshop paper deadlines**: TBD (typically 4-6 weeks before conference, ~May-June 2026)
- **Fit Assessment**: EXCELLENT for workshop paper. RSS is the premier venue for robotics + 3D perception. The "sim-ready generative assets" angle fits well with the sim2real and manipulation communities. Workshop paper submission likely late May to early June.
- **Action**: Too late for main paper. Watch for workshop CFPs in March-April. Strong target for a workshop paper with Phases 1-2 results.

### ICRA 2026

- **Conference**: June 1-5, 2026, Vienna, Austria
- **Main paper submission**: September 15, 2025 (PASSED)
- **Workshop proposals**: September 25, 2025 (PASSED)
- **Individual workshop paper deadlines**: Varies by workshop, typically February-March 2026
  - Field Robotics Workshop: March 15, 2026 (5 weeks from now)
  - Construction Robotics Workshop: TBD
  - Other workshops: Check individual pages
- **Workshop acceptance notifications**: ~April 15, 2026
- **Camera-ready**: ~April 30, 2026
- **Fit Assessment**: GOOD for specific workshops. The Field Robotics workshop (March 15 deadline) could work if we focus on agriculture applications. Look for workshops on sim2real, 3D generation for robotics, or digital twins.
- **Action**: Scan ICRA 2026 workshop list NOW for relevant workshops with March deadlines. Phase 1 (mesh quality threshold) could produce a focused workshop paper.

### CoRL 2026

- **Conference**: Late September 2026, Austin, Texas, US
- **Paper submission**: May 28, 2026, AoE
- **Workshops**: September 27, 2026
- **Workshop paper deadlines**: TBD (typically July-August 2026)
- **Fit Assessment**: BEST FIT for main paper. CoRL is specifically robot learning, and this work sits at the intersection of generative models + robot learning. The May 28 deadline gives ~16 weeks from now, enough time for Phases 1-3.
- **Action**: PRIMARY TARGET. Aim for CoRL 2026 main paper submission. Start Phase 1 immediately, have paper draft by mid-May.

### NeurIPS 2026

- **Conference**: December 6-12, 2026, Sydney, Australia
- **Paper submission**: Expected ~May 2026 (based on NeurIPS 2025 pattern: abstract May 11, paper May 15)
- **Workshop submissions**: Expected ~August-September 2026
- **Fit Assessment**: GOOD for the generative/physics angle. NeurIPS favors ML methodology. Frame as "physics-grounded generative 3D assets for embodied AI." The PhysX-3D group already published at NeurIPS 2025 on this topic.
- **Action**: SECONDARY TARGET. If CoRL submission is strong, submit to NeurIPS too (check overlap policies). Workshop is a good backup if main paper doesn't get in.

### Summary Timeline

```
Feb 2026:
  [Feb 6]  RSS 2026 supplementary deadline (passed/today)
  [Feb 12] RSS 2026 workshop proposal deadline

Mar 2026:
  [Mar 15] ICRA 2026 Field Robotics Workshop deadline
  [Mar ??] Other ICRA workshops TBD

Apr 2026:
  [Apr 27] RSS 2026 acceptance notification

May 2026:
  [May ~15] NeurIPS 2026 abstract/paper deadline (estimated)
  [May 28]  CoRL 2026 paper submission <<< PRIMARY TARGET

Jun 2026:
  [Jun 1-5] ICRA 2026 conference (Vienna)
  [Jun ??]  RSS 2026 workshop paper deadlines (estimated)

Jul 2026:
  [Jul 13-17] RSS 2026 conference (Sydney)
  [Jul-Aug]   CoRL 2026 reviews

Aug-Sep 2026:
  [Aug ??]  NeurIPS 2026 workshop submissions (estimated)
  [Sep 27]  CoRL 2026 workshops (Austin)
  [Sep ??]  CoRL 2026 main conference

Dec 2026:
  [Dec 6-12] NeurIPS 2026 conference (Sydney)
```

### Recommended Venue Strategy

1. **Immediate (Feb-Mar)**: Submit Phase 1 mesh quality results to an ICRA 2026 workshop (March 15 deadline).
2. **Primary (May 28)**: Submit full paper (RQ1 + RQ2 + RQ5) to CoRL 2026.
3. **Backup (May ~15)**: If results are strong on the ML side, also target NeurIPS 2026 main paper (framed differently -- methodology focus).
4. **Fallback (Jun-Aug)**: RSS 2026 workshop or NeurIPS 2026 workshop for any partial results.

---

## APPENDIX: Key URLs and Resources

- [DreamGaussian GitHub](https://github.com/dreamgaussian/dreamgaussian)
- [GaussianDreamerPro GitHub](https://github.com/hustvl/GaussianDreamerPro)
- [PhysX-Anything GitHub](https://github.com/ziangcao0312/PhysX-Anything)
- [PhysX-3D / PhysXGen GitHub](https://github.com/ziangcao0312/PhysX-3D)
- [PhysSplat arXiv](https://arxiv.org/abs/2411.12789)
- [ComGS Project](https://nju-3dv.github.io/projects/ComGS/)
- [D3DR arXiv](https://arxiv.org/abs/2503.06740)
- [FreeInsert arXiv](https://arxiv.org/abs/2505.01322)
- [World Labs Marble](https://www.worldlabs.ai/)
- [NVIDIA Isaac Sim + Marble Blog](https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble)
- [SplatSim Project](https://splatsim.github.io/)
- [RoboSplat GitHub](https://github.com/OpenRobotLab/RoboSplat)
- [MaGS Project](https://wcwac.github.io/MaGS-page/)
- [Ref-Gaussian ICLR 2025](https://openreview.net/forum?id=xPxHQHDH2u)
- [OnePoseviaGen GitHub](https://github.com/GZWSAMA/OnePoseviaGen)
- [CoRL 2026](https://www.corl.org/)
- [RSS 2026 CFP](https://roboticsconference.org/information/cfp/)
- [ICRA 2026](https://2026.ieee-icra.org/)
- [NeurIPS 2026](https://neurips.cc)
