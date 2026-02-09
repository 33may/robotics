# Research Prompt: Generative Gaussian Splatting Worlds for Robot Policy Training

## Researcher Profile

- **Background:** Industry engineer working full-time on simulation-driven robotics learning
- **Hardware:** SO-101 6-DOF robot arm, RTX 4070 Ti SUPER (16GB VRAM), NVIDIA Isaac Sim
- **Current state:** Working behavior cloning pipeline with SmolVLA (~80% task success), Isaac Sim inference pipeline operational
- **Task domain:** Agriculture/food manipulation (tomato picking)
- **Target output:** Workshop paper (CoRL, ICRA, NeurIPS workshop, or similar)
- **Timeline:** February - June 2026
- **Paper experience:** 1-2 prior papers

## Existing Technical Stack

- **BC training:** LeRobot + SmolVLA (vision-language-action model)
- **Simulation:** NVIDIA Isaac Sim + Isaac Lab (manager-based RL environments)
- **3D Reconstruction:** gsplat + 3DGUT + COLMAP/GLOMAP planned (not yet captured)
- **Robot:** SO-101 with leader-follower teleoperation for data collection
- **Integration:** LeIsaac bridge between LeRobot and Isaac Sim

## Research Context

The broader project aims to go beyond behavior cloning (80% ceiling) by training in simulation using RL, then transferring back to real hardware. The key enabler is creating high-fidelity, interactive digital twins of real workspaces using Gaussian Splatting.

**What already exists (do NOT re-prove these):**
- SplatSim (CMU, 2024) proved GS rendering closes the visual sim2real gap for manipulation (86.25% zero-shot)
- RoboSplat (RSS 2025) showed direct 3DGS editing generates diverse demonstrations, achieving 87.8% one-shot success across 6 augmentation types
- RL-GSBridge (2024) demonstrated RL training inside GS-reconstructed environments with zero-shot sim2real transfer
- RoboGSim (2024) built a Real2Sim2Real GS simulator with action-conditioned video generation
- GaussianEditor (CVPR 2024) enables text-guided GS editing in ~20 min on a single GPU

**The gap we target:** All existing work treats GS scenes as VISUALLY rich but PHYSICALLY inert. Nobody has built a pipeline that takes a GS reconstruction, decomposes it into semantically-labeled objects with collision geometry, then uses generative editing/composition to create diverse INTERACTIVE training environments where the robot physically grasps, pushes, and collides with scene objects.

---

## Three Research Branches

### Branch 1: GS Editing as Structured Domain Randomization for Contact-Rich Tasks

**The state of the art and its limits:**
RoboSplat demonstrated 6 types of GS augmentation (object replacement, appearance, pose, lighting, viewpoint, embodiment) for tabletop pick-and-place. However, all RoboSplat augmentations are VISUAL — they render modified images for behavior cloning but do NOT change physics, collision geometry, or enable RL training. The augmented scenes cannot be stepped through a physics simulator.

**Research Questions:**

1. **Can GS scene editing produce variations that are BOTH visually diverse AND physically consistent for RL training?** RoboSplat edits are visual-only (rendered for BC data). If you change a tomato's size via GS editing, the collision mesh must change too. What is the pipeline to keep visual edits and physics proxies synchronized?

2. **Which specific augmentation types yield the highest marginal gain for contact-rich agriculture tasks?** RoboSplat tested 6 types on rigid tabletop objects. Tomato picking involves deformable objects, vine structures, varying ripeness (visual + physical property change), and cluttered organic geometry. Which augmentations transfer? Which new ones are needed (e.g., deformability variation, stem attachment strength)?

3. **What is the scaling law for GS-edited scene variations vs. policy robustness in simulation?** At what point do additional edited variations stop improving the policy? Is there a diminishing returns curve, and does it differ from traditional domain randomization's known curves (Tobin et al., 2017)?

**Key prior work to analyze:** RoboSplat (RSS 2025), SplatSim (2024), Instruct-GS2GS, GaussCtrl, domain randomization scaling (OpenAI Dactyl, Tobin et al.)

---

### Branch 2: Semantic-Physical Scene Decomposition — From Static GS Backdrop to Interactive Digital Twin

**The state of the art and its limits:**
NVIDIA NuRec imports GS reconstructions into Isaac Sim as static visual backdrops with a ground-plane physics proxy. LangSplat/FlashSplat can segment GS scenes semantically. SuGaR/MILo can extract meshes from Gaussians. PhysGaussian/MaGS can simulate physics on Gaussians. BUT: nobody has connected these into a pipeline that produces a fully interactive scene from a single GS scan. The missing link is going from "I can segment a tomato in the GS scene" to "Isaac Sim treats that tomato as a graspable rigid/soft body with correct mass and friction."

**Research Questions:**

1. **What is the minimum mesh quality from GS extraction (SuGaR/MILo) required for stable contact simulation in Isaac Sim/PhysX?** Grasping requires watertight manifold meshes with proper normals. GS-extracted meshes are often noisy, non-manifold, and have holes. What post-processing pipeline (Poisson reconstruction, remeshing, convex decomposition) bridges this gap? At what point does mesh quality from GS become "good enough" compared to manual CAD for grasp success?

2. **Can language-grounded segmentation (LangSplat) reliably decompose agricultural scenes where objects are small, clustered, and partially occluded?** Tomatoes on a vine are visually similar, touching each other, and partially hidden by leaves. LangSplat was evaluated on room-scale scenes with distinct objects. Does it work on dense agricultural geometry? What's the failure mode and can SAM3D or Gaussian Grouping handle these cases better?

3. **Is it feasible to skip mesh extraction entirely by using PhysGaussian or MaGS for direct Gaussian-based physics simulation inside Isaac Sim?** PhysGaussian uses MPM directly on Gaussians (no mesh needed). MaGS adsorbs Gaussians onto meshes for coupled visual-physical simulation. Can either integrate with Isaac Sim's PhysX pipeline, or do they require their own physics engine? What's the trade-off: mesh extraction (lossy but Isaac-compatible) vs. Gaussian-native physics (accurate but potentially incompatible)?

4. **How should physics properties be assigned to semantically-segmented objects?** Given a LangSplat label "tomato", what's the method to assign mass (~150g), friction coefficient (~0.5), and deformability (Young's modulus ~1MPa)? Is there existing work on semantic-to-physical property lookup tables, or is this manual? Could LLMs be used to estimate these from semantic labels?

**Key prior work to analyze:** LangSplat V2 (NeurIPS 2025), FlashSplat (ECCV 2024), SuGaR, MILo (SIGGRAPH Asia 2025), PhysGaussian (CVPR 2024), MaGS (ICCV 2025), NVIDIA NuRec pipeline, DecoupledGaussian

---

### Branch 3: Compositional Generation — Spawning New Interactable Objects into Reconstructed Scenes

**The state of the art and its limits:**
DreamGaussian and GaussianDreamer can generate 3D Gaussian objects from text/image prompts in minutes. SplatSim demonstrated foreground/background composition (placing GS objects into GS scenes). BUT: generated GS objects have no collision mesh, no physics properties, no guaranteed scale consistency, and no deformability model. They are purely visual. The industry side (NVIDIA Edify 3D, World Labs) is building toward sim-ready asset generation, but current public tools produce display-quality assets, not simulation-quality ones.

**Research Questions:**

1. **Can text-to-3D Gaussian generators (DreamGaussian, GaussianDreamer) produce objects that, after mesh extraction + physics property assignment, function as graspable assets in Isaac Sim?** What is the full pipeline: prompt → generate GS object → extract collision mesh (SuGaR) → assign physics → place in scene → verify grasp success? What are the failure rates at each stage?

2. **How do you maintain lighting/scale/appearance consistency when inserting generated GS objects into a reconstructed GS scene?** SplatSim composites foreground scans onto background scans (both from real captures). Inserting GENERATED objects raises new problems: mismatched lighting (the generated object was "lit" by a different environment), incorrect scale (generation doesn't know the scene's metric scale), and appearance inconsistency (Gaussian density/color distribution differs). What techniques solve these? (Relighting: Ref-Gaussian? Scale: reference objects in scene?)

3. **Can parametric variation of generated objects create meaningful training diversity for agriculture?** Instead of generating one tomato, can we systematically vary: size (3-8cm), color (green→red gradient for ripeness), shape (round→oblong), surface texture (smooth→blemished), and have each variant be physically accurate (mass scales with size, ripe tomatoes are softer)? Is this parametric control achievable with current text-to-3D methods, or does it require a different approach (e.g., mesh deformation + retexturing)?

4. **How does compositional scene generation compare to RoboSplat's object replacement augmentation and to traditional asset libraries (e.g., NVIDIA Omniverse assets)?** RoboSplat replaces objects by swapping GS reconstructions of real alternatives. Omniverse has curated sim-ready assets but limited agricultural variety. Generated objects offer unlimited variety but unknown quality. What's the Pareto frontier of variety vs. simulation fidelity across these three approaches?

**Key prior work to analyze:** DreamGaussian (ICLR 2024), GaussianDreamer, SplatSim composition approach, NVIDIA Edify 3D, Ref-Gaussian (ICLR 2025), MaGS, RoboSplat object replacement

---

## Synthesis Question: The Unified Pipeline

The strongest contribution would unify all three branches into a single coherent pipeline:

```
Real Scene → GS Reconstruction → Semantic Decomposition (Branch 2)
                                        ↓
                              Per-object meshes + physics
                                        ↓
                    ┌───────────────────┼───────────────────┐
                    ↓                   ↓                   ↓
            Scene Editing         Object Generation    Object Variation
            (Branch 1)            (Branch 3)           (Branch 3)
                    ↓                   ↓                   ↓
                    └───────────────────┼───────────────────┘
                                        ↓
                        Diverse Interactive Training Scenes
                                        ↓
                              RL Training in Isaac Sim
                                        ↓
                              Sim2Real Transfer to SO-101
```

**The synthesis research question:** Does this full pipeline (reconstruct → decompose → augment/generate → train RL → deploy) produce measurably better real-world manipulation performance than: (a) BC only, (b) RL in hand-built simulation, or (c) RL in static GS backdrop?

---

## Output Requirements

### PRIMARY DELIVERABLE: Research Questions

The main output of this research is **refined, validated research questions** for each branch. For each branch, deliver:

1. **2-3 refined research questions** — Each question must be:
   - Grounded in what the literature review reveals (cite the papers that prove the gap exists)
   - Specific enough to be answerable with a concrete experiment
   - Novel (not already answered by existing work — state explicitly what prior work comes closest and why it doesn't answer this)
   - Feasible for our hardware/timeline (RTX 4070 Ti SUPER, Feb-June 2026)

2. **For each research question, provide:**
   - **Prior work context** — the 2-3 papers closest to answering it, and the specific gap they leave
   - **Proposed method** — a 1-paragraph sketch of how we would answer it
   - **Expected outcome** — what result would confirm or deny the hypothesis
   - **Why it matters** — the practical implication for robot policy training

### SUPPORTING EVIDENCE (per branch)

3. **Top 5 most relevant papers** — title, year, venue, arxiv link, and 2-sentence summary explaining SPECIFICALLY what they contribute and what they leave unsolved
4. **Feasibility assessment** — can this run on RTX 4070 Ti SUPER (16GB VRAM)? Cite specific VRAM requirements of the tools involved
5. **Minimal experiment** — the smallest concrete experiment (task, metrics, baselines, expected result) that validates the branch's core claim
6. **Risk assessment** — the single biggest technical risk and a concrete mitigation strategy

### CROSS-BRANCH SYNTHESIS

7. **Synthesized research questions** — Take the best questions from all 3 branches and propose 2-3 unified research questions that span multiple branches. These should form the backbone of a single workshop paper.
8. **Venue recommendation** — specific workshop name, expected deadline, and why it fits (check CoRL 2026, ICRA 2026, RSS 2026, NeurIPS 2026 workshop timelines)
9. **3-5 paper title options** — concise, specific, informative
10. **Synthesis recommendation** — which combination of branches (1+2, 2+3, 1+2+3, etc.) creates the strongest workshop paper given the timeline and hardware constraints, and why

### Save all findings to `/home/may33/projects/ml_portfolio/robotics/vbti/research/generative-worlds/research_results.md`
