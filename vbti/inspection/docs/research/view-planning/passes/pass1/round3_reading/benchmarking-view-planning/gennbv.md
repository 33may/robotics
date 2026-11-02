# GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction (CVPR 2024)

- arXiv: 2402.16174 (v3, 30 Jul 2024) — https://arxiv.org/html/2402.16174v3
- Authors: Xiao Chen, Quanyi Li, Tai Wang, Tianfan Xue, Jiangmiao Pang (Shanghai AI Lab, CUHK)
- Read via full HTML source, tables/equations extracted directly from the arXiv HTML (not just abstract/summary).

## Core method (for context — Branch 1/2 territory, kept brief here)

- RL (PPO)-based NBV policy for a simulated drone (CrazyFlie) doing active 3D reconstruction of **building-scale objects** (houses).
- **Action space (§3.1):** 5-DoF continuous — 3D position in a ~20m×20m×10m free-space box + 2D heading (yaw, pitch; roll excluded). This is explicitly *not* object-conditioned — it's a free-flight box around the whole scene, contrasted against prior work's "hand-crafted hemisphere" (Fig. 1). No notion of "orthogonal to a surface," "primitive shape," or "symmetry" anywhere in the paper (checked: 0 hits for "orthogonal," "primitive," "symmetry").
- **State embedding (§3.2, Eq. 7):** concatenation of geometric (probabilistic 3D occupancy grid, built by back-projecting depth + Bresenham ray casting, log-odds update), semantic (2-layer CNN on RGB), and action-history embeddings, each 256-d, linearly fused.
- **Reward (§3.3, Eq. 2–3):** `CR_t = Ñ_t / N* · 100%` (occupied voxels found / GT occupied voxels); reward `r^CR_{t+1} = CR_{t+1} − CR_t` (incremental coverage). Negative reward on collision (episode terminates) and on exceeding an empirical keyframe budget.
- Episode ends at 99% coverage, on collision, or at 100 steps (Appendix A.2). Trained 32M+ iterations, ~24h on one V100, 256 parallel Isaac Gym envs (one building per env).

## Benchmark design (the part in scope for this branch)

### Ground truth and simulation (§4.1, Appendix B)
- Ground-truth point cloud: **Poisson Disk sampling** on the mesh surface (Open3D), 100k points, then voxelized to get `N*`.
- Simulator: **NVIDIA Isaac Gym**, GPU-accelerated (~1000 FPS), RGB-D camera 400×400, 90° vertical FOV, fixed point light.
- Explicit quote (Appendix A.1): *"our experiments are based on the realistic simulator and accurate observations like depth maps."* → **No sensor noise, no depth dropout, no pose error, no fused-vs-single-view offset is injected anywhere in this benchmark.** This is a clean-simulation-only evaluation.

### Datasets (§4.1)
- **Houses3K**: 256 train / 50 test synthetic building meshes, disjoint batches ("distinct geometries").
- **OmniObject3D "house" category**: 43 real-scanned toy houses — used purely as a cross-domain generalization test (policy trained on Houses3K, evaluated zero-shot).
- **Objaverse 1.0 / Replica**: qualitative-only, non-house categories, no quantitative table.
- No primitive/near-primitive stratification of any kind — all objects are complex architectural forms; the benchmark's difficulty axis is "in-distribution vs cross-dataset," not shape complexity.

### Evaluation protocol — the confound fix (§4.1, direct quote)
> "most prior works separately use the coverage ratio (%) and the number of views to evaluate the reconstruction completeness and efficiency for NBV policies. However, the coverage ratio is highly correlated with the number of views. Thus we propose to unify the number of views to a fixed value for all NBV policies during evaluation and use the area under the curve (AUC) of coverage ratio as the main metric for comparison."

- **Fixed view budget**: 30 views for Houses3K, 20 for OmniObject3D (Table 1 caption). This is a hard cap applied identically to every method — no auto-stop-and-compare-budgets.
- Three reported metrics per table, all "along with the consistent number of views": (1) **Mean AUC of coverage ratio** (completeness+efficiency in one number), (2) **Final Coverage Ratio (FCR)** at budget exhaustion, (3) **Chamfer Distance accuracy (cm)** between scanned and GT point cloud (via PyTorch3D), computed presumably at final coverage.
- AUC is computed by tracking CR_t at every step up to the fixed budget and integrating — this is literally a per-step curve, i.e., exactly the "coverage % vs view count" plot our branch scope calls out, collapsed to one scalar.

### Baselines (Table 1, Appendix A.3) — exact recipe, transferable to our Q5
1. **Random**: random 5-DoF vector in the full action space, collision-rejected.
2. **Random Hemisphere**: random position on a fixed hemisphere (radius 9m, chosen > max object height, to guarantee no collision), heading always toward hemisphere center.
3. **Uniform Hemisphere**: positions **evenly distributed** on the same hemisphere, **at the same view-count budget as the policy being tested** — Houses3K: 5 heights × 6 positions = 30 views; OmniObject3D: 4 heights × 5 positions = 20 views. This is our field-standard analog to the "uniform {h,v} sphere" baseline in our setup, and critically it is budget-matched by construction (5×6=30, 4×5=20), directly answering our Q5/Q6: the paper enforces "denser/matched uniform sphere at matched view budget must be reported."
4. **Uncertainty-Guided** [Lee et al. / entropy-based NeRF uncertainty] and **ActiveRMap** [info-gain]: both select from a *pre-sampled 100-view candidate set per object* on the hemisphere, using a continually-optimized per-scene NeRF (TensoRF) — i.e., "oracle-ish" baselines that get to re-optimize on the test object itself (marked "†": trained+evaluated on the same test objects, not held-out).
5. **Scan-RL**: prior RL-based NBV baseline restricted to a 3-DoF hemisphere action space, RGB-only (6-frame) history.
6. Two **cross-implementation ablations** that isolate action-space contribution from representation contribution (directly answers our Q14 — separating "the action space is better" from "the policy used it better"):
   - **"Scan-RL w/ Our Repre."** — Scan-RL's 3-DoF hemisphere action space + GenNBV's multi-source state embedding.
   - **"Ours w/ Scan-RL Repre."** — GenNBV's 5-DoF free-space action space + Scan-RL's RGB-only representation.

### Exact Table 1 numbers (Houses3K test / OmniObject3D house, at 30/20 views resp.)

| Method | H3K AUC | H3K FCR | H3K Acc(cm) | Omni AUC | Omni FCR | Omni Acc(cm) |
|---|---|---|---|---|---|---|
| Random | 48.53% | 58.24% | 1.38 | 60.13% | 73.73% | 0.62 |
| Random Hemisphere | 71.19% | 79.72% | 0.63 | 74.03% | 84.74% | 0.48 |
| **Uniform Hemisphere** | 82.91% | 89.71% | 0.44 | 83.09% | 92.90% | 0.41 |
| Uncertainty-Guided† | 83.13% | 89.31% | 0.44 | 69.08% | 92.85% | 0.42 |
| ActiveRMap† | 84.86% | 90.77% | 0.44 | 69.47% | 93.15% | 0.42 |
| Scan-RL* | 84.49% | 91.61% | 0.40 | 82.17% | 92.53% | 0.34 |
| Scan-RL w/ Our Repre.* | 86.48% | 92.20% | 0.37 | 86.14% | 93.73% | 0.35 |
| Ours w/ Scan-RL Repre.* | 87.39% | 95.63% | 0.38 | 86.81% | 93.64% | 0.34 |
| **GenNBV (Ours)*** | **91.19%** | **98.26%** | 0.37 | **88.63%** | **97.12%** | 0.33 |

(* trained on Houses3K, held-out test; † per-scene NeRF-optimized, non-held-out.)

Key reading: the fully-uniform sphere baseline (82.91/83.09% AUC) is already close to the NeRF-based "oracle-ish" per-scene-optimized baselines (83–85%), and within ~8 AUC points of the full learned method (91.19%). **The absolute Chamfer accuracy spread across ALL nine methods is only 0.33–0.44 cm on OmniObject3D (i.e., ~1mm) and 0.37–1.38cm on Houses3K** (dominated entirely by the "Random" outlier at 1.38cm; everything else, including plain Uniform Hemisphere, sits in 0.37–0.44cm).

### Ablations (Table 3, Table 4)
- **Representation ablation (Table 3)**, uni-source vs multi-source, on Houses3K:
  - Probabilistic 3D Grid only: 81.06% AUC / 84.56% FCR
  - 5-DoF Pose only (a learned-trajectory baseline with no scene input): 69.53% / 76.61%
  - Semantic 2D map only: 81.24% / 87.90%
  - Geometric+Pose: 88.66% / 96.67%; Geometric+Semantic: 89.77% / 95.31%; Pose+Semantic: 88.30% / 96.29%
  - All three (full GenNBV): 91.19% / 98.26%
- **Depth representation ablation (Table 4)**: Probabilistic 3D Grid (81.06% AUC) > Binary 3D Grid (77.28%) > 2D BEV geometric map (71.86%) — justifies 3D probabilistic occupancy over cheaper alternatives.
- **Training-object-count ablation (Appendix C, Table 6)**: AUC on OmniObject3D rises monotonically with number of Houses3K training objects: 64→82.09%/93.70%, 128→87.21%/96.75%, 256→88.63%/97.12%. Diminishing returns visible but not yet flat at 256.

### Statistical treatment (weak — negative finding for our Q12)
- Baselines with stochastic policies (Random, Random Hemisphere) are "averaged over random seeds from 0 to 4" (5 seeds) — but **no variance, standard deviation, or confidence interval is reported anywhere in the paper** (grep-checked: 0 hits for "variance," "standard deviation," "confidence interval," "p-value"). All tables report single point estimates. This is a real methodological gap if we want to cite this paper as a model for our Q12 statistical rigor — we cannot borrow their stats practice, only their metric design.

## Load-bearing assumptions
1. Clean, noiseless simulated depth and known-exact camera pose at every step (explicitly stated, §A.1) — the entire benchmark assumes perfect perception primitives; view-planning quality is isolated from sensing quality by construction.
2. Ground truth is a mesh-derived, densely-sampled point cloud (Poisson disk, 100k pts) voxelized — always available because objects are synthetic/scanned CAD-quality meshes, not survey clouds with defects.
3. Reward = evaluation metric's core term (incremental coverage ratio is literally the training reward, and coverage-ratio AUC/FCR is the headline eval metric). Chamfer accuracy is the one held-out-of-reward metric, giving partial (not full) separation between judge and objective.
4. Objects are all large building-scale scenes with no primitive-shape structure exploited; "best view" has no analytical definition here — coverage of unknown voxels is the only quality signal, not orientation to a surface.
5. Fixed, hard view-count budget shared by all methods at eval — auto-stopping is not evaluated; the policy also has its own internal stop condition at training time (99% coverage) but eval reports metrics *up to* the fixed budget regardless of when/whether coverage saturates.

## What transfers to our setting
- **The confound fix itself** (Q5/Q6/Q19 core): report AUC-of-coverage-vs-views *and* final-metric-at-fixed-budget together, never coverage-% alone — directly reusable methodology, independent of domain.
- **Budget-matched uniform-sphere baseline recipe** (Q5): construct the uniform baseline so its view count exactly equals the tested method's budget (their 5×6=30, 4×5=20 grid), not a separately-sized "reasonable default" sphere. Our current uniform {h,v} sphere (12×3=36) should be reported at whatever budget we cap the geometry-aware policy to, not at its native 36.
- **Cross-implementation ablation design** (Q14): the "Scan-RL w/ Our Repre." / "Ours w/ Scan-RL Repre." pattern is a directly reusable template for separating "the action space is better" from "the agent used it better" — swap our geometry-conditioned action space into the uniform-sphere-era orchestrator and vice versa.
- **Chamfer-as-secondary-metric practice**: keeping a metric untouched by the reward function (here Chamfer distance) as a secondary check is a cheap partial answer to Q3/Q19 circularity, though see below for its limits.
- Multi-source ablation table format (uni-source vs combinations) is a clean template for our own action-space-representation ablations.

## What does NOT transfer, and why (first-class negative findings)
1. **No sensor-defect injection at all (Q9, Q10).** GenNBV's benchmark is pure clean-simulation; it never asks whether ranking among methods survives pose error, depth dropout, or a fused-vs-single-view offset. Our setting's ~4cm fused-cloud offset and per-view dropout are *entirely absent* as a benchmark axis here — we cannot borrow a defect-injection protocol from this paper, only its metric-reporting shape. This paper is evidence that the field's "current convention" (per the branch tasking) has not yet solved Q9; it's a gap we'd be first to fill, not a solved problem to copy.
2. **No orthogonality / surface-normal notion whatsoever (Q7, Q8).** Coverage-ratio (voxel-occupied-or-not) is blind to *view angle* — a grazing view that happens to reveal a voxel counts identically to an orthogonal one. This is the opposite of our core requirement (evidence views must be orthogonal to the inspected surface). GenNBV's metric family is structurally unable to express our Q7 target; we need a different metric, not an adaptation of theirs.
3. **No primitive-shape / symmetry handling (Q8, Q11).** All target objects are complex architectural meshes; there is no equivalence-class or primitive-decomposition concept, so nothing here informs how to handle "boxes/cylinders make the correct view a set, not a point."
4. **Ground truth requires a full watertight mesh (Q2, Q4).** GT voxels/point cloud come from Poisson-disk sampling a clean mesh — this whole approach is unavailable for a survey cloud with real defects (our Q2's exact concern: does the analytical GT survive contact with a defective cloud). GenNBV never has to answer this because its "GT" is definitionally the clean mesh, sidestepping Q2 entirely.
5. **Reward/metric circularity is present, not solved (Q3).** The paper's own headline metric (coverage AUC/FCR) is the literal training signal; this is exactly the "judge-benchmark circularity" flagged in the task rationale — GenNBV is a live example of the problem, not a counterexample. Chamfer distance is the only independent check, and its cross-method spread (0.33–0.44cm on OmniObject3D) is tiny — smaller than could plausibly separate real method quality from noise once any real sensor defect (mm-level pose error, cm-level fusion offset) is added. This directly informs our Q2/Q9: if we transplant Chamfer-style accuracy as a secondary metric, we must first check whether its cross-method spread survives being swamped by our ~4cm offset — on this paper's own numbers, it likely would not.
6. **No statistical rigor to borrow (Q12).** Despite 5-seed averaging for stochastic baselines, no variance/CI is ever reported — we should not treat this paper as a model for statistical treatment, only for point-metric design.
7. **Large-scale / drone-in-free-space setup doesn't map to our reachability constraints (givens).** The 5-DoF "any point in a 20m×20m×10m box" action space explicitly removes the kinematic/collision constraints that dominate our problem (IK, gripper+camera hull, table, minimum shell radius); their only real-space constraint is a single collision-avoidance reward term with hemisphere-radius padding, not a genuine reachable-shell computation.

## Relevance to our RQs
Most directly useful for **Q5/Q6/Q19** (confound-avoidance methodology, budget-matched uniform-sphere baseline, cross-implementation ablation for Q14) — this is the strongest "steal the eval scaffolding" source in the branch. Load-bearing but negative for **Q2/Q3/Q9/Q10**: it demonstrates the field's current benchmark convention still has unresolved reward-metric circularity and zero sensor-defect handling, meaning our benchmark cannot copy a defect-injection or GT-under-noise protocol from here — we have to originate one. Not useful for **Q7/Q8/Q11** (no orthogonality, no primitives, no symmetry) — these need to come from elsewhere.
