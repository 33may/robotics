# Branch 4 — `view-set-selection-under-constraints`
## Deriving the view set: coverage, selection, reachability

**Date:** 2026-08-27
**Role:** literature mapper (judgment tier), pass 1 / round 2.
**Branch plan:** [`../round1_plan.md` §Branch 4](../round1_plan.md)
**Contract:** [`../../../../geometry-aware-view-planning.md`](../../../../geometry-aware-view-planning.md)
**Feeds:** RQ2 (primary), RQ3 (secondary). Priority P1.

---

## How to read this map

Every candidate is filed into exactly one of three buckets, matching the
round-2 synthesis structure the branch plan asked for:

- **(a) Generation** — how candidate viewpoints are produced from known
  geometry. This is where the contract's "box → faces, cylinder → orbit"
  precedent lives.
- **(b) Selection** — how a subset is chosen from candidates (set cover,
  greedy, ILP, learned), and how set *size* is controlled.
- **(c) Feasibility coupling** — how kinematics, collision and stand-off
  distance enter, and what happens when the geometrically ideal view does not
  survive them. This is the branch's distinguishing question and our critic
  mandate #4.

Per-candidate tags:

- **RQ** — which research question it feeds.
- **Extract** — what the reader should pull out (this is the reading task, not
  a summary request).
- **⚠︎** — constraint-violation tag: an assumption the paper makes that our
  setting breaks (clean CAD/mesh input, free-flying camera, unlimited view
  space, continuous pose output, mobile base).
- **BM** — flag for Branch 2: its evaluation setup / dataset / ground-truth
  protocol could seed our benchmark.

**Dedup with other branches.** NBV papers appear here for their *selection
loop and feasibility handling only*; their score/utility functions belong to
Branch 1, their addressing schemes to Branch 5, their evaluation protocols to
Branch 2, their VLM interface to Branch 6. Where a paper is listed here and is
likely on another branch's list too, the ownership note is in the `Extract`
line.

---

## Bucket 0 — Spine (read these first, in this order)

### 1. `scott-2003-view-planning-survey` — **HIGH**
W. R. Scott, G. Roth, J.-F. Rivest, "View Planning for Automated
Three-Dimensional Object Reconstruction and Inspection", *ACM Computing
Surveys* 35(1):64–96, 2003.
<https://dl.acm.org/doi/10.1145/641865.641868> · paper

*Why:* The canonical taxonomy of the whole problem and the source of the
model-based vs. non-model-based split our project sits astride (we have a
survey cloud, so we are model-based over a *defective* model). Defines the
measurability matrix / set-cover reduction that every later formulation
inherits, and enumerates the constraint families (visibility, incidence,
resolution, field of view, positioning-system) that our IK/collision/shell
stack extends.
**RQ2, RQ3 · Extract:** the constraint taxonomy verbatim; where "positioning
system constraints" sit in their pipeline (up-front vs. post-filter); their
statement of what makes the VPP NP-hard. **⚠︎** assumes a laser range sensor
and a clean prior model.

### 2. `peuzin-jubert-2021-vpp-survey` — **HIGH**
M. Peuzin-Jubert, D. Nozais, J.-L. Mari, J.-P. Pernot, A. Polette, "Survey on
the View Planning Problem for Reverse Engineering and Automated Control
Applications", *Computer-Aided Design* 141:103094, 2021.
<https://www.sciencedirect.com/science/article/abs/pii/S0010448521001056> · paper

*Why:* The modern successor to Scott 2003, written from the industrial
quality-control side. Groups solution methods into macrocategories with an
explicit criteria grid — the fastest way to place our two candidate tracks
(analytical vs. optimization) inside an existing taxonomy instead of inventing
vocabulary.
**RQ2 · Extract:** the criteria grid (transcribe it — it is the skeleton of our
round-3 comparison table); which macrocategory handles sensor *placement
feasibility*; the listed open challenges.

### 3. `alsadik-2025-nbv-taxonomy` — **HIGH**
B. Alsadik, H. A. Mahdi, N. A. Abdulateef, "A structured review and taxonomy of
next-best-view strategies for 3D reconstruction", *ISPRS Open Journal of
Photogrammetry and Remote Sensing* 17:100098, 2025. Open access.
<https://www.sciencedirect.com/science/article/pii/S2667393225000171> · paper

*Why:* 100+ papers sorted into rule-based / uncertainty-based / sampling-based
/ learning-based / prediction-based. Recent enough to tell us whether learned
selection has actually displaced analytical selection, which is the question
the branch has to answer before we commit to the analytical track.
**RQ2 · Extract:** which strategy class dominates for *known-geometry,
object-scale, arm-mounted* settings; the open-access tool list (we want
something integrable). **BM** — it aggregates evaluation practice.

### 4. `lee-2024-autonomous-view-planning-review` — MED
I. D. Lee, J. H. Seo, B. Yoo, "Autonomous view planning methods for 3D
scanning", *Automation in Construction* 160:105291, 2024. Open access.
<https://www.sciencedirect.com/science/article/pii/S092658052400027X> · paper

*Why:* Bridges the graphics and robotics framings and explicitly covers
*partially known* geometry — our regime (we have a cloud, not a model, not
nothing). Sankey diagrams of research trends are a cheap way to spot dead
sub-lines we should not enter.
**RQ2 · Extract:** their "partially known geometry" branch; per-platform
performance comparison (manipulator vs. mobile vs. drone). **⚠︎** construction
scale.

---

## Bucket A — Viewpoint generation from known geometry

### 5. `tarbox-1995-complete-sensor-coverage` — **HIGH**
G. H. Tarbox, S. N. Gottschlich, "Planning for Complete Sensor Coverage in
Inspection", *Computer Vision and Image Understanding* 61(1):84–111, 1995.
<https://doi.org/10.1006/cviu.1995.1007> · paper

*Why:* The original CAD-based complete-coverage inspection planner: discretize
the viewpoint space, discretize the surface, build the visibility/measurability
relation, cover it. Our whole RQ2 is a re-run of this with a noisy surface and
a much smaller admissible viewpoint space.
**RQ2, RQ3 · Extract:** how the viewpoint space is discretized and how surface
points are discretized (this is also Branch 5 input — flag it there); how
occlusion and incidence enter the measurability relation.
**⚠︎** clean CAD input.

### 6. `magana-2023-viewpoint-generation-constrained-spaces` — **HIGH**
A. Magaña, J. Dirr, P. Bauer, G. Reinhart, "Viewpoint Generation using
Feature-Based Constrained Spaces for Robot Vision Systems", arXiv:2306.06969,
2023.
<https://arxiv.org/abs/2306.06969> · paper

*Why:* The closest published thing to our analytical track. Each viewpoint
constraint (resolution, incidence, field of view, working distance,
occlusion, robot limits) is formulated as a topological *ℂ-space* — the region
where the sensor may sit and still satisfy that constraint — and viewpoints
fall out of the intersection in closed form, deterministically, without
sampling-and-scoring. That is exactly "derive the pose from the geometry", and
it gives a principled home for our hard minimum shell radius as just another
constraint space.
**RQ2, RQ3 · Extract:** the full list of constraints they express as ℂ-spaces
and the closed forms; how they intersect spaces and what they do when the
intersection is empty (our unreachable-view case); whether robot constraints
are first-class or bolted on. Validated on two real sensors — record which.

### 7. `mosbach-2021-feature-driven-viewpoint-placement` — **HIGH**
D. Mosbach, P. Gospodnetić, M. Rauhut, B. Hamann, H. Hagen, "Feature-Driven
Viewpoint Placement for Model-Based Surface Inspection", *Machine Vision and
Applications* 32(1):8, 2021.
<https://dblp.org/rec/journals/mva/MosbachGRHH21.html> · paper

*Why:* Candidate density is driven by *geometry*, not by mesh resolution:
adaptive subdivision in parameter space with "feature functionals", dense where
curvature is high, sparse on flats. That is the principled generalization of
"one view per box face, a ring per cylinder" — a flat face genuinely needs one
viewpoint and the method emits one. Directly attacks the candidate-explosion
problem that makes Scott-style per-vertex generation useless to us.
**RQ2, RQ3 · Extract:** the feature functionals and their termination
condition; the resulting viewpoint counts per shape class; how they get from a
discrete mesh to B-splines (our input is worse than a mesh — does the
preprocessing survive noise? cross-tag Branch 3).
**⚠︎** assumes B-spline/mesh input, not a defective point cloud.

### 8. `gospodnetic-2022-viewpoint-placement-inspection-planning` — **HIGH**
P. Gospodnetić, D. Mosbach, M. Rauhut, H. Hagen, "Viewpoint placement for
inspection planning", *Machine Vision and Applications* 33(2), 2022.
<https://link.springer.com/article/10.1007/s00138-021-01252-z> · paper

*Why:* The companion that closes the loop: candidates → greedy NBV reduction →
camera path under the inspection system's DOF. It is the only entry that spans
generation, selection *and* mechanism DOF in one pipeline, which is the shape
our answer to RQ2 must take.
**RQ2 · Extract:** the greedy reduction criterion and stopping rule; how the
inspection system's degrees of freedom prune the candidate set; reported
final view counts.

### 9. `li-2015-freeform-structured-light-viewpoint-planning` — MED
"Viewpoint planning for freeform surface inspection using plane structured
light scanners", *Machine Intelligence Research* 12(1), 2015.
<https://link.springer.com/article/10.1007/s11633-015-0916-8> · paper

*Why:* An explicit four-step model-based recipe — topology reconstruction, mesh
refinement, scan-direction determination, viewpoint generation — with facet
grouping under total-area and normal-vector limits. The normal-vector grouping
rule is a concrete, implementable statement of "how far from orthogonal may a
patch be and still share a view", which our per-face rule needs.
**RQ1, RQ2 · Extract:** the normal-deviation and area thresholds and their
justification; how scan direction is chosen per group.
**⚠︎** STL input; structured-light, not RGB-D.

### 10. `staderini-2023-spatial-resolution-metric` — MED
V. Staderini et al., "Spatial Resolution Metric for Optimal Viewpoints
Generation in Visual Inspection Planning", *Computer Vision Systems* (ICVS
2023), LNCS 14253.
<https://link.springer.com/chapter/10.1007/978-3-031-44137-0_17> · paper

*Why:* Generates viewpoints that *guarantee* a specified spatial resolution
over the surface via ray tracing from a sensor model against a 3D mesh —
resolution as a hard generation constraint rather than a post-hoc score. Our
D405 has a fixed sensor model and a hard minimum stand-off; this is the
template for turning that into a viewpoint-generation constraint.
**RQ1, RQ2 · Extract:** the resolution metric definition and how it becomes a
generation constraint. Score-function internals → Branch 1.

### 11. `scott-2009-model-based-view-planning` — **HIGH**
W. R. Scott, "Model-based view planning", *Machine Vision and Applications*
20(1):47–69, 2009.
<https://link.springer.com/article/10.1007/s00138-007-0110-2> · paper

*Why:* The most careful treatment of *sampling the viewpoint space and the
object surface* under measurement and pose error, by the author of the survey.
Statistical models of camera and positioning-system performance — i.e. it does
not pretend the sensor and the robot are exact, which is the whole reason our
4 cm fused-vs-single-view offset matters.
**RQ1, RQ2 · Extract:** the viewpoint-space sampling scheme and its density
argument; how pose error is propagated into view planning; the reported
decimation ratio (36k vertices → ~700 → 72 candidates) as a candidate-reduction
precedent. **BM** — quality objectives and performance measures.

---

## Bucket B — Selection over candidates, and set-size control

### 12. `scott-2001-view-planning-as-set-covering` — **HIGH**
W. R. Scott, G. Roth, J.-F. Rivest, "View Planning as a Set Covering Problem",
NRC Technical Report 44892, 2001.
<https://www.researchgate.net/publication/2401756_View_Planning_as_a_Set_Covering_Problem> · paper

*Why:* The formal reduction — cover the rows of a binary measurability matrix
with a minimal subset of columns — plus an integer-programming formulation
*with a registration constraint*. The registration constraint is the precedent
for adding a non-coverage requirement (for us: reachability, and Albert's
orthogonality demand) to the cover.
**RQ2 · Extract:** the IP formulation verbatim; how the registration constraint
is encoded; their argument for why heuristics dominated the field.

### 13. `chvatal-1979-greedy-set-cover` — MED
V. Chvátal, "A Greedy Heuristic for the Set-Covering Problem", *Mathematics of
Operations Research* 4(3):233–235, 1979. (Read alongside Nemhauser–Wolsey–
Fisher 1978 for the 1−1/e maximum-coverage bound.)
<https://pubsonline.informs.org/doi/abs/10.1287/moor.4.3.233> · paper

*Why:* Settles which of the two problems we are actually solving. **Minimum set
cover** (ln n greedy guarantee) = "cover everything, use as few views as
possible". **Maximum k-coverage** (1−1/e guarantee) = "the VLM gets k views,
cover as much as possible with them". RQ3/RQ4 pressure for a *small* discrete
set pushes us toward the second, and the field mostly assumes the first without
saying so.
**RQ2, RQ3 · Extract:** the two guarantees and the exact conditions on the
coverage function; whether our coverage-with-orthogonality objective is
submodular (if it is not, the guarantee evaporates — critic ammunition).

### 14. `pito-1999-nbv-positional-space` — MED
R. Pito, "A Solution to the Next Best View Problem for Automated Surface
Acquisition", *IEEE TPAMI* 21(10):1016–1030, 1999.
<https://doi.org/10.1109/34.799908> · paper

*Why:* The positional-space formulation that decouples "which surface patches
does a view see" from "which view do I pick", so the selection step becomes a
lookup instead of a re-simulation. If we precompute a face↔view visibility
table over a geometry-conditioned candidate set, this is the classical version
of that data structure.
**RQ2, RQ3 · Extract:** the positional space construction and its cost;
whether it survives non-star-shaped objects.

### 15. `bircher-2015-structural-inspection-path-planning` — **HIGH**
A. Bircher, K. Alexis, M. Burri, P. Oettershagen, S. Omari, T. Mantel, R.
Siegwart, "Structural inspection path planning via iterative viewpoint
resampling with application to aerial robotics", *IEEE ICRA* 2015, 6423–6430.
Journal extension: *Autonomous Robots*, doi:10.1007/s10514-015-9517-1.
<https://ieeexplore.ieee.org/document/7140101/> · paper

*Why:* The single most important *mechanism* precedent in this branch. Instead
of accepting or rejecting a viewpoint, it lets each viewpoint **slide inside
its admissible region** while the tour is re-optimized — vehicle and sensor
limits are respected *inside* both steps, not by post-filtering. That is
"nearest-feasible projection" made into an algorithm, and it is directly
transplantable: replace "path cost" with "kinematic feasibility + orthogonality
loss" and the alternating scheme still runs.
**RQ2 · Extract:** the admissible-region definition per viewpoint and the
resampling operator; how vehicle constraints are enforced during resampling;
the documented weakness (it does *not* minimize the number of viewpoints —
it emits redundant views, which is fatal for us and must be noted).
**⚠︎** aerial, free-flying; tour-cost objective we explicitly do not care about.

### 16. `ethz-asl-structural-inspection-planner` — MED
ETH ASL, `StructuralInspectionPlanner`, open-source implementation of the
above, with datasets in the wiki.
<https://github.com/ethz-asl/StructuralInspectionPlanner> · repo

*Why:* A running reference implementation of viewpoint resampling we can read
rather than reimplement from the paper; the wiki datasets are a possible
benchmark seed. **BM**.

### 17. `mizutani-2024-fpt-inspection-planning` — MED
Y. Mizutani, D. Coimbra Salomao, A. Crane, M. Bentert, P. G. Drange, F. Reidl,
A. Kuntz, B. D. Sullivan, "Leveraging Fixed-Parameter Tractability for Robot
Inspection Planning", arXiv:2407.00251, 2024.
<https://arxiv.org/abs/2407.00251> · paper

*Why:* Exact algorithms (DP + ILP) for inspection planning by exploiting
structural parameters of the coverage graph. Our instances are *tiny* — a
handful of primitives, tens of candidate views — which is precisely the regime
where exact solutions beat greedy and the NP-hardness objection stops applying.
This is the paper that lets us say "we solve it optimally, and here is why
that is affordable".
**RQ2 · Extract:** which parameter is bounded and whether our
primitive-decomposed instances have it bounded; instance sizes at which exact
solving remains tractable. **BM** — their bridge-inspection and
surgical-inspection instances.

### 18. `vf-plan-2025` — MED
B. Xiong, L. Zhang, R. Huang, J. Zhou, S. R. U. N. Jafri, B. Wu, F. Li,
"VF-Plan: Bridging the Art Gallery Problem and Static LiDAR Scanning with
Visibility Field Optimization", arXiv:2503.01562, 2025. Code released.
<https://arxiv.org/abs/2503.01562> · paper

*Why:* The cleanest modern statement of the art-gallery→sensor-placement
transfer, with a **visibility field** that encodes directional and
range-dependent sensor behaviour rather than binary line-of-sight. Our D405 is
exactly a directional, range-limited sensor with a hard minimum stand-off; a
binary visibility model would let the planner pick views the sensor cannot use.
**RQ2 · Extract:** the visibility-field definition and how range/incidence
limits enter; the greedy construction and its connectivity requirement.
**⚠︎** static LiDAR, no manipulator.

### 19. `objview-bench-2026` — **HIGH**
S. Pan et al., "ObjView-Bench: Rethinking Difficulty and Deployment for
Object-Centric View Planning", arXiv:2605.10707, 2026.
<https://arxiv.org/abs/2605.10707> · paper

*Why:* **The most on-point paper found in this sweep.** It disentangles object
self-occlusion, observation saturation, and *planning difficulty measured via a
set-cover formulation*, then shows that budget regimes and **reachable-view
constraints change method rankings and failure modes**. The tabletop reachability
study is our exact setup: on a real arm, only ~45 of 128 candidate views were
reachable on average (range 26–74 across five objects × five placements), and
restricting the view space costs ~0.13–0.17 absolute surface coverage even for
the more robust planners. This is published evidence for the branch's core
claim — that a geometrically optimal view often does not exist kinematically —
and it also tells us that ranking planners on an unconstrained viewsphere is
meaningless for us.
**RQ2, RQ3 · Extract:** the set-cover difficulty measure; the reachability
filtering procedure (IK + collision) and its parameters; the budget regimes
(small K, large K, auto-stop) and the auto-stop criteria; the coverage deltas
under quarter vs. whole view space. **BM** — highest-value benchmark seed in
this branch; hand to Branch 2 immediately.

### 20. `pan-2022-scvp-one-shot-set-covering` — **HIGH**
S. Pan, H. Hu, H. Wei, "SCVP: Learning One-Shot View Planning via Set Covering
for Unknown Object Reconstruction", *IEEE RA-L* 7(2):1463–1470, 2022.
<https://ieeexplore.ieee.org/document/9670705/> · paper

*Why:* Learned selection whose *training labels* come from a set-cover solver
over known 3D models — so the network is a fast approximation of exactly the
optimization our analytical track would run. Crucially it has a **built-in
auto-stop**: the smallest view set achieving full coverage. Set-size minimality
as a learned property is the direct answer to the VLM's small-discrete-set
pressure.
**RQ2, RQ3 · Extract:** the set-cover labelling procedure (this is the part we
might use *without* the network); reported view counts vs. iterative NBV; the
stated failure mode (single-view input gives insufficient surface detail).

### 21. `pan-2024-ma-scvp` — **HIGH**
S. Pan, H. Hu, H. Wei, N. Dengler, T. Zaenker, M. Dawood, M. Bennewitz,
"Integrating One-Shot View Planning with a Single Next-Best View via Long-Tail
Multiview Sampling", *IEEE T-RO*, 2024. arXiv:2304.00910. Code released.
<https://arxiv.org/abs/2304.00910> · paper
Repo: <https://github.com/psc0628/MA-SCVP> · repo

*Why:* The hybrid that matters for us: gather *some* information first, then
plan the whole set in one shot. We already have a survey cloud, i.e. we start
where their pipeline starts after its NBV — so their one-shot branch is our
whole problem. Reports a 45% movement-cost reduction vs. iterative NBV. Repo
ships a view-planning simulator, a dataset generator and a ROS interface.
**RQ2, RQ3 · Extract:** how much prior information the one-shot predictor needs
before its predictions become reliable; the view-space discretization it
predicts over (also Branch 5); the ROS interface's arm assumptions.

### 22. `pan-2025-previous-session-view-planning` — MED–HIGH
S. Pan, L. Lobefaro, M. Taherkhani, X. Huang, R. Menon, C. Stachniss, M.
Bennewitz, "Efficient View Planning Guided by Previous-Session Reconstruction
for Repeated Plant Monitoring", arXiv:2510.07028, 2025.
<https://arxiv.org/abs/2510.07028> · paper

*Why:* Uses a *prior-session reconstruction* — degraded, incomplete, not a CAD
model — to drive one-shot view selection, replacing the iterative NBV loop.
That is structurally identical to "use the survey PLY to derive the view set",
and it is the only entry that treats the prior geometry as evidence rather than
as truth.
**RQ2 · Extract:** how they quantify trust in the previous-session
reconstruction; what degradation breaks the transfer (cross-tag Branch 3).
**⚠︎** plants deform between sessions; our object does not.

### 23. `pan-2024-how-many-views-nerf-prv` — MED
S. Pan, L. Jin, H. Hu, M. Popović, M. Bennewitz, "How Many Views Are Needed to
Reconstruct an Unknown Object Using NeRF?", *IEEE ICRA* 2024. arXiv:2310.00684.
Code: <https://github.com/psc0628/NeRF-PRV>
<https://arxiv.org/abs/2310.00684> · paper

*Why:* Predicts the *required number of views* from object complexity before
planning any of them. RQ3/RQ4 need the action set to be small and to have a
defensible size; "how many views does this shape deserve" is a question we will
have to answer per primitive, and this is the only paper that asks it directly.
**RQ2, RQ3 · Extract:** the complexity features that drive the predicted count;
whether the relation is simple enough to write analytically for primitives
(box/cylinder) instead of learned. **BM**.

### 24. `occamview-2026` — MED
H. Gao, W. Zhang, Z. Ni, D. Zhu, R. Li, Y. Wang, C. Xu, "OccamView:
Object-Conditioned View Selection for Frame-Budgeted Active 3D Gaussian
Reconstruction", arXiv:2608.16499, 2026.
<https://arxiv.org/abs/2608.16499> · paper

*Why:* Explicitly optimizes selection **under a tight frame budget** with an
occlusion-aware proxy-coverage score and a "Geo-Floor" mechanism that refuses
geometrically weak candidates. The budget regime is ours (a VLM cannot be
handed 128 views), and the Geo-Floor is a concrete pattern for "never offer a
view that is below a geometric quality floor" — i.e. never offer a badly
non-orthogonal view at all.
**RQ2, RQ3 · Extract:** the Geo-Floor admissibility rule; how ranking changes
as the budget shrinks. **⚠︎** 3DGS reconstruction objective, not defect
inspection.

### 25. `benbv-2024-boundary-exploration` — MED
R. Li, C. Wan, Y. Zhang, "Boundary Exploration of Next Best View Policy in 3D
Robotic Scanning" (BENBV), arXiv:2412.10444, 2024.
<https://arxiv.org/abs/2412.10444> · paper

*Why:* Contains the third-party head-to-head we need: learned (PC-NBV) vs.
search-based vs. random-sphere baselines on the same objects, and learned does
*not* dominate — PC-NBV wins on Chamfer/Hausdorff on ModelNet40 while losing on
overlap consistency. That is the evidence base for the branch's question "does
learning buy anything in our regime, or does analytical selection dominate?"
**RQ2 · Extract:** the comparison table and the exact baselines, including
"Random Uniform Sphere" — that baseline is *our current system*, so its numbers
are our floor. **BM** — strong benchmark seed; hand to Branch 2.

### 26. `gennbv-2024` — MED
R. Chen et al., "GenNBV: Generalizable Next-Best-View Policy for Active 3D
Reconstruction", *CVPR* 2024. arXiv:2402.16174. Code:
<https://github.com/zjwzcx/GenNBV>
<https://arxiv.org/abs/2402.16174> · paper

*Why:* The strongest argument for RL-based selection, and it makes the argument
by *enlarging* the action space to 5D free space — the opposite of our
direction. Read it as the counter-position: if the field's best learned policy
needs a free-flying 5D action space to beat heuristics, it is telling us
something about what a constrained, discrete, arm-mounted action space gives up.
**RQ2 · Extract:** the baselines actually benchmarked (they skip several,
citing pretraining dependence — a critic flag); the reported coverage on unseen
geometry. **⚠︎** free-flying drone, 5D continuous action space — violates both
the discrete-selection given and the reachability constraint.

---

## Bucket C — Feasibility coupling (the branch's distinguishing question)

### 27. `wang-2026-observation-constrained-joint-space-viewpoint` — **HIGH**
Y. Wang, R. Yin, B. Zhang, S. Schwertfeger, "Observation-Constrained
Joint-Space Viewpoint Optimization for Robotic Inspection of Cylindrical
Cavities", arXiv:2608.16442, 2026.
<https://arxiv.org/abs/2608.16442> · paper

*Why:* Optimizes in **joint space** rather than Cartesian view space, so
visibility, kinematic and collision constraints are satisfied by construction
instead of by filtering — the ideal-view-unreachable problem simply cannot
arise. Multistart derivative-free search then ranks feasible configurations by
motion economy *and* view quality (92% success on 100 targets). It is also
about a **cylinder**, our second primitive.
**RQ2, RQ3 · Extract:** the joint-space objective and how view quality is
expressed on joint variables; the multistart budget and its runtime; how they
handle IK multiplicity (a UR5e has up to 8 solutions per pose — which one
becomes "the" view?); their cylinder-specific viewpoint pattern.

### 28. `yu-gupta-2000-cspace-entropy-eye-in-hand` — **HIGH**
Y. Yu, K. Gupta, "View Planning via C-space Entropy for Efficient Exploration
with Eye-in-Hand Systems", *ISRR/Experimental Robotics*, LNCIS, 2000; and
"An information theoretical approach to view planning with kinematic and
geometric constraints", *IEEE ICRA* 2001.
<https://link.springer.com/chapter/10.1007/3-540-45118-8_38> · paper

*Why:* The founding work that says view planning for an **eye-in-hand arm** is
not view planning for a free camera: the quantity worth maximizing is
information about the robot's own configuration space, because that is what
determines which views exist next. Twenty-five years old and still the most
principled statement of why our constraint stack is not a post-processing step.
**RQ2, RQ3 · Extract:** the C-space entropy formulation and its computational
cost; the argument for why Cartesian-space view utility is the wrong objective
for an arm. **⚠︎** exploration of unknown space (our geometry is surveyed) —
borrow the coupling argument, not the loop.

### 29. `glorieux-2020-targetted-viewpoint-sampling` — **HIGH**
E. Glorieux, P. Franciosa, D. Ceglarek, "Coverage path planning with targetted
viewpoint sampling for robotic free-form surface inspection", *Robotics and
Computer-Integrated Manufacturing* 61:101843, 2020.
<https://www.sciencedirect.com/science/article/abs/pii/S0736584518305246> · paper

*Why:* Industrial, arm-mounted, and constrained by robot kinematics *and*
collision avoidance *and* full coverage simultaneously — the same triple we
have. Its contribution is replacing random viewpoint sampling with a targeted
scheme, cutting inspection cycle time up to 23.8% on real sheet-metal parts.
Targeted sampling from surface structure is the sampling-side sibling of our
per-primitive rules.
**RQ2 · Extract:** the targeting rule; where kinematics enters (sampling vs.
filtering vs. objective); the reported viewpoint counts and cycle times.
**BM** — real industrial parts, real cycle-time metric.

### 30. `spensieri-2025-unified-sampling-coverage-robot-placement` — **HIGH**
D. Spensieri, E. Åblad, R. Salman, J. S. Carlson, "A unified sampling method
for optimal feature coverage and robot placement", *Robotics and
Computer-Integrated Manufacturing* 93:102932, 2025.
<https://www.sciencedirect.com/science/article/pii/S0736584524002199> · paper

*Why:* Solves feature coverage and robot/sensor placement as **one** problem,
minimizing the number of sensor locations. This is the industrial-grade version
of "do not select views and then check reachability — select over the set of
reachable, covering configurations", and it is validated on real cells.
**RQ2 · Extract:** the unified formulation and what makes it customizable; how
they keep the joint problem tractable; whether the minimization is
cardinality-based (matches our set-size pressure).
**⚠︎** offline cell design; assumes exact CAD.

### 31. `zacharias-2007-capability-map` — **HIGH**
F. Zacharias, C. Borst, G. Hirzinger, "Capturing robot workspace structure:
representing robot capabilities", *IEEE/RSJ IROS* 2007, 3229–3236.
doi:10.1109/IROS.2007.4399105 · paper

*Why:* The capability map — a voxelized workspace where each cell stores the
fraction of orientations reachable there. This is the data structure that turns
"is this view reachable?" from a millisecond IK call into a lookup, and, more
usefully for us, turns it into a **continuous score** we can use to pick the
nearest-feasible alternative when the orthogonal view fails. Filter candidates
by map, run IK only on survivors.
**RQ2, RQ3 · Extract:** the SO(3) discretization and the reachability index
definition; reported prediction accuracy and false-positive/false-negative
behaviour (a false positive here means offering the VLM a view the robot cannot
execute — an action-space integrity failure, not just an inefficiency).

### 32. `reuleaux-2017-reachability-base-placement` — MED
A. Makhal, A. K. Goins, "Reuleaux: Robot Base Placement by Reachability
Analysis", arXiv:1710.01328, 2017. ROS package.
<https://arxiv.org/abs/1710.01328> · paper/repo

*Why:* The practical, open-source realization of reachability/inverse-
reachability maps in ROS. Even though we have a fixed base, the map generation
and query machinery is directly reusable to precompute the UR5e's reachable
shell around the tabletop workspace once, offline.
**RQ2 · Extract:** map resolution/build cost trade-off; whether the
implementation is usable standalone or ROS-locked (auxiliary goal wants
something integrable).

### 33. `cmm-visibility-maps-accessibility-cones` — MED
"Part orientations for CMM inspection using dimensioned visibility maps",
*Computer-Aided Design* 30(9), 1998 — read with Spyridi & Requicha's
accessibility cones and Vafaeesefat & ElMaraghy, "Automated accessibility
analysis and measurement clustering for CMMs", *Int. J. Production Research*
38(10):2215–2231, 2000.
<https://www.sciencedirect.com/science/article/abs/pii/S0010448598000347> · paper

*Why:* Metrology solved a structurally identical problem thirty years ago:
compute, per feature, the *set of directions from which it can be measured*
(accessibility cone / visibility map), then **cluster features that share
accessible directions** to minimize probe orientations and part setups. Swap
"probe orientation" for "camera pose" and "setup" for "reachable arm
configuration" and this is our selection problem with a minimality objective
built in. It is also the precedent Anton asked for: deterministic, per-feature
view derivation blessed by industrial practice.
**RQ2, RQ3 · Extract:** the accessibility-cone construction; the clustering
formulation and its objective (minimum number of orientations); how they treat
partial vs. complete visibility. Cross-tag Branch 5 for the spherical
visibility-map representation.

### 34. `cpp-quality-inspection-measurement-uncertainty-2022` — MED
"Coverage Path Planning for Robotic Quality Inspection with Control on
Measurement Uncertainty", arXiv:2201.04310, 2022.
<https://arxiv.org/abs/2201.04310> · paper

*Why:* Carries an explicit, minimal **feasibility predicate** — a viewpoint is
feasible iff it covers ≥1 measurement point *and* the robot does not collide —
inside an RRT*-style viewpoint search, and simultaneously controls measurement
uncertainty. It is the cleanest small example of feasibility-as-a-predicate we
can copy directly.
**RQ1, RQ2 · Extract:** the predicate and where it sits in the search loop; how
measurement uncertainty is modelled (cross-tag Branch 1).

### 35. `nbv-hand-eye-fixed-base-arm-2025` — MED
"A Next-Best-View Method for Complex 3D Environment Exploration Using Robotic
Arm with Hand-Eye System", *Applied Sciences* 15(14):7757, 2025. Open access.
<https://www.mdpi.com/2076-3417/15/14/7757> · paper

*Why:* States our objection to the mainstream literature in print: frontier-
based exploration from mobile robotics "does not translate well to the
kinematic constraints and limited reach of a fixed-base manipulator", and
learning-based methods generalize poorly to novel workcell layouts. Useful both
as a method and as a citable justification for why we are not simply adopting an
off-the-shelf NBV planner.
**RQ2 · Extract:** their reachability handling; the quantified failure of
transplanted mobile-robot methods.

### 36. `compass-2025-confined-space-active-sensing` — MED
"COMPASS: Confined-space Manipulation Planning with Active Sensing Strategy",
arXiv:2509.14787, 2025.
<https://arxiv.org/abs/2509.14787> · paper

*Why:* Handles the case where the manipulator's *own embodiment* shrinks the
reachable view set, using a multi-objective utility that trades informativeness
against downstream feasibility — +24.25% manipulation success over
information-gain-only planning. Our RG2+D405 hull and 4 cm minimum stand-off
produce the same pathology at tabletop scale.
**RQ2 · Extract:** the multi-objective utility weights and how the feasibility
term is computed; the ablation isolating the feasibility term.

### 37. `gs-nbv-2025-constrained-1d-viewpoint-circle` — MED
X. Song, K. Karydis, "GS-NBV: a Geometry-based, Semantics-aware Viewpoint
Planning Algorithm for Avocado Harvesting under Occlusions", arXiv:2506.23369,
2025. Code released.
<https://arxiv.org/abs/2506.23369> · paper

*Why:* Deliberately collapses the search space to a **1D circle** derived from
geometric cues, on an arm, and reports 100% success under heavy occlusion. That
is a cylinder-orbit action space in all but name, and it is direct evidence that
aggressively shrinking a geometry-derived view space *helps* rather than hurts —
the counter-argument to "geometry-conditioned means complicated".
**RQ2, RQ3 · Extract:** how the circle is derived and parameterized (Branch 5
will want the addressing); what is given up by the restriction; the picking-score
feasibility terms.

---

## Bucket D — Cross-tags (owned elsewhere, but read from here)

### 38. `scanbot-2026` — MED
Z. Chen, Y. Zhang, F. Jalil Piran, Q. Zhou, J. Tang, F. Imani, "ScanBot: A
Benchmark for Precision Robotic Surface Scanning with Industrial Laser
Profilers", arXiv:2505.17295 (v2 2026, IROS 2026). Data/code:
<https://github.com/Ed1sonChen/Scan-Bot>
<https://arxiv.org/abs/2505.17295> · paper

*Why:* UR3 + eye-in-hand profiler + D435i, twenty objects, instruction-
conditioned scanning, and a **two-stage protocol** (sensor-parameter selection,
then feasible target-covering motion "without extra moves") with an industrial
fixed-template baseline. The fixed-template baseline is ADAM; the "without extra
moves" criterion is set-size minimality made into a metric. Also reports that
VLA/MLLM models still fail to produce stable, feasible scanning motion.
**RQ2 · Extract:** the Stage-II feasibility and coverage metrics. **BM** —
strong protocol seed; primary ownership Branch 2.

### 39. `sun-2025-text-guided-inspection-plans` — MED
X. Sun, Z. Pan, X. Gao, K. Wu, A. Bera, "Text-guided Generation of Efficient
Personalized Inspection Plans", arXiv:2506.02917, 2025.
<https://arxiv.org/abs/2506.02917> · paper

*Why:* Training-free VLM in the loop over a **known** scene: the model picks
points of interest and waypoints where those POIs are salient, then classical
optimization takes over. It is the clearest existing division of labour between
"language model chooses what to look at" and "geometry decides where to stand" —
which is the architecture our RQ2/RQ4 boundary implies.
**RQ2, RQ4 · Extract:** exactly which decisions the VLM makes vs. the optimizer;
how candidate waypoints are presented to the model. Interface details →
Branch 6.

---

## Questions this branch must answer

Ordered by how load-bearing they are for RQ2/RQ3. These are the reading tasks;
each one names the documents most likely to hold the answer.

**On the formulation**

1. **Is our problem minimum set cover or maximum k-coverage?** "Cover the whole
   surface with as few views as possible" and "given the VLM k views, cover as
   much as possible" are different problems with different guarantees and
   different failure modes. Which one does Albert's requirement (evidence images
   are the most orthogonal view of *the inspected surface*) actually imply — is
   coverage even the objective, or is it per-region view *quality* subject to a
   coverage floor? (13, 12, 24)
2. **Is our coverage objective submodular?** Greedy's guarantee dies if adding
   an orthogonality or quality weighting breaks submodularity. If it breaks, do
   we care — see Q3. (13, 17)
3. **At our instance size, can we just solve it exactly?** A box gives ~6–10
   candidate faces; a decomposed object maybe tens of candidates and tens of
   surface regions. Does the NP-hardness objection apply at all, or is an ILP /
   FPT solve instantaneous and optimal? If it is, the field's entire heuristic
   apparatus is irrelevant to us and we should say so explicitly. (17, 12, 30)
4. **Does the "generate candidates then select" pipeline earn its keep, or
   should views be derived directly?** Magaña's constrained-ℂ-spaces (6) and
   Wang's joint-space optimization (27) both skip candidate enumeration.
   Enumeration is what makes a discrete action space presentable to a VLM (RQ4),
   so if we skip it we must re-enumerate afterwards — at what cost?

**On per-primitive view recipes**

5. **What is the published precedent for "one orthogonal view per face, a ring
   per cylinder"?** No paper found names this recipe. The nearest precedents are
   metrology accessibility cones + feature clustering (33), adaptive feature-
   driven subdivision (7), facet grouping under normal-deviation limits (9), and
   GS-NBV's 1D circle (37). Assemble the strongest citation chain — Anton needs
   this to defend "primitive-level deterministic rules" as standing on precedent
   rather than as a fresh hand-programmed recipe.
6. **How many views does a primitive deserve, and what sets the number?** For a
   cylinder: what fixes the number of orbit stations — field of view, incidence
   tolerance, overlap fraction, defect size? Extract the actual thresholds
   (9, 10, 23), and check whether a closed-form per-primitive answer exists so we
   never need a learned budget predictor.
7. **What is the maximum admissible deviation from orthogonality before a
   surface region needs its own view?** This single number converts "coverage"
   into "coverage at acceptable quality" and sets the view count for every
   primitive. (9, 10, 24 Geo-Floor; formula internals → Branch 1)

**On the constraint stack (critic mandate #4)**

8. **What does the field actually do when the ideal view is unreachable?** So
   far four distinct answers: (i) hard filter and discard (19, 34); (ii) slide
   the viewpoint inside an admissible region and re-optimize (15); (iii) never
   leave the feasible set — optimize in joint space (27); (iv) defer rather than
   delete, re-testing later (28, 36). Enumerate these as a **fallback ladder**
   with the conditions under which each is correct, and state which one our
   architecture takes. This is the branch's headline deliverable.
9. **Filter up front or post-filter?** Capability-map pre-filtering (31, 32) is
   cheap but approximate; per-candidate IK is exact but slow. What false-positive
   rate does a capability map have, and what does a false positive cost *us*
   specifically — namely offering the VLM an action it cannot execute, which is
   an action-space integrity failure rather than a wasted cycle? (31, 19)
10. **How much of an object's surface becomes unreachable on a tabletop arm?**
    ObjView-Bench (19) gives ~45/128 reachable views and a 0.13–0.17 absolute
    coverage loss. Reproduce the equivalent number for our UR5e + RG2 + D405 +
    minimum-shell setup — it bounds what any view-planning method can achieve
    for us and is the honest denominator for every later result.
11. **How does IK multiplicity interact with a discrete action space?** A UR5e
    pose admits up to eight solutions with different collision and joint-limit
    behaviour. Is "a view" a Cartesian pose or a joint configuration? If the
    latter, the action space is larger than the geometry suggests; if the former,
    execution may fail after the VLM has committed. (27, 31)
12. **Does the reachable set change the *shape* of the answer, or only prune
    it?** If the reachable shell systematically excludes, say, all top-down views
    of tall objects, then "coverage" is unattainable by construction and the
    objective must be redefined (best-available view per region, with an explicit
    unreachable-region report). (19, 30)

**On set size and the VLM's pressure**

13. **What set sizes do these methods actually emit, and how are they
    controlled?** Collect reported view counts per method and per object class
    (7, 8, 20, 21, 23, 29). Bircher (15) is explicitly documented as emitting
    redundant views — establish whether the same critique applies to each
    candidate we might adopt.
14. **Is there a principled stopping/minimality criterion we can implement?**
    SCVP's learned auto-stop (20), NeRF-PRV's predicted budget (23), Chvátal's
    ln n greedy termination (13), ObjView-Bench's auto-stop regimes (19) — pick
    one and be able to defend it.

**On learning vs. analysis**

15. **Does learned selection beat analytical selection for primitive-shaped
    objects with known geometry?** BENBV's third-party comparison (25) says
    learned methods win on speed and coverage-per-scan but not uniformly on
    accuracy; GenNBV (26) buys its wins with a free-flying 5D action space we do
    not have. Write the verdict for *our* regime, with the caveat that our
    "Random Uniform Sphere" baseline is literally the system we are replacing —
    so any number reported for that baseline is a free estimate of our current
    performance.
16. **Does the field ever plan views from a defective survey cloud rather than a
    model?** Only (22) treats prior geometry as evidence rather than truth. If
    the answer is essentially "no", that gap is a contribution claim — and a risk
    that must be handed to Branch 3.

---

## Gaps and negative results found in this sweep

Recorded so the synthesis does not have to rediscover them.

- **No paper found conditions viewpoint generation on a fitted primitive
  decomposition** (superquadric/cuboid/cylinder fitting → per-primitive view
  recipe). Superquadric decomposition (e.g. SuperDec, arXiv:2504.00992) and
  view planning are both mature and appear not to have been fused. If confirmed
  in round 3, this is the branch's clearest novelty claim — and its clearest
  risk, since it means no precedent for how well it works.
- **Almost nothing plans views from a defective, self-acquired survey cloud.**
  The literature splits into model-based (clean CAD/mesh) and non-model-based
  (nothing known, explore online). Our regime — a bad model — is between them and
  barely served; (22) is the only close entry.
- **Reachability is overwhelmingly treated as a post-filter**, not as part of
  the objective. Explicit joint-space formulations (27) and unified
  coverage+placement formulations (30) exist but are rare, and the arXiv sweep on
  `next-best-view AND manipulator` returned essentially no paper documenting IK
  filtering as a first-class methodological component for eye-in-hand cameras.
- **3D coverage has no clean art-gallery bound.** Unlike the 2D ⌊n/3⌋ result,
  interior guarding of polyhedra admits no bound in n alone (Seidel's Ω(n²)
  example), and boundary covering vs. interior covering are separately NP-hard
  with no known transformation between them. Any "minimum number of views"
  claim we make must be empirical or exact-per-instance, never a general bound.
- **Redundancy is a documented, recurring failure of coverage planners** — most
  explicitly for Bircher et al. (15). Given RQ3/RQ4's small-set pressure, every
  method we consider must be screened for it up front.

---

## Search trail

Tools: `WebSearch` (exhausted at 200 calls for the session), `WebFetch` against
arXiv abstract pages and the arXiv API, publisher landing pages, dblp, GitHub.
Abstracts and landing pages only — no deep reading, per the mapping brief.

Queries run (paraphrased, in order):

1. `"view planning problem" survey Scott Roth Rivest automated 3D object
   reconstruction inspection` → Scott 2003, Scott 2001 set covering, CAD 2021
   survey.
2. `model-based view planning CAD-based inspection planning viewpoint
   generation survey 2024` → Scott 2009, Mosbach 2021, Staderini 2023, Mavrinac
   2015, voxelization/STL pipelines.
3. `arxiv next best view planning survey 2023–2025 active vision object
   reconstruction review` → Alsadik 2025 taxonomy, GenNBV, ObjView-Bench,
   BENBV.
4. `reachability aware viewpoint planning manipulator capability map inspection
   IK feasible viewpoints` → capability/reachability maps, Reuleaux, apple-tree
   active perception, Glorieux, CPP-with-uncertainty.
5. `structural inspection path planning Bircher iterative viewpoint resampling`
   → Bircher 2015 + Autonomous Robots extension + ETH repo + the redundancy
   critique.
6. `"set covering" viewpoint selection greedy submodular sensor placement 3D
   inspection NP-hard` → Chvátal, Nemhauser–Wolsey–Fisher, Wolsey; min-cover vs.
   max-k-coverage distinction.
7. `photogrammetric network design camera network optimization close-range` →
   NDP framing, convergent/circular block geometry, Fraser; contested
   set-cover-equivalence claim.
8. `viewpoint planning robotic surface defect inspection industrial without CAD
   point cloud` → model-based vs. non-model-based split, RANSAC-segmented
   region planning (30–48 viewpoints/part), ScanBot.
9. `arxiv ObjView-Bench object-centric view planning benchmark 2026` →
   ObjView-Bench, E3VS-Bench, Planning with the Views.
10. `learned set covering view planning SCVP one-shot Pan Wei` → SCVP, MA-SCVP
    + repo, DM-OSVP++, one-shot lineage.
11. `Text-guided Generation of Efficient Personalized Inspection Plans` →
    arXiv:2506.02917.
12. `viewpoint planning unreachable fallback constraint relaxation nearest
    feasible collision-aware` → the four-answer fallback ladder (filter /
    resample / joint-space / defer).
13. `viewpoint planning geometric primitive decomposition box faces cylinder
    orbit scanning pattern` → Tarbox, Pito, Fan 2016, generalized-cylinder
    envelope tessellation; **no named box-face/cylinder-orbit recipe found**.
14. `CMM accessibility analysis probe orientation clustering setup planning` →
    accessibility cones (Spyridi & Requicha), dimensioned visibility maps,
    Vafaeesefat & ElMaraghy clustering.
15. `"Autonomous view planning methods for 3D scanning" review 2024` → Lee et
    al., *Automation in Construction* (corrected venue); pointer to Spensieri
    RCIM 2025.
16. `Tarbox Gottschlich planning for complete sensor coverage in inspection` →
    CVIU 1995 + IVIS companion.
17. `Spensieri unified sampling optimal feature coverage and robot placement` →
    RCIM 93:102932.
18. `Mosbach Gospodnetić feature-driven viewpoint placement` → MVA 2021 + MVA
    2022 companion + the "candidate explosion" critique of vertex-based
    generation.
19. `GenNBV PC-NBV NBV-Net learned NBV vs analytical baseline` → BENBV
    third-party comparison table.
20. `arXiv 2026 view planning manipulator eye-in-hand tabletop kinematic
    constraints reachable view space` → ObjView-Bench reachability numbers,
    COMPASS, Applied Sciences hand-eye NBV, Yu & Gupta lineage.
21. `Alsadik structured review taxonomy NBV` / `CAD 2021 VPP survey` →
    bibliographic confirmation.
22. `shape-conditioned view planning primitive decomposition superquadric` →
    SuperDec; **confirmed gap** between primitive decomposition and view
    planning.
23. `arXiv 2407.15883 quality surface coverage multi-view capture` /
    `arXiv 2505.17295 ScanBot` → coverage-quality and benchmark entries.
24. arXiv API sweeps (WebSearch budget exhausted):
    `abs:"view planning"`, `abs:"viewpoint planning" OR abs:"inspection
    planning"`, `all:"reachability" AND all:"viewpoint"`,
    `abs:"coverage" AND abs:"set cover" AND abs:"viewpoint"`,
    `abs:"next-best-view" AND abs:"manipulator"`, all sorted by submission date
    → surfaced OccamView (2608.16499), Observation-Constrained Joint-Space
    (2608.16442), previous-session one-shot planning (2510.07028), VF-Plan
    (2503.01562), FPT inspection planning (2407.00251), flow-based MILP
    inspection (2603.16593), GS-NBV (2506.23369), implicit inspection-planning
    primitives (2510.07611), clustering-based building inspection view planning
    (2408.01435).
25. Landing-page fetches to confirm scope: 2605.10707, 2605.29563, 2306.06969,
    2608.16442, 2608.16499, 2510.07028, 2510.07611, 2503.01562, 2407.00251,
    2506.23369, 2603.16471, 2605.17593, 2007.10452, 2503.10334, 2509.21370,
    2603.23112; GitHub `psc0628/MA-SCVP`.

Deliberately not pursued (out of branch scope): exploration NBV for unknown
scenes as an end in itself; inter-view trajectory/tour optimization and TSP
variants; UAV/mobile coverage path planning except where the viewpoint math
transfers (Bircher, VF-Plan); NBV score-function internals (Branch 1);
viewpoint-space addressing schemes (Branch 5); VLM interface encodings
(Branch 6).
