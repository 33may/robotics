# Branch 1 — `view-quality-scoring`
## Scoring a single view: metrics and local optimality — Round 2 document map

**Date:** 2026-08-27
**Role:** literature mapper (judgment tier), pass 1 / round 2
**Branch plan:** [`../round1_plan.md`](../round1_plan.md) § Branch 1
**Contract:** [`../../../../2026-08-27-geometry-aware-view-planning.md`](../../../../2026-08-27-geometry-aware-view-planning.md)
**Method:** abstracts and landing pages only. No deep reads. 40 documents, ranked.

---

## How to read this map

Every entry carries tags used by the readers and by the round-2 synthesis:

- **Space** — `PLY` (score computable from the point cloud + camera intrinsics),
  `IMG` (score needs the captured/rendered image), `BOTH`, `SIM` (needs a
  simulator/renderer we would have to build).
- **Violates** — the contract given the paper assumes away (`CAD` = assumes a
  clean mesh/CAD model, `FREE-CAM` = free-flying camera with no IK/collision,
  `CONT` = continuous pose output, `–` = compatible).
- **B2** — flag for Branch 2 (`benchmarking-view-planning`): its evaluation
  setup, dataset, or ground-truth protocol could seed our benchmark.
- **Formula** — `yes` if the abstract/landing page indicates a closed-form or
  otherwise directly implementable definition; `impl` if code exists; `?` if
  unknown from the abstract.

The single most important extraction question for this branch, applied to every
candidate: **can this score be computed on a noisy fused PLY with known D405
intrinsics, and does it change monotonically as the camera rotates off the
surface normal?** Anything that cannot answer both is low value regardless of
citation count.

---

## Tier 1 — spine (read first, in this order)

These five give the vocabulary. Everything else is a leaf off them.

### 1. `tarabanis-sensor-planning-survey`
**A Survey of Sensor Planning in Computer Vision** — Tarabanis, Allen, Tsai.
*IEEE T-RA* 11(1):86–105, 1995.
<https://www.semanticscholar.org/paper/A-survey-of-sensor-planning-in-computer-vision-Tarabanis-Allen/fca269c55ff935d4bcc52b11032d0cabee75e80c>
**Type:** paper · **Space:** PLY · **Violates:** CAD · **B2:** no · **Formula:** partial
**Priority: HIGH.** The origin of the constraint vocabulary we need — visibility,
field of view, focus/depth of field, resolution — as *separately characterizable
domains* rather than a blended heuristic. Read it to inherit the decomposition,
not the algorithms.

### 2. `tarabanis-feature-detectability-constraints`
**Analytical Characterization of the Feature-Detectability Constraints of
Resolution, Focus, and Field of View for Vision Sensor Planning** — Tarabanis,
Tsai, Allen. *CVGIP: Image Understanding* 59(3):340–358, 1994.
<https://www.academia.edu/23026191/The_MVP_sensor_planning_system_for_robotic_vision_tasks> (companion; primary via IEEE/Elsevier)
**Type:** paper · **Space:** PLY · **Violates:** CAD · **B2:** no · **Formula:** yes
**Priority: HIGH.** The highest-density source of *actual analytic formulas* in the
whole branch: each constraint is characterized as a closed-form region of
viewpoint space. This is the direct feedstock for the analytical track — if the
resolution and FOV constraints are already solved regions, our job on a box face
reduces to intersecting them with the orthogonality constraint.

### 3. `tarabanis-mvp-system`
**The MVP Sensor Planning System for Robotic Vision Tasks** — Tarabanis, Tsai,
Allen. *IEEE T-RA* 11(1):72–85, 1995.
<https://link.springer.com/chapter/10.1007/978-94-011-2560-4_33>
**Type:** paper · **Space:** PLY · **Violates:** CAD, FREE-CAM · **B2:** no · **Formula:** yes
**Priority: HIGH.** The "generalized viewpoint" — an 8-D imaging space (position,
orientation, focal length, aperture, principal-point distance) — plus the move
from constraint satisfaction to *optimization with a margin*. That margin concept
is exactly RQ1's "is this the best view within its neighbouring region", posed 30
years ago; extract how they define and maximize it.

### 4. `scott-roth-rivest-vpp-survey`
**View Planning for Automated Three-Dimensional Object Reconstruction and
Inspection** — Scott, Roth, Rivest. *ACM Computing Surveys* 35(1):64–96, 2003.
<https://dl.acm.org/doi/10.1145/641865.641868> · free copy at NRC Publications Canada
**Type:** paper · **Space:** PLY · **Violates:** CAD · **B2:** yes · **Formula:** partial
**Priority: HIGH.** The canonical taxonomy; shared spine with Branch 4 (they own
selection, we own the per-view quality criteria and the performance measures
section). Also the standard citation for "view planning remains an open problem",
which the review will need.

### 5. `zeng-view-planning-survey`
**View Planning in Robot Active Vision: A Survey of Systems, Algorithms, and
Applications** — Zeng, Wen, Zhao, Liu. *Computational Visual Media* 6(3):225–245,
2020. Open access.
<https://www.sciopen.com/article/10.1007/s41095-020-0179-3>
**Type:** paper · **Space:** BOTH · **Violates:** – · **B2:** yes · **Formula:** no
**Priority: HIGH.** The modern bridge: covers the 2003→2020 gap, is open access,
and is the survey a reviewer will expect us to have read. Extract its utility
function taxonomy only; systems/applications belong to Branch 4.

---

## Tier 2 — the orthogonality argument, grounded in measurement physics

This cluster is the branch's obligation to the critic: *why* is orthogonal
better? Not intuition — signal-to-noise and sampling density.

### 6. `nguyen-izadi-kinect-noise-model`
**Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking** —
Nguyen, Izadi, Lovell. *3DIMPVT* 2012, pp. 524–530.
<https://users.cecs.anu.edu.au/~nguyen/papers/conferences/Nguyen2012-ModelingKinectSensorNoise.pdf>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: HIGH.** The canonical empirical model of **axial and lateral depth
noise as a joint function of range and incidence angle**, with fitted curves. This
is the single citation that converts "Albert wants orthogonal" into "noise
explodes past ~60° and the D405 stops resolving the defect". Also a ready-made
noise injector for the Branch 2 benchmark.

### 7. `soudarissanane-incidence-angle-tls`
**Incidence Angle Influence on the Quality of Terrestrial Laser Scanning Points**
— Soudarissanane, Lindenbergh, Menenti, Teunissen. *ISPRS Workshop Laser
Scanning* 2009.
<https://gnss.curtin.edu.au/wp-content/uploads/sites/21/2016/04/Soudarissanane2009Incidence.pdf>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Derives noise growth with incidence angle from the
signal-to-noise ratio, validated on a plate scanned at controlled angles.
Defines α as the angle between beam and surface normal — our orthogonality
score's exact variable, in a metrology venue with a physical derivation.

### 8. `soudarissanane-scanning-geometry-isprsj`
**Scanning Geometry: Influencing Factor on the Quality of Terrestrial Laser
Scanning Points** — Soudarissanane et al. *ISPRS J. Photogrammetry & Remote
Sensing*, 2011.
<https://www.sciencedirect.com/science/article/abs/pii/S0924271611000098>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: HIGH.** The journal extension: joint range × incidence model,
per-point precision by least squares, and **propagation from per-point precision
to whole-cloud quality**. That propagation is the template for scoring a *surface
region* rather than a point — which is literally the RQ1 unit.

### 9. `intel-d405-product-and-datasheet`
**Intel RealSense Depth Camera D405 — product page + datasheet**
<https://www.intelrealsense.com/depth-camera-d405/>
**Type:** docs · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Our sensor's own numbers: ideal range 7–50 cm, sub-mm at the
near end, <2% depth error at 50 cm (≈10 mm). These bound the standoff term of any
view score and interact hard with the "hard minimum shell radius" given — worth
checking whether the reachable shell even lands inside the D405's good range.

### 10. `intel-realsense-tuning-whitepapers`
**Intel RealSense whitepapers: depth camera tuning, on-chip self-calibration,
depth post-processing**
<https://dev.intelrealsense.com/docs>
**Type:** docs · **Space:** PLY · **Violates:** – · **B2:** no · **Formula:** partial
**Priority: MED-HIGH.** Practitioner-grade: preset/resolution choices (848×480 is
the accuracy optimum), disparity shift to push minimum range to ~4 cm, and the
post-processing filter chain. Directly relevant to whether the measured ~4 cm
fused-vs-single-view offset is a calibration artefact rather than a fact of life.

### 11. `halmetschlager-funek-ten-depth-cameras`
**An Empirical Evaluation of Ten Depth Cameras: Bias, Precision, Lateral Noise,
Different Lighting Conditions and Materials** — Halmetschlager-Funek, Suchi,
Kampel, Vincze. *IEEE Robotics & Automation Magazine*, 2019.
Search title on IEEE Xplore / TU Wien publication database.
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** partial
**Priority: MED.** Cross-sensor comparison including **material dependence**
(glossy, dark) and lighting — the empirical counterpart to our "depth dropout on
glossy/dark surfaces" given. Use to decide whether material must enter the score
or can be handled as a dropout mask.

### 12. `khoshelham-elberink-kinect-accuracy`
**Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications**
— Khoshelham, Elberink. *Sensors* 12(2):1437–1454, 2012.
<https://www.mdpi.com/1424-8220/12/2/1437>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** no · **Formula:** yes
**Priority: MED.** The other canonical stereo/structured-light error model —
random error growing quadratically with range. Pair with #6 for a defensible
`σ(range, angle)` used as a weight in the view score.

### 13. `fringe-projection-accuracy-vs-surface-normal`
**Measurement Accuracy of Fringe Projection Depending on Surface Normal
Direction**
<https://www.researchgate.net/publication/46142299_Measurement_Accuracy_of_Fringe_Projection_Depending_on_Surface_Normal_Direction>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** no · **Formula:** ?
**Priority: MED.** Optical-metrology framing of the identical claim for triangulation
sensors. Worth one abstract-level read to see whether it gives a slope-dependent
uncertainty curve we can cite alongside #6–#8.

### 14. `fringe-projection-sensor-planning-sheet-metal`
**Sensor Planning System for Fringe Projection Scanning of Sheet Metal Parts** —
*Measurement*, 2016.
<https://www.sciencedirect.com/science/article/abs/pii/S0263224116304377>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** no · **Formula:** yes
**Priority: MED-HIGH.** The one paper found that **predicts whether the sensor will
return data at all** at a candidate view, by simulating a Nayar/Ellenrieder
diffuse+specular reflectance model on the CAD surface. This is the principled
form of "avoid the view where the D405's projector reflects back into the
imager" — the mechanism behind our glossy dropout. Validated against a real
fringe scanner on a robot (16%/26% sim-vs-real area error — note the honesty).

### 15. `reflective-metal-defect-detection-review`
**A Review of Vision-Based Defect Detection Research for Highly Reflective Metal
Surfaces** — *ScienceDirect*, 2026.
<https://www.sciencedirect.com/science/article/pii/S2215098626002181>
**Type:** paper · **Space:** IMG · **Violates:** – · **B2:** no · **Formula:** no
**Priority: MED.** Recent, and it explicitly recommends **robot–sensor path
planning that actively avoids specular highlights to secure optimal viewpoints** —
i.e. a review-level endorsement that viewpoint quality on shiny parts is not
purely geometric. Cite for the "orthogonal may be the *worst* view on a mirror"
counter-argument the critic will raise.

---

## Tier 3 — inspection-specific viewpoint quality (closest applied neighbours)

### 16. `staderini-spatial-resolution-metric`
**Spatial Resolution Metric for Optimal Viewpoints Generation in Visual
Inspection Planning** — Staderini, Glück, Mecca, Schneider, Kugi. *ICVS 2023*,
LNCS 14253:197–207.
<https://link.springer.com/chapter/10.1007/978-3-031-44137-0_17>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Puts **defect size → required spatial resolution** into the
viewpoint generator via ray tracing and a sampling-density matrix. This is the
clearest existing answer to "how well does this view frame/resolve the region"
in an industrial inspection context, and it is stated as a per-viewpoint metric,
not just a coverage constraint.

### 17. `staderini-visual-quality-inspection-planning`
**Visual Quality Inspection Planning: A Model-Based Framework for Generating
Optimal and Feasible Inspection Poses** — Staderini, Glück, Schneider, Kugi.
*IROS 2024*, pp. 10799–10806.
<https://ieeexplore.ieee.org/document/10802791/>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** yes · **Formula:** ?
**Priority: HIGH.** Same group, next step: couples object geometry with **robot
kinematics feasibility** in the pose generator, and reports full coverage from a
small pose count across geometries and materials. Small-pose-count + feasibility
+ materials is our exact operating point; shared read with Branch 4.

### 18. `mosbach-feature-driven-viewpoint-placement`
**Feature-Driven Viewpoint Placement for Model-Based Surface Inspection** —
Mosbach, Gospodnetić, Rauhut, Hamann, Hagen. *Machine Vision and Applications*
32(1):8, 2021.
<https://web.cs.ucdavis.edu/~hamann/MosbachGospodneticRauhutHamannHagenMachineVisionAndAppplicationsPaperRevision2AsPublished10292020.pdf>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Free PDF. Places virtual cameras **directed at pivot points on
the surface** — the constructive counterpart to a scoring function, and precisely
the "derive the pose from the surface" analytical move Albert is asking for.
Extract the per-viewpoint acceptance criteria and how "relevance" is quantified.

### 19. `gospodnetic-viewpoint-placement-inspection-planning`
**Viewpoint Placement for Inspection Planning** — Gospodnetić, Mosbach, Rauhut,
Hagen. *Machine Vision and Applications* 33(1), 2022. Open access.
<https://link.springer.com/article/10.1007/s00138-021-01252-z>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Its stated contribution is that the field optimizes *selection*
while never evaluating the *generation* method — and it compares generation
schemes by achieved coverage. That comparison protocol is a direct benchmark
candidate, and "generate-and-use vs generate-and-test" is the framing our
analytical-vs-optimization split needs.

### 20. `defect-detectability-viewpoint-optimization-aiat25`
**End-to-end Viewpoint Optimization for Inspection based on Defect Detectability
Assessment** — *AIAT '25*, ACM.
<https://dl.acm.org/doi/full/10.1145/3787120.3787121>
**Type:** paper · **Space:** SIM/IMG · **Violates:** CAD · **B2:** yes · **Formula:** yes
**Priority: HIGH.** The most on-target single paper in the map: a **viewpoint
quality metric built from defect visibility, size and position under
illumination**, assembled into a *viewpoint quality map*, then optimized
**end-to-end and differentiably** over viewpoint parameters. It is simultaneously
RQ1's judge and the optimization track's climber, and it explicitly criticizes
geometric/perceptual quality metrics as the wrong objective for inspection.
Its ROI-sampling trick (randomize defect sets in the region, maximize expected
detectability) is a ready answer to "we do not know where the defect is".

### 21. `mavrinac-semiautomatic-triangulation-inspection`
**Semiautomatic Model-Based View Planning for Active Triangulation 3-D
Inspection Systems** — Mavrinac, Chen, Alarcon-Herrera. *IEEE/ASME Trans.
Mechatronics* 20(2):799–811, 2015.
<https://ieeexplore.ieee.org/document/6818412/>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** no · **Formula:** yes
**Priority: MED-HIGH.** A **bounded scalar performance metric** for a triangulation
sensor configuration, used as an optimization objective, with probabilistic
modelling of positioning error. "Bounded scalar" matters: our VLM-facing score
should be normalized and comparable across views and objects.

### 22. `mavrinac-chen-camera-coverage-survey`
**Modeling Coverage in Camera Networks: A Survey** — Mavrinac, Chen. *IJCV*
101(1):205–226, 2013.
<https://link.springer.com/article/10.1007/s11263-012-0587-7>
**Type:** paper · **Space:** PLY · **Violates:** FREE-CAM · **B2:** no · **Formula:** yes
**Priority: MED-HIGH.** Survey of **geometric coverage models** — how the field
formalizes "this camera covers this point *well*" as a continuous strength
(resolution, focus, incidence angle, direction) rather than a boolean. That
continuous coverage-strength function is the cleanest existing template for our
per-region view score.

### 23. `rl-view-planning-surface-inspection`
**High-Efficient View Planning for Surface Inspection Based on Parallel Deep
Reinforcement Learning** — *Advanced Engineering Informatics*, 2022.
<https://www.sciencedirect.com/science/article/abs/pii/S147403462200307X>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** yes · **Formula:** partial
**Priority: MED.** Read for its **visibility estimation** module and the shape of
its reward (image quality under occlusion and illumination) — the reward *is* a
view-quality score. The RL loop itself belongs to Branch 4.

### 24. `uav-hierarchical-rl-viewpoint-precision`
**Hierarchical Reinforcement Learning for Viewpoint Planning with Scalable
Precision in UAV Inspection** — *Drones* 9(5):352, 2025. Open access.
<https://www.mdpi.com/2504-446X/9/5/352>
**Type:** paper · **Space:** SIM · **Violates:** FREE-CAM · **B2:** yes · **Formula:** yes
**Priority: MED.** Worth it purely for the **explicit three-part visibility
predicate** (FOV containment, incidence-angle threshold vs. surface normal,
occlusion ray check) plus a perspective-projection resolution test that includes
the **1/cos θ effective-GSD correction for tilt**. That correction is the
mathematical link between "orthogonal" and "resolves the defect" — cheap to
implement, easy to defend.

---

## Tier 4 — NBV utility / information-gain score functions
*(score function only; the selection loop is Branch 4's)*

### 25. `wu-quality-driven-poisson-autoscanning`
**Quality-Driven Poisson-Guided Autoscanning** — Wu, Sun, Long, Huang, Cohen-Or,
Gong, Deussen, Chen. *ACM TOG (SIGGRAPH Asia)* 33(6):203, 2014.
<https://graphics.uni-konstanz.de/publikationen/Wu2014QualitydrivenPoisson/index.html>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Explicitly rejects "minimize number of scans" for
"**maximize scan quality**" — our exact reframing. Builds a **per-region confidence
map** from the Poisson field of the current cloud, then derives a *viewing vector
field* from it. A confidence-weighted per-region score computed directly on
points, with the view direction falling out of the field, is the closest
published analogue to what RQ1 asks for. Validated on a PR2 and a one-arm
industrial robot.

### 26. `border-gammell-see`
**The Surface Edge Explorer (SEE): A Measurement-Direct Approach to Next Best
View Planning** — Border, Gammell. *IJRR* 43(10):1506–1532, 2024.
<https://arxiv.org/abs/2207.13684> · project: <https://robotic-esp.com/papers/border_ijrr24>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Scores views **directly from raw measurements** — no voxel grid,
no mesh, using local measurement density and occlusion estimates. This is the
methodological permission slip for judging a view on our defective PLY without
first reconstructing a surface, and its critique of rigid data structures
(fidelity loss, coupling to post-processing) is one we should reuse.

### 27. `delmerico-volumetric-ig-comparison`
**A Comparison of Volumetric Information Gain Metrics for Active 3D Object
Reconstruction** — Delmerico, Isler, Sabzevari, Scaramuzza. *Autonomous Robots*
42(2):197–208, 2018.
<https://link.springer.com/article/10.1007/s10514-017-9634-0> · PDF: <http://rpg.ifi.uzh.ch/docs/AURO17_Delmerico.pdf>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** **yes (strong)** · **Formula:** yes/impl
**Priority: HIGH.** The field's only head-to-head **comparison of score functions**
under one protocol, with formulas for each VI variant and open code. Two uses:
(a) a menu of candidate utility terms, (b) a template for how our benchmark
should compare judges — including how they avoided rewarding the metric that
matches the evaluation.

### 28. `isler-information-gain-formulation`
**An Information Gain Formulation for Active Volumetric 3D Reconstruction** —
Isler, Sabzevari, Delmerico, Scaramuzza. *ICRA* 2016.
<https://github.com/uzh-rpg/rpg_ig_active_reconstruction>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** no · **Formula:** yes
**Priority: MED-HIGH.** The precursor with the base formulation — read only if #27
leaves the derivations opaque.

### 29. `rpg-ig-active-reconstruction-repo`
**uzh-rpg / rpg_ig_active_reconstruction** — reference implementation of #27/#28.
<https://github.com/uzh-rpg/rpg_ig_active_reconstruction>
**Type:** repo · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** impl
**Priority: MED.** GPLv3, ROS Indigo / Ubuntu 14.04 — almost certainly not
directly runnable in our stack, so treat it as a *formula reference*, not a
dependency. Note the staleness in the map so nobody plans around it.

### 30. `kriegel-next-best-scan`
**Efficient Next-Best-Scan Planning for Autonomous 3D Surface Reconstruction of
Unknown Objects** — Kriegel, Rink, Bodenmüller, et al. (DLR). *J. Real-Time Image
Processing* 10:611–631, 2015.
<https://www.robotic.dlr.de/fileadmin/robotic/krie_si/kriegel13a.pdf>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Free PDF. Its utility function **explicitly separates an
exploration term from a mesh-quality term**, and it ablates information-gain-only
against information-gain-plus-quality on a real industrial robot with a laser
striper. That ablation is the experiment we will want to imitate: does adding a
quality/orthogonality term actually change the chosen views?

### 31. `vasquez-gomez-nbv-positioning-error`
**Volumetric Next-Best-View Planning for 3D Object Reconstruction with
Positioning Error** — Vasquez-Gomez, Sucar, Murrieta-Cid, Lopez-Damian. *IJARS*
11(159), 2014.
<https://journals.sagepub.com/doi/10.5772/58759>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: MED-HIGH.** A utility function with named factors (new information,
positioning, sensing, registration/overlap) *and* an explicit treatment of
**sensor positioning error** — our mm-level pose error and 4 cm offset. Extract the
occlusion factor (ratio of "occplane" voxels per ray-traced image) and the
expected-utility-under-pose-uncertainty machinery.

### 32. `vasquez-gomez-view-state-planning-uncertainty`
**View/State Planning for Three-Dimensional Object Reconstruction Under
Uncertainty** — Vasquez-Gomez, Sucar, Murrieta-Cid. *Autonomous Robots*, 2015.
<https://www.cimat.mx/~murrieta/Papersonline/VaSuMu_AR17.pdf>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: MED.** Free PDF. **Expected** utility by sampling the control space
from the robot's error distribution — i.e. score a view by what the arm will
*actually* achieve, not the commanded pose. If our 4 cm offset survives
calibration, this is how it enters the judge.

### 33. `jiang-fisherrf`
**FisherRF: Active View Selection and Uncertainty Quantification for Radiance
Fields using Fisher Information** — Jiang, Lei, Daniilidis. arXiv:2311.17874.
<https://jiangwenpl.github.io/FisherRF/>
**Type:** paper · **Space:** IMG · **Violates:** – · **B2:** yes · **Formula:** yes/impl
**Priority: MED-HIGH.** A **differentiable, per-view scalar** obtained from one
backward pass, with no ground truth needed and pixel-wise uncertainty as a
by-product. Read for whether the Fisher-information formulation transfers to a
plain point-cloud/rasterization model — if it does, the optimization track gets a
gradient for free. Note the stated limitation (linear-rendering assumption,
Gauss-Newton over-approximation).

### 34. `chen-cover-coverage-fisher`
**Coverage Optimization for Camera View Selection (COVER)** — Chen, Dai, Adang,
Gao, Schwager. arXiv:2604.05259, 2026.
<https://arxiv.org/abs/2604.05259>
**Type:** paper · **Space:** IMG · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: MED.** A cheap, tractable approximation of Fisher information gain
framed as "cover geometry insufficiently observed by past cameras", avoiding
transmittance estimation, integrated into Nerfstudio. Relevant as the *cheap
surrogate* end of the score-function spectrum.

---

## Tier 5 — graphics best-view scoring (transferable math)

### 35. `bonaventura-viewpoint-selection-survey`
**A Survey of Viewpoint Selection Methods for Polygonal Models** — Bonaventura,
Feixas, Sbert, Chuang, Wallraven. *Entropy* 20(5):370, 2018. Open access.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC7512891/>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** **yes (strong)** · **Formula:** yes/impl
**Priority: HIGH.** Twenty-two viewpoint-quality measures with formulas, grouped
into visibility / information-theoretic / silhouette / depth / curvature-saliency
families, **all evaluated against one benchmark (#38)**, and a **publicly released
framework implementing all of them**. This is the single highest formula-per-page
document in the branch and its comparison design is a benchmark template.

### 36. `vazquez-viewpoint-entropy`
**Viewpoint Selection Using Viewpoint Entropy** — Vázquez, Feixas, Sbert,
Heidrich. *VMV* 2001, pp. 273–280; see also the image-based-modeling application
paper.
<https://vccimaging.org/Publications/Vazquez2003AVS/Vazquez2003AVS.pdf>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** no · **Formula:** yes
**Priority: MED-HIGH.** The founding formula (entropy over projected-area
distribution of faces) and, critically, its **known failure mode**: the measure is
discretization-dependent, so two tessellations of the same shape give different
best views. On a noisy PLY that failure is not hypothetical — it is the reason
raw viewpoint entropy is probably wrong for us, and we should say so with a
citation.

### 37. `secord-perceptual-viewpoint-preference`
**Perceptual Models of Viewpoint Preference** — Secord, Lu, Finkelstein, Singh,
Nealen. *ACM TOG* 30(5):109, 2011.
<https://dl.acm.org/doi/10.1145/2019627.2019628>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** yes · **Formula:** yes
**Priority: MED-HIGH.** Fits a **linear combination of view attributes to human
preference** from a large user study, and reports the correlation structure of
the attributes (three clusters: area-like, silhouette-curvature, surface-curvature).
Two payoffs: a recipe for *learning* weights when no analytic ground truth
exists, and evidence about which attributes are redundant — we should not sum
five correlated terms and call it a score.

### 38. `dutagaci-best-view-benchmark`
**A Benchmark for Best View Selection of 3D Objects** — Dutagaci, Cheung, Godil.
*3DOR* 2010, pp. 45–50. Dataset: NIST, DOI 10.18434/M32209.
<https://data.nist.gov/pdr/lps/ark:/88434/mds2-2209>
**Type:** paper + dataset · **Space:** SIM · **Violates:** CAD · **B2:** **yes (strong)** · **Formula:** yes
**Priority: MED-HIGH.** 68 models × 26 human subjects' preferred views, plus a
published *view-selection-error* measure and released code. The one concrete
precedent for "human-labelled ground truth for the best view" — a Branch 2 must-see,
and a caution: it measures *representativeness for a human*, not *measurability
for a sensor*. Distinguishing those two notions is a load-bearing move in our review.

### 39. `lee-mesh-saliency`
**Mesh Saliency** — Lee, Varshney, Jacobs. *ACM SIGGRAPH* 2005, pp. 659–666.
<https://dl.acm.org/doi/10.1145/1186822.1073244>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** no · **Formula:** yes
**Priority: MED.** Center-surround on mean curvature; the standard "importance
weight" plugged into viewpoint scores (#35, #37). Relevant only as a *region
importance* term — plausibly the wrong prior for defect inspection, where
importance is the operator's question, not curvature. Read to argue that
explicitly.

### 40. `feixas-unified-viewpoint-saliency-framework`
**A Unified Information-Theoretic Framework for Viewpoint Selection and Mesh
Saliency** — Feixas, Sbert, González. *ACM TAP*, 2009.
<https://dl.acm.org/doi/10.1145/1462055.1462056>
**Type:** paper · **Space:** SIM · **Violates:** CAD · **B2:** no · **Formula:** yes
**Priority: MED.** Viewpoint mutual information over a viewpoint↔polygon channel,
and — the reason it is here — it treats **viewpoint stability and object
exploration inside the same formalism**. "Stability" is the formal cousin of RQ1's
"best within its neighbouring region"; check whether their stability measure is
computable for us.

---

## Tier 6 — local viewpoint optimization (the climbing machinery)

### 41. `burusa-gradient-local-nbv-plant-nodes`
**Gradient-Based Local Next-Best-View Planning for Improved Perception of
Targeted Plant Nodes** — arXiv:2311.16759.
<https://arxiv.org/abs/2311.16759>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: HIGH.** Exactly RQ1's second half: **estimate the local gradient
direction of view quality via differential ray sampling** and move the eye-in-hand
camera up it, for a *single targeted region* under occlusion. 10× faster than
sampling-based, 28% more efficient trajectories, on a robot arm. If any single
paper prototypes our optimization track, it is this one.

### 42. `active-implicit-uncertainty-nbv-optimization`
**Active Implicit Object Reconstruction Using Uncertainty-Guided Next-Best-View
Optimization** — arXiv:2303.16739.
<https://arxiv.org/abs/2303.16739>
**Type:** paper · **Space:** IMG/SIM · **Violates:** CONT · **B2:** yes · **Formula:** yes
**Priority: MED-HIGH.** Optimizes the view **on a continuous manifold** by
exploiting differentiability of the implicit representation, with information
gain as sampled entropy of the occupancy field — no auxiliary uncertainty map, no
learned module. Read for how they keep the utility landscape smooth despite
visibility discontinuities; that smoothing problem is the main technical risk of
the optimization track.

### 43. `viewactive-viewpoint-quality-field`
**ViewActive: Active Viewpoint Optimization from a Single Image** — Wu, Lin, He,
Fermüller, Aloimonos. arXiv:2409.09997 (v5, Jul 2025). Code released.
<https://arxiv.org/abs/2409.09997>
**Type:** paper + repo · **Space:** IMG · **Violates:** – · **B2:** yes · **Formula:** yes/impl
**Priority: HIGH.** Defines a **3D Viewpoint Quality Field** from three
general-purpose, *computable* metrics — **self-occlusion ratio,
occupancy-aware surface normal entropy, visual entropy** — then learns to predict
it from a single 2D image at 72 FPS. Two distinct values to us: (a) the three
metrics are directly implementable on our PLY as ground truth, (b) it is the
strongest existing evidence that view quality can be judged from the *captured
image* alone, which is the fork the contract flags (image-space judging → the
deferred generative-view question).

### 44. `oa-nbv-occlusion-aware`
**OA-NBV: Occlusion-Aware Next-Best-View Planning for Human-Centered Active
Perception on Mobile Robots** — arXiv:2603.11072, 2026.
<https://arxiv.org/abs/2603.11072>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** yes · **Formula:** yes
**Priority: MED.** A **target-centric** visibility score (occlusion, target scale,
target completeness) restricted to **kinematically feasible robot poses**, with
quantified gains (≥81% normalized target area, ≥58% keypoint visibility). Target-
centric + feasibility-filtered is our structure; the metric "normalized target
area" is a plausible benchmark metric for framing.

### 45. `graspview-render-and-score`
**GraspView: Active Perception Scoring and Best-View Optimization for Robotic
Grasping in Cluttered Environments** — Wang et al. arXiv:2511.04199.
<https://arxiv.org/abs/2511.04199>
**Type:** paper · **Space:** IMG · **Violates:** – · **B2:** yes · **Formula:** ?
**Priority: MED.** **Render-and-score**: render each candidate view off-screen from
the *current colored point cloud*, score the rendered image, pick the best. That
loop — render from the survey cloud, judge in image space, no new capture needed
— is a concrete architecture for our judge that sidesteps the generative
world-model question entirely. Extract exactly what does the scoring and how the
render artefacts (holes, sparsity) are handled. Cross-tag Branch 6.

### 46. `one-shot-multimodal-active-perception`
**A General One-Shot Multimodal Active Perception Framework for Robotic
Manipulation: Learning to Predict Optimal Viewpoint** — Qin, Liu, Luo, Liang,
Fang. arXiv:2601.13639, Jan 2026.
<https://arxiv.org/abs/2601.13639>
**Type:** paper · **Space:** IMG · **Violates:** – · **B2:** yes · **Formula:** no
**Priority: MED.** Cite mainly for its framing claim — **there is no unified
definition of an optimal viewpoint**, with existing theories evaluating views by
visible area, silhouette, depth, curvature, semantics, or aesthetics. That is the
review's thesis sentence, sourced from a 2026 paper rather than asserted.

---

## Tier 7 — image-space judging of the captured view

### 47. `qalign-lmm-visual-scoring`
**Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels** —
ICML 2024. arXiv:2312.17090.
<https://arxiv.org/abs/2312.17090>
**Type:** paper · **Space:** IMG · **Violates:** – · **B2:** yes · **Formula:** impl
**Priority: MED.** The standard method for turning a VLM into a calibrated image
*quality* rater via discrete text levels. Directly usable as a baseline judge in
the benchmark, and it is the natural thing our existing VLM orchestrator could do
today — so it must be beaten, or adopted. Cross-tag Branch 6.

### 48. `blur-nriqa-classification-review`
**Blur-Specific No-Reference Image Quality Assessment: A Classification and
Review of Representative Methods**
<https://www.researchgate.net/publication/327752994>
**Type:** paper · **Space:** IMG · **Violates:** – · **B2:** no · **Formula:** yes
**Priority: MED.** One survey to cover the whole family; from it, take JNB
(Ferzli & Karam, TIP 2009) and CPBD (Narvekar & Karam, TIP 2011) as the two
concrete sharpness metrics to implement. Sharpness/exposure is the cheap
image-space gate the contract explicitly permits ("or the captured image
itself"), and a cheap gate we can ship is worth more than a perfect one we cannot.

### 49. `focus-measure-operators-sml`
**Sum-Modified-Laplacian and energy-based focus measures** (shape-from-focus /
autofocus literature; robust energy-based focus measure for low-contrast images)
Entry point via #48's reference list and the SEM-blurriness paper:
<https://pmc.ncbi.nlm.nih.gov/articles/PMC6589194/>
**Type:** paper · **Space:** IMG · **Violates:** – · **B2:** no · **Formula:** yes
**Priority: LOW-MED.** Acquisition-side focus measures, cheaper and better-behaved
than perceptual IQA for a fixed camera. Relevant if the D405 near-field depth of
field turns out to bind at our minimum shell radius.

---

## Tier 8 — implementation primitives (make the score computable on a PLY)

### 50. `katz-tal-basri-hidden-point-removal`
**Direct Visibility of Point Sets** — Katz, Tal, Basri. *ACM TOG* 26(3), SIGGRAPH
2007.
<https://dl.acm.org/doi/10.1145/1276377.1276407>
**Type:** paper · **Space:** PLY · **Violates:** – · **B2:** no · **Formula:** yes
**Priority: HIGH.** Determines which points of a cloud are visible from a
viewpoint **without reconstructing a surface or estimating normals**, in O(n log n),
via spherical flipping + convex hull. This is the enabling primitive for
self-occlusion and visibility terms computed straight on our defective PLY — it
removes the "we need a mesh first" dependency that would otherwise chain RQ1 to
Branch 3. Note the abstract's own warning: visibility is *definitional* and
depends on sampling density, which is exactly where our dropout will bite.

### 51. `open3d-hidden-point-removal-docs`
**Open3D — `hidden_point_removal` (point cloud tutorial)**
<https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html>
**Type:** docs · **Space:** PLY · **Violates:** – · **B2:** no · **Formula:** impl
**Priority: MED-HIGH.** #50 already implemented in the library our stack uses.
Check the radius-parameter semantics and behaviour on non-uniform density — the
parameter is the practical hazard.

### 52. `scott-model-based-view-planning`
**Model-Based View Planning** — Scott. *Machine Vision and Applications*
20(1):47–69, 2009.
<https://link.springer.com/article/10.1007/s00138-007-0110-2>
**Type:** paper · **Space:** PLY · **Violates:** CAD · **B2:** **yes (strong)** · **Formula:** yes
**Priority: HIGH.** Defines **quality objectives and performance measures** for
high-fidelity inspection, models camera *and positioning-system* performance
statistically, and extends the measurability matrix (Tarbox & Gottschlich) with
**shadow effect, measurement precision and incident angle** — the three factors
most papers drop. Also proposes an "optimal scan zone" generated by **offsetting
from the sampled surface patch** rather than from a view sphere — precedent for
abandoning our uniform sphere in favour of surface-anchored views. Arguably the
best single formal treatment in the branch for our specific setting.

### 53. `olague-mohr-optimal-camera-placement`
**Optimal Camera Placement for Accurate Reconstruction** — Olague, Mohr. *Pattern
Recognition* 35(4):927–944, 2002.
<https://www.sciencedirect.com/science/article/abs/pii/S0031320301000760>
**Type:** paper · **Space:** PLY · **Violates:** FREE-CAM · **B2:** no · **Formula:** yes
**Priority: MED.** Photogrammetric network design done properly: propagate
detection uncertainty into a **reconstruction covariance matrix** and derive the
criterion from it. Read for the error-propagation criterion, and for the honest
critique attached to it (geometry only; ignores texture/appearance) which is a
warning for any purely geometric judge of ours.

---

## Explicitly not in this map (and why)

- **Aspect graphs, view-space tessellation, addressing schemes** → Branch 5.
- **Set-cover / greedy selection over candidate views, reachability maps, IK
  filtering as a method** → Branch 4. (Papers here that *mention* feasibility —
  #17, #20, #44 — are cross-tagged, but we extract only the per-view score.)
- **Benchmark datasets and protocols as such** → Branch 2. All `B2: yes` entries
  above are the handoff list.
- **Generative prediction of unseen views** (Cosmos-style) → deferred by the
  plan. #45 (`graspview-render-and-score`) is the deliberately non-generative
  alternative and should be evaluated before that door is reopened.
- **Learned end-to-end view policies** → deferred; #23, #24, #43 are read for
  their *scores/rewards*, not their policies.
- **Primitive fitting and normal estimation** → Branch 3. This branch *consumes*
  normals and must state its tolerance to normal error (see Q7).

---

## Questions this branch must answer

Tasks for the reading threads and for my own round-3 synthesis. Grouped; each
should end as a defensible paragraph with citations, or an explicit "the
literature does not answer this".

### A. The orthogonality claim (grounding Albert's requirement)

1. **What is the measured relationship between incidence angle and depth/measurement
   error for an active-stereo sensor like the D405 at 10–40 cm?** Produce a
   citable curve or closed form `σ(range, θ)`, with the sources' sensor and range
   stated. (#6, #7, #8, #11, #12, #9)
2. **Is there a threshold effect or a smooth penalty?** Does quality degrade
   gracefully with θ, or is there a knee (commonly cited near 60°) past which the
   sensor fails outright? This decides whether our score is a soft weight or a
   hard feasibility constraint — and whether "most orthogonal" is even the right
   objective versus "within tolerance and reachable".
3. **Does orthogonality ever *hurt*?** On glossy/dark surfaces, the specular lobe at
   normal incidence can reflect the projector straight back into the imager.
   Find the conditions under which the best view is deliberately off-normal, and
   whether any planner models this (reflectance-aware planning). (#14, #15, #11)
4. **Independently of depth error: does defect detectability actually peak at
   normal incidence?** Foreshortening costs resolution as 1/cos θ, but many
   surface defects (scratches, dents, coating flaws) are *more* visible at grazing
   angles or under raking light. Get the industrial-inspection answer, not the
   reconstruction answer. (#20, #15, #23, #24)

### B. The score function itself (RQ1 core)

5. **Enumerate every published view-quality term with a computable definition**,
   tagged PLY / IMG, with the formula and its inputs. Deliverable: one table.
   Minimum coverage — incidence/cosine, standoff/GSD, FOV framing/containment,
   depth of field, self-occlusion/visibility, sampling density, information gain,
   viewpoint entropy, surface-normal entropy, defect detectability. (#35, #2,
   #52, #22, #43, #27, #16, #20)
6. **Which terms are redundant?** Secord (#37) reports the attribute correlation
   clusters; check whether the terms we intend to combine are measuring the same
   thing. A score that sums five correlated terms is one term with extra steps.
7. **How much does the score degrade under our normal-estimation error?** An
   orthogonality term is a function of the estimated normal. Establish the
   sensitivity `d(score)/d(normal error)` and the normal error our clouds
   actually have — the second half is Branch 3's number, this branch must state
   the tolerance it needs. **This is the branch's top technical risk.**
8. **Is there precedent for a *bounded, normalized* per-view score comparable
   across views and across objects?** The VLM will eventually see these numbers
   (RQ4); an unbounded utility is unusable as an interface. (#21, #44)
9. **What is the unit of scoring — a point, a patch, a face, the whole view?**
   RQ1 says "a surface region". Collect how papers aggregate per-point quality to
   per-region and per-view (mean? min? area-weighted? propagated precision?), and
   whether min-pooling (worst point) or mean-pooling is defensible. (#8, #25, #52)

### C. Computability on *our* input

10. **Which of these scores survive a defective fused PLY** — no mesh, mm-level
    pose error, dropout holes, ~4 cm fused-vs-single-view offset? Explicitly
    partition the Q5 table into "computable today on our PLY", "needs Branch 3
    structure first", "needs a mesh/CAD we do not have".
11. **How is visibility/self-occlusion computed without a surface?** Validate the
    HPR route (#50, #51): parameter sensitivity, behaviour on non-uniform density,
    and what dropout holes do to it (do holes read as "visible through"?).
12. **Does any method score views against a cloud that is itself wrong?** Almost
    all NBV work assumes the accumulated map is the best current estimate. Find
    anyone who propagates *input geometry uncertainty* into the view score
    (#31, #32, #52, #53) — and if nobody does, that gap is a contribution claim.
13. **What does the 4 cm fused-vs-single-view offset do to a geometric judge?**
    If the judge reasons in fused-cloud coordinates and the camera reports depth
    in its own frame, an "orthogonal" pose may be orthogonal to the wrong surface.
    Is this calibration (fixable) or a modelling requirement (must enter the score)?

### D. Local optimality (the second half of RQ1)

14. **What formal definitions of "best view within a neighbourhood" exist?**
    Candidates seen: MVP's constraint-satisfaction-with-margin (#3), viewpoint
    stability in the mutual-information framework (#40), gradient magnitude of
    the utility field (#41, #42). Compare and pick one we can compute.
15. **Is the view-quality landscape smooth enough to climb?** Visibility is
    discontinuous in camera pose; the gradient literature smooths it (fuzzy
    filters, occupancy fields, differential ray sampling, alpha compositing).
    Catalogue the smoothing tricks and their cost. (#41, #42, #33, #20)
16. **Sampling vs. gradient: what does the field report about which wins for a
    small local refinement?** #41 claims 10× compute and 28% trajectory
    efficiency for gradients; is that reproduced elsewhere, and does it hold when
    the neighbourhood is small and already IK-constrained?
17. **How does local optimization interact with a discrete action space?** Our
    contract fixes discrete selection. Determine whether local optimization
    belongs *offline* (refine each candidate view once, at action-space
    construction time) or *online* (the VLM asks for "better"). This is a design
    fork the literature will not decide for us, but should inform.

### E. Image-space judging (the fork the contract flags)

18. **Can view quality be judged from the captured image alone, and how well?**
    #43 is the strongest evidence yes (single image → viewpoint quality field).
    Establish the accuracy and the object-class generalization, since our objects
    are unglamorous primitives, not ShapeNet categories.
19. **What can be judged in image space that PLY space cannot see?** Candidate
    answers: specular blowout, exposure, motion blur, focus, actual defect
    contrast. If the answer is "the things that matter for defect detection",
    that materially strengthens the image-space track — and only then does the
    deferred generative-unseen-view question deserve reopening.
20. **Is render-and-score from the existing cloud a viable middle path?** (#45)
    What do render artefacts (sparsity, holes, no BRDF) do to the score, and does
    the ranking of views survive them?
21. **How good is an off-the-shelf VLM as a view judge, right now?** #47 gives a
    method and a baseline. This is the "do nothing clever" control our benchmark
    needs; if it wins, the project's shape changes.

### F. Feeding the benchmark (Branch 2 handoff)

22. **What is ground truth for "the best view" of a box face or a cylinder?**
    Collect the three families the plan demands: analytical (face normal through
    the region centroid at optimal standoff), metric-based (argmax of a chosen
    score — note the circularity), human-labelled (#38's protocol). State the
    circularity risk explicitly for each.
23. **Which evaluations here can be replicated on primitives?** Tag each `B2: yes`
    entry with its object set and whether primitives were included; most of this
    literature evaluates on Stanford models or free-form industrial parts, and
    our regime (boxes, cylinders) may be *easier* in a way that makes published
    baselines uninformative.
24. **What ablation shows a quality term earns its place?** #30 ablates
    information-gain vs information-gain+quality; #27 compares score functions
    head-to-head. Extract both protocols verbatim — they are the two experiments
    our benchmark most needs to imitate.

---

## Search trail

Tools: `WebSearch` (200-call session budget, exhausted), `WebFetch` for
abstract/landing-page verification of the arXiv entries. Semantic Scholar and
arXiv APIs were attempted and returned HTTP 429 / were unreachable from the
sandbox — all bibliographic detail below therefore comes from search results,
publisher landing pages, and fetched arXiv abstracts.

Queries run, in order:

1. `Tarabanis survey of sensor planning in computer vision generalized viewpoint MVP` → #1–#4 lineage, incl. the CVGIP feature-detectability paper and "Computing occlusion-free viewpoints".
2. `viewpoint quality metric survey viewpoint entropy best view selection polygonal models` → #35, #36, #37, #39; the 22-measure taxonomy.
3. `comparison of volumetric information gain metrics active 3D object reconstruction Delmerico` → #27, #28, #29.
4. `RGB-D depth error vs incidence angle surface characterization RealSense accuracy model` → mostly off-axis-FoV studies; the search itself flagged that incidence-angle models live in the Nguyen/Kinect lineage → redirected to query 5.
5. `Nguyen Izadi Lovell modeling Kinect sensor noise axial lateral noise angle 3DIMPVT 2012` → #6, #12.
6. `terrestrial laser scanning incidence angle influence point quality Soudarissanane scanning geometry` → #7, #8.
7. `view planning automated 3D object reconstruction and inspection survey Scott Roth Rivest` → #4, and the Scott follow-ups incl. pose-error work.
8. `FisherRF Fisher information next best view Gaussian splatting uncertainty quantification` → #33, plus pointers to Next Best Sense, GauSS-MI, Bayes' Rays.
9. `viewpoint planning surface defect detection detectability viewing angle industrial inspection` → #20, #18, #19, #16, #23, #15 — the richest single query.
10. `"view planning" robot active vision survey systems algorithms applications Zeng 2020` → #5.
11. `Mosbach Gospodnetić viewpoint placement inspection planning Machine Vision Applications 2021` → #18, #19, #16.
12. `Dutagaci Cheung Godil best view selection benchmark human generated ground truth 3D models` → #38 + NIST dataset DOI.
13. `continuous next best view optimization gradient differentiable camera pose refinement view planning` → #41, #42, #34; also surfaced the visibility-discontinuity/smoothing caveat behind Q15.
14. `Mavrinac Chen modeling coverage in camera networks survey task-specific coverage strength` → #22, #21.
15. `Kriegel efficient next-best-scan planning autonomous 3D surface reconstruction surface quality estimation` → #30, #26.
16. `photogrammetric network design camera placement precision Olague Mohr` → #53.
17. `fringe projection structured light measurement uncertainty incidence angle shiny specular surface metrology` → #13, plus the FPP-vs-deflectometry split and VDI/VDE 2634 as a standards lead.
18. `Intel RealSense D405 depth quality accuracy short range whitepaper tuning` → #9, #10.
19. `Secord perceptual models of viewpoint preference mesh saliency Lee Varshney` → #37, #39, #40.
20. `Surface Edge Explorer SEE measurement-direct next best view planning Border Gammell` → #26.
21. `"Spatial Resolution Metric for Optimal Viewpoints Generation in Visual Inspection Planning" Staderini` → #16, #17.
22. `arXiv 2025 2026 viewpoint quality scoring active perception robot inspection orthogonal view surface normal metric` → #43, #44, #45, #46.
23. `Mavrinac Chen Alarcon-Herrera semiautomatic model-based view planning active triangulation` → #21.
24. `Vasquez-Gomez view planning positioning error hierarchical ray tracing utility function` → #31, #32.
25. `no-reference image quality assessment blur sharpness metric automated optical inspection` → #48, #49.
26. `Staderini "Visual Quality Inspection Planning" model-based framework optimal feasible inspection poses` → #17.
27. `"Quality-driven Poisson-guided autoscanning" Wu confidence scan quality next best view` → #25.
28. `view planning avoid specular reflection glossy object 3D scanning BRDF reflectance-aware viewpoint selection` → #14.
29. `"End-to-end Viewpoint Optimization for Inspection based on Defect Detectability Assessment" 2025` → #20.
30. `Scott "Model-based view planning" measurability matrix sampling density` → #52, plus Tarbox & Gottschlich as the measurability-matrix origin.
31. `Katz Tal Basri direct visibility of point sets hidden point removal operator` → #50, #51.
32. `ground sample distance viewpoint generation UAV infrastructure inspection incidence angle constraint` → #24, and the 1/cos θ effective-GSD relation.
33. `vision language model as judge image quality viewpoint selection scoring captured view robotics 2025` → #47; the search explicitly reported no paper using a VLM judge to score *viewpoint* quality for NBV — recorded as a gap.

Fetched for verification: arXiv 2409.09997 (#43), 2601.13639 (#46), 2311.16759
(#41), 2511.04199 (#45), 2604.05259 (#34).

### Known gaps in this map

- **No 2023–2026 NBV/active-vision survey** was located before the search budget
  ran out; #5 (2020) is the most recent spine. Worth one query in the next round.
- **Joint viewpoint + illumination planning** was not searched (budget). #20 lists
  it as its own future work, and for a D405 with an active projector the coupling
  is real. Flag for round 3.
- **Tarbox & Gottschlich (1995)** (measurability matrix origin) is cited second-hand
  via #52 and was not independently verified.
- **Bonaventura's released framework** implementing the 22 measures is referenced
  in #35 but its repository URL was not captured — retrieve from the paper.
- **#11** (ten depth cameras) is cited from prior knowledge, not from a search
  result; verify the exact venue/year before it enters the review.
