# Branch 3 — Extracting exploitable geometry from defective survey clouds
## Round 2: document map

**Date:** 2026-08-27
**Branch slug:** `primitive-structure-from-survey-clouds`
**Feeds:** RQ2 (shape → views), RQ1 (trustworthy normals for orthogonality scoring),
RQ3 (a face/axis decomposition is the natural skeleton of the action space); P1.
**Plan:** [`../round1_plan.md`](../round1_plan.md) — Branch 3
**Contract:** [`../../../../2026-08-27-geometry-aware-view-planning.md`](../../../../2026-08-27-geometry-aware-view-planning.md)

**Standing critic question for this branch:** *does decomposition survive our defect
profile?* Almost every candidate below evaluates on clean synthetic data (ShapeNet,
ABC, Fit4CAD sampled CAD). The mapping bias here is deliberate: candidates with
**real-scan evidence** are ranked up even when the method is weaker, and every
synthetic-only candidate carries an explicit constraint-violation tag.

---

## How to read the table

Each entry carries, where known:

- **Prim:** primitives covered — `pl` plane, `cy` cylinder, `sp` sphere, `co` cone,
  `to` torus, `sq` superquadric, `cub` cuboid, `bspl` B-spline patch.
- **Real:** does the paper show results on *real scanned* data (not just synthetic)?
- **Code:** off-the-shelf availability.
- **⚠:** constraint violated relative to our givens (clean-mesh assumption, canonical
  pose, CAD input, training data we don't have).
- **[B*]:** cross-branch tag — B1 view-quality scoring, B2 benchmark, B4 view-set
  selection, B5 viewpoint-space structure.

---

## A. Spine — surveys, benchmarks, and the default detector

| # | Slug | Doc | Why it matters | Prio |
|---|------|-----|----------------|------|
| 1 | `kaiser-primitive-detection-survey` | Kaiser, Ybáñez Zepeda, Boubekeur — *A Survey of Simple Geometric Primitives Detection Methods for Captured 3D Data*, CGF 38(1):167–196, 2019. [doi:10.1111/cgf.13451](https://onlinelibrary.wiley.com/doi/10.1111/cgf.13451) | The one survey written explicitly for **captured, corrupted** 3D data rather than clean meshes; organizes the field application-first and ends with evaluation guidance. This is the branch's spine — read before anything else. Prim: pl/cy/sp/co/to. Real: yes (survey scope). | **high** |
| 2 | `schnabel-efficient-ransac` | Schnabel, Wahl, Klein — *Efficient RANSAC for Point-Cloud Shape Detection*, CGF 26(2):214–226, 2007. [landing](https://cg.cs.uni-bonn.de/publication/schnabel-2007-efficient) | The default primitive detector we would actually run first, and the baseline every later paper compares to. Extract: the **probabilistic stopping criterion** (probability of missing the largest shape) — a ready-made confidence signal for the fallback trigger. Prim: pl/cy/sp/co/to. Code: CGAL, research-license original. ⚠ known noise sensitivity in *normals* of the minimal set, and per-input parameter tuning. | **high** |
| 3 | `shrec2022-primitive-fitting-track` | Romanengo et al. — *SHREC 2022: Fitting and recognition of simple geometric primitives on point clouds*, Computers & Graphics, 2022. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0097849322001224) | Six methods compared head-to-head under one protocol, including type recognition (not just parameter fitting). The fastest way to answer "which fitter do we pick" without reading six papers. [B2] | **high** |
| 4 | `fit4cad-benchmark` | Romanengo, Raffo, Qie, Anwer, Falcidieno — *Fit4CAD: A point cloud benchmark for fitting simple geometric primitives in CAD objects*, C&G 102:133–143, 2022. [arXiv:2105.06858](https://arxiv.org/abs/2105.06858) | 225 clouds with per-primitive ground truth (implicit **and** parametric form) plus a defined metric set. Its metrics are the most directly reusable material in this branch for our own benchmark. ⚠ sampled from CAD, i.e. *no* registration error / dropout — the corruption is ours to add. [B2] | **high** |
| 5 | `araujo-oliveira-robust-plane-detection` | Araújo & Oliveira — *A robust statistics approach for plane detection in unorganized point clouds*, Pattern Recognition 100:107115, 2020. [PDF](https://www.inf.ufrgs.br/~oliveira/pubs_files/RE/Araujo_Oliveira_plane_detection_low_res.pdf) | Deterministic, near parameter-free, and — critically for us — returns **delimited planar patches**, not infinite planes. A box face needs an extent and a centroid, which is exactly what this outputs. Shipped in Open3D as `detect_planar_patches`; standalone impl at [plusk01/pointcloud-plane-segmentation](https://github.com/plusk01/pointcloud-plane-segmentation) (~120 ms / 50k points). Prim: pl. Real: yes (noisy clouds). | **high** |
| 6 | `cgal-shape-detection-docs` | CGAL 6.x — *Shape Detection* user manual. [doc.cgal.org](https://doc.cgal.org/latest/Shape_detection/index.html) | The implementation we would integrate, and the precise semantics of the parameters that decide whether it works on our cloud: `epsilon`, `cluster_epsilon`, normal-deviation threshold, min region size. Also documents the custom-primitive interface. Prim: pl/cy/sp/co/to. | **high** |
| 7 | `open3d-geometry-api-docs` | Open3D — `PointCloud` API + geometry tutorials (`segment_plane`, `detect_planar_patches`, normal estimation, statistical/radius outlier removal). [python_api](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html) · [tutorials](https://www.open3d.org/docs/release/tutorial/geometry/index.html) | Establishes exactly what we get for free and what we don't: planes and planar patches yes, **cylinders no** (open feature request [#6602](https://github.com/isl-org/Open3D/issues/6602)). Since the cylinder-orbit behaviour is half the contract, this is a hard integration fact, not trivia. | **high** |

---

## B. Robust detection under noise: RANSAC descendants, voting, global selection

| # | Slug | Doc | Why it matters | Prio |
|---|------|-----|----------------|------|
| 8 | `raffo-localized-voting-primitives` | Raffo et al. — *Fitting and recognition of geometric primitives in segmented 3D point clouds using a localized voting procedure*, CAGD, 2022. [arXiv:2205.15426](https://arxiv.org/abs/2205.15426) | The Hough/voting alternative to RANSAC, motivated by exactly our failure: RANSAC "can be easily deceived in the presence of point cloud artifacts". Extract the type-recognition confidence, which RANSAC does not give cleanly. Prim: pl/cy/sp/co/to. | med |
| 9 | `barath-multi-x-multiclass-fitting` | Barath & Matas — *Multi-Class Model Fitting by Energy Minimization and Mode-Seeking (Multi-X)*, ECCV 2018. [arXiv:1706.00827](https://arxiv.org/abs/1706.00827) | Fits **planes and cylinders simultaneously** under a global energy — a principled answer to "which primitive type is here, and how many", instead of greedy sequential extraction that leaves phantom shapes. Evaluated on annotated LiDAR (signs=planes, balusters=cylinders). Real: yes. | med |
| 10 | `isack-boykov-pearl` | Isack & Boykov — *Energy-Based Geometric Multi-Model Fitting (PEARL)*, IJCV 97(2), 2012. [PDF via ResearchGate](https://www.researchgate.net/publication/220660141_Energy-Based_Geometric_Multi-model_Fitting) | The formulation Multi-X repairs: fitting error + spatial-coherence + **model-complexity (label cost)**. The label cost is the closest thing in the classical literature to an automatic "how many primitives does this object have" answer, which our decomposition needs. Read for the formulation, not the results. | med |
| 11 | `montebox-finder` | Ramamonjisoa, Stekovic, Lepetit — *MonteBoxFinder: Detecting and Filtering Primitives to Fit a Noisy Point Cloud*, ECCV 2022. [arXiv:2207.14268](https://arxiv.org/abs/2207.14268) | The only candidate whose *entire contribution* is filtering good primitives from noisy ones, evaluated on real ScanNet scans. Its stated premise — that RANSAC + relationship refinement "works only on very clean scans" — is the branch's central risk, stated by someone who measured it. Prim: cub. Real: yes. | med |
| 12 | `gnc-robust-spatial-perception` | Yang, Antonante, Tzoumas, Carlone — *Graduated Non-Convexity for Robust Spatial Perception*, RA-L 5(2), 2020. [arXiv:1909.08605](https://arxiv.org/abs/1909.08605) | Outlier-robust fitting with **no initial guess** and, in the wider TEASER line, a *certificate* of solution quality. Both halves matter: a fitter that tolerates dropout-induced structure, and a computable confidence to gate the fallback. | med |

---

## C. Shape abstraction — superquadrics, cuboids, learned decomposition

| # | Slug | Doc | Why it matters | Prio |
|---|------|-----|----------------|------|
| 13 | `ems-probabilistic-superquadric-recovery` | Liu, Wu, Ruan, Chirikjian — *Robust and Accurate Superquadric Recovery: a Probabilistic Approach (EMS)*, CVPR 2022. [arXiv:2111.14517](https://arxiv.org/abs/2111.14517) · [code](https://github.com/ChirikjianLab/EMS-superquadric_fitting) | Gaussian-uniform mixture on the superquadric surface → MLE, with **explicit outlier modelling** and three separate robustness ablations: partial data, outliers, noise. Yields a posterior outlier probability per point — a directly usable fit-quality signal. Also extended to hierarchies of superquadrics. Prim: sq. Code: yes. | **high** |
| 14 | `superflex-deformable-superquadrics` | Tavernini, Fedele, Novello, Guibas, Pollefeys, Engelmann — *SuperFlex: Deformable Superquadrics for Point Cloud Decomposition*, 2026. [arXiv:2607.01015](https://arxiv.org/abs/2607.01015) | The closest published match to our input profile: explicitly trained for robustness to **partial real-world point clouds**, and it names SuperDec's limitations (Chamfer-only supervision leaves gaps; rigid convex symmetric primitives; trained on complete clouds). Adds bending/tapering. Prim: deformable sq. Real: claimed. | **high** |
| 15 | `superdec-superquadric-decomposition` | Fedele, Sun, Guibas, Pollefeys, Engelmann — *SuperDec: 3D Scene Decomposition with Superquadric Primitives*, ICCV 2025. [arXiv:2504.00992](https://arxiv.org/abs/2504.00992) | Transformer decoder over point features producing per-primitive parameters + a soft point-to-primitive assignment, with a **parsimony loss** (compactness is an explicit objective, not an accident) and a normal-based loss. Trained on ShapeNet, shown to transfer to ScanNet++ objects and Replica. Downstream robotics use is a stated motivation. Prim: sq. Real: yes (transfer). ⚠ ShapeNet-trained. | **high** |
| 16 | `yang-chen-cuboid-abstraction` | Yang & Chen — *Unsupervised Learning for Cuboid Shape Abstraction via Joint Segmentation from Point Clouds*, SIGGRAPH 2021. [arXiv:2106.03437](https://arxiv.org/abs/2106.03437) · [code](https://github.com/SilenKZYoung/CuboidAbstractionViaSeg) | Cuboids are *literally* box faces — the most direct route from cloud to "6 faces to inspect". Joint segmentation + abstraction with no part labels. ⚠ ShapeNet-trained, assumes canonical orientation and complete clouds — a real constraint violation for a tabletop scan; read partly to quantify how badly that hurts. Prim: cub. | med |
| 17 | `spfn-supervised-primitive-fitting` | Li, Sung, Dubrovina, Yi, Guibas — *Supervised Fitting of Geometric Primitives to 3D Point Clouds (SPFN)*, CVPR 2019. [arXiv:1811.08988](https://arxiv.org/abs/1811.08988) | The supervised baseline the whole learned branch benchmarks against; predicts type **and** parameters without per-input tuning — the exact weakness of RANSAC. Establishes the accuracy ceiling learning offers on ABC. Prim: pl/sp/cy/co. ⚠ ABC-trained (CAD, clean). | med |
| 18 | `point2cad-reverse-engineering` | Liu, Obukhov, Wegner, Schindler, et al. — *Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds*, CVPR 2024. [project](https://www.obukhov.ai/point2cad.html) | Owns the step everyone else skips: **extend, intersect and clip** fitted surfaces so topology emerges — turning unbounded primitives into bounded faces with edges. "Face 3" as an addressable thing requires exactly this. Prim: pl/cy/sp/co + freeform (INR). ⚠ ABC benchmark. [B5] | med |
| 19 | `hierarchical-object-representation-superquadrics` | *Hierarchical Object Representation for Spatial Robot Perception: Points, Meshes, and Superquadrics*, 2026. [arXiv:2606.01545](https://arxiv.org/abs/2606.01545) | Four-layer per-object representation built **from RGB-D observations**: object cloud → mesh → superquadrics → *inflated* superquadrics for collision margins. Mirrors our own object-inflation constraint and is a candidate skeleton for the action space (object → primitive → view). Real: yes. [B5] | med |
| 20 | `find-superquadric-repo` | robotology — `find-superquadric`. [github](https://github.com/robotology/find-superquadric) | A robotics-grade, battle-tested superquadric fit **from a partial cloud on a table**, with density-clustering outlier removal and a table-plane prior — our exact deployment setting. The pragmatic baseline to try in an afternoon. Code: yes (C++/YARP). | med |
| 21 | `vezzani-superquadric-grasping` | Vezzani, Pattacini, Natale — *A grasping approach based on superquadric models*, ICRA 2017; pipeline paper in [Front. Robot. AI 4:59, 2017](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2017.00059/full) | Existence proof that a primitive fitted to a **single partial view** is accurate enough to plan *contact* poses — a far tighter tolerance than pointing a camera. If grasping works off partial-scan superquadrics, an orthogonal view plausibly does too. Read the symmetry-based completion pre-step and the shape-prior variant. | med |

---

## D. Normals and uncertainty — the input to any orthogonality score

| # | Slug | Doc | Why it matters | Prio |
|---|------|-----|----------------|------|
| 22 | `iterative-pca-robust-normal-estimation` | Sanchez et al. — *Robust normal vector estimation in 3D point clouds through iterative principal component analysis*, ISPRS J. 163, 2020. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0924271620300575) | M-estimator-reweighted PCA that also **separates neighbours into inliers/outliers** (a per-point reliability flag we can surface), plus an 8-method comparison against PCA, 2-jets, VCM, Hough, Hough-CNN, PCV, PCPNet. Best single entry point for "which normal estimator, and how wrong is it". | **high** |
| 23 | `plane-extraction-uncertainty-cramer-rao` | Pathak, Vaškevičius, Birk — *Uncertainty analysis for optimum plane extraction from noisy 3D range-sensor point-clouds*, Intelligent Service Robotics 3(4), 2010. [Springer](https://link.springer.com/article/10.1007/s11370-009-0057-4) | Closed-form covariance of the fitted plane's **normal and offset** under a range-error model that is a function of range *and incidence angle* — the propagation RQ1 needs to turn "orthogonal" into "orthogonal ± σ". Compares exact ML, renormalization, approximate LS, and eigenvector perturbation, with Cramér–Rao bounds. [B1] | **high** |
| 24 | `generalized-vcm-robust-geometry` | Cuel, Lachaud, Mérigot, Thibert — *Robust Geometry Estimation using the Generalized Voronoi Covariance Measure*, 2014. [arXiv:1408.6210](https://arxiv.org/abs/1408.6210) | Normal/curvature estimation with **theoretical stability guarantees parameterized by a noise level** (Hausdorff-type), benchmarked against VCM, jet fitting and randomized Hough on a noisy ellipsoid with outliers. Gives a defensible error bound rather than an empirical curve. [B1] | med |
| 25 | `boulch-marlet-hough-normals` | Boulch & Marlet — *Fast and Robust Normal Estimation for Point Clouds with Sharp Features*, CGF (SGP) 31(5), 2012. [doi:10.1111/j.1467-8659.2012.03181.x](https://doi.org/10.1111/j.1467-8659.2012.03181.x) | Box edges are precisely where PCA and jet normals smear, and face segmentation depends on those edges surviving. Plane-voting in a Hough accumulator is the classical feature-preserving answer; read for the sharp-feature failure analysis as much as the method. | med |
| 26 | `isprs-plane-uncertainty-propagation` | *Uncertainty modeling for point cloud-based automatic indoor modeling*, ISPRS Archives XLIII-B2-2022:395. [PDF](https://isprs-archives.copernicus.org/articles/XLIII-B2-2022/395/2022/isprs-archives-XLIII-B2-2022-395-2022.pdf) | Propagates plane-fit covariance into the **edges produced by intersecting two planes** — the same operation Point2CAD does to get faces, now with error bars. Reports that the smallest-support planes dominate the error, and our dropout regions are exactly the small-support ones. | med |
| 27 | `m3c2-ep-registration-error-propagation` | Winiwarter, Anders, Höfle et al. — *M3C2-EP: Pushing the limits of 3D topographic point cloud change detection by error propagation*, ISPRS J. 178, 2021. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0924271621001696) | The recipe for our defect profile: replaces roughness-based noise proxies with propagation of **measurement uncertainty *and* the full ICP alignment covariance** into per-point uncertainty. This is the template for "how do a few mm of pose error become an angular error on face 3's normal". | med |

---

## E. Our specific defect modes — dropout, depth bias, fusion offset

| # | Slug | Doc | Why it matters | Prio |
|---|------|-----|----------------|------|
| 28 | `d415-metrological-characterization` | Carfagni et al. — *Metrological and Critical Characterization of the Intel D415 Stereo Depth Camera*, Sensors 19(3), 2019. [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6387018/) | Measures the distribution of point-to-surface distances against **known planes, cylinders and spheres** — i.e. it characterizes the sensor using precisely the primitives we intend to fit. Also notes no international standard exists for depth-camera metrology, which shapes how we defend our own numbers. [B1][B2] | med |
| 29 | `transparent-translucent-depth-assessment` | *An Experimental Assessment of Depth Estimation in Transparent and Translucent Scenes for Intel RealSense D415, SR305 and L515*, Sensors 22(19), 2022. [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9572012/) | Quantifies the non-Lambertian failure across sensor families — missing values *and* outliers, not just holes. Needed to answer whether dropout biases a fitted primitive systematically or merely adds variance. | med |
| 30 | `structured-light-stereo-noise-model` | Nguyen, Izadi, Lovell / Landau et al. lineage — *Noise in Structured-Light Stereo Depth Cameras: Modeling and its Applications*. [arXiv:1505.01936](https://arxiv.org/abs/1505.01936) | An explicit axial + lateral noise model for active-stereo depth. This is the parameterization we would inject into synthetic clouds to test fitters at *our* noise level rather than generic Gaussian. [B2] | med |
| 31 | `intel-realsense-stereo-whitepaper` | Keselman, Woodfill, Grunnet-Jepsen, Bhowmik — *Intel RealSense Stereoscopic Depth Cameras*, CVPRW 2017. [arXiv:1705.05548](https://arxiv.org/abs/1705.05548) | Vendor-authored characterization of the exact sensor family the D405 belongs to: subpixel/disparity noise vs range, projector role, dropout causes. Cheap, authoritative grounding for the sensor-side givens. | low |
| 32 | `flange-based-handeye-calibration` | Peters et al. — *Flange-Based Hand-Eye Calibration Using a 3D Camera With High Resolution, Accuracy, and Frame Rate*, Front. Robot. AI 7:65, 2020. [Frontiers](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.00065/full) | Quantifies achievable hand-eye accuracy on UR e-series arms and — the useful part — the **anisotropy of the residual** (z-dominant translational error). A rotational hand-eye error scales with standoff, which is a concrete candidate explanation for our measured ~4 cm fused-vs-single-view offset. | med |
| 33 | `domain-randomization-depth-simulation` | Dai et al. — *Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects*, ECCV 2022. [arXiv:2208.03792](https://arxiv.org/abs/2208.03792) | Simulates specular/transparent depth artifacts by reproducing the project–capture–stereo-match pipeline, which is how we would generate defective clouds for the benchmark. ⚠ its *restoration* half crosses into the deferred "learned repair" topic — extract the simulator, flag the rest. [B2] | med |
| 34 | `mechmind-dark-reflective-scanning` | Mech-Mind — *How to get optimal results when 3D scanning dark and reflective objects?* [blog](https://www.mech-mind.com/blog/3d-scanning-dark-reflective-objects-tips.html) | Practitioner-level catalogue of dropout mitigations orderable by cost: exposure/HDR bracketing, cross-polarization, wavelength/power, viewpoint sweep angle, matting spray. Relevant because **changing the survey viewpoint** is itself one of the mitigations — a possible feedback loop into view planning. | low |
| 35 | `robust-extrinsic-symmetry-estimation` | *Robust Extrinsic Symmetry Estimation in 3D Point Clouds*, 2021. [arXiv:2109.09927](https://arxiv.org/abs/2109.09927) | A cylinder orbit needs the **axis**, and half the cylinder was never scanned. Statistical-estimator symmetry recovery that is explicitly robust to outliers *and missing parts*, where voting schemes fail. Cross-read with PotSAC (MLESAC axis-of-revolution estimation from noisy fragments) cited in the Structure-from-Sherds++ line. | med |

---

## F. Failure detection and the fallback trigger

| # | Slug | Doc | Why it matters | Prio |
|---|------|-----|----------------|------|
| 36 | `daftry-introspective-perception` | Daftry, Zeng, Bagnell, Hebert — *Introspective Perception: Learning to Predict Failures in Vision Systems*, IROS 2016. [arXiv:1607.08665](https://arxiv.org/abs/1607.08665) | The framing for "know when the decomposition is unreliable → fall back to the uniform sphere". Key claim to test against our case: they argue prediction from *raw input* beats model-internal uncertainty — if true for a geometric fitter too, an inlier-ratio threshold is not enough. | med |
| 37 | `safety-monitoring-perception-survey` | *Safety Monitoring of Machine Learning Perception Functions: a Survey*, 2024. [arXiv:2412.06869](https://arxiv.org/abs/2412.06869) | Taxonomy of runtime monitors, OOD detectors and confidence calibration for perception — supplies the vocabulary and the option space for designing our fallback gate, plus how such gates are evaluated (which our benchmark must mirror). [B2] | low |

---

## G. Bridging candidates — decomposition already used for view planning

| # | Slug | Doc | Why it matters | Prio |
|---|------|-----|----------------|------|
| 38 | `hierarchical-viewpoint-planning-inspection` | *Hierarchical Viewpoint Planning for Complex Surfaces in Industrial Product Inspection*, IEEE (2024). [IEEE Xplore 10375828](https://ieeexplore.ieee.org/document/10375828/) | The single closest published instance of our whole pipeline: **multilevel geometric representation of the object point cloud** → divide and conquer → viewpoints, with bottom-up gridization for regular surfaces and top-down confidence-driven optimization for irregular ones. If any paper has already done "decompose, then plan views", it is this one. [B1][B4] | **high** |
| 39 | `fc-planner-skeleton-decomposition` | Feng et al. — *FC-Planner: A Skeleton-guided Planning Framework for Fast Aerial Coverage of Complex 3D Scenes*, ICRA 2024. [arXiv:2309.13882](https://arxiv.org/abs/2309.13882) | Decomposition-for-view-planning precedent from a **different geometry family**: skeleton-based space decomposition into simple-geometry subspaces that then drive viewpoint sampling. Useful as the counterfactual — what we would use if primitive fitting proves infeasible. ⚠ UAV, free-flying camera, scene scale. [B4] | low |

---

## Toolchain quick-reference (integration reality check)

| Tool | Planes | Cylinders | Superquadrics | Notes |
|------|--------|-----------|---------------|-------|
| Open3D | ✅ `segment_plane`, ✅ `detect_planar_patches` (bounded!) | ❌ ([#6602](https://github.com/isl-org/Open3D/issues/6602)) | ❌ | Also normals, statistical/radius outlier removal |
| CGAL `Shape_detection` | ✅ | ✅ | ❌ | pl/cy/sp/co/to; custom primitive interface |
| PCL `SACSegmentationFromNormals` | ✅ | ✅ | ❌ | [cylinder tutorial](https://pcl.readthedocs.io/projects/tutorials/en/master/cylinder_segmentation.html); needs normals first |
| [pyRANSAC-3D](https://github.com/leomariga/pyRANSAC-3D) | ✅ | ✅ | ❌ | Python, Open3D-interop, quickest prototype |
| [EMS](https://github.com/ChirikjianLab/EMS-superquadric_fitting) | — | — | ✅ | Probabilistic, outlier posterior |
| [find-superquadric](https://github.com/robotology/find-superquadric) | — | — | ✅ | Partial cloud + table prior, YARP |
| SuperDec / SuperFlex | — | — | ✅ (learned) | Pretrained weights; real-scan transfer claimed |

The gap is stark and worth stating up front for the readthrough: **there is no single
library that gives us bounded box faces and a cylinder axis with confidence scores**.
The likely build is Open3D (normals, patches) + CGAL or PCL (cylinder) + one
abstraction method, glued.

---

## Questions this branch must answer

### Feasibility of decomposition on our input (RQ2, critic mandate)

1. On a fused multi-view PLY with a few mm of registration error and a measured ~4 cm
   fused-vs-single-view offset, what **angular error on a recovered face normal or
   cylinder axis** should we expect? At what noise / incompleteness level do published
   methods stop returning the correct *primitive type* at all?
2. Does the contract's object set close with plane + cylinder + sphere + cone RANSAC,
   or do real chamfered/rounded parts force superquadrics or cuboid decomposition —
   and what does each vocabulary buy in recovered-normal accuracy?
3. Which methods recover **bounded faces** (extent, boundary, centroid) rather than
   infinite primitives? A "scan this face orthogonally" action needs a face centre and
   a face size, not a plane equation. Candidates: Araújo & Oliveira patches, Point2CAD
   surface clipping, cuboid abstraction.
4. Is there **any** published system that decomposes a scanned object into geometric
   primitives and then generates per-primitive viewpoints? If not, what is the nearest
   precedent and what did it substitute for primitives (skeleton, gridization,
   mesh patches)? — this determines whether our contribution is novel or a re-run.
5. How repeatable is decomposition across re-scans of the same object? Does "face 3"
   stay face 3 — in identity, ordering, and orientation? RQ3's consistency requirement
   collapses if extraction output is order-unstable. (Feeds RQ3, B5.)
6. For a **partially observed cylinder** (roughly half the surface scanned, plus
   dropout on the glossy side), how well can the axis be recovered, and does
   symmetry-based completion (mirror-plane estimation, PotSAC-style axis estimation)
   improve it enough to plan an orbit?

### Trustworthy normals and propagated uncertainty (RQ1)

7. What is the quantified relationship between per-point noise σ, neighbourhood size
   k, and **normal angular error** for each estimator family (PCA, jets, VCM, Hough,
   learned)? We need an actual number to hand Branch 1.
8. Is there a closed-form covariance for a fitted plane's / cylinder's **orientation**
   under a realistic range-error model (range- and incidence-angle dependent)? Can we
   deliver "the normal of face 3, ± σ" instead of a point estimate?
9. How does **registration/alignment covariance** — not just per-point sensor noise —
   propagate into fitted-primitive orientation? Does the M3C2-EP treatment transfer
   from change detection to primitive fitting?
10. Given our sensor and the hard-minimum reachable shell radius, what **angular error
    budget** do we actually have before "orthogonal" stops being meaningful? Cross with
    Branch 1's depth-error-vs-incidence-angle numbers to get a threshold, not a vibe.

### Defect modes (constraints)

11. What are the documented mechanisms and magnitudes of depth dropout on glossy/dark
    surfaces for active stereo, and does dropout bias fitted primitives
    **systematically** (shrinking a face, shifting an axis, tilting a normal) rather
    than merely inflating variance? Systematic bias is the dangerous case: it is
    invisible to residual-based confidence.
12. What is the most likely source of the measured ~4 cm fused-vs-single-view offset —
    hand-eye rotational error scaled by standoff, depth bias, or fusion drift — and
    which of these does the literature say is correctable versus something the
    decomposition must simply tolerate?
13. Which preprocessing steps (voxel downsampling, statistical/radius outlier removal,
    denoising) measurably change fitting outcomes, and by how much? Are there reported
    cases where denoising **harms** primitive recovery by over-smoothing the edges that
    face segmentation depends on?

### Failure detection and the fallback (contract's fallback requirement)

14. What computable signals do papers use to declare a fit unreliable — inlier ratio,
    residual RMS, MDL / energy score, EMS posterior outlier probability, MonteBoxFinder
    score, GNC/TEASER optimality certificate? Which are obtainable from an
    off-the-shelf implementation with **no retraining**?
15. Has anyone calibrated such a score against **downstream task failure** ("fit score
    below X ⇒ the derived pose is wrong")? If not — and I expect not — we must build
    that mapping; what would the minimal experiment look like on our objects?
16. Does the introspection literature support a two-tier design (primitive
    decomposition, else uniform sphere)? Specifically: is a separate failure predictor
    over raw input required, or does the fitter's own confidence suffice for a
    *geometric* (not learned) estimator?

### Integrability (auxiliary goal: integrable, not a research programme)

17. Which combination of Open3D + CGAL/PCL + one abstraction method covers box **and**
    cylinder on a PLY with no training, and what is the realistic per-object runtime on
    our cloud sizes?
18. Which learned methods have released weights **and** demonstrated real-scan
    generalization (SuperDec → ScanNet++, SuperFlex → partial real), versus ShapeNet-
    only evaluation? Which of them assume canonical object orientation — a direct
    constraint violation for an arbitrarily posed tabletop object?

### Benchmark tagging (feeds Branch 2)

19. Which datasets and protocols here can seed our benchmark — Fit4CAD's metric suite,
    SHREC 2022's comparison protocol, ABC — and can any of them be corrupted with **our
    specific defect profile** (registration jitter, view-dependent dropout, depth bias)
    rather than generic isotropic Gaussian noise?
20. What is the ground-truth story for "the correct decomposition" of a *real* scanned
    object? Do we need CAD models of our test objects, or do published protocols accept
    human-annotated segmentation — and what does that imply for benchmark cost?

---

## Search trail

Executed via WebSearch/WebFetch on 2026-08-27. Queries, in order:

1. `geometric primitive fitting point cloud survey RANSAC plane cylinder cone`
2. `efficient RANSAC point-cloud shape detection Schnabel Wahl Klein`
3. `superquadric recovery point cloud robust noisy partial scan probabilistic`
4. `normal estimation point cloud survey comparison noisy deep learning 2023 2024`
5. `cuboid abstraction point cloud unsupervised shape abstraction primitives real scans`
6. `plane segmentation point cloud region growing survey comparison evaluation`
7. `uncertainty quantification plane fitting point cloud covariance propagation normal estimation error`
8. `Intel RealSense D405 depth accuracy evaluation bias specular surfaces characterization`
9. `SuperDec superquadric decomposition point clouds transformer 2025`
10. `RANSAC model selection MDL primitive fitting goodness of fit confidence shape detection`
11. `CONSAC multi-model fitting learned robust estimation planes point cloud energy minimization PEARL`
12. `SPFN ParseNet HPNet supervised primitive fitting network point cloud parametric surfaces`
13. `superquadric fitting grasping robot partial point cloud real sensor Vezzani`
14. `structured light stereo depth camera specular glossy dark surface dropout missing depth mitigation industrial`
15. `registration error propagation multi-view fused point cloud effect on surface fitting accuracy`
16. `symmetry detection partial point cloud axis of revolution estimation robust`
17. `primitive-based view planning inspection decompose object into primitives viewpoint generation point cloud`
18. `Open3D RANSAC plane segmentation cylinder fitting documentation robust`
19. `"Segment Any Primitive" zero-shot 3D primitive segmentation point cloud 2025`
20. `planar patch detection robust RGBD point cloud Araujo Oliveira 2020 boundary`
21. `survey reverse engineering point cloud to CAD segmentation primitive extraction review 2023 2024`
22. `comparison normal estimation methods point cloud PCA jet fitting Hough robust benchmark classic`
23. `Fit4CAD benchmark point cloud fitting simple geometric primitives CAD objects`
24. `graduated non-convexity robust estimation outliers 3D perception Yang Carlone certifiable`
25. `hand-eye calibration error effect fused point cloud accuracy robot arm eye-in-hand depth camera`
26. `depth sensor noise simulation model RGB-D point cloud realistic artifacts synthetic training`
27. `introspection perception failure prediction robotics reliability estimate model fitting fallback`
28. `hierarchical viewpoint planning appearance inspection point cloud multilevel geometric representation`
29. `survey 2025 2026 geometric primitive fitting shape abstraction point clouds review learning`
30. `arxiv 2606.01545 hierarchical object representation points meshes superquadrics robot perception`

Landing pages fetched for verification: CGF `10.1111/cgf.13451` (Kaiser survey),
arXiv `2607.01015` (SuperFlex), arXiv `2207.14268` (MonteBoxFinder), arXiv `2607.05568`
(training-free primitive abstraction).

**Search budget note:** the session WebSearch budget (200) was exhausted during this
pass. Two intended follow-ups were not run and should open the next round:
- a dedicated hunt for a **2024–2026 STAR/survey update** to Kaiser et al. (Eurographics
  State-of-the-Art Reports, Computers & Graphics) — none surfaced, which is itself a
  finding worth confirming;
- the **SHREC 2022 track results table** (which method won on which corruption), needed
  to convert entry #3 from "read this" into an actual method shortlist.

### Deliberately excluded (per branch scope)

- Semantic segmentation / object recognition pipelines (SAM3D, SAI3D, PartSLIP lineage)
  — surfaced repeatedly, out of scope; `segment-any-primitive` (CVPRW 2025) was
  considered and dropped: workshop paper, no released code, needs multi-view renders.
- Learned cloud completion / generative repair — deferred by the plan. Flagged only
  where a paper's *simulator* half is usable (#33) or where symmetry completion is a
  pre-step inside an in-scope method (#21, #35).
- SLAM / registration-pipeline improvements (global bundle adjustment, loop closure,
  Gauss–Helmert stochastic registration models) — the survey pipeline is a given. Kept
  only the **error-propagation** view of registration (#27), which is about consuming
  its output, not fixing it.
- Mesh reconstruction for rendering (Poisson, surface-reconstruction surveys),
  CAD-program synthesis (CAD-Recode, TransCAD, SECAD-Net), Gaussian-splatting
  primitive work — all downstream or orthogonal.

### Notes for the reader threads

- Every learned method here is trained on ShapeNet or ABC. When reading, record the
  **exact evidence of real-scan performance**: which dataset, how noisy, what
  degradation vs synthetic. That number is this branch's deliverable.
- Record every quantified claim of the form *"normal/axis error X° at noise level σ"* —
  Branch 1 consumes these directly to propagate uncertainty into the orthogonality
  score, and Branch 2 to set benchmark noise levels.
- Tag any paper that gives a computable fit-quality score with whether that score is
  exposed by a released implementation. The fallback gate is a deliverable, not a
  footnote.
