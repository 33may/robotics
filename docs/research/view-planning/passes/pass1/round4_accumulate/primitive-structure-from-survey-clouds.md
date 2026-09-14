# Branch 3 — Extracting exploitable geometry from defective survey clouds
## Round 4: branch synthesis

**Date:** 2026-08-27
**Branch slug:** `primitive-structure-from-survey-clouds`
**Feeds:** RQ2 (shape → views), RQ1 (trustworthy normals for orthogonality scoring),
RQ3 (face/axis decomposition as action-space skeleton), RQ4 (what gets presented);
benchmark methodology (Branch 2).
**Inputs:** [`round1_plan.md`](../round1_plan.md) Branch 3 ·
[`round2_map`](../round2_map/primitive-structure-from-survey-clouds.md) (39 candidates) ·
[`round3_reading`](../round3_reading/primitive-structure-from-survey-clouds/) (20 documents read in full)
**Contract:** [`2026-08-27-geometry-aware-view-planning.md`](../../../../2026-08-27-geometry-aware-view-planning.md)

**Standing critic question:** *does decomposition survive our defect profile?*

---

## 0. Coverage of this synthesis

20 of the map's 39 candidates were read. Sections A (spine, 7/7), B (robust
detection, 5/5), C (abstraction, 5/9), D (normals & uncertainty, 2/6), G
(bridging, 1/2). **Sections E (our specific defect modes: D405/D415
characterization, dropout physics, hand-eye error, depth-noise simulation —
8 documents) and F (failure detection and the fallback trigger — 2 documents)
were not read at all.** That is the largest known hole and it sits exactly on
top of the branch's central question; every claim below about our defect triad
is therefore inference from geometry-processing literature that does not model
it, not evidence from sensor literature that does.

Also unread and load-bearing: Point2CAD (#18, bounded faces via extend/
intersect/clip), find-superquadric (#20, partial cloud + table prior — the
closest deployment match in the whole map), M3C2-EP (#27, the one candidate for
registration-covariance propagation), robust extrinsic symmetry estimation
(#35, the only candidate for partial-cylinder axis completion).

---

## 1. Headline verdict

**Decomposition is available, fast, and cheap at our scale. It is not
validated for our defects, and the field has no instrument that would tell us
if it failed.**

Three findings carry that verdict:

1. **Runtime and availability are non-issues.** Efficient RANSAC runs 12k
   points in 0.57 s and 2M in 41 s (Schnabel Table 1); RSPD does 6M points in
   6.4 s; CGAL region growing is ~3.4 µs/point; EMS fits a superquadric to a
   1500-point PLY in 91 ms. Our tabletop clouds are at the small end of every
   published range. The auxiliary goal ("integrable, not a research
   programme") is satisfiable on the runtime axis.

2. **Every robustness number in this literature is measured under a noise
   model that excludes all three of our defects.** Across 20 documents the
   corruption models are: isotropic per-point Gaussian on coordinates
   (Schnabel, EMS, CSA, Raffo, Sanchez, SHREC A2), uniform coordinate noise
   (SHREC A1, Raffo), along-normal uniform ±1% (SPFN), uniformly-random
   spatial outliers (Schnabel, EMS, SPFN, PEARL, GNC), geometric hole-punching
   (SHREC A4, Fit4CAD via CloudCompare), and visibility-only occlusion (SuperFlex
   HPR + spherical masking). Exactly one document models range- and
   incidence-angle-dependent range noise (Pathak et al. 2010, σ ∝ d²/|n̂·m̂|³)
   and it is a ToF flash-ladar model with a κ borrowed from a *different*
   sensor's calibration. **Nobody models registration/fusion covariance,
   view-dependent glossy/dark dropout, or systematic depth bias.** Araújo &
   Oliveira contains the field's entire treatment of registration error in one
   throwaway line ("errors in registering partial point clouds also lends to
   noise"), unmodelled.

3. **The failure mode that would hurt us most is structurally invisible to
   every confidence signal the field offers.** Kaiser's survey notes that the
   whole robustness vocabulary — noise / outliers / incomplete data — has *no
   category* for systematic geometric bias from a directional sensor artifact.
   PEARL's per-point residual term would score a systematically shifted face as
   a *good* fit, because bias by definition does not raise per-point variance.
   Sanchez's M-estimator is built to separate two clean surfaces at an edge,
   not a true surface from a coherently displaced duplicate of itself — a 4 cm
   fused-vs-single-view offset would plausibly be embraced as surface, not
   rejected as outliers. Every fit score available off the shelf (inlier ratio,
   residual RMS, MFE, coverage, EMS posterior, MonteBoxFinder μ) is a
   *residual-family* statistic and inherits this blind spot.

The practical reading: build the decomposition, but treat its output as
*unvalidated for our input class* until we measure it ourselves, and do not
plan a fallback gate around a residual score alone.

---

## 2. What is known — the numbers census

The plan instructed this branch to record every quantified claim of the form
"normal/axis error X° at noise level σ". Here is everything that exists across
20 documents, with its noise model attached. **Read the noise-model column
first; without it none of these numbers transfer.**

### 2.1 Angular error on normals and axes

| Source | Quantity | Value | Noise model / conditions |
|---|---|---|---|
| SPFN Table 1 | point normal error | **8.66°** | ANSI/TraceParts CAD, complete surfaces, single fixed along-normal uniform ±1% of half-extent, 8192 pts |
| SPFN Table 1 | primitive axis error | **1.51°** | same — **conditioned on correct type** (96.93% of cases) |
| SPFN Table 1 row 1 | Eff. RANSAC (CGAL) + jet normals, 8k pts | normal 11.42°, **axis 7.54°**, type acc 52.92% | same data |
| SPFN Table 1 row 2 | Eff. RANSAC, 64k pts (8×) | normal 6.92°, **axis 2.42°**, type acc 43.90% | same data |
| SPFN Table 1 row 6 | Eff. RANSAC given SPFN's own normals+seg+type | **axis 7.02°** | segmentation quality lifts coverage/type but *not* axis precision |
| SPFN Table S1 | +10% uniform outliers | axis 1.51 → **1.97°** (+30%), normal 8.66 → 9.67° | outlier contamination, not our defect |
| SPFN ablation | no `L_axis` loss | **axis 6.27°** while type acc stays 96.47% | type-correctness and axis-accuracy diverge |
| Sanchez Fig. 9 | mean normal angular error, two-plane edge | **0.3° → 2.2°** as noise goes 0 → 434% of mean NN spacing | isotropic Gaussian; their robust IRLS-PCA |
| Sanchez Fig. 9 | PCA / 2-jets / VCM, same test | flat **~3–5°** across the whole noise range | error dominated by edge-smoothing bias, not noise |
| Sanchez Fig. 10 | pipe-bend (curved) | their method **0.3 → 1.4–2.2°**; PCA **0.85 → 1.85°** | isotropic Gaussian, complete synthetic cylinders |
| Sanchez Fig. 11 | neighbourhood sweep | their method monotone k=100 (0.065°) → k=400 (0.02°); **PCA/2-jets/BF peak at k≈150–200 then degrade** | averaged over 16 noise levels |
| Sanchez Fig. 13 | 14 CAD meshes, RMS | **11° → 17°** (noise 0 → 0.2% bbox diag); RMSτ 25° → 37°; PCP-net worst ~75–76°, flat | RMS dominated by hardest points/corners |
| Sanchez Fig. 14–16 | simulated indoor LiDAR | **3.7 – 7.8°** their method; PCA 5 – 10.5°; PCV up to 21° (unstable); PCP-net 8.5 – 14° | closest proxy to sampling anisotropy, but density anisotropy ≠ positional bias |
| Raffo Fig. 8 | cylinder axis, ‖a−â‖₂ → angle | 0° / 0° / 0° / **≈25°** | coordinate noise U(±0.1/±0.5/±1.0) on radius ~1.5 shape; **initial estimate only**, ~1/8 coverage |
| Raffo Fig. 8 | cone axis | 0° / 0° / **≈11°** / ≈21° | same |
| SuperFlex Table 9 | "normal completeness" \|dot(n,n_gt)\| | 0.83 – 0.85 (naively arccos ≈ **32°**) | complete ShapeNet clouds only; aggregate over all surface, not per-primitive |
| Araújo & Oliveira §4.2 | **the field's own correctness bar** | a detection counts as correct if normal error **≤ 30°** | this is the matching criterion, not an error measurement |

Two things fall out immediately. First, **the ceiling on achievable normal
accuracy under idealised noise is 0.3–2° for well-sampled flat regions**
(Sanchez), and the ceiling on axis accuracy for a well-segmented complete CAD
primitive is 1.5–2.5° (SPFN). Those are ceilings, not predictions. Second,
**Efficient RANSAC's axis error on clean synthetic CAD is 7.0–7.5°** (SPFN rows
1 and 6) — and that number does not improve when it is handed a perfect
segmentation, which locates the deficit in the *fitting* step, not the
segmentation step.

### 2.2 Incompleteness

| Source | Result |
|---|---|
| EMS Fig. 4(e) | global-optimality rate vs. surface coverage: 1.0→**100%**, 0.8→100%, 0.6→99.8%, 0.4→97.4%, **0.2→86.4%**. Multi-restart LSQ baseline: 99/78/70.4/68.8/**65.6%**. Single-restart LSQ: flat ~30% *regardless of coverage* (local optima dominate over incompleteness) |
| SuperFlex Table 3 | model trained on complete clouds **collapses to IoU 0.20 on partial input** (3.5× drop from 0.72). Occlusion-augmentation + fine-tuning recovers to IoU 0.54 but **F-score only 0.18** — worse than every complete-cloud baseline in their own Table 1 |
| SHREC2022 Table 7 | single spherical hole (A4): cylinder TPR 0.80 (M1) to 1.00 (M4) — weakest method loses ~10 TPR points to *one* hole |
| Fit4CAD §4.3 | HT (exhaustive parametric search) "hardly affected" by missing data; PG (curvature region growing) shows "more prominent variation". Binary with/without, no dose-response |
| Kaiser §4.2.4 | "**Region growing based methods do not perform well when data is missing**, as they only exploit existing data without trying to create a global model." Multiscale and multi-view-reconstruction methods handle it better |
| Schnabel §5.1 | only incompleteness test is a synthetic sphere *octant* — symmetric, single case |
| Raffo §3.4 | ~1/8-coverage partial segments work for the *initial estimate* at low noise; no coverage-fraction sweep, no final-fit accuracy |

**Nobody tests a half-scanned cylinder with one-sided dropout.** The closest
analogues are a symmetric sphere octant, a random spherical hole, and a
contiguous cap. Symmetry-based completion (mirror plane, PotSAC-style axis
estimation) appears in **zero** of the 20 documents read.

### 2.3 Type-recognition breakdown thresholds

| Source | Threshold |
|---|---|
| Kaiser §4.2.4 quoting Schnabel | "up to **20% of Gaussian noise**" still recovers the primitive; "**95% outliers**" tolerated |
| Schnabel §5.1 | "for a noise degree higher than **20%** the detection starts to become unstable and the algorithm sometimes returns false shape types"; 95% outliers tolerated *at 2% noise* |
| Raffo §4.2.1 | ABC, Gaussian σ=0.50: "**two cylinders are mislabeled as cones**" — geometric-similarity confusion, not random error |
| Raffo Table 3 | at σ=0.50 both their method and the compared method **collapse equally** (TPR 0.076 vs 0.076) — the robustness advantage vanishes at high noise |
| SHREC2022 Table 1 | all six methods >90% per-class accuracy, >95% macro. **Cylinders, cones and tori consistently hardest**; cone TPR as low as 0.7784, torus TPR as low as 0.7297 |
| Raffo Table 2 (Fit4CAD) | plane ACC 0.999; **sphere TPR only 0.667, torus TPR only 0.583** — recognition degrades well before overall accuracy suggests |
| PEARL Fig. 22 | **bimodal error histogram**: correct model count → 5–20% error; wrong model count → a distinct 40–70% "gross error" mode. **Count errors are discrete and catastrophic, not graceful** |

### 2.4 Runtime anchors

Efficient RANSAC 0.57 s @12k pts → 40.7 s @1.8M (Schnabel Table 1) · RSPD
0.42 s @360k → 6.39 s @6M, O(n log n), single-threaded (Araújo & Oliveira
Table 3) · CGAL region growing 0.76 s @300k → 4.06 s @1.2M, ~3.4 µs/pt · CGAL
head-to-head: Eff. RANSAC 78% shapes correct in 8 s vs region growing 100% in
15 s (Fig. 94.11) · Fit4CAD HT 2.6/50.7/**358 s** min/mean/max vs PG
1.7/286/**19,074 s (5.3 h)** worst case · SHREC2022 per pre-segmented cloud:
Hough/MATLAB 123 s, histogram+LSQ 4 s, PointNet GPU 0.28 s, **Minkowski+LSQ
0.04 s**, AlexNet CPU 0.5 s · EMS 91.1 ms (KIT) / 78.9 ms (BigBIRD) per object
@~1000–1500 pts · SuperDec 0.13 s/scene forward, LM step <1 s · SuperFlex 8 ms
feed-forward but **55 s** with the test-time optimisation that produces the
good numbers · Sanchez normals ~0.2 ms/point, fully parallelisable · ALSP plane
covariance **5.2 ms** per 25k-point fit (2009 MATLAB).

---

## 3. What the field actually does

### 3.1 Method families and their reputations

Kaiser's taxonomy (the branch spine) sorts the classical era into seven
frameworks. The ones that matter here, with their Table 1 ratings and what the
method papers actually showed:

- **RANSAC family (Efficient RANSAC, Schnabel 2007).** Rated *simple, general,
  accurate, robust to outliers*; weaknesses *many parameters, dependent on a
  minimal set, no spatial consistency*. Notably, Table 1's own row does **not**
  list noise robustness as a strength — the 20%-noise number comes from
  Schnabel's paper, not from the survey treating it as RANSAC's calling card.
  Ships in CGAL and PCL. Still the default and still the baseline everyone
  compares against; it reappears as a *sub-component* inside Fit4CAD's PG
  pipeline, SHREC's M2 cone branch, and MonteBoxFinder's candidate generator —
  i.e. the field's practical answer is usually "Efficient RANSAC plus
  something", not "instead of Efficient RANSAC."
- **Hough / voting (Raffo's localized voting, SHREC's M1, Fit4CAD's HT).**
  Motivated explicitly by RANSAC being "easily deceived in the presence of
  point cloud artifacts". Robust to missing data (Fit4CAD: "hardly affected"),
  weak on undersampling, slow (123 s/cloud MATLAB; 17–49 s per segment in
  Raffo). M1 and M4 "alternate first and second place" in SHREC across both
  classification and fitting — the one direct method and the one
  neural-with-LSQ-refinement method are the two most reliable end-to-end.
- **Region growing (RSPD, Fit4CAD's PG, CGAL Region Growing).** Rated
  *meaningful segmentation, spatial consistency*; weak on *missing data, noise,
  outliers, seed sensitivity*. Deterministic (CGAL states this explicitly).
  RSPD is the best-in-class robust-statistics instance: median/MAD instead of
  mean/PCA (50% breakdown point vs 0%), best average F1 (0.81) across seven
  real datasets, sub-7 s at 6M points.
- **Energy minimisation / global selection (PEARL 2012, Multi-X 2018).** Fits
  error + spatial coherence + MDL-style label cost jointly instead of greedily.
  PEARL needs ~3–5× fewer samples than plain RANSAC for equal accuracy, and 900
  vs 46,416 sampled models on a real scene. Multi-X adds mode-seeking to kill
  near-duplicate instances. **Both are demonstrated on the wrong data for us**
  — PEARL operates entirely on 2D image feature correspondences (never a point
  cloud), Multi-X on street-scale LiDAR with a 10 cm inlier threshold and
  ≤1000-point runtime experiments.
- **Learned abstraction (SPFN, CSA, SuperDec, SuperFlex).** Predict per-point
  properties and/or slot-based primitive sets. SPFN's central architectural
  claim — predict per-point properties then fit algebraically, don't regress
  parameters directly — is proven by its own DPPN ablation (type accuracy
  51.33% vs 96.93%) and independently corroborated by SHREC2022 (M3, pure
  PointNet regression with no LSQ refinement, has median MFE 4.49e-2, ~10–20×
  worse than every hybrid, *despite excellent classification on the same
  backbone*).
- **Probabilistic / MLE fitting (EMS).** Gaussian-uniform mixture on the
  superquadric surface, MLE with explicit outlier modelling, plus a symmetry-
  exploiting "S-step" that escapes local optima without random restarts.
  Produces a per-point posterior inlier probability as a free byproduct.

### 3.2 What the field does *not* do

This branch's most valuable output is a census of structural absences, because
several of them are things our design was implicitly assuming existed.

1. **No primitive detector outputs uncertainty.** Kaiser names this himself in
   §7.3: *"the amount of variability of primitives detected in a scene, as well
   as their uncertainty level, are so far not explicitly provided on the output
   channel of the detection systems."* His own proposed remedy is *interactive
   parameter-range exploration by a user* — not a statistical covariance. CGAL
   confirms it at the API level: no covariance, no confidence interval, no
   error bar anywhere in the Shape Detection component.
2. **No benchmark reports an angular error.** Fit4CAD's ground truth *contains*
   the axis/normal vectors in its parametric tuples and its metric suite never
   extracts an angle. SHREC2022's L2 conflates orientation with radius and
   position into one scalar. This is a field-wide reporting gap, not an
   oversight in one paper.
3. **Normals are exogenous everywhere.** Every algorithm in Kaiser §3 takes
   *"List of vertices and associated normals"* as its Input line. Schnabel,
   CGAL, RSPD, Raffo, SPFN's axis estimators — all consume normals and none
   propagate their error. Every primitive-fitting accuracy claim in this
   literature silently assumes normals of unstated, unquantified quality.
4. **No re-scan repeatability experiment exists.** Not one, in 20 documents,
   classical or learned. Kaiser's "temporal consistency" property is about
   dynamic scenes, not identity stability across independent re-scans of a
   static object.
5. **No calibration of any fit score against downstream task failure.**
   Confirmed absent in all 20. The branch's prior ("If not — and I expect not —
   we must build that mapping") is confirmed with certainty.
6. **No recovered-accuracy comparison between primitive vocabularies.** Kaiser
   is explicit that no such comparison exists in the classical era; nothing in
   the 2019–2026 papers read supplies one either.

### 3.3 The tooling reality (verified, not assumed)

| Tool | Planes | Cylinder | Bounded output? | Confidence? | Determinism |
|---|---|---|---|---|---|
| **Open3D** | ✅ `segment_plane` (infinite) + ✅ **`detect_planar_patches` → `list[OrientedBoundingBox]`** | ❌ (#6602 open since 2024-01, **zero maintainer response**; sphere also absent per thread) | **YES for planes**: `.center`, `.extent`, `.R` (z-col = normal) | ❌ (`probability` is a search hyperparameter, not a returned score) | RANSAC randomized |
| **CGAL Eff. RANSAC** | ✅ | ✅ (+ sphere/cone/torus) | ❌ — **verified accessor-by-accessor**: `Cylinder::axis()` returns an *infinite* `Line_3`; `Cone` has no height; `Torus` no arc extent | caller-computed `coverage` + mean residual only | **Non-reproducible run-to-run on identical input, and no seed parameter exists anywhere in the public API** |
| **CGAL Region Growing** | ✅ | ✅ (+ sphere; no cone/torus for point sets) | ❌ (doc literally titled "detecting **(infinite)** cylinders") | + `minimum_radius`/`maximum_radius` plausibility gate | **Deterministic** given identical input ordering |
| **PCL** | ✅ | ✅ `SACMODEL_CYLINDER` | ❌ | — | randomized |
| **EMS** | — | — (superquadric family) | bounded *primitive*, not faces | ✅ **per-point posterior inlier probability**, released MATLAB+Python, reads `.ply` directly | 1 init, no restarts |
| **SuperDec / SuperFlex** | — | — | bounded *primitive* (pose + half-extents) | existence probability (ablates to "marginal" over a mass threshold) | unordered slot prediction |
| **MonteBoxFinder** | via CGAL | ❌ cuboids only | ✅ **`Cuboid` = 6 face normals + 8 corners** | μ0/μ1 UCB scores, relative not absolute | stochastic search |
| **SPFN** | ✅ | ✅ | ❌ | point-count discard threshold | **no released weights**, TF 1.10 + custom CUDA → full retraining |
| **CSA (Yang & Chen)** | — | — cuboids | ✅ (t + s + R) | — | **category-specific ShapeNet checkpoints only** |

Three concrete integration traps found by reading rather than assuming:

- **CGAL Region Growing's point-set defaults are absolute, not scale-relative**
  (`maximum_distance = 1`, `sphere_radius = 1`). On a metric survey cloud a
  default 1.0 m threshold is meaningless. Efficient RANSAC's defaults *do*
  self-scale to the bounding box (`epsilon` and `cluster_epsilon` at 1% of the
  diagonal, `normal_threshold` 0.9 ≈ 25°). The manual itself warns the defaults
  "do not necessarily guarantee to produce pleasant results."
- **`cluster_epsilon`'s connectivity constraint interacts adversarially with
  dropout.** CGAL keeps "only one connected component… the one covering the
  largest number of inliers." A dropout gap larger than `cluster_epsilon`
  silently discards the far fragment as unassigned or as a separate small
  shape. This predicts *coverage/fragment loss*; the doc gives no evidence
  either way about whether the retained fragment's axis is biased.
- **Neighbourhood radius too small doesn't over-segment — it collapses
  coverage.** CGAL's 2D 70k sweep: `sphere_radius=1.0` → 794 regions but only
  6.4% of points assigned, because regions never reach `minimum_region_size`
  and are discarded outright. Directly relevant to sparse/partial coverage on a
  defective cloud.

The map's up-front claim survives the reading: **there is no single library
that gives bounded box faces and a cylinder axis with confidence scores.** The
likely build is Open3D (normals, bounded planar patches) + CGAL or PCL
(cylinder) + our own extent extraction + our own confidence layer.

---

## 4. What transfers to our setting

### 4.1 A concrete stack

- **Bounded box faces, today, no extra code**: Open3D `detect_planar_patches`
  returns `OrientedBoundingBox` per patch. `.center` is the face aim point,
  `.extent` (x,y) is the face size for framing, the third column of `.R` is the
  orthogonal approach direction. That is *literally* the "scan face N
  orthogonally" action tuple. Kaiser supplies the algorithmic pedigree —
  bounded-patch trimming via grid + connected components + convex hull of
  inliers, attributed to Schnabel/Andrew — so this is a citable mechanism, not
  just "it's in the library we already use."
- **The same tool carries a structural false-positive mode.** RSPD (which
  `detect_planar_patches` implements) has an acknowledged, unfixed limitation
  (§5): a purely local planarity test cannot distinguish a locally-flat patch
  of a large-radius cylinder from a true plane, so it emits *spurious
  rectangular plane patches tiled along a cylinder's axis*. Measured cost on
  the two cylinder-heavy datasets: Plant precision 0.63, Boiler Room recall
  0.60, against 0.85/0.79 averages. For our rounded/chamfered objects this is a
  concrete failure to test for, and it argues for a curvature-consistency guard
  downstream.
- **Cylinder axis**: CGAL Efficient RANSAC or Region Growing (the latter
  deterministic, with a free `minimum_radius`/`maximum_radius` geometric
  plausibility gate that needs no learned component), or PCL. Extent and
  centroid must be computed by us from the assigned-point indices.
- **Refit after detection.** Schnabel §4.6: least-squares refit using points
  within 3ε (a wider band than detection's ε) cuts sphere radius/centre error
  by 5–25× at low-to-moderate noise. Cheap, worth adopting regardless of
  detector. Caveat: that reduction is measured in *position/scale*, not
  orientation — we cannot claim refitting fixes angular error from this paper.
- **Normals**: Sanchez's iterative robust PCA is the best-documented option —
  published error-vs-σ-vs-k curves against 8 baselines, ~0.2 ms/point,
  parallelisable, two hyperparameters (assumed sensor σ, minimum curvature
  radius R_min) with a closed-form derivation for the convergence threshold.
  Its refinement step cuts error **up to 90× when neighbourhood sampling is
  non-uniform across an edge** — which is exactly the anisotropy a fused
  multi-view cloud has. No public repo was found; treat as
  "reimplementable from the equations."
- **Plane-normal uncertainty**: Pathak's ALSP (Eqs. 19–21) is the only
  closed-form, non-iterative covariance on a fitted normal+offset in the whole
  branch. Built from nothing but the weighted centroid and weighted scatter
  matrix of the inliers, 5 ms per fit, within ~1.2% of the Cramér–Rao bound
  across a ±30° incidence sweep. It has a free self-test: the fitted [n̂; d]
  must lie in the covariance's null space (Eq. 22) — cheap to unit-test.
- **Cross-scan primitive matching**: SPFN's Relaxed-IoU + Hungarian matching
  (Eq. 1) is designed for exactly "these two primitive sets have no guaranteed
  a priori correspondence." It is a training-time device in SPFN; we would use
  it at inference across scans.

### 4.2 Reusable ideas, not code

- **Robust statistics over PCA.** RSPD's median + MAD (50% breakdown) instead
  of mean + covariance eigenvalue ratio (0% breakdown) is a general recipe for
  fitting under localised dropout/outlier clusters.
- **Over-generate then select.** MonteBoxFinder's contribution — sort all
  candidates by confidence, update *every* candidate's statistics each
  iteration (not just visited ones), no tree — is a clean, implementable answer
  to "we ran the detector with several parameter settings and now have
  overlapping hypotheses." The paper explicitly argues genericity beyond
  cuboids (asserted, not demonstrated).
- **Type-correctness and parameter-accuracy are separate signals.** SPFN's
  no-`L_axis` ablation: type accuracy 96.47% while axis error jumps to 6.27°.
  A system can report "cylinder, 96% confident" and still be 6° off. Any
  confidence we surface must not conflate the two.
- **Normals optimised for fitting ≠ normals optimised for angular accuracy.**
  SPFN substituted *more accurate* jet normals (6.87° vs 8.66°) and coverage
  *dropped* (76.47% vs 86.63%). This is a direct methodological caution against
  Branch 1 selecting a normal estimator on angular error alone.
- **A uniform model can compete as a hypothesis.** PEARL's outlier label ∅ has
  constant fidelity γ and competes per-point inside the same energy as the real
  models. This is the one classical precedent for our two-tier fallback
  *mechanism* — but it is per-point, not a per-object "abandon the
  decomposition" decision. Extending it to object level is our extrapolation.
- **Bias warning from the uncertainty literature.** Pathak §2.2: the
  theoretically "correct" full-ML objective (β=1) is *biased* under a
  range-dependent noise model; only the least-squares variant (β=0) is
  unbiased. If we build our own incidence-weighted fitter, the textbook choice
  is not automatically the safe one.

### 4.3 The precedent question — answered, positively, with caveats

**Yes, a published system decomposes a scanned object into geometric
primitives and generates per-primitive viewpoints.** Wang, Tong, Shang, Zhang,
*Hierarchical Viewpoint Planning for Complex Surfaces in Industrial Product
Inspection*, IEEE/ASME T-Mech 29(5):3289–3299, 2024.

Its architecture is our contract almost verbatim:

- Objective (Eq. 6): minimise viewpoint count, maximise pose quality by
  penalising deviation from ideal orthogonal normal alignment, subject to every
  point being covered — a set-cover formulation with an explicit orthogonality
  term.
- Point cloud segmented into **plane / sphere / cylinder / cone** plus a
  residual "irregular" class.
- **Bottom-up for regular primitives**: each primitive's *bounded* surface is
  gridded, with grid cell size derived from the camera's field-of-view and
  depth-of-field (Eqs. 8–9), and **one viewpoint generated per grid cell,
  orthogonal to the local surface by construction**. This is the branch's
  cleanest answer to the bounded-face question and to the primitive-level
  deterministic rules the framing doc blesses.
- **Top-down for irregular geometry**: a Hybrid Viewpoint Confidence
  Representation (Eq. 17) combining boundary-edge probability × local curvature
  (Open3D), covered-point count, and a viewpoint-normal · surface-normal term
  with a `max()` that zeroes out wrong-side views — then a dynamic NBV search
  and a FOV-shift pose optimisation.
- Real hardware, including a 6-DoF arm; a companion paper by the same lab
  (Machine Intelligence Research 22(4), 2025, open access) reuses this exact
  module and reports 25 viewpoints / 82.75 s / 96.45% coverage on 40 complex
  ABC clouds vs. 29–39 viewpoints and 83.95–92.80% for three baselines; and 11
  to 63 viewpoints for 100% coverage on five real objects.

Four caveats that shape what this means for our novelty claim:

1. **Their primitive segmentation is a trained network**, not RANSAC — the
   lab's own Probabilistic Boundary-Guided Point Cloud Primitive Segmentation
   Network (IEEE TIM 72, 2023). No RANSAC-vs-theirs ablation was found, so we
   cannot say *why* they judged off-the-shelf fitting insufficient, only that
   they did.
2. **Their answer to non-primitive geometry is a different algorithm, not a
   coarser fallback and not a richer vocabulary.** Irregular geometry is routed
   to confidence-driven NBV search. There is no failure detection, no
   uncertainty, no degraded-safe tier.
3. **Their acquisition setting is far cleaner than ours**: CAD-derived (ABC),
   an unspecified "Synthesis" set, T-LESS, and turntable captures on a
   purpose-built rig with industrial lighting and a dedicated positioning
   camera. No defect-mode analysis of any kind appears anywhere.
4. **Sourcing is weak.** IEEE Xplore is paywalled; the reading note is built
   from the abstract, a third-party Chinese section-by-section walkthrough, and
   the open-access companion paper. **Not one number from the target paper's
   own tables was ever read.** Confidence in the structure is good; confidence
   in any specific reported result is low.

Against this, Kaiser's survey confirms the negative for the classical era:
across ~50 surveyed methods, **no method generates viewpoints or camera poses
from a primitive decomposition**. The nearest neighbours are
primitive→navigation-graph (robotics free-space culling) and
primitive→relation-graph (GlobFit's coplanarity/coaxiality/orthogonality graph;
Schnabel's own follow-up topology graph). Every other paper in this branch has
zero view-planning content.

**Net: our contribution is not "decompose then plan views."** That exists. It
is (a) doing it from a defective survey cloud with an explicit failure tier,
and (b) exposing the result as a VLM-navigable action space.

---

## 5. What is contested — contradictions kept visible

These are not smoothed. Each is a real tension between sources read this round.

**C1 — How robust is Efficient RANSAC, really?**
Kaiser quotes Schnabel's "20% Gaussian noise, 95% outliers" as the field's
headline robustness figure. SPFN measures the CGAL implementation of the same
algorithm at **52.92% type accuracy and 7.54° axis error** on clean synthetic
CAD at matched resolution, and 43.90% type accuracy even at 8× the point count.
The two are an order of magnitude apart in implied competence. They are not
strictly incompatible — Schnabel's number answers "does *a* correct shape get
found on a synthetic sphere octant," SPFN's answers "does the *right* shape get
found with precise parameters across 3,366 mechanical parts" — but the gap
matters enormously for whether "just run CGAL RANSAC" is a plausible plan. Note
also that Schnabel's is a single anecdotal operating point, not a curve, and
that Kaiser's Table 1 does not even list noise robustness in RANSAC's strengths
row.

**C2 — Region growing under incompleteness: bad or good?**
Kaiser states region-growing methods "do not perform well when data is
missing." Fit4CAD's measurement agrees on *its* region-growing pipeline (PG
shows "more prominent variation" under missing data; HT "hardly affected").
But CGAL's own head-to-head on a real scene has **region growing at 100% of
shapes detected vs Efficient RANSAC at 78%**, at 2× the runtime. Both can be
true — region growing wins on completeness on well-sampled data and loses under
holes — but the crossover point is unmeasured anywhere, and it is exactly the
regime our dropout puts us in.

**C3 — Do learned methods beat RANSAC at primitive fitting?**
SPFN: decisively yes on ABC-style CAD (axis 1.51° vs 7.54°, surface coverage
86.63% vs 43.42%), and the gap survives handing RANSAC the network's own
segmentation, normals *and* type. SHREC2022: pure neural regression is the
**worst** fitter in the field (M3 median MFE 4.49e-2, ~10–20× worse than
hybrids) despite the best classification on the same backbone; the two most
reliable methods end-to-end are one *direct* method (accelerated Hough) and one
hybrid (neural guess → least-squares polish). Both are true and the synthesis is
architectural, not tribal: **use learning for segmentation, type and
initialisation; use algebra for parameters.** SPFN's own DPPN ablation proves
the same point internally (type accuracy 51.33% vs 96.93%).

**C4 — What does "robust to real-world data" mean in the abstraction papers?**
SuperFlex's abstract claims robustness to partial real-world point clouds. The
quantitative evidence is entirely **synthetic** ASE occlusion (hidden-point
removal + random spherical masking) on CAD furniture; ScanNet++ and Replica
appear only as qualitative figures because no ground-truth decomposition exists
for real scans. SuperDec's real-data Table 2 is real (ScanNet++/Replica,
3–11× better Chamfer than baselines) but **only measures Chamfer distance** —
it never scores whether the primitive type or count is semantically right. The
map's rationale ("SuperDec → ScanNet++, SuperFlex → partial real") is optimistic
on both counts. "Works on real scans" in this literature means "reconstructs
surface geometry to low Chamfer distance," not "recovers the correct primitives."

**C5 — Does the four-primitive vocabulary close?**
SPFN: plane+sphere+cylinder+cone covers **94.0% of average per-model surface
area** on ANSI mechanical CAD parts. Kaiser: the field added a fourth category
(ellipsoids, tori, general parallelepipeds) *precisely because real parts need
it*, and answers organic/chamfered geometry with generalized cylinder
decomposition. SHREC and Fit4CAD both include torus as the intended answer to
fillets — and both find torus among the two hardest classes (TPR down to 0.583
in Fit4CAD, 0.7297 in SHREC). SuperFlex's own ablation says bending+tapering
deformations buy only **2–4 IoU points** over rigid superquadrics, versus 13
points from fixing the loss function — soft evidence that most everyday objects
are close to rigid-convex-decomposable. EMS silently **excludes 27–33% of KIT
and BigBIRD objects** as not single-superquadric-approximable and names
thin-walled non-convex objects (a cup) as poorly modelled by the family itself.
**And no source anywhere compares recovered-normal accuracy across
vocabularies.** The question is not settled by this literature; the honest
position is that vocabulary choice is currently a guess.

**C6 — MonteBoxFinder's premise was misread by the map.**
The map framed it as evidence that "RANSAC + relationship refinement works only
on very clean scans." That sentence is MonteBoxFinder's claim *about GlobFit*,
not about itself. Its actual position is the opposite: RANSAC **over-generates**
on noisy real scans (880 cuboids + 174 thin cuboids per ScanNet scene, mostly
false positives) and this is survivable by adding a combinatorial selection
stage. This flips the risk framing from "RANSAC fails on our data" to "RANSAC
over-produces and needs a selection stage." But MonteBoxFinder never measures
**recall of the true candidate into the pool** as a function of noise — filtering
cannot recover a candidate RANSAC never proposed, and that silent failure mode
is unaddressed.

**C7 — GNC does not give us a certificate.**
The map selected it for "a certificate of solution quality… to gate the
fallback." The paper's own abstract states *"Although GNC's global optimality
cannot be guaranteed…"* and its conclusion lists optimality conditions as future
work. The one certificate present is for an SOS relaxation of the *shape-
alignment inner subproblem* with fixed weights — not transferable to primitive
fitting. TEASER++ (a different, later paper) is where to look if a certificate
is genuinely needed. Independently, GNC does no multi-model segmentation at
all: it robustifies one global model per call given correspondences, which is
not the hard part of our problem.

**C8 — "Parameter-free" claims don't survive reading.**
RSPD's abstract says "virtually independent of parameter tuning." The method has
four global magic constants (75° planarity angle, 60° max normal deviation, 25%
outlier fraction, ε = 0.1% leaf size), tuned once across seven
architectural/industrial datasets, none of which resembles a tabletop object at
D405 standoff. MonteBoxFinder has the same pattern (α,β,γ = 0.3, 0.7, 0.025
fixed across all 1613 ScanNet scenes; adaptivity proposed but not implemented).
Schnabel's ε and α are hand-set per model across all nine examples. **The
classical methods substitute global constant tuning for training data; they do
not eliminate the tuning problem.**

**C9 — Residual-based confidence is demonstrably insufficient.**
SHREC2022 §4.2 documents a named failure: near-degenerate cone fits where the
half-angle collapses toward zero and the vertex flies far from the object.
M2 produced 11 such outliers, M6 produced 30; **mean L2 parameter error reaches
1.21e+11 while the point-to-surface residual can still look low.** This is a
concrete, citable case where MFE alone fails to flag a catastrophically wrong
fit. Set against Raffo's and Fit4CAD's use of MFE as *the* model-selection
criterion, and against every "just threshold the residual" instinct.

---

## 6. What is missing — the census

Ordered by how much it hurts us.

1. **Any angular-error-vs-noise curve under a sensor-realistic noise model.**
   Zero. Every number in §2.1 is under isotropic Gaussian, uniform coordinate
   noise, or along-normal uniform noise on complete synthetic surfaces.
2. **Registration/alignment covariance propagated into fitted-primitive
   orientation.** Pathak explicitly assumes R,t known and adds zero uncertainty
   for the registration step when fusing planes across views. M3C2-EP (map #27)
   is the only candidate and was not read this round.
3. **A closed-form covariance for a cylinder axis.** The one lineage that has
   it for planes states in its own follow-up (Vaskevicius et al., quadrics,
   ICRA 2010 / PhD thesis Ch. 3, cross-checked in the reading) that
   *"analytical expressions for the Hessian cannot be obtained for the quadric
   case."* Cylinders and cones need iterative ML refinement with a numerically
   evaluated Hessian. **Documented negative, from the research lineage itself.**
4. **Any re-scan repeatability experiment.** Zero, across all 20 documents.
   And the available evidence points negative (see §7, RQ3).
5. **Whether directional dropout biases a fitted primitive or merely inflates
   variance.** CGAL's connectivity mechanism predicts fragment/coverage loss;
   nobody measures whether the *retained* fragment's axis is systematically
   tilted. This is the single most dangerous open question in the branch,
   because a systematic tilt is invisible to residual-based confidence.
6. **Any calibration of a fit score against downstream task failure.** Zero.
7. **Any recovered-normal-accuracy comparison across primitive vocabularies.**
   Zero.
8. **Any evaluation on a small tabletop object scanned close-in by eye-in-hand
   active stereo.** Nearest are EMS's KIT (Minolta laser) and BigBIRD
   (PrimeSense Carmine) at ~1000–1500 points per object. `find-superquadric`
   (map #20, partial cloud + table plane prior) is the closest deployment match
   in the whole map and was **not read**.
9. **Any symmetry-based completion for a partially observed cylinder.** Absent
   from all 20 documents. Map #35 unread.
10. **Any preprocessing-effects study on primitive recovery.** One data point
    exists, and it is at the normals stage and negative: Sanchez shows a
    bilateral filter "tend[s] to corrupt the accuracy of the other normals" and
    produces >10° errors on a non-trivial fraction of points **even at zero
    noise** — i.e. it smooths points that plain PCA got exactly right.
    Schnabel adds the related tradeoff: enlarging the normal-estimation radius
    to fight noise pulls in points from adjoining shapes and destroys small
    shapes' normals.
11. **The entire sensor side** (map section E, 8 documents) and the entire
    failure-detection/introspection side (section F, 2 documents). Unread.

---

## 7. Implications per research question

### RQ1 — Judging a view (orthogonality scoring)

**What we can hand Branch 1:**

- An *achievable-accuracy ceiling*, not a prediction: 0.3–2° normal error for
  well-sampled flat regions under idealised isotropic noise (Sanchez); 8.66°
  per-point normal error and 1.51–2.42° axis error on complete synthetic CAD
  (SPFN); 7.0–7.5° axis error for Efficient RANSAC on the same data even with a
  perfect segmentation handed to it. **Every one of these is a lower bound on
  our error, not an estimate of it.**
- A neighbourhood-size warning: PCA, 2-jets and bilateral-filtered normals have
  a *sweet spot then degrade* shape in k (best ~150–200, worse after), while
  robust IRLS-PCA improves monotonically with k. "Just use more neighbours" is
  unsafe for the estimator Open3D ships.
- The one closed-form path to "orthogonal ± σ": Pathak's ALSP, for **planes
  only**, non-iterative, 5 ms, with a free null-space self-test. It requires a
  σ(range, incidence) model for the D405 that we do not have and that section E
  of the map was supposed to supply.
- A concrete, non-learned, computable view-quality score with production
  pedigree: HVCR (T-Mech 2024, Eq. 17) = boundary-edge probability × local
  curvature (Open3D-computable), plus covered-point count, plus
  viewpoint-normal · surface-normal with a `max()` zeroing wrong-side views.
  Usable independent of the rest of that paper's machinery.

**What Branch 1 must not do:**

- Do not adopt the field's correctness bar. Araújo & Oliveira count a detection
  as correct at **≤30° normal difference**. That is the literature's threshold
  for "the plane was found," not an orthogonality budget. An "evidence view
  orthogonal to the inspected surface" needs a bar we set and defend, and the
  angular-error budget question (map Q10) cannot be answered until Branch 1's
  depth-error-vs-incidence-angle numbers land.
- Do not select a normal estimator on angular error alone. SPFN substituted
  *more accurate* jet normals and downstream coverage *fell*. Benchmark
  estimators on the fitted-primitive outcome, not the intermediate.
- Do not treat type confidence as parameter confidence. SPFN's no-`L_axis`
  ablation: 96.47% type accuracy with 6.27° axis error.

### RQ2 — Selecting best views (shape → views)

**Feasibility verdict: proceed, but instrument.** The blockers the branch was
opened to find are *not* runtime, vocabulary availability, or library support.
They are:

- **Validation gap.** Nothing in this literature evaluates decomposition under
  registration jitter, view-dependent dropout, or systematic depth offset. Our
  ~4 cm fused-vs-single-view offset has no analogue anywhere in the 20
  documents. This must be measured on our own objects; the literature will not
  supply it.
- **Bounded faces are the pinch point.** Only Open3D `detect_planar_patches`
  (planes) and MonteBoxFinder's `Cuboid` (boxes, via CGAL + selection search)
  return a face centroid + extent natively. **CGAL returns infinite primitives
  for all five types — verified accessor by accessor.** For cylinders nothing in
  the toolchain gives extent; we write it, using Kaiser's citable recipe (grid +
  connected components + convex hull of inliers in the shape's parameterisation).
- **The tool we would reach for first has a structural failure on the geometry
  we care about.** `detect_planar_patches` (= RSPD) will tile a large-radius
  cylinder with spurious rectangular plane patches. Acknowledged by the authors,
  unfixed, and directly relevant since half the contract is cylinder-orbit.
  A curvature-consistency guard is required, not optional.
- **Partial cylinder is the least-supported case.** Best available
  incompleteness evidence is EMS's 86.4% global-optimality at 20% surface
  coverage (superquadric family, contiguous synthetic cap) and Raffo's ~1/8
  coverage working for a *coarse initial estimate* at low noise. Nobody tests a
  half-scanned cylinder with one-sided glossy dropout, and symmetry completion
  is absent from everything read.

**Design consequence for the analytical track**: the bottom-up path in Wang et
al. (bounded primitive surface → grid by camera FOV/DoF → one orthogonal
viewpoint per cell) is a validated template for exactly the primitive-level
deterministic rules the framing doc blesses. Adopt its *shape*; do not assume
its numbers, and note that they used a trained segmenter where we intend to use
off-the-shelf fitting.

### RQ3 — Composing the action space

**This is where the branch found the sharpest risk, and it is a risk nobody has
measured.**

RQ3's consistency requirement — "does face 3 stay face 3, in identity, ordering
and orientation" — has **zero supporting evidence and several structural
counter-indications**:

- CGAL states plainly that Efficient RANSAC is *"randomized and so the output
  may vary at each run"* — on identical input, identical parameters. And the
  API exposes **no random-seed parameter anywhere**, so it cannot be worked
  around through the documented interface. `shapes()` returns results "in the
  order of detection… ordered with decreasing size" — order is a *consequence*
  of a randomized search, not an identifier.
- Every learned method uses unordered slot prediction. SuperDec and SuperFlex
  resolve slot correspondence by Hungarian matching **at training time only**;
  nothing enforces that primitive index 3 is the same physical part across two
  scans. CSA's claimed fixed-slot correspondence is across *different instances
  of one trained category*, never across re-scans of one object.
- PEARL and Multi-X are stochastic proposal + local-minimum methods by
  construction, with no persistent identity.
- Not one paper in the branch runs the experiment.

**Design consequences:**

1. **Address the action space by geometric content, never by extraction
   index.** A face must be identified by its normal direction (relative to a
   stable frame, e.g. table plane + gravity), its area, and its position — not
   by "the third thing the detector returned." This is a hard design
   constraint, derived from a documented library property, not a hunch.
2. **Prefer deterministic detectors where identity matters.** CGAL Region
   Growing is deterministic given identical input ordering; RSPD is
   deterministic; EMS needs one initialisation. Efficient RANSAC is not.
   Determinism on identical input is necessary but *not sufficient* — it says
   nothing about stability across two different scans.
3. **Build a matching layer.** SPFN's RIoU + Hungarian matching over point
   membership, or nearest-neighbour matching on returned primitive parameters,
   is the mechanism. It is a layer we write.
4. **Measure it.** "Re-scan the same object N times, match primitives, report
   count stability / parameter dispersion / identity flip rate" is a
   first-class experiment for our own benchmark, because no published protocol
   exists.

On the positive side, the *skeleton* is sound: bounded-primitive representations
(superquadric pose + half-extents; cuboid t+s+R; Open3D OBB center+extent+R) are
all strictly richer action-space skeletons than infinite primitive equations,
and typical primitive counts are small — SuperDec 5.18–6.58 per real object,
SuperFlex 4.3–6.3, CSA 8.5–9.75 cuboids, EMS one-or-few. A 4–8-primitive
decomposition is a tractable structure to address.

### RQ4 — Presenting to the VLM

No document in this branch touches VLMs. What the branch contributes is the
*content and size* of the payload:

- A primitive is a small typed tuple: `(type, centroid[3], extent[3],
  axis/normal[3], coverage_fraction, mean_residual)` — roughly 10 numbers plus a
  label. At 4–8 primitives per object that is a far smaller text payload than
  enumerating a 12×3 viewsphere, and it is *semantically* addressable ("the
  large flat face pointing away from the table") in a way a `{h,v}` cell is not.
- The confidence numbers we can attach are **uncalibrated**. Coverage fraction,
  residual RMS, MND, EMS posterior — none has ever been mapped to task success,
  in any of the 20 documents. If they are surfaced to the orchestrator they must
  be framed as qualitative hints, not probabilities, and the prompt design must
  say so rather than imply otherwise.
- The two-tier fallback (decomposition, else uniform sphere) has exactly one
  classical precedent for its *mechanism*: PEARL's constant-fidelity outlier
  label ∅ competing as a hypothesis inside the same energy. It is per-point,
  not per-object. Wang et al.'s system — the closest full precedent — has **no
  failure tier at all**: irregular geometry gets a different algorithm, not a
  degraded-safe one. We are on our own for the gate design, and section F of the
  map (introspective perception, safety monitoring) was not read.

---

## 8. Implications for the benchmark (Branch 2)

### 8.1 What is directly reusable

- **Fit4CAD's metric suite and ground-truth schema.** Segmentation quality via
  point-membership TPR/TNR/PPV/NPV/ACC/DSC; approximation via Mean Fitting Error
  (point-to-surface distance normalised by bounding-box diagonal) and directed
  Hausdorff; a coefficient distance in normalised implicit-polynomial space. The
  four-file GT schema (points; point-index → primitive-ID; parametric tuple —
  which *embeds the axis/normal vector*; implicit polynomial) is a ready
  template. Normalising all distance metrics by the bbox diagonal is a clean
  scale-invariant convention worth copying.
- **SHREC2022's protocol and public assets** (github.com/chiararomanengo/SHREC2022)
  as an evaluation-harness skeleton, plus its A0–A9 perturbation taxonomy as a
  *shape* to extend: clean / uniform / Gaussian / undersampling / missing-parts /
  four combinations / localised deformation.
- **Pathak's ε1/ε2/ε3** (Mahalanobis distance to the true covariance,
  weighted angular error of principal uncertainty directions, relative total
  uncertainty against the Cramér–Rao bound) as a ready-made validation metric
  set for any normal-uncertainty estimator we build — independent of
  Fit4CAD/SHREC/ABC.
- **Araújo & Oliveira's scoring-fairness step**: voxel-regularise the cloud
  *for evaluation only* to remove sensor-proximity oversampling bias, then match
  detections to ground truth on ≥50% voxel overlap plus an angular tolerance.
  The mechanism transfers even though their 30° tolerance does not.
- **Runtime envelopes**: Fit4CAD's HT worst case (358 s per CAD object) is a
  workable per-object budget; PG's 5.3 h worst case is the cautionary anchor for
  naive curvature-region-growing at scale.

### 8.2 What we must build because it does not exist

1. **An angular-error metric.** No benchmark in this literature reports one.
   Fit4CAD's GT *contains* the axis vectors; the extraction procedure (compare
   fitted axis to GT axis from the parametric tuple) transfers, the numbers do
   not. This is the single most important addition, because the whole contract
   turns on orthogonality.
2. **A corruption suite matching our defect triad.** Every published corruption
   model is sensor-agnostic. Ours needs: (a) **registration jitter** injected as
   a per-view rigid perturbation *before* fusion, not as per-point noise after —
   this is the structural difference that makes existing benchmarks
   inapplicable; (b) **view-dependent dropout** keyed to incidence angle and
   simulated material, not an isotropic spherical hole; (c) a **systematic
   per-view depth offset** to reproduce the fused-vs-single-view discrepancy.
   Map #30 (structured-light stereo noise model) and #33 (domain-randomised
   depth simulation) are the intended sources and were not read.
3. **A re-scan repeatability protocol.** No published protocol exists. Metrics:
   primitive count stability, parameter dispersion across scans, identity flip
   rate after matching. This is a Branch 3 → Branch 2 hand-off item.
4. **A fit-score → task-failure calibration experiment.** Confirmed absent
   everywhere. The minimal version: for each primitive fitted from a corrupted
   cloud, compute every available score (coverage, residual RMS, MND, EMS
   posterior, radius plausibility), derive the orthogonal viewpoint from that
   primitive, and measure the angular error of that viewpoint against the CAD
   ground truth. The output is a score → error mapping, and a threshold with a
   measured false-accept/false-reject rate. **Without this the two-tier fallback
   gate is a guess.**

### 8.3 Ground truth — and a circularity trap

The field's accepted substitute for CAD is **human-annotated segmentation**:
Kaiser §6.1 (segmentation correctness "require[s] ground truth information…
prior manual and user-assisted work"), Araújo & Oliveira §4.1 (interactively
hand-selecting inlier points per plane), PEARL Table 1 (human-clicked
intersection lines on a real scene), Schnabel §5.2 (the USF manually-annotated
range-image benchmark), Multi-X (points manually assigned to signs and
balusters). This is cheap enough to be viable at our scale.

**But watch the circularity.** Araújo & Oliveira build their ground-truth plane
parameters by fitting hand-selected inliers *with the same median-position /
FAST-MCD-normal recipe their detector uses*. A benchmark built that way cannot
detect its own fitter's systematic bias — which is precisely the failure mode we
most need to catch. This is a concrete instance of Branch 2's standing
judge–benchmark circularity question, and there is a second: Raffo's M1 and
Fit4CAD's HT both *select the primitive type by lowest MFE*, and MFE is also the
reported fitting metric. A method that picks the type minimising the metric will
score well on that metric.

The escape is a CAD model (or a metrology-grade reference scan) for at least a
calibration subset of test objects. Note that both SHREC2022's authors (2022,
§5) and SuperFlex's (2026) name real-scan ground truth as **unsolved future work
on their own roadmaps**. We should not expect to inherit it.

### 8.4 Failure-mode test cases the literature hands us for free

Every one of these is a documented, named, reproducible failure that our
benchmark should contain as an explicit case:

- **Cone near-degeneracy** — half-angle collapses toward zero, vertex flies far
  from the object, L2 parameter error reaches 1e11 while the point-to-surface
  residual stays low (SHREC2022 §4.2: M2 11 outliers, M6 30 outliers).
- **Cylinder ↔ cone confusion under noise** (Raffo, ABC at σ=0.50).
- **Spurious planar patches tiled along a large-radius cylinder** (RSPD §5,
  acknowledged and unfixed; inherited by Open3D `detect_planar_patches`).
- **Wrong primitive count as a discrete catastrophe** — PEARL's bimodal error
  histogram (Fig. 22): correct count → 5–20% error; wrong count → a separate
  40–70% mode. Count errors are not graceful degradation.
- **Type-correct but axis-wrong** (SPFN no-`L_axis`: 96.47% type accuracy at
  6.27° axis error).
- **Dropout gap larger than `cluster_epsilon`** → the far fragment is silently
  dropped, unflagged (CGAL's documented connectivity mechanism).
- **Denoising over-smoothing edges** — bilateral filtering produces >10° normal
  errors on points that plain PCA got right, at *zero* noise (Sanchez §6.1.1).

---

## 9. Recommended next actions from this branch

1. **Read map section E (8 documents) and F (2 documents).** The branch's
   central question — does decomposition survive *our* defects — cannot be
   closed without the sensor characterisation and dropout-physics literature.
   Priority within E: #30 (structured-light stereo noise model, needed for the
   corruption suite), #32 (flange-based hand-eye calibration, the candidate
   explanation for the 4 cm offset), #28 (D415 metrological characterisation,
   which measures against known planes/cylinders/spheres — the primitives we
   intend to fit).
2. **Read #27 (M3C2-EP)** — the only candidate in the whole map for propagating
   registration covariance into geometry, and the one thing that could turn our
   4 cm offset from an unmodelled hazard into a number.
3. **Read #20 (`find-superquadric`)** — a robotics-grade superquadric fit from a
   *partial cloud on a table* with a table-plane prior and density-clustering
   outlier removal. The closest deployment match in the map, and cheap to trial.
4. **Read #18 (Point2CAD)** and **#35 (robust extrinsic symmetry)** — the two
   remaining candidates for bounded faces via clipping, and for partial-cylinder
   axis recovery.
5. **Run the two-hour empirical check** rather than reading further on
   integrability: Open3D `detect_planar_patches` + CGAL/PCL cylinder on one of
   our actual fused PLYs, and EMS (`.ply` in, ~90 ms) on the same. This answers
   more of the integrability question than any further paper will.
6. **Chase the T-Mech 2024 paper properly.** It is the branch's single closest
   precedent and not one number from its own tables has been read. If
   institutional IEEE access is available, this upgrades every claim in §4.3
   from third-party summary to primary source.

---

## 10. One-paragraph answer to the critic mandate

*Does decomposition survive our defect profile?* **Unknown, and the field
cannot tell us.** Decomposition into plane/cylinder/sphere/cone is available
off the shelf, runs in seconds at our cloud sizes, and has one strong published
precedent for driving per-primitive orthogonal viewpoints (Wang et al., T-Mech
2024). But every robustness number in the literature is measured under
isotropic Gaussian noise, random outliers, or geometric hole-punching; not one
paper models registration/fusion error, view-dependent glossy dropout, or
systematic depth bias; no detector outputs uncertainty (the field's own survey
names this as an open problem); no benchmark reports an angular error on a
normal or an axis; no paper has ever run a re-scan repeatability experiment;
and the confidence signals that *are* available are all residual-family
statistics, which are structurally blind to exactly the systematic-bias failure
our 4 cm fused-vs-single-view offset represents. The branch's honest output is
therefore not a robustness estimate but a **measurement plan**: build the
decomposition on Open3D + CGAL, add the extent extraction and the cross-scan
matching layer the libraries do not provide, and make "angular error of the
derived orthogonal viewpoint under our corruption suite" the first number our
benchmark produces — because it does not exist anywhere else.
