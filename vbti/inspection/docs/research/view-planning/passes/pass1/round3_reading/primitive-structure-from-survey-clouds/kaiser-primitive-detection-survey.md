# A Survey of Simple Geometric Primitives Detection Methods for Captured 3D Data

**Source:** Kaiser, Ybanez Zepeda, Boubekeur, *Computer Graphics Forum* 38(1), 167-196,
2019 (also presented as a Eurographics 2019 STAR). DOI `10.1111/cgf.13451`. Wiley page
(`onlinelibrary.wiley.com/doi/10.1111/cgf.13451`) is paywalled to abstract-only. Full text
obtained from the authors' own open-access mirror: the paper's landing page is linked
from Tamy Boubekeur's homepage publication list (`perso.telecom-paristech.fr/boubek/`,
entry for 2019, `papers/GeoPrimFitSurvey/`), which hosts the authors' PDF directly
(`papers/GeoPrimFitSurvey/GeoPrimFitSurvey.pdf`, "authors' draft" per page footer, 30 pp,
final version in CGF). All page/section/table/figure numbers below refer to this PDF.
**Type:** survey / State-of-the-Art Report (STAR) — taxonomy and classification of prior
methods, not a method paper and not an experimental study. No new experiments, no new
error curves; its numeric content is limited to what it quotes from the surveyed papers
(mostly qualitative "handles up to X% noise/outliers" claims, not curves).

**Verification note (this pass):** the authors' PDF link above was independently
re-fetched (`curl`, HTTP 200, `application/pdf`, 24.8 MB, 30 pages per `pdfinfo`,
last-modified 2018-07-03 — consistent with "authors' draft") and converted with
`pdftotext -layout` to spot-check the claims below directly against source text: Table 1
(framework strengths/weaknesses), the RANSAC/Hough/Clustering "Input: List of vertices
and associated normals" algorithm headers, the 20%-noise/95%-outliers quotes (Sec 4.2.4,
p.15), the "Robustness to Incomplete Data" paragraph, Sec 6.1-6.3 (evaluation
methodology, fitting-error/Hausdorff formulas, the QEM/SQEM "does not make much sense to
evaluate... the output" caveat), and the Sec 7.3 "Interpretation" quote on primitive
uncertainty. All matched verbatim. **One error was found and is now corrected below:**
an earlier draft of this note claimed CGAL is never mentioned in the document; Table 5
(p.20) in fact lists "Fast RANSAC (CGAL)" as a distinct implementation row alongside
"Fast RANSAC (original)" and "Variants of RANSAC (PCL)" — see the Tooling section.

## What this document actually is, and the central caveat for our branch

This is the taxonomy backbone the branch chose it for — but it is a **2019** survey
whose own reference list runs 1972-2016 (newest entries [ZXTZ15], [MKRVG15], [CLW*14],
[KL15] are 2014-2016; abstract mentions "the rise of single-view 3D capture based on
machine learning" as a 2019-era future direction, not yet covered). Consequently it
predates, and says nothing about: superquadric/superellipsoid recovery via deep learning
(EMS, SuperDec, SuperFlex — all 2022+), learned failure/confidence scores (MonteBoxFinder,
GNC/TEASER certificates — 2022+), modern point-cloud benchmarks (Fit4CAD, SHREC 2022, ABC
— ABC itself is 2019, just after this survey's cutoff), and normal-estimation error models
(PCA/jets/VCM angular-error-vs-sigma curves are absent entirely — see below). For those
RQs this document is a **negative source**: it establishes the taxonomy and the
classical-era ceiling, and explicitly names some of these as *open* research challenges
in its own concluding section (Section 7.3, p.21) rather than closing them.

## Core structure of the paper

- **Sec 2 (Background):** input data types (depth image, image sequence, point cloud,
  mesh — Sec 2.1, p.3-4) and the definition of "simple geometric primitive" (Sec 2.2,
  p.4-5).
- **Sec 3 (Theoretical Foundations):** three families — stochastic (RANSAC, local
  statistics), parameter-space (Hough transform, parameter-space clustering), clustering
  (region growing, Lloyd-based automatic clustering, segmentation-then-fitting). Table 1
  (p.5) and Figure 4 (p.5, radar chart) compare these seven frameworks on Speed,
  Repeatability, Simplicity, Generality, Robustness to outliers/noise/incomplete data,
  Meaningful segmentation, Data fidelity, Scalability.
- **Sec 4 (Characterization):** the classification axes used throughout — Context
  (detected primitives, application context: indoor/outdoor/individual-object) and
  Properties (Accuracy: Data Fidelity Table 2 + Abstraction Level Table 3; Practicality:
  timing, scalability, user assistance, learning phase, intuitive tuning, temporal
  consistency; Information: semantics, meta-data; Robustness: noise, incomplete data,
  outliers).
- **Sec 5 (Methods and Applications):** ~50 methods surveyed, organized by primitive type
  (planes; boxes/cuboids; spheres/cylinders/cones; ellipsoids/tori/parallelepipeds) crossed
  with application context (indoor / outdoor / individual objects).
- **Sec 6 (Metrics and Evaluation):** fitting-error metrics, processing metrics, available
  code (Table 5) and datasets (Table 6).
- **Sec 7 (Discussion):** concluding remarks, "Towards Spatial Reasoning" (Sec 7.2), and
  Research Challenges (Sec 7.3) — Interpretation, Improvement of Processing, Connection to
  Deep Learning.
- **Table 4** (p.19, "Comparison of presented algorithms") is the master lookup table: one
  row per method with all classification axes as columns. **Table 7** (pp.23-25) is a
  visual compendium — one thumbnail image per method, useful for quickly recognizing what
  each cited acronym actually looks like.

## Definition of "simple geometric primitive" (Sec 2.2, p.4) — answers the vocabulary RQ

A simple 3D primitive is defined by: (1) fixed and limited number of global intrinsic
parameters (size, orientation, position only — except the torus), (2) convex except the
torus, (3) symmetric, (4) a basic shape assembled with others to build complex shapes
(explicitly the CSG view, citing Foley 96 [Fol96]). Four categories, increasing parameter
count:
- **planes** — normal **n** + distance d;
- **cuboids and boxes** — center **C**, orientation **n**, three side lengths (also
  representable as 8 vertices);
- **spheres, cylinders, cones** ("natural quadrics" per Hakala et al. [HHNM80]) — sphere:
  center + radius; cylinder: axis point **C**, radius r, axis direction **n**; cone: apex
  **C**, axis **n**, half-angle α;
- **other: ellipsoids, tori, non-rectangular parallelepipeds** — ellipsoid: center, axis,
  3 semi-axes a,b,c; torus: center, axis, minor/major radii m,M; general parallelepiped:
  center, orientation, 3 side lengths, 3 interaxial angles.

**Explicit exclusions the survey states** (p.4-5): conservative bounding envelopes (AABB,
OBB, k-DOP), CAD-specific primitives (rolling-ball blends [FS96]) are out of scope — the
survey's interest is fitting primitives to *surfaces*, not building bounding volumes.

**Answer to "does the object set close with plane+cylinder+sphere+cone, or do real
chamfered/rounded parts force more":** the survey's own taxonomy already goes beyond the
four-primitive vocabulary — cuboids/boxes are a first-class separate category (not
decomposed into 6 planes in the classification, though they can be built from planes
algorithmically), and a fourth bucket (ellipsoids, tori, general parallelepipeds) exists
precisely because real parts need it. Sec 5.4 "Ellipsoids, Tori, Parallelepipeds" (p.18)
is explicit that ellipsoids model "organically-shaped meshes" and are "quite different
from the original model" even so ("results are quite different from the original model,
this allows very light modeling of complex meshes"); full tori appear in mechanical parts
while **partial tori model blends or elbows in pipes** — i.e., rounded/chamfered
mechanical features are handled in this literature via *tori*, not superquadrics (which
this survey never mentions — the term "superquadric" does not appear anywhere in the
text; the closest concept is "non-natural quadric surfaces," used only to describe
general second-degree implicit surfaces fit via quadric-error-metric-style least squares,
e.g. [YWLY12] "Quadric Surface Fitting," not superellipsoid shape recovery). Also
directly relevant: **Generalized Cylinder Decomposition** [ZYH*15] (cited repeatedly, Sec
5.3 p.17, Sec 7.1 p.20-21) segments a mesh into a set of *warped* (non-circular,
non-straight-axis) cylinders via region growing — this is the survey's own answer to
"what buys more than the four rigid primitives" for organic/chamfered shapes, and it is
explicitly framed as trading simplicity for fidelity ("This type of primitives allows
modeling faithfully trivial objects from our daily lives that cannot be constructed only
with the simpler primitives, while keeping a rather simple set of parameters," p.21-22).
**No recovered-normal-accuracy comparison between vocabularies is given anywhere** — the
survey never quantifies how much normal/pose accuracy is bought or lost by choosing a
richer primitive set. That comparison is not in this literature generation at all
(negative finding).

## Bounded faces vs. infinite primitives — "Primitive Shape Trimming" (Sec 2.2, p.5)

Directly answers the RQ "which methods recover bounded faces (extent, boundary,
centroid)." Planes, cylinders, and cones are *infinite by definition* (spheres, cuboids,
boxes, parallelepipeds, ellipsoids are already bounded/full-extent). The survey names
three trimming strategies used in the literature to bound the infinite ones:
1. **Regular grid + connected components on the shape's surface**, then **convex hull of
   inliers in shape space** to define the patch boundary — attributed to Schnabel et al.
   [SWK07] (Efficient RANSAC), citing Andrew's convex hull algorithm [And79] for the hull
   step.
2. **Fit smaller shape patches and merge them into the final model** — [BV11] (Plane
   Filtering).
3. **Region growing directly on the surface** — Feng et al. [FTK14] (Agglomerative
   Clustering / "fast plane extraction in organized point clouds using agglomerative
   hierarchical clustering").

The survey adds an explicit caution (p.5): trimming curves should stay simple (rectangles,
circles, regular convex polygons) "in the spirit of shape approximation" — i.e., don't let
the boundary-recovery step reintroduce the complexity the primitive was meant to remove.
This is directly actionable for our "face centre + face size" requirement: Efficient
RANSAC's convex-hull-of-inliers approach is the citable mechanism, and it is the same
family Open3D's `detect_planar_patches` implements (cross-ref: sibling note in this
folder, `open3d-geometry-api-docs.md` — Open3D's bounded-patch output is architecturally
the same idea this survey attributes to Schnabel/Andrew, though the survey predates and
does not mention Open3D specifically).

## Quantitative robustness numbers actually stated (the closest this paper gets to "a number")

These are the only calibrated noise/outlier numbers anywhere in the document, both about
the same reference algorithm (Efficient RANSAC [SWK07]), stated in Sec 4.2.4 (p.15):

- **Noise:** "[Schnabel et al.] explicitly shows their robustness by corrupting input
  data with up to **20% of Gaussian noise** and correctly reconstructing the primitive."
- **Outliers:** "In their study, Schnabel et al. show that their RANSAC-based method can
  handle up to **95% outliers**."

Caveats for our use: (a) these are single reported operating points ("still gets the
right primitive type at this level"), **not** a curve of angular error vs. noise sigma —
they answer the coarse "when does the method stop returning the correct primitive type"
half of the branch question, not the fine "what angular error results" half; (b) "20% of
Gaussian noise" is not defined here relative to what scale (point spacing? bounding-box
diagonal? sensor sigma?) — that definition lives in the original Schnabel et al. 2007
paper, not in this survey, so it is not independently checkable from this document alone;
(c) no equivalent numbers are given for Hough-transform or region-growing families — the
survey states qualitatively that Hough is robust to noise ("methods that best handle noisy
data are based on robust theoretical paradigms such as the Hough transform [WO02] or
RANSAC [SWK07]," p.15) without a number. **This is the single most load-bearing gap for
Branch Question "what angular error... at what noise/incompleteness level do methods stop
returning the correct primitive type" — the survey gives one anecdotal operating point for
one algorithm and nothing resembling a calibrated curve.**

## Normal estimation is an *assumed input*, not a topic this survey covers

This is a critical, easy-to-miss negative finding. Every core algorithm in Sec 3
(Algorithm 1 RANSAC, p.6; Algorithm 2 Hough Transform, p.7; Algorithm 3 Clustering Plane
Parameter Space, p.8) takes **"List of vertices and associated normals"** as its stated
Input line — normals are a precondition, supplied by the acquisition pipeline, not
something any surveyed method computes or whose error is analyzed. Section 3.1 (p.6)
states the RANSAC minimal-set claim explicitly: "they claim that three 3D points and
associated normals are sufficient to estimate the parameters for all these shapes,"
crediting Schnabel et al.'s own Section 4.1. The word "normal estimation" as a topic (PCA
vs. jets vs. VCM vs. Hough-based estimators, or any angular-error-vs-k-vs-sigma
relationship) **does not appear anywhere in this document.** This means:
- **Branch Question "quantified relationship between per-point noise sigma, neighbourhood
  size k, and normal angular error for each estimator family" is entirely unaddressed
  here** — must be sourced from dedicated normal-estimation literature (Mitra et al.,
  jet-fitting/CGAL, VCM papers), not this survey.
- It also means every primitive-fitting method surveyed here **inherits** whatever normal
  error the upstream (unspecified) estimator produces, silently. For our defect modes
  (registration jitter, glossy/dark dropout, ~4 cm fused-vs-single-view offset), any
  primitive-fitting accuracy claim quoted from this literature implicitly assumes
  normals of unstated, unquantified quality — a load-bearing assumption the survey never
  flags as such.

## Framework comparison table — Table 1 (p.5) and radar chart Figure 4 (p.5)

| Framework | Strengths | Weaknesses | Reference algorithm |
|---|---|---|---|
| RANSAC | Simple, general, accurate, robust to outliers | Many parameters to tune, dependent on a minimum set, no spatial consistency | Fast RANSAC [SWK07] |
| Local statistics | Application-specific | Model-dependent | Monocular Occupancy Maps [CSM12] |
| Hough transform | Handles missing data, supports many model instances, relatively robust to noise | Unbounded space size, dependent on parameter space quantization | Primitive-based registration [RDvdHV07] |
| Clustering parameter space | Robust to outliers | Restricted to low dimensions | Cluster Normal Space [HHRB11] |
| Primitive growing | Meaningful segmentation, spatial consistency | Slow, local, sensitive to initial conditions (seeds), sensitive to noise, sensitive to outliers | Hierarchical Modeling [AP10] |
| Automatic clustering | No prior on location, few parameters | Dependent on seeds, sensitive to noise, can require numerous clusters (K-means) | Quadric Surface Fitting [YWLY12] |
| Segmentation then fitting | Vast literature for segmentation | Can merge different primitives, sensitive to noise, sensitive to outliers | Hybrid City Representation [LM12] |

Directly relevant to picking a *fallback-detection* approach for our two-tier design:
RANSAC is rated "robust to outliers" but explicitly **not** rated for spatial consistency
or robustness beyond outliers in this summary row — noise robustness for RANSAC is not
even listed as a strength in Table 1's own row (only outlier-robustness is), which is
consistent with the fact that the paper's only quoted noise number (20% Gaussian, above)
comes from the original Schnabel paper, not from the survey treating noise-robustness as
RANSAC's calling card.

## Incomplete data — partial-cylinder-relevant findings (Sec 4.2.4, p.15)

"Robustness to Incomplete Data" paragraph, verbatim-relevant content: "The robustness of
a method against incomplete data is measured by the completeness of shapes obtained when
only parts of it were captured... **Region growing based methods do not perform well when
data is missing**, as they only exploit existing data without trying to create a global
model of the input point clouds... Other methods that handle missing data well include
**multiscale methods [AP10] and multiple view reconstruction [FCSS09b]**." Earlier in the
same subsection: "some methods fail to handle missing data [OVWK16, CZS*13], robust ones
have been developed [BFF15, VLA15]." **No PotSAC, no symmetry/mirror-plane completion, no
quantified axis-recovery-vs-coverage-fraction relationship anywhere** — the survey predates
this specific technique family entirely (PotSAC-style axis estimation is 2020s work). This
is a hard negative for the "partially observed cylinder... symmetry-based completion" RQ:
this document only tells us *which method families are reputationally good/bad at missing
data* (RANSAC/stochastic and local-statistics methods better than pure region growing),
not *how good*, and not via symmetry-completion mechanisms at all.

## Fit-confidence / failure-detection signals — what this survey actually offers (Sec 6)

Section 6.1 "Evaluation Methodology" (p.18) lists the metrics the field uses: fitting
error (Sec 6.2), processing time / frames-per-second, and "simplicity of the model and
overdetection: number of primitives." Segmentation correctness and spatial consistency
are named as metrics that "require ground truth information... prior manual and
user-assisted work on the data" (p.18) — directly answers the ground-truth RQ: **as of
this survey, ground truth for "correct decomposition" is manual/human-annotated
segmentation, not CAD-model comparison** (no CAD-vs-scan protocol is described anywhere).

Section 6.2 "Evaluation Metrics" gives exactly two formulas:
- **Fitting error** (unnumbered equation, p.18): for inliers **P**ʲᵢ of primitive Sᵢ,
  ε = Σᵢ Σⱼ ‖**P**ʲᵢ − proj(**P**ʲᵢ, Sᵢ)‖² — plain sum of squared point-to-primitive
  distances.
- **Hausdorff distance** (p.18, citing [CRS98]): H_AB = max_{a∈A} min_{b∈B} d(a,b).

Section 6.3 "Processing Metrics" (p.18-20) explicitly separates these from quality
metrics: quadric error metric (QEM, [GH97]) and its extension spherical QEM (SQEM,
[TGB13]) are described as metrics **used to drive algorithms** (mesh/sphere-mesh
simplification cost functions), with the survey's own caveat stated plainly: **"The
following metrics are used to drive algorithms, although their values do not make much
sense to evaluate the quality of the output"** (p.18, section header text). This is a
directly relevant negative/cautionary finding for the "what computable signals do papers
use to declare a fit unreliable" RQ: **this survey does not mention inlier ratio, residual
RMS as a *confidence* signal (only as a *fitting-error* metric, not calibrated to
downstream failure), MDL/energy score as a *confidence* signal, EMS posterior outlier
probability, MonteBoxFinder score, or GNC/TEASER optimality certificates at all** — none
of those terms appear in the document. The survey's own Sec 7.3 "Interpretation" (p.21)
names this exact gap as unsolved future work (see below), which is itself the most
important single finding of this reading for that branch question: **as of 2019, the
field had not yet built fit-confidence/uncertainty output into primitive detectors**, so
the calibration question ("has anyone calibrated a score against downstream task
failure") has **no candidate answer in this literature generation** — must be sourced from
2020s-era work explicitly, which this survey cannot point to (it doesn't exist yet at its
time of writing).

## Section 7.3 "Interpretation" — the paper's own naming of our exact gap (p.21)

Quoted because it is the single highest-value passage in the document for our branch's
core uncertainty questions: *"beyond the extraction of per-primitive semantics, a global
consistency check is often missing in state-of-the-art methods... **the amount of
variability of primitives detected in a scene, as well as their uncertainty level, are so
far not explicitly provided on the output channel of the detection systems. Instead of a
static primitive, new systems could be built to provide parametric primitives with
restricted parameter spaces. Allowing the user to explore the space of possible fitted
primitives could be achieved interactively through specifically modulated ranges of
parameters and interfaces.**"* This directly confirms: (a) as of this survey, no reviewed
method outputs a covariance/uncertainty on a fitted primitive's parameters — answering
"is there a closed-form covariance for a fitted plane's/cylinder's orientation" in the
negative *for this literature*; (b) the paper's own forward-looking suggestion is
interactive parameter-range exploration by a *user*, not an automatic statistical
covariance or M3C2-EP-style propagation — it does not anticipate or point toward the
change-detection literature (M3C2-EP) as a transfer source; that connection, if it
exists, has to be made independently by us, not inherited from this survey.

## Spatial reasoning / graphs of primitives — closest precedent to "per-primitive
viewpoints" (Sec 7.2, p.21-22)

No method in this survey generates viewpoints or camera poses from a primitive
decomposition — confirmed absent by a full read of Sec 5 (Methods and Applications) and
Sec 7 (Discussion). The nearest related idea is graphs of *spatial relations between
primitives* (not observation viewpoints):
- Li et al. [LWC*11] GlobFit: builds a graph of geometric relations (coplanarity,
  coaxiality, orthogonality) between RANSAC-detected parts, simplified by merging nodes.
- Rusu et al. [RBM*07] "3D Object Maps": relations between detected household objects via
  3D rigid transformation matrices.
- Silberman et al. [SHKF12]: hierarchical scene segmentation into objects + adjacency
  ("supporting") relations, building an object-support graph.
- Ochmann et al. [OVWK14]: detects doors/windows to build a room-connectivity graph.
- Schnabel/Wahl/Klein follow-up [SWWK08]: builds a *topology graph* of object/object-parts
  from Efficient-RANSAC-detected primitives for recognition and semantic interpretation
  across scans — this is the closest single precedent to "decompose into primitives, then
  reason over the structure," but the output is a recognition/topology graph, not
  viewpoints.
- **Robotics/SLAM applications** (Sec 1.3 "Robotics," p.2-3; Sec 5.1-5.3 "Indoor Scenes"
  passim) use detected planes/primitives for **localization and navigation-space
  computation** (e.g., "when computing the navigation spaces in a 3D scene, simple
  primitives help quickly cull away entire regions... restrict authorized navigation to
  areas that are free from object occlusion," Sec 1.3 "Interaction," p.3) — i.e., the
  substitution this literature makes for "primitives → viewpoints" is "primitives →
  free-space/navigation graph," a related but distinct problem (occupancy, not
  observation). **No skeleton- or gridization-based viewpoint precedent appears either**
  — those aren't discussed. **Answer to "what is the nearest precedent and what did it
  substitute for primitives": nothing in this survey substitutes anything for primitives
  in a viewpoint-planning context — the entire idea of primitive-conditioned camera view
  generation is absent from this literature**, confirming the branch's premise that this
  is unexplored territory, at least as of 2019 and within this survey's scope (SLAM/
  robotics papers, not graphics/vision primitive-fitting papers, are the more likely place
  to find any precedent, and this survey only lightly touches robotics applications).

## Face-identity / repeatability across re-scans (RQ3 relevance)

Not addressed. "Temporal Consistency" is listed as a formal property (Sec 4.2.2, p.13-14,
"Temporal Consistency" paragraph) but is scoped to *dynamic scenes* (objects/people moving,
or the camera moving) and whether the *model* stays stable across time — it is about
scene-level model stability, not about per-primitive identity/ordering/index stability
across independent re-scans of a static object. No method is described as tracking or
guaranteeing "face 3 stays face 3." This RQ needs a different source.

## Tooling — cross-check against Open3D + CGAL/PCL (feeds the sibling Open3D note)

Table 5 "Available implementations" (p.20, links accessed Feb 2018 per caption, so likely
stale today) lists source code for: 3D Kernel Hough Transform [LO15] (C++), RAPter
[MMBM15] (C++), Generalized Cylinder [ZYH*15] (C++), Agglomerative Clustering [FTK14]
(C++), Fitting Cuboids [JX13] (Matlab), Learning Object Templates [KLM*13] (Matlab),
Geosemantic Snapping [SAG*13] (C#), Sphere Meshes [TGB13] (C++), Indoor Robot Navigation
[BV12] (C++), Object Support [SHKF12] (Matlab), Plane Filtering [BV11] (C++), GlobFit
[LWC*11] (C++), Plane Detection for SLAM [XZZ*11] — listed in-text as available too,
Hierarchical Segmentation [AFS06] (C++), Billboard Clouds [DDSD03] (C++), and — as three
separate rows — "Fast RANSAC (CGAL)" [SWK07] (C++), "Fast RANSAC (original)" [SWK07]
(C++), and "Variants of RANSAC (PCL)" (C++). **Correction: an earlier draft of this note
claimed CGAL is never mentioned — that is wrong.** Table 5 (verified directly against
`pdftotext -layout` output of the authors' PDF) explicitly lists the CGAL build of
Schnabel et al.'s Efficient RANSAC as a distinct, separately-linked row from both the
authors' original implementation and PCL's SAC module. So the survey's own tooling
pointers for RANSAC-family fitting are **both CGAL and PCL**, not PCL alone — consistent
with (and confirming) the external knowledge that CGAL's `Efficient_RANSAC`/
`Shape_Detection` package implements Schnabel-style plane/sphere/cylinder/cone/torus
fitting. Table
6 (p.20) datasets are all mid-2010s indoor-scene RGB-D sets (SUN3D [XOT13], NYU Depth V2
[SHKF12], plus per-paper point-cloud dumps from [LO15], [MMBM15], [ZYH*15], [CLW*14],
[MPM*14], [KLM*13], [SXZ*12], [LWC*11]) — **none is a CAD-vs-scan or corruption-injection
benchmark**; Fit4CAD, SHREC 2022, and ABC are entirely absent (temporal cutoff, as noted
above).

## Preprocessing (voxel downsampling, outlier removal, denoising) — not covered

The survey does not discuss preprocessing steps (voxel downsampling, statistical/radius
outlier removal, denoising) as a distinct topic, and gives no numbers on how such steps
change fitting outcomes or whether denoising over-smooths edges that segmentation depends
on. This is a full negative finding — the paper's scope starts *after* whatever
preprocessing the input data received; it is not a preprocessing-effects study. (Compare
sibling note in this folder on Open3D docs, which is similarly silent on this point.)

## Registration/alignment covariance, M3C2-EP, depth-dropout-bias mechanisms — not covered

None of these appear: no discussion of registration/alignment error propagating into
fitted-primitive covariance (the survey's fitting-error metric, Sec 6.2, is a point-to-
primitive residual sum computed *after* the fact, not a forward error-propagation model);
no mention of M3C2 or M3C2-EP (those are geomorphology/change-detection literature,
outside this survey's citation graph entirely); no discussion of *systematic* bias
mechanisms from glossy/dark-surface depth dropout (shrinking a face, shifting an axis,
tilting a normal) — dropout/missing-data is discussed only as *incompleteness* (Sec
4.2.4, see above), with no distinction drawn between random incompleteness and
systematically *biased* incompleteness. This is an important negative finding in its own
right: **the survey's entire robustness vocabulary (noise, outliers, incomplete data)
has no category for "systematic geometric bias from a directional sensor artifact"** — our
branch's concern that dropout could be a bias rather than just added variance is simply
outside this paper's conceptual frame, not addressed and not contradicted.

## What transfers to our setting

- **Bounded-patch trimming mechanism** (grid + connected components + convex hull of
  inliers, Schnabel/Andrew) is a concrete, citable algorithmic answer to "how do you get
  a face centre + face size" — and matches what Open3D's `detect_planar_patches` already
  implements (see sibling note), giving us a load-bearing algorithmic justification for
  that tool choice beyond "it's in the library we use."
- **Table 4 / Table 1's per-framework robustness ratings** are a legitimate first-pass
  filter for candidate methods/frameworks given our defect profile (noisy, incomplete,
  outlier-containing fused clouds): RANSAC-family ranks well on outliers and simplicity/
  generality but is not singled out for noise robustness beyond one anecdotal number;
  region growing ranks well on meaningful/spatially-consistent segmentation but poorly on
  missing data and noise — directly relevant since region growing is a natural fit-quality
  competitor to RANSAC for our face-segmentation needs but is the worse choice under our
  incompleteness.
- **Generalized Cylinder Decomposition** [ZYH*15] is the concrete, named answer (within
  this literature) to "what does a richer vocabulary buy for chamfered/organic parts,"
  even though this survey gives no quantified normal-accuracy comparison for it.
- The survey's own naming of "primitive uncertainty is not yet provided by detection
  systems" (Sec 7.3) is itself a useful citation: it lets us assert "this is a genuinely
  open problem as of the classical era of this field" with a source, framing our
  covariance/confidence work as filling a named gap rather than reinventing a solved
  problem.

## What does NOT transfer / negative findings (first-class, per task instructions)

- No angular-error-vs-noise-sigma curves, for any estimator family or any primitive type.
  Only one anecdotal robustness operating point exists in the whole document (20% Gaussian
  noise, RANSAC, [SWK07]), not a quantified function.
- No coverage of normal estimation at all — every algorithm here assumes normals as a
  given input; the accuracy of that input is never discussed or propagated.
- No closed-form covariance model for fitted-primitive orientation, no M3C2/M3C2-EP
  connection, no registration-covariance-to-fit-covariance propagation.
- No systematic-bias framing for glossy/dark dropout — only generic "incompleteness."
- No superquadric coverage (term never appears); richer-than-four-primitive vocabulary is
  represented instead by generalized cylinders and tori, with no recovered-accuracy
  numbers for either.
- No fit-confidence/introspection signals beyond fitting-error (SSE) and Hausdorff
  distance as *evaluation* (not *live decision*) metrics; explicit caveat that QEM/SQEM
  (the closest thing to an energy score) "does not make much sense to evaluate the
  quality of the output" when used as a driving cost.
- No calibration of any score against downstream task failure — the concept doesn't
  appear.
- No face-identity/ordering-repeatability treatment across independent re-scans.
- No primitive-to-viewpoint generation precedent of any kind; nearest neighbours are
  primitive-to-navigation-graph (robotics) and primitive-to-relation-graph (GlobFit,
  SWWK08) uses, both structurally different problems.
- No modern benchmark protocols (Fit4CAD/SHREC2022/ABC) — datasets listed (Table 6) are
  all pre-2016 indoor RGB-D sets with manual/semantic ground truth, not CAD-vs-scan.
- No preprocessing-effects study (denoising/downsampling vs. fitting outcome).

## Load-bearing assumptions in this source

- Every classical method surveyed assumes point normals are already computed and
  reasonably accurate (RANSAC minimal sets use 3 oriented points; Hough plane detection
  needs a normal per point for the Gauss-map step, Fig. 6-7 p.8). Our fused-cloud defect
  modes (registration jitter, dropout, depth bias) attack exactly this precondition, and
  this survey has no machinery to reason about that attack because it treats normals as
  exogenous.
- The survey's whole "robustness" vocabulary (noise / outliers / incomplete data) is a
  three-bucket qualitative rating system (Table 4's Robustness columns use ●/○/— per
  method), not a quantitative one — so even where a method is rated "good" on noise or
  incompleteness in Table 4, that rating is the *original authors' own claim*, filtered
  through this survey's qualitative bucketing, not an independently reproduced number.

## Checkable citations (page/section anchors, this PDF)

- Definition of simple primitive, 4 categories: Sec 2.2, p.4-5.
- Primitive shape trimming (bounded faces): Sec 2.2, p.5, "Primitive Shape Trimming"
  paragraph, citing [SWK07], [And79], [BV11], [FTK14].
- Table 1 (framework strengths/weaknesses) and Figure 4 (radar chart): p.5.
- RANSAC minimal-set claim, Algorithm 1: Sec 3.1, p.6.
- 20% Gaussian noise / 95% outliers numbers: Sec 4.2.4, p.15, "Examples of Robustness to
  Noise" and "Robustness to Outliers" paragraphs, both citing [SWK07].
- Incomplete-data paragraph: Sec 4.2.4, p.15, "Robustness to Incomplete Data."
- Fitting error and Hausdorff distance formulas: Sec 6.2, p.18.
- QEM/SQEM "doesn't evaluate quality" caveat: Sec 6.3, p.18-20.
- Ground-truth requiring manual/user-assisted work: Sec 6.1, p.18.
- Table 4 master comparison: p.19. Table 5 (code) and Table 6 (datasets): p.20.
- Interpretation / uncertainty-not-provided passage: Sec 7.3, p.21, "Interpretation"
  paragraph.
- Spatial reasoning graph precedents: Sec 7.2, p.21-22, citing [LWC*11], [RBM*07],
  [SHKF12], [OVWK14], [SWWK08].
- Generalized Cylinder Decomposition discussion: Sec 5.3, p.17; Sec 7.1, p.20-21.
- Ellipsoids/tori for organic/chamfered shapes: Sec 5.4, p.18.
- CGAL and PCL as the survey's own RANSAC tooling pointers ("Fast RANSAC (CGAL)," "Fast
  RANSAC (original)," "Variants of RANSAC (PCL)"): Table 5, p.20.
