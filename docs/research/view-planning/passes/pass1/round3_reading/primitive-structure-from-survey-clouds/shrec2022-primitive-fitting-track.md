# Romanengo, Raffo, Biasotti, Falcidieno et al. (2022) — "SHREC 2022: Fitting and recognition of simple geometric primitives on point clouds"

**Source:** Computers & Graphics 107 (2022), DOI 10.1016/j.cag.2022.07.004. ScienceDirect page is
paywalled (403 on direct fetch); full text obtained via the authors' arXiv preprint
[2206.07636](https://arxiv.org/abs/2206.07636) (identical content, confirmed via OpenAlex
`oa_status: green`, `oa_url` pointing to this arXiv id). Read in full: 32 pages, all sections + both
appendices (A: per-perturbation classification tables 3–12; B: per-perturbation fitting tables 13–22),
via `pdftotext -layout`. Local copy: `/tmp/shrec2022.{pdf,txt}`.

Benchmark data + code: https://github.com/chiararomanengo/SHREC2022.git. Per-method repos linked in
§3.1–3.6 below.

---

## 0. What this paper actually is (important framing correction)

This is **not** a head-to-head of RANSAC-family fitters. Only one of the six methods cites Efficient
RANSAC (Schnabel et al. [17]) at all, and only as a sub-step inside one method's cone-fitting branch
(§3.2.1, M2). The six compared pipelines are:

| ID | Family | Classification | Parameter fitting |
|----|--------|-----------------|--------------------|
| M1 | Direct | Accelerated algebraic Hough transform | Same HT, per-primitive-family search |
| M2 | Direct | k-NN over HONV / surflet-pair histograms | Per-primitive least-squares (RANSAC only for cones) |
| M3 | Neural | PointNet classifier | PointNet regression heads (no geometric refinement) |
| M4 | Neural | Minkowski Engine (sparse voxel conv) classifier | Same backbone regression + least-squares refinement (`geomfitty`) |
| M5 | Hybrid | PointNet ensemble (3-model vote) | Least-squares (`geomfitty`) using network output as init |
| M6 | Hybrid | AlexNet on multi-view renders | Least-squares per ISO GPS reconstruction standard |

So the paper answers "Hough vs. neural-classification-then-fit vs. end-to-end-neural-regression", not
"which RANSAC descendant wins." Useful shortlist signal, but readers expecting an Efficient-RANSAC /
region-growing comparison (as our task brief implied) won't find one here — Schnabel et al. is cited,
not benchmarked.

**Also critical:** every method operates on a **single pre-segmented point cloud known to contain
exactly one primitive**. The track explicitly excludes the segmentation problem (§1.1: "we are not
interested in the whole segmentation process, but... classification and recognition of segments").
There is no whole-object decomposition into multiple primitives anywhere in this paper — the authors
flag that as future work (§5, Concluding remarks: "We believe that a similar comparative analysis will
be possible in the future on models with multiple primitives").

---

## 1. Dataset and ground truth (§2.1, Fig. 1)

- 46,925 point clouds total: 46,000 train / 925 test. Five primitive families: **plane, cylinder,
  sphere, cone, torus** (T1–T5) — the classical CSG vocabulary, **no superquadrics, no cuboid/box
  primitive, no free-form/learned abstraction**. Torus is the vocabulary's answer to filleted/rounded
  edges (see §5 below on relevance to chamfers).
- Generation procedure (§2.1): sample the primitive in canonical form (centered, axis-aligned) →
  segment it with random cutting planes (this is what bounds the point cloud spatially) → apply random
  rigid transform → apply one of 10 perturbation types.
- **Perturbation types A0–A9** (§2.1, exact definitions):
  - A0 clean; A1 uniform noise `U(−1/n,1/n)`, `n∈[3,20]` random, applied to a random point subset; A2
    Gaussian noise `N(−1/n, 4/n²)`, `n∈[10,30]`; A3 undersampling (random point removal); **A4 missing
    parts** — pick a random point, remove everything inside a randomly-sized sphere centered there (this
    is the closest analog in the paper to "dropout", but it is an isotropic spatial hole, not a
    view-dependent / incidence-angle-dependent sensor dropout); A5–A8 = combinations of {uniform,
    Gaussian} × {undersampling, missing parts}; A9 = one localized bivariate-Gaussian bump deformation.
  - **This noise model is entirely sensor-agnostic**: no range-dependent or incidence-angle-dependent
    depth bias, no registration/multi-view fusion step, no specular/glossy dropout physics — it is
    generic isotropic per-point noise plus geometric hole-punching. Every branch question that asks
    about range-and-incidence-angle-dependent error models, registration covariance propagation, or
    sensor-realistic dropout mechanisms gets **no data from this paper**; the corruption model would
    need to be replaced entirely to match our defect profile.
- **Ground-truth vectors are infinite-primitive analytic parameters, not bounded faces** (§2.1):
  plane → `[type, normal(3), one point on plane(3)]`; cylinder → `[type, radius, axis(3), one point on
  axis(3)]`; sphere → `[type, radius, center(3)]`; cone → `[type, half-aperture, axis(3), vertex(3)]`;
  torus → `[type, minor r, major R, axis(3), center(3)]`. **No extent, no boundary polygon, no bounded
  centroid is ever recorded or predicted by any of the six methods.** Directly negative for the branch
  question about "bounded faces (extent, boundary, centroid)" — this benchmark and all six competing
  methods only ever produce/score infinite-primitive equations.
- Ground truth is **exact by construction** (clouds sampled directly from the analytic primitive) —
  there is no CAD-scan-then-manually-annotate step and no real-scanner data anywhere in the dataset.
  The authors explicitly propose, as their own future work (§5), building a real-scanner dataset with
  CAD-derived ground truth as the natural next step — i.e., this exact gap (real-scan ground truth
  protocol) is *unsolved on the authors' own roadmap* as of 2022.

## 2. Evaluation measures (§2.2)

- **Classification**: confusion matrix + TPR/TNR/PPV/NPV/Accuracy, macro-averaged across the 5 classes
  (§2.2.1).
- **Fitting** (§2.2.2, Eq. 1): Mean Fitting Error `MFE(P,SP) = (1/|P|) Σ d(x,SP) / l`, point-to-surface
  distance normalized by the bounding-box diagonal `l` of the input cloud — a scale-normalized residual,
  not an angular quantity. Also directed Hausdorff distance `max_x min_y d(x,y)`.
- **Recognition**: `L2` distance between predicted and ground-truth parameter vectors (with the
  type-index and, for planes/cylinders, the arbitrary point dropped since it isn't uniquely comparable).
- **Critical gap for our RQs**: `L2` conflates orientation error (normal/axis direction) with
  scale/position error (radius, center) into one scalar. **The paper never decomposes recognition error
  into a standalone angular error in degrees for a normal or an axis.** No number in this paper can be
  quoted directly as "X° angular error at noise level Y." M1's own companion journal paper (Raffo et
  al., CAGD 2022, ref [11], "Fitting and recognition of geometric primitives in segmented 3D point
  clouds using a localized voting procedure") is the more likely place to find a decomposed angular
  error analysis — not fetched here, flagged as a follow-up read.

## 3. Method sketches (§3, only what's structurally informative)

- **M1 — accelerated Hough transform** (Romanengo/Raffo/Biasotti/Falcidieno, §3.1; code:
  github.com/chiararomanengo/fitting_geometric_primitives). Normals first estimated via a Hough-based
  plane-in-Hesse-form voting scheme (cites Limberger & Oliveira 2015 [14], the classic real-time planar
  region detector) — robust to most artifacts because it's a voting method, but no noise-vs-angular-error
  ablation reported for this normal step specifically. Then per-primitive-family HT search in a
  dimension-reduced parameter space (rotation/translation brings the cloud to standard form first, using
  primitive-specific geometric priors). **Runs the fit for all 5 families and returns the one with lowest
  MFE** — this is the closest thing in this paper to a goodness-of-fit-based type-selection rule; not
  exposed/calibrated as a confidence threshold, just used internally to pick a winner.
  - Runtime: 123 s/cloud average (MATLAB, single CPU, i9). Slow — direct/interpretable but the slowest
    method by ~2 orders of magnitude vs. the GPU neural methods.
- **M2 — HONV/surflet histograms + k-NN + per-family least squares** (§3.2; code:
  github.com/ccbien/SHREC22-Track2). Cascaded k-NN classifiers (plane vs. non-planar → sphere/torus vs.
  cylinder/cone → final split), each on 2D histogram features (HONV [15], surflet-pair [16]). Fitting:
  closed-form/least-squares per family, cone via Efficient RANSAC [17] (the only literal RANSAC use in
  the whole paper). Runtime: ~4 s/cloud total (fast, CPU).
- **M3 — PointNet, fully neural, no geometric refinement** (§3.3; code:
  github.com/ivansipiran/shrec2022-geometric-primitives). Classifier network + 5 primitive-specific
  regression heads off a shared PointNet backbone; composite losses per primitive (scalar/point/vector
  losses, notably a cosine^8 + L2 vector loss for normals/axes, Eq. in §3.3.1). **No least-squares
  post-fit.** Runtime 280 ms/cloud (GPU).
- **M4 — Minkowski sparse-voxel network + least-squares refinement** (§3.4; code:
  github.com/JohnRomanelis/SHREC2022_PrimitiveRecognition). Voxelizes to a sparse grid, sparse-conv
  backbone (from Minkowski Engine [19]), classification + per-family regression, **then refines with
  `geomfitty` least-squares using the network output as initial guess** — explicitly the network's job is
  to produce "an excellent initial guess" for a fast subsequent LSQ fit (§3.4.1). Includes a
  backtracking heuristic: if fit residual exceeds a threshold after the top-1 class guess, retry with
  the second-most-probable class. Runtime: fastest of all six, ~40 ms/cloud total (0.010+0.009+0.021 s
  for classify+regress+LSQ, GPU).
- **M5 — PointNet ensemble + least squares** (§3.5; code: github.com/ToTuanAn/SHREC2022). 3-model
  majority vote for classification, `geomfitty` LSQ for parameters. Runtime ~36 s classification (925
  clouds) + 10 min parameter fitting (925 clouds) ⇒ ~650 ms/cloud average total (GPU/Colab P100).
- **M6 — AlexNet on multi-view renders + ISO-GPS least squares** (§3.6; code:
  github.com/YifanQie/SHREC22_fitting_LURPA). Point cloud centered, scaled to unit sphere, **rotated so
  the first principal axis (PCA) aligns with z** — a per-object canonicalization step, not a hard
  assumption of known canonical pose in the world frame; turns the cloud into rendered images for
  transfer-learned AlexNet classification, then LSQ fit per ISO GPS reconstruction standard [23,24].
  Runtime ~0.5 s/cloud (CPU).

All four neural/hybrid methods (M3–M6) are trained **only** on this synthetic dataset; none are
evaluated on real scans anywhere in this paper (confirmed in §5 as an explicit limitation/future work).
Code is public per-method (GitHub links above), but "off-the-shelf, no retraining" does not hold for
M3–M6 on our real defective clouds — they'd need retraining or at minimum fine-tuning/domain
adaptation to be trustworthy off this dataset. M1 (Hough) and M2 (the LSQ/RANSAC fitting *sub-step*,
not its k-NN classifier) are the only pieces that are genuinely training-free at inference.

## 4. Results (§4, Tables 1–22, Figs. 8–10)

### 4.1 Classification (§4.1, Table 1 = whole test set; Tables 3–12 in Appendix A = per-perturbation)

- All six methods: **accuracy > 90%** on every primitive type over the whole test set; **macro-averaged
  accuracy > 95%** for all methods (Table 1, "Avg" column: M1 0.9676, M2 0.9494, M3 0.9831, M4 0.9935,
  M5 0.9663, M6 0.9542).
- **Cylinders, cones and tori are consistently the hardest classes**; planes and spheres are easiest
  (near-perfect for most methods). E.g. per-class TPR (Table 1): cylinder TPR ranges 0.88 (M5) to 0.9847
  (M1... note M1's own NPV is listed higher, TPR for cylinder is 0.9405); cone TPR as low as 0.8432 (M1)
  and 0.7784 (M2); torus TPR as low as 0.7297 (M6).
- **Best overall classifier: M4** (Minkowski + refinement) — top or near-top on every metric, macro ACC
  0.9935. **M3** (plain PointNet classifier) close second, macro ACC 0.9831. The two fully-neural
  classification stages beat both direct methods (M1, M2) and both hybrid classifiers (M5, M6) on raw
  classification.
- **Degradation by perturbation type** (Fig. 9, quantified in Tables 3–12):
  - Text of §4.1 states M1 (Hough) "suffers the most from undersampled data" but "is not much affected
    by missing data," attributed to the Hough voting paradigm degrading gracefully with holes but needing
    point density to converge. **Verified against the raw per-class tables, this needs a correction and a
    caveat**: macro-avg TPR for M1 is actually *lower* under missing data (Table 7, avg TPR 0.8400, cylinder
    T2 TPR 0.8000, down from clean-data 0.9000) than under undersampling (Table 6, avg TPR 0.9500, cylinder
    T2 TPR 0.9833, *up* from clean). So at the per-class-TPR level the numbers don't straightforwardly
    support the "robust to missing data" claim for cylinders specifically — the authors' statement likely
    reflects a different aggregate (e.g. classification accuracy trend across all artifact types in Fig. 9,
    or a cross-method rather than cross-perturbation comparison). **Caveat that matters for our RQs**: each
    perturbation-type test bucket has only 100 point clouds split across 5 classes (~20/class), so per-class
    per-perturbation TPR swings of ±0.05–0.15 are within single-sample noise — these tables are too small to
    support precise "at what dropout level does X° error occur" claims; only broad-strokes ordering is
    trustworthy.
  - Noise+undersampling combinations (A5/A6, Tables 8–9) are the worst case across the board for most
    methods — e.g. M6 cylinder TPR drops to 0.85 (Table 8) / 0.85 (Table 9), well below its clean-data
    0.91 (wait — check: Table 3 clean M6 cylinder wasn't listed for T2 TPR at 1.0000, cone 0.9000 —
    combos consistently push all methods below their clean-data numbers).
  - Small local deformations (A9, Table 12) are, surprisingly, where M2/M4/M6 do relatively **better**
    than on some noise conditions — the paper notes these methods "extract some additional hidden
    information" from localized bumps (§4.1), an unexplained but consistent effect.
- **No decomposition into a literal angular-error number vs. noise σ is given** — classification tables
  answer "does the method still call it the right primitive type" as perturbation severity increases,
  which is the closest this paper gets to "at what noise/incompleteness level do methods stop returning
  the correct primitive type" — but severity is bucketed by artifact type (A0…A9), not by a continuous
  noise-σ sweep with a clean threshold crossing.

### 4.2 Fitting and recognition (§4.2, Table 2 = whole test set; Tables 13–22 = per-perturbation)

- Table 2 (whole test set), mean/median (Q2) MFE (normalized point-to-surface residual):
  - **M4 lowest mean MFE**: 5.82e-3. M1 close: 9.19e-3 mean, but **best median (Q2) MFE is M2 at
    1.94e-3**, M4 second at 3.59e-3, M6 third at 3.43e-3.
  - **M3 (plain PointNet regression, no LSQ refinement) has by far the worst typical fitting accuracy**:
    median MFE 4.49e-2 — roughly 10–20× worse than M2/M4/M6's medians, despite M3 being competitive on
    *classification*. This is the paper's clearest actionable finding: **end-to-end neural regression
    without a subsequent geometric least-squares refinement step is a poor primitive fitter**, even when
    the classification stage built on the same backbone is excellent. M4/M5's strategy (network guess →
    LSQ polish) structurally fixes this.
  - M2, M5, M6 show catastrophic mean/std blowups (e.g., M2 mean L2 = 1.21e+11, M6 mean MFE =
    2.56e+05) driven by a **small number of outliers** — degenerate cone fits where the estimated
    half-angle collapses near zero and the vertex flies far from the object (§4.2, explicitly diagnosed:
    "M2 presents 11 outliers corresponding to recognition of cones in which the estimated half-angle
    estimate is close to zero..."; "M6 presents 30 outliers... analogue to that of M2"). **This is a
    concrete, named failure mode for cone fitting under noise**: near-degenerate cone geometry (small
    aperture) is numerically unstable for least-squares solvers, producing wildly wrong parameters while
    (per M5's case) the point-to-surface residual can still look deceptively low — i.e., **MFE alone can
    fail to flag a bad fit** when the surface family itself is degenerate; this is a direct data point for
    "what computable signals detect an unreliable fit" — MFE/Hausdorff by themselves are shown here to be
    insufficient for cone near-degeneracy.
  - M1 and M4 "alternate first and second place" in both fitting and recognition (§4.2) — the one direct
    method (accelerated Hough) and the one hybrid-with-refinement neural method are the two most reliable
    performers end-to-end across both classification and fitting; the "pure neural, no refinement" (M3)
    and "pure neural classification with fragile downstream LSQ" (M2/M5/M6, when they hit degenerate
    cases) are the two failure-prone extremes.
  - Per-perturbation fitting tables (13–22) show the general pattern holds across artifact types: ranking
    stays roughly stable, magnitudes grow somewhat under combined noise+undersampling/missing-data
    (A5–A8, Tables 18–21) but no artifact type qualitatively flips which method wins.

## 5. Relevance to specific branch questions (explicit mapping)

- **Angular error vs. noise/incompleteness, and failure threshold for correct type ID** — *partially
  served, weakly*. No angular-degree numbers are extractable (§2/4.2 above); classification-accuracy
  degradation vs. artifact type is given (Tables 3–12) but bucketed by artifact category, not a
  continuous σ sweep. Use this paper only for qualitative degradation ordering (undersampling worst for
  Hough, noise+undersampling worst overall), not for a numeric angular budget.
- **Vocabulary closure — planes/cylinder/sphere/cone/torus vs. superquadrics/cuboids** — *served*: this
  is exactly the classical 5-primitive CSG vocabulary, confirmed as the field's default (torus
  specifically covers fillets/rounds — chamfers/rounds do **not** force superquadrics per this
  vocabulary's own design intent, though torus is also shown to be one of the two hardest classes to fit
  reliably, so "the vocabulary closes" ≠ "closes cheaply"). No superquadric or cuboid-decomposition
  method appears anywhere in this track.
- **Bounded faces (extent/boundary/centroid) vs. infinite primitives** — *negative finding, directly
  relevant*: none of the 6 methods, nor the ground truth format itself, ever produce a bounded face
  descriptor. Every output is an infinite-primitive equation. A "scan this face orthogonally" pipeline
  cannot use this paper's outputs as-is; would need a separate boundary/extent extraction stage on top
  of whichever fitter is chosen.
- **Precedent for decompose-then-generate-viewpoints** — *not addressed*: single-primitive-per-cloud
  benchmark, no multi-primitive decomposition, no viewpoint generation of any kind. Confirms (via
  absence) that this specific line of work stops at "fit one primitive to one pre-segmented cloud."
- **Order/identity stability across re-scans (RQ3)** — *not addressed*: no multi-primitive objects, no
  notion of "face 3" persisting across anything.
- **Partial cylinder + symmetry completion (PotSAC-style)** — *not addressed directly*: the closest
  analog, A4 "missing parts" (a single random spherical hole per cloud), is a much milder and
  differently-shaped defect than "half the surface scanned plus glossy-side dropout" — it removes one
  contiguous blob, not ~50% of the surface. Table 7 cylinder (T2) TPR under missing-data ranges 0.80 (M1)
  to 1.00 (M4) across the six methods (corrected from an earlier misreading — see §4.1 note above), i.e.
  a *single* spherical hole already costs the weakest method ~10 pts of TPR vs. its clean baseline (0.90);
  a true half-surface-missing scenario is a substantially larger perturbation than anything tested here,
  so this table should be read as a lower bound on how bad it gets, not a predictor. No symmetry-based
  completion method is tested or cited anywhere in this paper.
- **Noise σ / neighborhood k / normal-angular-error per estimator family** — *not addressed
  quantitatively*: M1 cites a Hough-based normal estimator [14] as a black-box preprocessing step; no
  ablation of its angular error vs. noise is reported in this paper.
- **Closed-form orientation covariance; registration-covariance propagation; incidence-angle-dependent
  range error; dropout-mechanism physics; 4 cm-offset diagnosis** — *entirely unaddressed*: the dataset's
  noise model is generic isotropic per-point noise/hole-punching, sensor- and registration-agnostic by
  construction (§2.1). None of these questions can be answered from this paper.
- **Preprocessing ablations (denoising harming edge-dependent segmentation, etc.)** — *not addressed*:
  no systematic preprocessing-vs-outcome study; each team's robustness is a property of their chosen
  method, not an isolated preprocessing variable.
- **Computable unreliability signals (inlier ratio, MDL, GNC certificate, etc.)** — *weak precedent
  only*: M1's use of "lowest MFE across all 5 primitive families" as an implicit type-decision rule is
  the only self-scoring mechanism in the paper, and it is not calibrated or exposed as a confidence
  threshold. The M5/cone-degeneracy finding (§4.2 above) is a useful **negative** data point: MFE can
  look fine even when the recovered parameters (L2 distance) are badly wrong for near-degenerate cones —
  i.e., residual-only confidence is shown here to be an unreliable signal in at least one concrete,
  named failure mode.
- **Calibrating a fit-score against downstream failure** — *not addressed*.
- **Two-tier design (decomposition else uniform sphere) support** — *not addressed*.
- **Open3D/CGAL/PCL/no-retraining runtime budget** — *not directly addressed* (none of these libraries
  are used by any of the six methods), but gives concrete per-object runtime anchors for planning:
  Hough/MATLAB CPU ≈ 123 s; histogram+LSQ CPU ≈ 4 s; PointNet GPU ≈ 0.28 s; Minkowski-voxel+LSQ GPU ≈
  0.04 s (fastest); PointNet-ensemble GPU ≈ 0.65 s; AlexNet CPU ≈ 0.5 s. All per single pre-segmented
  primitive, not a whole multi-primitive scene.
- **Released weights + real-scan generalization vs. ShapeNet-only; canonical-orientation assumptions** —
  *served, negatively*: all 4 neural/hybrid methods (M3–M6) are open-sourced but trained/evaluated
  **exclusively** on this synthetic dataset — zero real-scan validation, explicitly named as future work
  by the authors (§5). M6 does canonicalize orientation via PCA-align-to-z as a preprocessing step
  (§3.6.1) — a per-object, not global, canonicalization, so it doesn't literally violate "arbitrarily
  posed object" (it re-derives its own canonical frame per object) but is worth flagging as a fragile
  step for near-symmetric shapes (e.g., near-spherical or near-cylindrical objects where the principal
  axis is ill-defined/unstable).
- **Datasets/protocols to seed our benchmark** — *served, concretely*: SHREC2022's own dataset/code
  (github.com/chiararomanengo/SHREC2022) and its metric trio (MFE, directed Hausdorff, L2 param
  distance) are directly reusable as an evaluation harness skeleton. The paper's own predecessor,
  **Fit4CAD** (ref [8]: Romanengo, Raffo, Qie, Anwer, Falcidieno, Computers & Graphics 102 (2022)
  133–143), is cited as the closer point-cloud-benchmark ancestor and shares the same organizing group —
  flagged as the next most relevant read for the Fit4CAD-metric-suite branch question.
- **Ground-truth story for "the correct decomposition" of a real object** — *served, negatively*: this
  benchmark's ground truth is exact-by-construction from synthetic sampling; no CAD-vs-scan or
  human-annotation protocol is used or proposed here beyond being named as necessary future work (§5).
  Confirms that if we want real-object ground truth, we're on our own (or need to follow whatever
  Fit4CAD or ABC does — not detailed in this paper beyond citation).

## 6. Bottom line for the shortlist task

This paper is useful for **scoping the primitive vocabulary and for a rough method-family shortlist**
(Hough-transform direct methods and hybrid neural-classify-then-LSQ-refine methods are the two
strongest/most reliable patterns; pure end-to-end neural regression without refinement is measurably
worse at fitting despite good classification; cone fitting has a specific, named near-degeneracy failure
mode). It is **not** useful for: angular-error-vs-noise budgets, registration/sensor-realistic defect
modeling, bounded-face recovery, multi-primitive decomposition, viewpoint planning, or real-scan
generalization evidence — all of these are either absent or explicitly named by the authors as
unaddressed future work.
