# EMS — Robust and Accurate Superquadric Recovery: a Probabilistic Approach

**Liu, Wu, Ruan, Chirikjian — CVPR 2022 (oral).** arXiv:2111.14517 (this read used **v3**,
5 Jul 2023, HTML/full text via `arxiv.org/html/2111.14517v3` + the PDF downloaded and
`pdftotext`/`pdftoppm`-rendered for exact figure/table values — several numbers below are
read directly off printed tables in the PDF, others off chart curves and are flagged as
visually-approximate). Code: [github.com/bmlklwx/EMS-superquadric_fitting](https://github.com/bmlklwx/EMS-superquadric_fitting)
(MATLAB original + Python/NUMBA port, no license file). Branch: `primitive-structure-from-survey-clouds`.
Catalogued as entry **#13** in [round2 map](../../../round2_map/primitive-structure-from-survey-clouds.md).

---

## 1. Core method / idea

**Generative model (Sec. 3.1).** A point is generated either from a uniform outlier
density `p_o(x) = 1/V` over a bounding volume `V` (probability `w_o`), or from an
isotropic Gaussian `N(x|μ,σ²I)` centred at a point `μ` sampled uniformly on the
superquadric's parametric surface `S_θ` (probability `1-w_o`) — Eq. (2)-(4), the
**Gaussian-Uniform Mixture (GUM)**. Superquadrics are parameterised by `θ = {ε1, ε2,
ax, ay, az, R, t}` via the implicit ("inside-outside") function, Eq. (1); `ε1, ε2 ∈
(0,2]` control shape (cuboid↔ellipsoid↔cylinder-like family), `ax,ay,az` scale, `(R,t)`
pose. Recovery = MLE over `θ, σ²` (Eq. 5-6).

**EMS algorithm (Sec. 3.2-3.5)**, an EM variant:
- **E-step:** nearest surface point via radial projection (Eq. 9, closed-form,
  approximates the true closest-point problem which has no analytical solution); then
  the **posterior inlier probability** per point via Bayes' rule (Eq. 10) —
  `E(z_i=1|x_i,μ̂_i) = N(x_i|μ̂_i,σ²I) / (N(x_i|μ̂_i,σ²I) + w_o·p_o/(1-w_o))`. This is
  the directly-reusable **per-point fit-quality signal**.
- **M-step:** trust-region-reflective optimisation of `θ` (not Levenberg-Marquardt —
  LM is numerically unstable as `ε1` or `ε2` → 0, forcing prior methods to floor them
  at 0.1, which degrades accuracy on sharp-edged shapes like cuboids/cylinders; EMS
  floors at 0 and stays stable, Supp. Sec. 2 / Fig. 1). `σ²` updated in closed form.
  Surface area `A_θ` (needed for the `N·log(A_θ)` small-shape-preference term) has no
  closed form and is approximated by bilinear interpolation over `(ε1,ε2)`, mean
  relative error **<10%** vs. mesh-computed area (Supp. Sec. 1).
- **S-step (the actual contribution):** superquadrics have algebraic/geometric
  symmetries — when `ε1≈ε2` (axis-mismatch similarity) or `ax≈ay` (duality similarity,
  `ε2 ↔ 2-ε2`) — that let the algorithm jump to a **distant point in parameter space
  encoding a geometrically similar shape** and re-run EM from there, escaping local
  optima without random restarts (Fig. 2, Supp. Sec. 3, Algorithm 1 in main text is
  actually the multi-SQ algorithm; the S-step pseudocode is in Supp.). Triggered when
  relative log-likelihood decrease falls below threshold `δ=0.1`. **EMS needs only one
  initialisation vs. 3 PCA-based restarts for every baseline.**

**Multi-superquadric extension (Sec. 3.6):** hierarchical — fit one superquadric,
cluster the posterior-outlier points (Euclidean clustering), recurse on each cluster
(max depth 3, prune clusters <60 pts in experiments), yielding a **primitive hierarchy +
graph + full point-cloud segmentation** (Fig. 3). No pre-segmentation required, unlike
prior multi-SQ work (Leonardis, Borges, Chevalier et al., cited Sec. 2.2).

---

## 2. Concrete results and numbers

### Synthetic ablations (Sec. 4.1)
500 random superquadrics, `ε1,ε2 ~ U(0,2]`, `ax,ay,az ~ U[0.5,3]`, random `SO(3)` pose,
`t ~ U[-1,1]³`; error metric Eq. (12) = mean nearest-neighbour point-to-sample distance
(unitless, in the scale of the `[0.5,3]` box). 4 baselines: Implicit-LSQ, Radial-LSQ,
Robust-fitting, Numerical Stable (NS) — all LM-based, all given 3 PCA-derived inits
unless noted; EMS given 1 init.

- **Partial data (Fig. 4)** — partial ratio (fraction of surface kept, contiguous cap
  around a random seed point) ∈ {1.0, 0.8, 0.6, 0.4, 0.2}, 2500 clouds total.
  - **Global optimality rate, Fig. 4(e), exact printed table** (success = error < 0.01):

    | Partial ratio | 1.0 | 0.8 | 0.6 | 0.4 | 0.2 |
    |---|---|---|---|---|---|
    | Radial-LSQ (1 init) | 33.2% | 28.6% | 30.8% | 29.4% | 34.4% |
    | Radial-LSQ (3 inits) | 99.0% | 78.0% | 70.4% | 68.8% | 65.6% |
    | **EMS (1 init)** | **100%** | **100%** | **99.8%** | **97.4%** | **86.4%** |

    Reading for our branch: even at 20% surface coverage EMS reaches the global optimum
    86.4% of the time vs. 65.6% for the best multi-restart LSQ baseline; a single-restart
    LSQ baseline is essentially flat at ~30% success **regardless of partial ratio** —
    for that baseline family, susceptibility to local optima dominates over incompleteness
    per se.
  - Error curves Fig. 4(a) (baselines ×3 init, visually read): at ratio 1.0 all methods
    ≈0.008-0.01; baselines rise to a peak ≈0.026-0.029 around ratio 0.4 (Robust-fitting
    worst, Radial-LSQ best-of-baselines at ≈0.0165); **EMS stays ≈0.001-0.005 flat down
    to ratio 0.4**, rising only mildly at 0.2.
- **Outliers (Fig. 5)** — Gaussian outliers added, outlier ratio (outliers/inliers)
  0→0.5, 2500 clouds, log-scale error axis. Implicit-LSQ/Radial-LSQ/NS diverge almost
  immediately (not designed to reject outliers at all); Robust-fitting's heuristic
  rejection degrades more gracefully but is *worse than EMS even at outlier ratio 0*
  (text, Sec. 4.1) — its heuristic rejection actively hurts clean data. **EMS stays flat
  and low across the whole 0-0.5 range** — attributed directly to the posterior
  (Eq. 10) vs. heuristic outlier handling.
- **Noise (Fig. 6)** — isotropic Gaussian noise, variance 0→0.01, evaluated against
  the *uncorrupted* cloud. Explicit, important qualifier in the text (Sec. 4.1,
  "Robustness to noise"): **error grows with noise level for every method, including
  EMS** — "unlike outliers, the noise inherently corrupts the geometric information."
  EMS retains the lowest error and runtime throughout, but the gap to baselines is
  visibly narrower here than in the partial-data or outlier ablations.

### Real-world data (Sec. 4.2)
- **KIT ObjectModels** (laser scanner, Minolta VI-900; 97 of 145 objects used — the
  rest excluded as "better recovered with multiple superquadrics", deferred to 4.3;
  downsampled to ~1500 pts). **Fig. 7(c), exact printed table**, error = Eq. (12) at
  0.1 mm sampling density:

  | Method | Radial | Implicit | NS | Robust | **EMS** |
  |---|---|---|---|---|---|
  | Error (mm) | 0.650 | 0.647 | 0.638 | 0.748 | **0.584** |
  | Time (ms) | 290.7 | 249.7 | 256.0 | 1065.0 | **91.1** |

- **BigBIRD** (RGB-D, PrimeSense Carmine 1.08; 91 of 125 objects used; downsampled to
  ~1000 pts). **Fig. 8(b), exact printed table**:

  | Method | Radial | Implicit | NS | Robust | **EMS** |
  |---|---|---|---|---|---|
  | Error (mm) | 0.821 | 0.826 | 0.847 | 0.807 | **0.753** |
  | Time (ms) | 175.0 | 139.0 | 133.9 | 834.9 | **78.9** |

  **Important, checkable, and sobering:** EMS's margin over the *best* classical
  baseline on real data is small — **8.5% lower error on KIT** (0.584 vs. 0.638),
  **6.7% lower on BigBIRD** (0.753 vs. 0.807) — nothing like the multi-fold gaps seen
  in the synthetic partial/outlier ablations. EMS is uniformly the fastest (2-13×) and
  most accurate, but the real-world accuracy story is a modest, not dramatic, win.
  Both real datasets also silently **exclude ~27-33% of objects** as not reasonably
  single-superquadric-approximable — the headline mm numbers describe only the
  superquadric-friendly subset.
- Prior outlier probability `w_o` is **hand-set per dataset/experiment**, not inferred:
  0 (partial-data ablation), 0.2 (outlier ablation), 0.01 (noise ablation and KIT),
  0.05 (BigBIRD — "since the point clouds are captured by a RGB-D camera, resulting in
  more outliers and noise"), 0.8 (multi-SQ extension) — Supplementary Sec. 5.

### Multi-superquadric recovery (Sec. 4.3, Fig. 3, 9, 10)
Purely **qualitative** — a Seal (body cylinder + head sphere + parts, Fig. 3), a bottle
(cylinder body, cap correctly identified as posterior-outliers and recursed into its own
primitive, Fig. 10), assorted animals with ears/mouths recovered as sub-primitives
(Fig. 9). **No ground-truth part-segmentation accuracy is reported anywhere** — there is
no number to cite for "how good is the automatic decomposition," only visual plausibility
on a handful of hand-picked KIT objects.

---

## 3. Load-bearing assumptions

- **Isotropic per-point Gaussian noise, `Σ=σ²I`** (Eq. 3) — a generic noise model, not
  a range-and-incidence-angle-dependent depth-sensor model.
- **Outlier model is spatially uniform random noise in a bounding volume** (`p_o(x)=1/V`)
  — i.e. scattered random 3D points, not a coherent missing patch. Their "partial data"
  experiment (contiguous cap removed around a random seed point) is the much closer
  proxy to real scan-dropout than their "outlier" experiment.
- **No registration/multi-view fusion error model at all.** Corruption is always i.i.d.
  per-point noise or randomly-placed outliers or a single contiguous missing region —
  never a systematic, pose-dependent bias of the kind a few-mm hand-eye/registration
  error or the ~4 cm fused-vs-single-view offset would produce.
- Superquadric pose `(R,t)` is fit jointly with shape/scale — **no canonical-orientation
  assumption**, unlike some learned baselines flagged elsewhere in this branch's map.
- Real-world quantitative evaluation is restricted to objects **pre-selected as
  reasonably approximable by a single superquadric** (see numbers above) — a curation
  step, not automatic; the paper doesn't test or report what happens on the excluded
  objects.
- Runtime figures (91.1 ms / 78.9 ms per object) are on downsampled ~1000-1500-point
  clouds, MATLAB, Intel i9-9900K (2019 desktop CPU) — the Python/NUMBA port claims 3-5×
  further speedup (repo README) but that number is un-benchmarked in the paper itself.

---

## 4. What transfers to our setting

- **The per-point posterior inlier probability (Eq. 10) is a real, off-the-shelf,
  no-retraining fit-quality signal**, directly answering branch question F.14/15: it is
  exposed by the released code (both MATLAB and Python), requires no training, and is
  computed as a normal byproduct of fitting, not an extra step.
- **Fig. 4(e)'s global-optimality-vs-partial-ratio table is a genuinely reusable
  incompleteness benchmark**, even though it's for the superquadric family specifically:
  a robust MLE-style fitter can hold >85% correct-parameter recovery down to 20% surface
  coverage, while naive multi-restart LSQ degrades to ~65-70% and single-restart LSQ is
  a ~30% coin-flip *independent of coverage*. Useful anchor number for our RQ1/RQ2
  "at what incompleteness level does fitting stop working" question, albeit for a
  different failure signature (local-optimum convergence, not incompleteness per se)
  than we might have assumed.
- **No canonical-pose requirement** matches our arbitrarily-posed tabletop object
  directly — a genuine (if unremarked) point in EMS's favour relative to several
  ShapeNet-trained learned methods in this branch's candidate list.
- **The demo script reads a `.ply` directly** and returns fitted parameters + runtime +
  visualisation (`test_script.py file.ply --result --runtime --visualize`) — this is
  literally our input format, zero-friction to trial.
- Runtime (~80-90 ms/object at our-scale downsampled clouds, 2019 desktop CPU) is
  compatible with an interactive/inspection-loop budget.
- **Inference, not a paper claim:** since the fitted surface has a closed-form
  parametric equation (Supp. Eq. 7) and implicit function (Eq. 1), surface normals
  anywhere on a fitted superquadric are analytically computable from the gradient of
  the implicit function — a fitted EMS primitive could in principle hand us exact
  normals (no PCA/jet estimation, no k-neighbourhood tuning) for any point on its
  surface. The paper never states or exploits this; it's an opportunity worth testing,
  not a validated result.
- Multi-superquadric hierarchy (Sec. 3.6) is a plausible **candidate skeleton** for
  "decompose object → treat parts as primitives → plan per-part views" (bridging branch
  question G) — but see the explicit limitation below before leaning on it.

---

## 5. What does NOT transfer / negative findings (first-class)

- **No angular/orientation error is reported anywhere in this paper.** Verified by
  full-text search: zero occurrences of "angle," "degree," "rotation error," "axis
  error," or "orientation error" as a *result*. All accuracy is a single scalar
  point-to-surface Euclidean distance (Eq. 12) or, in the supplementary stability
  figure, a combined log-L2 norm over shape+scale parameters only (pose isn't even in
  that metric). **This paper gives zero direct evidence for branch question B1
  ("what angular error on a recovered face normal/cylinder axis should we expect")** —
  we would have to compute normal/axis angular error ourselves from the released code
  against known ground truth; it is not a byproduct of their evaluation protocol.
- **"Cylinder" is never fit or evaluated as a distinct primitive type.** A cylinder is
  one point in the continuous `(ε1,ε2)` shape space (roughly `ε1→0, ε2→1`); the
  synthetic benchmark samples `(ε1,ε2)` uniformly over the whole convex region, not
  specifically cylindrical or axis-aligned shapes. EMS has **no model-selection /
  primitive-type-recognition story** at all — it doesn't decide "is this a plane, a
  cylinder, or a superquadric," it only ever fits the superquadric family. Branch
  question A2 ("does the object set close with plane+cylinder+sphere+cone, or does it
  force superquadrics") is not engaged either way by this paper.
- **Outlier robustness (Fig. 5) is not evidence about our dropout failure mode.** The
  outlier model is spatially-random scattered points; our concern (glossy/dark-surface
  dropout) is a *coherent missing region*, which is structurally closer to their
  partial-data experiment (Fig. 4) — but even that is a single contiguous synthetic cap,
  not dropout correlated with material/incidence angle the way ours would be.
- **No registration-error modelling whatsoever** — nothing in this paper speaks to
  branch questions A1/D9/E12 about how few-mm registration error or a ~4 cm
  fused-vs-single-view offset propagates into fitted-primitive orientation. The M3C2-EP
  line (separately catalogued, entry #27) is the actual candidate for that; EMS is
  silent on it.
- **Real-world accuracy gain over the best classical LSQ baseline is small (7-8%)**,
  and both real-data benchmarks quietly exclude ~30% of objects as not
  single-superquadric-friendly. Anyone citing "EMS beats baselines on real data" as
  strong evidence of general real-world robustness should read the actual margin —
  it's real but modest, and the exclusion criterion (which objects were dropped and
  why) is not quantified.
- **Multi-superquadric decomposition has zero quantitative validation** — no
  ground-truth part segmentation, no boundary/centroid accuracy, nothing to compare
  against Point2CAD-style bounded-face extraction. The paper is explicit about the
  scope limit (Sec. 5, Discussion & Limitations): the hierarchical extension
  **"requires the object being recovered [to] possess an inherent hierarchical
  geometric structure. For more general cases, a preliminary point cloud segmentation
  is preferred"** — i.e. EMS's multi-primitive mode is not a general-purpose
  segmentation-then-fit pipeline, it only works when outlier-clustering happens to
  align with real object parts.
- **EMS produces one blob per object/part, not addressable bounded faces.** A box
  approximated by a single superquadric is one smooth implicit volume with rounded
  corners (explicitly discussed for KIT box-like objects, Sec. 4.2/Fig. 7b, where the
  fitted shape is "close to cuboids" but not 6 separate face patches). For the branch's
  "scan face 3 orthogonally, need a face centre and a face size" requirement (branch
  question A3), EMS's natural output granularity is object- or part-level, not
  face-level — getting individual face centroids/normals off a fitted superquadric
  would need an extra, unvalidated step of intersecting/sampling the analytic surface,
  not something the paper does or measures.
- **Thin-walled non-convex objects (e.g., a cup) are stated as poorly modelled** by the
  superquadric vocabulary itself (Sec. 5) — a vocabulary-coverage limitation independent
  of the fitting algorithm's quality.
- Requires a **hand-set hyperparameter `w_o`** (prior outlier probability) tuned per
  noise/dataset regime (values ranged 0-0.8 across their own experiments) — "no
  retraining" is true, but "no tuning" is not; we'd need to pick a value for our sensor
  and defect profile, and the paper gives no principled rule for choosing it beyond
  "RGB-D → higher than laser scan."

---

## 6. Mapping to branch questions (direct answers only; see §5 for the rest as negatives)

- **A1 (angular error at our noise/incompleteness level):** not answered by this paper
  — no angular metric exists in it (see §5). Closest transferable anchor is the
  Global-Optimality table (§2) for *parameter-recovery success rate* vs. partial ratio,
  not angle.
- **A3 (bounded faces):** not this paper — EMS gives a whole-object/part implicit
  volume, not per-face extent/centroid (see §5). Point2CAD / Araújo-Oliveira (other
  branch entries) remain the candidates for that.
- **A6 (partial cylinder, symmetry completion):** EMS's partial-data ablation (Fig. 4,
  86.4% global-optimality at 20% coverage) is encouraging *for the superquadric family
  in general*, but was never run specifically on a half-scanned cylinder with
  glossy-side dropout, and gives no axis-angle number — only a scalar shape-fit-error
  success/fail label.
- **F.14/15 (computable, off-the-shelf fit-quality signal):** **yes** — Eq. (10)
  posterior inlier probability, per point, no retraining, released in both MATLAB and
  Python. No paper-internal calibration of this score against downstream task failure
  exists (F.15's second half is still open, as expected).
- **Integrability (17):** demo reads `.ply` directly, pip-installable Python package,
  ~80-90 ms/object at our rough cloud-size scale on a 2019 desktop CPU — a strong
  integrability candidate for the "cover box and cylinder with no training" question,
  *if* box/cylinder are acceptable as "one superquadric" rather than requiring 6
  addressable faces or an explicit axis vector.

---

## Gist (for structured return)

GUM+EM (EMS) is a probabilistic single-superquadric fitter with a genuinely novel
local-optima-avoidance step (S-step) and a directly reusable per-point posterior
outlier/inlier probability. It is dramatically more robust than classical LSQ baselines
on synthetic partial/outlier ablations (e.g. 86.4% vs. 65.6% global-optimality success
at 20% surface coverage, Fig. 4e) but the real-world (KIT/BigBIRD) accuracy margin over
the best classical baseline is modest (~7-8%), and both real-data evaluations quietly
exclude ~30% of objects as not single-superquadric-friendly. Critically for this
branch, **the paper reports no angular/orientation error metric at all** — only a
scalar point-to-surface distance — so it gives zero direct evidence on face-normal or
cylinder-axis angular accuracy. It also has no registration-error model, no
coherent-dropout model (its "outliers" are spatially random points, not missing
patches), no primitive-type recognition (fits superquadrics only, never chooses
plane/cylinder/sphere/cone), and no bounded-face output — a fitted superquadric is one
smooth blob per object/part, not 6 addressable faces. The multi-superquadric hierarchy
extension is qualitative-only, with an explicit stated requirement that the object have
"an inherent hierarchical geometric structure," undermining its use as a general
decomposition-for-view-planning skeleton.
