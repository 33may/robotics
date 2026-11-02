# Reading notes — Schnabel, Wahl, Klein, "Efficient RANSAC for Point-Cloud Shape Detection"

**Venue:** Computer Graphics Forum 26(2), 2007, pp. 214–226 (Eurographics 2007).
**Landing page:** https://cg.cs.uni-bonn.de/publication/schnabel-2007-efficient (JS-only SPA,
no static content — fetched instead via Wayback Machine snapshot of the live PDF endpoint,
`web.archive.org/web/20230325100737/https://cg.cs.uni-bonn.de/backend/v1/files/publications/schnabel-2007-efficient.pdf`,
12 pages, read in full).
**Branch:** `primitive-structure-from-survey-clouds` (Branch 3).
**Map entry:** #2, section A ("Spine"), priority **high**.

---

## 1. Core method / idea

A randomized RANSAC variant that detects **plane, sphere, cylinder, cone, and torus**
primitives in an unorganized 3D point cloud with per-point normals, greedily extracting
the highest-scoring shape, removing its points, and repeating (Algorithm 1, §3). It is a
pure geometry-detection paper — no learning, no training data, no semantics. Two ideas
carry the whole "efficient" claim:

- **Localized (octree-biased) sampling (§4.3.1).** Instead of drawing minimal sets
  uniformly, draw the first point unrestricted, then pick a random octree cell containing
  it (at a randomly weighted level) and draw the remaining points of the minimal set only
  from within that cell. Because shapes are spatially coherent, this makes it far more
  likely that a randomly drawn minimal set is entirely inliers. Eq. 5–6 give the resulting
  per-sample success probability `P_local(n) = n / (N·d·2^(k-1))` where `d` is octree
  depth and `k` the minimal-set size — versus the naive `P(n) ≈ (n/N)^k` (Eq. 1). A
  level-weighting scheme (Eq. 7, §4.3.1.1) adapts which octree level is sampled from as
  point removal proceeds, keeping ≥10% of draws uniform across levels to detect newly
  important scales.
- **Lazy score evaluation via random subsets (§4.5.1).** Rather than scoring every
  candidate shape against the *entire* remaining point cloud, the cloud is split into
  disjoint random subsets `S1...Sr`. A candidate is scored only against `S1` first; its
  true score on the full cloud is estimated as a **confidence interval** via the
  hypergeometric distribution (Eq. 8–9: mean ± stddev of `f(N,x,n)`). Candidates are only
  evaluated against additional subsets when their confidence intervals overlap with the
  current best candidate's — i.e., only when the ranking is actually ambiguous.

**Score function (§4.4, Eq. as `σ_P(ψ) = |P_ψ|`):** count of points that (a) lie within
distance `ε` of the shape, (b) have normals within angle `α` of the shape's local normal,
and (c) belong to the **largest connected component** of compatible points in the shape's
own parameter domain (a 2D bitmap per shape type, resolution `β` ≈ local sampling
density; hemispherical parametrization for spheres, two parametrizations for cones
depending on opening angle — §4.4.1). This connectivity filter is what keeps the
algorithm from merging two coplanar-but-disjoint patches into one "shape."

**Probabilistic stopping criterion (§4.2.1, Eq. 1–4).** For a shape of size `n` in a
cloud of size `N`, minimal-set size `k`, the number of candidate draws `T` needed to
detect it with probability `p_t` is `T ≈ -ln(1-p_t) / P(n)`. The algorithm repeatedly
draws small batches of candidates and, after each batch, checks `P(|ψ_m|, s) > p_t`
(with `p_t = 0.99` throughout the paper) for the current best candidate `ψ_m` — i.e. "how
likely is it that a better candidate exists but wasn't sampled." Once that probability
crosses the threshold, `ψ_m` is accepted and extracted; the whole algorithm terminates
once this holds even for the smallest accepted shape size `τ`.

**Refitting (§4.6):** once a shape is accepted, a least-squares refit (Shakarji '98) is
run using all points within `3ε` of the shape (wider band than the `ε` used for
scoring), "to remove unnecessary clutter."

Minimal-set construction per primitive (§4.1): plane = 3 points (normals only used to
*verify*, not construct); sphere = 2 points+normals (midpoint of shortest segment between
the two normal lines is the center); cylinder = 2 points+normals (axis = `n1 × n2`);
cone = 3 points+normals (apex from intersecting 3 point-normal planes, opening angle
from averaged angle to axis); torus = 4 points+normals (axis from intersecting the 4
point-normal lines, two candidate axes disambiguated by re-fit error, minor radius from
a 3-point circle in a rotated plane).

---

## 2. Concrete results and numbers

### 2.1 Runtime and scale (Table 1, §5)

| model | \|P\| | ε (frac. bbox width) | α (°) | τ | \|Ψ\| shapes | \|R\| remaining | time (s) |
|---|---|---|---|---|---|---|---|
| fandisk | 12k | 0.01 | 10 | 50 | 24 | 38 | 0.57 |
| rocker arm | 40k | 0.003 | 20 | 50 | 73 | 1k | 6.5 |
| carter | 546k | 0.001 | 20 | 200 | 138 | 47k | 29.1 |
| rolling stage | 606k | 0.003 | 20 | 300 | 61 | 16k | 15.1 |
| oil pump | 542k | 0.0015 | 30 | 100 | 202 | 15k | 30.9 |
| master cylinder | 418k | 0.003 | 35 | 300 | 37 | 7k | 12.1 |
| house | 379k | 0.002 | 20 | 100 | 130 | 19k | 10.7 |
| church | 1,802k | 0.002 | 20 | 1000 | 160 | 690k | 40.7 |
| choir screen | 1,922k | 0.002 | 20 | 4000 / 500 | 81 / 372 | 543k / 236k | 20.8 / 61.5 |

(choir screen run twice at two `τ` settings; the `τ=500`, 372-shape run is the one shown
in Fig. 1.) Averaged over 5 runs. All models are architectural/mechanical CAD or laser
scans up to ~2M points — **two to three orders of magnitude larger than a single tabletop
object cloud**, so runtime is very unlikely to be a constraint for us (positive read on
RQ17).

**Ablation (Fig. 4 caption):** on the oil pump model, total runtime was 272.5 s with
subset-scoring disabled, 199.1 s with local sampling disabled, and 12.3 s with both
optimizations on — roughly 20× from stacking the two. (Note: this 12.3 s figure sits
below Table 1's 30.9 s average for the same model; likely a single best-case run vs. a
5-run average, or slightly different parameters — the paper doesn't reconcile the two
numbers.)

**Sampling-strategy example (§4.3.1, Fig. 2):** a small cylinder of ~1066 points found
among ~341.5k remaining points (from an original 2M-point cloud, i.e. relative size
≈1/3000) would require **151,522,829** candidate draws under uniform sampling for 99%
detection probability, vs. **64,929** under localized sampling — "an improvement by three
orders of magnitude, i.e. in this case that is the difference between hours and seconds."

### 2.2 Noise/outlier robustness (§5.1, synthetic sphere octant, Fig. 5, Table 2)

Test: points on one octant of a sphere with synthetic Gaussian noise (`σ`, as % of
sphere diameter) and uniform-random outliers (`o`, %). Normals from local least-squares
plane fits with search radius scaled to noise level.

| σ (%) | o (%) | before-refit μ_r | before-refit μ_c | before-refit σ_r | before-refit σ_c | after-refit μ_r | after-refit μ_c | after-refit σ_r | after-refit σ_c |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.02 | 0.02 | 0.02 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 25 | 1.73 | 2.24 | 1.16 | 0.95 | 0.07 | 0.07 | 0.004 | 0.004 |
| 2 | 25 | 1.79 | 2.60 | 1.43 | 1.47 | 0.31 | 0.31 | 0.021 | 0.023 |
| 5 | 50 | 2.25 | 3.96 | 1.65 | 2.19 | 0.35 | 0.26 | 0.054 | 0.025 |
| 10 | 50 | 8.11 | 11.87 | 4.15 | 4.44 | 4.32 | 7.20 | 0.53 | 0.91 |
| 10 | 80 | 7.17 | 10.58 | 5.18 | 4.68 | 5.12 | 5.99 | 1.65 | 1.96 |

`μ_r`/`μ_c` = mean radius/center-position error, `σ_r`/`σ_c` = their standard deviations
— **all in percent of sphere diameter**, i.e. this is a *position/scale* error, **not an
orientation/normal-angle error** (see §4 negative findings below). Refitting cuts error by
roughly 5–25× at low-to-moderate noise, less so at 10% noise + 80% outliers.

Qualitative statements (§5.1): "for a noise degree higher than 20% the detection starts
to become unstable and the algorithm sometimes returns false shape types. Up to 95%
outliers are tolerated for a noise degree of 2%." On the oil-pump model with Gaussian
noise `σ = 1%` of the bounding-box diagonal, "only small or very narrow shapes (e.g. some
of the screw heads) can no longer be reliably segmented... larger shapes, however, are
unaffected and therefore are successfully and stably detected despite the heavy noise."
Reason given: small shapes both lack enough support points *and* their local normal
estimates become unreliable because the larger neighborhood radius needed to smooth noise
starts including points from adjoining shapes.

**Real-world registration-error example (§5, Fig. 1/6):** the "choir screen" model is a
multi-scan **laser-registered** point cloud ("several registered scans... there also are
some registration errors"). Stated result: "the planar regions and the pillars are well
detected but even for the persons good approximating shapes are found... the small casks
in the hands of the two leftmost persons are reliably detected despite their small size."
This is the paper's only real-scan-with-registration-error qualitative claim — **no
angular or positional error number is given for it.**

### 2.3 Comparison against other range-image segmenters (Table 3, §5.2)

Tested (without adaptation) on the Segmentation Comparison Project benchmark (40 range
images, USF; Powell/Bowyer/Jiang/Bunke '98), scored by required region-overlap threshold
between ground truth and detected regions:

| overlap threshold | BJ88 [Besl & Jain] | UB (Jiang & Bunke '97) | Schnabel et al. |
|---|---|---|---|
| 0.7 | 16.65% | 66.93% | **71.06%** |
| 0.8 | 16.12% | 66.50% | **68.04%** |
| 0.9 | 16.14% | 62.42% | 62.35% |

"Similar or even slightly higher rates... while clearly outperforming the BJ segmenter,"
despite this being a domain (structured range images) the method wasn't designed for.

---

## 3. Load-bearing assumptions

- **Per-point normals are a precomputed input, not something this algorithm produces.**
  §3: "Given a point-cloud P... with associated normals..."; §4.1 references
  Hoppe et al. '92 (local PCA plane fit) as the normal source used in their experiments.
  Any normal-estimation error is inherited wholesale — this paper is silent on how
  sensitive its output is to *which* normal estimator is used or its angular error (that
  question is delegated entirely to the normals literature, map entries #22–26).
- **Per-model manual parameter tuning.** Table 1 shows `ε` (0.001–0.01 × bbox width) and
  `α` (10°–35°) hand-set per model across the 9 examples — there is no universal default.
  This complicates the "off-the-shelf, no training" story: it substitutes parameter
  tuning for training data (relevant to RQ17).
- **The stopping criterion (`P(τ,s) > p_t`, §4.2.1/4.3.2) measures search completeness,
  not correctness.** It answers "how likely is it that a larger/better-scoring candidate
  of the accepted primitive *type* was overlooked" — it is not a statement about whether
  the accepted shape's *type* is the right one, nor about the accuracy of its parameters.
  This distinction matters directly for RQ14/RQ15 (see §5 below).
- **Score extrapolation (Eq. 8–9, §4.5.1) assumes the random subsets `S1...Sr` are
  representative of the whole cloud's inlier distribution.** The paper's own
  level-weighting scheme (§4.3.1.1) exists precisely because outlier/point density is
  *not* homogeneous across real models — but that adaptation targets the *sampling* step,
  not the *scoring* step. Whether the scoring extrapolation is safe under our specific
  defect profile (dropout concentrated on the glossy side of an object, not uniformly
  spread) is not discussed or tested.
- **Deliberately incomplete decomposition.** §6 (Conclusion): "our algorithm, in contrast
  to previous methods in computer graphics, does not find shape proxies for every part of
  the surface... we stress that this is not a requirement in many applications." The
  `|R|` remaining-points column in Table 1 is large and structural, not a bug (e.g. 690k
  of ~1.8M points unexplained for "church").

---

## 4. What transfers to our setting

- **The exact algorithm is available off-the-shelf via CGAL's `Shape_detection` package**
  (map entry #6) — no reimplementation needed, matching RQ17's integrability ask.
- **The probabilistic stopping criterion is a real, already-computed confidence signal**
  we could surface as one input to a fallback gate (RQ14) — but see the caveat above: it
  is evidence about *how exhaustively the shape was searched for*, not evidence that the
  *returned type or pose is correct*. It would need to be paired with something else
  (residual/inlier statistics, or `|R|`/coverage fraction) to be a task-failure predictor.
  This is a genuine, checkable partial answer to RQ14, with an explicit caveat for RQ15.
- **Refitting is cheap and clearly worth adopting regardless of detector choice** — in
  the sphere experiment (Table 2) it reduces radius/center error by roughly one order of
  magnitude at low-moderate noise. Any pipeline we build should keep this stage.
  Refitting uses points within `3ε`, a wider band than detection's `ε` (§4.6).
  Note that this reduction is measured in *position/scale* error, not orientation error —
  we cannot claim refitting fixes normal-angle error from this paper alone.
  (Directly relevant scope for RQ13: this is the one "preprocessing/postprocessing choice
  measurably changes outcome, by how much" data point the paper supplies — though it's a
  post-fit step, not a cloud-preprocessing step like denoising/downsampling.)
- **Real-scan tolerance to registration error and heavy noise is claimed qualitatively**
  on the choir screen (multi-scan, registration error present, "well detected") — weak
  evidence in our favor for RQ1's registration-error question, but not quantified.
- **|R| (unexplained point count/fraction) is already computed and could seed a coverage-
  based fallback signal** — "how much of the object's surface is *not* explained by any
  primitive" is a natural, cheap, complementary criterion to a per-shape confidence score
  (RQ16's two-tier framing).
- **Runtime is a non-issue at our scale.** Their smallest/fastest model (fandisk, 12k
  points) still ran in 0.57 s; our single-object tabletop clouds will likely be smaller
  than every model in Table 1 except possibly fandisk/rocker-arm. This closes RQ17's
  runtime half favorably.
- **Torus inclusion partially helps RQ2** (chamfered/rounded real parts): a torus patch
  can capture a filleted edge or rounded corner locally, so plane+torus can approximate
  *some* rounding without needing a superquadric — though it will not produce one
  unified "rounded-box" primitive; the corner/blend region is still likely to end up in
  `|R|` unless explicitly modeled (see oil pump result, §5: "areas of blending patches or
  small scale details are ignored").

---

## 5. What does NOT transfer / negative findings (checked against every serviced RQ)

1. **No normal/axis angular error is reported anywhere in this paper** (RQ1, RQ7, RQ8,
   RQ10). Table 2 — the paper's only quantitative accuracy table — reports sphere
   radius and center-position error in % of diameter. There is no equivalent table for
   plane-normal or cylinder-axis angular error vs. noise level. **This paper cannot
   answer the branch's central numeric ask** ("what angular error should we expect at what
   noise level"); that number has to come from the normal-estimation literature (map
   entries #22, #23, #24) or from running the algorithm ourselves on our own defect
   profile and measuring it — the paper leaves this measurement to us.
2. **All five primitives are infinite/unbounded parametric shapes** (RQ3). §4.1 defines
   plane, sphere, cylinder, cone, torus purely as parametric equations. Boundedness only
   exists implicitly through (a) the assigned inlier set `P_ψ` and (b) the 2D
   connected-component bitmap used internally for scoring (§4.4.1) — the algorithm never
   emits a face centroid, extent, or boundary polygon as output. To get "face 3's center
   and size" for a "scan orthogonally" action, we would need to post-process `P_ψ`
   ourselves (bounding box / PCA extent in the shape's own parameter plane) or pair this
   detector with something that does emit bounded regions natively — Araújo & Oliveira
   (map entry #5, returns delimited planar patches directly) or Point2CAD's clip/extend
   step (map entry #18) are the better direct hits for RQ3.
3. **Zero connection to view planning.** This is a pure 2007 shape-detection paper; there
   is no mention of cameras, viewpoints, coverage, or downstream robotic use anywhere in
   the text. It does **not** answer RQ4 ("is there a published system that decomposes
   into primitives and then generates per-primitive viewpoints") — it is purely the
   upstream half of such a system, and the paper itself is silent on what would come
   after detection besides "compression, hybrid rendering, shape classification, meshing,
   simplification, approximation and reverse engineering" (§1, Introduction). Map entries
   #38/#39 remain the candidates for actually answering RQ4.
4. **No repeatability / re-scan experiment** (RQ5). The paper never re-runs detection on
   two independent scans (or two noise realizations) of the same object and compares
   segmentation identity or ordering. Extraction order is greedy-by-score (largest first,
   Algorithm 1), which gives a *plausible* argument for rough stability of the largest
   shapes' identity across runs, but this is an inference from the algorithm's structure,
   not a result the paper demonstrates. RQ5 is open.
5. **No partial/occluded-cylinder experiment** (RQ6). The only incompleteness test is a
   synthetic sphere *octant* (Fig. 5/§5.1) — a symmetric, single partial case, not "half a
   cylinder with dropout concentrated on the glossy side." No symmetry-based completion
   (mirror-plane, PotSAC-style axis fitting) is used or discussed anywhere. RQ6 needs map
   entry #35 (robust extrinsic symmetry estimation) or a dedicated experiment of our own.
6. **Noise model is generic isotropic Gaussian + uniform-random outliers** (§5.1), not
   our defect profile (registration jitter, glossy/dark view-dependent dropout, range- and
   incidence-angle-dependent depth bias). The two quantitative degradation thresholds
   given — "false shape types above 20% noise," "95% outlier tolerance at 2% noise" — are
   both measured on the synthetic sphere octant under this generic model, so they
   **cannot be read as a direct threshold for our sensor's actual defect structure**
   (relevant caveat for RQ11, RQ13, RQ19).
7. **No closed-form uncertainty / covariance output** (RQ8, RQ9). The confidence interval
   in Eq. 8–9 is over the *point count* (`σ_P(ψ)`, i.e. how many points support the
   shape), derived from the hypergeometric distribution over subset sampling — it is not
   a covariance on the fitted shape's parameters (position, orientation, radius), and
   nothing here propagates registration/alignment covariance the way M3C2-EP does (map
   entry #27). RQ8/RQ9 remain unanswered by this paper.
8. **No explicit denoising/downsampling ablation** (RQ13). The only "preprocessing"
   response to noise mentioned is enlarging the *normal-estimation* neighborhood radius
   as noise increases (§5.1) — which the paper itself flags as a double-edged tradeoff
   (larger radius smooths noise but pulls in points from adjoining shapes, degrading small
   shapes' normals and hence detectability). There is no voxel-downsampling or
   statistical/radius-outlier-removal experiment. RQ13 is open.
9. **No cuboid or superquadric primitive, and no unified rounded-box vocabulary** (RQ2).
   A box in this framework is always 6 (or fewer, if occluded) separate plane instances
   with no explicit "this is a box" grouping; chamfers/fillets are only covered
   opportunistically by torus fits at the edges, with blend regions typically dropped
   into `|R|` (oil pump result, §5). This paper alone does not close the object-set
   question; map entries #13–17 (superquadric/cuboid abstraction) are needed for that.
10. **Fit-quality-vs-downstream-task-failure calibration is entirely absent** (RQ15). No
    experiment in the paper relates the acceptance probability `p_t`, the score `|P_ψ|`,
    or any other computed quantity to whether a downstream consumer (grasp, pose, view)
    derived from the shape actually succeeds or fails. Confirms the branch's own
    expectation ("If not — and I expect not — we must build that mapping") — this paper
    supplies raw ingredients (score, stopping probability, `|R|`) but no calibration.
11. **Dataset/benchmark questions (RQ19, RQ20) are out of this paper's scope** — its own
    evaluation datasets (CAD/scanned mechanical parts from AIM@SHAPE, USF range-image
    benchmark) are neither Fit4CAD, SHREC2022, nor ABC, and none carry our specific
    defect profile. Not usable as ground-truth infrastructure for our benchmark, only as
    a runtime/robustness sanity check.

---

## 6. Direct mapping to branch questions

| # | Question (abridged) | This paper's contribution |
|---|---|---|
| 1 | Angular error vs. noise/incompleteness | **Not answered.** Only position/radius error (Table 2); no angular number at all. §5 |
| 2 | Does pl+cy+sp+co+to close the object set? | **Partial.** Covers 5 canonical types + torus for local rounding; no unified cuboid/superquadric; chamfer blends fall to `|R|`. §4.1, §5 |
| 3 | Bounded faces (extent/boundary/centroid)? | **No.** All primitives unbounded; boundedness only implicit in `P_ψ` / bitmap. §4.1, §4.4.1 |
| 4 | Precedent for primitive→viewpoint pipeline? | **None in this paper.** Pure detection, no downstream use discussed. §1, §6 |
| 5 | Repeatability across re-scans ("face 3" stability)? | **Not tested.** Greedy-largest-first order is suggestive but undemonstrated. Algorithm 1 |
| 6 | Partial cylinder + symmetry completion? | **Not tested** for cylinders; only a sphere-octant partial-data test exists (Fig. 5). §5.1 |
| 7 | Noise σ / neighborhood k / normal angular error? | **Not answered**; normals are an external input (Hoppe et al. '92), no error curve given. §4.1 |
| 8 | Closed-form orientation covariance? | **No.** Only a hypergeometric CI on point-count score (Eq. 8–9), not shape-parameter covariance. §4.5.1 |
| 9 | Registration covariance propagation (M3C2-EP transfer)? | **No mechanism**; only qualitative "well detected despite registration errors" on choir screen. §5 |
| 10 | Angular error budget vs. reachable-shell/incidence-angle | Not addressed; no incidence-angle-dependent error model here. |
| 11 | Dropout mechanism/magnitude, systematic bias? | **Not addressed** (generic Gaussian+uniform-outlier noise model only). §5.1 |
| 12 | Source of ~4 cm fused-vs-single-view offset | Not addressed (no hand-eye/registration-drift analysis in this paper). |
| 13 | Preprocessing effect on fitting outcomes | **Weak partial**: refitting (post-fit) cuts position error ~5–25×, Table 2; no denoising/downsampling ablation. |
| 14 | Computable unreliability signals? | **Yes, partially** — stopping probability `P(τ,s)` and `|R|` coverage are both computed for free; both are *search-completeness*/*coverage* signals, not correctness signals. §4.2.1, Table 1 |
| 15 | Calibrated against downstream task failure? | **No.** Not attempted; confirms branch's own prior that this must be built. |
| 16 | Two-tier design support (decomposition, else fallback)? | Not discussed; paper doesn't address failure/fallback design at all. |
| 17 | Toolchain coverage + runtime at our scale | **Yes** — CGAL ships this exact algorithm; runtime at 12k–2M points is 0.5–60 s, i.e. a non-issue at tabletop-object scale. Table 1 |
| 18 | Learned methods w/ real-scan generalization | N/A — this is a classical (non-learned) method, no weights/training involved. |
| 19 | Datasets/protocols to seed benchmark | **No overlap** with Fit4CAD/SHREC2022/ABC; own benchmarks (AIM@SHAPE CAD/scans, USF range-image set) are not directly reusable. §5, §5.2 |
| 20 | Ground-truth story for "correct decomposition" | Their own comparison (Table 3) borrows USF's manually-annotated range-image ground truth — evidence that **human-annotated segmentation is an accepted ground-truth protocol** in this literature, relevant precedent for RQ20's cost question. §5.2 |

---

## 7. Bibliographic/provenance note

Fetched from the University of Bonn CG group's current site
(`cg.cs.uni-bonn.de/publication/schnabel-2007-efficient`) — that page is now a JS-only SPA
with no server-rendered content, and its live PDF backend endpoint
(`/backend/v1/files/publications/schnabel-2007-efficient.pdf`) currently 200s but returns
the SPA shell (`text/html`), not the PDF, so it is effectively dead for scripted fetches.
Recovered the actual PDF via the Wayback Machine CDX index, which had a valid snapshot of
that same backend URL from 2023-03-25 (4.3 MB, matches the earlier 2007-era snapshot's
content digest exactly — `NRH4BSQUU5G5OW5HRV6XQO3HYTD4FMK3` — so this is confirmed to be
the unmodified original paper, not a stale/different version). Local copy:
`/tmp/schnabel/schnabel-2007-efficient.pdf` (not committed to the repo).
