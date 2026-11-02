# Sanchez, Denis, Coeurjolly, Dupont, Trassoudaine, Checchin (2020) — "Robust normal vector estimation in 3D point clouds through iterative principal component analysis"

**Source:** ISPRS Journal of Photogrammetry and Remote Sensing 163 (2020), pp. 18–35.
DOI 10.1016/j.isprsjprs.2020.02.018. ScienceDirect abstract page returned HTTP 403 (paywalled,
`curl`/WebFetch both blocked). Full text obtained via the open-access author preprint on HAL
(`https://hal.science/hal-02514851`, submitted-version PDF, 41 pages, downloaded with `curl -L
.../document`). HAL's preprint title is "Robust **and edge-aware** normal vector estimation in 3D
point clouds" — same paper, pre-final-title. All content below (sections, equations, figures,
tables) is read from that PDF, cross-checked against the published section/page numbers via the
HAL citation banner (ISPRS JPRS 2020, 163, pp.18-35). Code/data promised "available online" in the
abstract; I could not locate a public repository under the authors' names via GitHub search (no
`julia-sanchez` or LIRIS-affiliated repo matching this paper) — **treat "off-the-shelf, no
retraining" as "reimplementable from the equations" rather than "downloadable," pending a direct
check of the paper's stated code URL (not printed in the PDF I could extract; may be in the final
Elsevier-hosted version behind the paywall).**

---

## 1. Core method

A **per-point normal estimator** (not a primitive fitter, not a segmenter) that replaces plain PCA
with an iteratively-reweighted, M-estimator-robustified PCA, run twice per point to survive sharp
edges. Four load-bearing pieces:

1. **Weighted PCA (§3.1, Eq. 1–2).** Normal is the eigenvector of smallest eigenvalue of a
   *weighted* covariance matrix `C_M = (1/Σm_i) Pᵀ diag(M) P`, with `P` built relative to the query
   point `p₀` itself (not the neighborhood centroid — a deliberate change from classical weighted
   PCA (Pauly et al. 2002) that "leads to better results when introducing a robust estimator," §3.1).

2. **M-estimator weighting (§3.2, §4.1, Eq. 6–8).** Weights come from a scaled Geman-McClure kernel
   `ρ(rᵢ) = μrᵢ² / (2(μ+rᵢ²))`, giving IRLS weight `wᵢ(n) = (μ/(μ+rᵢ(n)²))²` where `rᵢ(n) = n·(pᵢ-p₀)`
   is the point-to-candidate-plane residual. This is the **continuous inlier/outlier separation**
   the paper is selected for: `wᵢ → 1` for points that fit the current normal estimate, `wᵢ → 0` for
   points on the other side of an edge or far outliers (Fig. 1, Fig. 2 color scale 0=blue/outlier to
   1=red/inlier). **This weight is per-neighbor-relative-to-query-point, not a single scalar
   "reliability flag" on the query point itself** — to get a per-point confidence for downstream use
   you'd have to aggregate it yourself (e.g., mean neighbor weight, or fraction below a cutoff); the
   paper does not define or report such an aggregate.

3. **Two-phase optimization per initialization (§4.2.1).**
   - *Rough estimation:* `μ` starts at `μ_init = (max_i rᵢ)²` and is **shrunk by a fixed factor 1.01
     per IRLS iteration** until it reaches a target `μ_lim`; number of iterations =
     `ln(μ_init/μ_lim)/ln(1.01)`. Too-fast decay → local minima; too-slow → wasted iterations. The
     1.01 factor was tuned once and reused everywhere (Fig. B.22 shows the accuracy/time tradeoff:
     mean angle error rises from ~0.75° to ~1.25° as the factor goes 1.0→1.15, while compute time
     collapses from 330s to ~10s — they kept 1.01, near the cheap end, accepting close to the worst
     accuracy in that sweep for speed).
   - *Refinement:* the PCA reference point `p₀` (fixed at the literal query point in the rough
     phase) is iteratively shifted along the candidate normal by `Σwᵢrᵢ(n)/Σwᵢ` and re-optimized.
     Ablation (Fig. B.18/B.19): refinement cuts mean angle error **up to 6×** under Gaussian noise,
     and **up to 90×** when neighborhood sampling is non-uniform across the two sides of an edge
     (density ratio far from 1). This refinement step is the direct fix for exactly the anisotropic
     multi-view point-density problem a fused survey cloud has.

4. **Double initialization for sharp edges (§4.2.2, Fig. 2).** One classical-PCA init (`n₁^init`)
   plus a second init forced 90° away from both `n₁` and the inferred edge direction
   `ν₁ = n₁^init × n₁` (i.e. `n₂^init := n₁ ∧ ν₁`). Both are optimized independently; the final
   choice between `n₁`/`n₂` uses two heuristics: (a) both must point away from the local curvature
   (Eq. 9: `Σᵢ(n_k·(pᵢ-p₀)) < 0`), (b) pick whichever PCA reference point ended up closer to `p₀`
   (`min_k n_k·(p₀ₖ-p₀)`). **This explicitly models exactly one sharp edge (two smooth pieces) in
   the neighborhood — it is not a general junction/corner solver**, though the CAD benchmark (14
   objects with corners) suggests it degrades gracefully rather than failing outright there.

**Convergence parameter `μ_lim` is derived analytically, not hand-picked per point** (§4.2.3, Eq.
not numbered but text-given): `μ_lim = r_max²`, `r_max = X_max + 0.5σ'` where `σ' = σ/√3` (noise std
in one axis) and `X_max = d_max²/(2R_min)` is the max plane-residual expected purely from curvature
of radius `R_min` at the farthest neighbor distance `d_max` (geometric derivation in Fig. 3). So the
**two user-facing hyperparameters are: (1) an estimate of sensor noise σ, (2) a minimum curvature
radius `R_min` the user is willing to "tolerate as flat."** `R_min = ∞` means "assume purely
piecewise-planar" (used for all their indoor-scene / real-LiDAR runs, Table 1); `R_min` finite
(8% of bbox diagonal for the CAD set, 10 cm for a synthetic cylinder model) engages curvature-aware
loosening of the inlier threshold. **Neither parameter is estimated automatically from the data —
both must be supplied**, which matters directly for us since a scanned tabletop object's "reasonable
minimum curvature radius" is not obviously knowable a priori across arbitrary primitive types
(a chamfer radius vs a sphere radius vs a flat face all want different `R_min`).

A cheap **pre-selection filter** (§4.2.4, Eq. 10–12) runs a classical-PCA pass first and only invokes
the expensive iterative optimization for points whose residual std exceeds a closed-form threshold
`τ(R_min, d_max)` — i.e., points already flat within noise skip the M-estimator entirely. Table B.6:
cuts total runtime **34s vs 53s (planes model)** and **14s vs 59s (pipe-bend model)** (~1.6–4×) for a
negligible accuracy cost (mean angle error 0.83° vs 0.72° planes; 0.71° vs 0.68° pipes, both over the
same 16-noise-level sweep).

---

## 2. Concrete results / numbers (checkable against cited figures)

All errors are **mean/RMS angular error of the estimated normal vs. ground-truth**, degrees, unless
noted. 8 baselines compared throughout: PCA (Hoppe et al. 1992), 2-Jets (Cazals & Pouget 2005), VCM
(Mérigot et al. 2011, Voronoi-cell), Bilateral Filter+LOP post-process (Huang et al. 2013), Hough
voting (Boulch & Marlet 2012) and its CNN variant (Boulch & Marlet 2016), PCV (Zhang et al. 2018),
PCP-net (Guerrero et al. 2018, learned).

- **Synthetic two-plane edge** (1×0.5 m, 90° dihedral, 15 000 pts, k=300 neighbors; Fig. 6, 7, 9):
  - Noise-free (σ=0%): method's cumulative-error curve is essentially perfect above the 90th
    percentile (Fig. 6a) — i.e. correctly handles the edge with near-zero smoothing, unlike PCA/2-
    jets/VCM which smooth normals near the edge to errors of tens of degrees for a nontrivial
    fraction of points (>50° tail, §6.1.1 text).
  - Across noise 0→434% of mean nearest-neighbor spacing (Fig. 9), **mean angle error rises smoothly
    from ~0.3° to ~2.2°**, staying below every baseline at every noise level tested; PCA/2-jets/VCM
    sit flat-ish around 3–5° (dominated by edge-smoothing bias that noise barely worsens); Hough-CNN
    is worst at high noise (>6° by 400%).
  - At σ=108% (i.e. noise std ≈ the mean point spacing — plausibly the closest analog to "a
    survey-scan noise floor comparable to point density"), method + PCV are near-tied for best in
    the cumulative histogram (Fig. 6b).

- **Synthetic pipe-bend** (two 0.2 m / r=0.1 m cylinder segments at 90°, 100 000 pts, k=200; Fig. 8,
  10): qualitatively the only method with a clean thin error band exactly on the seam and low error
  (<1°, blue) elsewhere (Fig. 8g); mean angle error over noise 0→220% rises **~0.3°→~1.4–2.2°**,
  again below all baselines (Fig. 10) — PCA sits at ~0.85→1.85°, i.e. **roughly 1.3–3× worse** than
  this method depending on noise level, on a case that is mostly smooth curved surface, not edges.

- **Neighborhood-size sweep** (Fig. 11, averaged over 16 noise levels 0–400%, two-plane model):
  method is essentially monotonically best from k=100 (~0.065°) to k=400 (~0.02°); PCA/2-jets/BF
  have a *sweet-spot-then-degrade* shape (best near k≈150–200, then worse) — meaning "just add more
  neighbors" is not universally safe for the naive baselines but is safe for this method.

- **14 CAD meshes with sharp+curved features** (54k–493k pts each, noise 0–0.2% of bbox diagonal,
  k=200, `R_min`=8% diagonal; Fig. 12, 13): method achieves the **minimal RMS angular error** on this
  set (roughly 11°→17° as noise rises 0→0.2%, reading Fig. 13a), and lowest RMSτ (thresholded RMS,
  errors >10° clipped to 90° before averaging — penalizes smoothing) at roughly 25°→37° vs PCA/2-
  jets/Hough flat near 48–50° and **PCP-net worst at ~75–76° (flat, i.e. it doesn't even improve with
  less noise)** — direct evidence PCP-net does not generalize to this kind of piecewise-CAD geometry.
  PCV ties this method's accuracy but costs **up to 25× more compute** (Table showing PCV up to 6 h
  per CAD model vs this method's max 12 min, and Hough's max 40 min — §6.2 text).
  *(Caveat: even the "best" RMS here is ~11-17°, which sounds large — this is RMS dominated by the
  hardest points across 14 diverse objects including small/thin features and multi-way corners, not
  representative of typical flat-face error; mean-error numbers on the simpler synthetic models above
  are the honest low end, ~0.3–2°.)*

- **Simulated indoor LiDAR** (4 scenes, 320 000 pts, 0.1% Gaussian noise, k=200, `R_min=∞`; Fig.
  14–16): mean angle error per scene — Conference room ≈7.8° (method) vs PCA≈10.5°, BF/PCV/PCP-net
  ≈13–14°; Living room 1 ≈4.0° vs PCA≈6, PCV≈17.5(!), PCP-net≈9.5; Living room 2 ≈3.7° vs PCA≈5,
  PCV≈21(!), PCP-net≈8.5; House ≈4.0° vs PCA≈6.3. **This is the closest proxy in the paper to "real
  sensor sampling anisotropy without registration/fusion error"** — LiDAR raster sampling density
  varies hugely with range and incidence, which is structurally similar (but not identical) to a
  multi-view fused cloud's non-uniform density. Method cuts mean angle error roughly **25–45%**
  relative to plain PCA here, and PCV/PCP-net become unstable/worst (§6.1.3 text: "PCV and Hough-CNN
  algorithms are unstable on noisy inputs with non-uniform samplings").

- **Real LiDAR data**, 3 sensors, no ground truth available — **purely qualitative** (Table 2, Fig.
  17, §6.1.4): Hokuyo/apartment (±3 cm accuracy, 370k pts, 6 mm resolution), Leica/patio (±3 mm,
  3.21M pts, 3 mm resolution), Velodyne/PAVIN (±2 cm, 60k pts, 8 cm resolution). Claims: robust to
  low/anisotropic sampling, recovers curved parts, detects narrow planes (door boundaries in the
  Leica scan). **No quantitative angular-error number exists for any real (non-CAD, non-synthetic)
  point cloud in this paper** — a real caveat for us, since our object is exactly "real cloud, no
  ground truth."

- **Compute time** (Table 3–5, single-threaded, Intel i5-7440HQ 2.8 GHz, fully parallelizable across
  points since each is independent — explicitly noted in Conclusion): on the pipe-bend model,
  11–29 s across k=50→400 (roughly linear in k), competitive with or faster than every non-trivial
  baseline except plain PCA/2-jets/VCM. On real data: **147 s (Hokuyo, 370k pts), 630 s (Leica,
  3.21M pts), 134 s (Velodyne, 60k pts)** at k=100 — i.e. **order ~0.2 ms/point single-threaded** for
  this setting, which scales to our (presumably far smaller, single-object tabletop) survey clouds in
  low seconds even unparallelized. Hough and the bilateral filter either ran far slower (1235 s on
  Hokuyo) or **exceeded memory and failed to complete** on the 3.21M-point Leica cloud (marked "n.a."
  in Table 5) — a data point that dense real scans can break some baselines outright, something this
  method survived.

---

## 3. What transfers to our setting

- **Direct answer to the branch's "noise σ / neighborhood k / angular error" quantification RQ.**
  Fig. 9, 10, 11 give exact curves (not just this method — 6–8 baselines each) relating Gaussian
  noise level (as % of mean NN spacing) and neighborhood size to normal angular error, for both a
  sharp-edge case and a smooth-curvature case. This is the best "get an actual number" source found
  so far for that RQ, with the load-bearing caveat below about *noise model mismatch*.
- **The per-neighbor M-estimator weight `wᵢ` is a legitimate, off-the-shelf-reimplementable,
  no-retraining reliability signal** — directly answers "what computable signals do papers use to
  declare a fit unreliable" for the normal-estimation stage specifically (as opposed to the
  primitive-fitting stage, which is a different literature). It requires no learned model, just the
  IRLS-PCA loop (a few dozen lines of NumPy/Eigen).
- **Evidence that denoising/smoothing filters (bilateral filter here) measurably harm edge accuracy**
  even relative to doing nothing: §6.1.1 explicitly states the bilateral filter "usually correct[s]
  the normals whose neighborhood contains the edge but tend[s] to corrupt the accuracy of the other
  normals," and Fig. 6 shows it has a non-trivial fraction of >10° errors even at **zero noise** —
  i.e. it is smoothing points that PCA alone would have gotten exactly right. Direct, citable evidence
  for the branch RQ "are there reported cases where denoising harms primitive recovery by
  over-smoothing edges."
- **Runtime is cheap and parallelizable** — no obstacle to using this as a preprocessing normal
  estimator ahead of a RANSAC-family primitive fitter on our (much smaller than 3.21M-point) survey
  clouds.
- **A concrete, reusable numeric recipe for setting the noise/curvature hyperparameters** (§4.2.3,
  Fig. 3) if we do adopt this method: `μ_lim` from an assumed sensor σ and a chosen `R_min` — usable
  once we know the D405's per-point depth noise floor.

## 4. What does NOT transfer, and why (negative findings)

- **The paper's entire noise model is isotropic per-point Gaussian jitter added to a synthetic or
  CAD-derived ground-truth surface.** It never models: (a) **registration/fusion error** between
  multiple viewpoints (our few-mm pose error), (b) **the ~4 cm fused-vs-single-view systematic
  offset** flagged in our framing doc, (c) **specular/dark dropout** (missing points forming
  boundary-shaped holes, not added noise), or (d) **range-and-incidence-angle-dependent depth bias**
  (systematic, not zero-mean). The closest proxy tested is *sampling density anisotropy* from LiDAR
  raster geometry (§6.1.3), which is a **different failure mode** from ours — anisotropic *density*
  with correct mean position, vs. our case which is a low-frequency **positional bias** that can
  survive averaging. **This paper gives no evidence about how weighted-PCA-with-M-estimator responds
  to a systematic (spatially correlated, non-zero-mean) bias like a hand-eye or fusion offset** — a
  4 cm offset would very plausibly get treated as a coherent "surface" by the M-estimator rather than
  rejected as an outlier, since the M-estimator is built to distinguish two clean surface pieces, not
  a true surface from a systematically displaced duplicate of itself. This is the single largest gap
  between this paper's guarantees and our defect profile, and it is a **first-class negative finding**
  for the branch: **do not assume this paper's noise-robustness numbers bound our registration/fusion
  error case.**
- **No uncertainty/covariance output.** The method produces a point estimate of the normal plus
  per-neighbor weights; there is no closed-form covariance on the returned normal direction. It
  cannot directly answer the branch's "normal of face 3, ± σ" RQ — at best the weight vector or the
  empirical noise-vs-error curves here could seed a *post-hoc* calibration, but that calibration is
  not in the paper and would need to be built by us.
- **No boundary/extent/centroid output, no primitive fitting at all.** This is exclusively a
  per-point normal (direction) estimator; it says nothing about bounded faces, face size, or face
  centroids. Entirely orthogonal to (and a plausible input for) the branch's primitive-extraction and
  bounded-face RQs — it does not answer them.
- **No treatment of partial/occluded surfaces or symmetry completion.** The pipe-bend test is a fully
  sampled synthetic cylinder pair, not an occluded half-cylinder. Says nothing about our "roughly
  half the surface scanned" cylinder-axis RQ.
- **No repeatability-across-rescans experiment.** Nothing here speaks to RQ3's face-identity/ordering
  stability requirement.
- **Two hand-set hyperparameters per dataset, not automatically inferred.** `σ` (assumed sensor
  noise) and `R_min` (minimum curvature radius to tolerate as flat) are supplied per dataset in Table
  1 (e.g. `R_min=∞` for planar-only scenes, 8% bbox diagonal for the mixed CAD set, 10 cm for a
  synthetic cylinder). For an arbitrarily-shaped tabletop object with mixed geometry (a genuine
  chamfer radius here, a genuine sphere there), picking one global `R_min` is itself a design
  decision this paper does not solve — it assumes the operator already knows it.
- **PCP-net (the one learned method in the comparison) performs worst or near-worst on every
  non-ShapeNet-style test** (CAD objects: worst RMSτ ~75-76°, flat regardless of noise; indoor LiDAR:
  consistently mid-to-worst). This is direct, citable negative evidence against relying on a
  ShapeNet/PCPNet-style learned normal estimator for our kind of piecewise-CAD/real-scan geometry —
  relevant context for the branch's "does SuperDec/SuperFlex generalize to real scans" RQ, even though
  this paper predates both those specific systems.
- **No quantitative evaluation on any real (sensor-acquired) point cloud** — §6.1.4 is qualitative
  only. All the numeric error bounds above come from synthetic or CAD-mesh ground truth. Treat the
  "0.3–2°" best-case numbers as a **ceiling on what's achievable under idealized noise**, not a
  number this paper actually demonstrated on real defective data.

## 5. Bottom line for the branch

Useful, checkable **normal-estimation front end** with a genuine per-neighbor reliability signal and
real noise/neighborhood-size error curves — directly serves the "noise σ vs k vs angular error" RQ
and the "computable unreliability signal" RQ for the *normal-estimation* sub-stage. It does **not**
address primitive fitting, bounded faces, partial-view symmetry completion, covariance propagation,
or — critically — our specific defect modes (registration error, fusion offset, dropout, depth bias),
none of which resemble the isotropic-Gaussian-noise model this paper validates against. Its
error-vs-noise curves should be treated as a **best-case lower bound**, not a number to budget
against for our actual sensor/registration pipeline.
