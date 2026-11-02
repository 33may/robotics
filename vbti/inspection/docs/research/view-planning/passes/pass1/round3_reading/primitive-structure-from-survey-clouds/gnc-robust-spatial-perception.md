# Yang, Antonante, Tzoumas & Carlone (2020) — "Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection"

**Source:** IEEE RA-L 2020; arXiv:1909.08605v4 (11 Jun 2020). Full text fetched as PDF
(`https://arxiv.org/pdf/1909.08605`) and read via `pdftotext -layout` — all 10 pages (main paper §I–VI +
supplementary proofs §VII–IX + references), including all 5 figures' captions and axis labels. Local copy:
`/tmp/gnc_paper/{gnc.pdf,gnc.txt}`.

**Scope-fit warning up front:** this is a **robust registration / pose-estimation** paper (aligning a point
cloud, mesh, pose graph, or 2D image to a target model given noisy *correspondences*), not a **primitive
extraction / segmentation** paper. It never fits a plane, cylinder, sphere, or cone to an unlabeled point
cloud, never estimates normals, and never produces a bounded face. It is relevant to the branch only as (a)
a candidate drop-in replacement for RANSAC inside a single-primitive robust fit, and (b) a cautionary data
point on how narrow "certificate of solution quality" claims actually are in this line of work. Treat the
rest of these notes accordingly — most branch questions are **not** answered here; that is itself the
finding.

---

## 1. Core method (§III–IV, Fig. 1, Propositions 3–4)

Goal: solve robust estimation `min_x Σᵢ ρ(r(yᵢ,x))` (Eq. 2) **without an initial guess**, for any problem
where a **non-minimal (least-squares) solver already exists for the outlier-free version** (Eq. 1). Three
pieces:

1. **Black-Rangarajan duality (§III-A, Lemma 1, restated from Black & Rangarajan 1996 [11]).** Any robust
   cost `ρ(·)` satisfying mild conditions can be rewritten exactly as a weighted least-squares problem plus
   a penalty on weights `wᵢ∈[0,1]`:
   `min_{x,wᵢ} Σᵢ [ wᵢ r²(yᵢ,x) + Φ_ρ(wᵢ) ]` (Eq. 3). This turns "minimize a nonconvex robust cost" into
   "alternate: (a) weighted-LS solve for x, (b) closed-form solve for weights" — i.e. it's IRLS with a
   principled weight update, not the classic ad-hoc IRLS.

2. **Graduated Non-Convexity (§III-B).** Rather than optimizing `ρ(·)` directly, introduce a surrogate
   `ρ_µ(·)` controlled by a scalar `µ`: convex at one extreme of `µ`, exactly recovers `ρ(·)` at the other.
   Solve a sequence of surrogates from convex → original, using each solution as the next initial guess —
   this is what removes the need for an external initial guess (the convex surrogate can be solved from
   scratch).

3. **Two concrete cost families, worked out in closed form (§IV-B):**
   - **GNC-GM (Geman-McClure, Prop. 3):** `ρ(r)=c̄²r²/(c̄²+r²)`, surrogate `ρ_µ(r)=µc̄²r²/(µc̄²+r²)` (Eq. 4).
     Weight update (Eq. 12): `wᵢ = (µc̄² / (r̂ᵢ²+µc̄²))²` — smooth soft-weight, closed form, no case-split.
     Schedule: start `µ=2r²_max/c̄²` (convex regime), decrease `µ←µ/1.4` each outer iteration, stop at `µ<1`.
   - **GNC-TLS (Truncated Least Squares, Prop. 4):** `ρ(r)=r²` if `r²≤c̄²`, else `c̄²` (Eq. 5) — literal
     hard-inlier/outlier truncation, the same cost RANSAC's consensus set implicitly optimizes. Weight
     update (Eq. 14) is a **3-piece closed form**: `wᵢ=1` for small residual, `wᵢ=0` for large residual, and
     a smooth interpolation formula `wᵢ = (c̄/r̂ᵢ)√(µ(µ+1)) − µ` in between. Schedule: start `µ=c̄²/(2r²_max−c̄²)`
     (convex regime), increase `µ←1.4µ`, stop when `Σwᵢr̂ᵢ²` converges.
   - **Both require a hand-set threshold `c̄`** ("the maximum error expected for the inliers", Remark 5) —
     functionally the same knob as RANSAC's inlier distance threshold. GNC does **not** remove this
     hyperparameter; it removes the *random sampling* and *initial guess*, not the threshold-tuning problem.

**Algorithm shape (§IV-A, Eq. 9–10):** alternate (1) variable update — weighted-LS solve for `x` using
*whatever non-minimal solver already exists* for the outlier-free problem, and (2) weight update — closed
form per Prop. 3/4. All weights initialized to 1. This is the whole method; no learned components, no
training data, deterministic given `c̄`.

---

## 2. Applications and concrete numbers (§V, Figs. 2–5)

All four applications are **registration/alignment-with-known-correspondences** problems, not primitive
discovery. Each swaps in a different existing non-minimal solver as the "variable update" step.

### 2a. Point cloud registration (P-REG), Stanford Bunny — Fig. 2
Setup: Bunny scaled to unit cube, `N=100` correspondences, Gaussian noise `σ=0.01` on inliers, outliers are
random points (not a sensor-realistic noise model). Compared: RANSAC (1000 iter, 99% conf, Horn 3-point +
consensus refinement), ADAPT [27], **TEASER** [2] (the *original*, non-"++" TEASER, Yang & Carlone RSS
2019 — a *different* paper from this one, cited as a specialized baseline here).
- RANSAC, GNC-GM, GNC-TLS, ADAPT **all break at 90% outliers**; below 90% all four achieve similar
  rotation/translation accuracy (Fig. 2a–b).
- **TEASER outperforms all of them in accuracy** but "currently does not scale to large problem instances
  (>5 min runtime) and does not extend to other registration problems" (§V-A, explicit negative finding
  about the TEASER line — the accuracy leader is also the one that doesn't scale).
- Runtime at 80% outliers (Fig. 2c numbers stated in text): **RANSAC 218 ms, GNC-GM 22 ms, GNC-TLS 23 ms**
  — ~10x speedup over RANSAC, with GNC's iteration count staying roughly constant vs. outlier rate while
  RANSAC's grows very fast and ADAPT's grows linearly.
- Note: "GNC-GM is essentially the same as Fast Global Registration [Zhou et al. ECCV 2016]" (§V-A) — GNC-GM
  is not new for this specific problem, just reframed under the general theory.

### 2b. Mesh (generalized) registration, PASCAL+ "car-2" — Fig. 3
Point cloud → mesh registration using point-to-point/line/plane correspondences (non-minimal solver from
Briales & Gonzalez-Jimenez [8]). Gaussian noise `σ=0.05`; 40 point-to-point + 80 point-to-line + 80
point-to-plane correspondences. **This is the closest thing in the paper to "primitive-aware" fitting** —
but the planes/lines are **known target primitives on a known mesh model**, not primitives discovered from
an unlabeled scan. It solves "given a point and a plane it's supposed to lie on, find the rigid transform,"
not "find the planes."
- GNC-GM, GNC-TLS, ADAPT robust to **80% outliers**; RANSAC (12-point minimal solver [17], 10,000 iter)
  **breaks at 50%**.
- Qualitative: GNC-TLS succeeds, RANSAC fails, at 70% outliers (Fig. 3e–f).

### 2c. Pose graph optimization (PGO), INTEL/CSAIL — Fig. 4
Not point-cloud-relevant to us directly, included for completeness: GNC-GM/TLS/ADAPT robust to 70–80%
(INTEL) and up to 90% (CSAIL) outliers; g2o (non-robust) degrades immediately; DCS/PCM (specialized local
robust solvers) degrade gradually with outlier %.

### 2d. Shape alignment (SA) — 2D image ↔ 3D shape, FG3DCar — Fig. 5
**New contribution:** first non-minimal solver for shape alignment (scale `s`, rotation `R`, translation `t`
from weak-perspective 2D-3D correspondences), via quaternion reparameterization (Prop. 6, Eq. 15–17) reduced
to minimizing a degree-4 polynomial in 4 variables, relaxed to an SOS/SDP problem (Eq. 17), solved with
GloptiPoly 3 in Matlab, "empirically the relaxation is always exact," **SDP solve ≈80 ms** on a desktop.
- GNC-GM, GNC-TLS, ADAPT robust to 70% outliers; RANSAC breaks at 60%; Zhou's convex-relaxation method [15]
  "degrades quickly with increasing outliers" even at low outlier rates.

**Aggregate headline claim (Abstract, Conclusion):** "robust to 70-80% of outliers, outperform RANSAC, more
accurate than specialized local solvers, faster than specialized global solvers" — this is a rounded summary
across four *different, non-primitive-fitting* tasks, each with its own outlier-type definition (random
points / bad correspondences / bad loop closures / bad landmark matches), not a single universal number.

---

## 3. On the "certificate of solution quality" — the load-bearing negative finding for this branch

The task brief selected this paper partly for "a certificate of solution quality... the confidence needed
to gate the uniform-sphere fallback." Reading the actual text, this claim needs sharp qualification:

- **The abstract itself states GNC has no such certificate:** *"Although GNC's global optimality cannot be
  guaranteed, we demonstrate the empirical robustness..."* (Abstract). The Conclusion (§VI) restates this as
  open future work: *"Future work includes investigating a priori and a posteriori conditions that
  guarantee convergence of GNC to globally optimal solutions."* As of this paper (2020), **no certificate
  exists for the GNC-robustified solution** — only empirical robustness curves (Figs. 2–5).
- **The one certificate that does exist in this paper is narrower still.** It applies only to the SOS
  relaxation of the *shape-alignment non-minimal solver's inner subproblem* (Eq. 17, Prop. 6) — i.e., given
  fixed weights `wᵢ`, is the weighted-least-squares pose/scale solve for 2D-3D correspondences globally
  optimal. It says nothing about whether the *overall* outlier-robust GNC iteration (weights + pose jointly)
  found the correct/optimal answer, and it is specific to weak-perspective 2D-3D shape alignment — not
  transferable to plane/cylinder/sphere/cone primitive fitting.
- **The actual TEASER-line certificate the branch brief is probably thinking of is in a different paper.**
  This paper cites and compares against "TEASER" [2] (Yang & Carlone, RSS 2019, arXiv:1905.12536, "A
  polynomial-time solution for robust registration with extreme outlier rates") as a *baseline*, and notes
  it doesn't scale past small problems and doesn't generalize beyond one registration variant. The
  sub-optimality-bound / certificate machinery associated with "TEASER++" (Yang, Shi & Carlone, T-RO 2020,
  arXiv:2001.07715) is a **separate, later paper**, not covered by this reading. **Action for the branch:
  if a computable optimality certificate is actually needed, TEASER++ (not this GNC paper) is the correct
  document to read next** — this GNC paper explicitly disclaims having one for its general method.
- Net: for the failure-detection RQ ("what computable signals do papers use to declare a fit unreliable"),
  this paper's only usable computable signal is the **converged weight vector `wᵢ`** (soft inlier/outlier
  membership, Eq. 12/14) and the **final weighted residual sum** used as its own convergence criterion
  (Remark 5) — both are heuristic diagnostics from the fit itself, not a certified bound, and neither is
  calibrated in this paper against any downstream task-success outcome.

---

## 4. Load-bearing assumptions

- **A non-minimal (closed-form or globally-solvable) least-squares solver must already exist for the
  outlier-free version of the problem** (footnote 1, §I). GNC contributes nothing to *finding* candidate
  models — it only robustifies the *fitting* step once a parametric model and correspondence structure are
  given. For plane/cylinder/sphere/cone fitting this is plausible (closed-form or SVD-based LS fits exist
  for all four primitive types), but GNC does not supply the missing piece our branch actually needs most:
  **multi-model segmentation** (deciding *which* points belong to *which* primitive, and how many
  primitives exist) — see §6 below.
- **A hand-tuned inlier threshold `c̄`** is required for both GM and TLS variants (Remark 5) — same class of
  tuning burden as RANSAC's distance threshold, not eliminated.
- **All four experiments use synthetic/correspondence-level outliers** (random 3D points, random incorrect
  correspondences, random relative-pose measurements, random landmark mismatches) — **not** a sensor-driven
  noise model (no range bias, no incidence-angle dependence, no specular dropout, no registration-covariance
  propagation). None of the branch's specific defect modes (registration jitter, glossy/dark dropout, ~4cm
  fused-vs-single-view offset) are modeled or tested anywhere in this paper.
- **GNC assumes a single global model per problem instance** (one transform, one pose, one shape) with
  correspondences that are individually right-or-wrong (inlier/outlier), not a *multi-instance* setting
  where the cloud must be partitioned into several coexisting primitives.

---

## 5. What transfers to our setting

- **GNC-TLS/GNC-GM as a candidate RANSAC replacement for a single, already-hypothesized primitive fit.**
  If a plane/cylinder/sphere/cone LS solver exists (all do — SVD plane fit, cylinder-axis LS fit, etc.),
  GNC-TLS could robustify that specific fit deterministically, without random restarts, at ~10x the speed
  of RANSAC in the one apples-to-apples timing comparison given (22ms vs 218ms at 80% outliers, point cloud
  registration case, Fig. 2c) — a concrete, checkable number, but from a registration task, not primitive
  fitting; would need re-validation on our problem before trusting the ratio.
- **The closed-form weight-update formulas (Eq. 12, 14) are cheap, off-the-shelf, no-training signals** —
  directly relevant to the "computable signals from an off-the-shelf implementation with no retraining"
  branch question, as a soft per-point inlier confidence, though not calibrated to task outcomes (see §3).
- **The explicit disclaimer that GNC lacks a global-optimality certificate, and that TEASER's specialized
  certificate doesn't scale / doesn't generalize**, is itself useful negative information for scoping what
  "GNC/TEASER-family confidence" can realistically deliver for the uniform-sphere-fallback gate — don't
  budget on getting a certified bound from this line of work without reading the TEASER++ paper specifically
  and checking whether its certificate applies to a fit type we need (it targets registration, likely not
  primitive fitting either).

## 6. What does NOT transfer, and why

- **No primitive extraction/segmentation at all** — no plane, cylinder, sphere, or cone is ever fit to raw
  unlabeled points in this paper; every application assumes correspondences to a *known* target (mesh,
  previous pose graph node, 3D shape model, 3D landmark set) are already given, just corrupted by outliers.
  Directly negative for the branch's core "does the object set close with plane+cylinder+sphere+cone RANSAC"
  question — this paper offers zero primitive-fitting evidence either way.
- **No normal estimation, no bounded-face/extent recovery, no centroid+size output** — none of the
  branch's "scan this face orthogonally" geometry (face centre, face size) appears anywhere; the outputs are
  rigid transforms (R,t[,s]), not shapes.
- **No uncertainty/covariance model for fitted parameters** — weights `wᵢ` are soft inlier indicators, not a
  calibrated variance; no propagation from sensor range/incidence-angle error or registration covariance
  into any output quantity. Does not touch the M3C2-EP-style covariance-propagation branch question.
- **No multi-instance / multi-primitive discovery** — GNC solves one global model per call; deciding how
  many primitives exist and which points belong to which (the actual hard part of "decompose this object
  into primitives") is out of scope and unaddressed. This is the single biggest scope gap versus the
  branch's needs.
- **No repeat-scan identity/consistency experiment** — nothing here speaks to RQ3's "does face 3 stay face
  3" concern; there is no notion of a persistent primitive identity across runs.
- **No real sensor defect profile tested** — all outlier/noise injection is synthetic and correspondence-
  level (random points/pairs), not derived from an actual depth-sensor error model; doesn't inform the
  depth-dropout-bias or registration-covariance branch questions.
- **Accuracy leader (TEASER) explicitly doesn't scale** (>5 min runtime past a certain problem size, per
  §V-A) — a caution against assuming "more certified/more accurate" implies "usable at our cloud sizes."

---

## 7. Section/figure/equation index (for checkability)

- Abstract: no-certificate disclaimer, 70–80% outlier robustness headline claim.
- §I (Intro): problem setup Eq. 1–2, contributions list, footnote 1 (non-minimal-solver-required assumption).
- §III-A (Black-Rangarajan duality): Lemma 1, Eq. 3.
- §III-B (GNC): general idea, Fig. 1 (GM and TLS surrogate shapes vs. µ).
- §IV-A (GNC algorithm overview): Eq. 7–10, alternating variable/weight update steps.
- §IV-B (GM/TLS instantiation): Prop. 3 (GNC-GM, Eq. 11–12), Prop. 4 (GNC-TLS, Eq. 13–14), Remark 5
  (implementation/scheduling details, threshold `c̄`).
- §V-A (P-REG/G-REG): Fig. 2 (Bunny, outlier rates 60–95%, incl. 218/22/23ms runtime numbers in text),
  Fig. 3 (mesh registration "car-2", outlier rates 10–90%).
- §V-B (PGO): Fig. 4 (INTEL/CSAIL).
- §V-C (Shape alignment): Prop. 6 + Eq. 15–17 (quaternion/SOS solver), Fig. 5 (FG3DCar, outlier rates
  10–80%), 80ms SDP solve time (text, §V-C).
- §VI (Conclusion): explicit statement that a priori/a posteriori global-optimality certificates for GNC
  remain future work.
- Supplementary §VII–IX: closed-form proofs of Prop. 3, 4, 6 (weight-update derivations, Eq. A18–A34).
- Baseline "TEASER" = ref [2] = Yang & Carlone, RSS 2019 (arXiv:1905.12536) — distinct paper from
  "TEASER++" (Yang, Shi & Carlone, T-RO 2020, arXiv:2001.07715), which is NOT covered by this reading and
  is the more likely source of an actual point-cloud-registration optimality certificate.
