# Fit4CAD: A point cloud benchmark for fitting simple geometric primitives in CAD objects

- arXiv: 2105.06858 (v3, HTML full text read: https://arxiv.org/html/2105.06858v3)
- Published: Computers & Graphics 102 (2022), pp. 133–143
- Authors: Romanengo, Raffo, Biasotti, Falcidieno et al. (CNR IMATI Genova + LURPA ENS Paris-Saclay)

## Core idea

A benchmark dataset + metric suite for the classic "fit a plane/cylinder/cone/sphere/torus to a
segmented point cloud" task on CAD objects. It is a *measurement instrument*, not a new fitting
method — the paper's own contribution is the dataset, the ground-truth format, and the metric
definitions (Sections 3.1–3.3); it then demonstrates the benchmark on two baseline methods
(Sections 4.1–4.2) as a worked example, not as a comparison of the field.

## Dataset construction (Section 3.1, Fig. 1, Fig. 3)

- 225 point clouds total: 190 train / 35 test.
- Source CAD models: filtered subset of the ABC dataset (mechanical CAD, ~1M models total) plus
  models built directly in Onshape. Authors manually rejected low-quality ABC models.
- Each CAD object → exactly one point cloud, sampled at a chosen density, "optionally manually
  postprocessed by using CloudCompare... to simulate missing data" (verbatim, §3.1). Missing-data
  perturbation is a **manual, geometric point-removal step in CloudCompare** — not a sensor noise
  model. There is **no additive range noise, no registration-error model, no incidence-angle
  dependent noise, no glossy/specular dropout model**. This is purely synthetic CAD sampling with
  optional hole-punching for occlusion.
- 5 primitive families: plane, cylinder, cone, sphere, torus — "polynomial primitives" defined as
  zero-sets of a polynomial (§3.1 intro). No superquadrics, no cuboid/box-as-composite, no
  free-form/B-spline in the fittable vocabulary — B-spline patches are deliberately left
  **unsegmented** in ground truth specifically to test whether methods avoid false-positive
  primitive fits on non-primitive surface (§3.2).
- Primitive-type balance: authors aimed for "each family... should appear in a sufficient number
  of point clouds" (§3.1) but no exact per-type counts are given in the running text (Fig. 1 has
  the actual bar chart, not reproduced here).

## Ground truth format (Section 3.2, Appendix A.1/A.2)

Four files per cloud: (1) raw point coordinates; (2) point-index → primitive-ID assignment, e.g.
`Primitive6:=[4 9 184 185 ...]`; (3) parametric representation per primitive, e.g.
`Primitive6:=[Cylinder, [a1 a2 a3 b1 b2 b3 c1 c2 c3 d1 d2 d3]]` where the surface is the explicit
`u,v`-parametrised map (plane, cylinder, cone, sphere, torus formulas given in full in Appendix
A.1); (4) implicit polynomial coefficient vector (Appendix A.2).

Load-bearing structural fact: **primitives are stored/fit as infinite algebraic surfaces.** The
parametric equations have no bound on `u,v`; there is no boundary curve, no extent/half-length,
no centroid field anywhere in the ground-truth schema or in either evaluated method. A "cylinder"
in this benchmark is an infinite tube with a radius and an axis direction (`c1,c2,c3` in the
parametric form above is literally the axis direction vector) — not a bounded can with two end
caps and a centroid. **This directly answers one of our branch questions: Fit4CAD does not
recover bounded faces (extent/boundary/centroid); it is a pure surface-type-and-pose benchmark.**

## Metrics (Section 3.3)

Two families, both point-set based, **neither is an angular/orientation error metric**:

1. **Classification metrics (§3.3.1)** — per detected primitive vs. its best-matching ground-truth
   primitive, computed over point *membership*, not geometry: TPR = TP/(TP+FN), TNR = TN/(TN+FP),
   PPV = TP/(TP+FP), NPV = TN/(TN+FN), ACC = (TP+TN)/(TP+TN+FP+FN), and Sørensen–Dice/F1
   DSC = 2·TP/(2TP+FP+FN). These measure *segmentation* quality (did the right points get
   assigned to the right patch), not fit *orientation*.
2. **Approximation accuracy (§3.3.2)**, only computed for the parametric/implicit representations:
   - Mean Fitting Error, Eq. (1): `MFE(P_S, S) := (1/|P_S|) Σ_x d(x,S) / l`, point-to-surface
     distance normalized by the bounding-box diagonal `l` — a scale-normalized RMS-ish distance,
     not an angle.
   - Directed Hausdorff distance, similarly normalized by `l`.
   - Coefficient distance `d1(v,v') = ||v - v'||_1` between normalized, sign-standardized implicit
     polynomial coefficient vectors — an abstract parameter-space distance, not directly an angle
     either (though it is dominated by axis/normal mismatch for cylinders/cones/tori, since those
     coefficients encode direction).

**Negative finding, load-bearing for our RQs on angular error and covariance (RQ7/RQ8/RQ9):**
Fit4CAD never computes a normal/axis angular error, anywhere. The ground-truth parametric form
*contains* the axis/normal vector implicitly (visible in the Appendix A.1 formulas), so the raw
material to compute one exists, but the paper's own metric suite does not extract or report it.
**We cannot lift an angular-error number from this paper — we would have to re-derive it
ourselves from the released parametric ground truth**, and even then the dataset has no noise
model resembling ours (no registration jitter, no range-error-vs-incidence-angle, no dropout), so
a number we computed from it still would not transfer quantitatively to our defect profile — only
the extraction *procedure* (compare fitted axis vector to GT axis vector from the parametric
tuple) would transfer.

## Methods evaluated (Sections 4.1, 4.2) — worked examples, not the benchmark itself

**PG ("Primitive Growing"), §4.1** — clustering/region-growing pipeline:
1. Per-point *surface variation* `σ_n(p) := λ1/(λ1+λ2+λ3)` from PCA eigenvalues of a local
   k-neighbourhood covariance matrix (Pauly, Gross & Kobbelt 2002 [22]) — a curvature proxy used
   to find sharp-edge points via a threshold (η = 0.8 in their implementation, i.e. top-20%
   surface-variation points flagged as edges).
2. Region growing between edges gives coarse segments.
3. **Slippage analysis** (Gelfand & Guibas 2004 [24]) refines each coarse segment: finds the
   motion `[r,t]` minimizing `Σ((r×x_i + t)·n_i)²` (Eq. 2) using **per-point normals `n_i` as a
   required input** — slippage class (0/1/2/3-slippable) maps to primitive family via an ISO GPS
   invariance-class table (Table 1): planar↔3-slippable, spherical↔3, cylindrical↔2, prismatic↔1
   (ambiguous, needs further splitting), revolute↔1 (cone/torus, ambiguous), complex↔0
   (undersegmented, needs further splitting).
4. For the ambiguous 1-/0-slippable classes (prismatic/revolute/complex), **Efficient RANSAC**
   (Schnabel, Wahl & Klein 2007 [25]) is invoked as a fallback to split further.

This is directly in our branch's vocabulary: PG is essentially "PCA-normal-based curvature
segmentation + Efficient-RANSAC fallback," so its **per-point normal quality is a hard dependency
for edge/segment detection**, but the paper never ablates normal-estimation error against PG's
output quality — it only reports end-to-end classification/approximation numbers.

**HT ("Hough Transform"), §4.2** — parametric voting: for each candidate primitive family, define
a bounded region of parameter space from the point set's geometric bounds, discretize into a
Hough accumulator, vote, then pick peaks by **topological persistence** (peaks with persistence
>10% of the accumulator's max value are kept). When multiple families fit the same dense subset,
Eq. (1) MFE is used to pick the winning family. **Primitive type is inferred, never given as an
oracle**, in both methods — confirmed directly from method description (HT compares MFE across
competing family fits; PG maps slippage class to family and disambiguates via RANSAC).

## Results (Table 2, Table 3, §4.3; 35 test clouds)

| | PG | HT |
|---|---|---|
| DSC (F1) range | 0.370 – 0.916 | 0.568 – 0.998 |
| TPR range | 0.332 – 0.983 | 0.456 – 1.000 |
| TNR range | 0.968 – 0.998 | 0.995 – 1.000 |
| PPV range | 0.332 – 0.982 | 0.456 – 1.000 |
| ACC range | 0.706 – 0.997 | 0.973 – 0.999 |
| Runtime min/mean/max | 1.7 s / 286.0 s / **19,074 s (~5.3 h)** | 2.6 s / 50.7 s / 358 s (~6 min) |

Hardware: desktop, Intel Core i9 @ 3.6 GHz, Windows 10 (§4.3, "Computational time").

HT dominates PG on every classification metric's lower bound and is ~40–70× more predictable in
worst-case runtime. **HT's worst case (358 s) is a usable per-object budget at our scale; PG's
worst case (5.3 hours) is not** — directly relevant to the runtime branch question, modulo their
cloud sizes (up to ~17,000 points for the worst case, PC 9) not being disclosed as a systematic
size sweep.

**Approximation accuracy (Table 3, HT only — PG has no parametric/implicit fit to score this
way):** MFE 0.001–0.010 (i.e. ~0.1–1.0% of bbox diagonal), directed Hausdorff 0.002–0.019
(~0.2–1.9%), and coefficient distance `d1` mostly small but with two bad outliers on **torus**
primitives: PC 10 → d1 = 1.203, PC 14 → d1 = 1.264 (vs. typical values ≤0.3 elsewhere) — the paper
flags torus fits as the worst case for parameter-space accuracy, though it does not decompose this
into an axis-vs-radius-vs-tube-radius breakdown.

**Failure mode called out explicitly (§4.3):** "both methods rarely suffer from oversegmentation,
[but it] is more likely for them to undersegment" — worst case is PC 9 (17,000 points, 104
ground-truth primitives): PG detects only 43, HT only 40. Authors attribute this to "thin or small
primitives" in the source CAD model, i.e. **failure is driven by primitive scale relative to
sampling density, not by noise** (there is no noise in this dataset). No per-type breakdown of
which family (plane/cylinder/cone/sphere/torus) is hardest is reported anywhere — confirmed
absent by direct search of the text.

**Missing-data robustness (§4.3, Fig. 10):** HT "hardly affected... inter-quartile range and
whiskers do not significantly vary," only TPR shows a slight drop. PG shows "more prominent
variation" under missing data — i.e. **the curvature/region-growing pipeline is materially less
robust to incompleteness than the exhaustive parametric-family search**, on this benchmark's
occlusion model. No quantitative noise/incompleteness *level* sweep exists (no SNR parameter, no
% of points removed reported as a controlled variable) — it's a binary with/without-missing-data
split, not a dose-response curve.

## Author-stated limitations (Section 5)

- Deliberately not aiming for ABC-scale coverage ("we do not aim at a large scale dataset");
  225 clouds is explicit and acknowledged as modest against ABC's ~1M models.
- Only "simple (polynomial)" primitives; explicitly flags future expansion toward "more complex
  primitives and possibly... assembly models" — superquadrics/cuboid decomposition are out of
  scope by the authors' own admission, not just an oversight.
- Anticipate but have not evaluated learning-based fitting methods.
- Computational time flagged by the authors themselves as "the main bottleneck in practice" —
  echoing PG's catastrophic worst case above.

## What transfers to our setting

- **The metric *design pattern* transfers, not the numbers.** DSC/TPR/TNR/PPV/ACC for
  segmentation quality, MFE/Hausdorff for point-to-surface distance, and (with our own extension)
  axis/normal angular error extracted from the same parametric ground-truth tuples — all
  reusable as a metric *shape* for our own benchmark. Normalizing distance metrics by bounding-box
  diagonal (Eq. 1) is a clean, scale-invariant convention worth copying.
- **The ground-truth file schema is directly reusable**: point-index-to-primitive assignment +
  parametric tuple (which embeds axis/normal) + implicit polynomial. If we build CAD-based
  synthetic ground truth for our own objects, this schema (Appendix A.1/A.2 formulas) is a ready
  template, and it already includes what we need to compute axis/normal error ourselves.
  Real-object ground truth (no CAD model) would need the human-annotation substitute this paper
  does not exercise (all of Fit4CAD's models come from parametric CAD source, §3.1) — confirming
  we cannot reuse Fit4CAD's *data*, only its *format*, for the real-scan-with-no-CAD case central
  to our own benchmark question.
- **Efficient RANSAC (Schnabel et al. 2007) and slippage analysis (Gelfand & Guibas 2004)
  reappear here as load-bearing sub-components of a 2021-era "SOTA-adjacent" pipeline**, useful
  corroboration that Efficient-RANSAC-family methods are still the practical fallback even in a
  paper that also tries pure Hough voting.
- **Runtime numbers are a reference point** for what "acceptable" vs. "catastrophic" looks like at
  CAD-object cloud sizes (10^4 points): HT's <6 min worst case is a workable envelope for a
  per-object primitive-fitting budget; PG's 5.3 h worst case is a concrete cautionary data point
  against naive curvature-region-growing pipelines at scale.

## What does NOT transfer, and why

- **No noise model resembling ours.** No registration/pose error, no range-error-vs-incidence
  model, no glossy/dark dropout, no additive Gaussian sensor noise at all — missing data is
  manual geometric hole-punching on a noiseless CAD sample. Any robustness claim ("HT hardly
  affected by missing data") is about *occlusion*, not about *our* defect modes (registration
  jitter + depth bias + dropout), and should not be read as evidence our defects will be similarly
  benign.
- **No angular/orientation-error metric exists in the paper**, despite the ground truth
  containing the needed axis/normal vectors — cannot cite a number for expected axis/normal
  angular error from this source; would have to compute it ourselves from released
  ground-truth + method outputs if code/data were rerun (not done here).
- **No bounded-face recovery** (extent, boundary, centroid) anywhere in the benchmark — the
  "orthogonal-view-needs-a-face-centre-and-size" requirement from our framing doc is entirely
  outside what Fit4CAD measures or what either evaluated method outputs. A method scoring well on
  Fit4CAD's metrics could still be useless for our face-centering action space.
- **Objects are all real mechanical CAD parts (fasteners, brackets, etc.) sampled synthetically,
  never real sensor scans** — so "real-scan generalization" claims cannot be sourced from this
  paper at all; it is a synthetic-CAD-sampling benchmark end to end.
- **No per-primitive-type difficulty breakdown** (which of plane/cylinder/cone/sphere/torus is
  hardest) is reported — despite one strong hint (torus d1 outliers in Table 3) it is not
  generalized into a stated conclusion by the authors; we should not over-read the two torus
  outliers as a general "torus is hard" claim without more evidence.
- **Primitive vocabulary caps at 5 quadric/algebraic families** with torus as the most complex
  member — no superquadric, no cuboid decomposition, no learned abstraction baseline anywhere in
  this paper (consistent with the authors' own §5 admission that this is future work), so
  Fit4CAD gives us zero evidence either way on whether chamfered/rounded real parts force
  superquadrics — that question needs a different source.

## Citation map (for checkability)

| Claim | Section | Figure/Table/Eq |
|---|---|---|
| Dataset size/split | §3.1 | Fig. 1 |
| CAD sourcing (ABC + Onshape) | §3.1 | Fig. 2, Fig. 3 |
| Missing-data simulation via CloudCompare | §3.1 | Fig. 3 |
| Ground-truth file schema | §3.2 | — |
| Parametric surface equations (axis/normal embedded) | Appendix A.1 | — |
| Implicit polynomial form | Appendix A.2 | — |
| Classification metrics (TPR/TNR/PPV/NPV/ACC/DSC) | §3.3.1 | — |
| MFE, Hausdorff, coefficient distance | §3.3.2 | Eq. (1), Eq. (2 is PG's slippage eq, not a metric) |
| PG method (surface variation, slippage, RANSAC fallback) | §4.1 | Fig. 5, Table 1 |
| HT method (accumulator, persistence peaks, MFE-based type selection) | §4.2 | Fig. 7, Fig. 8 |
| Classification results, undersegmentation (PC 9: 104→43/40) | §4.3 | Table 2, Fig. 9 |
| Missing-data robustness comparison | §4.3 | Fig. 10 |
| Approximation accuracy, torus outliers (PC10 d1=1.203, PC14 d1=1.264) | §4.3 | Table 3 |
| Runtime stats, hardware spec | §4.3 | — (inline paragraph) |
| Limitations / future work | §5 | — |
