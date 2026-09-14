# Supervised Fitting of Geometric Primitives to 3D Point Clouds (SPFN, CVPR 2019)

**arXiv:** 1811.08988v4 (28 Jan 2020) | **PDF:** https://arxiv.org/pdf/1811.08988
**Authors:** Lingxiao Li*, Minhyuk Sung*, Anastasia Dubrovina, Li Yi, Leonidas Guibas (Stanford / Facebook AI Research)
**Code:** https://github.com/lingxiaoli94/SPFN — MIT license, TensorFlow 1.10, PointNet++ backend
requiring custom CUDA op compilation. **No pretrained weights released** — training scripts and
the ANSI dataset are provided, but any use on our data requires training from scratch on our own
labeled primitives. (Repo existence/contents checked via GitHub fetch this session, not exhaustively
audited.)

Read via direct PDF extraction (11 pages, all figures/tables/equations rendered) — high confidence
in transcribed numbers and equations; cross-checked against a separate HTML-summary fetch for
consistency (both agreed).

## Core method

End-to-end network that fits a *varying number* of primitives (plane/sphere/cylinder/cone) to a
point cloud with no per-input parameter tuning. Three-stage pipeline (Fig. 2, Sec. 3):

1. **Backbone:** PointNet++ over input cloud `P ∈ R^{N×3}` (N=8192 in all experiments) → three
   per-point predictions: soft point→primitive membership `Ŵ ∈ [0,1]^{N×K}` (row-softmax), unoriented
   unit normals `N̂ ∈ R^{N×3}`, and primitive type `T̂ ∈ [0,1]^{N×L}` (L=4: plane, sphere, cylinder,
   cone).
2. **Primitive reordering (Sec. 3.1):** ground-truth primitives are matched to predicted columns via
   Hungarian matching on pairwise **Relaxed IoU** (Eq. 1: `RIoU(w,ŵ) = w^Tŵ / (‖w‖₁+‖ŵ‖₁-w^Tŵ)`).
   Network does **not** assume any ground-truth column ordering — order is resolved purely by the
   matching step at train time.
3. **Differentiable model estimation (Sec. 3.2):** given the predicted membership column, normals,
   and points, each primitive's parameters are computed by a **closed-form / linear-least-squares**
   solve (not gradient descent), independently per primitive, fully backprop-able through SVD and
   Cholesky:
   - **Plane** `(a,d)`: `D²=(a^Tp-d)²` (Eq. 2); weighted least squares reduces to homogeneous LSQ
     solved by SVD, normal = smallest right singular vector (Eq. 3–4).
   - **Sphere** `(c,r)`: true distance `(‖p-c‖-r)²` (Eq. 5) is non-quadratic, so they instead
     minimize a *different* distance `(‖p-c‖²-r²)²` (Eq. 6) — quadratic in `c`, solved via Cholesky
     (Eq. 7). This substitute-objective trick (not the true residual) is reused for cylinder/cone.
   - **Cylinder** `(a,c,r)`: `D²=(√(v^Tv-(a^Tv)²)-r)²`, `v=p-c` (Eq. 8). Two-stage: axis `a` first
     estimated from **point normals** via homogeneous LSQ (Eq. 9, "inspired by Nurunnabi et al."),
     then points projected onto the plane ⊥ `a` and circle-fit using the sphere formulation (Eq. 6)
     in 2D.
   - **Cone** `(a,c,θ)`: `D²=(‖v‖·sin(min(|α-θ|,π/2)))²` (Eq. 10). Multi-stage: apex `c` = intersection
     of tangent planes via LSQ using predicted normals (Eq. 11), axis `a` fit as the plane-normal
     through the point normals (reuses Eq. 3/4 form with `P`→`N̂`), half-angle `θ` = weighted average
     (Eq. 12).
   - **Cylinder and cone axis estimation are both directly driven by the network's predicted point
     normals**, not by the points alone — normal quality is load-bearing for axis accuracy in this
     architecture, not just a downstream nicety.

**Loss (Eq. 13):** unweighted sum `L = L_seg + L_norm + L_type + L_res + L_axis`.
- `L_seg` (Eq. 14): `(1/K)Σ(1-RIoU(W_k,Ŵ_k))`.
- `L_norm` (Eq. 15): `(1/N)Σ(1-|N_i^T·N̂_i|)` — absolute cosine (normals are **unoriented**,
  i.e. no inside/outside convention).
- `L_type` (Eq. 16): per-point cross-entropy, unassigned points ignored.
- `L_res` (Eq. 17): expected squared distance from **M=512 points sampled uniformly on the
  bounded ground-truth surface `S_k`** to the fitted primitive — every primitive weighted **equally
  regardless of physical scale/area**, explicitly to keep tiny primitives from being ignored ("This
  allows us to detect small primitives that can be missed by other unsupervised methods"). Uses the
  **ground-truth type** `t_k` at train time (not predicted type), to decouple type-prediction error
  from fitting-loss instability — at test time, type is decided separately (Eq. 18, per-point-vote
  argmax over `T̂` weighted by `Ŵ`).
- `L_axis` (Eq. 19): cosine loss `1-|a^Tâ|` on plane normal / cylinder axis / cone axis (zero for
  sphere) — added specifically because SVD-based axis estimation is numerically unstable when the
  weighted point set is nearly degenerate (near-empty, or points lying near a line/narrow band).

**Numerical stability controls (Supp. S.1)**, relevant because our fused clouds are exactly the kind
of imperfect input that stresses these paths: custom SVD-gradient clipping
`K_ij=1/(sign(σ_i-σ_j)·max(|σ_i-σ_j|,ε))`, `ε=1e-10`; Cholesky solved with L2 regularizer `λ=1e-8`;
**if the LSQ system's condition number exceeds 1e5, the problem is trivialized (`X=0`) to kill
gradient flow rather than let it blow up.** This is a real, explicit acknowledgment that degenerate
segments (near-planar strips going into an axis fit, thin point sets) require special-cased fallback
handling even inside a from-scratch-trained network — not just an inference-time RANSAC problem.

## Dataset

**Novel benchmark introduced by this paper:** ANSI Mechanical Component Dataset, built from
TraceParts CAD models. 504 categories, 100 models/category max, **13,831 train / 3,366 test,
split by category** (train and test categories are disjoint — this tests cross-category
generalization of the *fitting* skill, not memorization of specific part shapes). The four
primitive types (plane, sphere, cylinder, cone) cover **94.0% of average per-model surface area**
(Sec. 4.1) — the residual 6% is presumably fillets/chamfers/threads/blends not covered by the
vocabulary, though the paper does not break this down further.

Points: N=8192 sampled from the exact CAD surface (mesh sampled then reprojected to true geometry,
so ground truth points are exact, not mesh-facet-biased). Shapes normalized to unit bounding box
(center of mass at origin, axis-aligned bbox ⊂ `[-1,1]^3`). **Noise: a single fixed level** —
per-point offset along the surface normal, uniform in `[-0.01, 0.01]` (i.e. ±1% of the half-extent
of the normalized box) — applied to every training and test sample. **No noise-magnitude sweep is
run anywhere in the paper.** `K_max=24` (20 max ground-truth primitives + 4 slack columns);
predicted primitives with fewer than `ε_discard=0.005N` assigned points are discarded at test time
(a bare point-count threshold, not a calibrated confidence score).

## Evaluation metrics (Sec. 4.2)

Segmentation mean IoU; primitive **type accuracy** (%); mean **point normal** angular error (°);
mean **primitive axis** angular error (°, measured **only when predicted type is correct** — this
matters, it means the headline 1.51° axis number below is conditioned on already having gotten the
type right); `{S_k}` fitting residual mean±std and coverage at ε=0.01/0.02 (fraction of ground-truth
surface within ε of the fitted primitive); `P` coverage at ε=0.01/0.02 (fraction of *input points*
within ε of their matched fitted primitive).

## Quantitative results

**Table 1, full SPFN (row 13):** Seg IoU **77.14%**, primitive type accuracy **96.93%**, point
normal error **8.66°**, primitive axis error **1.51°** (conditional on correct type), `{S_k}`
residual **0.011 ± 0.131** (normalized units), `{S_k}` coverage **86.63% (ε=0.01) / 91.64% (ε=0.02)**,
`P` coverage **88.31% (ε=0.01) / 96.30% (ε=0.02)**.

**vs. Efficient RANSAC (CGAL impl., Schnabel et al.), Table 1 rows 1–6** — the paper's headline
comparison:
| Row | Config | Seg IoU | Type acc | Normal° | Axis° | `{S_k}` cov ε=0.01 |
|---|---|---|---|---|---|---|
| 1 | Eff.RANSAC + jet normals, 8k pts (same res as SPFN) | 43.68% | 52.92% | 11.42° | 7.54° | 43.42% |
| 2 | Eff.RANSAC + jet normals, **64k pts** (8× SPFN's input) | 56.07% | 43.90% | 6.92° | 2.42° | 56.95% |
| 6 | Eff.RANSAC given the network's **own** predicted normals+segmentation+type | 60.56% | 93.13% | 8.15° | 7.02° | 61.94% |
| 13 | **SPFN** | **77.14%** | **96.93%** | **8.66°** | **1.51°** | **86.63%** |

Even at **8× the point resolution**, RANSAC's coverage stays ~30 points below SPFN's; even when
handed the network's *own* accurate segmentation+normals+type predictions (row 6), RANSAC's axis
error (7.02°) is still **>4.6× worse than SPFN's (1.51°)** — segmentation quality lifts RANSAC's
coverage/type accuracy substantially but does **not** fix its parameter/axis precision. This is
direct evidence that the precision gap is in the *fitting* step, not just the segmentation step —
relevant to the branch's efficient-RANSAC-vs-learned question.

**vs. Direct Parameter Prediction Network (DPPN, Sec. 4.4, row 7)** — a same-backbone ablation that
skips the per-point-property intermediate and regresses `4·K_max` primitive parameter sets directly,
matched to ground truth via Hungarian matching on `L_res` alone: Seg IoU 44.05%, type acc **51.33%**,
axis 3.68°, `{S_k}` cov 46.99%. This is the paper's direct evidence for its central architectural
claim — "predict per-point properties, then fit algebraically" clearly beats "regress parameters
directly," even when both are supervised end-to-end with the same backbone and same amount of
supervision. Relevant precedent if we ever consider a direct-regression learned alternative.

**Ablations (Table 1, rows 8–11), each removing one loss term:**
- No `L_seg`: **worst degradation of all** — `{S_k}` cov drops to 50.04% (vs 86.63% full). Segmentation
  quality is the single most load-bearing loss term for the whole pipeline.
- No `L_norm`, substituting normals from **jet fitting on 64k points** (Cazals & Pouget) instead of
  predicted: normal error itself *improves* to 6.87° (more accurate normals, in isolation), yet
  `{S_k}` coverage still **drops** to 76.47% vs SPFN's 86.63%. Direct textual finding (Sec. 4.5):
  *"This implies that SPFN predicts point normals in a way that better fits primitives rather than
  to just accurately predict the normals."* — i.e., normals jointly optimized for the downstream fit
  outperform normals that are individually more geometrically accurate. Important nuance for the
  branch's normal-estimation-accuracy questions: **normal angular accuracy alone is not the
  right proxy for fitting quality** in a system where normals feed a task-specific estimator.
- No `L_res`: `{S_k}` cov drops to 79.81%.
- No `L_axis`: type accuracy barely moves (96.47% vs 96.93%), but **axis error jumps to 6.27° from
  1.51°** (>4× worse) — this loss term is specifically what buys tight axis precision; without it,
  segmentation/type can still be "correct" while the geometric axis is badly off. This is a sharp
  illustration of the branch's introspection concern: **type-correctness and axis-accuracy are
  not the same signal and can diverge** — a system could report "cylinder, 96% confident" while
  still being off by 6°.

**Supplementary Table S1 — additional robustness experiments (closest thing in the paper to a
noise/defect sweep, but still only single extra data points, not a curve):**
- 64k test input (same 8k-trained model): small improvement (cov 87.03%/91.87%) — PointNet++'s
  known test-time resolution invariance, not a new capability.
- **10% uniform random outliers** (added to both train and test): Seg IoU 72.38% (from 77.14%), type
  acc 95.94% (from 96.93%), normal error 9.67° (from 8.66°), **axis error 1.97° (from 1.51°, +30%
  relative)**, `{S_k}` cov 82.57%/88.78% (from 86.63%/91.64%). Paper's own characterization: "a
  little drop but still comparable." This is the single closest data point in the paper to a
  "how much does defect X degrade axis accuracy" number, but it is **outlier contamination**, not
  registration error, not depth-dependent bias, not view-dependent dropout — a different defect
  mode than our fused-cloud problem.
- `K_max=48` (vs 24): no substantial change — architecture is insensitive to this hyperparameter
  once it exceeds the true max primitive count.

## Real-scan results (Sec. 4.6, Fig. 6) — qualitative only, no numbers

3D-printed a subset of held-out **test-set** CAD models, scanned with a **DAVID SLS-2** structured-
light 3D scanner. Claim: *"SPFN trained on synthesized noises successfully reconstructed all
primitives including the small segments."* **No metrics are reported for this experiment** — no
IoU, no axis error, no coverage number, nothing quantitative, just a rendered comparison figure
(photo of printed parts + reconstructed primitive renders). Not stated whether the input to the
network was a single scan or a fused/registered multi-view cloud, nor the point count, nor whether
any preprocessing (denoising, downsampling) was applied before feeding the scanner output to SPFN.
**This is the paper's only real-scan evidence and it is a demo figure, not a benchmark** — it cannot
supply any of the quantified numbers the branch questions ask for (noise σ → axis error, coverage
vs. defect magnitude, etc.).

## Load-bearing assumptions (what could break transfer)

1. **Full-surface sampling only — no partial/single-view/occluded scans anywhere in training or
   eval.** Both the 8192-point synthetic inputs and the M=512 residual-eval samples are drawn
   uniformly over the *entire* CAD surface. The real-scan demo (Fig. 6) is the one place actual
   scanner-shaped incompleteness could appear, and it reports zero quantitative detail. **The paper
   supplies no evidence at all about behavior on a half-observed cylinder or a single-view partial
   cloud** — directly negative for the branch's partial-cylinder question; SPFN is untested there.
2. **Noise model is a single fixed magnitude, along-normal, uniform-distributed** — not a sweep, not
   registration jitter, not incidence-angle-dependent range bias, not view-dependent dropout. The
   10%-outlier experiment (Table S1) is the only other defect mode tested, and outliers are a
   different failure mechanism than our systematic depth-bias/dropout problem. **No noise-vs-
   accuracy curve exists in this paper** — cannot directly answer Branch Q7 ("quantified relationship
   between per-point noise sigma... and normal angular error").
3. **Requires full retraining on labeled ground-truth primitives (type + parameters + point-to-
   primitive membership + surface boundary) — no pretrained weights are released.** The GitHub repo
   ships training code and the ANSI dataset only. To use SPFN on our object set we would need to (a)
   generate labeled ground-truth primitive decompositions for our own training shapes (CAD models or
   manually-labeled scans) and (b) train from scratch on TensorFlow 1.10 with custom-compiled
   PointNet++ CUDA ops — a materially heavier lift than "run an off-the-shelf implementation," and a
   direct hit against the branch's "obtainable with no retraining" framing.
4. **No calibrated per-primitive confidence/fit-reliability score.** The only signal available at
   test time for "is this primitive real" is `ε_discard=0.005N`, a bare point-count threshold on the
   assigned segment — structurally identical to SuperDec's weak existence gate (see
   `superdec-superquadric-decomposition.md` in this same folder). No MDL-style score, no residual-
   based posterior, no outlier probability. `L_res`'s residual **is** computed per primitive and
   could in principle be read off at test time as a fit-quality signal, but the paper never proposes
   or evaluates it as one.
5. **No explicit boundary/extent parametrization beyond the assigned point segment.** Fig. 3 caption:
   *"the point segment `W_:,k` provides an approximation to this bounded region."* Ground-truth `S_k`
   does carry boundary info (used to sample residual points), but the network's *output* at test time
   is type + infinite-primitive parameters + a soft point membership column — extracting "face
   centre + face size" requires an extra downstream step (e.g., bounding the assigned points), not a
   primitive-native boundary output the way SuperDec's scale parameters are.
6. **No canonical-pose assumption is stated, but also not tested against.** Unlike SuperDec, this
   paper documents no rotation augmentation scheme at all — CAD models are apparently used in native
   TraceParts orientation, centered and bbox-normalized only. Whether the network is robust to
   arbitrary SO(3) input pose is simply not discussed; treat as **unaddressed**, not confirmed either
   way.
7. **Type accuracy conditions everything downstream, and the axis-error metric itself is conditioned
   on correct type** — the reported 1.51° axis number only ever gets measured on the 96.93% of cases
   where type was already right. There is no reported number for "axis error including wrong-type
   cases" (i.e., what happens to axis quality is unknown once you're in the ~3% failure regime) —
   the paper's own metric design makes the failure regime invisible rather than characterized.

## What transfers to our setting

- **The differentiable-least-squares-per-primitive-type machinery (Sec. 3.2, Eqs. 2–12) is directly
  reusable as classical (non-learned) closed-form fitting code**, independent of whether the rest of
  the network is adopted — e.g., their cylinder axis-then-circle two-stage fit, or their cone
  apex/axis/half-angle decomposition, could be lifted as plain geometry-processing utilities on top
  of normals from any source (ours, CGAL RANSAC, PCA) and slotted into a classical (not learned)
  primitive-fitting stage, entirely sidestepping the retraining problem in point 3 above.
- **The RIoU/Hungarian-matching primitive-reordering scheme (Sec. 3.1, Eq. 1)** is a clean, reusable
  idea for the branch's RQ3 (re-scan identity stability) *if* we ever need to match a new scan's
  primitives against a previous scan's — it's designed exactly for "these two primitive sets have no
  guaranteed a priori correspondence," which is our re-scan problem too.
- **The "normals optimized for fitting ≠ normals optimized for angular accuracy" finding (ablation,
  no-`L_norm` row)** is a genuinely transferable methodological caution: if we build a normal-
  estimation + primitive-fitting pipeline, benchmarking normal estimators purely on angular error in
  isolation (as several other papers in this branch do) may not predict which one yields the best
  *fitted primitive*. Worth flagging when evaluating estimator families for Branch Q7.
- **The 94.0% surface-area coverage number by plane+sphere+cylinder+cone on real mechanical CAD
  parts (Sec. 4.1)** is a concrete, citable data point for the branch's "does the contract's object
  set close with 4-primitive RANSAC" question — for parts resembling ANSI mechanical components
  (bolts, fittings, standard hardware — see Fig. 6), the four-primitive vocabulary is empirically
  sufficient for the large majority of surface area, with the residual likely fillets/chamfers/
  threads (not confirmed by the paper, inferred).
- **Explicit numerical-degeneracy handling (Supp. S.1)** — the condition-number trivialization and
  SVD-gradient clipping are a direct precedent for "near-degenerate point sets (thin strip, near-
  collinear) need special-cased fallback logic," reinforcing that our own primitive-fitting stage
  will need equivalent guards regardless of whether we use a learned or classical fitter.

## What does NOT transfer / negative findings

- **Zero partial/occluded-scan evidence.** This is the single biggest gap against the branch's
  incompleteness questions (RQ on partial cylinder, dropout, symmetry completion) — SPFN's entire
  train/test protocol samples full closed CAD surfaces; it offers no data on half-observed objects.
- **Zero registration-error / view-fusion-defect evidence.** No mention of multi-view fusion, hand-
  eye calibration error, or any of our specific defect modes (specular dropout, depth bias vs.
  incidence angle). The 10%-outlier and single fixed along-normal noise experiments are the closest
  analogues and are mechanistically different from our defects.
- **No usable off-the-shelf path.** No pretrained weights, old TF1 stack with custom CUDA
  compilation, and would need our own labeled ground-truth primitive dataset to retrain — this fails
  the branch's "obtainable from an off-the-shelf implementation with no retraining" bar outright.
  (Contrast with SuperDec, which does ship weights for exactly this use case.)
- **No calibrated fit-confidence signal** — same weak-gate pattern as SuperDec's existence head;
  does not answer the introspection/failure-detection branch questions.
- **No bounded-face extraction as a first-class output** — boundary is only available as "whichever
  points got assigned," requiring an extra downstream step to get a face centroid+extent, unlike
  SuperDec's native scale parameters.
- **No view-planning or per-primitive viewpoint content whatsoever.** Same as SuperDec: not a
  precedent for "decompose then generate per-primitive viewpoints."
- **No re-scan/repeatability study on the *same* object.** The closest thing (Supp. S.4, Fig. S2) is
  cross-*instance*, same-*category* correspondence ("column k tends to mean the same semantic part
  across different bolts in the bolt category") — a related but distinct claim from "does re-scanning
  the *same physical object* twice keep face 3 as face 3." Do not conflate the two; SPFN provides
  no evidence for the latter.
- **Axis-error metric is conditioned on correct type**, so the paper cannot answer "how bad is the
  axis once you're already in a wrong-type failure" — exactly the failure-mode information the
  branch's introspection questions need, and it's structurally absent from this evaluation protocol.

## Answers to specific Branch questions (direct hits only)

- **"At what noise/incompleteness level do methods stop returning the correct primitive type?"** —
  Not answered. Single noise level tested (Sec. 4.1); zero incompleteness testing. Only adjacent
  data point: 10% outlier contamination → type accuracy 96.93%→95.94% (Table S1), i.e. barely moves,
  but this is not the failure mode in question.
- **"Does the vocabulary close with plane+cylinder+sphere+cone?"** — Concrete number for real
  mechanical CAD parts: **94.0% average surface-area coverage** (Sec. 4.1) with these four types; no
  breakdown of what constitutes the remaining 6%.
- **"Which methods recover bounded faces (extent, boundary, centroid)?"** — SPFN: only implicitly,
  via the point segment assigned to each primitive (Fig. 3 caption); no native boundary/extent
  parameter the way SuperDec has `(s_x,s_y,s_z)`.
- **"Precedent for decompose-then-generate-viewpoints"?** — None; not addressed at all.
- **"Repeatability of decomposition across re-scans"?** — Not tested for the same object; only a
  related but distinct cross-instance/same-category correspondence result (Supp. S.4) exists.
- **"Quantified noise σ → normal/axis angular error"?** — Not available as a curve; only two
  single-point measurements exist (baseline 8.66°/1.51° at the one trained noise level; 9.67°/1.97°
  under added 10% outliers).
- **"Off-the-shelf, no retraining"?** — No: code exists (MIT, TF1.10) but ships no weights, requiring
  full retraining on a labeled dataset we'd have to build ourselves.
- **"Does introspection literature support a fitter's own confidence sufficing"?** — No: SPFN's only
  test-time reliability signal is a bare point-count discard threshold, not a calibrated score;
  matches SuperDec's pattern of weak/absent introspection signal.
