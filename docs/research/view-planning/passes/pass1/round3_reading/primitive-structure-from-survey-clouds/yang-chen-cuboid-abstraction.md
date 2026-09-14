# Unsupervised Learning for Cuboid Shape Abstraction via Joint Segmentation from Point Clouds (SIGGRAPH 2021)

**arXiv:** 2106.03437 (v1) | **HTML:** https://arxiv.org/html/2106.03437v1
**Authors:** Kaizhi Yang, Xuejin Chen (USTC)
**Code:** https://github.com/SilenKZYoung/CuboidAbstractionViaSeg (PyTorch, MIT license) —
data + pretrained-checkpoint Google Drive links in README, confirmed present.
Called "CSA" (Cuboid Shape Abstraction) elsewhere in the literature (e.g. cited
by name in SuperDec's loss description, see cross-reference below).

Read via arXiv HTML render + two targeted follow-up fetches on specific
sections (loss/normal details, ablation, applications/correspondence). Section
numbers below as rendered by the fetch tool; not independently checked against
the PDF typesetting.

## Core method

Single-pass (non-iterative, feed-forward) network, trained **per object
category**, that jointly predicts (a) a fixed bank of `M` parametric cuboids
and (b) a point→cuboid soft segmentation, with a training objective that
forces the two to agree with each other (no ground-truth part labels used —
"unsupervised" refers to no part-segmentation supervision, not no 3D shape
supervision: it trains and evaluates against ShapeNet meshes/point clouds).

1. **Feature embedding (Sec 3.1):** two EdgeConv layers on **raw point
   coordinates only** (no normals in the input features) → 64D each,
   concatenated to 128D per-point features; max-pooled to a 1024D global
   feature.
2. **Shape abstraction / VAE (Sec 3.2):** global feature → μ, σ ∈ ℝ⁵¹²,
   reparameterized to latent code `z`. For each of `M` cuboid slots, `z` is
   combined with a slot-specific embedding into a 128D feature that decodes to
   cuboid parameters: translation `t∈ℝ³`, quaternion rotation `r∈ℝ⁴`, scale
   `s∈ℝ³`, and a binary **existence indicator** `δ∈{0,1}` (soft during
   training, thresholded at inference) — so cuboid *count* is adaptive up to a
   fixed maximum `M`, not fixed per instance.
3. **Cuboid-associated segmentation (Sec 3.3):** point-cuboid affinity via dot
   product of per-point and per-cuboid features, `A_{mn} = g^c_m · g^p_n`,
   softmax-normalized into an allocation matrix `W`. This is the mechanism
   that lets segmentation and cuboid shape supervise each other without part
   labels.
4. **Slot indices are fixed, not permutation-matched.** Unlike DETR-style
   set-prediction architectures (e.g. SuperDec, see below), there is **no
   bipartite/Hungarian matching** between predicted and any target set —
   because there is no ground-truth cuboid set to match against in the first
   place. Slot `m` is just "the m-th output head," and the paper states
   directly: *"The fixed order of cuboids naturally conveys the part
   correspondence in different instances of the same category."* (Sec 4.4).
   This correspondence is a byproduct of training one shared network per
   category to convergence, not an explicit correspondence-enforcing loss.

## The four losses (Sec 3.4), total objective

`L = L_recons + 0.1·L_compact + 0.01·L_exist + 6e-6·L_kl-div`

1. **Reconstruction (Sec 3.4.1):** `L_recons = (1/N) Σ_n Σ_m W_{m,n} · d(p_n, C_m)`.
   The point-to-cuboid distance `d` is **normal-aware**: *"we calculate the
   distance from p_n to the cuboid surface with the most similar normal
   direction"* and additionally samples an auxiliary point `p_n^s` along the
   point's own normal direction at a random Gaussian offset (σ_s) before
   measuring distance — explicitly to make the loss sensitive to surface
   orientation, not just position. The paper does not state whether these
   normals are ground-truth mesh normals or point-cloud-estimated; per the
   EdgeConv detail above, normals are **not** part of the learned input
   features, so they are most plausibly computed from ShapeNet mesh ground
   truth as a training-time signal, not something the network itself outputs
   or needs at inference on a real scan. **This is a load-bearing assumption
   for us**: the training signal that makes cuboid faces align well with true
   surface orientation depends on clean, oriented normals unavailable to this
   quality on a noisy fused PLY.
2. **Cuboid compactness (Sec 3.4.2):** `L_compact = (Σ_m √(w_m + ε_sps))²`,
   an L_0.5-style sparsity penalty on the fraction of points `w_m` assigned to
   each cuboid — pushes toward fewer, larger cuboids.
3. **Cuboid existence (Sec 3.4.3):** BCE between predicted `δ_m` and a
   pseudo-ground-truth existence derived by thresholding allocation at
   `ε_ext = 0.05`.
4. **KL-divergence (Sec 3.4.4):** standard VAE regularizer on the 512-D latent.

Training: Adam, 1000 epochs, batch 32, lr 6e-4, single GTX 1080Ti; trained
**separately per category**.

## Concrete results

Verified directly against arXiv HTML source (Sec 4.1 "Structured shape
reconstruction" / Table 1; Sec 4.2 "Shape co-segmentation" / Table 2; Sec 4.3
"Ablation study and analysis" / Tables 3–5) by fetching and grepping the raw
HTML — table numbers below are exact, not fetch-tool guesses. Institution
confirmed from the HTML header: University of Science and Technology of China
(Hefei); journal record: ACM TOG vol. 40, no. 4, article 152 (DOI
10.1145/3450626.3459873).

**Reconstruction quality, Chamfer distance (Table 1; lower better), vs.
Tulsiani et al. 2017 (VP) and Sun et al. 2019 (HA):**

| Method | Airplane | Chair | Table | Animal |
|---|---|---|---|---|
| VP (Tulsiani 2017) | 0.725 | 1.006 | 1.525 | 1.896 |
| HA (Sun 2019) | 0.713 | 1.109 | 1.449 | 0.575 |
| **Ours (CSA)** | **0.329** | **0.399** | **0.848** | **0.350** |

**Co-segmentation, mIoU (Table 2) vs. more baselines (incl. SuperQuadrics/Paschalidou,
BAE-Net, BSP-Net):**

| Method | Airplane | Chair | Table |
|---|---|---|---|
| VP | 37.6 | 64.7 | 62.1 |
| SuperQuadrics | 48.9 | 65.6 | 77.7 |
| HA | 55.6 | 80.4 | 67.4 |
| BAE-Net | 61.1 | 65.5 | 87.0 |
| BSP-Net | 74.5 | 82.1 | 90.3 |
| **Ours (CSA)** | **64.2** | **82.0** | **89.2** |

Note BSP-Net beats CSA on airplane/table mIoU — CSA is not universally best,
just competitive with a much simpler, geometrically-interpretable (cuboid, not
implicit-field) output.

**Max cuboids M vs. average cuboids used (adaptive count via existence + compactness loss):**
M=16 for chair → average used N_ac = 9.753; M=16 for table → N_ac = 8.524.
M=20 airplane, M=16 animal (category-specific maxima).

**Noise robustness (Table 4), Chamfer distance vs. Gaussian noise σ_n added to
input points, chair category, σ_s=0.05 sampling:**

| σ_n | 0.00 | 0.01 | 0.02 | 0.03 |
|---|---|---|---|---|
| CD | 0.399 | 0.442 | 0.572 | 0.754 |

This is an isotropic, uniform Gaussian on point *position* — not the
registration-error / view-dependent dropout / range-bias defect profile this
branch cares about. From 0→0.03σ (on a unit-normalized shape scale) Chamfer
distance nearly doubles (0.399→0.754), a ~89% degradation — no threshold is
given for when reconstruction becomes qualitatively wrong (wrong cuboid
count/topology) vs. merely blurrier; the paper frames this purely as a
graceful-degradation curve, not a correctness cliff.

**Ablation, point-to-cuboid distance formulation (Sec 4.3, Table 3, chair CD):**
P2C-Dis (hard nearest-cuboid assignment, no segmentation branch) = 3.356,
Chamfer-Dis (plain bidirectional Chamfer, no segmentation branch) = 0.834,
**P2C-Seg (this paper's soft segmentation-weighted distance) = 0.436** — i.e.
the full method beats plain Chamfer by ~2x and beats hard nearest-cuboid
assignment by ~7.7x on chair. (Airplane: 4.836 / 0.286 / 0.237; Table: 3.335 /
0.992 / 0.846 — same ordering on all three categories.) Confirmed by direct
HTML grep, not just fetch-tool paraphrase.

**Robustness on sparse point clouds (Sec 4.3, Fig. 11, chair category only):**
the network was **trained separately** at three point counts — 256, 1024, and
4096 — and produces qualitatively "reasonable and consistent" cuboid structure
at each. No Chamfer/mIoU numbers are reported for this experiment (qualitative
figure only). Important caveat for our transfer question: this is a **per-density
retrain**, not a single model tested at varying test-time density — it says
nothing about a fixed model's robustness to unexpected point count/density at
inference, which is our actual situation (survey cloud density is not a
training-time choice we control per-object).

**Compactness weight λ₁ sweep:** 0→0.20 trades cuboid count 15.5→5.5 against
CD 0.381→1.452 — i.e., forcing fewer cuboids costs real reconstruction
fidelity, not a free simplification.

## Load-bearing assumptions (what could break transfer)

1. **Canonical, pre-aligned, unit-scale input, trained per category.**
   Sec 4.1: *"All shapes are pre-aligned and normalized to the unit scale."*
   No PCA-alignment or pose-normalization procedure is described in the paper
   itself — it inherits ShapeNet's canonical per-category orientation
   convention. The only rotation augmentation mentioned is ±5°/±10° around
   the z-axis, and **only for the four-legged-animal category** (not
   airplane/chair/table). This is a direct, hard violation of our setting: an
   arbitrarily posed tabletop object has no canonical frame, and the released
   pretrained checkpoints are category-specific ShapeNet models (airplane,
   chair, table, animal) with no "generic object, any pose" checkpoint —
   contrast with SuperDec, which ships a `normalized`-checkpoint explicitly
   for "objects from generic 3D scenes" and tests on real, arbitrarily-posed
   ScanNet++/Replica captures (see cross-reference below). CSA has no
   equivalent evaluation or checkpoint.
2. **Complete-coverage point clouds only; the one density experiment doesn't
   test what we need.** Main experiments use 4096 points sampled uniformly
   from complete watertight ShapeNet meshes. Sec 4.3's "Robustness on sparse
   point clouds" (Fig. 11, chair only) additionally trains at 256 and 1024
   points, but this is uniform density reduction over a still-*complete*,
   all-around shape, and — critically — a **separate model retrained per
   density**, not one model tested on unexpected/variable density at
   inference. There is no partial-view, occlusion, or single-viewpoint
   experiment anywhere in the paper — the failure-case discussion (Sec 4.5,
   below) is entirely about *complete* shapes that are hard to segment, not
   incomplete scans. This is an unfilled gap for our fused-but-partial,
   dropout-afflicted PLYs, and the sparse-density result cannot be read as
   evidence against that gap.
3. **Trained per-category, closed-world.** The network has no mechanism for
   an object category it wasn't trained on; "multi-class training" is listed
   explicitly as *future work*, not present capability (Sec 4.5/Conclusions).
   For an inspection setting with arbitrary tabletop objects, this means
   either (a) training a bespoke model per object class (defeats the "no
   hand-programmed per-object recipe" goal) or (b) not using this method
   as-is.
4. **Noise model tested is isotropic positional Gaussian on ShapeNet-scale
   shapes**, not our defect profile (mm-level registration error at real
   metric scale, ~4cm fused-vs-single-view offset, glossy/dark dropout,
   range-dependent depth bias). The 0→0.03σ sweep cannot be mapped onto our
   physical error budget without knowing the shape's unit-normalization
   scale, which is not given per-object.
5. **Fixed-slot part correspondence is a category-level training artifact,
   not a validated per-object re-scan guarantee.** The claim *"the fixed
   order of cuboids naturally conveys the part correspondence in different
   instances of the same category"* is about correspondence **across
   different object instances of the same trained category** (e.g., chair A's
   seat and chair B's seat both land in slot 3), which is a much weaker claim
   than "re-scanning the *same* physical object twice yields the same slot for
   the same face." The paper never runs a re-scan/repeatability experiment;
   whether slot assignment is stable under the small input perturbations a
   re-scan would introduce (different point sampling, registration noise, view
   coverage) is untested. Given the network is deterministic (single forward
   pass, no stochastic sampling at inference beyond the VAE's mean-code
   usage), *some* stability is plausible for small input perturbations, but
   this is inference from architecture, not evidence from the paper.

## Failure cases (Sec 4.5, exact-quoted)

1. *"the instance with very unique structures may not be well reconstructed,
   such as the aircraft [with atypical structure]"* — the method assumes
   category-consistent structure across the training set; genuine
   category-outliers are poorly abstracted.
2. *"points sampled for fine structures are too scarce to provide sufficient
   geometric information, e.g. the table legs"* — thin/fine structures under
   sparse sampling are missed or merged.
3. *"For shapes with excessive small parts, our method fails to precisely
   recover all the parts with limited M"* — six chair slats collapse into two
   cuboids when the part count exceeds the category's `M`.

Listed future work: multi-class (cross-category) training, primitives beyond
cuboids, unsupervised part-relationship discovery. No mention of noise/partial
data as a listed limitation — the paper treats its Table 4 Gaussian-noise
sweep as already "robustness demonstrated," not as an open problem.

## Applications (Sec 4.4) — relevant to identity/consistency question

- **Structural shape clustering:** uses each shape's `δ` (existence indicator)
  vector directly as a structure descriptor — *"shapes in which the same
  cuboids appear are considered to have the same structure"* — which only
  works because slot identity is assumed stable within a category. This is
  the paper's own strongest implicit endorsement that fixed-slot ordering is
  useful downstream, but again only demonstrated across different shape
  instances of one trained category, not across re-scans of one instance.
- **Shape interpolation:** linear interpolation in the 512-D VAE latent space
  between two shapes, decoded to smoothly-varying cuboid sets — relies on the
  same per-category slot consistency.

Neither application involves camera viewpoints, robots, or any inspection-like
downstream task.

## Cross-reference: contrast with SuperDec (ICCV 2025)

(Reading notes at
`docs/research/view-planning/passes/pass1/round3_reading/primitive-structure-from-survey-clouds/superdec-superquadric-decomposition.md`)
SuperDec explicitly reuses this paper's reconstruction-loss normal term
(`L_N ... "is defined as the reconstruction loss from Yang et al. [CSA, 2021]"`)
but pairs it with a DETR-style bipartite-matched slot architecture instead of
CSA's fixed, unmatched per-category slots — and crucially trains/evaluates on
real, noisy ScanNet++/Replica data with a checkpoint meant for arbitrary
object pose. On every axis this branch cares about (real-scan generalization,
released weights, non-canonical pose, boundedness, closed-form surface
distance for grasp/view geometry), SuperDec is the more directly applicable
descendant; CSA is best read as the **origin of the normal-aware
reconstruction-loss idea** and as an illustration of how much a "clean
ShapeNet, canonical pose, complete cloud" assumption costs when compared
against a later paper (SuperDec) that explicitly targeted and reported on
real, arbitrarily-posed scans.

## What transfers to our setting

- **The normal-aware point-to-primitive distance formulation** (project a
  point along its own normal, measure distance to the primitive with most
  similar normal) is a reusable idea for *any* future primitive-fitting loss
  or scoring function we might build, cuboid or otherwise — it's the
  mechanism that makes reconstruction respect surface orientation rather than
  just point position, directly relevant to Branch's concern about recovered
  normal accuracy.
- **Existence-indicator + compactness-loss pattern** (soft head thresholded to
  decide "does this primitive exist") is a reusable pattern for adaptive
  primitive count, seen again (independently) in SuperDec — evidence this is
  a converged idiom, not a one-off.
- **Boundedness:** cuboids are bounded by construction (`s∈ℝ³` scale), so each
  recovered cuboid directly gives a centroid (`t`) and extent (`s`) — this
  satisfies Branch's "face centre + face size, not a plane equation" ask
  structurally, same as SuperDec, better than infinite-plane RANSAC output.
- **Fast, single feed-forward pass** (no LM/iterative refinement stage,
  unlike SuperDec) — cheap if it were applicable to our data, but see below.

## What does NOT transfer / negative findings

- **Canonical-pose and complete-cloud assumptions are both hard, unaddressed
  requirements**, not soft degradations — the paper offers zero evidence
  about partial views or arbitrary object pose. This directly answers the
  task's framing question ("read partly to quantify how badly its
  canonical-pose and complete-cloud assumptions hurt"): the honest answer is
  the paper contains **no data to quantify this from** — it simply never
  tests either condition. We cannot extract a degradation number; we can only
  say the assumption is present, undiscussed, and would need to be tested by
  us if we wanted to use CSA specifically (we likely would not, given
  SuperDec exists as a direct, real-scan-tested descendant using a related
  loss).
- **No RANSAC or classical-geometry baseline whatsoever** — cannot inform the
  plane/cylinder/sphere/cone-vs-cuboid vocabulary question from this paper.
- **No per-primitive confidence/fit-reliability score** — existence indicator
  `δ` is trained from a fixed allocation-mass threshold (`ε_ext=0.05`), a
  training-time pseudo-label, not an inference-time uncertainty estimate
  usable as a fallback trigger.
- **No closed-form or empirical orientation-error/covariance result** — the
  paper reports Chamfer distance (position error), never a normal-angle error
  in degrees; it cannot supply Branch 1's or Branch 8's requested numbers.
- **No cross-category / open-world capability** — a fundamentally
  closed-world method (train one network per category); not usable as a
  drop-in "any tabletop object" tool without retraining per class, which
  reproduces the ADAM-style per-object-recipe problem this project is trying
  to avoid, just moved from "hand-coded trajectory" to "trained-per-category
  network."
- **No view-planning precedent** — like SuperDec, this paper's downstream
  applications (shape interpolation, structural clustering) never touch
  camera viewpoints or robotic inspection; it is not a precedent for
  "decompose then generate per-primitive viewpoints."
- **Re-scan/repeatability of slot identity is untested**, only cross-instance
  (within a trained category) correspondence is demonstrated and only via a
  downstream clustering task, not a direct stability metric.

## Answers to specific Branch questions (direct hits only)

- **"Does the contract's object set close with plane+cylinder+sphere+cone
  RANSAC, or force cuboid decomposition?"** — This paper offers no comparative
  evidence (no RANSAC baseline), only a demonstration that cuboid abstraction
  beats other *learned* abstraction baselines (VP, HA, SuperQuadrics) on
  ShapeNet Chamfer/mIoU — not informative for the RANSAC-vocabulary question.
- **"Which methods recover bounded faces (extent, boundary, centroid)?"** —
  Yes structurally: each cuboid gives translation (centroid) + scale (extent)
  + rotation directly, same pattern as SuperDec's superquadrics, but here the
  face is flat (rectangular) by construction rather than superquadric-curved
  — arguably an even cleaner match to "face centre + face size" than
  SuperDec's superquadrics, *if* the canonical-pose/complete-cloud
  requirements could be met.
- **"Repeatable decomposition across re-scans — does face 3 stay face 3?"** —
  Paper claims fixed-slot correspondence *across different instances of a
  category* (Sec 4.4 clustering application) but never tests re-scan
  repeatability of the *same* object; this is inferred plausibility from a
  deterministic architecture, not demonstrated evidence.
- **"Released weights and real-scan generalization — vs. ShapeNet-only
  evaluation, vs. canonical-orientation assumption — a constraint
  violation?"** — Confirmed constraint violation: pretrained checkpoints are
  category-specific (airplane/chair/table/animal), ShapeNet-only, canonical
  pose, no real-scan evaluation anywhere in the paper. This is the paper the
  Branch's "canonical-pose constraint violation" question was written to
  flag, and it is a clean, citable confirmation.
- **"What computable signals declare a fit unreliable?"** — None offered
  beyond the training-time existence-threshold heuristic; not an inference-time
  reliability signal.

## Runtime

Not reported (no per-sample inference time, no complexity analysis). Training
details given (1000 epochs, batch 32, single 1080Ti) but this is not a
runtime-at-our-cloud-sizes answer for Branch's practical question — would need
to be measured directly if we ran the released code.
