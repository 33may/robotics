# Ramamonjisoa, Stekovic & Lepetit (2022) — "MonteBoxFinder: Detecting and Filtering Primitives to Fit a Noisy Point Cloud"

**Source:** ECCV 2022, arXiv:2207.14268v1 (28 Jul 2022). Full text (14 pages main + 8 pages supplementary)
read via `pdftotext -layout` on the arXiv PDF. Local copy: `/tmp/paper_dl/montebox.{pdf,txt}`.
Project page: https://michaelramamonjisoa.github.io/projects/MonteBoxFinder — code:
https://github.com/MichaelRamamonjisoa/MonteBoxFinder (checked README: no runtime/hardware numbers,
no released weights — consistent with the method being training-free; no cylinder/sphere/cone anywhere
in the repo, cuboids only).

---

## 0. Correcting the selection rationale before anything else

The task brief frames this paper's premise as "RANSAC-based extraction works only on very clean scans."
**That sentence is not MonteBoxFinder's claim about itself — it is MonteBoxFinder's claim about a different,
prior method** (GlobFit, Li et al. 2011 [ref 20]), stated in Related Work (§2.1):

> "Li et al [20] decompose a real scan of an object into primitives by extracting a set of primitives with
> RANSAC, which they refine by reasoning on relationship between these primitives. **However their method
> works only on very clean scans**, and using object that were built as a set of primitives. Furthermore,
> since they reason about interaction between primitives using a graph, the complexity of there method
> quickly becomes untractable." (§2.1, p.4)

MonteBoxFinder's own position is the opposite: it explicitly targets **noisy, incomplete real scans**
(ScanNet) and its entire contribution is a search algorithm that tolerates a RANSAC-generated candidate
pool containing many false positives, rather than requiring a clean pool. From the abstract and §1:
"point clouds from 3D reconstruction or scans are typically very noisy with missing data, and robust
methods are required to handle this real data" — and "We expect this algorithm [Efficient-RANSAC] to
generate correct primitives but also **many false positives**. Our problem then becomes the identification
of the correct primitives while rejecting the incorrect ones" (§1, p.2). So the paper's actual, measured
premise is: **RANSAC over-generates on noisy real scans (many spurious plane/cuboid candidates), and this
is survivable by adding a combinatorial-selection stage on top — not that RANSAC fails outright on noise.**
This reframes the paper's relevance to our branch: it does not measure "at what noise level RANSAC stops
returning the correct primitive type" (§ branch Q1) — it sidesteps that question by treating false
positives as an expected, tolerable byproduct to be filtered downstream, never quantifying at what noise
level the *true* primitive fails to appear in the candidate pool at all (a silent, unaddressed failure
mode: filtering can only reject bad candidates that exist, not recover a candidate RANSAC never proposed).

---

## 1. Core method (§3, Fig. 2–3, Algorithms 1–3)

Two decoupled stages: (A) over-generate a large pool of cuboid **candidates** from RANSAC plane detections,
(B) run a discrete combinatorial **search** to select the best non-intersecting subset. The paper's sole
contribution is stage (B); stage (A) is admittedly "a simple ad hoc algorithm" (§1) built on an existing
library.

### 1.1 Candidate generation (§3.1, Fig. 3)
1. **Plane extraction:** Efficient-RANSAC (Schnabel et al. 2007 [25], via CGAL) on the raw point cloud.
   Controlled by 3 hyperparameters — plane-to-point distance threshold, normal-cosine-similarity threshold,
   connectivity radius — **the same fixed values used for every ScanNet scene** ("We use the same
   hyperparameters for all the scenes in ScanNet, although we could run RANSAC multiple times with various
   geometric parameters in order to adapt to various types of noise... " §3.1, p.6 — floated as future
   work, not implemented). Same single-global-tuning pattern seen in the RSPD plane-detection paper.
2. **Box-from-plane-pairs:** for every pair of plane segments (π_A, π_B) satisfying **alignment**
   (near-orthogonal `|N_A·N_B| < α` or near-colinear `|N_A·N_B| > β`) and **proximity** (some point pair
   within distance `γ`), build an oriented box via two Gram-Schmidt orthonormalizations (Supp. Mat. Eq.
   1a/1b) that encloses `X_A ∪ X_B`. Fixed constants used for all scenes: **(α, β, γ) = (0.3, 0.7, 0.025)**
   (Supp. §1.1).
3. **Thin cuboids from single planes:** each individual plane segment also gets its own oriented bounding
   box via the SO(3) OBB-optimization method of Chang et al. 2011 [5] (also via CGAL), so pure planar
   surfaces (no adjacent partner) are still represented as a (degenerate/thin) box.
4. **Yield:** "On average, we obtain **880 cuboids and 174 thin cuboids per scene**" (§3.1, p.6) — i.e.
   a candidate pool of roughly ~1050 boxes per ScanNet *scene* (whole room), most of which are false
   positives to be filtered by stage B.
5. **Cuboid representation:** a `Cuboid` object is defined by **its 6 face normals and 8 corners** (§3.2,
   "Cuboid Class", p.7) — i.e. every candidate is already a fully bounded, parametric box, not an infinite
   plane. Face centroid/extent for any face is trivially derivable from its 4 corners. This is the
   strongest direct hit on the "bounded faces, not infinite primitives" branch question.

### 1.2 Objective function (Algorithm 1, p.7)
For a candidate solution (set of cuboids) `S`, sample points+normals `(X, N(X))` from the cuboid mesh
surfaces, compare against the real scan `(Y, N(Y))`:
- `ℓ_c` = bidirectional Chamfer Distance (CD) between `X` and `Y`, **truncated at τ=0.1** and normalized
  by τ.
- `ℓ_n` = bidirectional Cosine Dissimilarity between `N(X)` and `N(Y)`.
- Combined loss: **`ℓ = ℓ_c · (1 + 0.25·exp(ℓ_n))`** — geometric fit dominates, normal mismatch acts as
  a multiplicative penalty.
This is a *scene reconstruction fidelity* loss (how well does the box arrangement reproduce the input
cloud), not a per-primitive correctness/uncertainty measure — no notion of ground-truth box identity is
used anywhere in training or scoring.

### 1.3 Selection problem (§3.2)
Choose `S' ⊆ S` minimizing `ℓ(S')` subject to **mutual non-intersection**: `isCompatible(s1,s2,η)`
computes an IoU-like ratio — intersection volume over the *smaller* of the two cuboid volumes (not the
union) — approximated by Monte-Carlo point sampling (`N_samples=5000`) inside both boxes, thresholded at
**η=10%** (Supp. §1.2, Algorithm 4). This pairwise-compatibility matrix is precomputed once per scene for
efficiency. Search space is exponential in |S| (~1050 candidates), constrained by this compatibility
matrix.

### 1.4 MonteBoxFinder search algorithm (§3.4, Algorithms 2–3, Fig. 2)
Inspired by MCTS-for-scene-understanding (Hampali et al. CVPR'21 [13]) but **explicitly drops the tree
structure**, motivated by two stated observations (§3.4, p.9):
- **Order invariance:** "the order we select the primitives does not matter" — but a tree-based search
  (MCTS) still wastes iterations re-exploring subtrees rooted at an early bad choice.
- **Missing complementarity:** MCTS's node-update step only touches *visited* nodes, so a high confidence
  for "keep primitive P" never automatically lowers confidence for "reject primitive P" (or vice-versa).

Algorithm (per iteration): (1) sort all candidates by their current confidence `s.μ1` descending; (2) walk
the sorted list, for each candidate sample `ε~U[0,1]`, with probability `P=0.3` **exploit** (add `s` to the
tentative solution iff `s.μ1 > s.μ0`), else **explore** (add iff `s.μ1 < s.μ0`), respecting
`isCompatible`; (3) evaluate `ℓ` on the assembled solution; (4) **update ALL candidates' states** (not just
those touched this iteration) via a modified UCB-style rule:

`s.μ0 = -s.ℓ0 + √(ln(1/δ)/s.n0)`,  `s.μ1 = -s.ℓ1 + √(ln(1/δ)/s.n1)`   (Eq. 1, §3.4)

where `ℓ0`/`ℓ1` are the best (min) losses achieved so far when `s` was rejected/selected, `n0`/`n1` are
reject/select visit counts, and `δ=0.03` is the exploration-rate hyperparameter (smaller δ → more
exploration). Track the best solution seen across iterations as the final output. This "update everyone,
every iteration" step (vs. MCTS's "update only the visited path") is presented as the source of the
speed advantage; the paper frames the whole method as "conceptually simple... a few lines of pseudo-code"
(Conclusion, §5).

---

## 2. Concrete results and numbers (Table 2, Fig. 4–5, §4)

**Dataset:** ScanNet [8], full dataset, **1613 indoor scenes**, using ScanNet's own "decimated and cleaned"
point clouds (§4.1) — i.e. the paper does **not** test on raw uncleaned scans; whatever ScanNet's own
preprocessing removed is invisible here (a gap for the "does denoising/preprocessing change outcomes"
branch question — no data either way).

**Evaluation protocol (§4.3) — important methodological caveat:** run Hill-Climbing first to convergence,
record the number of objective-function evaluations `N_eval` it needed; then run MCTS, MCTS-Binary, and
MonteBoxFinder under that **same fixed evaluation budget** `N_eval` (not to their own convergence) "to
ensure the three methods are compared fairly" and because evaluating `ℓ` is "by far the most costly step."
This means Table 2's numbers are a **budget-matched snapshot**, not asymptotic/converged performance for
the non-hill-climbing methods — a genuinely different (and arguably harder-to-generalize) protocol than a
fixed-iteration-count or fixed-wall-clock comparison.

**Table 2 (whole-dataset averages, all 1613 ScanNet scenes):**

| Method | Loss ↓ | Precision Pr_τ ↑ | AUC ↓ | AUC(norm) ↓ | Avg. #cuboids ↑ |
|---|---|---|---|---|---|
| Hill-Climbing | 0.383 | 0.928 | 0.871 | 0.998 | 12 |
| MCTS | 0.247 | 0.966 | 0.427 | 0.225 | 28 |
| MCTS-Binary | 0.292 | 0.961 | 0.370 | 0.102 | 35 |
| **MonteBoxFinder (ours)** | **0.201** | **0.982** | **0.322** | **0.018** | **37** |

- **Precision metric `Pr_τ`** (Eq. 2, §4.2): bidirectional point-matching ratio between the box-sampled
  synthetic cloud `X` and the real scan `Y`, counting a point "matched" if its Chamfer Distance to the
  other cloud is `≤ τ=0.2`. This is a **surface-coverage/reconstruction-fidelity** precision, not a
  primitive-identity precision (it says nothing about whether any individual detected box corresponds to
  a real distinct object/part) — must not be read as "correct primitive rate."
- **AUC / AUC(norm):** area under the loss-vs-#evaluations curve (Fig. 4, log-scale x-axis 10⁰–10³
  evaluations), min-max normalized 0–1 across methods per scene, then averaged. Lower = converges faster.
  This is a **relative convergence-speed** measure — no absolute wall-clock seconds or hardware spec is
  reported anywhere in paper or repo (checked both).
- **Avg. #cuboids retrieved:** the paper notes "bad solutions tend to contain a small number of selected
  primitives" because dense non-intersecting packing is hard to find (§4.2) — so cuboid count is used as a
  rough complexity/completeness proxy, not validated against a ground-truth object count anywhere.

**Qualitative (§4.5, Fig. 5):** Hill-Climbing selects large cuboids spanning multiple real objects (bad);
MCTS "often selects many true positives but misses some... because it cannot explore deeper levels of the
tree for the given iteration budget"; MonteBoxFinder "successfully retrieve[s] cuboid primitives for
objects of different sizes, such as walls, floors, and furniture" and reconstructs "many chairs" and
"bedroom furniture" — the only qualitative claim of object-level correctness, and it is visual/anecdotal
(figure inspection), not a quantified per-object correctness metric.

**No angular-error number is reported anywhere in the paper.** Neither for RANSAC-extracted plane normals,
nor for assembled cuboid face normals, at any noise level. Cosine dissimilarity `ℓ_n` is used only inside
the aggregate scene-level loss, never reported as a standalone per-primitive orientation-error statistic
or broken out by noise/incompleteness level. This is a direct, load-bearing negative finding against
several of our branch's numeric-angular-error questions.

---

## 3. Load-bearing assumptions

- **The correct primitive must already be present in the RANSAC-generated candidate pool.** MonteBoxFinder
  can only *filter*, never *recover a candidate that RANSAC never proposed*. If our defect profile
  (registration jitter, specular/dark dropout, ~4cm fused-vs-single-view offset) causes Efficient-RANSAC to
  simply fail to segment a face/cylinder at all (not just segment it noisily), this method's entire
  contribution is moot for that face — a silent, structural failure mode the paper never measures or
  discusses (it only ever discusses *rejecting* bad candidates, never *recall of the true candidate into
  the pool* as a function of noise level).
- **Scene-level, not object-level, framing.** ScanNet inputs are whole rooms; "objects" here are
  incidental parts of a room-scale reconstruction (chairs, furniture, walls, floors). There is no
  single-isolated-tabletop-object condition tested — the closest partial analog is "furniture," still
  embedded in room clutter with other objects providing spatial/compatibility constraints the selection
  algorithm exploits (fewer intersecting neighbors = easier selection). An isolated single object survey
  (our setting) has a different candidate-pool structure (fewer, larger, less mutually-constraining boxes)
  that is untested here.
- **Boxes only — no cylinder/sphere/cone primitive at all**, not even as a baseline comparison. "Thin
  cuboids" from single planes are the only fallback for non-paired planar surfaces; there is no analogous
  fallback construction for curved geometry. A true cylindrical part would presumably be tiled by many
  small locally-planar Efficient-RANSAC segments (same underlying mechanism the RSPD paper explicitly
  documents as a failure mode on cylinders) and then converted into many small "thin cuboids" — this is an
  **inference from the method's construction**, not a claim tested or discussed in the paper itself.
- **Global, once-fixed hyperparameters** for RANSAC extraction (§3.1) and for the alignment/proximity
  cuboid-construction thresholds `(α,β,γ)=(0.3,0.7,0.025)` (Supp §1.1) — used identically across all 1613
  ScanNet scenes, with adaptivity to different noise types explicitly proposed but not implemented (§3.1).
  Same pattern as RSPD: apparent parameter-freedom is actually "tuned once, reused everywhere," with no
  guarantee of transfer to a qualitatively different noise/defect profile (our tabletop D405 survey vs.
  ScanNet's structured-light/BundleFusion room scans).
- **Ground truth = the input scan itself**, not any external object model or human segmentation. Fit
  quality is entirely "does the box arrangement reconstruct the scan," never "does box #k correspond to
  real physical part #k." This sidesteps rather than answers the "what is the ground-truth story for the
  correct decomposition" branch question — there is no primitive-identity ground truth anywhere in this
  paper's evaluation.
- **Fixed evaluation budget matched to Hill-Climbing's convergence point** (§4.3) — an unusual protocol
  choice that makes Table 2 a snapshot comparison, not a converged-performance comparison; generalizing
  these numbers to "how good is MonteBoxFinder if run to its own convergence" requires re-reading Fig. 4,
  which shows it is still improving in some regime but is the best of the four at every budget shown.

---

## 4. What transfers to our setting

- **The core algorithmic insight (order-invariant scoring → sort-and-update-all-candidates each iteration,
  no tree) is a clean, implementable idea** for any "select the best non-overlapping subset of primitive
  hypotheses from an over-generated noisy pool" problem — directly reusable if our pipeline also
  over-generates candidate faces/cylinders per object (e.g., running RANSAC/region-growing with multiple
  parameter settings, or getting redundant/overlapping fits) and needs a principled way to pick the
  non-conflicting subset that best explains the survey cloud. Not specific to cuboids; the paper explicitly
  argues genericity to "other primitives" (Conclusion, §5), though this is asserted, not demonstrated with
  a second primitive type.
- **The `Cuboid` representation (6 face normals + 8 corners) is directly a "bounded face" output** —
  exactly matches the "face centre and face size, not a plane equation" need from our branch questions,
  for the box case. Any face centroid/extent is a trivial derived quantity from the 4 corners of that face.
- **Per-candidate confidence signals `μ0`/`μ1` (Eq. 1) are a cheap, off-the-shelf, no-retraining reliability
  score** computed purely from Chamfer-distance-based loss statistics accumulated during search — directly
  relevant to "what computable signals do papers use to declare a fit unreliable... obtainable without
  retraining." Caveat: these are *relative/comparative* scores used for sorting within one scene's search,
  never calibrated as an absolute confidence or validated against downstream task success — building that
  calibration would be entirely on us, same gap as RSPD's MDP/MND.
- **CGAL's Efficient-RANSAC [25] + SO(3) OBB optimizer [5] are both off-the-shelf, no-training components**
  used directly by this paper — concrete evidence for the "CGAL covers box... on a PLY with no training"
  branch question: yes, CGAL ships both pieces needed for the box-fitting half of a box+cylinder pipeline.
- **Qualitative acknowledgment that dark/specular dropout is a first-class noise source** for real RGB-D
  scans (§1: "dark or specular materials are not well captured by the RGB-D cameras") — same defect
  family as our D405 dropout, though purely qualitative, no magnitude given.
- **No training data required at all** (repeated emphasis in abstract, §1, Conclusion) — fits our
  deterministic/no-training constraint directly, unlike learned shape-abstraction baselines it cites
  (Tulsiani et al., Paschalidou et al. superquadrics) which need training data and are noted (§2.1) to
  "only operate at the scale of a single object, on synthetic data."

## 5. What does NOT transfer, and why

- **No cylinder/sphere/cone primitive, and no discussion of how the method would extend to curved
  geometry** — cuboids (incl. degenerate "thin" cuboids for bare planes) are the entire vocabulary. Does
  not close the plane+cylinder+sphere+cone contract; would need to be paired with a separate
  cylinder/sphere/cone detector, and the paper gives zero guidance on how such primitives would interact
  with the isCompatible/intersection-volume machinery (which is written specifically for box-box
  intersection tests).
- **No angular-error number for any recovered normal, at any noise level** — the single most direct miss
  against this branch's core numeric ask (Branch 1: expected angular error vs. noise/incompleteness). The
  paper's only orientation-related metric (cosine dissimilarity `ℓ_n`) is folded into an aggregate scene
  loss and never reported standalone or as a function of noise magnitude.
- **No registration/multi-view fusion error treatment.** ScanNet clouds are themselves BundleFusion-style
  multi-view registrations, so registration artifacts are *implicitly* present in the "noisy" input, but
  the paper never separates registration error from sensor noise from dropout, and never propagates any of
  them into a stated uncertainty on the output boxes. Does not touch the M3C2-EP-style covariance-transfer
  question at all.
- **No repeat-scan / primitive-identity-stability experiment.** "Order invariance" here is a statement
  about the *search process* (insertion order doesn't change the optimum found within one run on one fixed
  input), **not** about detection stability across two independent noisy realizations of the same scene —
  these are different properties and should not be conflated. RQ3's "does face 3 stay face 3" question is
  untested; there is no persistent-ID mechanism across runs at all in this design.
- **No absolute runtime numbers** (confirmed absent from both paper and GitHub README) — only a relative,
  normalized AUC-over-evaluations curve. Cannot directly answer "realistic per-object runtime at our cloud
  sizes"; would need to benchmark CGAL's Efficient-RANSAC + a reimplementation of the search ourselves.
- **No failure-predictor calibration against downstream task success** — `μ0`/`μ1` and the aggregate loss
  `ℓ` are used only as internal search-guidance/acceptance signals, never validated against "is the
  resulting box's pose actually correct/usable." Same gap as RSPD; the fit-score→task-success mapping
  would have to be built by us from scratch, and this paper offers no precedent for doing so.
- **No preprocessing ablation.** Uses ScanNet's pre-decimated/pre-cleaned clouds as a given; never
  compares raw vs. cleaned, never tests voxel size or outlier-removal sensitivity. Silent on "does
  denoising help or hurt edge-dependent segmentation" — no data either direction.
- **Scene-level (whole room, many objects) not single-object framing** — "objects" fall out as an
  emergent byproduct of scene decomposition, embedded in a multi-object compatibility graph; the paper
  never isolates a single scanned object the way our tabletop-survey setting does, so the relative ease/
  difficulty of the search problem (pool size, mutual-constraint density) at our scale is unknown from
  this paper. No per-primitive-type breakdown (e.g. "boxes vs desks vs walls") of precision either.
- **No view-planning / next-best-view component whatsoever.** This is a static, offline scene-decomposition
  method for representation/labeling purposes (its own suggested future use: "generate labeled data for
  training a deep architecture for fast inference... to predict cuboids from point clouds... or from
  RGB-D images," §5) — it stops at geometry extraction and never proposes viewpoints from the recovered
  primitives. For the "nearest precedent for decompose-then-generate-viewpoints" branch question, this
  paper is evidence of a precedent for the *decompose* half only; the *generate viewpoints from primitives*
  half has no analog here and would be entirely our own contribution on top of a method like this one.

---

## 6. Section/figure/equation/table index (for checkability)

- Related-work "clean scans only" claim about GlobFit/Li et al. [20] (not about MonteBoxFinder itself):
  §2.1, p.4.
- Candidate generation pipeline: §3.1, Fig. 3 (pipeline diagram), p.5–6. 880/174 cuboids-per-scene figure:
  §3.1, p.6. Cuboid-from-plane-pair construction (Gram-Schmidt, α/β/γ constants): Supp. §1.1, Eq. 1a/1b,
  p.18–19.
- `Cuboid` class definition (6 normals + 8 corners): §3.2, p.7.
- Objective/loss function: Algorithm 1, p.7 (ℓ_c, ℓ_n, τ=0.1 truncation, combination formula).
- Intersection/compatibility test (isCompatible, η=10%, N_samples=5000): §3.2 p.7, Supp. §1.2 Algorithm 4,
  p.19.
- MCTS baseline description: §3.3, p.8–9. Binary-tree MCTS variant: Supp. §2.2, p.21, Fig. 1 (supp).
- MonteBoxFinder algorithm: §3.4, Algorithm 2 (p.10), Algorithm 3 / Simulate+Update (p.10), UCB-style
  criterion Eq. 1 (μ0/μ1, δ=0.03) p.11.
- Dataset: §4.1, p.11 (ScanNet, 1613 scenes, decimated/cleaned clouds).
- Metrics: §4.2 — fitness loss ℓ; precision Pr_τ Eq. 2 (τ=0.2), p.11–12; AUC efficiency measure, p.12;
  complexity measure (avg #cuboids), p.12.
- Evaluation protocol (budget-matching to Hill-Climbing's N_eval): §4.3, p.13.
- Main results table: Table 2, p.13. Convergence curves: Fig. 4, p.12.
- Qualitative results: §4.5, Fig. 5, p.14; more in Supp. §3.1 Fig. 2, p.23.
- Conclusion / proposed future extensions (other primitives, auto-labeling, RGB-D reprojection): §5, p.13–14.
- Dark/specular dropout acknowledgment (qualitative only): §1, p.3 ("dark or specular materials are not
  well captured by the RGB-D cameras").
