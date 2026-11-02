# PEARL — Energy-based Geometric Multi-Model Fitting

**Isack, Boykov — IJCV vol. 97, no. 2, pp. 123-147, April 2012** (received June 2010,
accepted June 2011). Full text obtained as the author's own PDF,
`www.csd.uwo.ca/~yboykov/Papers/ijcv10_pearl.pdf` (33 pp., LaTeX/pdfTeX, matches the
journal pagination — in-text page numbers below are journal pp. 1-33 as printed, i.e.
`p.1`=`p.123` of the volume). ResearchGate link in the task is paywalled (403); this PDF
is the free author copy and is the actual source read. Branch: `primitive-structure-from-survey-clouds`.
Selected because it's the classical energy-optimization ancestor that Multi-X and related
"how many primitives" work repairs/extends.

---

## 1. Core method / idea

**Problem framing (Sec. 1, p.2-6).** Standard multi-model fitting (sequential RANSAC,
J-linkage, mean-shift/Hough) greedily selects one model at a time by maximizing inlier
count within a threshold, then removes its inliers and repeats. The paper argues this is
"fundamentally flawed" for multi-model data: Fig. 1 (p.2) shows a synthetic 8-line
example where, at higher noise, a random spurious diagonal line accumulates *more*
inliers (125) than either true line (89, 91) — greedy inlier-maximization picks the wrong
model. This is the paper's foundational negative finding about the entire "greedy
inlier-count" family, not a critique of just RANSAC's classic single-model use.

**Energy formulation.** Multi-model fitting is posed as a discrete labeling problem: each
data point `p` gets a label `L_p` from a set of candidate models (including a special
"outlier" label `∅`). The core energy, Eq. (3), p.10:

```
E(L) = Σ_p ||p - L_p||  +  λ · Σ_(p,q)∈N  w_pq · δ(L_p ≠ L_q)
```

- **Term 1 (fitting error):** per-point geometric error to its assigned model (any error
  measure — perpendicular distance for lines, Eq. after Fig.4 p.10; transfer/reprojection
  error for affine/homography, Eqs. 7-8 p.19-20; Sampson distance for fundamental
  matrices, Eq.10 p.26).
- **Term 2 (spatial coherence, Potts model):** a smoothness penalty over a neighborhood
  graph `N` (built via Delaunay triangulation of the data points, Fig.5 p.11 — "k-nearest
  or other techniques can be used as well, particularly for higher dimensional data,"
  p.11), weight `w_pq = exp(-||p-q||²/ς²)` for line-fitting (ς=5, heuristic) or `w_pq=1`
  for the vision applications (p.11).
- **Extended energy, Eq. (5), p.15** adds a third, MDL-style **label-cost term**
  `β·|L_L|` (penalty proportional to the number of *distinct* labels/models actually
  used) — this is the "how many primitives" regularizer the task description points at;
  Eq.(5) generalizes Eq.(3) and Li's earlier Eq.(2) (p.5) which combined fitting error
  with only the label-count term (no spatial term).

**PEARL algorithm (Propose–Expand–Re-estimate–Loop), Sec. 2.5 p.13, pseudocode:**
1. **Propose:** randomly sample minimal point-subsets to generate an initial finite pool
   of candidate models `L0 ⊂ R^n` (n=2 for lines, n=6 for affine — exactly RANSAC's
   sampling step), optionally add the outlier label `∅`.
2. **Expand:** run α-expansion graph cuts (Boykov-Veksler-Zabih) to solve the labeling
   for the *current finite* label pool — this is the step that requires Eq.(3)/(5) to
   have a finite label set; continuous parameter spaces (e.g. `R²` for lines) are handled
   only by restricting to the finite proposed pool at each iteration.
3. **Re-estimate:** for each label with non-empty inlier support, re-fit its parameters
   in closed form by minimizing the sum of geometric errors over its current inliers,
   Eq.(4) p.12: `L̂ = argmin_l Σ_{p∈P(L)} ||p-l||`. Discard labels with empty support.
4. Iterate 2-3 (optionally re-**Propose** by sampling more or merging/splitting current
   models) until the energy stops decreasing → **local minimum** (both expand and
   re-estimate are proven to monotonically decrease Eq.3/5, p.13).

**Outlier label `∅` (Sec. 2.3, p.11-12) — directly relevant to our fallback design.**
PEARL introduces a special always-present label `∅` with **constant fidelity**
`||p-∅|| = γ` for every point — i.e., a literal "uniform model" competing on equal
footing with the real geometric models inside the same energy. A point gets assigned `∅`
when no real model fits it well enough to be worth the label/smoothness cost. This is the
paper's formal mechanism for "this point doesn't belong to any recovered structure,"
operating per-point.

---

## 2. Concrete results and numbers

**Synthetic line fitting (Sec. 2, Figs. 6-9, pp.13-18).** 2D points supporting 8 lines
(`n=2` param space), compared against multi-RANSAC, mean-shift, J-linkage, all tuned per
noise level (p.13).
- **Fig. 6, p.14** (quantitative): PEARL vs. RANSAC given the *identical* initial random
  proposal pool `L0`, single-line ground truth, 80% outliers / 20% noisy inliers, errors
  averaged over 400 random tests, x-axis = `|L0|` (5 to ~80). RANSAC error starts ≈1.6-1.7
  at `|L0|=5` and only reaches ≈0.1-0.2 by `|L0|≈60-80`; PEARL starts ≈0.9 at `|L0|=5`
  and is already near-zero by `|L0|≈20-30`. **PEARL needs roughly 3-5× fewer initial
  RANSAC-style samples than plain RANSAC for equivalent parameter accuracy**, on the same
  proposal pool.
- **Figs. 7-9, pp.15-17** (qualitative, 8-line scenes at 25%/35%/45% outlier fractions,
  σ=0.01/0.02/0.025): multi-RANSAC, mean-shift, and J-linkage visibly degrade — splitting
  single lines into pieces, merging distinct lines, or fitting spurious diagonals through
  outlier clusters — especially at the 45%-outlier level (Fig.9); PEARL recovers 8 clean
  lines at all three noise levels. No single scalar error number is given for Figs.7-9
  (qualitative only) — **this is a limitation of the paper's own reporting**, not just
  ours to note.

**Real multi-view stereo, affine/plane fitting ("Clorox" pair), Sec. 3.1, Table 1 p.21.**
Ground truth = manually-annotated intersection lines between real planes in the scene
(human-clicked, not CAD — a directly relevant precedent for the branch's "what is the
ground-truth story" question). Errors = sum of corner-to-line distances vs. this
hand-annotated ground truth:

| Line | BT [dense photoconsistency baseline] | PEARL (transfer error, Eq.7) | PEARL (reprojection error, Eq.8) |
|---|---|---|---|
| L1 | 34.55 | 8.80 | 4.37 |
| L2 | 16.83 | 4.82 | 7.5 |
| L3 | 5.56 | 13.27 | 9.53 |
| L4 | 5.99 | 4.46 | 8.47 |
| **Total** | **62.93** | **31.35** | **29.87** |

PEARL (either error measure) is ~2× more accurate in total than the dense-photoconsistency
baseline BT, using only a *sparse* set of SIFT feature correspondences. Text (p.21) also
states sequential-RANSAC and J-linkage errors "were significantly larger than those
above" for the same data (Fig.15) — **no number given**, qualitative claim only.

**Multi-homography fitting, real images (Sec. 3.2, pp.22-24).**
- Merton College I (Fig.16): PEARL found **7 planes using only 900 randomly sampled
  homographies**, converged in 3 iterations. Multi-RANSAC [Zuliani et al.] needed 11,604
  iterations × 4 samples/iteration = **46,416 total sampled homographies** to fit 4
  models to the *same* data. PEARL doesn't need the number of planes given a priori;
  multi-RANSAC in this comparison did not either but ran far more samples.
- Merton College III (Fig.17): 8 planes found from 3000 initial samples, 4 iterations;
  inlier counts per plane ranged from 13 (small roof patch) to 786 (large wall) —
  demonstrates recovery of very weak/small-support models when the initial sample count
  is generous.
- Raglan Castle Tower (Fig.19): 13 planes from 6000 initial samples, 4 iterations;
  cross-validated visually against a second photo of the same building from a different
  viewpoint (different floors = different parallel planes, confirmed).

**Motion segmentation / multibody structure-from-motion (Sec. 3.3, Hopkins/"checkerboard"
sequences from Vidal's benchmark, 26 sequences, 3 true motions each).**

*Two-frame fundamental-matrix fitting (Eq.11), Table 2, p.26* — misclassification error,
compared to methods given the true model count (3) and using all N frames:

| | **PEARL** | ER1 (PEARL, 15/26 seqs where it got exactly 3 models) | ER2 (PEARL init'd w/ 3 GT models) | REF | GPCA | LSA-4n | RANSAC |
|---|---|---|---|---|---|---|---|
| avg misclass. | **21.2%** | 6.93% | 4.2% | 6.28% | 31.95% | 5.80% | 25.78% |
| median | **11.9%** | 4.15% | 1.2% | 5.06% | 32.93% | 1.77% | 26.01% |

**Critical negative finding — Fig. 22, p.28 (bimodal error histogram).** PEARL's raw
21.2%/11.9% average conceals a **bimodal distribution**: sequences where PEARL correctly
recovered exactly 3 models cluster at low error (5-20%); sequences where it recovered the
*wrong number* of models form a distinct "gross error" mode (40-70% misclassification).
This is the paper's own direct evidence that **getting the model count wrong is a
discrete, catastrophic failure mode, not a smooth degradation** — directly relevant to
the branch's "at what noise/incompleteness level do published methods stop returning the
correct primitive type at all" question, generalized from "primitive type" to "primitive
count."

**Sample-complexity limit, explicit and load-bearing (p.28).** For the checkerboard
sequences (`m1=20, m2=24, m3=56` inliers, `mo=0` outliers, 3 models needed from 8-point
minimal samples), the paper computes `Pr(3-out-of-n samples all good)` and finds **even
n=1,000,000 samples gives only 0.92 confidence** of covering all 3 models — and that many
samples is stated as impractical ("too large to be practical"; their MATLAB
implementation "can not work with very large number of samples due to memory
restrictions," p.28). Their workaround: batch 5000 samples at a time, mix best-so-far
models into the next batch, iterate — 100 iterations ≈ 1 hour; a reduced 30 iterations ≈
20 min but with worse accuracy (ER1 average/median error rises from 6.93%/4.15% to
11.26%/4.8%, p.29). **This is a concrete negative finding about scaling combinatorial
RANSAC-style sampling to guarantee coverage of every true model — PEARL reduces but does
not eliminate this cost**, even for a problem with only ~100 total inlier points.

**Energy-as-fit-quality-signal, Fig. 23, p.28.** Scatter plot across many different random
sample sets (same scene): **energy value (Eq.11) is positively correlated with
misclassification error.** This is presented purely as an *observed correlation used to
select among multiple PEARL runs* (pick lowest-energy solution), not as a calibrated,
externally-validated confidence score — no threshold is proposed or tested against
downstream task pass/fail.

*N-frame / 4D hyperplane formulation (Eq.13, Sec 3.3.2), Table 3, p.31* — same sequences,
different geometric model (motion trajectories as 4D affine subspaces, following
Tomasi-Kanade factorization): PEARL avg/median 19.57%/19.28%, ER1 8.97%/3.53%, ER2
8.15%/3.74%, vs. REF 6.28%/5.06%, GPCA 31.95%/32.93%, LSA-4n 5.80%/1.77%, RANSAC
25.78%/26.01%. The paper explicitly declines to compare the 2-frame vs. N-frame PEARL
formulations to each other ("we do not make any conclusions from this fact as we tested
only one and probably not the most difficult example," p.31) — an honest **unresolved
comparison**, and a caution against over-reading either table as "PEARL's number."

**Hyperparameter sensitivity, Fig. 25 p.31** — a 3D surface of misclassification error vs.
`λ` (smoothness weight) and `β` (label-cost weight) for one checkerboard sequence: error
forms a shallow basin (fairly flat near the optimum) but is clearly sensitive outside it.
Conclusions (p.32) state plainly: **"We did not analyze any specific technique for
selecting these parameters, but we demonstrate that the method is fairly stable with
respect to them."** No principled selection rule is given — a hand-tuning burden the
branch would inherit if it adopted this energy form.

---

## 3. Load-bearing assumptions

- **Discrete/finite label pool is required for the graph-cut solver.** The continuum of
  model parameters (`R^n`) is never actually optimized over directly — it's approximated
  by restricting to a finite, randomly-sampled candidate pool at each Propose step, exactly
  like RANSAC's sampling, just iterated and jointly optimized rather than greedily
  selected. This inherits RANSAC's minimal-sample-set combinatorics (see sample-complexity
  finding above).
- **Neighborhood graph / spatial coherence must be generatively or discriminatively
  justified** (Sec 1, Fig.2 p.3, explicit caveat in footnote 3, p.6: "One can not expect
  spatial regularization to work well for i.i.d. data, in general"). The paper's own line-
  fitting demos are stated to be a *special case* where it happens to work for i.i.d.
  points; the "primary target... [is] applications in vision (Sec.3) where spatial
  coherence is well-founded" (footnote 3, p.6) — i.e., the method leans on points being
  spatially clustered by real surfaces, which does hold for our tabletop-object point
  clouds, but the paper is explicit this is an assumption, not a universal guarantee.
- **Outlier/noise model is unstructured**: outliers are uniformly-distributed random
  points (2D synthetic) or mismatched SIFT correspondences (real images) — i.e., scattered,
  not a coherent missing region or a systematic bias. The fitting-error term and the `∅`
  label are built to catch points that don't fit any model well; they are **not** designed
  to catch a *whole model* that is fit to a systematically biased inlier set (e.g., a face
  whose points are all shifted the same way) — such a bias would not raise per-point
  residuals and so would not trigger the outlier mechanism at all.
- **Runtime/implementation ceiling.** 2012 MATLAB implementation; even ~100-inlier-point,
  3-model motion problems require batched-sampling workarounds and up to ~1 hour runtime
  for full accuracy (p.29). No claim or evidence this scales to survey-cloud sizes
  (thousands-to-millions of points).
- **Stochastic, run-to-run non-determinism.** The Propose step is random sampling; the
  Expand/Re-estimate loop converges only to *a* local minimum, and which one depends on
  initialization (explicit, p.13: "Depending on initialization, PEARL's iterative expand
  and re-estimate steps would converge to different local minima"). Fig. 23's scatter (same
  scene, different random sample sets) empirically shows this variance in both energy value
  and misclassification error.

---

## 4. What transfers to our setting

- **The energy-formulation idea itself** (fitting error + spatial coherence + label-count
  penalty, jointly optimized rather than greedily searched) is the direct conceptual
  ancestor of Multi-X and other "how many primitives" work in this branch's map — citable
  as the origin of that framing (Eqs. 3 and 5, p.10 & p.15) when writing up why our
  primitive-count problem shouldn't be solved by naive sequential RANSAC + threshold.
- **The outlier label `∅` with constant fidelity `γ`** (Sec 2.3, p.11-12) is a genuine,
  citable precedent for "a uniform/null model competing as one of the hypotheses" — i.e.
  it's evidence that the *mechanism* of a uniform fallback is principled within a
  classical energy-fitting framework, not something we'd be inventing from scratch. Note
  the granularity mismatch flagged in §5: this is a **per-point** mechanism, not a
  per-object "abandon the whole decomposition" decision — extending it to an object-level
  two-tier trigger (our "else uniform sphere" contract) is our own extrapolation, not
  something the paper does.
- **Sample-efficiency argument** (Fig.6: ~3-5× fewer samples than RANSAC for equal
  accuracy given the same pool; Merton College I: 900 vs. 46,416 total sampled models,
  p.23) is relevant if we ever consider adopting joint energy-based fitting instead of
  greedy sequential RANSAC — though this baseline predates modern efficient-RANSAC
  descendants (Schnabel et al.), so the specific multiples shouldn't be taken as current
  state-of-the-art comparisons, only as *directional* evidence that joint optimization
  beats naive greedy multi-model sampling.
- **Energy value as an internal fit-quality proxy** (Fig.23, p.28: energy correlates with
  misclassification error across different random runs on the same scene) is weak but
  real evidence for the branch's "computable signals to declare a fit unreliable"
  question — it's an off-the-shelf-obtainable number (the objective being minimized) that
  tracks quality, though never calibrated to an absolute threshold.
- **Bimodal failure histogram (Fig.22)** is a genuinely useful data point for framing our
  own failure-detection experiments: it argues we should specifically watch for and
  measure "wrong primitive count" as a distinct, discrete failure category, separate from
  "correct count, noisy parameters."
- **Ground-truth protocol precedent** (Table 1, p.21: human-clicked intersection lines on
  a real captured scene, not CAD models) is a cheap, directly reusable answer to the
  branch's "what is the ground-truth story" question for at least a coarse sanity-check
  benchmark tier.

---

## 5. What does NOT transfer / negative findings (first-class)

- **No 3D point clouds anywhere in this paper.** All experiments fit models to sparse 2D
  SIFT feature *correspondences* between pairs of camera images (narrow-baseline stereo,
  wide-baseline uncalibrated pairs, or motion-tracked features) — "planes" are recovered
  indirectly as 2D affine transforms / planar homographies between two images of a planar
  patch, never as a plane equation fit directly to depth/range points. There is no depth
  sensor, no PLY input, no range data of any kind in the entire paper. This is a **major
  scope mismatch**: none of the branch's plane/cylinder/sphere/cone-on-a-point-cloud
  machinery is demonstrated or even attempted here.
- **No curved primitives at all** — only lines (2D synthetic), affine transforms/planar
  homographies (2D-image proxies for 3D planes), fundamental matrices (rigid motions), and
  4D hyperplanes (motion trajectories). **No cylinder, sphere, cone, or superquadric is
  ever fit.** Branch questions about the plane+cylinder+sphere+cone vocabulary, or
  superquadrics vs. cuboid decomposition, are not engaged by this paper in either
  direction.
- **No normal estimation, no angular error metric, no uncertainty/covariance output
  anywhere.** All accuracy numbers are point-to-model geometric-error distances (pixels,
  or arbitrary line-distance units) or classification-error percentages. Zero occurrences
  of angular/orientation error as a reported metric. This paper gives **no evidence at all**
  for the branch's core Branch-1 numeric question (angular error on a face normal/cylinder
  axis vs. noise level).
- **No bounded-face extraction beyond point-cluster membership.** PEARL's output is a set
  of (model parameters, inlier point subset) pairs; nothing computes or reports a face
  centroid, extent, or boundary polygon. The Delaunay/neighborhood-graph mechanism *could*
  in principle be repurposed to derive a bounded region from an inlier cluster (this is
  noted as a transferable **idea**, §4), but the paper itself never does this or measures
  it — it's our own extrapolation, not a validated result.
- **No per-primitive viewpoint or downstream action-planning story whatsoever.** The paper
  stops at model fitting/point-labeling; there's no precedent here for "generate a view per
  primitive," positive or negative.
- **Run-to-run instability directly undermines a "face 3 stays face 3" identity contract**
  (branch RQ3). The Propose step is random sampling; the algorithm provably converges only
  to *a* local minimum, which the authors state explicitly depends on initialization
  (p.13), and Fig.22/Fig.23 empirically show both which models get discovered and the
  resulting error can vary meaningfully across runs on the identical scene when sampling is
  inadequate. Nothing in the paper addresses matching/re-identifying labels across separate
  runs or re-scans — that would be entirely additional engineering on top of PEARL, not
  something the energy formulation provides.
- **Outlier/dropout model is the wrong shape for our defect modes.** PEARL's outliers are
  spatially scattered random points (2D) or mismatched feature correspondences — not a
  coherent missing region correlated with material/incidence angle (our glossy/dark
  dropout), and structurally incapable of catching a *systematically biased* inlier set
  (a whole face shifted/tilted consistently) — the paper's own fitting-error term would
  score such a face as a **good fit** (low residual), since bias, by definition, doesn't
  increase per-point variance. This is a genuine, structural blind spot relative to the
  branch's explicit concern that "systematic bias is the dangerous case — it is invisible
  to residual-based confidence."
- **No registration/alignment error modeling, no depth-bias modeling, no discussion of the
  ~4cm fused-vs-single-view offset's likely cause** — entirely out of scope for this paper
  (2D image feature matching, not multi-view depth fusion).
- **No preprocessing (voxel downsampling, outlier removal) discussion** — the paper
  operates on already-sparse SIFT correspondences, never on raw dense point clouds, so
  there's nothing to say about how denoising/downsampling affects primitive recovery.
- **No calibration of the energy-value fit-quality signal against downstream task
  failure** — Fig.23's correlation is observational (used only to pick among PEARL's own
  runs), never turned into a validated "energy below X ⇒ trust the fit" threshold; the
  branch's "if not, we must build that mapping" caveat applies in full.
- **No modern tooling, datasets, or learned-method comparison** — 2012 MATLAB
  implementation, predates Open3D/CGAL-as-commonly-used-today, predates Fit4CAD/SHREC
  2022/ABC benchmarks, predates all deep-learning shape-abstraction methods (SuperDec,
  SuperFlex, etc.) referenced elsewhere in this branch's map.
- **λ and β require hand-tuning with no principled selection rule** (explicit
  self-critique, Conclusions p.32) — inherits directly if we ever adopt Eq.(3)/(5) as a
  formulation.

---

## 6. Mapping to branch questions (direct answers only; rest is negative evidence in §5)

- **Angular error / noise threshold (RQ1, RQ7-11):** not answered — no angular metric
  exists anywhere in the paper (see §5). The closest quantitative analogue is the checkerboard
  motion-segmentation misclassification-error tables (Tables 2-3) and the bimodal "wrong
  model count" failure mode (Fig.22), which speaks to *count* failure, not *angle* accuracy.
- **Object vocabulary closure (RQ2):** not engaged — only lines/planes/rigid-motion
  models are fit; no curved primitives, no superquadrics.
- **Bounded faces (RQ3):** not directly delivered, but the spatial-coherence/neighborhood-
  graph mechanism (Eq.3's second term) is the right *conceptual* tool for turning an
  infinite model equation into a bounded inlier-point region from which a centroid/extent
  could be computed — untested by the paper itself.
- **Primitive decomposition → per-primitive viewpoints precedent (RQ4):** none; paper
  stops at fitting/labeling.
- **Decomposition repeatability / label identity stability (RQ5):** negative evidence —
  the algorithm is explicitly stochastic and only locally-optimal; run-to-run variance in
  both recovered models and energy is empirically shown (Fig.22, Fig.23, p.13 discussion).
- **Partial/occluded cylinder + symmetry completion (RQ6):** not addressed — no cylinders,
  no partial-view completion anywhere.
- **Failure-detection signals (RQ14-15):** the minimized energy value is a real,
  off-the-shelf, no-retraining candidate signal, shown to correlate with misclassification
  error (Fig.23) — but never calibrated into an absolute pass/fail threshold; that mapping
  is left as future work by omission.
- **Two-tier decomposition-or-fallback design (RQ16):** partial precedent only — the
  outlier label `∅` (constant-fidelity uniform model) is a real classical-literature
  instance of "uniform model competes as a hypothesis," but it operates per-point, not as
  a whole-object "abandon decomposition, use bounding sphere" decision; extending it to
  our object-level contract is not something this paper does or validates.
- **Depth dropout / systematic bias (RQ11 continued):** not addressed directly, but the
  paper's per-point residual mechanism is structurally blind to systematic bias by
  construction (see §5) — relevant negative framing for why residual-based confidence
  alone (as flagged in the branch questions) is insufficient for our defect modes.
- **Tooling/benchmarks/learned methods (RQ17-20):** not applicable — 2012 MATLAB, predates
  all named tools/benchmarks/learned baselines.

---

## Gist (for structured return)

PEARL formulates multi-model geometric fitting as global energy minimization — fitting
error + spatial (Potts) coherence + optional MDL-style label-count penalty — solved by
iterating RANSAC-style random model proposal, α-expansion graph-cut labeling, and
closed-form model re-estimation to a local minimum. It is the direct conceptual ancestor
of "how many primitives" energy methods (Multi-X etc.) and needs far fewer samples than
plain multi-RANSAC for equal accuracy (Fig.6; 900 vs. 46,416 samples on a real Oxford
homography scene, p.23). Its outlier label — a constant-fidelity "uniform model"
competing per-point in the same energy — is a real classical precedent for a uniform
fallback mechanism, though at point, not object, granularity. Critically, it operates
entirely on 2D image feature correspondences (never 3D point clouds), fits only
lines/affine transforms/fundamental matrices/hyperplanes (never curved primitives, never
normals, never uncertainty), and its own motion-segmentation results (Fig.22) show a
bimodal failure mode where getting the model *count* wrong causes a discrete jump to
40-70% misclassification — evidence that primitive-count errors are catastrophic, not
graceful. The algorithm is also explicitly stochastic/only-locally-optimal (p.13, Fig.23),
directly undermining any "face 3 stays face 3" repeatability contract without extra
engineering, and its per-point residual mechanism is structurally blind to systematic
(non-random) bias — the exact failure mode this branch is most worried about.
