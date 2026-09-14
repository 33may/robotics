# Integrating One-Shot View Planning with a Single Next-Best View via Long-Tail Multiview Sampling

- arXiv: 2304.00910 (v4, Oct 30 2024; PDF read in full, 21 pages: https://arxiv.org/pdf/2304.00910)
- Venue: IEEE Transactions on Robotics, accepted Oct 2024 (preprint header "IEEE TRANSACTIONS ON
  ROBOTICS. PREPRINT VERSION. ACCEPTED OCTOBER, 2024")
- Authors: Sicong Pan, Hao Hu, Hui Wei (Fudan Univ.), Nils Dengler, Tobias Zaenker, Murad Dawood,
  Maren Bennewitz (Univ. of Bonn, Humanoid Robots Lab)
- Code: https://github.com/psc0628/MA-SCVP · datasets: https://www.kaggle.com/datasets/sicongpan/ma-scvp-dataset
  · demo video: https://youtu.be/gy2bHbBI168
- Prior work this extends: SCVP (Pan & Wei, RA-L 2022, arXiv "Scvp: Learning one-shot view planning
  for unknown object reconstruction") — SCVP is the one-shot baseline throughout this paper.

## Core idea

Two existing paradigms for object-reconstruction view planning both fail at high-quality
reconstruction: **iterative NBV** (search or learning selects one view at a time, hand-crafted stop
criterion, no global path planning — "lacks global path planning capabilities," Fig. 1(a)) and
**one-shot set-covering view planning (SCVP)** (predicts the whole view set at once from a *single*
initial view, so it "struggles to capture sufficient surface details" because a single view gives too
little information about the unknown object, Sec. I). The paper's hypothesis: **more information about
the object improves the one-shot network's prediction quality**, so it combines a small number `k` of
NBVs (to gather more information) with a one-shot network retrained to consume multiview input
(MA-SCVP). The final chosen design uses **k=1** NBV before the one-shot network fires — "1-NBV+MA-SCVP".

Two mechanisms carry this:
1. **Long-tail multiview sampling** (Sec. IV-A): training input cases for the one-shot network are
   object+view-subset pairs; because coverage gain follows a long-tail curve across NBV iterations
   (see below), the dataset is deliberately imbalanced toward multiview cases with *fewer* selected
   views, rather than sampled at random or via the field's usual NBV-reconstruction-trial (NBVR)
   sampling.
2. **MA-SCVP network** (Sec. IV-C): a multilabel-classification CNN (Fig. 11) that takes the current
   OctoMap occupancy grid `M` (32×32×32) plus a binary "view state" vector `V_state` (which of the 32
   candidate views have already been taken) and predicts a 32-bit vector of view scores; thresholding
   at γ=0.5 yields the predicted covering set `V*_cover` directly. **The one-shot network's output size
   *is* the plan length — there is no separate stop-decision step for the one-shot stage** (only the
   preceding `k` NBV iterations need a stop rule, and `k` itself is a fixed hyperparameter chosen by
   ablation, not learned or adapted per-object).

## Problem formulation and system architecture (Sec. III, Fig. 2, Algorithm 1)

- **View space**: 6-DOF pose space `V ⊂ ℝ³ × SO(3)`; candidate views live on the **upper hemisphere**
  around the object (tabletop scenario — "we can ignore the tabletop because the candidate views are
  all above it," Sec. III-C), positioned by **Spherical Codes** (maximize minimal pairwise distance on
  a sphere) — a fixed, geometry-agnostic, dense sampling identical in spirit to a uniform sphere.
  **`|V| = 32`** candidates (Fig. 4(b)), chosen because prior work SCVP uses 32, NBVNet uses 14, PCNBV
  uses 33 (Sec. III-C, footnote-style comparison) — i.e. the field has no consensus candidate count,
  each paper picked its own. Pose is effectively 5-DOF (camera z-axis points at object centroid; the
  6th DOF — roll around that axis — is fixed to "most upward" for minimal change from the previous
  pose, following NBVNet's convention, Sec. III-C, Eq. 1, Fig. 4(a)).
- **3D representation**: OctoMap occupancy grid, dynamically resized so the network always sees a
  32³ grid regardless of absolute object size (`M_res = 2×o_size/32`, capped at object size ≤ 15 cm
  radius) — this is how the network generalizes across object sizes without retraining per size class.
- **Combined pipeline** (Algorithm 1, Definition 3): run `k` NBV iterations (any NBV submodule) →
  compute `V*_cover = f_MA-SCVP(M, V_state)` (one-shot set-covering prediction over the *remaining*
  object) → sequence `V*_cover ∪ {v_k}` into a shortest-path tour via global path planning → execute →
  filter the table out of the final point cloud.
- **Global path planning** (Sec. III-C2, Eq. 2, Definition 4, Fig. 5): local path between any two views
  is straight-line unless an "obstacle sphere" (bounding sphere around the object) is crossed, in which
  case route along the great-circle arc (Eq. 2); the global tour over the selected views is the
  **shortest Hamiltonian path** (not a cycle — no return-to-start), NP-hard in general but solved exactly
  via `O(n²2ⁿ)` dynamic programming since candidate counts stay small (<20) in this setting.

## Set-covering formulation for the one-shot label (Sec. III-B Def. 2, Sec. IV-B, Eq. 4)

Ground-truth training labels are generated by literally solving the covering problem the network is
meant to approximate:

```
minimize: Σ_{v∈V_rest} z_v
subject to: (a) Σ_{v: e∈v} z_v ≥ 1  for all e ∈ U_rest   (every uncovered voxel must be covered
                                                             by ≥1 chosen view)
            (b) z_v ∈ {0,1}
```

Solved via **Gurobi** (an ILP solver) in under 10 s per label (Sec. IV-B1, IV-B3). `U_rest` = uncovered
object voxels after the already-selected views; `V_rest` = the view→voxel-visibility sets with already-
observed voxels removed. This SCOP (set-covering optimization problem) formulation, and its ILP
encoding, is the direct analogue the task brief names for our "action-space-as-set-cover" framing.

## Long-tail sampling mechanism (Sec. IV-A, Fig. 6–7, Algorithm 2, Eq. 3)

- **Empirical finding that motivates everything**: plotting surface-coverage gain vs. NBV iteration
  (Fig. 7(a)) shows "the first 1–3 views (an initial view and 2 NBVs) cover more than 80% of the object
  surfaces, while subsequent views only contribute to the remaining 20%" — a classic long-tail curve.
  Separately (Fig. 6(b)): "most objects in reconstruction trials require **12–15 NBVs** for 100%
  coverage" under random sampling — this number becomes load-bearing later (see Oracle-Count Critique
  below).
- **Sampling rule** (Eq. 3): for object case `c_obj`, cap the number of training samples with
  `n_select` views at `L(c_obj, n_select) = n_single` if `n_select=1`, else
  `⌈n_single × NS(c_obj, n_select)/NS(c_obj,1)⌉`, where `NS` is the surface-coverage-gain function and
  `n_single` (default 32) is a user-set number of single-view cases. This deliberately shapes the
  sampled-case distribution to mirror the long-tail coverage-gain curve (fewer, but not zero, samples
  at large view counts) rather than the near-uniform distribution random/NBVR sampling produces
  (Fig. 7(b): "Whole" curve stays near-constant to 20 views, "Long-Tail" drops sharply after ~4).

## Evaluation metrics (Sec. V-A3, Eqs. 8–10)

Four metrics, computed per test case and reported as mean ± std over the test set:
- **Visible Surface Coverage** `VSC = |U_cover| / |U|` (Eq. 8) — `U` is the set of **visible** surface
  voxels only (bottom/interior voxels invisible to any candidate view are excluded by construction,
  Fig. 13) — a pure occupancy-count ratio, **no incidence-angle term of any kind**.
- **Required Views** `RV = |V*|` (Eq. 9) — total views including the initial view; this is the paper's
  stand-in for "view budget used," reported as an outcome, not fixed as an input, for the one-shot and
  combined systems (their auto-stop criterion decides it).
- **Movement Cost** `MC = Σ_{i=0}^{RV-2} Path(v_i, v_{i+1})` (Eq. 10) — sum of local path lengths (Eq. 2)
  over the executed view sequence, in meters. This is the metric the task brief calls the "movement-cost
  comparison axis."
- **Reconstruction Time** `RT` — total wall-clock execution time (real-world experiments only).

Statistical testing: **paired t-test AND Wilcoxon signed-rank test** (chosen "due to some non-normally
distributed data"), α=0.05, applied to **matched test cases** (every method evaluated on the identical
30 objects × 5 rotations × 20 initial views = 3,000 cases, Sec. V-A2) — a directly reusable statistical
protocol for Q12: paired significance tests over an identical case set, not independent-sample
comparison.

## Concrete results

**Main comparison, Table VII (simulation, 3,000 test cases)** — iterative NBV under two stop-criterion
strictness levels (`m`-based frontier-voxel-variance stop, Delmerico et al. movement-weighted utility
variant tagged `+Mov`), one-shot SCVP, and the combined system:

| System | RV | VSC (%) | MC (m) |
|---|---|---|---|
| PCNBV (strict, <0.1%) | 10.40±3.30 | 97.03±2.33 | 5.21±1.36 |
| GMC (strict, <0.1%) — 2nd-highest VSC | 11.27±3.82 | **97.86±2.29** | 5.57±1.50 |
| MCMF (strict, <0.1%) | 11.39±3.83 | 97.85±2.40 | 5.73±1.94 |
| One-shot SCVP (auto-stop) | 9.61±0.96 | 96.12±6.29 | 2.28±0.15 |
| **Combined: 1-NBV+MA-SCVP (auto-stop, ours)** | 13.01±1.46 | **98.41±2.67** | **3.07±0.24** |

Abstract/Sec. V-E headline: "45% reduction in movement cost compared to state-of-the-art systems" —
verified arithmetically against Table VII: `(5.57 − 3.07)/5.57 ≈ 44.9%` against GMC-strict, the
second-highest-VSC baseline (text, p.14: "reduces MC by 45% compared to the second-highest VSC method
(GMC with <0.1%)"). Lenient-threshold (<0.5%) iterative methods reach comparable MC but "exhibit much
lower VSC"; strict-threshold methods reach comparable VSC but "exhibit much higher MC" — the combined
system is presented as dominating both ends of that trade-off, not merely averaging it.

**Ablation over `k` (number of NBVs before the one-shot network fires), Table V/VI:**

| k (NBVs) combined with MA-SCVP (long-tail dataset) | RV | VSC (%) | MC (m) |
|---|---|---|---|
| 0 | 13.13±1.43 | 98.10±3.73 | 2.59±0.23 |
| 1 (chosen) | 12.90±1.44 | 98.41±2.67 | 3.07±0.24 |
| 2 | 13.06±1.51 | 98.56±2.35 | 4.29±0.25 |
| 3 | 13.29±1.56 | 98.60±2.25 | 5.30±0.31 |

Table VI significance: 0-NBV+MA-SCVP vs. 0-NBV+SCVP(singleview) is `***` (p<0.001) VSC gain, 14% MC
growth; 1-NBV+MA-SCVP vs. 0-NBV+MA-SCVP is `***` VSC gain, 19% MC growth; **2-NBV and 3-NBV vs. 1-NBV
are `n.s.`** (not significant) on VSC while costing 20%/40% more MC. The paper's own conclusion,
stated plainly: "a stable plateau is observed" beyond k=1 — **an explicitly reported negative/no-benefit
result** (more information stops helping past one extra view, but keeps costing movement), directly
useful precedent for our Q13 (pre-registering and reporting a "more views doesn't help" outcome).

**Object-complexity stratification, Table XII** — objects bucketed by SCOP-solved minimum-covering-view-
count into simple (4–7 views, 314 cases), medium (8–12, 1,729 cases), complex (13–16, 957 cases):

| Method | Simple | Medium | Complex |
|---|---|---|---|
| 0-NBV+SCVP | 99.67±0.41 | 97.87±2.75 | 91.83±9.08 |
| 0-NBV+MA-SCVP | 99.95±0.11 | 98.99±1.73 | 95.87±5.55 |
| 1-NBV+MA-SCVP (ours) | 99.95±0.14 | 99.01±1.48 | **96.84±3.80** |
| 2-NBV+MA-SCVP | 99.96±0.06 | 98.99±1.36 | 96.83±4.02 |

The extra online NBV mainly helps **complex** cases (95.87→96.84%); simple cases are already
saturated at ~99.9–100% regardless of method — a concrete instance of "does the ranking/benefit change
under harder conditions" (relevant template for our Q9/Q11 stratification, though the stratifier here
is an independently-computed complexity proxy — min-cover-set size — not a primitive/compound/near-
primitive taxonomy).

**Inference/timing, Table VIII**: combined-system inference is **0.119±0.038 s**, needing only **2**
network calls total (1 for initial view, 1 for the NBV) vs. **~7–11** iterative calls for search-based
NBV methods (their stop criterion requires repeated re-evaluation). Total system time ≈ 0.564 s
(inference + 2 OctoMap updates at 0.163±0.111 s each) — the "one-shot" design's efficiency argument is
architectural (fewer perception/planning round-trips), not just movement-cost.

**NBV submodule swap ablation, Table XIV** (holding MA-SCVP fixed, varying which NBV method supplies
the k=1 view): PCNBV gives the best downstream VSC (98.41±2.67, lowest std); NBVNet 98.16±3.56; even
**Random** NBV gives 98.34±3.19 — i.e. the one-shot network is fairly robust to *which* single view it's
given, though "a generally positive correlation between the performance of the NBV method and the
performance of our MA-SCVP network" holds. This is a controlled hold-one-stage-fixed-vary-the-other
methodology, analogous in structure (not content) to the oracle-selector ablation our Q14 needs.

## The "oracle view count" critique (Appendix B-C, Table XVI) — directly answers the task brief's flag

Verbatim (p.18, "Stop Criterion Analysis," second experiment): *"we randomly select a fixed number of
views and utilize our global path planning for comparison. The fixed number of views is determined by
prior knowledge regarding how many NBVs are required to fully cover an object, specifically 12–16, as
shown in Fig. 6. **It is important to note that these values are derived from training objects, and
acquiring this knowledge in advance for unknown testing objects is unrealistic.**"* Table XVI then
shows random-view-selection-with-optimal-global-path-planning at fixed budgets 12/13/14/16 views all
scoring **lower VSC** than 1-NBV+MA-SCVP's auto-stop result (98.41±2.67% at RV=13.01±1.46) even at
matched or larger view counts — e.g. at the closest matched budget (13 views) random scores ~95.5% VSC,
several points below the auto-stop system at essentially the same view count. The paper's own reading
(p.19): "our method not only predicts a sufficient RV but also effectively anticipates the view
configuration, i.e., which candidate views need to be selected" — the win is attributed to *which*
views, not merely *how many*.

**Why this matters for us**: this is a direct, citable warning against baking a view-count constant
(learned from a training set) into a fixed-budget baseline for unseen test objects — precisely the
risk our Q5/Q6 flag ("is a denser uniform sphere at matched budget a fair baseline" presupposes we know
what budget to match to, and this paper shows that number is training-set-derived and shouldn't be
trusted to transfer).

## Ground-truth strategy — two different answers for sim vs. real (Q2/Q4/Q10 relevance)

- **Simulation** (Appendix A, Algorithm 4, Fig. 17): GT is the object **mesh** itself; per-view visible
  voxels are obtained by ray-casting through the mesh ("de-projecting pixels of the virtual image plane
  until it hits an occupied voxel") — Q4's exit (i) (mesh/CAD as GT), but **no injected sensor defects
  at all** in the simulated pipeline (clean raycasting, no noise/dropout/pose-error model anywhere in
  the paper for the 3,000-case simulation study).
- **Real world** (Sec. V-F1): GT is built by **exhaustively capturing all 32 candidate views** and
  voxelizing the union into a 0.002 m grid — i.e. Q4's exit (ii), "densely pre-capture and treat the
  dense set as GT," concretely instantiated. Load-bearing caveat, stated by the authors themselves:
  *"slight calibration errors may result in a small offset in the point cloud for each view. To address
  this issue, for a voxel in the ground truth model, we search for its corresponding neighbors. If any
  neighboring voxels are covered, it is included in the coverage count."* This is a **neighbor-tolerance
  trick to absorb small per-view registration offset directly into the coverage GT** — a concrete,
  reusable precedent for our Q10 (fused-vs-single-view offset): score coverage/quality with a spatial
  tolerance band rather than requiring exact-voxel correspondence. Caveat for us: their offset is
  described only as "slight" (< the 0.002 m GT voxel size, implicitly) — nothing here validates that the
  same trick scales cleanly to our reported ~4 cm offset, which is ~2000× their voxel size; the
  *mechanism* transfers, the *magnitude* it was validated at does not.

## Symmetry / view-space rotational invariance (Q8 relevance)

Appendix A states explicitly why object rotation augmentation is required: *"this implies the inputs to
the network have lost the information of rotational invariance and symmetry in the hemispherical view
space. Consequently, we incorporate `R` and `V` to enable the network to generalize across various
object rotations and initial views."* Concretely: 40 training objects × **8 rotations** (45° steps about
the up-axis) × the fixed 32-view candidate space = 10,240 whole-sampling-space cases (Appendix A). This
is a materially **different** answer to view-equivalence-under-symmetry than Dutagaci's explicit
symmetry-set/Fourier-Mellin equivalence classes (round-2 map, entry `dutagaci-best-view-benchmark`): here
symmetry is handled by **brute-force rotation augmentation of training data**, not by an explicit
equivalence relation in the scoring function — cheaper to implement, but it only helps a *learned*
predictor generalize; it gives us no formal equivalence-class definition to reuse for an analytical or
rule-based benchmark scorer.

## Reachability / candidate-view-space rigidity (RQ3/RQ4, framing-doc constraint relevance)

Two places this surfaces:
- **Real-world workaround** (Sec. V-F2): unreachable candidate views are handled by *nudging the whole
  view space* — "we adjust the candidate view positions by moving them up by 0.008 m," "we lower the
  table height... allowing for less constrained view poses," and as a last resort "we directly solve
  the joint inverse solution for the target view" — i.e. the action space is perturbed globally to fit
  the arm's workspace, never dynamically pruned per-object.
- **Author-acknowledged limitation** (Sec. VI Discussion, 2nd point, verbatim): *"the candidate view
  space is pre-defined, and the network output is bound to a certain candidate view. This does not
  allow the system to drop some unreachable candidate views, such as due to obstacles and robot
  workspace. A possible solution is using a regression network... and training it with the state of
  unreachable views."* This is a direct, citable confirmation that a **fixed discrete candidate space
  that cannot exclude kinematically-infeasible views is a known, unsolved limitation in this exact
  problem family** — relevant to our RQ3/RQ4 (our framing doc's "not every desired view is reachable"
  constraint) as evidence we are not the first to hit this, and that the field's own proposed fix
  (predict reachability as an extra learned signal) is still speculative, not demonstrated.

## What transfers to our setting

- **Movement-cost formalization** (Eq. 2, Eq. 10: local obstacle-aware path + shortest Hamiltonian tour)
  is a ready-made, non-trivial "path efficiency" axis to report alongside any orthogonality/coverage
  metric we define — directly reusable if we want a third comparison axis beyond view-quality and view-
  budget.
- **Paired-test-on-matched-cases statistical protocol** (identical object/rotation/initial-view triples
  across all compared methods, paired t-test + Wilcoxon, α=0.05, significance stars per pairwise
  comparison) is a directly copyable template for Q12, and scales down fine to our smaller object count
  since it's about *pairing*, not *sample size*.
- **Neighbor-tolerance GT construction** for absorbing small per-view registration offset (Sec. V-F1) is
  a concrete mechanism for Q10, though validated only at "slight" (sub-2mm) magnitudes.
- **Complexity stratification via an independently-computed proxy** (SCOP min-cover-set size → simple/
  medium/complex tertiles, Table XII) is a reusable *method* for Q11's difficulty stratification, usable
  even though this paper's own object set (organic/freeform scans) doesn't match ours.
- **The "derived-from-training-objects" critique of fixed view-count baselines** (Appendix B-C, Table
  XVI) is directly citable ammunition for Q5/Q6: don't calibrate a fixed-budget baseline's view count
  from training-time statistics and assume it transfers to unseen test objects.
- **Auto-stop-as-plan-length** (the one-shot network's output size *is* the stopping decision, no
  separate criterion needed for that stage) is the concrete mechanism the task brief flagged this paper
  for — directly informs Q6's "auto-stop-with-cap" option: here the "cap" is implicit in `|V|=32`
  candidates, and the network reliably converges to 12–16 without ever needing an explicit ceiling in
  practice.
- **Tabletop, single-object, eye-in-hand-RGB-D-on-a-manipulator setting** (Sec. V-F: UR5-class arm +
  Intel RealSense D435, close cousin of our D405) with real-world validation is a reasonably close
  platform analogue, lending some real-world credibility to numbers that otherwise come from simulation.

## What does NOT transfer / negative findings (first-class)

- **Zero notion of view-angle/orthogonality quality anywhere in the metric stack.** `VSC` (Eq. 8) is a
  pure voxel-was-seen-by-any-ray occupancy ratio; the raycasting visibility test (Appendix A, Fig. 17)
  applies no incidence-angle threshold or weighting — a voxel counts as "covered" even from a grazing,
  highly oblique viewing angle, as long as line-of-sight exists at the chosen OctoMap resolution. This
  is the single most important negative-transfer finding for our branch: **this paper's entire
  evaluation machinery cannot express, and never attempts to express, "is this view orthogonal to the
  inspected surface."** Achieving 98% VSC with 13 views from a fixed 32-view sphere is evidence that
  *coverage* is achievable with a smart subset-selector on a generic dense sphere — it is **not**
  evidence that those 13 views are good *evidence* views by our (orthogonality) standard. Any temptation
  to treat this paper's strong coverage numbers as validating "fixed generic action space + smart
  selection is good enough, skip geometry-conditioning" should be resisted: it answers a different,
  weaker question than ours.
- **No sensor-defect injection or robustness sweep of any kind** (Q9) — simulation uses clean mesh
  raycasting throughout the 3,000-case study; real-world section only *observes* incidental noise
  post-hoc (Fig. 14, "<1% noise points") without parameterizing or sweeping it. Cannot tell us whether
  method rankings are stable under pose error/dropout — this paper offers a simulation-*harness* pattern
  (fixed candidate pool, mesh-based GT) to extend with defect injection, but supplies none of the
  defect-robustness result itself.
- **Object set is the structural opposite of ours**: 70 scanned meshes from Stanford 3D Scanning Repo /
  Linemod / HomebrewedDB / Clutter (dragons, cats, cups, mechanical parts, cluttered scenes — Fig. 12) —
  organic/freeform/complex shapes, explicitly *not* primitives, with no box/cylinder/sphere stratification
  and no "near-primitive, ambiguous fitting" test case. The action space itself (32-view generic sphere)
  is geometry-agnostic by design and never adapted per-object-shape — this paper does not attempt, and
  gives no evidence for or against, a geometry-conditioned action space (our RQ1–RQ3's actual subject).
- **No VLM, no language model, no agentic orchestrator anywhere.** Q3 (judge circularity), Q15–Q17
  (grounded success vs. lucky answer, named VLM spatial failure modes, judge controls) have zero
  applicable material — this is a pure geometric/learned-scalar-fitness NBV+one-shot system, no
  perception-language loop at all.
- **`k` (number of NBVs before one-shot fires) is a fixed, globally-tuned hyperparameter (k=1), not an
  adaptive per-object decision.** The paper explicitly ablates k=0..5 and picks one constant for the
  whole system — it does not investigate letting an agent (learned or otherwise) *decide* per-object
  how much information to gather before committing to a one-shot plan, which is closer to what a VLM
  orchestrator in our system would need to do.
- **View-quality vs. downstream-task-outcome relationship never tested.** No grasping, no watertight-
  meshing, no defect-detection experiment — future-use downstream tasks are named in the Introduction
  and Fig. 1 caption as motivation only, never evaluated. Cannot serve as evidence either way for Q19's
  "does view quality diverge from inspection outcome" question — it's silent, not disconfirming.
- **Unit of evaluation is whole-object-trial, not per-surface-region** (Q1): `VSC`/`RV`/`MC`/`RT` are all
  computed once per (object, rotation, initial-view) test case over the *entire* reconstruction run — no
  per-face or per-region score exists anywhere in the paper. If we adopt a per-region evidence-view
  score, we are extending past this paper's evaluation unit, not adapting it.

## Load-bearing assumptions

- Single object in an otherwise-known/emptied workspace ("we assume that there is only one object to be
  reconstructed," Sec. III-B); multi-object or region-of-interest scenes require external segmentation
  (Sec. VI, 3rd limitation) — matches our tabletop-single-object setting, but would not extend to
  compound multi-part scenes without that same caveat.
- Object size bounded to ≤15 cm radius (Sec. IV-C, "we assume `o_size` is not greater than 15 cm") —
  the dynamic OctoMap resizing scheme depends on this bound; larger objects are out of scope by
  construction, not merely under-tested.
- Bottom/interior surfaces are permanently unobservable and explicitly excluded from `U` (the coverage
  denominator, Fig. 13) — a tabletop assumption baked into the metric itself, not just the setup; VSC=
  100% by definition never requires seeing the underside.
- The whole evaluation assumes a **watertight, hole-free mesh** as implicit ground truth; the Discussion
  (Sec. VI, 1st point) names objects with holes (e.g. the "Cheff" object, Fig. 12) as an explicit failure
  mode the system is not designed to handle, calling it out of scope for "the majority of real-world
  objects [which] are solid inside."

## Citation map (for checkability)

| Claim | Section | Figure/Table/Eq |
|---|---|---|
| Motivation / gap between iterative and one-shot | §I | Fig. 1 |
| System overview, view space definition | §III-A | — |
| Combined pipeline algorithm | §III-B | Algorithm 1, Definitions 1–3 |
| View pose parameterization (5-DOF), 32 candidates | §III-C | Eq. 1, Fig. 4 |
| Local/global path planning, Hamiltonian path | §III-C1–2 | Eq. 2, Definition 4, Fig. 5 |
| Long-tail coverage-gain phenomenon | §IV-A2 | Fig. 7(a) |
| Long-tail sampling rule | §IV-A3 | Eq. 3, Algorithm 2, Fig. 7(b) |
| Random-sampling unsuitability (12–15 NBVs to converge) | §IV-A1 | Fig. 6 |
| SCOP ILP formulation | §IV-B3 | Eq. 4 |
| Automatic dataset generation | §IV-B1 | Algorithm 3, Fig. 8–9 |
| MA-SCVP network architecture, loss | §IV-C | Eq. 5–7, Fig. 11 |
| Evaluation metrics (VSC, RV, MC) | §V-A3 | Eq. 8–10 |
| Test-case construction (3,000 cases) | §V-A2 | — |
| Comparative baselines list | §V-A4 | — |
| Network-training ablation (sampling method × arch) | §V-B | Table I–III |
| NBV-method-selection ablation | §V-D1 | Table IV |
| k-NBV ablation (final choice k=1) | §V-D2 | Table V, VI |
| Main system comparison, 45% MC reduction | §V-E | Table VII |
| Inference/update timing | §V-E3 | Table VIII |
| Real-world GT construction, neighbor-tolerance | §V-F1 | Fig. 14 |
| Dynamic bounding box vs. GT bounding box | §V-F2 | Table IX, Fig. 15 |
| Real-world system comparison | §V-F5 | Table X |
| Real-world stop-criterion parity check | §V-F6 | Table XI |
| Object test-case complexity stratification | §V-G | Table XII, Fig. 16 |
| Author-stated limitations (holes, unreachable views, single-object, size bound, no bottom surface) | §VI | — |
| λ/γ ablation | Appendix B-A | Table XIII |
| NBV submodule swap ablation | Appendix B-B | Table XIV |
| Stop-criterion parity + oracle-view-count critique | Appendix B-C | Table XV, XVI |
| Whole-sampling-space generation, rotation augmentation rationale | Appendix A | Algorithm 4, Fig. 17 |

## Open gaps from this pass

- Exact Spherical-Codes angular layout of the 32-view candidate space (elevations/azimuths) is
  described only as "Spherical Codes... maximize the minimal distance between them" (Sec. III-C) —
  the specific 32-point configuration is not tabulated in-paper; would need the released code
  (`github.com/psc0628/MA-SCVP`) to recover exact angles if we want to reproduce or compare against it
  directly.
- Table XVI's exact per-fixed-budget numbers (12/13/14/16 views) were read from the rendered PDF table
  image rather than machine-extracted text; the qualitative conclusion (all fixed-budget random
  baselines underperform auto-stop VSC at matched/larger view counts) is solid, but if exact digits
  become load-bearing for a citation, re-verify against the PDF table directly rather than this note.
- No sensitivity analysis of `|V|=32` itself (would fewer/more candidates change the coverage-vs-MC
  trade-off) — not run in this paper.
