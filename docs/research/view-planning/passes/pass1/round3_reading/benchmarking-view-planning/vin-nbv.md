# VIN-NBV: A View Introspection Network for Next-Best-View Selection for Resource-Efficient 3D Reconstruction

- arXiv: 2505.06219 (v3, HTML full text read: https://arxiv.org/html/2505.06219v3; PDF: https://arxiv.org/pdf/2505.06219)
- Venue: not stated in the fetched text (arXiv preprint, 2025)

## Core idea

VIN-NBV replaces coverage-maximization (the field's default NBV proxy) with a learned predictor of
**Relative Reconstruction Improvement (RRI)** — the fractional drop in Chamfer Distance a candidate
view would cause if acquired — and runs a simple greedy sequential policy on top of it (imitation
learning, no RL). Motivating claim (Introduction, verbatim): "if the fence of the house occludes a
part of the wall, a single viewpoint capture may satisfy the coverage criterion, but will lead to a
poor reconstruction with holes in the wall unless additional viewpoints are captured" — i.e. coverage
(empty-pixel counting) is blind to *how well* a region will reconstruct, only whether it was *seen at
all*.

## Problem formulation (Section 3.1)

Agent has `k` base images `I_base={I_1,…,I_k}` with known camera params `C_base` and depth maps
`D_base`. Goal: predict `m` next-best-view camera poses `C_nbv={C_1,…,C_m}`. Fitness (Eq. 1):

```
C*_nbv = argmax_{C_nbv} [CD(R_base, R_GT) − CD(R_final, R_GT)] / CD(R_base, R_GT)
```

i.e. the objective is defined directly in terms of a **ground-truth point cloud `R_GT`** and Chamfer
Distance to it — the benchmark's entire fitness signal is unavailable without GT geometry. This is
the paper's own load-bearing assumption, not incidental to one baseline.

## Method (Section 3.2–3.4)

- **RRI (Eq. 2)**: `RRI(q) = [CD(R_base,R_GT) − CD(R_base∪q, R_GT)] / CD(R_base,R_GT)` — normalized,
  per-candidate-view improvement if that one view were added. Computable at training time only
  because GT is available for every training scene.
- **VIN network (Eq. 3–4)**: `RRI_hat(q) = M_φ(E_θ(F_v ⊕ F_p) ⊕ F_empty ⊕ F_base)`. Inputs: point
  cloud enriched with surface normals, visibility count, depth; projected to a 512×512 grid
  downsampled to 256×256 giving per-pixel variance `F_v` and point features `F_p`; a 2-element
  coverage feature `F_empty` (Section: "counts 'empty' pixels inside/outside reconstruction hull");
  `F_base` = acquisition-stage indicator. 4-layer CNN encoder (256-dim) → concat → 3-layer MLP → RRI
  prediction.
- **Training as ordinal classification**: RRI is discretized into **15 ordinal classes** (0 =
  least improvement, 14 = most), trained with **CORAL** ranking-aware loss instead of regression,
  justified as reducing "catastrophic misclassifications." Stage-dependent z-score normalization
  (per acquisition-stage mean/std, tanh soft-clip, then re-binned into 15 roughly-equal-count bins)
  handles distribution shift across acquisition stages — i.e. "good enough at step 2" and "good
  enough at step 15" are on different scales and the label space is re-normalized per stage.
- **Greedy rollout (Algorithm 1)**: reconstruct → sample `n` query views → score each with VIN →
  acquire highest-scoring → update reconstruction → repeat until a termination criterion (view/time
  budget) is hit. No RL, no lookahead beyond one step — pure greedy.
- Implementation: PyTorch + PyTorch Lightning + PyTorch3D, 4-layer CNN / 3-layer MLP, AdamW,
  cosine LR schedule (1e-3), 60 epochs, ~24h on 4×A6000 (Section 7.2, appendix).

## Experimental setup (Section 4)

- **Candidate/action space**: "we uniformly render 120 viewpoints in 3 hemispherical shells around
  the object from which the policies can sample during evaluation" — **the same fixed 120-view set
  for every object**, not per-object adaptive sampling. Exact per-shell elevation angles, azimuth
  spacing, and radii are **not given anywhere in the main text or the appendix sections fetched** —
  only the total count (120) and shell count (3) are stated. This is the "identical fixed 120-view
  candidate set in hemispherical shells" the branch brief refers to; I could not recover the finer
  geometry (deg spacing) from the HTML — would need the appendix figure or released code to confirm
  exact angles.
- **Base view init**: "the first base view randomly and then set the second base view to be the one
  closest to the first" — random start, not fixed canonical pose.
- **Datasets**: training on a modified subset of **Houses3K**; testing on **OmniObject3D** categories
  — houses (26 usable objects after excluding one with missing GT, out of 27), plus dinosaurs,
  trucks, toy animals, toy motorcycles used only as unseen-category generalization tests, **exact
  object counts per non-house category not stated** in the fetched text. Houses are explicitly
  building-scale: "all objects [scaled] to a size of 14 meters to simulate a more realistic capture
  setting" (Section 4.2) — this is aerial/drone-scale scanning, not tabletop.
- **Platform framing**: Section 4.2 states "we assume that the agent is a drone equipped with a
  depth sensor and camera and travels at a constant velocity of 4 mph" — all fully simulated
  (rendered RGB-D), no real drone, no real sensor.
- **Ground truth**: "we use ground-truth depth maps in RGB-D capture" — explicitly **noise-free**,
  acknowledged as a limitation (Section 5, verbatim below). Reconstruction is built by
  back-projecting RGB-D into 3D with known camera poses (`C_base` given, not estimated) — **no pose
  noise, no registration/ICP error, no dropout, no sensor-defect model of any kind** anywhere in the
  paper.
- **Budget**: Table 1/2 main results use **20 acquisitions**. Time-limited variant (Section 4.2)
  sweeps time budgets including a 15 s case; a 1 m collision-clearance constraint is tested
  separately (Section 4.3).
- **Statistics**: results are reported as a single mean Chamfer Distance per method per dataset;
  **no variance/std, no per-object breakdown, no significance testing** found anywhere in the
  fetched sections — a benchmark-methodology gap directly relevant to Q12.

## Baselines (Section 4)

1. **Random Hemisphere** — random sampling from the 120-view set.
2. **Uniform Hemisphere** — deterministic uniform sampling from the same set.
3. **Uncertainty-Guided** — (named in Table 1, not otherwise characterized in fetched text).
4. **ActiveRMAP** (arXiv'22) — prior NBV method.
5. **Scan-RL** (ECCV'20) — RL with coverage-based reward.
6. **GenNBV** (CVPR'24) — RL with coverage-based reward, current SOTA prior to this paper.
7. **Cov-NBV** — same greedy sampling-based sequential strategy as VIN-NBV, but fitness = coverage
   (empty-pixel count, Eq. 5) instead of RRI. This is the paper's controlled ablation isolating
   "fitness signal" from "search strategy" — **only comparison in this paper that holds the search
   procedure fixed and swaps only the scoring function**, which is exactly the Oracle/Cov/VIN triple
   the branch brief flags as clean template.
8. **Oracle-NBV** — same greedy strategy as VIN-NBV but with `RRI_hat` replaced by the **true** RRI
   computed against ground-truth geometry (i.e. cheats by having GT at test time). Defined as "a
   variant of the VIN-NBV policy that selects the best image at each step where the fitness
   function... is replaced with the ground-truth Relative Reconstruction Improvement." This is a
   genuine upper bound: same action space, same greedy search, only the fitness oracle differs.
   Numeric values only appear as curves in Figure 4, not tabulated — Oracle-NBV Chamfer Distance is
   "substantially lower... throughout acquisition stages, with the gap largest early on."

## Results

**Table 1 (OmniObject3D Houses, 20 acquisitions, Chamfer Distance in cm, lower is better):**

| Method | CD (cm) |
|---|---|
| Random Hemisphere | 0.48 |
| Uniform Hemisphere | 0.41 |
| Uncertainty-Guided | 0.41 |
| ActiveRMAP (arXiv'22) | 0.38 |
| Scan-RL (ECCV'20) | 0.37 |
| GenNBV (CVPR'24) | 0.33 |
| **VIN-NBV (ours)** | **0.20** |

**Table 2 (unseen-category generalization, 20 acquisitions, CD in cm):**

| Method | Dinosaurs | Trucks | Animals |
|---|---|---|---|
| Scan-RL | 0.06 | 0.88 | 0.17 |
| GenNBV | **0.03** | 0.81 | 0.16 |
| VIN-NBV (ours) | 0.04 | **0.57** | **0.08** |

**Negative result buried in Table 2**: on Dinosaurs, VIN-NBV (0.04) is *worse* than GenNBV (0.03) —
the paper's headline "~40% better than RL baselines" does not hold category-by-category; it holds
on average / on the two categories with the largest gaps (Trucks, Animals). Not flagged as a
limitation by the authors — I surfaced it by reading Table 2 row-by-row, not from their prose.

- **Coverage-baseline comparison**: "~30% gain in reconstruction quality over a coverage-based
  criterion using the same greedy strategy" (Abstract) — this is the Cov-NBV comparison; per
  Figure 4, the gap is **stage-dependent**: "large improvements... happen during the initial
  stages... while towards the later stages, both policies result in similar reconstruction
  quality" — i.e. VIN-NBV's advantage over coverage-only is front-loaded (early views matter most),
  converging as the budget is exhausted. This matters for our small-fixed-budget regime (Q6):
  if we only get 5–8 views per object, we are exactly in the regime where their reported gain is
  largest; if we allow many views, their own data says the advantage shrinks.
- **Coverage/F1 vs. step (Figure 5)**: "mean coverage percentage and F1 score across all
  OmniObject3D house objects at each acquisition step" — both metrics improve monotonically with
  more views, VIN-NBV ahead of Cov-NBV throughout, but exact F1/coverage numbers and any
  distance-threshold definition for F1 are **not given anywhere in the fetched text** (Figure only,
  no accompanying equation or threshold value found).
- **Time-limited (Section 4.2, Figure 6)**: at 4 mph drone speed and a 15 s time limit, VIN-NBV
  beats Cov-NBV Chamfer Distance by "~25%"; sweep over multiple time budgets shown only graphically.
- **Collision avoidance (Section 4.3, Figure 7)**: adding a straight-line-path ≥1 m clearance
  constraint to NBV selection, VIN-NBV "still performs well and outperforms the Cov-NBV baseline" —
  no numeric table, qualitative claim only.
- **Ablation**: `F_empty` (the coverage feature) contributes "minimal impact early (stages 0–5) but
  substantial contribution in later stages" (Figure 4 ablation) — i.e. even VIN-NBV's own best
  model leans on a coverage-style signal once early-stage reconstruction-quality gains are
  exhausted, partially undercutting the coverage-vs-quality framing as a clean dichotomy.

## Limitations (Section 5, verbatim)

"We use ground truth depth maps in RGB-D capture to match the evaluation setting of the prior works
GenNBV and ScanRL, but these depth maps are noise-free and not reflective of true depth sensor
readings. The performance of our method has not been evaluated with real world data and could have
different performance in such settings. In Fig. 4 we see a gap to the Oracle-NBV in early capture
stages, indicating that our method still has room for improvement."

Three explicit, author-stated gaps: (1) no real sensor noise tested, (2) no real-world data at all
(fully simulated), (3) even the best learned method underperforms its own oracle early in the
capture sequence.

## What transfers to our setting

- **The Oracle/learned/coverage-baseline triple, holding search strategy fixed and swapping only
  the fitness function** (Oracle-NBV vs. VIN-NBV vs. Cov-NBV, all three using the identical greedy
  loop over the identical 120-view set) is a clean, directly reusable benchmark-design pattern for
  Q5/Q14: it isolates "is the scoring signal good" from "is the search/action-space good," which is
  exactly the ablation Q14 asks for (separating "action space is better" from "VLM used it better").
  We would substitute: Oracle = analytical best-view-per-primitive-region; our-system = VLM-driven
  selection; Cov-NBV-analog = a coverage/uniform-sphere baseline.
- **Fixed, shared, dataset-wide candidate view set** (120 views, identical across all objects) is
  the direct precedent the branch brief cites for a uniform hemisphere baseline design — reusable
  as a methodological template ("policies sample from the same fixed candidate pool") even though
  we cannot recover their exact angular spacing from the text.
  **Answer to Q5's specific question**: this paper's own "Uniform Hemisphere" baseline (0.41 cm)
  is a *weak* baseline in their results — beaten by every learned/RL method and only marginally
  ahead of Random Hemisphere (0.48 cm) — evidence that uniform-sphere-only is not a strong enough
  fallback to treat as "good enough," supporting the branch's instinct (Q5) that a denser
  uniform-sphere-at-matched-budget check is warranted before trusting it as a fair floor.
  Uniform beats Random by a comparatively small margin (0.41 vs 0.48, ~15%) while informed methods
  (VIN-NBV, GenNBV) beat Uniform by much more (0.20 vs 0.41, ~51%) — most of the benchmark's
  discriminative range is between "informed" and "uninformed," not within uninformed variants.
- **Stage-dependent behavior (Figure 4 finding that quality-vs-coverage gap shrinks with more
  views)** transfers as a hypothesis to test in our own budget sweep (Q6): expect any
  geometry-aware advantage to be largest at small view counts and to compress as budget grows —
  argues for reporting a small-fixed-budget regime specifically because that is where a method
  difference is most visible, not just because it matches our real per-object time constraints.
- **The RRI metric family** ("fractional Chamfer-Distance improvement from one candidate view") is
  a clean instance of the "metric admits a per-candidate analytical optimum" pattern Q7 asks for —
  though it requires a GT mesh/point cloud (see below), so it only transfers as a *design pattern*
  for simulation-based evaluation, not as a metric usable on our real survey clouds without a
  separately-acquired dense GT.
- **CORAL ordinal classification instead of regression** for a noisy, hard-to-regress scalar
  fitness score is a reusable technique if we ever want to train (rather than hand-derive) a
  view-scoring function from noisy simulated labels — not directly a benchmark-methodology point,
  but adjacent.

## What does NOT transfer, and why

- **The entire fitness signal (Eq. 1, Eq. 2) requires a ground-truth point cloud at test time for
  Oracle-NBV and at train time for VIN's labels.** Confirms Q4's exit #1 (simulate from CAD/scanned
  mesh with injected defects) is the only route this paper demonstrates working — it offers zero
  evidence on Q4's other two exits (dense-real-capture-as-GT, or a mesh-free per-region view-angle
  metric), because it never attempts either.
- **No sensor-defect injection whatsoever** (author-acknowledged, Section 5) — directly negative
  evidence for Q9: this benchmark cannot tell us whether method rankings change under pose
  error/dropout/fused-vs-single-view offset, because it was never tested. If we want that evidence,
  we have to build it ourselves; this paper is a template for *how* to structure the simulation
  (fixed candidate pool + GT-based fitness) but supplies none of the defect-robustness result itself.
- **Object scale and platform are wrong for our setting.** Houses scaled to 14 m, drone at 4 mph,
  fully aerial/exterior scanning — the opposite regime from tabletop eye-in-hand D405 inspection at
  centimeter scale. The "resource-efficient" framing (time-in-motion budget, collision clearance
  from a flight path) doesn't map onto a UR5e's much shorter, differently-constrained reachability
  envelope. Use this paper for its *benchmark methodology*, not its numbers or its object set.
- **No per-region / per-surface evaluation unit at all — evaluation is whole-object Chamfer
  Distance only.** Directly answers Q1: VIN-NBV's unit of evaluation is "(object, view-budget)
  scored by whole-reconstruction fidelity," not "(object, surface region) scored by evidence-view
  quality for that region." The rejected option (per-region scoring) is not even considered in this
  paper — it has no notion of a distinguished surface region needing inspection, only "the whole
  object reconstructed well." This is a meaningful gap relative to our RQ: nothing here validates
  or informs a per-region orthogonal-view metric; we'd be extending past what this paper does, not
  adapting it.
- **No symmetry/view-equivalence handling (Q8)** — not applicable to their setup since there's no
  notion of "the one correct view of a labeled surface," only "did the whole point cloud converge."
- **No VLM, no language, no orchestrator-in-the-loop at all** — Q15–Q17 (grounded success vs. lucky
  answer, VLM failure modes, judge controls) have zero applicable material here; this is a pure
  geometric NBV paper with a learned scalar fitness function, not a VLM-driven action-selection
  system.
- **No statistics/variance reporting** (Q12) — single mean-CD-per-method numbers only, no per-object
  spread, no confidence intervals, no significance tests anywhere found — this paper is a negative
  example for benchmark statistical rigor, not a template to copy for Q12.
- **Exact hemispherical-shell geometry (elevation angles, radii, azimuth spacing) is not recoverable
  from the paper text** — only "120 views, 3 shells" is stated. If we want to replicate or compare
  against their exact candidate-view geometry we would need the released code/supplementary
  material (not confirmed accessible from the HTML version fetched here).

## Citation map (for checkability)

| Claim | Section | Figure/Table/Eq |
|---|---|---|
| Problem formulation, `C_base`/`I_base`/`D_base` | §3.1 | — |
| Top-level fitness objective | §3.1 | Eq. 1 |
| RRI definition | §3.2 | Eq. 2 |
| VIN architecture / featurization | §3.3 | Eq. 3, Eq. 4, Fig. 2(b) |
| CORAL ordinal loss, 15 classes, stage-dependent normalization | §3.4 | — |
| Greedy rollout loop | §3 (Algorithm 1) | Alg. 1, Fig. 2(a) |
| Coverage metric (Cov-NBV fitness) | §4 (baselines) | Eq. 5 |
| 120-view / 3-shell candidate set | §4 | — |
| Base-view init (random + nearest) | §4 | — |
| Houses3K training set, OmniObject3D test set | §4 | — |
| House scale (14 m) | §4.2 | — |
| Main results, houses | §4.1 | Table 1 |
| Main results, unseen categories | §4.1 | Table 2 |
| Coverage-quality gap shrinking over stages, Oracle-NBV gap | §4.1 | Fig. 4 |
| Coverage %/F1 vs. step | §4.1 | Fig. 5 |
| Time-limited evaluation (drone speed, 15s result) | §4.2 | Fig. 6 |
| Collision-avoidance constraint | §4.3 | Fig. 7 |
| Generalization to unseen categories | §4.4 | — |
| Limitations (noise-free depth, no real data, oracle gap) | §5 | — |
| Implementation/training details | §7.2 (appendix) | — |
| Teaser figure caption | — | Fig. 1 |

## Open gaps from this pass (not recoverable from HTML fetch alone)

- Exact per-shell elevation/azimuth angles and radii for the 120-view candidate set.
- Exact object counts for Dinosaurs/Trucks/Animals/Motorcycles test splits.
- F1-score distance threshold and coverage-percentage formula (only Eq. 5's empty-pixel count is
  given; F1 itself is undefined in the fetched text).
- Numeric (non-graphical) Cov-NBV and Oracle-NBV Chamfer Distance values (only shown in Figure 4,
  not tabulated).
- Point-cloud cardinality for `R_base`/`R_GT`.
Would need the PDF's figures/appendix tables or the released code repository to close these; worth
a follow-up fetch of the PDF (not just HTML) or the project/code page if this paper becomes
load-bearing for our own benchmark design rather than background reading.
