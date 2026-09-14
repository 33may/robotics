# Efficient View Planning Guided by Previous-Session Reconstruction for Repeated Plant Monitoring

- arXiv: 2510.07028 (v2), HTML full text read at https://arxiv.org/html/2510.07028v2
- Selected for: pose-error injection template — eye-in-hand manipulator, Gaussian
  pose-noise sweep at three magnitudes starting at (0.002 m, 0.5°), which is the
  paper our own defect-injection protocol is modeled on.
- Read via WebFetch (abstract page + three targeted full-text passes: pipeline/
  setup, §IV-D noise robustness + Table VII, candidate-view-space/related-work/
  ablations). No PDF download; relying on the arXiv HTML render, so equation
  numbers/table numbers are as rendered there — treat as high-confidence but not
  hand-verified against the PDF typesetting.

## Core method (not our RQ territory, but needed for context)

Problem: repeated multi-session monitoring of the *same* plant. Instead of
running next-best-view (NBV) from scratch each session, reuse the previous
session's reconstruction as a geometry prior for the current session.

Pipeline (§III):
1. **Non-rigid registration** of the previous-session reconstruction to a
   coarse current-session scan, via embedded deformation graphs (handles plant
   growth/deformation between sessions) — Eq. (2).
2. **Conservative inflation**: add points from the current coarse scan that are
   near current-scan points but far from the aligned previous reconstruction
   (catches new growth the prior doesn't cover) — points added if within
   `γ_near` of the current coarse cloud and beyond `γ_far` of the aligned prior.
3. **One-shot set-cover view selection**: choose the minimum view set such that
   every surface point is covered by at least one selected view — an integer
   program, `minimize Σ x_v s.t. Σ I(p,v) x_v ≥ 1 ∀p`, solved with Gurobi
   (Eq. 6). This replaces iterative NBV with a single-shot combinatorial
   optimization over a *fixed candidate view set*.
4. **Global path planning**: shortest Hamiltonian path connecting the selected
   views (Eq. 1) — a TSP-like tour, not per-view greedy travel.

This is a one-shot-from-prior architecture, structurally different from our
setting (we have no previous session / no growth-tracking prior; our object is
static and scanned once per inspection). Not directly transferable as a
method. What *is* transferable is (a) the visibility/coverage formalism, (b)
the pose-noise injection protocol, (c) the coverage + Chamfer/F-score metric
pairing, and (d) how they structure ablations and baseline comparisons.

## Sensor / robot setup

- **Simulated, not physically deployed.** Uses "a view-planning simulator from
  [8] [Pan et al., MA-SCVP, T-RO 2024], which implements a virtual sensor via
  OctoMap-based ray casting" (§IV-A1). Camera intrinsics are "copied from an
  Intel RealSense D435" at 1280×720 — **not the D405 we use**; D435 has a wider
  min range and different depth characteristics than D405, so any depth-noise
  characteristics implicit in their simulator do not transfer to our D405
  numbers (they don't model D435-specific IR-stereo noise anyway — see below,
  their "sensor" is a ray-casting oracle, not a noisy depth-camera model).
- Greenhouse deployment is only described as an inspiration architecture: "an
  eye-in-hand RGB-D sensor mounted on a manipulator" on "a rail-based base"
  (§V-1), citing Pathobot (Smitt et al., ICRA 2021, ref [6]) as the platform
  archetype. This is eye-in-hand like our UR5e+D405, but on a linear rail
  gantry across crop rows, not a 6-DOF arm around a tabletop object.
- **Explicit, load-bearing negative finding**: "we have not yet validated the
  full end-to-end closed-loop system on live crops in an operating greenhouse"
  (§V-1), attributed to seasonal constraints. All robustness numbers below are
  therefore simulated pose-perturbation studies, not measured real-hardware
  error — this is exactly analogous to what we'd be doing (injecting synthetic
  defects into a simulator/dataset), so it's a same-genre precedent, not an
  empirically-validated hardware number to borrow directly.
- **Direct arm-model mention (§IV-C3), closest concrete match to our own
  platform found in this paper**: "Because the manipulator workspace is
  limited (e.g., a UR5-class arm), we split the scan into upper and lower
  height bands" — this is the design justification for splitting the
  greenhouse-row candidate space into two height bands, not a claim that a
  UR5 was actually deployed. It's an illustrative example, but it's the one
  place the paper explicitly names our exact arm class (UR5), which
  strengthens the "eye-in-hand manipulator" match beyond generic
  eye-in-hand-RGB-D phrasing — treat as circumstantial platform-similarity,
  not evidence the pose-noise numbers were measured on a UR5.

## Pose-noise injection protocol (§IV-D) — the reason this paper was pulled

**§IV-D "Pre-Deployment Pose-Noise Robustness."** Motivation, quoted:
"In practical deployment, our greenhouse data acquisition is performed with an
eye-in-hand RGB-D sensor mounted on a manipulator... We therefore conduct a
pre-deployment pose-noise robustness study to quantify how execution errors
affect reconstruction quality and to evaluate whether lightweight ICP
refinement can recover geometric fidelity."

Stated error sources (this is the "our exact error sources" match the framing
doc flagged): "The dominant sources of error for multi-view fusion are pose
perturbations, including hand-eye calibration errors, joint/actuation
imprecision of the manipulator, and residual vibration or drift introduced by
base motion on the rail." — i.e. calibration + kinematic + platform-vibration
error, all folded into a single Gaussian pose-noise model. No separate
modeling of per-source magnitude; it's one combined SE(3) noise budget.

**Protocol (§IV-D1):**
- Add zero-mean Gaussian translational noise with std `σ_t` (meters) to camera
  position, and Gaussian rotational noise with std `σ_r` (degrees) to camera
  orientation.
- Three noise levels tested: `(σ_t, σ_r) ∈ {(0.002, 0.5), (0.005, 1.0), (0.01,
  2.0)}` — i.e. a linear 1×/2.5×/5× sweep from a base pair of 2 mm / 0.5°.
- Noise is applied **only to active (executed) views**; the passive warm-start
  observations (from the reused previous-session reconstruction) remain
  noise-free. This is a deliberate asymmetry: it isolates degradation caused
  by *this session's* execution error from the prior-session geometry, which
  is treated as already-trusted.
- Reported both **with and without lightweight ICP refinement** applied
  per-view before fusion — this is their proposed mitigation, and it's the
  main independent variable in the results, not the noise level itself.

**Table VII results (Chamfer distance in mm; F-score = P/R/F₁ at τ=0.01m and
τ=0.02m):**

| σ_t (m) | σ_r (°) | ICP | Chamfer (mm) | F@0.01 (P/R/F₁) | F@0.02 (P/R/F₁) |
|---|---|---|---|---|---|
| 0 | 0 | — | 5.44±0.13 | 100.0/87.3/93.2 | 100.0/97.9/98.9 |
| 0.002 | 0.5 | OFF | 7.69±0.15 | 57.8/91.8/70.9 | 93.7/98.6/96.1 |
| 0.005 | 1.0 | OFF | 9.89±0.27 | 40.0/93.1/56.0 | 76.7/99.0/86.4 |
| 0.01 | 2.0 | OFF | 13.87±0.47 | 27.3/92.2/42.1 | 55.3/99.3/71.0 |
| 0.002 | 0.5 | ON | 5.89±0.13 | 99.2/85.7/91.9 | 100.0/97.7/98.8 |
| 0.005 | 1.0 | ON | 6.03±0.18 | 98.9/85.0/91.4 | 100.0/97.6/98.8 |
| 0.01 | 2.0 | ON | 6.08±0.16 | 98.7/85.0/91.4 | 99.7/97.6/98.7 |

**Reading of the numbers:** Without ICP, Chamfer distance degrades roughly
linearly with noise (5.44→13.87 mm, ~2.5×), and *precision* at the tight
0.01 m threshold collapses catastrophically (100%→27.3%) while *recall* stays
essentially flat (~87–93%) — noise adds spurious/misaligned points (hurts
precision) without removing coverage (recall unaffected). With ICP, all three
noise levels converge to nearly the no-noise baseline (Chamfer 5.9–6.1 mm vs.
5.44 mm baseline; F₁@0.01 ~91.4–91.9% vs 93.2% baseline) — the paper's claim
that reconstruction becomes "largely insensitive to the tested noise levels"
after ICP is well supported by this table.

**What this experiment does NOT do (important for Q9 / our benchmark
design):** this is a single-method robustness sweep, not a multi-method
ranking-stability study. Table VII only reports "Ours" (with/without ICP) at
each noise level — there is no baseline (GMC, MA-SCVP, Block-Wise NBV) re-run
under noise. So the paper cannot and does not report whether *ranking between
methods* changes under injected pose noise — it only shows that one method's
*absolute* reconstruction-quality metrics degrade and can be recovered by ICP.
For our Q9 ("does any candidate method's ranking change under injected
defects — ranking changes are the finding"), this paper is a protocol
template (how to parameterize and sweep pose noise) but not a precedent for
the ranking-stability analysis itself — that comparison is not in scope here
and would need to be original to our benchmark.

Also confirmed: **no dropout injection and no fused-vs-single-view offset
experiment anywhere in the paper.** I searched specifically for both (per the
framing doc's ~4 cm offset and depth-dropout defect axes) — the paper's
degradation model is pose noise only. The "dominant sources of error" sentence
lists calibration/actuation/vibration but the paper operationalizes all of
them as a single combined SE(3) Gaussian, not as separately injected dropout
or fusion-offset terms. So the "our exact error sources" match from the task
brief is **partial**: pose error is a first-class, well-specified axis here;
dropout and fused-vs-single-view offset are not — those two defect types have
no precedent in this paper and would need another source.

## Evaluation protocol (relevant to Q4, Q7)

- **Coverage metric**: `|C|/|Ω|` where `Ω` is the union of visible sets over
  the *entire discretized candidate view space* (an oracle "reachable
  coverage ceiling," computed offline once) and `C` is the accumulated
  covered set from the executed views (Eqs. 8–10). This requires no mesh —
  it's computed directly from ray-casting against the reference point cloud
  in the simulator. Visibility indicator: `I(p,v) = 1` if point `p` is in the
  visible set of view `v` (occlusion-only via OctoMap ray casting), `0`
  otherwise — **no incidence-angle / surface-normal condition** is used to
  decide visibility. This is a meaningfully different (weaker) notion of
  "covered" than an orthogonality-based evidence-view criterion: a grazing
  view of a point counts as full coverage here, which would not satisfy our
  "evidence view must be orthogonal to the inspected surface" requirement.
  Relevant to Q7 — this metric family answers "was the point seen at all," not
  "was the point seen well/orthogonally," so it is not itself a candidate for
  our orthogonality-score metric, only an example of a no-mesh-needed,
  ray-casting-based coverage metric (relevant to Q4's "per-region metric that
  needs no mesh" exit).
- **Reconstruction-fidelity metrics** (used in the noise-robustness study):
  symmetric Chamfer distance, and F-score (precision/recall/F₁) at two
  distance thresholds τ ∈ {0.01, 0.02} m, computed against a ground-truth
  surface point cloud independent of the oracle-visible set.
- **Acquisition-cost metrics**: number of executed views, and accumulated
  Euclidean path length connecting consecutive executed views (their global
  Hamiltonian-path objective, evaluated post hoc).

## Ground truth construction (relevant to Q2, Q4)

Two different GT strategies for two different dataset types — worth noting as
a template for stratified GT provisioning:
- **Public individual-plant dataset [11]** (maize + tomato, 3 instances each,
  last two consecutive sessions used per instance): GT presumably comes from
  the dataset's own reference reconstruction (full citation for [11] could not
  be resolved from the HTML render — cite string was truncated in fetch;
  flagging as unresolved, would need the PDF references list to pin down).
- **New greenhouse crop-row dataset** (tomato + cherry tomato, 5 sessions over
  2 weeks, 4 consecutive session-pairs → 8 test cases): GT acquired via **an
  independent, higher-fidelity capture device** — "Polycam on iPhone/iPad Pro
  equipped with LiDAR," producing RGB + per-frame depth + estimated poses,
  fused via "Nerfstudio depth-nerfacto pipeline" into a reference point cloud.
  This is a clean example of the "third-party/independent-instrument" GT
  strategy — GT provenance is a *different device and different reconstruction
  pipeline* than the one being evaluated, which is directly relevant to our
  Q3 (avoiding circularity: separate GT provenance from the system under
  test) and Q21 (metrology-style independent-instrument GT as an alternative
  to analytical/CAD GT). It is not an artefact-standard (VDI/VDE 2634-style)
  approach, but it is the same *spirit*: GT from a separately-calibrated,
  higher-precision sensing pipeline, not from the system under test re-run
  more times.

## Candidate view space (relevant to Q5, Q8)

Not an angular {h,v} grid — **discrete point-sampled** candidate sets:
- Individual plants: 32 candidates on a hemisphere (or 63 on a full sphere) of
  radius 0.4 m centered on the plant, positions distributed via Tammes-problem
  packing (near-uniform point spacing on the sphere, not a regular az/el
  grid). This is a *denser, more uniform* candidate set than a coarse angular
  grid — relevant precedent for Q5's "must a denser uniform sphere at matched
  budget also be reported" question: this paper's own uniform-sphere
  candidate generation already uses geodesic/Tammes packing rather than naive
  {h,v} steps, suggesting naive angular grids are treated as the weaker
  baseline in this literature, not the state of the art for "uniform."
- Greenhouse rows: 1,000 candidates via Poisson-disk sampling inside an
  oriented bounding box (OBB) around the row segment — reflects the
  elongated/planar row geometry, not a sphere at all.
- **Table I is the one directly comparable "our baseline vs their method"
  result**: on the *sphere* view space, their method achieves 98.44±0.68%
  coverage with 9.22±1.66 views vs. MA-SCVP's 92.92±4.29% coverage with
  12.72±1.09 views — i.e., the prior-guided one-shot method needs *fewer*
  views and gets *higher* coverage than a strong one-shot-with-warm-up
  baseline, but only when a previous-session prior exists (their whole
  premise). On the hemisphere, the gap nearly disappears (98.77% vs 98.80%
  coverage, comparable view counts) — the method's advantage is
  session-repetition-specific, not a general view-selection improvement. This
  does **not** transfer as evidence that geometry-conditioned view selection
  beats uniform baselines in our single-session setting.

## Baselines (relevant to Q5)

- **GMC** [43]: iterative NBV, next view maximizes an information-gain
  objective (classic greedy iterative baseline).
- **MA-SCVP** [8] (Pan et al., T-RO 2024): "combined pipeline that triggers
  set-covering stage after a short NBV warm-up, using a learned shape prior" —
  their strongest baseline, and the one this paper is a direct architectural
  descendant of (same simulator, same set-cover formalism, difference is the
  previous-session prior replacing the learned shape prior).
- **Block-Wise NBV** (greenhouse only): reimplementation of Zaenker et al.'s
  information-gain NBV, partitioned into 10 spatial blocks (5 upper, 5 lower)
  — a domain-adapted iterative baseline, not a generic sphere baseline.
- **PriorOBBRandom / PriorPassiveRandom / PriorTransferRandom** (greenhouse
  only): one-shot look-at sampling with random selection under three
  different priors (OBB-only, passive-observation prior, transfer prior) —
  random-selection baselines repeated 5× each. This ablation ladder (Table V)
  isolates how much of the coverage ceiling comes from the *candidate
  generation prior* vs. the *selection algorithm* — a useful ablation
  structure for our own baseline design (separate "how good is the candidate
  set" from "how good is the selector").

No baseline in this paper is a plain uniform-angular-grid sphere like our
current 12×30°×3-vertical design; their "uniform" candidate sets are
Tammes/Poisson-disk point distributions, which are a stronger uniform
baseline than a naive az/el grid. If we cite this paper as precedent for
denser uniform-sphere baselines (Q5), note the discretization scheme differs
from ours and would need restating in az/el terms for a fair comparison.

## Statistics (relevant to Q12)

All results mean±std across test cases (36 for individual plants — 2 plant
types × 3 instances × 2 rotations × 3 initial views; 8 for greenhouse — 2 crop
types × 4 session-pairs). Random baselines repeated 5× and presumably
averaged into the same mean±std reporting. **No significance testing (no
t-tests, no confidence intervals) anywhere in the paper** — this is a
negative-precedent data point for Q12: even a paper with 36 test cases (more
than we're likely to have) does not do formal statistical testing, just
descriptive mean±std. Not a methodological endorsement, just documents field
practice — we should not treat "36 cases, mean±std, no significance test" as
sufficient rigor to copy uncritically.

## Ablations (structure worth reusing)

- **Table III** (inflation step on/off): coverage improves with inflation
  (maize 96.99%→98.06%, tomato 98.98%→99.15%) — small effect size, ablation
  isolates one pipeline component.
- **Table IV** (NBV warm-up on/off): registration Chamfer distance
  2.764±1.195 mm (initial view only) → 1.743±0.444 mm (initial + one NBV
  step) — shows a single extra warm-up view meaningfully improves downstream
  registration quality; relevant if we ever consider a warm-up/coarse-scan
  step before geometry-conditioned view selection.
- **Table V** (prior ladder: OBB-only / passive / transfer): reachable
  coverage ceiling 94.48% / 97.21% / 97.34% — passive and transfer priors
  give nearly identical ceilings, meaning most of the benefit comes from
  *any* observation-grounded prior over a bare bounding box, with diminishing
  returns from a more sophisticated prior. Relevant analogy for us: primitive
  fit vs. full survey-cloud-conditioned candidate generation may show a
  similar "most of the gain comes from leaving pure-bbox/pure-uniform" pattern
  — worth designing our ablation ladder analogously (uniform sphere → OBB/
  primitive-conditioned → full survey-cloud-conditioned) to see where returns
  diminish.

## What transfers to our setting

1. **Pose-noise injection protocol structure** — Gaussian SE(3) noise on
   camera pose, swept at multiple magnitudes, reported with/without a
   mitigation (ICP here), evaluated via Chamfer/F-score against independent
   GT. Directly reusable as a template for our pose-error axis. The specific
   magnitudes (2/5/10 mm, 0.5/1/2°) are a reasonable starting sweep to anchor
   to, though sourced from a different platform (rail-mounted gantry, not a
   6-DOF arm) so should be treated as a starting point, not validated for our
   UR5e+D405 kinematic error budget.
2. **Independent-instrument GT strategy** (Polycam+LiDAR / Nerfstudio pipeline
   distinct from the evaluated system) — a concrete precedent for Q3/Q21's
   "different provenance" GT separation.
3. **Ablation-ladder structure** (candidate-generation prior quality ×
   selection-algorithm quality, tested independently) — reusable pattern for
   isolating what our geometry-conditioning actually buys us.
4. **Coverage metric as ray-cast visible-set fraction against a reference
   cloud** — mesh-free, reusable for Q4, though it lacks any
   orthogonality/incidence condition so needs augmentation for our "evidence
   view must be orthogonal" requirement (Q7).

## What does NOT transfer, and why

1. **The core method (previous-session prior reuse)** — solves a repeated-
   session growth-tracking problem we don't have; our object is scanned once
   per inspection with no session history.
2. **No dropout or fused-vs-single-view-offset injection precedent** — despite
   being flagged as a close match, this paper only injects pose noise, not
   depth dropout or the fusion-offset defect. Framing doc's "our exact error
   sources" claim holds only for the pose-error axis, not the other two
   defect types in our constraints list (§88-91 of the framing doc).
3. **No ranking-stability analysis under noise** — Table VII is single-method
   robustness only; cannot be cited as precedent for "does ranking between
   methods change under defects" (Q9) — that analysis has to be original to
   us.
4. **Sensor is D435, not D405**, and is a ray-casting oracle (no depth-sensor
   noise model at all, e.g. no IR-stereo dropout, no material-dependent
   noise) — the "sensor" in their simulator is geometrically ideal except for
   the injected pose noise; realistic D405-specific depth defects (glossy/
   dark-surface dropout, the kind in our framing doc) are not modeled here at
   all.
5. **No live-hardware validation** — all numbers, including the noise study,
   are simulated; treat as a protocol precedent, not as empirical evidence
   about real eye-in-hand pose error magnitudes.
6. **Discretization scheme differs** — Tammes/Poisson-disk point sampling, not
   an angular {h,v} grid — if borrowing their "uniform sphere" as our
   stronger baseline (Q5), the parameterization needs translating.

## Open / unresolved

- Could not resolve the full citation for reference [11] (public multi-session
  individual-plant dataset) from the HTML fetch — the citation text was
  truncated in the fetched content. Would need to pull the references list
  from the PDF or arXiv listing directly if we need to cite/reuse that
  dataset.
- Did not verify equation numbering/table numbering against the arXiv PDF
  typesetting (HTML render only) — table/section labels (e.g. "Table VII",
  "§IV-D1") should be treated as very likely correct but not hand-checked
  against the PDF.
