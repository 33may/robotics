# ObjView-Bench: Rethinking Difficulty and Deployment for Object-Centric View Planning

- **arXiv**: 2605.10707v1 [cs.RO], submitted 11 May 2026
- **Authors**: Sicong Pan, Xuying Huang, Benno Wingender, Maren Bennewitz (University of Bonn); Hao Hu (independent)
- **Funding**: DFG AID4Crops, PhenoRob (EXC-2070), BMFTR Robotics Institute Germany
- **Source used**: full HTML (`arxiv.org/html/2605.10707`), fetched and parsed directly (not abstract-only). All section/table numbers below are as in the paper.
- **Code/data**: not yet released — "we will release the object pool, benchmark code, and baseline implementations with the final version" (§1). No URL, no license. Future: "a public evaluation server and leaderboard" (§8).

## Core idea (§1, §3)

The paper's central move is disentangling three quantities that prior view-planning evaluations conflate into a single "surface coverage" number:

1. **Omnidirectional self-occlusion ratio** `r_vis^(r)` (§3.3, Eq. 2) — object-side attribute: fraction of the discretized GT surface (voxelized at resolution `r`) that is *externally observable at all*, estimated as the saturated coverage `Y_sat^(r)` of a dense Tammes view sphere divided by total surface `|S_gt^(r)|`. This is a ceiling, not a planning outcome — low raw coverage on a self-occluded object doesn't mean bad planning.
2. **Observation saturation difficulty** `d_sat^(r) = N*^(r)` (§3.3) — the first view-set size (on a Tammes sphere) at which the marginal-gain curve `Y^(r)(N)` (Eq. 1) enters a "stable low-gain regime." Captures objects whose observable surface is revealed only by *dense* sampling (cavities, apertures, inter-part occlusion) even though it's not occluded in principle.
3. **Planning difficulty** `d_plan^(r)` (§3.4, Eq. 3) — defined as a **Set Cover ILP** over a *fixed candidate-view protocol* `V_plan`: minimum number of views from `V_plan` needed to cover the protocol-reachable target universe `U^(r) = ⋃ O(v_i; r)`. This is explicitly protocol-dependent, not an object-intrinsic quantity — the same object gets a different `d_plan` under a different candidate-view set.

Visibility operator `O(v; r)` = ray-based first-hit visibility from camera pose `v`, no noise model (§3.2).

This separation is the paper's actual contribution: it lets them argue that a low coverage score can come from (a) truly unobservable surface, (b) surface that needs many views to reveal, or (c) a genuine planner failure — and these require different fixes.

## Dataset (§4)

- Source: Objaverse++ (quality-filtered Objaverse), NOT primitive shapes. Pipeline: quality filter → "conservative geometry filtering" + stratify by "lightweight shape and fill statistics" → run the §3 difficulty pipeline → manual reliability review.
- Final pool: **5,691 objects**. Split into **main pool (5,523)** + **168 slow-saturation objects** (objects whose `d_sat` exceeds the 128-view threshold chosen for the candidate protocol).
- Main pool → **Main-Train-Public (4,995)** + **Main-Hidden-Test (528)**, sampled with a **two-stage difficulty-aware rule**: first cover every non-empty `d_plan` bucket, then allocate remaining quota ~proportional to bucket frequency (avoids a test set dominated by easy objects while preserving the natural long-tailed distribution).
- Canonical resolution `r = 0.02` (normalized object-scale units) used throughout for difficulty annotation and for the main reported metrics (NSC@0.02, SC@0.02).
- Candidate view protocol: Tammes-128 sphere (chosen because 128 views is enough for the observable surface of non-slow-saturation objects to saturate, per §3.5 — "conservative coverage criterion" using the max `d_sat` over non-slow-saturation objects). Classical/hybrid planners additionally get evaluated on a denser **Tammes-360** set since they aren't tied to a fixed-size action classifier.
- **No injected sensor defects, no depth noise, no calibration error** — renders are idealized RGB-D. Explicitly flagged as future work in §8 ("incorporating depth noise and calibration error... beyond the current idealized rendered RGB-D setting").

## Evaluation protocol (§5)

- **Online, hidden-geometry** setup: a benchmark server holds the GT mesh; the method never sees geometry, only sequential rendered RGB-D from the poses it requests. Server checks feasibility, renders, fuses into an accumulated point cloud, logs per-step metrics. Evaluation is done on **raw observation point clouds** (decoupled from any particular mapping/fusion representation).
- **Metrics** (§5.1): NSC (Normalized Surface Coverage), PC (Path Cost — accumulated travel distance, collision-avoiding shell distance when straight-line crosses the unit object), #Views, PR (Planning Runtime, coarse-binned because wall-clock is system-load-dependent).
  - `SC@τ`: fraction of GT surface point cloud covered by accumulated observations, τ = coverage distance threshold.
  - `NSC@τ`: SC@τ normalized by a **view-space-specific observable-surface reference** computed from a dense 360-view Tammes set *after applying the reachability constraint*. **No explicit closed-form equation is given for either** — both are defined only in prose, despite being the headline metrics. (Relevant to our Branch-1 metric-formula scope: this is a paper to check for a formula and it doesn't have one usable as-is.)
- **Budget regimes** (§5.2) — three, directly comparable to our Q6:
  1. Small fixed budget **K=5**
  2. Large fixed budget **K=30**
  3. **Automatic stopping**, capped at 128 views. Methods with a native finite-plan stop on their own criterion; iterative methods without one use a benchmark-side **Map-Stabilization criterion MS@δ**: voxelize accumulated cloud at resolution δ, stop when relative point-count gain < 1% for 3 consecutive steps.
- **Reachability** (§5.3) — two view-space settings, **whole** vs **quarter** (one quarter of the full sphere). This is a coarse *geometric* proxy for reachability, not live per-object IK. The only actual IK/collision number in the paper is a single validation statistic (§6.3): a real tabletop rig check found **only 45.4/128 candidate views reachable on average (range 26–74/128) across 5 objects × 5 placements** — used to *motivate* the quarter-sphere abstraction, not as a live constraint inside the benchmark loop. No robot model, arm series, or camera is named; this is the only real-hardware touchpoint in the whole paper, and it's a feasibility-count sanity check, not an evaluation.
- Feasibility handling is method-family-specific (Table 1): score-based methods take best feasible candidate; set-prediction methods filter infeasible views pre-execution; classical/hybrid optimize directly over the feasible subset. NBV-Net is the outlier: single discrete-index output, so an infeasible prediction **terminates the episode** (no fallback).

## Baselines (Table 1, §6.1) — 9 methods, 3 families

| Method | Family | Execution mode |
|---|---|---|
| Random+TSP | Classical | Budgeted plan (feasible views, TSP-ordered) |
| RSE (Delmerico et al. 2018) | Classical | Iterative, best feasible view |
| RSE-Mov (Delmerico et al. 2018) | Classical | Iterative |
| NBV-Net (Mendoza et al. 2020) | Learned | Iterative, single-index output |
| BENBV-Net (Li et al. 2024) | Learned | Iterative, boundary-point representation |
| MA-SCVP (Pan et al. 2024) | Learned | Set-cover plan |
| PoinTr-C+NBV (Dhami et al. 2023) | Hybrid | Iterative (shape completion + NBV) |
| PoinTr-C+SCP (Pan et al. 2025) | Hybrid | Set-cover plan |
| PoinTr-C+MCP (Pan and Wei 2023) | Hybrid | Budgeted plan |

Note: **no uniform-sphere-sweep policy is tested as a competing baseline.** The Tammes-128/360 sphere is used throughout only as the *candidate action set* that every method selects from — not as a naive "visit all points on a fixed sphere" policy in the results tables. Random+TSP is the closest analog to a "random-over-candidates" baseline, not a deterministic uniform-sweep one. This is directly actionable for our Q5: a denser uniform-sphere-sweep-as-policy is a gap even in this paper.

Learned methods trained only on Main-Train-Public, evaluated on Main-Hidden-Test, trained under **whole** view space only (so quarter-space eval is a genuine train/test reachability shift, not something they were adapted for).

## Concrete results

**Table 2 (quality top-3 by NSC@0.02, whole view space)** — budget changes the winner:
- K=5: RSE (classical) NSC 0.980, 6.0 views; NBV-Net 0.978; PoinTr-C+MCP 0.977
- K=30: PoinTr-C+MCP (hybrid) NSC 0.998, 31.0 views, path cost 51.35; Random+TSP 0.996 at <0.1s runtime; PoinTr-C+NBV 0.996 but path cost 99.11, runtime >5s
- Automatic: PoinTr-C+SCP (hybrid) NSC 0.993 with only 15.5 views; RSE 0.991 at 9.6 views; MA-SCVP 0.990 but needs 49.5 views — "the canonical 128-view protocol and higher planning difficulty make learned set prediction more challenging" (§6.2)

**Table 3 (path-efficient top-3, NSC≥0.95 filter)**: BENBV-Net wins path-efficiency at K=5 and automatic despite not winning quality — ranking by quality vs. by efficiency-at-quality-floor produces different winners at the same budget.

**Table 4 (reachability degradation, whole vs quarter, each method's max-view setting)**: ΔSC ≈ 0.13–0.17 for most robust methods (Random+TSP ΔSC=0.137, RSE=0.128, PoinTr-C+MCP=0.131). **NBV-Net collapses**: NSC 0.988→0.472, executed views 9.9→1.2 (single-index action interface + infeasible-prediction termination with no fallback). BENBV-Net and MA-SCVP degrade less catastrophically but still worse than classical/hybrid (ΔNSC 0.016 and 0.023 vs. ≤0.003 for classical). **Finding: reachability constraints hurt learned planners' action representations specifically, not just coverage generically.**

**Table 5 (slow-saturation diagnostic, oracle greedy rollout)**:
- Main-Hidden (N=528): d_sat=32.2, d_plan=18.5, MS stops at 9.0 views, MS-NSC=0.997, Tammes-128-NSC=1.000, gap Δ=0.0026
- Slow-Saturation (N=168): d_sat=161.7, d_plan=45.9, MS stops at 12.7 views, MS-NSC=0.988, Tammes-128-NSC=0.996, gap Δ=0.0081 (**~3× larger residual gap**)
- Interpretation: "map stabilization is a practical stopping signal, but it measures observed-map stabilization rather than long-horizon observable-surface saturation" (§6.4).

**§6.5 case study**: Main-Balanced-Train (oversampled medium/high-d_plan objects) vs Main-Raw-Train (natural long-tailed), both evaluated on held-out Main-Analysis-Test with MA-SCVP. Result reported only graphically (Fig. 3b) — "improves ... achieved voxel coverage across the planning-difficulty axis," **no exact before/after numbers given in text**.

## Statistical treatment — explicit gap

No confidence intervals, no variance/std, no significance testing, no stated number of seeds or trials anywhere in the paper. All numbers are point estimates aggregated over hundreds-to-thousands of objects (528–5,691). They substitute *object-count scale* for *trial replication* — viable for them because their object pool is large; **not a transferable recipe for our few-objects/few-trials setting** (Q12) — we can't buy precision by scale the way they do.

## What transfers to our setting

- **Budget-regime template (Q6)**: small-fixed / large-fixed / auto-with-cap-and-stabilization-criterion is a validated 3-way split that demonstrably flips rankings (their headline empirical result). Directly reusable structure for our own budget-regime decision, including the MS@δ (relative-gain-below-threshold-for-N-steps) stopping rule as a concrete, implementable auto-stop criterion if the VLM's own stopping isn't what we want to measure.
- **Difficulty-bucketed test-set sampling (Q11/Q12)**: two-stage rule (cover every difficulty bucket first, then allocate remaining quota proportional to natural frequency) is a clean, reusable recipe for building a stratified object set from a small pool, independent of their primitive-vs-freeform distinction.
- **Reachability-as-ranking-changer, methodologically (Q9 analog)**: even though they inject reachability/budget constraints rather than sensor defects, the paper is a strong existence proof that *"does the ranking change under constraint X" is the right question to ask and report* — and that some method classes (here: single-index learned action interfaces) fail catastrophically rather than gracefully under constraint. This is direct precedent for our Q9 (defect injection should look for rank flips, not just score drops) and specifically flags "single-choice/no-fallback action interfaces" as a fragile pattern to watch for in our own VLM action space design.
- **Held-out hidden-geometry server design (Q3/Q20)**: GT mesh lives server-side, method only sees rendered observations — a clean judge-free, contamination-free evaluation harness. Reinforces that "GT from geometry, evaluation decoupled from any judgment model" (their whole design) is a working precedent for our Q3 separation between GT-from-geometry and judge-from-image.
- **"Lessons for future work" as a first-class negative-results section (Q13)**: §7 explicitly reports catastrophic failure modes (NBV-Net under quarter-space), scaling limits (learned set-prediction struggling at 128-view/high-d_plan regime vs. "many prior object-centric settings"), and an unresolved gap (slow-saturation objects) as *findings*, not swept under a rug. Good structural precedent for how to write up our own negative results.
- **Oracle-greedy-rollout mechanic (Q14 partial)**: their diagnostic oracle ("select the candidate view with the largest additional GT coverage at each step," §6.4) is a directly reusable mechanism for constructing an oracle-selector condition, though they use it only as a diagnostic ceiling for one difficulty subset, not as a controlled ablation isolating action-space quality from planner-policy quality across all methods (so it's a partial, not full, precedent for Q14).

## What does NOT transfer, and why

- **No primitive shapes at all (Q2, Q11)**: objects are general Objaverse meshes; difficulty is defined by *geometric self-occlusion/saturation/set-cover statistics*, never by primitive-fit quality. There is no "face normal through centroid" GT-view concept here, and no discussion of how a fitted-primitive GT normal degrades against a defective/survey cloud. This paper simply doesn't have the primitive-decomposition problem our setting has — it sidesteps Q2 entirely rather than answering it.
- **No orthogonality/view-angle metric at all (Q7)**: all metrics are surface-coverage/reconstruction-completeness (NSC, SC, PC). There is no notion of "is this view orthogonal to the surface it's evidencing" — their views only matter through what they reveal in aggregate, not through per-view geometric quality relative to a target region. Zero transfer for the "orthogonal evidence view" scoring family.
- **No symmetry / view-equivalence treatment at all (Q8)**: not discussed anywhere; confirmed absent in a full-text pass, not just an omission from search.
- **No sensor-defect injection (Q9, Q4)**: idealized rendered RGB-D only, explicitly deferred to future work. Cannot be used as evidence for "does ranking change under injected pose error/dropout/fused-vs-single-view offset" — that experiment doesn't exist in this paper despite being structurally similar to what they *did* do for reachability/budget.
- **No fused-vs-single-view offset discussion (Q10)**: fusion is mentioned only as "accumulate rendered point clouds," no register/offset error modeled — not applicable.
- **No VLM/LLM anywhere (Q3, Q15, Q16, Q17)**: purely classical/learned/hybrid geometric planners. No judge model, no VQA-style episode, no spatial-reasoning failure-mode taxonomy. None of the VLM-specific branch questions get evidence from this paper — it's a pure geometry/coverage benchmark, adjacent to but categorically different from the "does the VLM correctly interpret the view" layer our branch also needs.
- **Real-hardware validation is a single feasibility statistic, not an evaluation**: the 45.4/128-reachable number is a sanity check for choosing the quarter-sphere abstraction, not a table of real-robot method performance. No robot/camera hardware is named. Don't over-read this as "field-validated on a real UR-style arm" — it isn't.
- **Unit of evaluation is whole-episode aggregate coverage, not per-region (Q1 analog)**: their scoring is "how much of the whole object surface did the accumulated run cover," which is neither our (object, surface-region) unit nor an (object, question) episode — it's a third option: (object, run) scored by cumulative coverage. Useful as a documented alternative to reject-and-state-why for Q1, since it doesn't decompose by region or by question at all.

## Relevance summary

Most directly useful as a **methodology precedent**, not a metric or GT source: validates that budget regime and reachability constraints change method rankings (our branch's selection rationale for citing this paper), gives a reusable stratified-sampling recipe and a reusable auto-stop criterion, and is a clean judge-free hidden-GT harness design. It has **zero primitive-shape framing, zero orthogonality metric, zero sensor-defect injection, zero symmetry handling, and zero VLM component** — so for Q2, Q3 (partial), Q7, Q8, Q9 (empirical evidence), Q10, and the VLM-specific questions (Q15–Q17) it supplies context/precedent at best, not answers.
