# Explore until Confident: Efficient Exploration for Embodied Question Answering

**arXiv:2403.15941v3** (Ren, Clark, Dixit, Itkina, Majumdar, Sadigh — Princeton/Stanford; RSS 2024).
Full text: https://arxiv.org/html/2403.15941v3 · Project page: https://explore-eqa.github.io/

## 0. One-line identity

An **Embodied Question Answering (EQA)** paper: a mobile robot must explore an unseen 3D scene (house/office, 100–800 m²) and answer a natural-language question (identification/counting/existence/state/location) about it, deciding *when it has seen enough to answer*. It supplies exactly one thing our branch asked for by name: a **conformal-prediction-calibrated stopping rule** built on top of a set-of-marks (letter-annotated) visual-prompting frontier selector. It is not a viewpoint-relation, ego/allo, or camera-pose-encoding paper — it has nothing to say about most of the branch's other questions.

## 1. Core method (two separable halves)

### 1a. Semantic-value-weighted frontier exploration (Section IV)

- Builds a voxel occupancy map via TSDF fusion from depth images at each visited pose (Sec. IV-B): "Each voxel corresponds to a cube with side length l... we apply Volumetric Truncated Signed Distance Function (TSDF) Fusion to update (1) occupancy... (2) if they are explored/seen." Projected to a 2D free/explored map (free if all voxels up to 1.5 m height are free; explored if all voxels up to H are marked explored).
- At each step, **farthest-point-sampling** picks `|P|=3` candidate directions in the current free space: "we find [3] sufficient to cover the possible distinct regions in an image" (Sec. IV-C). These are de-projected onto the current RGB image and **annotated with letters** `Y_P = {'A','B','C'}` — i.e. set-of-marks visual prompting, one letter per candidate direction, always exactly 3 marks.
- Two VLM queries per step, extracting **token probabilities, not free-text answers**:
  - **LSV** (Local Semantic Value): "where should I look to answer the question?" → probability mass on each letter token (Sec. IV-C).
  - **GSV** (Global Semantic Value): a yes/no query per direction, "is it worth exploring this direction?" → `GSV_p = f̂_'Yes'(...)` (Eq. 2).
  - Combined via temperature-scaled softmax: `SV_p = exp(τ_LSV·LSV_p + τ_GSV·GSV_p)` (Eq. 3), τ∈{1,3,5} tested, τ=1,1 used for frontier weighting (Appendix B, Eq. A6: `w_p = exp(τ_SV·SV_p + τ_SV,Normal·SV_p,normal)`, τ=11).
- Frontier sampling then draws the next navigation goal weighted by `w_p`, i.e. VLM semantic judgment biases *which frontier to walk to*, not which discrete "view" to select from a fixed menu.
- **Exact prompt text for LSV/GSV is not given anywhere in the paper (main text or appendix)** — only the letter-annotation mechanism and the probability-extraction math are specified. This is a real gap if anyone tries to reproduce the exact interface.

### 1b. Conformal-prediction stopping rule (Section V) — the load-bearing piece for this branch

- The VLM also predicts a probability distribution `f̂_y(x^t)` over candidate answers `y ∈ Y` at each step (for HM-EQA, answers are multiple-choice-style).
- A separate **relevance** query scores whether the current view is relevant to the question at all: `Rel(x^t) = f̂_'Yes'(I_c^t, (q, s_Rel,q)) ∈ [0,1]` (Eq. 5). This is the mechanism that prevents a confidently-wrong answer from an irrelevant view being trusted: confidence is *gated* by relevance, not just the answer probability.
- Per-step, per-candidate-answer confidence: `ρ_y(x) := Rel(x)(f̂_y(x) - 1)` (Eq. 7) — note this is deliberately non-standard (uses `f̂_y - 1`, i.e. a negative-ish quantity so the ordering/thresholding works out for the prediction-set construction; the paper's own Eq. 7/9/10 chain is the actual mechanism, reproduce from the paper if implementing).
- **Episode-level confidence** takes the *minimum* over all timesteps seen so far: `ρ̄_y(x̄) := min_{t∈[T]} ρ_y^t(x^t)` (Eq. 9) — a monotonic, "worst-case-so-far" running confidence, not a running average.
- **Conformal calibration**: non-conformity score `κ_i := 1 - ρ̄_{y_i}(x̄_i)` computed over a calibration set. Threshold `q̂` = the `⌈(N+1)(1-ε)⌉ / N` empirical quantile of `κ_1,...,κ_N` (Sec. V-A) — this is the standard split-conformal-prediction recipe, applied to embodied confidence rather than classification.
- **Prediction set at test time**: `C^t(x_test^t) := {y ∈ Y | ρ_y^t(x_test^t) ≥ 1 - q̂}` (Eq. 10). The robot intersects `C^t` across timesteps (monotonically shrinking, since it's an AND over time) and **halts when the intersection contains exactly one element** — a strict singleton rule, no size-≥2-but-close-enough fallback described.
- Calibration/test split: **300 calibration scenarios / 200 test scenarios** out of the 500-question HM-EQA set (Sec. VII-F). ε (the conformal miscoverage target) is swept from **0.27 to 0.99** in the efficiency-vs-confidence experiment (Sec. VII-G).

## 2. Concrete numbers

| Quantity | Value | Source |
|---|---|---|
| HM-EQA dataset | 500 questions, 267 HM3D scenes, scene size 100–800 m² | Sec. VI |
| Question categories | Identification 16.2%, Counting 16.0%, Existence 21.6%, State 20.2%, Location 26.0% | Sec. VI |
| Max timestep budget | `T_ε = √(scene_size) × 3` | Sec. VI |
| Candidate frontiers/marks per step | always 3 (fixed, via FPS) | Sec. IV-C |
| Main-VLM success @ full budget | 58.4% (Prismatic VLM, open-weight, needed for logprobs) | Sec. VII-C |
| FBE baseline (unweighted frontier exploration) | ~52% @ full budget | Sec. VII-C, Fig. 7 |
| Advantage over FBE | modest at full budget, **larger at 20–30% of budget** (early-exploration efficiency gain is the headline, not final accuracy) | Sec. VII-C |
| Ablation: No-LSV (GSV only) | ~50%, ≈FBE level | Fig. 7b |
| Ablation: No-GSV (LSV only) | higher early, plateaus lower — "planner being myopic" | Fig. 7b, Sec. VII-C |
| Temperature τ sweep | τ=3 best early gain; τ=5 spikes then degrades (over-prioritizes) | Fig. 7c |
| View-orientation ablation | 58.4% → 59.3% (+0.9 pt) when orientation is added to the semantic value | Appendix C |
| Calibration / test split | 300 / 200 scenarios | Sec. VII-F |
| ε sweep (conformal) | 0.27–0.99 | Sec. VII-G |
| Stopping efficiency @ 58% success | ~71% of max timesteps used, vs. 85% for the Relevance-only baseline | Sec. VII-G, Fig. 9 |
| Hardware trial | Fetch robot + iPhone-12 RGB, 6 scenarios (3 home-office scenes × 2 Q/A) | Sec. VII-H |
| Hardware success | 4/6 correct, matching Relevance baseline; stops earlier in 2 of the 4 | Sec. VII-H, Fig. 10 |
| Finetuned Prismatic (300 scenarios) | 68.1% (vs. 56.2% zero-shot) | Appendix C |
| GPT4-V (official access) | 73.9% success | Appendix C |
| GPT4-V (unofficial) | 58.4% | Appendix C |
| LLaVA-1.6 | 65.3% | Appendix C |

**No error bars, seeds, or repeated trials anywhere** — single-run point estimates throughout, including the hardware 6-scenario result. Flag per branch's evidence-discipline requirement.

## 3. Load-bearing assumptions (why the numbers hold, and where they'd break)

- **Requires token-probability access from the VLM.** The main experiments deliberately use the open-weight Prismatic VLM instead of GPT-4V because "our setup requires probability outputs from the VLM" (Sec. VII-A) — LSV, GSV, Rel, and the answer-confidence `f̂_y` are all built from softmax/logit extraction over specific tokens (letters, "Yes"/"No", answer choices), not from the model's free-text output. GPT-4V numbers exist only as a side experiment with degraded/unofficial access to logprobs.
- **Requires depth + accurate odometry + a collision-free planner** for the TSDF map and frontier sampling (Sec. III, IV-B) — this is a navigation-stack dependency orthogonal to the VLM-interface question.
- **Fixed cardinality of 3 marks per step**, chosen because 3 "sufficient to cover the possible distinct regions in an image" for *this* domain (indoor room frontiers). Never tested at larger menu sizes — **no evidence here about scaling the mark count**, which several other branch questions (option-count degradation, order sensitivity) directly need and this paper cannot supply.
- **View orientation is explicitly not modeled** in the main semantic-value computation: "Our current setup does not consider the view orientation when considering the semantic value, which can be important" (their own limitations statement). The ablation that does add it yields a **+0.9-point** gain (58.4%→59.3%, Appendix C) — small, but this is measured on room-scale "which direction is the object in" navigation, not on centimeter-scale "is the camera orthogonal to this surface" — the two tasks likely have very different orientation sensitivity, so this near-null result should **not** be read as "orientation doesn't matter," only as "orientation didn't matter much for coarse room navigation toward an object."
- **The calibration set must be representative of the deployment distribution** (standard conformal-prediction requirement, not spelled out as a limitation but structurally true): 300 HM-EQA scenarios calibrate the threshold `q̂`; a deployment on a materially different distribution (e.g., our tabletop defect-inspection setting) would need its own calibration set of comparable size with ground-truth labels, which we may not have at that scale for real defects.
- **The paper does not discuss failure modes of the visual-marking scheme itself** — no reported cases of the VLM misreading letters A/B/C, marks landing on ambiguous/non-navigable points, or marks overlapping. This is a **gap, not a reassurance**: it cannot be cited either for or against mark/OCR-dependence robustness in the branch's failure-mode catalog.
- **"Confident" means "answer-probability high AND view judged relevant," not "surface fully seen."** The conformal target label space `Y` is the multiple-choice answer set for a question, not a coverage/completeness variable. Porting this machinery to inspection requires redefining what `Y` is (e.g., "defect present / absent / need-more-views" per surface region) — the conformal recipe transfers, the label semantics do not.

## 4. What transfers to our setting

- **The stopping-criterion recipe is the exact answer to the branch's named question** ("What is the stopping-criterion literature... can it give us a defensible 'enough views' rule?"). The transferable skeleton, independent of EQA specifics:
  1. Define a per-view confidence score that is *relevance-gated* (don't trust a high answer-probability from an irrelevant/off-target view) — directly useful, since our "is this the orthogonal evidence view for surface region k" judgment has the same shape: a view of the wrong face could still produce a confident-sounding but wrong VLM judgment.
  2. Take the **running minimum**, not average, of confidence across views taken so far (Eq. 9) — a conservative "worst view still constrains us" aggregator, appropriate if a single bad/occluded view should not be washed out by several good ones.
  3. Use **split conformal prediction** on a held-out calibration set to convert a heuristic confidence score into a statistically-covered stopping threshold (`q̂` from the empirical quantile) rather than hand-tuning a probability cutoff.
  4. Stop when the "still-possible" set — states/labels consistent with everything seen so far — **collapses to one hypothesis**, tracked by *intersecting* per-timestep sets rather than re-deciding fresh each step.
- **The early-budget efficiency gain (not final accuracy) is the actual empirical win** — semantic weighting mainly helps use *fewer* steps for the same success rate (71% vs 85% of budget at matched 58% success, Sec. VII-G), which is the framing most relevant to us: we don't need to beat a ceiling accuracy, we need to stop taking views sooner without losing coverage.
- **The set-of-marks-with-letters interface (Sec IV-C)** is a clean, minimal existence proof that discrete-candidate-marking + logprob extraction is implementable and gets used by an open VLM in a real pipeline (also on real hardware, Sec. VII-H) — useful as a pattern reference, though it offers no failure-mode data (see §3).

## 5. What does NOT transfer, and why

- **EQA question-answering ≠ view-selection-for-inspection.** The target quantity here is "which multiple-choice answer is correct," evaluated by exact/semantic match to a ground-truth string. Our task has no such discrete ground-truth answer space per view — "enough views of this surface" is a coverage/defect-detection judgment, not a QA-answer judgment. Porting the machinery requires us to invent an analogous label space; the paper offers no guidance on how to do that for a coverage task.
- **Requires VLM logprob/token-probability access**, which constrains us to either an open-weight model or an API that exposes logprobs for the exact tokens used (letters, yes/no, answer choices). If our orchestrator is committed to a closed frontier model without reliable per-token logprob access for custom label sets, this exact recipe doesn't port as-is; a workaround (e.g., self-consistency sampling to approximate a probability, or explicit confidence-elicitation prompting) would be needed and is untested by this paper.
- **Only 3 candidates were ever tested.** Nothing here bears on menu sizes in the tens (our likely view-menu scale) or on option similarity/distinctiveness — the paper picked 3 because that's "sufficient to cover the possible distinct regions" of a room frontier, an empirical choice for a different domain, not a principled bound.
- **Orientation-blindness is a known, flagged gap in the source paper**, and their own +0.9pt ablation is weak evidence in a domain (room-scale navigation) where our core requirement (surface-orthogonality) barely exists as a concept. Do not cite this paper's near-null orientation ablation as evidence that orientation is unimportant for us — the tasks are not comparable on this axis.
- **No comparison to a random/uniform view-selection baseline is given here** — the comparison baselines are FBE (unweighted frontier exploration, itself already a reasonably strong non-random heuristic) and "Relevance" (single-VLM-query, no conformal calibration). Neither is a pure-random baseline, so this paper cannot supply the branch's separately-tracked "what is the random baseline" answer (that question is explicitly delegated to a different branch/paper, e.g. SPAZER).
- **Single-run numbers throughout, no variance.** Any specific percentage above (58.4%, +0.9pt orientation gain, etc.) should be treated as a point estimate, not a statistically defended delta.

## 6. Branch-question-by-branch disposition

- **"What is the stopping-criterion literature (Explore-until-Confident's conformal calibration) and can it give us a defensible 'enough views' rule that the current design lacks?"** — **Directly and fully answered.** The recipe: relevance-gated per-view confidence → running-minimum aggregation over time → split-conformal calibration on a held-out set → stop when the surviving-hypothesis set collapses to one. This is portable *as a pattern*; the label space, calibration set, and relevance-gating function all need to be redesigned for coverage/defect-inspection rather than QA. Calibration set size in the source paper (300 scenarios) is a scale reference, not a requirement — our calibration set (real defects, likely far smaller) is a first-order risk to flag, not resolved by this paper.
- **Random baseline question** — not answered here; this paper's FBE and Relevance baselines are both non-random heuristics. Delegate to SPAZER / classical NBV literature (already tracked elsewhere per the branch list).
- **Failure-mode catalog (handoff to Branch 2)** — this paper contributes two specific, quotable failure modes: (a) **VLM answers confidently wrong despite viewing the correct/relevant object** ("largely due to the wrong VLM answer prediction even when the robot sees the relevant object," Sec. VII-C; 2 of 6 hardware failures were exactly this, Sec. VII-H) — a "seeing ≠ correctly reasoning" failure distinct from the mark/OCR/ego-allo failures already catalogued; (b) **miscalibrated relevance scores** — successful episodes often had top relevance scores below 0.5 (Appendix C, Fig. A5), meaning the raw relevance signal alone is a poor confidence proxy without conformal correction, which is exactly why the paper doesn't use it raw.
- **Marks/set-of-marks failure modes** — this paper is silent (see §3, §5); do not cite it as either evidence for or against mark-reading robustness.
- **Ego/allo, viewpoint-relation, pose-encoding, compositional-operation, option-count/order-sensitivity, chicken-and-egg render-viewpoint questions** — **out of scope for this paper**; it neither measures nor discusses any of them. Do not force citations here.

## 7. Evidence-discipline entry

Paper = own-authors' method vs. own-authors' baselines (FBE, Relevance, Entropy), single environment family (HM3D), **single-run, no variance/seeds reported anywhere** (flagged, not cited as statistically robust). Ablations (LSV/GSV components, temperature, orientation) are internal ablations of the *exploration* half; the *conformal-stopping* half (Sec. V) is validated via the standard theoretical coverage guarantee of split conformal prediction (a distribution-free guarantee, not an empirical ablation) plus the efficiency-vs-baseline comparison in Fig. 9 — that comparison (71% vs 85% of budget at matched accuracy) **is** a real empirical delta, but again single-run.
