# Branch 2 — `benchmarking-view-planning`
## Round-4 accumulation: Benchmarks and evaluation methodology for view planning

**Date:** 2026-08-27
**Role:** branch synthesizer, pass 1 / round 4.
**Branch plan:** [`../round1_plan.md`](../round1_plan.md) § Branch 2
**Round-2 map:** [`../round2_map/benchmarking-view-planning.md`](../round2_map/benchmarking-view-planning.md)
**Reading notes:** [`../round3_reading/benchmarking-view-planning/`](../round3_reading/benchmarking-view-planning/) — 20 documents, all read in full for this synthesis.
**Contract:** [`../../../../geometry-aware-view-planning.md`](../../../../geometry-aware-view-planning.md)

---

## 0. Corpus and coverage — what this synthesis is built on

Twenty documents were read to depth out of 58 mapped candidates. Coverage by
map group:

| Group | Read / mapped | What was read | What the gap costs |
|---|---|---|---|
| A — benchmark artifacts & protocol design | 5 / 11 | `objview-bench`, `gennbv`, `vin-nbv`, `ma-scvp-longtail`, `nbv-taxonomy-survey` | The spine is covered. Missing `neu-nbv` (fixed-renderer confound isolation) and `how-many-views-prv` (per-object budget prediction). |
| B — reconstruction-error metric precedent | 0 / 2 | — | **Tanks & Temples / DTU unread.** We have no first-hand read of the precision/recall-at-τ convention or the median+variance reporting discipline. Second-hand only, via `pose-noise-robustness-plant-vp`'s F-score usage. |
| C — GT for "best view" | 3 / 5 | `dutagaci-best-view-benchmark`, `bonaventura-viewpoint-survey`, `secord-viewpoint-preference` | Strong. Missing `quoniam` (the third-party implementation that was the concrete anti-circularity instrument) and `which-viewpoint-shows-it-best` (the 78.5% inter-rater number). |
| D — simulation, synthetic GT, injected defects | 3 / 8 | `pose-noise-robustness-plant-vp`, `depthsynth`, `pbr-structured-light-sim` | Pose-error axis is covered. Dropout and fusion-offset axes are **not**. |
| E — benchmark methodology, statistics, circularity | 5 / 13 | `benchmark-lottery`, `measuring-what-matters`, `self-preference-bias-judge`, `embodied-iqa`, `binomial-eval-statistics`, `ycb-protocols` (6 incl. YCB) | Strong. Missing `mirrer` (cross-lab variance → local relative ranking) and `lbm-careful-examination` (blind randomized A/B). |
| F — VLM/agentic view selection protocols | 3 / 7 | `view-level-evidence-benchmark`, `e3vs-bench`, `planning-with-the-views` | The three highest-priority ones. Missing `scoutvla` (three independent judges + causal ablation of whether active perception yielded more answerable evidence). |
| G — defect-visibility ground truth | **0 / 3** | — | **The most consequential gap.** M2AD / Real-IAD / Real-IAD-MVN were the sanctioned exception to the 2D-AD exclusion and the only entries that would ground "defects hide from the wrong viewing angle" in data rather than in Albert's intuition. Nothing in the read corpus supplies defect-visibility ground truth. |
| H — applied inspection-planning evaluation | **0 / 5** | — | `cpp-measurement-uncertainty` (IK-feasibility + collision indicators per viewpoint, honest trade-off reporting) was rated **high** and was not read. The closest published evaluation design to ours is unread. |
| I — object sets / CAD ground truth | 0 / 2 | — | Deliberately deferred, downstream of the Q4 decision. Still deferred. |

**Read this synthesis knowing:** the branch is well covered on *protocol shape*,
*best-view GT strategies*, *circularity*, and *statistics*. It is thin-to-absent
on *defect-visibility evidence*, *applied inspection evaluation*, and *the
reconstruction-error metric conventions themselves*.

---

## 1. Corrections to the round-2 map — claims falsified by reading

These are recorded first because the round-2 map is a live document that other
branches and later passes will cite. Four of its per-entry claims did not
survive the read.

1. **`measuring-what-matters` does not contain the LLM-judge circularity /
   self-preference content the map attributed to it.** The map's headline
   finding #3 stated its "diagnosed failure modes are ours verbatim" including
   judge circularity. The paper mentions LLM-as-judge only as a *metric-type
   statistic* (17.1% partial / 3.1% exclusive use) and flags extractor-LLM bias
   as a *parsing* concern (Rec. 3.2), never as self-preference or judge
   contamination. Cite it for construct validity and statistical hygiene only.
   Q3's circularity mechanism must be sourced from `self-preference-bias-judge`.

2. **`pbr-structured-light-sim` (arXiv:2407.12449) does not simulate active
   stereo.** The map selected it for simulating "the D405's sensing modality."
   It simulates a **single projector + single camera gray-code sequential
   structured-light scanner** (Photoneo class). The paper matching the D405's
   modality (two IR cameras + speckle projector + block matching, single-shot)
   is its own reference [32]: **X. Zhang et al., "Close the visual domain gap by
   physics-grounded active stereovision depth sensor simulation,"
   arXiv:2201.11924 (2022)** — unread, and now the correct entry point for
   believable D405 dropout. The two sensors' failure modes are mechanically
   different (projector-occlusion + gray-code binarization ambiguity vs.
   stereo-correspondence failure on textureless/specular IR return), so
   2407.12449's defect statistics do not transfer even though its methodology
   does.

3. **`binomial-eval-statistics` (SureSim) never uses Clopper–Pearson and does
   not report the "70 rollouts → 15.4-point interval → 15× more rollouts"
   scenario.** The reader grepped for it; it is absent. The paper's machinery is
   the Waudby-Smith–Ramdas (WSR) betting interval, a *generalization* of exact
   binomial intervals to bounded non-Bernoulli metrics. The reader independently
   verified the curator's own arithmetic: 15.4 points at n=70, 90% CP **is
   correct** for p̂≈0.85, but the "~15×" cost ratio for tightening 10→2 points is
   wrong — the true ratio is **~24×**. Do not repeat "15×" as a checked number.
   The substantive upgrade: **WSR, not Clopper–Pearson, is the right tool** if
   our view-quality score is a bounded partial-credit scalar rather than a
   binary pass/fail.

4. **`objview-bench`'s reachability constraint is not live per-object IK.** The
   map's headline finding #1 called it "reachable-view subsets filtered by IK
   and collision." In the paper, reachability is abstracted to a coarse
   geometric proxy — **whole sphere vs. quarter sphere**. The only real IK/collision
   number is a single validation statistic (§6.3): a tabletop rig check found
   **45.4/128 candidate views reachable on average, range 26–74**, across 5
   objects × 5 placements, used to *motivate* the quarter-sphere abstraction. No
   robot or camera is named. The external-validation claim for our constraint
   stack is therefore weaker than the map asserted — but the 45.4/128 figure is
   itself a strong, citable independent confirmation that **roughly two-thirds of
   a naive candidate sphere is kinematically unavailable in a tabletop rig.**

A fifth, softer correction: the map credited `dm-osvp-plus-plus`, `pb-nbv`,
`gleam-bench` and others with payloads that were never verified because they
were not read. Treat un-read map entries as *claims about papers*, not findings.

---

## 2. Headline findings

Ten things the reading established. Ordered by how much they constrain the
benchmark design.

**F1 — The field's standard protocol stack exists, is coherent, and is
structurally unable to express our construct.** The modern convention, in three
layers: (a) a fixed shared candidate view set, (b) a unified view budget, (c)
surface-coverage ratio + AUC-over-the-trajectory + Chamfer distance to a GT
mesh. `gennbv` established (b) and (c) explicitly and stated the flaw it fixes
("the coverage ratio is highly correlated with the number of views"). Every
geometric paper read — `gennbv`, `vin-nbv`, `ma-scvp-longtail`, `objview-bench`,
`pose-noise-robustness-plant-vp` — computes visibility as a **binary ray-cast
line-of-sight test with no incidence-angle term.** A voxel seen from an 85°
grazing angle counts identically to one seen from 0°. This is not an omission we
can patch; it is the construct. **Our requirement (evidence views orthogonal to
the inspected surface) is invisible to the entire standard metric family.**

**F2 — Nobody has a validated orthogonality-to-surface view score.**
`bonaventura-viewpoint-survey` compares 22 viewpoint-quality measures head-to-head
against human GT — and **Kamada & Kawai's angle-between-view-direction-and-face-normal
measure (1988), the literature's most direct analogue to our criterion, is
explicitly excluded from the 22 tested.** It appears only as pioneering
background. The five best-performing measures (projected saliency, visible
triangle count, viewpoint entropy/I2, curvature entropy, Plemenos & Benayada) are
all *whole-object* informativeness scores. So: the idea "angle to normal" is old
and citable, but **there is no published measure of it with any human-agreement
or task-outcome track record.** Building one is original work, not adaptation.

**F3 — Whether the uniform sphere is a strong or a weak baseline depends
entirely on which metric you pick, and the two readings are both in the
corpus.** On coverage metrics the uniform sphere nearly saturates: `gennbv`'s
budget-matched Uniform Hemisphere scores 82.91% coverage-AUC vs. the full learned
method's 91.19%, and *matches* the per-scene-NeRF-optimized "oracle-ish"
baselines (83.13%, 84.86%); `ma-scvp-longtail` gets **98.41% visible-surface
coverage from 13 views selected out of a generic 32-view sphere**;
`objview-bench`'s Random+TSP reaches NSC 0.996 at K=30. On reconstruction-fidelity
metrics it is clearly beatable: `vin-nbv` shows Uniform 0.41 cm vs. VIN-NBV
0.20 cm Chamfer, and its reader's conclusion is that "most of the benchmark's
discriminative range is between *informed* and *uninformed*, not within uninformed
variants." **Both are true. The implication is uncomfortable and load-bearing:
if we adopt a coverage metric, we will very likely fail to beat the uniform
sphere — not because our action space is worse, but because the metric ceiling
is at 98%+ and the sphere already reaches it.**

**F4 — "More views is better" is false, and the field has three independent
demonstrations.** `ma-scvp-longtail` ablates k = 0…3 extra NBVs before its
one-shot planner fires: k=2 and k=3 are **not statistically significant** against
k=1 on coverage (paired t-test + Wilcoxon, α=0.05) while costing 20% and 40%
more movement — an explicitly reported no-benefit result. `vin-nbv` reports its
advantage over a coverage baseline is **front-loaded**: "large improvements
during the initial stages... towards the later stages, both policies result in
similar reconstruction quality." `objview-bench` shows the *winner changes with
budget*: RSE (classical) at K=5, PoinTr-C+MCP at K=30, PoinTr-C+SCP under
auto-stop. Corroborated from the map (unread): M2AD reports diminishing or
negative returns from excessive view/illumination addition.

**F5 — Reachability and budget change method rankings, and some action-space
designs fail catastrophically rather than gracefully.** `objview-bench`'s
whole→quarter view-space shift degrades classical/hybrid planners by ΔNSC ≤0.003
but collapses NBV-Net from **0.988 → 0.472**, with executed views dropping
9.9 → 1.2. Cause: a single-index discrete action output with no fallback — an
infeasible prediction terminates the episode. This is the sharpest available
warning for our own action-space design: **a discrete selection interface that
cannot express "that one is unreachable, give me the next best" is a
catastrophic-failure pattern, not a degradation pattern.** `ma-scvp-longtail`'s
authors name the same problem as an open limitation in their own system: "the
candidate view space is pre-defined... This does not allow the system to drop
some unreachable candidate views."

**F6 — The circularity mechanism is *shared assumptions*, not shared code, and
the empirical size of the resulting error is measurable.** `self-preference-bias-judge`
shows GPT-4's self-preference bias at 0.520 (Equal Opportunity form) / 0.749
(Demographic Parity form), and — the load-bearing part — that the bias tracks
**perplexity/familiarity rather than authorship detection**: the
low-perplexity-preference curve is essentially identical whether or not the text
is the judge's own output. So "we reimplemented the metric" is not a defense.
`embodied-iqa` supplies the magnitude: full-reference IQA metrics that score
**~0.9 SRCC against human perception** drop to **<0.65 (recomputed: 0.582) against
an embodied-task construct**, and the chain degrades again downstream —
Cognition→real-Execution SRCC **<0.5**, Decision→Execution **~0.6**. Two
compounding gaps, each reported separately. And: **rotation is the hardest
Decision dimension for every proxy metric tested** (TOPIQ-FR: Position 0.775 >
State 0.668 > Rotation 0.643) — which is exactly the axis our orthogonality
metric lives on.

**F7 — End-to-end outcome scoring overstates system quality by up to 70
percentage points when the evidence-selection step is unmeasured.**
`view-level-evidence-benchmark`'s 2×2 grid (view-correct × answer-correct) gives
the number: Qwen2.5-VL-7B's **raw answer accuracy is 78.7%; its grounded
(view-and-answer-correct) accuracy is 8.2%.** Pooled across six models, **26.2%
of all trials are wrong-view/right-answer**. `e3vs-bench` independently
self-reports the same failure without measuring it: "agent's final observation
differs substantially from human-annotated goal viewpoint while receiving similar
judge score, indicating automated evaluation may insufficiently capture viewpoint
adequacy." `planning-with-the-views` finds the mechanism-level analogue: **≥90%
of successful multi-turn view-localization runs (99.1% for Gemini 3.1 Pro)
involve the agent visually passing through the target view before submitting** —
i.e. success by wander-and-match, not by prospective reasoning.

**F8 — Frontier VLMs compose multi-turn view plans badly, and the failure grows
with viewpoint distance.** `planning-with-the-views`: single-turn view-transition
tasks sit at 20–53%, multi-turn interactive view planning collapses to
single-digit-to-low-20s%, and **every model's long-distance score is far below
its short-distance score** (GPT-5.4 Pro 34.8% → 11.7%). Best frontier = 21.3%.
More turn budget does not rescue it ("diminishing returns suggest planning
bottleneck, not exploration horizon"). Mechanistic decomposition worth keeping:
**rotation distance drives single-step view-prediction failure; position distance
drives multi-turn compositional failure.** `e3vs-bench` independently shows the
same shape: best model 2.79/5 vs. human 3.53, but **VQA-at-goal (hand the model
the right view) = 3.58 ≈ human** — the bottleneck is finding, not reasoning.

**F9 — The field's statistical practice is worse than the methodology
literature's own minimum bar, and the escape route it uses is closed to us.**
No confidence intervals, no variance, no significance testing in `gennbv`,
`vin-nbv`, `objview-bench`, `e3vs-bench`, `pose-noise-robustness-plant-vp`,
`dutagaci`, `bonaventura`, `pbr-structured-light-sim`. Two exceptions:
`ma-scvp-longtail` (paired t-test **and** Wilcoxon signed-rank on 3,000 matched
cases, α=0.05, significance stars per pairwise comparison) and
`view-level-evidence-benchmark` (mean/SD/95% CI over 5 runs at N=122).
`measuring-what-matters` puts the field-wide rate at **16% of 445 benchmarks
using any uncertainty estimate or statistical test.** The big benchmarks
substitute *object-count scale* (528–5,691 objects) for *trial replication* —
**we cannot buy precision that way.**

**F10 — Physical sensor behaviour supplies a hard anchor for the orthogonality
requirement at ~70°.** `depthsynth`'s error-vs-tilt validation (Fig. 8B, real
Kinect + simulator, 0–80°) finds that past roughly **70° incidence, both real
scans and simulated ones have most of the depth information missing** — "due to
the poor reflection and stretching of the projected pattern(s), heavily impairing
the reconstruction." This is a data-availability cliff, not a cosmetic
preference. It is the strongest physics-grounded justification in the corpus for
treating incidence angle as a first-class quantity, and it gives a candidate
outer bound for any angular-tolerance envelope. Caveat: it is Kinect-v1-specific
in magnitude; the taxonomy is sensor-agnostic within the structured-light /
active-stereo family, the numbers are not.

---

## 3. What the field actually does — the protocol stack, laid out

### 3.1 The per-paper matrix

| Paper | Unit of evaluation | Primary metric | GT source | Budget / stop | Baselines | Defects injected | Statistics |
|---|---|---|---|---|---|---|---|
| `gennbv` | (object, run) | Coverage-AUC + FCR + Chamfer | Clean mesh, Poisson-disk 100k pts, voxelized | Fixed 30 / 20 views, hard cap | Random, Random-Hemi, **budget-matched Uniform Hemi**, 2 NeRF-based, Scan-RL, 2 cross-impl. ablations | **None** ("accurate observations like depth maps") | 5 seeds averaged, **no variance reported** |
| `vin-nbv` | (object, budget) | Chamfer distance | Clean mesh + noise-free GT depth | Fixed 20 acquisitions | Random-Hemi, Uniform-Hemi, ActiveRMAP, Scan-RL, GenNBV, **Cov-NBV, Oracle-NBV** | **None** (author-acknowledged limitation) | Single mean per method, **none** |
| `ma-scvp-longtail` | (object, rotation, initial view) — 3,000 cases | VSC, Required Views, Movement Cost, Recon. Time | Sim: mesh raycast. **Real: exhaustively capture all 32 views + neighbor-tolerance** | Plan length *is* the stop; k tuned by ablation | PCNBV, GMC, MCMF (2 strictness levels), SCVP | **None** (real-world noise observed post-hoc only) | **Paired t-test + Wilcoxon on matched cases, α=0.05** |
| `objview-bench` | (object, run) | NSC@τ, SC@τ, Path Cost, #Views, Runtime | Hidden GT mesh, server-side | **K=5 / K=30 / auto-stop cap 128 with MS@δ** | 9 methods, 3 families. **No uniform-sweep policy** | **None** (deferred to future work) | **None** — point estimates over 528–5,691 objects |
| `pose-noise-robustness-plant-vp` | (session-pair) | Coverage (raycast, mesh-free) + Chamfer + F-score@{0.01,0.02}m | Public dataset + **independent instrument (Polycam LiDAR + Nerfstudio)** | One-shot set cover | GMC, MA-SCVP, Block-Wise NBV, 3 random-prior variants | **Pose noise: σ_t/σ_r ∈ {(2mm,0.5°),(5mm,1°),(10mm,2°)}** | mean±std over 36 / 8 cases, **no significance testing** |
| `dutagaci` | (object, single view) | View Selection Error (geodesic dist. to nearest symmetry-set member, /π) | **26 human subjects**, 258-vertex geodesic sphere | Single view, no budget | 7 geometric measures | None (clean mesh both sides) | mean VSE + win-count, **no CI** |
| `bonaventura` | (object, single view) | Same VSE machinery, 642-vertex sphere | Reuses Dutagaci GT frozen | Single view | 22 measures reimplemented in one framework | None (mesh discretization only) | box plots only |
| `secord` | (object, view pair) | Bradley-Terry fit, **naive-floor/oracle-ceiling normalized F ∈ [0,1]** | **127,140 crowdsourced 2AFC choices**, 524 workers | Pairwise, no budget | Naive (random) floor, empirical-agreement Oracle ceiling | None | **100× repeated half/half object splits + Tukey HSD "indistinguishable pool"** |
| `view-level-evidence-benchmark` | (scene, question) with **view-level label** | Acc_view / Acc_oracle / **Acc_joint** + 2×2 grid | 3D-bbox reprojection **+ human verification**; ambiguous items **excluded** | One-shot 7-way choice | 6 MLLMs, naive-uniform reference | None (real NuScenes captures) | **mean / SD / 95% CI over 5 runs**, N=122 |
| `e3vs-bench` | (scene, question) episode | **VLM-judge score 1–5** + steps + collision rate | Human-annotated answerable-viewpoint set; **goal image to judge only** | Max 25 steps + model `stop` action | Blind / VQA-at-start / VQA-at-birdview / **VQA-at-goal** / random / human | None (3DGS RGB only) | **None** |
| `planning-with-the-views` | (view pair) instance / episode | Pose-distance success (≤1 action step in *both* translation and rotation) | **ScanNet camera extrinsics** — pure geometry, **no judge anywhere** | Max turns ablated 10/20/30 | Random Response reported in the main table | None (real scans used as-is) | Single evaluation, **no CI** |
| `ycb-protocols` | (task, trial) | Task-specific partial-credit scores | **Physical measurement** (scale, printed target overlap) | — | Uniform-random-pick heuristic baseline, reported with full rigor | Peg insertion: 5mm / 10mm positioning perturbation | mean ± SD at N=6–10, **no CI** |
| `binomial-eval-statistics` | (environment, rollout) | Bounded [0,1] partial-credit success | Real robot trial, rubric-scored | — | "Classical" real-only WSR interval | — | **WSR non-asymptotic CIs, PPI, finite-sample Type-I guarantee** |

### 3.2 What is genuinely converged

- **Fixed, shared, dataset-wide candidate view set that every method selects
  from.** Universal: 120 views / 3 shells (`vin-nbv`), 32 spherical-codes
  (`ma-scvp-longtail`), Tammes-128/360 (`objview-bench`), 258 (`dutagaci`), 642
  (`bonaventura`), 32/63 Tammes or 1000 Poisson-disk (`plant-vp`). Note the
  spread — **the field has no consensus candidate count**, each paper picked its
  own, and `bonaventura` silently changed Dutagaci's 258 to 642 without
  re-justifying it.
- **Budget must be unified across compared methods** (`gennbv`, stated as the
  fix to a named methodological flaw).
- **Report a curve-derived aggregate, not just a terminal score** (coverage-AUC).
- **Hold one stage fixed and swap only the other.** `vin-nbv`'s
  Oracle-NBV / VIN-NBV / Cov-NBV triple (identical greedy loop, identical 120-view
  set, only the fitness function differs) is the cleanest instance; `gennbv`'s
  cross-implementation ablations ("Scan-RL w/ Our Repre." × "Ours w/ Scan-RL
  Repre.") is the second; `ma-scvp-longtail`'s NBV-submodule swap is the third.
- **Ship the artifact, not just the numbers** (`ycb-protocols`, argued from 33
  prior object sets of which nearly all failed on physical availability).

### 3.3 What the field has *not* solved, and knows it

- **Sensor defects inside the evaluation loop.** Zero of the geometric benchmarks
  inject any. `objview-bench` defers it explicitly ("incorporating depth noise
  and calibration error... beyond the current idealized rendered RGB-D setting").
  `vin-nbv` names it as limitation #1. `gennbv` states clean depth as an
  assumption. `pose-noise-robustness-plant-vp` is the one paper that does a
  defect sweep, and it is **single-method** — no baseline is re-run under noise,
  so it cannot and does not report ranking stability.
- **Unreachable candidate views inside a fixed discrete action space.**
  `ma-scvp-longtail` names it as an unsolved limitation and proposes a
  speculative fix (learn reachability as an extra signal) that is not
  demonstrated. Their actual workaround was to *nudge the whole view space*
  (+8 mm up, lower the table).
- **View equivalence under symmetry.** Four incompatible answers exist (§5.6).
- **Statistical treatment** (F9).

---

## 4. Question-by-question accumulation

The Q-numbering is the round-2 map's. Each entry: what is now **known**, what
**transfers**, what is **contested**, what is **missing**, and — where the
evidence supports one — a **recommendation** (a recommendation, not a decision;
the decisions are the architect's, listed in §8.4).

### Q1 — What is the unit of evaluation?

**Known.** The corpus contains four distinct answers, all instantiated:
1. **(object, run) scored by cumulative coverage** — `objview-bench`, `gennbv`,
   `ma-scvp-longtail`, `vin-nbv`. Not per-region and not per-question. Dominant
   in geometric view planning.
2. **(object, single view) scored against a best-view label** — `dutagaci`,
   `bonaventura`. No sequence, no budget, no stopping.
3. **(object/scene, question) episode scored end-to-end** — `e3vs-bench`
   (holistic 1–5 judge score), `planning-with-the-views` (pose-localization
   success).
4. **(scene, question) episode with a view-level label attached** —
   `view-level-evidence-benchmark`. The only design in the corpus that scores
   *both* the evidence selection and the outcome, separately, on the same item.

**Transfers.** Option 4's three-number-plus-grid reporting (`Acc_view`,
`Acc_oracle`, `Acc_joint` + the 2×2 count table, never blended into one score)
is directly adoptable and is the only design that detects F7's failure.
`e3vs-bench` is the working counter-example: a well-resourced episode-scored
benchmark (1,120 human-annotation hours) that ran into exactly this problem and
left it unresolved in its own limitations section.

**Contested.** Option 1's readers argue coverage-per-run is the field norm and
is what makes cross-paper comparison possible. Option 4's reader argues it
inflates by up to 70 points. Both are right about their own construct: the
geometric papers have *no distinguished surface region* — the object is either
reconstructed or not — so a per-region unit is not merely absent, it is
structurally inapplicable to them.

**Missing.** **No paper in the corpus uses an (object, surface region) unit.**
`vin-nbv`'s reader: "the rejected option (per-region scoring) is not even
considered." `planning-with-the-views`'s reader: "structurally inapplicable."
`ma-scvp-longtail`'s reader: "if we adopt a per-region evidence-view score, we
are extending past this paper's evaluation unit, not adapting it." This is
consistent across all four geometric readers — **the per-region unit our
requirement implies has no precedent and would be original.**

**Recommendation.** A two-level unit: the *primary* scoring unit is
**(object, surface region)** scored by evidence-view quality (this is what RQ1
and Albert's requirement are actually about, and it is what admits an analytical
ceiling); the *secondary* unit is **(object, question) episode** carrying the
view-level label, reported as `view-correct × outcome-correct`. State explicitly
that the (object, run) coverage unit was rejected and what it would have measured
(whole-object reconstruction completeness, which is not our construct).

### Q2 — Ground truth for "the orthogonal/best view" on a primitive

**Known.** Four strategies exist, three of them read:

| Strategy | Instance | Mechanism | Known noise floor |
|---|---|---|---|
| Human-labelled | `dutagaci` (68 models × 26 subjects), `secord` (127k 2AFC choices) | Web/AMT collection, snapped to a discrete sphere, symmetry-set equivalence | **Human–human Inconsistency 0.107–0.456** on the same [0,1] scale as the algorithm score; picks span a **20–40° fan** and can be **bimodal** (`bonaventura` Fig. 1, cow) |
| Metric-based-against-humans | `bonaventura` (22 measures on Dutagaci's GT) | Frozen third-party GT, closed-form candidates ranked by error | **Best measure ≈0.38 error; nothing approaches 0**; mid-pack 0.43–0.47 heavily overlapping |
| Oracle-by-construction | `vin-nbv` Oracle-NBV, `objview-bench` §6.4 greedy oracle rollout | Same search loop with GT access substituted for the learned fitness | Gap to oracle largest early in the sequence |
| **Analytical** (ours) | *none in corpus* | Face normal through face centroid; cylinder radial | **Unknown — this is the open question** |

**Transfers.** (a) `secord`'s **naive-floor / oracle-ceiling normalized fitness
scale** — every candidate metric reported between a random-guess floor (F=0) and
an *empirical-human-agreement* ceiling (F=1), not against a hypothetical 1.0.
(b) `dutagaci`'s **Inconsistency metric** (Eq. 3–4): the same distance used to
score algorithms, applied leave-one-rater-out, giving a per-object noise floor
directly comparable to the algorithm score. (c) `pose-noise-robustness-plant-vp`'s
**independent-instrument GT** (Polycam LiDAR + Nerfstudio, a different device and
a different reconstruction pipeline from the system under test).

**Contested — and this is the sharpest contradiction in the branch.** The map
treated human "best view" labels as a first-class GT option. Two readings
undercut that for *our* construct:
- `dutagaci`'s GT criterion is explicitly "most informative for
  recognition/thumbnailing," and its reader is blunt: "A bunny's 'best view' is
  not necessarily surface-orthogonal to any particular patch. This paper's human
  label GT strategy is a *different GT paradigm*."
- `embodied-iqa`'s companion result (via the map, from EPD): human-vs-robot
  preference correlation of **0.13–0.21**. Human aesthetic/legibility preference
  and machine-usable evidence quality are weakly correlated constructs.
So: **the human-label strategy is the best-developed of the three, and is
probably measuring the wrong thing for us.** Do not smooth this — it means the
"three established answers" framing from round 2 is really "three established
answers to a different question."

**Missing.** The actual Q2 question — *does the analytical GT survive contact
with a defective survey cloud, and what is its angular error?* — has **zero
evidence in the corpus.** `dutagaci`, `bonaventura`, `secord`, `gennbv`,
`vin-nbv`, `ma-scvp-longtail`, `objview-bench` all compute GT from a clean mesh
and sidestep the question by construction. This is Branch 3's number to supply
and it must be asked for explicitly.

**Recommendation.** Analytical GT (primitive fit → face normal / cylinder
radial), *with a measured angular-uncertainty band*, refereed only if Branch 3's
normal-error is small relative to the method differences we want to resolve. The
`dutagaci` Inconsistency construct is the template for reporting that band: the
GT's own noise floor, on the same scale as the score. If the band is not small,
the analytical GT cannot referee and we fall back to relative ranking against a
permanent in-house baseline.

### Q3 — How does the benchmark avoid being circular with the RQ1 judge?

**Known.** Three separate results now bound this question.
1. **The mechanism** (`self-preference-bias-judge`): bias tracks
   perplexity/familiarity, *not* authorship. The judge never needs to detect
   self-generation; the candidate only needs to *look like* what the judge would
   have produced. Therefore reimplementation is not separation.
2. **The magnitude** (`embodied-iqa`): a proxy at ~0.9 SRCC on its native
   construct drops to 0.58 on an adjacent one, and to <0.5–0.6 against real
   execution. And **fine-tuning toward the new construct actively destroys the
   old one** (SRCC <0.4 on LIVE) — the divergence is bidirectional, not just
   "the new task is harder."
3. **The structural risk** (`benchmark-lottery` §6, "rigging the lottery"): a
   field with no external canonical benchmark leads researchers to fit the
   protocol to their own method, without bad intent. Our case is *worse* than
   the paper's own examples — RecSys and ALE at least had N>1 independent groups
   picking different self-serving setups; **we are N=1 team designing the
   benchmark, the RQ1 judge, and the RQ2–4 method, and iterating all three
   together.** The paper offers no technical fix, only procedural ones
   (pre-registration, statistical protocol decided in advance, reviewer
   checklists).

**Transfers.** Working separations demonstrated in the corpus:
- **`planning-with-the-views`: no judge at all.** Success is a pure geometric
  computation from GT camera extrinsics; the only place human judgment enters is
  a one-shot threshold calibration. This is the cleanest existence proof that a
  view-selection benchmark can be built judge-free.
- **`ycb-protocols`: every score is a physical measurement** (weight scale,
  printed-target overlap) — a whole benchmark suite that never has to confront
  circularity because it chose constructs with no learned component.
- **`objview-bench`: hidden-geometry server.** GT mesh lives server-side; the
  method sees only rendered observations from poses it requests.
- **`view-level-evidence-benchmark`: partial, and honest about where it fails.**
  View-correctness GT is reprojection geometry + human verification, no model in
  the loop — genuinely clean. But the free-form-answer LLM judge's **model
  identity is never disclosed**, and Claude is one of the six evaluated systems,
  so an unquantified self-preference channel sits inside an otherwise clean
  design. Only the view-accuracy numbers are judge-free.
- **`e3vs-bench`: GT-for-scoring-only.** The goal image reaches the judge, never
  the agent. But construction-time filtering and evaluation-time scoring both use
  the GPT-5.1 family — **not a construct separation, an information-flow
  separation.** Its own judge–human agreement is **ρ=0.54**, which the authors
  call "reasonable" and the reader flags as moderate.

**Contested — and unresolved.** The `self-preference-bias-judge` reader's
argument is that geometric GT is *categorically* outside the contaminated space,
because a normal-vector computation has no perplexity/familiarity substrate. That
is right as far as it goes. But our geometric GT is **a primitive fit to a
defective cloud**, and the same fitting assumptions would inform the RQ1 judge
and the RQ2 selector. The paper's actual mechanism (shared substrate → spurious
agreement) applies to *any* shared assumption, including a geometric one. A
second data point in the same direction: `dutagaci`'s seven algorithms all share
the "maximize a geometric-complexity descriptor over a sampled sphere" paradigm,
and the paper's own conclusion — "none is consistently best; they are rather
complementary," clustering at VSE 0.396–0.517 — reads as **diversity of
implementation without diversity of construct.** So: geometric GT removes the
*model-familiarity* channel; it does not remove the *shared-assumption* channel.

**Missing.** The concrete anti-circularity instrument the map named — `quoniam`,
a third-party implementation of the 22 measures usable as a held-out score — was
not read. No defect-visibility outcome labels (Group G unread) exist to serve as
the "different outcome" separation.

**Recommendation.** Stack three separations rather than claim one:
- **Different construct** — GT from geometry (analytical normal from the fit),
  judged with no learned component. Removes the familiarity channel.
- **Different provenance** — a held-out score whose *assumptions* differ, not
  just whose code differs. `quoniam` is the named candidate; an
  independent-instrument capture (`plant-vp`'s Polycam/Nerfstudio pattern) is
  the other.
- **Different outcome** — downstream defect-detection success, and report the
  **rank correlation between the judge score and that outcome as a headline
  benchmark result**, exactly the way `embodied-iqa` reports Table 2 and §5.4 as
  its two findings rather than asserting transferability. A low correlation is a
  finding, not a failure.
And state explicitly, in advance, **which candidate solutions our metric
advantages and why that is acceptable** — `benchmark-lottery`'s only usable
mitigation for the N=1 case is pre-registration before iterating.

### Q4 — Is view-set quality scorable without a ground-truth mesh?

**Known.** All three exits named in round 2 are instantiated in the corpus, and
a fourth appeared.
- **(i) Simulate from a mesh with injected defects.** `pbr-structured-light-sim`
  is a working instance with a controlled A/B built in (clean Z-buffer depth vs.
  physically-decoded noisy depth as two separate conditions). Its result is the
  interesting part — see Q9.
- **(ii) Dense pre-capture as GT.** `ma-scvp-longtail` §V-F1 does this
  concretely: exhaustively capture **all 32 candidate views**, voxelize the union
  at 0.002 m, and — the load-bearing detail — absorb per-view registration offset
  with a **neighbour-tolerance rule** ("for a voxel in the ground truth model, we
  search for its corresponding neighbors; if any neighboring voxels are covered,
  it is included in the coverage count").
- **(iii) Mesh-free per-region metric.** `pose-noise-robustness-plant-vp`'s
  coverage is a ray-cast visible-set fraction computed directly against a
  reference point cloud, needing no mesh. **But its visibility indicator has no
  incidence-angle condition** — a grazing view counts as full coverage — so it is
  an existence proof that mesh-free coverage is computable, not a candidate for
  our orthogonality score.
- **(iv, new) Independent higher-fidelity instrument as GT** —
  `pose-noise-robustness-plant-vp`'s greenhouse dataset: Polycam-on-iPhone-LiDAR
  + Nerfstudio depth-nerfacto, a different device *and* a different
  reconstruction pipeline from the system under test. Doubles as a Q3 separation.

**Transfers.** `binomial-eval-statistics` supplies the decisive constraint on
exit (i): **the precision of any cheap-proxy-plus-few-real-trials design is
capped by the proxy–outcome correlation, not by how much proxy data you
generate.** Their rectifier interval width does not shrink as N→∞ (validated to
N=50,000); benefit converges by N≈500. And the go/no-go test is cheap and
pre-registerable: augmentation only helps when **σ²(Y−f) < σ²(Y)** — the
rectifier variance must be below the real-outcome variance. In their own low-ρ
case (ρ=−0.051, σ²_Y=0.014 vs σ²_{Y−f}=0.029) they report plainly: **"none of our
methods beat Classical."**

**Contested.** Exit (i)'s cost was assumed to be "we need materials/CAD."
`depthsynth` confirms that: its specular/non-specular/shadow simulation depends
on BRDF properties baked into the CAD asset, and **a raw defective point cloud
has no material channel.** So exit (i) for us requires primitive/mesh fitting
*from* the survey cloud first, then synthetic material assignment — an extra,
lossy step. Meanwhile exit (ii)'s neighbour-tolerance trick was validated at
"slight" (sub-2 mm, below the GT voxel size) offsets; **our ~4 cm offset is
~2000× that voxel size.** The mechanism transfers; the magnitude it was
validated at does not.

**Missing.** Tanks & Temples / DTU unread, so the precision/recall-at-τ
convention is second-hand. And nothing in the corpus scores view-set quality on a
*defective* cloud that is itself the planning input — every paper either has a
clean mesh or treats real captures as ground truth.

**Recommendation.** A hybrid, because the exits answer different questions:
exit (i) (simulate from a fitted mesh, inject defects) for **ranking stability
under controlled defect magnitudes**, since only simulation lets us sweep;
exit (iii) (per-region angular metric on the cloud, no mesh) as the **primary
score**, because it is the only one that expresses our construct and it needs no
GT reconstruction at all; exit (ii) (dense pre-capture) reserved for a small
real-data validation slice. Cost exit (ii) honestly: `ma-scvp-longtail` needed a
full 32-view exhaustive capture per object, per placement.

### Q5 — What are the baselines, and is the uniform sphere fair?

**Known.** The field's baseline family is stable across papers: **Random in the
full space / Random on the sphere / Uniform on the sphere / a greedy or
information-gain classical planner / an oracle with GT access.** `vin-nbv` has
all of them in one table plus the Cov-NBV controlled ablation; `gennbv` has all
of them plus two cross-implementation swaps.

Two concrete recipes worth lifting verbatim:
- **`gennbv`: the uniform baseline must be budget-matched by construction** —
  5 heights × 6 azimuths = 30 views to match a 30-view budget, 4 × 5 = 20 to
  match 20. Not "a reasonable default sphere," a grid sized to the budget under
  test.
- **`ycb-protocols`: the baseline gets the same reporting rigor as the methods** —
  their uniform-random-pick heuristic is specified precisely enough to reproduce
  and is run with 10 trials, a full per-trial table, and mean/SD.
- **`planning-with-the-views`: a random-response row sits in the main table**,
  every time, not in a footnote.
- **`e3vs-bench`: a four-point calibration ladder** — Blind (1.82) → VQA-at-start
  (2.14) → VQA-at-birdview (**1.48**) → VQA-at-goal (**3.58**, ≈ human 3.53). The
  birdview number is the interesting one: **a generically "helpful-looking"
  overview viewpoint scored worse than the start view.** A plausible vantage is
  not an informative one.

**Contested — see F3.** The uniform sphere is near-saturating on coverage and
clearly beatable on fidelity. Two further complications:
- `objview-bench` **never tests a uniform-sweep policy at all.** The Tammes
  sphere is the candidate action set every method draws from, not a competitor.
  A denser uniform sphere *as a policy* is a gap even in the field's most recent
  dedicated benchmark.
- `pose-noise-robustness-plant-vp` uses **Tammes / Poisson-disk packings** as its
  "uniform" candidate sets, not angular {h,v} grids — suggesting a naive az/el
  grid is treated as the weaker construction in this literature. Our 12×30° × 3
  is a naive az/el grid. **A geometry-conditioned space beating a naive az/el
  grid is a weaker claim than beating a Tammes-packed sphere at matched budget.**
- `ma-scvp-longtail` supplies the counter-warning against over-correcting: their
  Appendix B-C explicitly criticizes fixed-budget baselines whose view count was
  "derived from training objects, and acquiring this knowledge in advance for
  unknown testing objects is unrealistic," and shows random-selection-with-optimal-path
  at matched budgets (12/13/14/16 views) scoring several points below their
  auto-stop system at the same view count. So a matched-budget uniform baseline
  needs a defensible source for the budget number.

**Missing.** `Karaszewski et al. 2016` — "Assessment of next-best-view algorithms
performance with various 3D scanners and manipulator" — surfaced by the ISPRS
survey as an actual comparative benchmark across scanners *and manipulators*, and
unread. The ISPRS reader flagged it as "the single highest-priority lead this
survey surfaces for Q5 and Q12."

**Recommendation.** Report **three** uniform variants, always: the current
12×30°×3 az/el grid (as the permanent in-house control — it is the thing being
obsoleted, so it must never disappear from a table), a **budget-matched** uniform
sphere sized to whatever the geometry-conditioned space uses, and a **Tammes-packed
sphere at the same budget** (the stronger uniform construction). Plus a random
floor and an oracle ceiling. Justify the budget number from geometry (set-cover
on the fitted primitives), not from training-set statistics.

### Q6 — Budget regimes and stopping

**Known.** Four distinct handling patterns, all instantiated:
- **Single hard fixed budget** — `gennbv` (30/20 views), `vin-nbv` (20
  acquisitions). Simplest, and `gennbv`'s stated rationale is the confound fix.
- **Three regimes** — `objview-bench`: K=5 (small fixed), K=30 (large fixed),
  auto-stop with a 128-view cap. **Empirically the winner changes across all
  three.** Methods with no native stop get a benchmark-side rule.
- **Plan length is the stop** — `ma-scvp-longtail`: the one-shot network's output
  vector size *is* the plan; no separate stopping decision exists for that stage.
- **Agent-emitted stop with a hard cap** — `e3vs-bench` (25 steps + a `stop`
  action), `planning-with-the-views` (turn budget ablated at 10/20/30).

The reusable generic stopping rule is `objview-bench`'s **MS@δ (map
stabilization)**: voxelize the accumulated cloud at resolution δ, stop when the
relative point-count gain is **<1% for 3 consecutive steps**. Its own diagnostic
of MS@δ is honest and worth carrying: on the main test pool MS stops at 9.0 views
with NSC 0.997 (gap to the full 128-view result Δ=0.0026); on slow-saturation
objects it stops at 12.7 views with a **~3× larger residual gap** (Δ=0.0081) —
"map stabilization is a practical stopping signal, but it measures observed-map
stabilization rather than long-horizon observable-surface saturation."

**Transfers.** `planning-with-the-views`'s turn-budget ablation is the check to
run before declaring a small budget unfair: they found **marginal gains from
10→20→30 turns**, concluding the bottleneck is planning, not exploration
horizon. If our VLM plateaus of its own accord, a small-fixed-budget regime is
the *right* regime to report in, not merely a resource concession.
`ycb-protocols`'s peg-insertion grid (execution duration × perturbation level,
with the plateau and the cliff named explicitly in the text) is the reporting
shape for a multi-condition sweep.

**Contested.** Whether the VLM's own stopping decision is part of what we
measure. `objview-bench` holds it fixed (benchmark-side MS@δ for planners with no
native stop) so that planners are comparable; `e3vs-bench` and
`planning-with-the-views` let the agent stop and measure the consequence.
Measuring both is possible but they are different claims: "this action space
supports better views" vs. "this agent uses this action space better" — which is
Q14 again.

**Missing.** `how-many-views-prv` (is the budget a per-object quantity to be
predicted?) unread. The set-cover-derived budget — the number our analytical
track would naturally produce — has no evaluation precedent as a *budget-setting
mechanism* other than `ma-scvp-longtail`'s plan-length-is-the-stop.

**Recommendation.** Small-fixed / large-fixed / auto-with-cap, following
`objview-bench`, because it is the only design that demonstrated rankings flip
across regimes. Hold the VLM's stopping decision fixed in the action-space
comparison (so the comparison is about the space) and measure it separately as
an agent-competence number.

### Q7 — A metric family that expresses "orthogonal to the inspected surface"

**Known.** Nothing off the shelf. Specifically:
- Coverage-AUC and Chamfer cannot express angular quality (F1).
- `bonaventura`'s 22 measures **do not include an angle-to-normal score** (F2),
  and all 22 are **whole-object, single-viewpoint** scores with **no per-region
  decomposition** — applying one to "how good is this view of *this* surface"
  requires restricting the polygon/point set to the region first, an extension
  the survey never performs.
- `nbv-photography-inspection` (mapped, unread) is the one candidate whose metric
  decomposes into distortion + scale + coverage, i.e. demotes coverage to one
  term of three — the closest published thing to what Q7 wants.

**Transfers — three usable skeletons:**
1. **`dutagaci`'s VSE structure**: `W_s(v) = (1/π) · min_{v' ∈ Σ} GD(v', v)`,
   averaged over raters then over objects. Swap "geodesic distance to the nearest
   ground-truth vertex" for "angular deviation from the analytic surface normal
   through the region centroid," keep the min-over-equivalence-set, keep the
   [0,1] normalization, keep the average-over-instances. A ready-made scoring
   skeleton with a known ceiling.
2. **`secord`'s attribute a12 ("above preference")**: `G(φ; 3π/8, π/4)`, a
   Gaussian bump in latitude peaked at an ideal viewing direction. This is
   structurally exactly an orthogonality score — an analytically defined angular
   preference with a known peak and a falloff — just aimed at a global prior
   instead of a per-surface normal.
3. **`secord`'s empirical warning against stacking terms**: surface visibility
   *alone* (a single ratio: visible area / total area) is the best single
   predictor at F=0.38, beating mesh saliency and everything else; the best
   5-attribute linear model reaches F=0.58; **"using more than five attributes
   does not improve performance."** And their attribute correlation matrix shows
   three tight clusters, with an explicitly discarded attribute that turned out
   to be a proxy for another. Argument for one dominant geometric term plus at
   most a couple of others, not a composite of many.

**Contested.** Two findings pull in opposite directions on how sharp the optimum
should be. `bonaventura`'s Figure 3 (642-viewpoint score heatmaps for 21
measures on 3 models) shows **broad, smooth, unimodal plateaus covering a third
to a half of the visible sphere** for the great majority of measures — "good" is
a region, not a peak. That supports building our GT as a **tolerance band** around
the normal. But it also means an empirical or learned scorer **will not
spontaneously discover a sharp, face-aligned optimum** — the sharpness has to
come from the geometry-aware GT construction itself. Meanwhile `depthsynth`'s
~70° data cliff (F10) argues the *physical* score is not smooth at all past a
threshold — it falls off a shelf. These are compatible only if the metric is
smooth in the usable band and hard-gated at the edge.

**Missing.** Two structural checks the corpus hands us that we must run on our
own candidates:
- **Provable rank-equivalence** (`bonaventura` §3.2, Eq. 8): viewpoint entropy
  and I2 are provably rank-identical on a fixed mesh (I2 = H(Z) − H(Z|v), H(Z)
  constant). Check algebraically whether two candidate orthogonality scores are
  *forced* to agree before reporting them as independent evidence.
- **Feature-basis counting** (`bonaventura` §3.1): 22 formulas collapse to ~5
  independent feature bases. Count bases, not formula variants, when claiming
  "we tested N candidate metrics."
- **Density sensitivity**: 10 of 22 measures are sensitive to mesh
  discretization; viewpoint entropy **diverges as resolution increases** while
  VMI/VKL are provably stable. The point-cloud analogue is direct — an
  entropy-style score over segmented cloud regions would trend upward simply
  because a region is densely sampled. **Prefer normalized (VMI/VKL-style) forms
  over raw-entropy forms whenever point density is uncontrolled**, which for a
  survey cloud with dropout it never is.

**Recommendation.** A per-region angular score with an analytical ceiling, built
on the VSE skeleton, min-taken over a symmetry-equivalence set, hard-gated at an
incidence threshold justified from sensor physics (the ~70° cliff as the outer
bound; the operating threshold to be set from Branch 1's depth-error-vs-angle
numbers). Report worst-case per region alongside the mean — a single grazing
evidence view is the failure Albert is complaining about, and a mean hides it.

### Q8 — View equivalence under symmetry

**Known.** Four incompatible answers, and only one is usable on the scoring side:

| Answer | Source | Mechanism | Usable for us? |
|---|---|---|---|
| Explicit equivalence class in the *scoring function* | `dutagaci` | Σ_v = {w : d(I_w, I_v) ≤ t}, image dissimilarity via rotation-invariant Fourier–Mellin matching on rendered depth images; score = min over Σ | **Yes** — the only scoring-side equivalence relation in the corpus |
| Brute-force rotation augmentation of *training data* | `ma-scvp-longtail` | 40 objects × 8 rotations (45° steps), so a learned predictor generalizes | No — helps a learned predictor, gives no equivalence relation for a rule-based scorer |
| Exclude ambiguous items from the dataset | `view-level-evidence-benchmark` | Annotators drop any sample where "no single view offers equally direct evidence" | **No — structurally unavailable to us.** Symmetric primitives are the common case, not a rare edge case |
| Loose unspecified answerable-viewpoint set | `e3vs-bench` | 𝒱_ans, plural, "valid but unannotated alternatives handled via reference comparison" | Weak — no equivalence relation, no reported set cardinality |

Everything else read (`gennbv`, `vin-nbv`, `objview-bench`, `secord`,
`planning-with-the-views`) has **zero symmetry handling**. `secord`'s authors
explicitly list object symmetry as an attribute they did not include and left to
future work. `planning-with-the-views` has a *point-identity* target (an exact
ScanNet camera pose) — a class-of-correct-views concept does not exist there.

**Transfers.** `dutagaci`'s recipe is directly implementable on our candidate
action space: render a depth/silhouette image from each candidate pose, cluster
near-identical ones under a low threshold, and score with min-over-cluster. And
its *validation* of the mechanism is a worked example we should copy: the vase
model has **very low human inconsistency despite geometrically diverse GT picks**
precisely because rotational symmetry was accounted for.

**Contested.** `bonaventura`'s Figure 1 shows human picks are not only
symmetry-spread but genuinely **bimodal** on at least one object (the cow: a
dominant cluster plus a distinct smaller cluster from below). **Symmetry
equivalence and multi-modality are different things** and a min-over-set score
handles the first but silently accepts the second. For a cube, the six faces are
interchangeable under relabelling but they are not interchangeable *as answers to
"inspect face 3"* — so our equivalence relation must be defined per-region, not
per-object.

**Missing.** No paper defines an equivalence relation over a *geometry-conditioned*
addressing scheme (Branch 5's territory). The interaction — does our addressing
scheme respect the equivalence relation, or does it break it by naming faces that
are physically indistinguishable? — is unexamined anywhere.

**Recommendation.** `dutagaci`'s Σ construction, computed per (object, region)
rather than per object, with the equivalence threshold reported as a benchmark
parameter. Ask Branch 5 explicitly whether its addressing scheme is invariant
under the equivalence relation, and treat a "no" as a benchmark-detectable defect
rather than a naming inconvenience.

### Q9 — Injected sensor defects: which, at what magnitude, and do they change *rankings*?

**Known.** The defect taxonomy is solved; the ranking question is not.

**The taxonomy** (`depthsynth` §3.1, Table 1) — eight cause-attributed
categories, not artifact shapes:
- **Axial** — pattern density falls quadratically with distance → range-dependent
  depth variance.
- **Lateral** — non-uniformity at depth discontinuities/edges.
- **Shadow** — objects obstructing the *emitter*, the *camera*, or both. Because
  projector and camera are separated by a baseline, each can be occluded
  independently → depth holes purely from viewing geometry.
- **Structural** — sensor spatial resolution / lens pattern warping at capture,
  plus the nonlinear z = f·b/d disparity-to-depth transform at reconstruction.
- **Quantization** — depth resolution coarsens quadratically with distance
  (integer disparity, inverse relation).
- **Band** — SAD-window bias during block correlation.
- **Specular / non-specular surface response** — pattern reflected too strongly
  or too weakly.
- **Lens distortion, grain, motion blur, rolling shutter.**

Table 1 scores three simulators across these categories and is **directly
citable as the argument that ad-hoc Gaussian jitter is a known-inferior defect
model** — the rhetorical move Q9 needs.

**The pose-error sweep template** (`pose-noise-robustness-plant-vp` §IV-D):
zero-mean Gaussian on camera position and orientation, three levels
**(σ_t, σ_r) ∈ {(0.002 m, 0.5°), (0.005, 1.0), (0.01, 2.0)}**, applied only to
actively executed views, reported with and without a lightweight-ICP mitigation.
The results shape matters more than the numbers:

| σ_t / σ_r | ICP | Chamfer (mm) | F@0.01 P/R/F₁ |
|---|---|---|---|
| 0 / 0 | — | 5.44 ± 0.13 | 100.0 / 87.3 / 93.2 |
| 2 mm / 0.5° | off | 7.69 ± 0.15 | **57.8** / 91.8 / 70.9 |
| 5 mm / 1.0° | off | 9.89 ± 0.27 | **40.0** / 93.1 / 56.0 |
| 10 mm / 2.0° | off | 13.87 ± 0.47 | **27.3** / 92.2 / 42.1 |
| 10 mm / 2.0° | **on** | 6.08 ± 0.16 | 98.7 / 85.0 / 91.4 |

**Precision at the tight threshold collapses 100% → 27.3% while recall stays flat
at ~92%.** Pose noise adds spurious/misaligned points without removing coverage.
Any metric that only counts "was this region seen" is blind to the entire effect.
And ICP recovers essentially all of it — which means **the presence or absence of
a registration-refinement step, not the noise level, is the dominant variable.**

**Severity calibration** (`embodied-iqa` §4, Fig. 7): bin severity into
Mild/Medium/Severe by **tertiles of the downstream models' own score
distribution**, not by physically-uniform corruption magnitude, because different
corruptions have wildly different just-noticeable-differences ("Dis16 causes
serious distortion at level 1; Dis25 has no significant effect even at level 5").
Caveat from the reader: their binning is *retrospective*, defined after
collecting scores — it characterizes sensitivity, it is not a protocol for
choosing magnitudes in advance.

**Evidence that defect injection changes rankings — from an adjacent domain.**
`pbr-structured-light-sim` Table I: training on physically-defect-injected
synthetic depth transfers better to real than training on **clean** synthetic
depth, for 4 of 5 objects, and usually better than RGB. Metal Workpiece 1
detection sim2real gap: RGB 0.077, clean GT depth 0.086, **defect-injected depth
0.017**. Clean depth is consistently the *worst* transferer. This is an existence
proof that the clean/defect-injected choice flips which input "wins."

**Contested / missing — the actual Q9 question is unanswered.**
`pose-noise-robustness-plant-vp` is **single-method**: Table VII reports only
"Ours" with/without ICP at each noise level; no baseline is re-run under noise.
It cannot report ranking stability and does not claim to. `objview-bench` *does*
do ranking-under-constraint analysis but the constraint is reachability and
budget, not sensor defects. So: **the "does the ranking change under injected
defects" experiment does not exist in the read corpus for view planning. It
would be original.** `objview-bench` is nonetheless a strong methodological
precedent for the *shape* — it demonstrated that asking "does the ranking change
under constraint X" is the productive question, and that some method classes fail
catastrophically rather than gracefully.

**Two of our three named defects have no precedent at all.**
`pose-noise-robustness-plant-vp` injects **only** pose noise — no dropout, no
fusion offset. Its "sensor" is a ray-casting oracle (OctoMap), with **no depth-sensor
noise model whatsoever**, and its intrinsics are copied from a **D435, not a
D405**. `depthsynth` requires a textured CAD model we do not have.
`pbr-structured-light-sim` is the wrong sensor modality (§1, correction 2), and
its own residual sim2real gap is **largest precisely on the glossy object**
(YellowSaltCube retains 0.137–0.173 AP gap after defect injection vs. ~0.02–0.05
for the other four) — i.e. **simulating dropout on glossy/dark surfaces is the
case that even physically-based simulation has not closed.**

**Recommendation.** Build the injected-defect list from `depthsynth`'s taxonomy
(so it is citable, not ad hoc), sweep pose error at
`pose-noise-robustness-plant-vp`'s three levels (our stated "few mm" sits inside
that range), and — the finding to hunt — **report method *rankings* per defect
condition, not just scores.** Report the registration/ICP step as an explicit
condition, since it dominates. Pre-register that a rank flip is the result and a
uniform score drop is not. Read arXiv:2201.11924 before attempting dropout
simulation.

### Q10 — Is the ~4 cm fused-vs-single-view offset a condition or a subject?

**Known.** Almost nothing. This is the branch's largest unfilled hole and the
round-2 map's own assessment ("the highest-risk unexamined assumption in the
whole evaluation design") is confirmed rather than resolved.

What exists:
- **`ma-scvp-longtail`'s neighbour-tolerance GT** — the only mechanism in the
  corpus for absorbing registration offset into a coverage metric, and its
  authors state the motivation directly: "slight calibration errors may result in
  a small offset in the point cloud for each view. To address this issue, for a
  voxel in the ground truth model, we search for its corresponding neighbors."
  **Validated at sub-2 mm. Ours is ~2000× the GT voxel size.** The mechanism
  transfers; the magnitude does not.
- **`pbr-structured-light-sim`'s sim2real-gap-as-a-column** — report the *same*
  metric under two conditions and treat the delta as a first-class number rather
  than folding it into one score. Directly operationalizes Q10: report the
  view-quality score computed in fused-cloud space **and** in execution space, and
  report the gap as its own row.
- **`binomial-eval-statistics`'s finding that gap magnitude is condition-dependent,
  not a constant** — changing only which initial conditions were included flipped
  their real–sim correlation from ρ=0.59 to ρ=−0.05 for the same policy and task.
  Their conclusion transfers directly: **treat the offset as an
  empirically-measured-per-condition benchmark variable, not a fixed constant.**
- **`depthsynth` cannot help**: it is single-shot-per-pose with no multi-view
  fusion stage at all, so it has no analogue to a fused-vs-single-view offset.

**Contested.** Nothing, because nobody has taken a position. The question is
open.

**Missing.** Everything. No paper scores plans in both fused-cloud coordinates
and execution coordinates. No paper treats a registration offset of this
magnitude as a planning-relevant quantity. The mapped-but-unread
`nbv-under-positioning-error` (the classical Scott-lineage entry point) and
`motion-uncertainty-nbv` are the remaining leads.

**Recommendation.** Make it a **subject**, reported as an explicit gap row:
score every candidate in fused-cloud space *and* in execution space, and report
the delta per method. If a method's advantage exists only in fused-cloud space,
that is a finding — and it is exactly the finding that would invalidate a
benchmark that scored only in fused-cloud space. Cheap to add, and it is the one
place where the benchmark can catch its own most dangerous assumption.

### Q11 — Object set and difficulty stratification

**Known.** Three reusable stratification mechanisms, none matching our primitive
taxonomy:
- **`objview-bench`**: difficulty split into three independently-computed
  quantities — omnidirectional self-occlusion ratio (a *ceiling*, not a planning
  outcome), observation-saturation difficulty (marginal-gain curve knee), and
  planning difficulty as a **Set Cover ILP over a fixed candidate protocol**
  (explicitly protocol-dependent, not object-intrinsic). Plus a **two-stage
  difficulty-aware sampling rule**: cover every non-empty difficulty bucket
  first, then allocate remaining quota proportional to natural frequency.
- **`ma-scvp-longtail`**: bucket objects by SCOP minimum-cover-set size into
  simple (4–7 views) / medium (8–12) / complex (13–16), and report per bucket.
  Their result is instructive: the extra online view helps **complex** cases
  (95.87→96.84%) while **simple cases saturate at ~99.9% regardless of method** —
  a pooled average would have hidden the entire effect.
- **`ycb-protocols`**: five categories including an explicit **"shape items"**
  bucket of primitives (spheres at 8 sizes, a 10-piece stacking cup set, foam
  bricks, dice, washers at 7 sizes), plus a selection-criteria checklist
  (variety of shape/size/texture/deformability, explicit difficulty spread,
  durability, cost, portability). Caveat: **their difficulty spread is asserted
  in prose, never measured or validated** on the released set.

**Transfers.** `benchmark-lottery` supplies the audit that makes stratification
mandatory rather than nice-to-have: recompute the aggregate ranking under many
subsets of the evaluation set and check whether the winner changes. Their
numbers — SuperGLUE, 55 models × 8 tasks, ~60 of 70 four-task subsets disagreeing
at top-3/top-5; VTAB, **12 distinct top-1 winners among 19 single-task subsets** —
say that ranking instability under subset choice is the normal case, not the
exception. `dutagaci`'s Table 2 makes the same point in our own domain: **no
single algorithm dominates across the 68 objects; the win-count column sums to 68
with seven different winners.**

**Contested.** `objview-bench`'s difficulty axes are all about self-occlusion and
saturation — quantities that are near-trivial for primitives. Our difficulty axis
is different in kind: **fitting ambiguity** (is this object cleanly a primitive?)
and **symmetry degeneracy** (is the correct view a continuum?). Neither has
precedent.

**Missing.** No object set with analytic primitive labels was read (`abc-cad-dataset`
unread). No near-primitive/ambiguous-fitting test case exists anywhere in the
corpus. `secord`'s only relevant data point is incidental: **the "city" model —
their boxiest, most planar, most multi-part validation object — fit worst of the
four held-out models** (t=45.0 vs 57–71 for organic ones), which the authors
attribute to its qualitative difference from the training set. Weak, incidental
evidence that a universal geometric view-quality model degrades on box-like
structure — mildly supportive of per-primitive rules, not a designed test.

**Recommendation.** Pure primitives (box, cylinder, sphere, cone) / compounds of
2–3 primitives / at least one near-primitive where fitting is ambiguous, with
`objview-bench`'s two-stage bucket-covering sampling rule and per-stratum
reporting, never a pooled headline. The near-primitive case is where the uniform
sphere fallback should trigger, and **the benchmark must be able to detect that
it correctly did** — which makes "correctly abstained / correctly fell back" a
scored outcome, not a footnote (see Q16, abstention).

### Q12 — Statistical treatment

**Known.** The field's practice and the methodology literature's bar are far
apart (F9). Concretely available tools, ordered by fit to our regime:

1. **`ma-scvp-longtail`'s paired protocol** — every method evaluated on the
   *identical* case set (30 objects × 5 rotations × 20 initial views = 3,000
   matched cases), compared with **paired t-test AND Wilcoxon signed-rank**
   (both, "due to some non-normally distributed data"), α=0.05, significance
   stars per pairwise comparison. The reader's key observation: **this scales
   down to small N fine, because it is about pairing, not sample size.**
2. **`secord`'s resampling + Tukey HSD** — 100 repeated random half/half
   object-level splits, mean ± σ reported, and a **"pool of statistically
   indistinguishable top performers"** identified rather than a single winner.
   Designed for exactly our problem: few objects, noisy comparisons.
3. **`binomial-eval-statistics`'s WSR intervals** — finite-sample exact coverage
   for **bounded non-binary** metrics, which is what a partial-credit
   view-quality score is. Strictly better-fitting than Clopper–Pearson, which is
   Bernoulli-only. Plus the **"trials saved" reporting framing** (compute the
   competitor's interval at fixed n, binary-search the n the baseline needs to
   match it) — far more decision-relevant for a resource-constrained lab than raw
   interval width.
4. **`benchmark-lottery` §7.2** — Wilcoxon signed-rank (Demšar 2006) for "better
   on average across few datasets," per-dataset testing with multiple-testing
   correction (Dror 2017) for "better on all," and pre-specification of the test,
   the replicate count, the variance sources, and α *before* iterating.
5. **`view-level-evidence-benchmark`** — the small-N existence proof: mean / SD /
   95% CI over 5 repeated runs at N=122 items.

**Contested.** `benchmark-lottery`'s guidance assumes small-N *between* datasets
with each dataset itself large. **We are small at both levels** — few objects and
few trials per object. That compounded case is not solved by any source read; we
can borrow which test to reach for, but the required-N calculation is ours to
derive. `binomial-eval-statistics` gives the machinery but explicitly no n/α
values to reuse.

**The verified sizing number**, from the reader's own scipy check rather than
from any paper: a two-sided 90% Clopper–Pearson interval at **n=70** has width
**20.8 points at p̂=0.5** and **≈15.1–15.6 points at p̂≈0.85**. Going from a
10-point to a 2-point width costs **~24×** more trials (width 10.1 pts at n=150 →
≈2.0 pts at n≈3,600 at p̂=0.85). **Any success-rate claim at n≲100 is resolving
differences of roughly 15–20 percentage points, not 5.**

**Missing.** A minimum-detectable-effect derivation for our actual metric.
`measuring-what-matters` recommends "justify statistical power" and supplies no
formula or worked example. `mirrer` (unread) is the source for the ~20%
cross-lab reproduction variance that would argue for local relative ranking.

**Recommendation.** Paired design over matched cases (`ma-scvp-longtail`) as the
backbone, since it buys power without buying objects; Wilcoxon + paired t
together; `secord`'s repeated-split + indistinguishable-pool reporting instead of
a single winner; WSR intervals on any bounded score; and the "trials saved"
framing for any real-robot component. Report medians and variances, not means
alone. And decide the required effect size **before** running anything — if we
can only resolve 15-point differences, say so in the protocol.

### Q13 — What does a negative result look like, and is it reportable?

**Known.** The corpus is unusually rich here — four papers report clean negative
results with named mechanisms:
- **`ma-scvp-longtail`**: k=2 and k=3 are `n.s.` against k=1 while costing 20%
  and 40% more movement. "A stable plateau is observed." Reported as a finding.
- **`binomial-eval-statistics`**: at ρ=−0.051, **"none of our methods beat
  Classical,"** with the mechanism stated (rectifier variance 0.029 exceeded real
  variance 0.014) and a *pre-specifiable diagnostic* for when it will happen.
- **`objview-bench` §7**: catastrophic failure modes (NBV-Net under quarter
  space), scaling limits, and an unresolved gap (slow-saturation objects)
  reported as first-class lessons.
- **`ycb-protocols`**: every one of six worked benchmarks explains its failures
  in the same breath as its successes, and the benchmark template makes
  **"reasons for failed attempts" a mandatory submission field**, not an optional
  discussion.
- **`vin-nbv`** is the counter-example: on the Dinosaurs category VIN-NBV (0.04)
  is *worse* than GenNBV (0.03), which the authors do not flag — the reader
  surfaced it by reading Table 2 row-by-row. Their headline "~40% better than RL
  baselines" does not hold category-by-category.

**Transfers.** `benchmark-lottery` gives institutional cover (the EMNLP policy
that "not being SOTA" is not valid grounds for rejection; the Appendix G
checklist item on valuing papers that deviate from trend) and the concrete
mitigation for our N=1 situation: **pre-register the evaluation protocol —
metric, GT source, statistical test, defect conditions — before iterating the
method against it.**

**Contested.** Nothing. This is the least contested question in the branch.

**Recommendation.** State in the protocol, before running: the geometry-conditioned
action space loses if, at matched budget and under the permanent uniform control,
its per-region angular score is not better by more than the effect size we can
resolve (Q12), or if its advantage vanishes in execution space (Q10), or if the
ranking does not survive defect injection (Q9). Commit to reporting each. Adopt
`ycb-protocols`'s mandatory failure-narrative field in our own reporting template.

### Q14 — Separating "the action space is better" from "the VLM used it better"

**Known.** Four distinct mechanisms, all instantiated, none quite ours:
- **`vin-nbv`'s fitness-swap**: Oracle-NBV / VIN-NBV / Cov-NBV — identical greedy
  loop, identical 120-view candidate set, only the scoring function differs. The
  cleanest instance of "hold search fixed, vary the signal."
- **`gennbv`'s cross-implementation swap**: "Scan-RL w/ Our Representation" ×
  "Ours w/ Scan-RL Representation" — a full 2×2 over (action space) ×
  (state representation). Directly transposable to (action space) × (agent).
- **`ma-scvp-longtail`'s submodule swap**: hold the one-shot network fixed, vary
  which NBV method supplies the single warm-up view. Their result is a useful
  negative: even **Random** as the NBV submodule gives 98.34% vs. PCNBV's 98.41%
  — the downstream stage was largely insensitive to the upstream one.
- **`e3vs-bench`'s VQA-at-goal / goal-view-initialization**: hand the model the
  right view and measure only the reasoning (3.58 vs. human 3.53); or start the
  agent at a sufficient viewpoint (2.79 → 3.41 at half the steps). Both isolate
  *finding* from *reasoning*, and both say finding is the bottleneck.
- **`view-level-evidence-benchmark`'s Oracle-QA**: same mechanic, opposite
  direction — supply the golden view, measure downstream answering.

**Transfers.** The 2×2 is the right shape: {uniform sphere, geometry-conditioned
space} × {oracle selector, VLM selector}. The oracle-selector cell measures
action-space quality with the agent removed; the VLM cell measures agent
competence given the space; the interaction is the confound Q14 names. Both
`objview-bench` (greedy oracle rollout: "select the candidate view with the
largest additional GT coverage at each step") and `vin-nbv` (true-RRI oracle)
supply implementable oracle mechanics.

**Contested.** `gennbv`'s reader notes the field's oracle baselines are often
"oracle-ish" and not held-out — `gennbv`'s Uncertainty-Guided and ActiveRMap
baselines are marked † because they were *trained and evaluated on the same test
objects*. An oracle that re-optimizes on the test object is a different construct
from an oracle that has GT access to a fixed quantity.

**Missing.** Nobody runs the full 2×2. `objview-bench` uses its oracle only as a
diagnostic ceiling for one difficulty subset, not as a controlled ablation across
all methods. `neu-nbv`'s fixed-renderer trick — the map's named pattern for this
— is unread.

**Recommendation.** Run the full 2×2 and report all four cells. It is the single
highest-value experiment the branch can specify, because without it a win is
uninterpretable: **a geometry-conditioned space with 8 well-named views may win
because it is smaller and more legible, not because its views are more
orthogonal.** The oracle-selector row settles that.

### Q15 — Grounded success versus a lucky answer

**Known.** Fully answered, with the strongest single number in the branch (F7).
`view-level-evidence-benchmark`'s 2×2 grid, four cells: grounded success
(view✓/answer✓) 55.9% pooled; right-view/wrong-answer 6.7%; **wrong-view/right-answer
26.2%**; total failure 11.2%. Worst case Qwen2.5-VL-7B: raw answer accuracy
78.7%, grounded accuracy 8.2%. Free-form answering exposes the problem rather
than hiding it behind option-guessing (309/732 grounded vs. 409/732 for MC, with
149 right-view/wrong-answer).

Corroboration from two independent directions: `e3vs-bench` self-reports the
failure without measuring it; `planning-with-the-views` finds the mechanism —
**≥90% of successes are wander-and-match, not prospective inference** — and its
Table 4 diagnostic ("did the agent visually pass through the target view before
submitting?") is the same logic applied to camera localization.

**Transfers.** Adopt the grid near-verbatim. Also adopt the three-number
reporting discipline (`Acc_view`, `Acc_oracle`, `Acc_joint`, never blended) and
the practice of publishing the raw counts, not just the derived rates —
`self-preference-bias-judge` makes the same point about reporting the confusion
matrix behind an aggregate, and notes it matters *more* at small N, not less.

**Contested.** Nothing. Both the map and every relevant reader agree.

**Missing.** The grid requires view-level labels, which requires Q2's GT to
exist. It is downstream of the analytical-GT decision.

**Recommendation.** Mandatory. Any end-to-end inspection number we report ships
with the grid, or it is not trustworthy on its own. Add a fifth cell for
abstention (see Q16).

### Q16 — Named VLM spatial failure modes as explicit test cases

**Known.** The corpus supplies six, each with numbers:

| Failure mode | Source | Evidence |
|---|---|---|
| **Multi-turn compositional collapse** | `planning-with-the-views` | Single-turn 20–53% → multi-turn 2.5–21.3%; more turn budget doesn't help |
| **Degradation with viewpoint distance, decomposed** | `planning-with-the-views` | Every model: IVP-Short ≫ IVP-Long (GPT-5.4 Pro 34.8 → 11.7). **Rotation distance drives single-step failure; position distance drives multi-turn failure** |
| **Dominant-view bias** | `view-level-evidence-benchmark` | Models over-select `CAM_FRONT` even when evidence is elsewhere; Claude front-view accuracy 96.59% vs **side-view 46.15%** |
| **Adjacent-region confusion** | `view-level-evidence-benchmark` | `CAM_FRONT` vs `CAM_FRONT_LEFT` when evidence spans a boundary |
| **Failure to abstain** | `view-level-evidence-benchmark` | On genuinely unanswerable items, abstention accuracy **16.7–33.3%** even for frontier models — "even strong proprietary models rarely abstain correctly" |
| **Search-and-match instead of prospective reasoning** | `planning-with-the-views` | ≥90% of successes involve visiting the target view first |

**Transfers.** Each becomes a named test case with its own reported number, per
`measuring-what-matters` Rec. 3.7 (error analysis decomposed by construct, and
check whether failures correlate with constructs the benchmark did *not* intend
to measure). The **abstention** failure is a genuine addition beyond the round-2
map's list, and it maps directly onto our near-primitive stratum: *does the agent
correctly signal "no good orthogonal view exists, fall back to the uniform
sphere," or does it confidently name a wrong face?*

**Contested.** `planning-with-the-views` finds rendering fidelity is **not** the
bottleneck — switching point-cloud rendering to Gaussian splatting changed
multi-turn success by only +0.2 to +1.9 points, and GPT-5.4 Pro **lost 14.4
points** on one single-turn task with the better renderer. `e3vs-bench` chose
3DGS specifically *because* fidelity preserves fine detail. Both cannot be
generally true; the resolution is probably task-dependent (fidelity matters for
fine-attribute disambiguation, not for spatial composition). Worth flagging
because it bears on how much effort our benchmark's rendering deserves.

**Missing.** Allocentric/perspective-taking degradation (`viewspatial-bench`,
unread) — and it matters, because "orthogonal to face 3" is an allocentric claim.

**Recommendation.** Six named test cases, each with its own number, none buried
in an aggregate. Add abstention as a scored outcome on the near-primitive
stratum. Mirror `planning-with-the-views`'s Short/Long distance split in our own
reporting since our action space has both rotation-dominant (orbit) and
translation-dominant (approach/retreat) moves.

### Q17 — Controls when a VLM or LLM does the scoring

**Known.** The corpus is mostly a catalogue of *missing* controls, which is
itself useful:
- `e3vs-bench`: single judge, no anonymization, no position swapping, no
  multi-judge consensus, no held-out human slice. Judge–human agreement
  **ρ=0.54**, self-described as "reasonable." And construction-time filtering and
  evaluation-time scoring use the same model family.
- `view-level-evidence-benchmark`: single fixed judge, **model identity never
  disclosed**, while one of the evaluated systems is Claude. Only control:
  "keep prompting and judge criteria fixed across models" — consistency, not bias
  control.
- `self-preference-bias-judge`: its only control is position-swapping (A/B
  averaged over two orderings). Its proposed mitigation (down-weight a model's
  evaluation when it shows low perplexity on the sample) is **untested — no
  experiment validates it.**
- `embodied-iqa`: **15 independent judge models agree with each other at only
  ~0.25–0.31 pairwise SRCC**, on a narrow, well-defined task. The authors'
  conclusion: "using only one model as the subject is far from sufficient... It
  is necessary to collect their general preference."

**Transfers.** A realistic bar, not an aspirational one: **near-perfect
judge agreement is not achievable and should not be the criterion.** The
concrete controls with support are (a) multiple independent judges and report
their mutual agreement as a number, (b) position swap, (c) versioned judge, (d) a
held-out human-labelled slice checked periodically for divergence, (e) publish
the raw counts behind any aggregate. `secord`'s AMT consistency filter is the
recipe for (d) if we collect human labels: show every pair twice, keep only
raters with ≥22/30 repeat agreement (p>0.99 under a guessing null) — 524 workers
produced 127,140 usable choices under that filter.

**Contested.** `e3vs-bench` supplies a soft counter-data-point on self-preference:
GPT-5.1 is both the construction-time judge and one of the evaluated agents, yet
scores mid-pack (2.42, below three Gemini variants). No visible score inflation
for the judge's own family. One paper, one setting — it does not rule out subtler
effects (e.g. inflation specifically on episodes the same model both filtered and
later solved, which the paper does not check).

**Missing.** `scoutvla`'s three-independent-judge design and `llm-judge-bias-audit`'s
three concrete audits are unread. The fuller control list is **not sourced from
`self-preference-bias-judge`** — the reader flags this explicitly and names the
follow-up reads (Panickssery et al. 2024 arXiv:2404.13076; Xu et al. 2024
arXiv:2402.11436; Stureborg et al. 2024 arXiv:2405.01724).

**Recommendation.** Prefer the judge-free path (`planning-with-the-views`,
`ycb-protocols`) for the primary metric. If a judge scores anything, apply the
five controls, report inter-judge agreement as a number, and **disclose the judge
model and version** — the one control every read benchmark either applied
trivially or failed to.

### Q18 — Cross-branch intake format

**Known.** Two ready-made schemas:
- **`ycb-protocols`'s paired Protocol/Benchmark template**: a versioned Reference
  No + Version, Authors, Institution, Contact on both documents; Protocol has
  five fields (Task Description / Setup / Robot-Hardware-Subject incl. **what
  prior information the system is given** / Procedure / Execution Constraints);
  Benchmark has three (Adopted Protocol by reference / Scoring, with explicit
  guidance to **avoid binary success-fail in favour of partial credit** / To
  Submit, with **"reasons for failed attempts" mandatory**).
- **`measuring-what-matters`'s phenomenon → task → metric decomposition**, with
  failure classified by where the chain breaks: phenomenon ill-defined or
  contested / task unrepresentative / metric measures something else / nobody
  checked.

**Transfers.** Merge them. `benchmark-lottery` adds the field neither has: a
**statistical-protocol field** (replicate count, variance sources, significance
test, α) per source, forcing that decision to be made rather than deferred.
`bonaventura` adds a second: tag each candidate metric by its **underlying
feature basis**, not its name, so that "five new candidate metrics" that are one
feature basis in five notations is visible at intake.

**Recommendation intake schema** (for the other five branches to fill):
`ref-no/version` · `phenomenon (what it claims to measure)` · `unit of
evaluation` · `metric + feature basis` · `dataset / object set` · `GT source and
provenance` · `baselines (incl. whether a random floor and an oracle ceiling are
present)` · `budget / stopping treatment` · `defects injected (which, magnitude)`
· `statistical protocol (replicates, variance sources, test, α)` · `survives our
constraints? (survey-cloud input / reachability / discrete selection)` ·
`circularity tag [CIRC] / [CIRC-FIX]` · `reasons-for-failure narrative`.

### Q19 — View quality or inspection outcome, and can we report both?

**Known.** `embodied-iqa` is the branch's designated answer and it is
unambiguous: **report each link of the proxy chain as a separate correlation,
never fuse them.** Their design reports (a) IQA-metric vs. Decision-score SRCC
and (b) Decision-score vs. real-Execution SRCC as two distinct numbers, and the
two gaps compound (0.58 and 0.6 respectively; Cognition→Execution <0.5).
`view-level-evidence-benchmark` gives the structural version: three separate
numbers plus the grid, no blending.

`self-preference-bias-judge` supplies the formal reason: their Demographic-Parity
bias variant, computed **without conditioning on ground-truth quality**, cannot
distinguish "the judge recognizes quality that happens to correlate with
self-similarity" from "the judge rewards self-similarity regardless of quality" —
and the authors caveat it accordingly. The Equal-Opportunity form, conditioned on
an independent quality signal, is the stronger formalization. **Our Q19 needs the
conditioned form.**

**Contested.** Whether reporting both contaminates either. The corpus says no,
provided they are never combined into a single score and provided the
relationship between them is reported as a *result*.

**Recommendation.** Report both, plus their rank correlation, and treat the
correlation as a headline benchmark finding rather than an assumption. **A low
correlation is the empirical version of the Q3 circularity check** and is the
single most informative number the benchmark can produce about its own validity.

### Q20 — What do we release, and does releasing it change what we build?

**Known.** `ycb-protocols`'s argument, from a survey of 33 prior object/task sets
in which nearly all failed on one column — *are the physical objects available?* —
is that **availability of the artifact, not description of it, is the
comparability bottleneck.** Translated by its reader: for us the equivalent
must-ship artifact is **the survey clouds**, because the defective fused cloud
*is* the input, not a clean CAD model. Also: version every protocol and benchmark
artifact with a reference number and host it somewhere citable and updatable.

`objview-bench`'s hidden-geometry server (GT mesh server-side, method sees only
rendered observations from poses it requests) is the contamination-free release
architecture, and it plans a public evaluation server and leaderboard.
`view-level-evidence-benchmark` ships **test-only, no training split**,
deliberately, "to reduce annotation leakage into evaluation."

**Transfers — and this one bites during development, not at release.**
`benchmark-lottery` §5's stateful-benchmark / adaptive-overfitting argument
applies **intra-team, across our own design iterations**: iterating the action
space against the same small object set reproduces the exact "researcher sees
prior errors, tunes accordingly" pattern. Mitigation: reserve some survey clouds
as a **design-iteration-untouched holdout**, distinct from the objects used while
iterating. Note the paper does not resolve whether Dwork's theoretical bound or
Recht/Roelofs's empirical reassurance is the right prior — it says check
case-by-case.

**Contested.** Nothing substantive.

**Recommendation.** Ship survey clouds + object set + versioned protocol +
evaluation script + the analytical GT construction code. Test-only split. Reserve
an untouched holdout from day one. This constrains what we build: the input
format must be shippable, the harness must be deterministic and seeded, and
the GT must be reconstructible by someone else from what we ship.

---

## 5. Contradictions ledger — kept open

Recorded as contradictions, not resolved into a position. Each is a place where
the branch's evidence points two ways and a design decision will have to pick a
side knowingly.

**C1 — The uniform sphere is both near-saturating and clearly beatable.**
Coverage metrics: `gennbv` Uniform Hemi 82.91% AUC (vs. 91.19% best, and equal to
the NeRF oracles); `ma-scvp-longtail` 98.41% VSC from a generic 32-view sphere;
`objview-bench` Random+TSP 0.996 NSC. Fidelity metrics: `vin-nbv` Uniform 0.41 cm
vs. VIN 0.20 cm. **The metric decides the answer.** Consequence: our choice of
primary metric partly determines whether the project's central claim can be
demonstrated at all — which is itself a Q3 circularity exposure we should name
rather than exploit.

**C2 — Chamfer's cross-method spread is smaller than our known error budget.**
`gennbv` Table 1: the entire spread across nine methods on OmniObject3D is
**0.33–0.44 cm**, i.e. ~1 mm. Our stated givens include a few-mm pose error and a
**~4 cm** fused-vs-single-view offset. Transplanting Chamfer-style accuracy as a
secondary metric plausibly puts the entire between-method signal below our noise
floor. Unresolved: whether a per-region, region-restricted fidelity metric
escapes this.

**C3 — Human best-view labels are the best-developed GT and probably the wrong
construct.** `dutagaci` + `secord` supply mature machinery; `dutagaci`'s reader
and `embodied-iqa`'s companion (human-vs-robot preference 0.13–0.21) say the
construct is recognition/legibility, not evidence quality. Unresolved: whether a
*task-specific* human study ("which of these two views is better evidence of this
defect") inherits the machinery without inheriting the construct mismatch.

**C4 — Geometric GT escapes the familiarity channel but not the
shared-assumption channel.** `self-preference-bias-judge`'s reader argues a
normal-vector computation has no perplexity substrate. But our normal comes from
a fit whose assumptions would also inform the judge and the selector; and
`dutagaci`'s seven algorithms clustering at VSE 0.396–0.517 with "none dominates"
is evidence that implementation diversity is not construct diversity. Unresolved:
what counts as an assumption-independent held-out score in a geometric setting.

**C5 — Rendering fidelity matters / doesn't matter.** `planning-with-the-views`:
point cloud → Gaussian splatting changes multi-turn success +0.2 to +1.9, and one
model *lost* 14.4 points on a single-turn task with better rendering. `e3vs-bench`:
chose 3DGS specifically because mesh-based simulators degrade fine detail.
Unresolved and consequential for how much our harness should spend on rendering.

**C6 — More views: monotone benefit / plateau / negative.** `ma-scvp-longtail`
k>1 n.s. at +20–40% cost; `vin-nbv` advantage front-loaded and converging;
`objview-bench` winner changes with budget; M2AD (unread) reportedly shows
negative returns. **Our view-efficiency metric must be able to express a negative
return**, which coverage-AUC cannot (it is monotone by construction).

**C7 — Statistical rigor: the field's own practice vs. the field's own
prescriptions.** Eight of the read papers report no variance at all; the
methodology papers put field compliance at 16%; and the two papers that do it
properly (`ma-scvp-longtail`, `secord`) did so by having 3,000 matched cases or
127,140 human choices. Unresolved: whether *any* small-N design can support a
significance claim at the effect sizes we expect, or whether we should abandon
significance claims for relative ranking against a permanent control.

**C8 — Symmetry: four incompatible mechanisms, one usable.** See Q8's table.
Unresolved: whether an image-similarity equivalence class (`dutagaci`) computed on
a *defective cloud's* rendered depth is stable — the whole mechanism depends on a
low dissimilarity threshold on rendered images, and dropout changes those images.

**C9 — The one paper that swept sensor defects cannot answer the question we
swept them for.** `pose-noise-robustness-plant-vp` is single-method by design.
Its most striking result (precision 100→27.3% while recall holds at ~92%) is also
the strongest argument that our metric family must be precision-sensitive — but
its ICP result says the *mitigation*, not the noise, dominates. Unresolved:
whether our benchmark scores the pipeline with or without registration
refinement, and whether that choice determines the outcome.

---

## 6. What is missing

**Missing from the corpus (would be original work):**
1. Any (object, surface-region) evaluation unit.
2. Any validated orthogonality-to-surface view score.
3. Any ranking-stability analysis under injected sensor defects for view planning.
4. Any treatment of a fused-vs-single-view offset at our magnitude.
5. Any object set with near-primitive/ambiguous-fitting difficulty.
6. Any benchmark combining per-region scoring + orthogonality + defect injection
   + symmetry handling + statistical rigor. Each exists somewhere; none together.

**Missing from this branch's reading (present in the map, unread):**
- **Group G entirely** — `m2ad`, `real-iad`, `real-iad-mvn`. The only entries that
  would ground "defects hide from the wrong viewing angle" in data. **Highest
  priority remaining read.**
- **`cpp-measurement-uncertainty`** (rated high, unread) — the closest published
  evaluation design to ours: viewpoint count + average measurement uncertainty as
  a joint cost, IK-feasibility and collision as **binary per-viewpoint
  indicators**, and an honestly reported trade-off (traversal time +18.64% while
  the baseline left 37.33% of features failing the uncertainty requirement).
- **`quoniam`** — the concrete third-party held-out metric implementation for Q3.
- **`tanks-and-temples` / `dtu-mvs-eval`** — the precision/recall-at-τ and
  median+variance conventions, currently second-hand.
- **`neu-nbv`** — the fixed-renderer confound isolation for Q14.
- **`mirrer`** — cross-lab ~20% variation → local relative ranking, the direct
  argument for making the uniform sphere a permanent control.
- **`scoutvla`** — three independent judges + the causal ablation of whether
  active perception actually yielded more answerable evidence.
- **`viewspatial-bench`** — allocentric degradation, which bounds what our
  addressing scheme may assume.
- **`abc-cad-dataset`** — analytic primitive labels for the object set.

**Missing from the search entirely (carried from round 2, still unexplored):**
- **Metrology standards** — VDI/VDE 2634, ISO 10360-13 acceptance testing of
  optical 3D measuring systems. Potentially the most rigorous available answer to
  Q2 and Q4 (artefact-based ground truth, instrument verification). Zero coverage.
- **Photogrammetric network design** — first/second/third-order design, criterion
  matrices, "Fraser network design," "simulation-based precision estimation." An
  analytical benchmark score that needs **no ground-truth reconstruction at all**.
  Zero coverage, and directly relevant to Q4's third exit.
- **Negative-result and pre-registration practice in robotics venues** — needed to
  make Q13 more than an intention.
- **`Karaszewski et al. 2016`** — comparative NBV benchmark across multiple 3D
  scanners and manipulators. Surfaced by the ISPRS survey, never read, and the
  single best lead for Q5 and Q12.
- **arXiv:2201.11924** — the correct active-stereo sensor simulation paper for
  D405 dropout.

**Cross-branch asks (explicit):**
- **Branch 3 → Q2:** the angular error of a fitted primitive's normal on our
  actual defect profile, with its distribution, not just a mean. Without it the
  analytical GT cannot be shown to referee.
- **Branch 1 → Q7:** the depth-error-vs-incidence-angle curve for the D405, to
  set the operating angular threshold (`depthsynth`'s ~70° is the outer physical
  bound, not the operating point).
- **Branch 5 → Q8:** whether the addressing scheme is invariant under the
  symmetry equivalence relation.
- **Branch 6 → Q16:** the failure modes to promote into named test cases, and
  whether allocentric addressing degrades measurably.

---

## 7. Implications per research question

### RQ1 — Judging a view

- **The judge's construct must be per-(surface region, view), not per-object.**
  Every existing view-quality measure in the corpus — all 22 in `bonaventura`,
  all 14 attributes in `secord` — is a whole-object single-viewpoint score with
  no region decomposition. Region-restriction (limit the polygon/point set to the
  region before scoring) is an extension nobody has performed or evaluated.
- **The judge should be geometric and learned-component-free if at all
  possible.** `planning-with-the-views` and `ycb-protocols` demonstrate whole
  benchmarks that never confront circularity because they chose constructs with
  no learned component. That is the strongest available Q3 position.
- **Keep it simple.** `secord`'s strongest empirical result is that a single
  ratio (surface visibility) beats every other individual attribute and dominates
  combined models, and that **more than five attributes stops helping**. Argues
  for one dominant angular term plus at most a couple of others.
- **Check for forced agreement before claiming independent evidence.**
  `bonaventura`'s I2 ≡ viewpoint-entropy proof is a cheap algebraic check that
  should be run on any two candidate orthogonality scores.
- **Prefer density-normalized forms.** 10 of 22 measures are discretization-sensitive;
  viewpoint entropy *diverges* with resolution. Point density in a survey cloud
  with dropout is never controlled, so raw-entropy-style forms will drift with
  dropout rather than with view quality.
- **Expect a broad plateau, not a peak, and build the sharpness into the GT.**
  `bonaventura`'s heatmaps show most measures produce a plateau over a third to a
  half of the sphere. An empirical scorer will not spontaneously find a
  face-aligned optimum; the geometry-aware GT construction has to supply it.
- **The judge's validity claim is a correlation, not an assertion.** Per
  `embodied-iqa`: report the rank correlation between the judge's score and an
  independently-constructed downstream outcome, and expect it to be well below
  0.9. Rotation/orientation is the hardest axis for any proxy to predict.

### RQ2 — Selecting best views

- **The set-cover formulation is the field's answer and it is well-specified.**
  `ma-scvp-longtail` (Gurobi ILP, <10 s per label) and
  `pose-noise-robustness-plant-vp` both use it; `objview-bench` uses it as a
  *difficulty measure* and stresses that it is **protocol-dependent, not
  object-intrinsic** — the same object gets a different planning difficulty under
  a different candidate-view set. Our benchmark must state its candidate set
  whenever it reports a set-cover-derived quantity.
- **Plan-length-as-stopping-rule is validated.** `ma-scvp-longtail`'s one-shot
  output size *is* the plan; no separate stop criterion is needed for that stage,
  and in practice it converges to 12–16 views without ever hitting the 32-view
  ceiling. This is the closest analogue to our analytical track and it works.
- **Do not calibrate a fixed budget from training-set statistics.**
  `ma-scvp-longtail` Appendix B-C is explicit: view counts "derived from training
  objects... acquiring this knowledge in advance for unknown testing objects is
  unrealistic," and their fixed-budget random baselines underperform auto-stop at
  matched view counts. Our budget number should come from geometry (set cover on
  the fitted primitives), not from a tuned constant.
- **Reachability must be inside the selection, not after it.**
  `objview-bench`'s 45.4/128-reachable statistic and NBV-Net's 0.988→0.472
  collapse together say that filtering after selection is a catastrophic-failure
  pattern. `ma-scvp-longtail`'s authors name the same gap as unsolved and their
  workaround was to move the whole view space by 8 mm.
- **The expected advantage is front-loaded and stratum-dependent.** `vin-nbv`:
  informed selection wins most in the early views. `ma-scvp-longtail`: simple
  objects saturate at ~99.9% regardless of method; the gain is in the complex
  stratum. **Design the experiment to look where the effect is, and report per
  stratum.**

### RQ3 — Composing the action space

- **A fixed, shared candidate set is the field norm and it is what makes
  comparison possible** — but the field has no consensus on its size (32 / 120 /
  128 / 258 / 360 / 642 across the corpus), and `bonaventura` silently changed
  Dutagaci's 258 to 642 without justification. **Whatever we choose, the density
  is a benchmark parameter that must be stated and held fixed across compared
  methods**, and a density change is a confound.
- **A naive az/el grid is the weaker uniform construction.**
  `pose-noise-robustness-plant-vp` and `ma-scvp-longtail` both use Tammes /
  spherical-codes packings; `objview-bench` uses Tammes-128/360. Beating our
  current 12×30°×3 grid is a weaker claim than beating a Tammes sphere at matched
  budget.
- **The action space must be able to express "unreachable, next best."** F5. This
  is the single strongest design constraint the branch produces for RQ3, and it
  comes with a measured failure magnitude.
- **Symmetry equivalence has to be a property of the addressing scheme, not a
  post-hoc scoring fix.** Only `dutagaci` gives a scoring-side equivalence
  relation; the alternatives (rotation augmentation, exclusion) are unavailable to
  us. If the addressing scheme names physically-indistinguishable views
  differently, the benchmark will penalize correct answers.
- **Small and legible is a confound, not just a virtue.** An 8-view named space
  may beat a 36-cell grid because it is easier, independent of view quality. Q14's
  2×2 is the only thing that separates them.

### RQ4 — Presenting the space to the VLM

- **Expect compositional failure and design the interface against it.**
  `planning-with-the-views`: frontier best is 21.3% on multi-turn view planning,
  ≥90% of successes are wander-and-match, and **more turns do not help**. If the
  VLM cannot compose, an interface that requires composition will fail regardless
  of how good the underlying space is.
- **Two concrete alternative interface shapes exist in the corpus, and they are
  not the same design.** `e3vs-bench` and `planning-with-the-views` both use
  small fixed *incremental* action sets (0.25–0.5 m translations, 30° rotations,
  camera-relative, geometry-blind) — closer to a keyboard than to a menu. Our
  design is *pick a labeled view from a pre-enumerated set*. E3VS's approach
  sidesteps the "how do I describe N labeled viewpoints" problem entirely by
  making the space small, fixed, and geometry-independent — which is exactly what
  our contract wants to move past. The two are different points on the same axis
  and the comparison is worth making explicitly.
- **Report a random floor in the same table.** `planning-with-the-views` shows
  random at 0.8% on multi-turn but **23–26% on 4-way multiple choice** — the floor
  depends entirely on the interface's branching factor, so any claimed VLM
  competence is uninterpretable without it. `e3vs-bench`'s random-action baseline
  (2.28) beats three of its nine tested models.
- **Abstention is a capability we must measure, not assume.** 16.7–33.3%
  abstention accuracy for frontier models means "no good orthogonal view exists,
  fall back" will not happen reliably unless the interface makes it easy and the
  benchmark scores it.
- **A "helpful-looking" default vantage can be worse than nothing.**
  `e3vs-bench`'s VQA-at-birdview (1.48) scored *below* VQA-at-start (2.14). If
  our interface offers an overview view as an aid, its value must be measured,
  not assumed.

---

## 8. Implications for the benchmark methodology

### 8.1 The shape the evidence supports

A benchmark with **two scoring layers that are reported separately and never
blended**:

**Layer 1 — view quality, per (object, surface region).**
Primary metric: a per-region angular score on the `dutagaci` VSE skeleton —
normalized to [0,1], minimum taken over a symmetry-equivalence set, averaged over
regions then over objects, **with worst-case-per-object reported alongside the
mean**. Computed judge-free from geometry. Ceiling = the analytical optimum
(known by construction on primitives); floor = a random view from the candidate
set; and — following `secord` — every candidate metric reported on a
**naive-floor / achievable-ceiling normalized scale**, where the ceiling is
empirical (GT's own uncertainty band) rather than a hypothetical 1.0.

**Layer 2 — inspection outcome, per (object, question) episode**, carrying the
view-level label, reported as `view-correct × outcome-correct` (the
`view-level-evidence-benchmark` grid) plus an abstention cell. Never reported as
a single end-to-end number.

**The bridge between them is a result, not an assumption**: report the rank
correlation between Layer 1 and Layer 2 (`embodied-iqa`'s design), and treat a
low correlation as the branch's most informative finding about its own validity.

### 8.2 Conditions to sweep

| Axis | Levels | Source |
|---|---|---|
| Budget | small fixed / large fixed / auto-stop with cap (MS@δ for methods with no native stop) | `objview-bench` |
| Action space | current 12×30°×3 az/el (permanent control) / budget-matched uniform / Tammes at matched budget / geometry-conditioned | `gennbv`, `plant-vp` |
| Selector | oracle (metric-optimal, VLM removed) / VLM | `vin-nbv`, `gennbv`, `e3vs-bench` |
| Reachability | unconstrained / IK+collision filtered | `objview-bench`, `cpp-measurement-uncertainty` (unread) |
| Sensor defects | clean / pose error at 3 levels / dropout / fusion offset — each with and without registration refinement | `plant-vp`, `depthsynth` |
| Scoring frame | fused-cloud space / execution space, with the **gap reported as its own row** | `pbr-structured-light-sim` |
| Object stratum | pure primitive / compound / near-primitive-ambiguous | `objview-bench`, `ma-scvp-longtail` |

The **finding to hunt in every sweep is a ranking change, not a score drop**
(`objview-bench`, `benchmark-lottery`). Report the full grid, name plateaus and
cliffs explicitly in the text (`ycb-protocols`).

### 8.3 Reporting and statistics

- Paired design over matched cases; paired t-test **and** Wilcoxon; α pre-specified
  (`ma-scvp-longtail`).
- Repeated random object-level splits with a **pool of statistically
  indistinguishable top performers** rather than a single winner (`secord`).
- WSR intervals on any bounded score; Clopper–Pearson only if the score is
  genuinely binary (`binomial-eval-statistics`).
- Medians and variances, not means alone; per-trial tables, not just aggregates
  (`ycb-protocols`).
- Raw counts published behind every aggregate rate
  (`self-preference-bias-judge`).
- **Per-stratum breakdowns always; no pooled headline number**
  (`benchmark-lottery`, `dutagaci`).
- A random floor and an oracle ceiling in every table.
- Mandatory reasons-for-failure narrative (`ycb-protocols`).
- Honest sizing statement: at n≲100 we resolve ~15–20 point differences, not 5.

### 8.4 Decisions that remain the architect's

The evidence narrows these but does not settle them. Listed as decisions, with
what each costs:

1. **Primary unit: per-region view quality, or per-episode outcome, or both.**
   Both is defensible and is what §8.1 assumes, but it roughly doubles the
   protocol and requires view-level labels to exist (Q2).
2. **Whether the analytical GT is allowed to referee.** Contingent on Branch 3's
   normal-error number. If the GT's angular uncertainty is not small relative to
   the method differences, the fallback is relative ranking against the permanent
   uniform control (`mirrer`'s position, unread) and no absolute claims.
3. **Coverage in or out.** Including it invites C1 (the sphere saturates and we
   lose on a metric that does not measure our construct); excluding it means no
   comparability with the entire NBV literature.
4. **The fused-vs-single-view offset: condition or subject.** §4/Q10 recommends
   subject. Making it a condition is cheaper and hides the project's most
   dangerous assumption.
5. **Simulation-first or real-capture-first.** Simulation is the only way to
   sweep defect magnitudes; real capture is the only way to make claims about the
   actual pipeline. `binomial-eval-statistics` gives the pre-registerable go/no-go
   test (σ²(Y−f) < σ²(Y)) for whether a cheap proxy is worth building at all.
6. **Whether a VLM judge scores anything.** The judge-free path exists and is
   cleaner; it costs the ability to score free-form inspection conclusions.
7. **How much of the survey-cloud set is reserved as a design-iteration-untouched
   holdout**, and enforced from day one rather than retrofitted.

### 8.5 Pre-registration items (write before running anything)

Per `benchmark-lottery` §7.2 and `measuring-what-matters` Rec. 3.1/3.6, decided
in advance and published with the protocol:

- The operational definition of "orthogonal enough," including what is excluded
  (self-occluded regions, regions below a minimum area, unreachable regions).
- The candidate view set and its density.
- The symmetry equivalence relation and its threshold.
- The budget regimes and the stopping rule.
- The defect conditions and magnitudes.
- The statistical test, the replicate count, the variance sources randomized,
  and α.
- **The losing condition** — the outcome under which the geometry-conditioned
  action space is reported as not better than the uniform sphere.
- Which candidate solutions our metric advantages, and why that is acceptable.

---

## 9. Reading queue for the next round

Ranked by what they unblock.

1. **`m2ad`** + **`real-iad-mvn`** — defect-visibility ground truth. Unblocks Q2's
   "different outcome" separation and grounds the orthogonality requirement in
   data. Group G is 0/3 and it is the branch's biggest evidential hole.
2. **`cpp-measurement-uncertainty`** — the closest published evaluation design to
   ours (viewpoint count + measurement uncertainty joint cost, binary IK and
   collision indicators, honest trade-off reporting). Rated high in round 2,
   never read.
3. **arXiv:2201.11924** (active stereovision sensor simulation) — the correct
   paper for D405 dropout, per correction 2.
4. **`Karaszewski et al. 2016`** — comparative NBV benchmark across scanners and
   manipulators. Best lead for Q5 and Q12.
5. **`quoniam`** — the third-party held-out metric implementation for Q3.
6. **VDI/VDE 2634 / ISO 10360-13** and **photogrammetric network design /
   criterion matrices** — the two entirely unsearched lines, both aimed squarely
   at Q2 and Q4, the second offering an analytical benchmark score requiring no
   ground-truth reconstruction.
7. **`neu-nbv`** (fixed-renderer confound isolation, Q14), **`mirrer`** (local
   relative ranking, Q12), **`scoutvla`** (three judges + causal ablation, Q17),
   **`tanks-and-temples`/`dtu-mvs-eval`** (metric conventions, Q4).
</content>
</invoke>
