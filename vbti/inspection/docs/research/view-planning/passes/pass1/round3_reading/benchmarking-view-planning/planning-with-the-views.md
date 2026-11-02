# Planning with the Views (ViewSuite)

- arXiv: 2605.29563 (v4, HTML full text read: https://arxiv.org/html/2605.29563v4; abs:
  https://arxiv.org/abs/2605.29563)
- Submitted 2026-05-28, latest revision (v4) 2026-07-02.
- Authors: Kangrui Wang¹, Linjie Li², Zhengyuan Yang³, Shiqi Chen⁴, Zihan Wang¹, Li Fei-Fei⁵,
  Jiajun Wu⁵, Leonidas Guibas⁵, Lijuan Wang³, Manling Li¹ — ¹Northwestern, ²U. Washington,
  ³Microsoft, ⁴Oxford, ⁵Stanford.
- Venue: not stated in fetched text (arXiv preprint).

## Core idea

The paper defines **View Planning**: localizing an unseen target view of a 3D scene by composing
a sequence of discrete camera-move actions, then asks whether frontier VLMs can do it. It factors
the ability into two coupled parts (§1, verbatim): "_tracking_ how a path changes the view, and
actively _composing_ a path to find where an unseen target view was taken." The benchmark,
**ViewSuite**, is built on 286 real ScanNet indoor scenes with full 6-DoF camera control and three
tasks (§2.1, Fig. 1):

1. **Path-to-View (P2V)** — given a start view + action sequence, predict the resulting view
   (4-way multiple choice).
2. **View-to-Path (V2P)** — given two views, infer the action sequence between them (4-way MC).
3. **Interactive View Planning (IVP)** — multi-turn: agent issues actions one at a time, observing
   the rendered view after each, and must submit a final camera-pose estimate that localizes an
   unseen target view.

The paper's headline finding (abstract, verbatim): "a sharp planning gap emerges: models track
local view transitions but collapse when they must plan toward an unseen target view." Best
frontier IVP score is 21.3% (Gemini 3.1 Pro); base Qwen2.5-VL-7B scores 2.5%. The fix — **iterative
self-exploration + view graph distillation** — is a training method (PPO self-exploration under
sparse +1/0 reward, +0.1 format bonus; every trajectory's edges, successful or not, are compressed
into a directed "view graph"; graph paths are reformulated via an operator ℛ, Eq. 4 §4.2, into
supervised IVP demonstrations with arbitrary start/end nodes; alternates RL and SFT-on-distilled-
graph over 4 iterations). This lifts Qwen2.5-VL-7B from 2.5% to 47.8% IVP, "surpassing GPT-5.4 Pro
(19.9%) and Gemini 3.1 Pro (21.3%)" (abstract). This training method is orthogonal to our RQs (we
are not training a VLM) — **the load-bearing part for us is the benchmark design and the failure
analysis, not the distillation method.**

## Task/action-space definition (§2.2, Appendix A.1)

- **Discrete 12-action interface**: 6 translations (0.5 m step: forward/back/left/right/up/down) +
  6 rotations (30° step: turn_left/turn_right/look_up/look_down/rotate_cw/rotate_ccw).
- Rationale (§2.2, verbatim): "models control camera by selecting actions rather than specifying
  precise motion parameters, while each action still produces a visibly distinct view."
- Action space is decoupled from the 3D backend — agnostic to point cloud vs. mesh vs. simulator
  (§2.2). Deterministic transition: p_{t+1} = T(p_t, a_t) (§4.1) — no stochastic execution error
  modeled anywhere.
- Rendering: Open3D point-cloud rendering is the default (§2.2); a Gaussian-Splatting alternative
  (SceneSplat-7K reconstructions) is tested as a rendering-fidelity robustness check (§3.2, Table 3).

## Ground truth and success metric (§2.3, Eq. 1–3; Appendix A.3)

- GT camera poses come directly from ScanNet video-frame extrinsics (6-DoF camera-to-world:
  position t ∈ ℝ³, rotation R ∈ SO(3)) — **real, not synthetic, ground truth**, but it is
  camera-pose ground truth, not object-surface/normal ground truth (search for "orthogonal view",
  "surface normal", "inspection", "primitive shape" in the full text returns **zero hits** —
  confirmed directly, this paper has no notion of surface-relative view quality at all).
- **IVP success (Eq. 3)** requires BOTH: translation error d_pos ≤ β_t·s_t and rotation error
  d_rot ≤ β_r·s_r, with step sizes s_t = 0.5 m, s_r = 30° matching the action granularity, and
  β_t = β_r = 1 (i.e., success = within one action-step of GT pose in both position and rotation).
  d_pos = ‖t₁−t₂‖₂; d_rot = arccos((tr(R₁ᵀR₂)−1)/2) (Eq. 1, §2.3).
- **Threshold calibration is empirical, not assumed** (Appendix A.3, Table 9): β_t, β_r were swept
  and the F₁-score against **human binary judgments** of "do these two rendered views depict the
  same place" was maximized; β_t=β_r=1 was the optimum found. Sample size/annotation protocol for
  the human study are not given in the fetched text.
- **Unified view distance** (Eq. 2, §2.3): d = √((d_pos/s_t)² + (d_rot/s_r)²) — approximates the
  shortest-action-plan length between two views. Test pairs are split into **Short (d<3, 185
  pairs)** and **Long (d≥3, 345 pairs)**; full range 1.4–6.8, mean 3.7 (Appendix A.4, Fig. 6).
- **No LLM-as-judge anywhere in evaluation.** Success is computed purely geometrically from GT
  pose distances; the only place human judgment enters is the one-shot threshold-calibration study,
  not per-instance scoring. No mention of multiple judges, position swapping, anonymization, or a
  held-out human-labelled slice re-checked for divergence during the actual benchmark runs (as
  opposed to the threshold-calibration step).

## Dataset (§2.3, Appendix A.2)

- 286 ScanNet scenes → ~55K view pairs sampled from video frames → ~165K task instances total
  (ViewSuite-5K released now, ViewSuite-50K reserved for future work).
- Train/dev/test split 8:1:1 **by scene, no scene overlap**.
- Test set: 530 view pairs × 3 tasks = **1,590 instances**, evaluated once (no repeated trials, no
  confidence intervals or significance tests anywhere in the main tables).
- Filtering pipeline (Appendix A.2): frame sampling → action planning between frames → distractor
  generation → scene-level filtering (criteria unspecified in fetched text).
- Training: Qwen2.5-VL-7B / Qwen3-VL-8B trained on 3,419 ViewSuite training environments, 4
  iterations (3 iterations = 60-step RL + 3-epoch SFT on distilled graph; final iteration = RL to
  convergence, no SFT) (§5.1).

## Results (Table 2, Table 5 — full numbers)

**Table 2 (13 VLMs + random baseline, held-out test set, percentages):**

| Model | P2V All | V2P All | IVP Short | IVP Long | IVP All |
|---|---|---|---|---|---|
| Random Response | 23.3 | 25.7 | 2.2 | 0.0 | 0.8 |
| GPT-5.4 Pro | 53.2 | 50.6 | 34.8 | 11.7 | **19.9** |
| Gemini 3.1 Pro | 48.9 | 49.4 | 28.6 | 17.4 | **21.3** (best frontier) |
| GPT-5.4 | 47.9 | 45.5 | 33.5 | 7.5 | 16.6 |
| Grok 4.20 Beta | 46.2 | 44.5 | 17.3 | 2.9 | 7.9 |
| GPT-5.1 | 44.0 | 40.0 | 11.9 | 3.2 | 6.2 |
| Claude Opus 4.6 | 34.7 | 41.5 | 23.8 | 3.8 | 10.8 |
| Gemini 3 Pro | 37.7 | 38.7 | 13.5 | 7.0 | 9.2 |
| Qwen3.5-397B | 39.8 | 35.7 | 12.4 | 0.0 | 4.3 |
| GLM-4.6V | 27.8 | 30.2 | 9.2 | 1.2 | 4.0 |
| Qwen2.5-VL-72B | 28.9 | 32.1 | 2.2 | 0.6 | 1.1 |
| Qwen3-VL-32B | 27.4 | 33.0 | 4.3 | 0.0 | 1.5 |
| Kimi K2.5 | 28.7 | 25.5 | 4.9 | 1.2 | 2.5 |
| Qwen2.5-VL-7B (base) | 29.4 | 24.3 | 7.0 | 0.0 | **2.5** |

Note the collapse pattern: P2V/V2P (single-turn, local) sit at 20–53%, well above the 23–26%
random-choice floor for the 4-way MC tasks; IVP (multi-turn, compositional) collapses to
single-digit-to-low-20s%, and **every model's IVP-Long score is far below its IVP-Short score**
(e.g. GPT-5.4 Pro: 34.8%→11.7%, a ~3× drop; Qwen3.5-397B: 12.4%→0.0%). This is the "composition
breaks down as viewpoint distance grows" finding the branch brief names — it is visible in every
row of Table 2, not just an aggregate claim.

**Table 5 (training ablations, Qwen2.5-VL-7B backbone, IVP %):**

| Method | Short | Long | All |
|---|---|---|---|
| Base model | 7.0 | 0.0 | 2.5 |
| Direct PPO | 7.0 | 1.2 | 3.2 |
| Direct GRPO (filter) | 10.8 | 2.2 | 5.2 |
| Success-Only Bootstrapping | 14.0 | 2.0 | 6.2 |
| Random-graph ablation | 25.4 | 6.4 | 13.0 |
| 1 iter + RL | 24.3 | 5.4 | 12.0 |
| 2 iter + RL | 49.7 | 16.2 | 27.9 |
| **Full method (4 iter)** | **67.2** | **36.9** | **47.8** |

Sparse-reward RL alone barely moves the needle (2.5%→3.2%→5.2%→6.2% across three different direct-
RL/bootstrapping variants) — abstract's stated reason: "with success near 2.5%, reward is too
sparse for RL to bootstrap." The view-graph distillation is what produces the large jumps.

**Other findings:**
- **Turn-budget ablation (Table 3, §3.2)**: marginal gains going from 10→20→30 max turns;
  "diminishing returns suggest planning bottleneck, not exploration horizon" — more budget does not
  rescue frontier models, consistent with the failure being compositional/reasoning, not
  search-depth-limited.
- **Rendering fidelity (Table 3, §3.2)**: switching Open3D point-cloud rendering → Gaussian
  Splatting rendering changes IVP only +0.2 to +1.9 points; P2V/V2P changes are larger and mixed
  (e.g. GPT-5.4 Pro *loses* 14.4 points on V2P with better rendering) — rendering quality is not
  the bottleneck, and better rendering does not reliably help.
- **Success-mechanism check (Table 4, §3.2)**: ≥90% of successful IVP runs (up to 99.1% for
  Gemini 3.1 Pro) involve the agent visually passing through/matching the target view during
  exploration before submitting; the paper's own reading (§3.1, verbatim): models succeed "by
  moving until they happen to see the target view, then matching it, rather than by inferring
  where the target view lies beforehand." Genuine prospective inference (submitting correctly
  *without* having visited/matched the target view) accounts for ≤~10% of successes, even for the
  trained model.
- **Distance-dependent degradation mechanism (Fig. 2, Appendix B.2–B.3)**: Spearman-ρ factor
  analysis separates the two tasks' bottlenecks — **P2V/V2P degrade primarily with rotation
  distance** (GPT-5.4 Pro loses ~25 points across rotation bins), while **IVP success collapses
  primarily with position distance** (~7× drop for GPT-5.4 Pro). So "viewpoint distance" is not one
  scalar — rotation hurts single-step view prediction, translation/position hurts multi-turn
  compositional planning. This is a mechanistic decomposition of the "composition breaks down"
  headline finding, worth preserving if we ever cite the distance-degradation shape.
- **Transfer (Table 6, §5.3)**: the trained model, under identical post-training on P2V/V2P,
  beats the base model by 8–12 points on those tasks and ~10 points on an external benchmark
  (MindCube) — evidence the learned spatial prior generalizes somewhat beyond ViewSuite's own
  scenes, though this is about the *training* method's transfer, not the *benchmark's* portability.
- **Limitations (§7, Conclusion, verbatim)**: "We study static indoor scenes through a discrete
  12-action interface, and validate the framework on two 7–8B backbones; outdoor or dynamic
  environments, continuous control, and larger model scales are natural next steps. More
  fundamentally, our trained model still localizes largely by _approaching until the target view
  becomes observable_, but localizing a target view without ever seeing it, mentally inferring
  where it lies, remains open. Finally, our framework places little weight on explicit reasoning
  traces; at our model scale we found it hard to acquire effective explicit reasoning without
  external expert demonstrations, leaving reasoning-aware training as a further direction." — three
  explicit author-stated open problems: (1) indoor/static/discrete-action-only scope, (2) even the
  trained model doesn't do true prospective localization, it "explores toward observability," (3)
  no successful explicit-reasoning-trace training at this model scale.

## What transfers to our setting

- **The distance-decomposition methodology (Fig. 2 + Spearman-ρ factor analysis)** is a directly
  reusable technique for Q9/Q16: don't just report "performance drops with viewpoint distance" as
  one number — separate rotation-distance-driven degradation from position/translation-distance-
  driven degradation and report both. Our UR5e eye-in-hand setting has both kinds of viewpoint
  change (orbit = rotation-dominant, approach/retreat = translation-dominant); this paper's finding
  that different task shapes have different dominant failure axes is a hypothesis worth testing in
  our own multi-turn VLM orchestrator.
- **Threshold calibration against human judgment via F₁ maximization (Appendix A.3)** is a concrete,
  reusable recipe for Q2/Q17-adjacent work: when defining "close enough to correct," don't assume a
  threshold — sweep it and calibrate to what humans agree is "the same place" (their analogue of
  our "orthogonal enough"/"same view region"). This is a lightweight, one-shot calibration study
  design we could copy for calibrating our own "orthogonal enough" angular tolerance.
  **However**: their calibration is stated as a single optimum with no reported sample size, no
  inter-rater agreement, and no cross-validation — a Q17 negative example (a judge/threshold
  process without the audit trail we'd want), not a template to copy uncritically.
  they set for their evaluator suggest the same kind of empirical-calibration argument as a Q17
  worked example: they don't defend β=1 analytically, they defend it by matching human agreement.
- **View-Matching-Strategy diagnostic (Table 4: "did the agent visit the target view before
  submitting?")** is directly reusable for Q15 (grounded success vs. lucky answer): it is exactly
  the "view-correct × answer-correct grid" logic Q15 asks for, applied to camera localization
  instead of inspection Q&A. A system that "wanders until it happens to match" is the view-planning
  analogue of "wrong-view/right-answer inflates the metric" — evidence that success-rate alone,
  without a mechanism check, overstates a system's spatial reasoning. We should build an analogous
  "did the VLM actually reason to the orthogonal view, or did it get there by exhaustive/lucky
  search" diagnostic.
- **Turn-budget ablation showing diminishing returns (Table 3)** transfers as a check to run
  ourselves for Q6: before concluding a small fixed view budget is "unfair," verify (as this paper
  did) whether giving the VLM more turns/views actually helps — if gains saturate early, a
  small-fixed-budget regime is not just a resource constraint but the right regime to report in,
  because the model plateaus of its own accord.
- **Composition-collapses-under-distance as a general VLM property, independent of domain**: this
  paper is further evidence (alongside whatever cited it in the branch brief) that multi-turn
  spatial composition is a *known, general* frontier-VLM weakness, not something specific to
  inspection tasks — strengthens the case (Q16) that "multi-turn composition failure" and
  "distance-dependent degradation" deserve to be their own named benchmark test cases with their
  own reported numbers in our benchmark, exactly mirroring this paper's P2V/V2P/IVP × Short/Long
  breakdown structure.
- **Random-baseline-as-floor practice** (Random Response row in Table 2, always reported alongside
  every model) is a Q5-adjacent minimum-hygiene bar: any VLM-driven benchmark we build should
  likewise report a random/near-random baseline in the same table as the real methods, not as a
  footnote.

## What does NOT transfer, and why

- **No notion of surface-relative view quality at all** (confirmed by direct text search: zero
  hits for "orthogonal," "normal," "inspection," "primitive"). ViewSuite's ground truth is "the
  exact 6-DoF pose a specific ScanNet video frame was captured from" — a point-identity target, not
  a region/surface-relative target. This paper offers **no evidence at all** for Q2 (orthogonal-view
  GT on a primitive survives contact with a survey cloud) or Q7 (metric with a per-primitive
  analytical optimum) — it never faces that problem because its "correct view" is an arbitrary human
  camera position in a room, not a geometrically-motivated best view of an object surface.
- **Real depth data (ScanNet) but no defect injection or sensor-noise ablation of any kind** — no
  pose error, no dropout, no fused-vs-single-view offset, no synthetic-vs-real comparison anywhere
  in the fetched text. Negative evidence for Q9/Q10: this benchmark cannot and does not tell us
  whether method rankings change under sensor defects, because ScanNet scans are used as-is (single
  fixed reconstruction per scene, not re-injected with controllable defects). Contrast with VIN-NBV
  (round3 note in this same folder) which is fully synthetic/noise-free at the other extreme — this
  paper sits at "real sensor data, but no controlled defect axis," a third point on the space Q4/Q9
  are mapping, not a solution to either.
- **Unit of evaluation is per-episode pose-localization success (IVP) or per-instance MC accuracy
  (P2V/V2P), never per-(object, surface-region)** — directly answers Q1 in the negative: the
  rejected option here (region-scored evidence-view quality) is not just absent, it is
  structurally inapplicable, because ViewSuite scenes are whole rooms with a target *camera pose*,
  not objects with target *surfaces*. Confirms (alongside VIN-NBV) that neither major NBV/view-
  planning paper class we've read gives us a ready-made per-region unit of evaluation — we are
  extending past both, not adapting either.
- **No view-equivalence/symmetry handling (Q8)** — ScanNet target poses are singular ground-truth
  camera positions; there is no "class of correct views" concept (unlike a box face or cylinder
  orbit, where many poses are equally correct). Not applicable to their formulation, and offers no
  template for defining our equivalence relation.
- **Discrete 12-action interface is action-space-agnostic and NOT geometry-conditioned** — actions
  are fixed metric steps (0.5 m / 30°) in the camera's own egocentric frame, identical everywhere
  in every scene, with no relation to object/room geometry. This is architecturally the same
  "uniform, geometry-blind action space" our framing doc explicitly wants to move past (see
  /home/anton/projects/robotics/docs/research/geometry-aware-view-planning.md) — useful
  as a contrast case (their action space is uniform-and-that's-fine for room navigation; our
  problem statement is precisely that uniform bins are *not* fine for surface-orthogonal
  inspection) rather than as a design to copy.
- **No LLM-as-judge in the actual scoring loop** — so this paper contributes nothing to Q17's
  judge-control question (multiple judges, position swapping, anonymization, versioning) beyond the
  one-shot human-threshold-calibration precedent noted above; if RQ1's judge is LLM-based, this
  paper's clean-geometric-GT approach is actually the alternative to a judge, not an example of
  controlling one.
- **Scale and platform mismatch**: 286 whole indoor rooms/scenes, DoF-6 free camera flight, action
  step sizes tuned for room-scale navigation (0.5 m translations) — the opposite regime from
  tabletop centimeter-scale eye-in-hand D405 inspection of single objects. Use this paper for its
  *benchmark-structure and failure-analysis methodology* (Short/Long distance split, per-task
  MC-vs-multi-turn split, distance-decomposition-by-rotation-vs-position, view-matching-strategy
  diagnostic), not for its object set, its action-space granularity, or its numeric results as
  direct baselines.

## Citation map (for checkability)

| Claim | Section | Figure/Table/Eq |
|---|---|---|
| Task definitions (P2V/V2P/IVP) | §2.1 | Fig. 1 |
| Discrete 12-action interface, rendering | §2.2 | Appendix A.1 |
| GT pose distance metrics, success threshold | §2.3 | Eq. 1–3 |
| Unified distance metric, Short/Long split | §2.3 | Eq. 2, Fig. 6 (Appendix A.4) |
| Threshold calibration (β_t=β_r=1) vs. human judgment | §2.3 | Appendix A.3, Table 9 |
| Dataset scale, filtering pipeline | §2.3 | Appendix A.2 |
| Main VLM comparison (13 models + random) | §3.1 | Table 2 |
| View-graph distillation method, reformulation operator ℛ | §4.2 | Eq. 4 |
| Iterative self-exploration loop | §4.1–4.2 | Fig. 3 |
| Turn-budget ablation, rendering-fidelity check | §3.2 | Table 3 |
| Distance-dependent degradation (rotation vs. position) | §3.2 | Fig. 2, Appendix B.2–B.3 |
| View-matching-strategy diagnostic | §3.1, §3.2 | Table 4 |
| Training ablations (PPO/GRPO/bootstrap/graph) | §5.1–5.2 | Table 5 |
| Point-cloud/target-coverage over training | §5.3 | Fig. 4 |
| Transfer to P2V/V2P/MindCube | §5.3 | Table 6 |
| Limitations / open problems | §7 | — |

## Open gaps from this pass (not recoverable from HTML fetch alone)

- Sample size and annotation protocol for the human threshold-calibration study (Appendix A.3) —
  only the optimal β values and the F₁-maximization method are stated in the fetched text.
- Exact scene-level filtering criteria (Appendix A.2) beyond the pipeline stage names.
- Whether any of the 286 ScanNet scenes contain tabletop-scale objects specifically (vs. only
  room-scale navigation) — not addressed, would need the dataset release/appendix scene list.
- Full appendix (B.1 and beyond) was not fetched in detail; only the Spearman-ρ factor-analysis
  results cited in §3.2 were extracted. Worth a follow-up fetch of Appendix B in full if the
  distance-decomposition mechanism becomes load-bearing for our own failure-mode taxonomy (Q16).
