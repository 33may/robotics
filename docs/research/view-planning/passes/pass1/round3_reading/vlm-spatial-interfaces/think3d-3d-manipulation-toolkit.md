# Think3D: Thinking with Space for Spatial Reasoning

- arXiv: 2601.13029v3 (17 Mar 2026 revision; cs.CV), "Think3D: Thinking with Space
  for Spatial Reasoning"
- Authors: Zaibin Zhang, Yuhan Wu, Lianjie Jia, Yifan Wang, Zhongbo Zhang, Yijiang
  Li, Binghao Ran, Fuxi Zhang, Zhuohan Sun, Zhenfei Yin, Lijun Wang, Huchuan Lu
  (Dalian University of Technology / UCSD / Oxford). Code: github.com/zhangzaibin/spagent
- Read via: arXiv HTML full text (all sections + appendix), plus direct PDF render
  of Figures 8–14 (the actual system/tool/continuation prompts — these are images
  in the HTML render and were not otherwise extractable as text)

## Core method

Think3D equips a VLM agent with a **3D Manipulation Toolkit** (reconstruction +
transformation + rendering) so it can run an iterative **observe → manipulate →
reflect** loop over a reconstructed point cloud instead of reasoning over static
2D input images (Sec. 3.1, Fig. 2).

**Pipeline (Sec. 3.2):**
1. **3D Reconstruction.** Given multi-view images `{I_t}`, the agent optionally
   (binary decision `r ∈ {0,1}`, called once per query at most) invokes **Pi3**
   [Wang et al. 2025] to estimate per-view camera poses `C_t=(K_t,R_t,t_t)` and a
   fused colored point cloud `𝒳 = {(x_n, c_n)}` (Eq. 1–2). "Depth and confidence
   predictions are fused across views to obtain a **clean** colored point cloud"
   (Sec. 3.2, verbatim) — no discussion anywhere of holes, sparsity, or defects;
   "clean" is asserted, not measured.
2. **3D Transformation — the discrete-ish action space.** At each iteration k the
   agent predicts a 4-tuple (Eq. 7):
   `a_k = (i_k, m_k, Δα_k, Δβ_k)`
   - `i_k ∈ {1,...,T}`: a **1-based discrete anchor-camera index** selecting one
     of the original input views as the rotation reference (`rotation_reference_camera`
     in the tool schema).
   - `m_k ∈ {global, ego}`: a **binary mode toggle**. `global`: rotate (and
     optionally scale, `s=1` by default) the *scene* around its centroid while the
     camera stays fixed (Eq. 3) — "keeping the camera fixed." `ego`: keep the
     point cloud fixed and rotate the *camera* around the anchor camera's own
     center (Eq. 4), i.e. a first-person view from that camera's position.
   - `Δα_k, Δβ_k`: **azimuth and elevation rotation angles**. Confirmed from the
     actual tool-prompt figure (Fig. 9, PDF p.18, verbatim): `azimuth_angle` is
     "-180° to 180°, **integer only**: Controls left-right rotation";
     `elevation_angle` is "-90° to 90°, **integer only**: Controls up-down
     rotation." Convention: `(azimuth=0, elevation=0)` = the first input image's
     viewpoint (cam1) exactly, defined in the *input camera's own coordinate
     frame* ("azimuth rotates left/right around the camera's vertical axis;
     elevation rotates up/down around the camera's right axis") — i.e. the
     transformation is specified **egocentrically relative to a chosen anchor
     camera**, not in a fixed world/allocentric frame.
3. **Novel View Rendering (Eq. 5).** `Î_k = Render(𝒳^(m), C^(m), m)` — "a
   lightweight, point-based renderer." Global mode renders the rotated scene
   through the fixed anchor camera; ego mode first crops `𝒳` to "a wide
   field-of-view cone aligned with the forward direction of the virtual camera"
   before rendering. No further detail on hole-filling, splat size, or
   colorless/low-confidence point handling.

**Loop and stopping rule (Sec. 3.1, verbatim):** "We run the loop for at most K
iterations (default K=3), with at most one reconstruction call per query and at
most K rendering calls. The agent can terminate early by issuing a STOP action
when it deems the evidence sufficient." In practice "STOP" is not a distinct
token — it is simply *not* emitting a `<tool_call>` and instead emitting
`<answer>` (confirmed from the actual continuation-prompt text, Fig. 12: "If NO:
output your thinking process ... and your final answer"). **There is no
confidence threshold, calibration, or quantitative stopping rule** — the decision
is left entirely to the model's own judgment each turn, with only a hard cap (K)
as a backstop and a final forced no-tool turn once K is reached (Fig. 13: "WARNING:
This is the FINAL turn ... You MUST NOT call any tools").

**RL variant (Think3D-RL, Sec. 3.4):** GRPO-trained on Qwen3-VL-4B, trajectory-level
reward only (`R = R_ans + R_fmt`, Eq. 11), 977 MindCube training samples, point
clouds precomputed offline to avoid calling Pi3 during rollouts. Gradient masked
to action/answer tokens only, following DeepEyes (token-wise mask on observation/
rendered-image tokens, Sec. 3.4 "Optimization").

**Critical action-space detail found only in the actual prompt images (not in the
prose text), directly load-bearing for the branch's encoding/menu-size RQs:** the
**training-free** workflow (Fig. 8/9, GPT-4.1 / Gemini-2.5-Pro) exposes the *full*
continuous integer azimuth×elevation space with only *soft, recommended* named
viewpoints as scaffolding text ("Left views: azimuth=-45° or -90°"; "Right:
45°/90°"; "Top: elevation=30° to 60°"; "Back: 180° or ±135°"; "Diagonal: combine
azimuth and elevation e.g. 45°,30°"). But the **RL-trained / RL-inference**
workflow (Fig. 11/12) **hard-restricts the model to exactly 8 enumerated
(azimuth, elevation) pairs** and states this explicitly: *"You only have 8
viewing angles to choose: Left views (-45,0)/(-90,0); Right views (45,0)/(90,0);
Top views (0,60); Opposite views (45,60)/(45,30)/(-45,30)."* Plus an explicit
anti-repeat rule: *"Do not request the same angle as before."* This is a
hybrid **named-category-plus-explicit-numeric-pair** encoding, not pure free-form
numbers and not pure symbolic indices — and the paper silently collapses from an
open continuous menu (for strong closed models) to a small fixed 8-item menu
(for the small open model being RL-trained), without ever ablating the menu-size
choice itself.

## Concrete results and numbers

**Benchmarks used: BLINK (Multi-view), MindCube (rotation/among/around subsets,
40 Q each = 120 total), VSI-Bench-tiny (7 sampled frames/video).** ⚠️ **None of
MMSI-Bench, ViewSpatial, or SpinBench are used anywhere in this paper** — the
branch's named capability-floor benchmarks are absent; MindCube (camera-motion
type classification: rotation / around / among) and BLINK-MV (relative camera
motion inference) are the closest measured proxies, not substitutes.

**Table 2 — BLINK(MV) + MindCube, no-tool baselines (capability-floor evidence,
closest available proxy for RQ1):**

| Model | BLINK(MV) | MC-Rotation | MC-Among | MC-Around | Avg |
|---|---|---|---|---|---|
| GPT-4.1 | 36.82 | 60.00 | 46.67 | 55.00 | 49.62 |
| Gemini-2.5-Pro | 44.86 | 85.00 | 49.17 | 58.33 | 59.34 |
| Qwen3-VL-4B | 47.87 | 34.17 | 20.00 | 41.67 | 35.92 |
| GLM-4.5V | 39.85 | 45.00 | 45.00 | 22.50 | 38.09 |
| Doubao-1.5 | 50.93 | 72.50 | 40.00 | 35.00 | 49.61 |

Reading: BLINK-MV is near-chance-to-below for every model including frontier
closed models (36.82–50.93% — for a task described as inferring relative camera
motion between views); MindCube-Rotation is markedly higher for GPT-4.1/Gemini
(60/85%) but MindCube-Among is near-floor for everyone (20–49%). **No single
consistent "above/below chance" verdict — it is highly task/sub-type dependent
even within one paper's own benchmark selection**, which itself is evidence
against a flat "VLMs can/can't relate viewpoints" claim and for RQ1's more
granular framing.

**Table 2 — with Think3D (training-free tool use, up to 3 iterations):**

| Model | BLINK(MV) | MC-Rot | MC-Among | MC-Around | Avg |
|---|---|---|---|---|---|
| Think3D(GPT-4.1) | 63.91 (+27.09) | 63.33 (+3.33) | 57.50 (+5.00) | 60.83 (+14.16) | 61.19 (+11.57) |
| Think3D(Gemini-2.5-Pro) | 52.88 (+8.02) | 86.67 (+1.67) | 54.17 (+5.00) | 60.83 (+2.50) | 63.34 (+4.00) |
| Think3D(Qwen3-VL-4B, no RL) | 48.62 (+0.75) | 35.83 (+1.66) | 28.33 (+8.33) | 33.33 **(-8.34)** | 36.53 (+0.61) |
| Think3D(Qwen3-VL-4B-T3RL) | 53.39 (+7.28) | 42.50 (+11.67) | 37.47 (+11.64) | 42.50 (+6.67) | 43.97 (+9.32) |

**Table 1 — VSI-Bench-tiny, no-tool baselines:** GPT-4.1 48.18, Gemini-2.5-Pro
45.16, Qwen3-VL-4B 38.28 average (Route Plan / Rel. Dir. / Rel. Dist. / App.
Order). With Think3D: GPT-4.1→51.14 (+2.96), Gemini→51.61 (+6.45), Qwen3-VL-4B
(no RL)→39.08 (+0.80, and **Route Plan regresses 34.69→30.61, -4.08**), Qwen3-VL-4B-T3RL→45.41 (+12.05, from an RL-baseline 33.36).

**Key negative finding (Sec. 4.2, verbatim):** "for smaller models such as
Qwen3-VL-4B, the gain is marginal (0.61%), suggesting **limited spatial
reasoning capacity constrains the benefit of exploration**." Giving a weak model
the same discrete camera-control tool as a frontier model does **not**
automatically help it — on several sub-metrics tool use actively **hurts**
(BLINK MC-Around -8.34%, VSI Route-Plan -4.08%, both for un-tuned Qwen3-VL-4B).
This is the paper's own stated capability-floor finding: exploration only pays
off once the base model's spatial-reasoning capacity clears some threshold, or
once RL has taught it *when/where* to look.

**Table 3 — Ablation of components (GPT-4.1, BLINK/MindCube):**

| 3D Rec. | Cam. Anchor | Cam. Choice | Ego-view | BLINK | MindCube |
|---|---|---|---|---|---|
| – | – | – | – | 36.82 | 55.83 |
| ✓ | – | – | – | 41.17 | 54.59 |
| ✓ | ✓ | – | – | 55.46 | 57.22 |
| ✓ | ✓ | ✓ | – | 61.65 | 58.89 |
| ✓ | ✓ | ✓ | ✓ | 63.91 | 63.33 |

Reading: raw 3D reconstruction alone is nearly worthless (+4.35 BLINK, **-1.24
MindCube — i.e. it can hurt**) without a camera anchor to make rotations
interpretable. "Raw 3D input without anchor cameras causes a mild performance
drop... the model must actively explore multiple viewpoints" (Sec. 5.1). Camera
anchor is the single biggest jump (+14.29 BLINK). Ego-view is the smallest single
increment (+2.26 BLINK, +4.44 MindCube) but still positive and non-trivial.

**Table 4 — reconstructor robustness (Sec. 5.7):**

| Tool | BLINK | MindCube |
|---|---|---|
| w/o Tool | 36.82 | 53.89 |
| VGGT (CVPR'25) | 59.65 | 59.59 |
| Pi3 (ICLR'26) | 63.91 | 60.55 |

Framework is "largely reconstructor-agnostic" — VGGT retains most (not all) of
Pi3's gain. No comparison against a fundamentally different render source (mesh,
NeRF, defective/sparse survey cloud) — only two feed-forward transformer
reconstructors on presumably clean multi-view input sets.

**Table 6 (Appendix D) — angle *strategy* ablation, the paper's closest thing to
an option-selection-policy study:**

| Strategy | BLINK | MindCube | Avg |
|---|---|---|---|
| Random (2 of 8 candidate angles) | 45.86 | 27.50 | 36.68 |
| Heuristic (fixed (-45,30) & (45,30)) | 47.37 | 30.00 | 38.69 |
| RL-trained backbone, random exploration | 45.11 | 30.00 | 37.56 |
| RL-trained backbone, learned policy | **53.39** | **40.82** | **47.11** |

"RL_random performs similarly to Random... indicating that gains mainly come
from **spatial policy learning**, [not from RL just improving the backbone's
raw perception]." This is a genuine **random-baseline comparison** (RQ15-relevant):
a *learned* view-selection policy beats random selection from the *same* 8-item
menu by +10.4pp avg (47.11 vs 36.68), and merely having an RL-tuned backbone
without a learned policy (RL_random) gives almost nothing over Random with the
un-tuned backbone (37.56 vs 36.68, +0.88pp) — i.e., nearly all of the measured
benefit is attributable to *which* views get picked, not to the backbone being
RL-tuned per se, and picking *randomly* from a small, well-spread 8-view menu is
close to a floor, not a strong baseline.

**Table 7 (Appendix E) — more open models, no-tool vs Think3D, only one row with
a comparison:** GPT-4.1-mini 47.20 → Think3D(GPT-4.1-mini) 48.72 (+1.52, small).
Other rows (InternVL3.5-1B/2B/4B, LLaVA-OV, Qwen3-VL-8B/30B) are baseline-only,
**no Think3D numbers reported for them** — an evidence gap, not a null result.

**Efficiency ablation (Sec. 5.6, Fig. 7):** matched-rounds (3) Self-Refine
(Madaan et al. 2023, no 3D tools) baseline gives "only marginal improvements"
vs Think3D at comparable-or-higher token/time cost — the gain is attributed to
"explicit 3D interaction rather than multi-round prompting," i.e. more reasoning
turns alone (without new visual evidence) buys little.

**Turn-count ablation (Sec. 5.5, Fig. 6):** "for models without RL training,
increasing the number of interaction turns **does not yield a clear performance
gain**." Only after RL does turn count start correlating positively with
accuracy. This directly caps any assumption that "just let the untrained model
call the tool more" is a viable path to better coverage.

**Ego/global usage split (Appendix B, Fig. 16, GPT-4.1 only):** fine-grained
tasks (MindCube, object-direction) show much higher ego-view usage; broad-context
tasks (Route Planning) favor global-view — qualitative behavioral finding, no
numeric table given for this split.

## Evidence discipline (per-claim ablation status)

| Claim | Ablated against alternative? | Delta | Variance reported? |
|---|---|---|---|
| Camera-anchor + selection + ego-view components each help (Table 3) | Yes, additive ablation, same model (GPT-4.1) | +4.35 → +14.29 → +6.19 → +2.26/+4.44 pp per stage (BLINK/MindCube) | No (Appendix C says 3 runs averaged for "all experiments," but no std/CI shown in any table) |
| Reconstructor choice (Pi3 vs VGGT) doesn't matter much (Table 4) | Yes, swap ablation | -4.26 BLINK / -0.96 MindCube (VGGT vs Pi3) | No |
| Learned RL view-policy vs Random (Table 6) | Yes | +10.43 pp avg | No |
| **Integer-azimuth/elevation numeric encoding vs. named-only or free-form-language encoding** | **No — never ablated.** Only the *policy* (random/heuristic/RL) is ablated, not the *encoding format* itself | N/A | N/A |
| **Ego-centric parameterization vs. allocentric/world-frame parameterization** | **No** — ego/global is a *mode toggle* the agent picks, both modes share the same integer-angle encoding; the paper never compares this encoding against an allocentric-labeled alternative | N/A | N/A |
| "Clean" point cloud assumption | **No** — asserted once ("clean colored point cloud," Sec. 3.2), never tested against sparse/defective input | N/A | N/A |
| 8-item discrete menu size for RL (vs. e.g. 4 or 16 candidate angles) | **No** — 8 is simply the number chosen for the RL prompt; no sweep over menu size | N/A | N/A |

## Load-bearing assumptions

1. **Point cloud is "clean."** Pi3 is described as producing a clean, colored,
   fused reconstruction from a modest number of clean multi-view input images
   (T not stated exactly but implied small — MindCube/BLINK-MV items). No
   discussion anywhere of holes, sparsity, low-confidence/dropped points, or
   glossy-surface dropout — the render source assumption is the opposite of our
   fused, real-defect survey cloud with expected gaps.
2. **The candidate-view space is dense/continuous, not a small fixed geometric
   set**, in the training-free deployment — the model can request *any* integer
   (azimuth, elevation) pair, not just orthogonal-to-surface views from a
   pre-enumerated menu. Our setting (fixed orthogonal-to-face candidates) is
   structurally closer to the **RL-time 8-item hard menu** than to the
   open-ended training-free tool.
3. **Anchor camera is always one of the T *original input* images**, not an
   arbitrary point in space and not a node in a pre-built adjacency graph over
   candidate views — there is no "face 1 / face 2 / ... / opposite face"
   graph-topology concept anywhere in this paper; rotations are always relative
   to *some previously-seen camera pose*, which is a genuinely different
   primitive from our "orthogonal to face N" query.
4. **No infeasibility/collision handling.** Since rendering is purely geometric
   (point-cloud reprojection), every requested (azimuth, elevation, camera,
   mode) combination is always renderable — there is no notion of an
   IK/collision-infeasible view, no affordance scoring, nothing analogous to
   SayCan-style down-weighting. Not evidence either way for RQ11 (infeasible-view
   presentation), just silent by construction.
5. **Stopping is unconstrained model judgment plus a hard cap (K=3)** — no
   calibrated or measured "enough views" rule (contrast with the branch's
   interest in Explore-until-Confident-style conformal stopping).
6. **Menu size and encoding format are engineering choices, not researched
   variables** — the paper never asks "how many candidate views is too many" or
   "does the format of the angle description matter," it only asks "does having
   the tool at all help" and "does a *learned policy* over 8 fixed options beat
   random."

## What transfers to our setting

- **The 4-tuple action schema itself** — `(anchor_index, mode∈{ego,global},
  Δazimuth, Δelevation)`, with azimuth/elevation as **signed integer degrees**
  — is a clean, reusable, load-bearing template for a discrete camera-control
  tool interface. It is the fullest published spec we've found of exactly this
  kind of tool (matches the framing doc's stated reason for selecting this
  paper). Directly reusable as a starting schema for our tool's function
  signature.
- **The RL-time collapse from open continuous space to an 8-item named+numeric
  menu is itself a transferable design signal**: even this paper's authors
  found that a small open-source VLM (Qwen3-VL-4B) needed the space *reduced to
  a small fixed enumerated menu* to be RL-trainable/usable at all — the
  continuous/open version was reserved for frontier closed models only. This is
  direct evidence for RQ9 (menu size) and for the branch's premise that our
  action space "must be flat and independently addressed" for the models we can
  actually deploy — Think3D's own engineering choice already concluded this for
  a comparably-sized open model.
- **Random-vs-learned-policy comparison on a small, well-spread menu (Table 6)**
  is a genuine, directly quotable random-baseline data point: learned policy
  beats random by +10.4pp avg; RL-tuning-the-backbone-without-learning-the-policy
  buys almost nothing (+0.9pp) over random. Strong support for RQ15's premise
  that a random/uniform baseline must be beaten, not assumed beaten.
- **Anti-repeat instruction ("Do not request the same angle as before")** is a
  cheap, explicit pattern worth reusing regardless of menu size — cost-free
  redundancy guard in the prompt itself, independent of the model's own judgment.
- **Ego-mode rotation is defined relative to the *anchor camera's own frame***
  (azimuth around the camera's vertical axis, elevation around its right axis)
  — a working example of restating a manipulation egocentrically relative to a
  chosen reference view rather than in world/allocentric coordinates. Directly
  relevant to RQ2 (can "orthogonal to face 3" be restated egocentrically):
  Think3D's answer is "yes, by anchoring the rotation to that face's own
  captured camera pose and expressing the target as a small integer delta from
  it" — though note this is a *design choice made without an ego/allo ablation*
  (see Evidence discipline table), so it's a working pattern, not measured proof
  the reframing preserves accuracy.
- **Self-Refine-matched-rounds control (Sec. 5.6)** is a clean methodological
  pattern for any future ablation we run: to prove a gain comes from *new visual
  evidence* rather than *more reasoning turns*, compare against a same-turn-count,
  no-new-evidence baseline.

## What does NOT transfer, and why

- **No treatment of defective/sparse/holed point clouds** — the single biggest
  gap relative to our setting (per the framing doc's "~4 cm fused-vs-single-view
  offset and glossy-surface dropout"). Pi3's "clean colored point cloud" claim
  is unexamined; nothing here tells us whether the point-based renderer degrades
  gracefully or catastrophically on gappy/glossy input, and no other paper's
  findings should be assumed to transfer through this one — it's a genuine
  blank, not a reassurance.
- **No adjacency/topology structure over candidate views** — Think3D's "anchor
  camera" concept is always one of the *original captured* multi-view images
  (T of them, from the source dataset), never a constructed graph over a larger
  candidate space (e.g., our object's N faces with "opposite," "adjacent," etc.
  relations). RQ17's handoff question ("does any measured capability limit
  forbid a structure Branch 5 wants to build") gets a **null answer from this
  paper specifically** — Think3D neither builds nor needs a view-adjacency
  graph, so it can't be cited as either supporting or forbidding one; it simply
  operates one level below that abstraction (raw camera pose deltas, not named
  face relations).
- **No MMSI-Bench / ViewSpatial / SpinBench numbers** — cannot directly answer
  RQ1's specific capability-floor question. The closest proxies (BLINK-MV,
  MindCube) give an inconsistent picture (near-floor on BLINK-MV and
  MindCube-Among; well-above on MindCube-Rotation for strong models) that
  argues against extracting one flat "above/below chance" verdict even from
  this paper's own benchmark choices — flag as a genuine sub-task-dependent
  finding, not resolvable by this paper alone. **Handoff to Branch 2**: carry
  BLINK-MV / MindCube numbers as an *additional* data point, not a substitute
  for the named benchmarks.
- **No permutation/order test** — the RL menu's 8 items are always listed in
  the same fixed order (Left, Right, Top, Opposite) in every prompt shown; the
  paper never tests whether shuffling this order changes model behavior. RQ10
  gets no evidence here.
- **No infeasible-view handling** — geometric point-cloud rendering never fails
  or gets rejected the way an IK/collision-checked robot camera pose can; RQ11
  is untestable from this paper.
- **No hierarchical/on-demand query structure** — the whole reconstructed cloud
  is always available at once; there's no collapse/expand or query-tool pattern
  (RQ12) to compare against a "dump it all" baseline, because this paper's menu
  is already small enough that the question never arises.
- **"Reconstructor-agnostic" claim (Table 4) is about swapping Pi3↔VGGT, both
  feed-forward multi-view transformers with broadly similar output
  characteristics** — this should not be read as "any point-cloud source works
  equally well," since neither reconstructor was tested against defective or
  single-view input; do not over-extend the robustness claim to our survey-cloud
  regime.

## Handoffs

- **Branch 2 (benchmarks):** BLINK-MV / MindCube no-tool baseline numbers above
  are usable as *additional* capability-floor evidence, tagged clearly as
  non-MMSI/ViewSpatial/SpinBench proxies. The Table 6 random-vs-learned-policy
  comparison is a genuine random-baseline data point (RQ15) — flag its menu
  (8 hand-picked, well-spread angles) as the comparison condition, not a sphere
  sampling baseline.
- **Branch 5 (geometric construction / topology):** No evidence found either
  supporting or forbidding an adjacency-graph structure — Think3D operates
  below that abstraction level entirely (see "What does NOT transfer").
- **Failure-mode list for Branch 2:** (1) untuned small models can be *actively
  hurt* by having the tool at all, on specific sub-metrics (BLINK MC-Around
  -8.34pp, VSI Route-Plan -4.08pp, both Qwen3-VL-4B no-RL); (2) raw 3D
  reconstruction without an interpretable camera anchor can hurt vs. no
  reconstruction at all (MindCube 55.83→54.59, Table 3 row 2); (3) naive
  increase in tool-call turns does not improve accuracy absent RL-taught
  judgment of *when* to call (Sec. 5.5); (4) weaker models "drift toward
  redundant or even misleading camera poses" (Sec. 1, qualitative, no metric
  attached).

## Relevance note

Highest-value paper in this sub-branch for the concrete tool-schema RQs
(action-space encoding, menu-size collapse for weak models, random-baseline
comparison) — but it is silent or negative on every geometry-construction,
defective-cloud, and topology question central to our RQ2/RQ3/RQ7/RQ8/RQ12/RQ17,
and it does not use any of the three named capability-floor benchmarks, so its
contribution to RQ1 is proxy evidence only, not a direct answer.
