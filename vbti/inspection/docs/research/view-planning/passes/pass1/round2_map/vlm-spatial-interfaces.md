# Branch 6 — `vlm-spatial-interfaces`
## Presenting spatial action spaces to LLM/VLM agents — round-2 document map

**Date:** 2026-08-27 · **Pass 1, round 2** · **Feeds:** RQ4 (primary), RQ3
(tractability pressure back onto the representation), architecture decision
(blind vs. seeing orchestrator) · **Priority:** P2

**Plan:** [`../round1_plan.md#branch-6--vlm-spatial-interfaces`](../round1_plan.md)
**Contract:** [`../../../../geometry-aware-view-planning.md`](../../../../geometry-aware-view-planning.md)

---

## How this list is organised

The branch plan demanded every candidate be assigned to one of three interface
families up front, because **the pass-2 decision is a choice among these
families**:

| Family | Meaning | Our instantiation |
|---|---|---|
| **A** | Text/symbolic encoding of the structure, consumed by a model that never sees the object | Blind orchestrator navigating a serialized face/orbit/view graph |
| **B** | Views rendered and marked; the model picks a label off an image | Orchestrator sees a contact sheet / marked render of the PLY and picks view `7` |
| **C** | Hybrid — text carries the structure, a render carries what text cannot | Text lists faces + a marked render disambiguates which face is which |

Two further tiers have no family because they bound or engineer *any* choice:

- **CAP** — measured capability limits (benchmarks, diagnostics). These say
  what an encoding may assume the model can do at all.
- **ENG** — interface engineering for discrete action sets and the
  blind-orchestrator/vision-subagent architecture itself.

**Per-entry tags.** `EXTRACT:` what the reader must pull out (this branch cares
about the *literal encoding* and any *quantitative comparison between
encodings*). `VIOLATES:` givens from the contract that the work assumes away.
`BENCH:` flag for Branch 2 (`benchmarking-view-planning`). `CRITIC:` the
standing branch-6 question — *is the claimed VLM capability demonstrated or
demoed?*

**Count:** 63 candidates — 23 high, 2 med/high, 35 med, 3 low. Well over the
plan's 18–28 target, deliberately: the branch spans three interface families
*plus* a capability-bounds tier and an architecture tier that no other branch
owns, and the subfield is young enough that the negative results are scattered
across many papers rather than concentrated in a survey. **The priority column
is the real filter.** The 23 highs are the reading list and cover all three
families, the capability floor, and both sides of the architecture argument;
the meds are the depth bench to pull from once a family is chosen.

---

## Family A — text-encoded structure for a blind orchestrator

The direct precedent for keeping the current architecture: give the model a
serialized spatial structure and let it navigate by name.

### `sayplan-3dsg-llm-planning`
**SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable
Robot Task Planning** — Rana, Haviland, Garg, Abou-Chakra, Reid, Suenderhauf.
CoRL 2023, PMLR 229:23–72. [arXiv:2307.06135](https://arxiv.org/abs/2307.06135)
· [project](https://sayplan.github.io/) · **A · high · RQ4, RQ3**

The canonical blind-LLM-over-spatial-structure interface: a hierarchical 3D
scene graph serialized as JSON, with the model issuing `expand`/`contract` ops
on nodes to reveal detail only where the task needs it — the collapse/expand
pattern our own agentic-stack synthesis already flagged as the thing to steal.

- `EXTRACT:` the literal JSON schema fed to the model; the expand/contract op
  vocabulary; the scalability analysis (token footprint vs. graph size); the
  iterative-replanning loop that catches infeasible actions.
- `VIOLATES:` room/floor-scale semantic graph, not an object-surface geometry;
  the hierarchy is semantic (floor→room→asset), ours would be geometric
  (object→primitive→view). Test whether the addressing idea survives the swap.
- `BENCH:` its scene-graph simulator as a feasibility oracle is a template for
  a dry-run view-space validator.

### `structured-interfaces-3dsg-cypher`
**Structured Interfaces for Automated Reasoning with 3D Scene Graphs** — Ray,
Arkin, Biggie, Fan, Carlone, Roy (MIT). Oct 2025.
[arXiv:2510.16643](https://arxiv.org/abs/2510.16643) · **A · high · RQ4, RQ3**

The strongest argument against dumping the action space into context at all:
encode the structure in a graph database and expose a **query language as a
tool**; RAG-style retrieval of the task-relevant subgraph beat both
context-window and code-generation baselines and cut token counts sharply on
graphs far past context limits.

- `EXTRACT:` the Cypher tool schema; the three-arm comparison (context-window
  vs. code-gen vs. query interface) with token counts and accuracy; where the
  query interface *fails* (queries the model cannot formulate).
- `VIOLATES:` scale motivation is 15k places / 1km traversal — our view space
  is tens of items, so the token argument does not transfer; the *interface*
  argument (query vs. dump) does. Flag this asymmetry explicitly.

### `mapgpt-map-guided-prompting`
**MapGPT: Map-Guided Prompting with Adaptive Path Planning for
Vision-and-Language Navigation** — Chen, Lin et al. ACL 2024.
[arXiv:2401.07314](https://arxiv.org/abs/2401.07314) · **A · high · RQ4, RQ3**

Closest existing textual encoding of a *view/place graph*: topology verbalised
as "Place 0 is connected with Places 1, 2, 3", action-space labels augmented
with node IDs, updated online — i.e. exactly the question of how to say
"orthogonal to face 3, adjacent to view 7" in prose.

- `EXTRACT:` verbatim Map Connectivity Prompt and action-space prompt; the
  stated reason precise coordinates were abandoned ("GPT-4V struggles to
  understand navigation environments from multiple precise coordinates") — that
  claim is load-bearing for our {h,v}-vs-geometry decision; the adaptive
  multi-step planning over the map.
- `CRITIC:` gains are reported against zero-shot baselines, not against an
  ablation isolating the topology text — see `tagavlm-topology-critique`.

### `tagavlm-topology-critique`
**TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language
Navigation** — 2026. [arXiv:2603.02972](https://arxiv.org/abs/2603.02972)
· **A · med · RQ4**

Negative evidence, and the ammunition the critic will demand: adding
MapGPT-style textual topological-map prompts reportedly raised success rate by
only **+0.94%** over baseline, which the authors read as a real limit of
understanding topology through purely textual representation vs.
attention-based spatial encoding.

- `EXTRACT:` the exact ablation setup and number; their argued mechanism for
  why text topology under-delivers.
- `CRITIC:` this is the single most direct published threat to the blind
  orchestrator over a geometry-conditioned graph. Verify the number in the
  paper, not the summary.

### `navgpt-explicit-reasoning-vln`
**NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large
Language Models** — Zhou, Hong, Wu. AAAI 2024.
[arXiv:2305.16986](https://arxiv.org/abs/2305.16986) · **A · med · RQ4**

A fully blind LLM selecting among M navigable viewpoints presented as textual
descriptions with unique IDs, with vision foundation models acting purely as
translators — structurally identical to our orchestrator + vision subagents.

- `EXTRACT:` the observation/action prompt template; the explicit "do not
  fabricate nonexistent viewpoint IDs" system rule (an ID-hallucination
  failure mode we will hit); the history-summarisation buffer.
- `VIOLATES:` pre-defined connectivity graph of static viewpoints — actually
  *aligned* with our discrete-selection given; note as precedent, not violation.

### `can-llms-see-without-pixels-sitbench`
**Can LLMs See Without Pixels? Benchmarking Spatial Intelligence from Textual
Descriptions (SiT-Bench)** — Guo, Yang, Li et al. Jan 2026.
[arXiv:2601.03590](https://arxiv.org/abs/2601.03590) · **A · high · RQ4 ·
BENCH**

The measurement of the blind-orchestrator hypothesis: >3,800 items, scenes
converted to coordinate-aware textual descriptions, spanning perspective
transformation and fine-grained manipulation. Reports strong local semantic
performance but a persistent "spatial gap" in global consistency, and that
explicit spatial reasoning prompts materially help.

- `EXTRACT:` the textual scene-description format (this is a candidate encoding
  for us, off the shelf); per-subtask scores, especially perspective
  transformation; the prompting that recovered performance.
- `BENCH:` its text-description generator is a ready-made way to build our own
  blind-encoding test set over PLY-derived geometry.

### `conceptgraphs-openvocab-3dsg`
**ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning**
— Gu, Kuwajerwala, Morin, Jatavallabhula et al. ICRA 2024.
[arXiv:2309.16650](https://arxiv.org/abs/2309.16650) ·
[project](https://concept-graphs.github.io/) · **A · med · RQ3, RQ4**

The standard object-centric graph that gets serialized to text for LLM
planning; matters here as the *baseline representation* that Family-B papers
(3D-Mem, SPAZER) attack for oversimplifying spatial relations.

- `EXTRACT:` node/edge text schema; what spatial information survives
  serialization and what is lost.
- `VIOLATES:` object-level granularity — our structure is sub-object
  (faces/orbits of a single part).

### `clio-task-driven-scene-graphs`
**Clio: Real-Time Task-Driven Open-Set 3D Scene Graphs** — Maggio, Chang,
Hughes, Schmid, Carlone (MIT SPARK). RA-L 2024, 9:8921–8928.
[arXiv:2404.13696](https://arxiv.org/abs/2404.13696) · **A · med · RQ3, RQ4**

Formalises the exact question RQ3/RQ4 collide on — *what granularity of
structure should the map expose?* — as an Information Bottleneck problem where
the answer is task-dependent, with the same parameters yielding room-level or
knob-level structure depending on the task list.

- `EXTRACT:` the IB formulation and whether it is computable over our
  primitive decomposition; evidence that granularity must be task-conditioned
  (i.e. the inspection question should shape the action space, not just the
  geometry).

### `found-it-granularity-on-demand`
**FOUND-IT: Foundation-model-first Task-driven 3D Scene Graphs with Granularity
on Demand** — 2026. [arXiv:2605.25371](https://arxiv.org/abs/2605.25371)
· **A · med · RQ3, RQ4**

Clio's 2026 successor: granularity adapts as the task list *evolves* mid-run
(stove vs. stove knob), plus an agentic scene-graph querying interface — the
closest thing to "expand the face into its sub-views only when the inspection
question demands it".

- `EXTRACT:` the agentic query interface (compare against
  `structured-interfaces-3dsg-cypher`); the dynamic re-granularisation trigger.

### `3d-mem-snapshot-memory`
**3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning** (orig.
*SnapMem*) — Yang et al. CVPR 2025.
[arXiv:2411.17735](https://arxiv.org/abs/2411.17735) · **A→B · med · RQ4, RQ3**

The explicit rebuttal to Family A: argues object-centric scene graphs
"oversimplify spatial relationships … with restrictive textual relationships"
and replaces them with image snapshots — memory snapshots for explored regions,
**frontier snapshots** for unexplored directions — and has the VLM pick a
frontier snapshot each step.

- `EXTRACT:` the frontier-snapshot construct (a rendered stand-in for "a
  direction you haven't looked from" — directly reusable for uncovered surface
  regions); the co-visibility clustering that decides which single image
  represents a cluster; the retrieval mechanism as the snapshot set grows.
- `CRITIC:` its anti-scene-graph claim is argued, then benchmarked on EQA — check
  whether the ablation isolates representation from exploration policy.

### `agentic-scene-policies`
**Agentic Scene Policies: Unifying Space, Semantics, and Affordances for Robot
Action** — 2025. [arXiv:2509.19571](https://arxiv.org/abs/2509.19571)
· **A · med · RQ4**

Scene queries exposed as freely-callable tools (open-vocab retrieval, size and
distance tools, an `interact` tool over 3D affordances) returning basic types
or a symbolic state — a worked example of the "spatial state as tool API rather
than as context blob" pattern.

- `EXTRACT:` the tool list and return schemas; how spatial queries are typed.

---

## Family B — rendered and marked views for a seeing orchestrator

The alternative the contract explicitly permits. This family is demo-heavy;
prioritise papers with ablations.

### `set-of-mark-prompting`
**Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V** —
Yang, Zhang, Li, Zou, Li, Gao (MSR). Oct 2023.
[arXiv:2310.11441](https://arxiv.org/abs/2310.11441) ·
[repo](https://github.com/microsoft/SoM) · **B · high · RQ4**

The founding technique of the seeing-orchestrator option: segment the image,
overlay alphanumeric marks, let the model refer to regions by label instead of
by coordinates. Everything downstream (MOKA, SeeGround, GUI agents) is a
descendant.

- `EXTRACT:` mark design choices (alphanumerics vs. masks vs. boxes, placement,
  granularity); the stated rationale for *why* coordinate emission is avoided —
  it is the same reason our contract says VLMs cannot emit metric poses.
- `VIOLATES:` assumes a segmentable RGB image of a scene; our marks would be on
  a *render of a point cloud*, where SAM-style segmentation is not the natural
  region source. Note this substitution risk.

### `som-limitations-contrastive-region-guidance`
**Contrastive Region Guidance: Improving Grounding in Vision-Language Models
without Training** — Wan et al. Mar 2024.
[arXiv:2403.02325](https://arxiv.org/abs/2403.02325) · **B · med · RQ4**

The best-documented SoM failure evidence: marks *decrease* performance on
LLaVA-family models, with the gain on GPT-4V attributed partly to OCR ability
(48.5% vs 28.7%), and a reported **−16.2% on relational questions** even where
overall accuracy rose.

- `EXTRACT:` the per-category breakdown (recognition up, relations down); the
  OCR-dependence argument.
- `CRITIC:` if marks trade relational fidelity for referential convenience,
  a marked render is a bad interface for "which of these views is most
  orthogonal to this face" — that is a *relational* question. Highest-value
  single critique in Family B.

### `pivot-iterative-visual-prompting`
**PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs** —
Nasiriany, Xia, Yu, Xiao, Liang, Dasgupta, Driess et al. (Google DeepMind).
ICML 2024. [arXiv:2402.07872](https://arxiv.org/abs/2402.07872) ·
[project](https://pivot-prompt.github.io/) · **B · high · RQ4, RQ1**

The load-bearing insight for our whole design: VLMs cannot emit precise spatial
outputs but *can* select among a discrete set of coarse annotated choices,
which can be refined by re-sampling around the selection. Numbered arrows
annotated on the image, colour/size encoding forward-back depth, parallel calls
for robustness.

- `EXTRACT:` the annotation grammar (arrows, circled numbers, colour/size for
  the third dimension — we need exactly this to draw a viewsphere on an image);
  the refinement loop and its convergence behaviour; the parallel-call
  aggregation; the honest "performance is far from perfect" limitations
  section.
- `BENCH:` its zero-shot localization tasks are a template for measuring
  view-choice accuracy against a known-correct answer.

### `scaffold-coordinates-prompting`
**SCAFFOLD: Scaffolding Coordinates to Promote Vision-Language Coordination in
Large Multi-Modal Models** — Lei et al. ACL 2024.
[arXiv:2402.12058](https://arxiv.org/abs/2402.12058) · **B · high · RQ4**

Dot-matrix overlay giving the model textual coordinate anchors; reports **+13.3
average on spatial reasoning** over CoT prompting across benchmarks — the
cleanest quantified evidence that *how you draw the addressing scheme on the
image* changes spatial competence.

- `EXTRACT:` grid density (8×8 dot matrix in follow-ups), label format, and the
  paired textual prompt; the ablation separating the overlay from the prompt;
  the SoM-vs-SCAFFOLD contrast (semantic regions vs. regular spatial lattice) —
  our viewsphere is a lattice, not a segmentation, so this is the closer
  ancestor.

### `moka-mark-based-manipulation`
**MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting** —
Liu, Fang, Abbeel, Levine. RSS 2024.
[arXiv:2403.03174](https://arxiv.org/abs/2403.03174) ·
[project](https://moka-manipulation.github.io/) · **B · high · RQ4, RQ3**

Converts spatial choice into literal multiple choice: a 5×5 grid with cells
labelled `a1`–`e5`, candidate keypoints marked `P1, P2, …`, and hierarchical
prompting that decomposes the task before asking the low-level "pick a label"
question. This is the "annotated action menu" pattern in its purest robotic form.

- `EXTRACT:` the grid/label scheme and why a 5×5 granularity; the two-level
  prompt split (decompose then select); the point→SE(3) lift (our analogue:
  label→camera pose); in-context example design.
- `VIOLATES:` acts on a single fixed RGB view of a tabletop; our marks live on
  a render whose viewpoint is itself the thing being chosen (chicken/egg — flag
  for synthesis).

### `ap-vlm-active-perception-grid`
**AP-VLM: Active Perception Enabled by Vision-Language Models** (v2 retitled
*Scene Exploration by Vision-Language Models*) — Sripada, Carter, Guerin,
Ghalamzan. Sep 2024. [arXiv:2409.17641](https://arxiv.org/abs/2409.17641)
· **B · high · RQ4, RQ2 · BENCH**

Closest published prior art to our system, already named as such in the
project's own agentic-stack synthesis: a **virtual 3D grid overlaid on the
scene image**, anchored to a calibrated marker, from which a zero-shot VLM picks
the next viewpoint and orientation for a UR5 / Franka with an in-hand camera,
looping until it can answer.

- `EXTRACT:` the grid geometry and how orientation options are presented
  alongside position; the exact query prompt; the stopping criterion; the
  baselines beaten (passive perception, TGCSR) and by how much; failure cases.
- `CRITIC:` grid is workspace-anchored and geometry-agnostic — the *same*
  criticism Albert levelled at our {h,v} sphere. Establish whether AP-VLM ever
  achieves surface-orthogonal views or only "better than fixed camera".
- `BENCH:` its task set (occluded / inclined object identification) is a
  candidate protocol for evaluating view selection with a semantic success
  criterion.

### `graspview-vlm-nbv-scoring`
**GraspView: Active Perception Scoring and Best-View Optimization for Robotic
Grasping in Cluttered Environments** — Nov 2025.
[arXiv:2511.04199](https://arxiv.org/abs/2511.04199) · **B · high · RQ4, RQ1**

Nearly our pipeline: renders candidate viewpoints **off-screen from the current
coloured point cloud**, has a VLM score each for expected visibility and
utility, executes only the top-rated pose — explicitly replacing hand-crafted
coverage/information-gain utilities with VLM judgement.

- `EXTRACT:` the rendering setup (how a partial coloured cloud is made legible
  to a VLM — the single most transferable engineering detail in this branch);
  the scoring prompt; whether scores are calibrated or merely ranked; how many
  candidates are shown at once.
- `CRITIC:` "hand-crafted utilities are brittle" is asserted — check whether
  they benchmark VLM scoring *against* a classical utility, or only against no
  NBV. If the former, it bears directly on RQ1 (algorithmic judge) vs. RQ4
  (let the model judge).
- `BENCH:` a direct comparison protocol candidate for "VLM judge vs. geometric
  judge".

### `spazer-progressive-3d-grounding`
**SPAZER: Spatial-Semantic Progressive Reasoning Agent for Zero-shot 3D Visual
Grounding** — Jun 2025. [arXiv:2506.21924](https://arxiv.org/abs/2506.21924)
· **B · med · RQ4**

An agent that renders holistic views of a point cloud from multiple
perspectives and selects the best one for the query — *and reports the ablation
we most need*: agent-selected views beat a **random view by only +3.6%** (and
BEV by +6.3%).

- `EXTRACT:` the render/selection loop; the random-view baseline construction;
  any variance reporting.
- `CRITIC:` +3.6% over random is the branch's headline caution — most of the
  benefit of "look from somewhere else" may be multi-view aggregation, not
  viewpoint reasoning. Cross-reference the classical active-recognition result
  that random motion already delivers ~75% of the NBV gain.
- `BENCH:` mandates a **random-view baseline** in our benchmark.

### `manipulate-anything-view-menu`
**Manipulate-Anything: Automating Real-World Robots using Vision-Language
Models** — Duan et al. CoRL 2024.
[arXiv:2406.18915](https://arxiv.org/abs/2406.18915) · **B · med · RQ4**

Concatenates all available viewpoints (multi-camera or re-rendered from a
single RGB-D) into one tiled frame with an index number on each tile and asks
the VLM to choose — the "contact sheet" encoding of a discrete view space.

- `EXTRACT:` tile count and layout; label placement; the stated motivation
  (single-viewpoint VLM reasoning is poor); any sensitivity to tile ordering.

### `antigrounding-render-then-evaluate`
**AntiGrounding: Lifting Robotic Actions into VLM Representation Space for
Decision Making** — Jun 2025.
[arXiv:2506.12374](https://arxiv.org/abs/2506.12374) · **B · med · RQ4, RQ1**

Render-then-evaluate with a *dynamic* view set: iteratively replaces candidates
to converge on top-k (k=3) viewpoints, and evaluates each rendered candidate
with four structured sub-questions rather than one holistic judgement.

- `EXTRACT:` the four sub-questions (decomposed rubric — directly applicable to
  "is this view orthogonal / well-framed / unoccluded"); the iterative view
  replacement schedule.

### `coarse-correspondences`
**Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal
Language Model** — Liu et al. CVPR 2025.
[arXiv:2408.00754](https://arxiv.org/abs/2408.00754) · **B · med · RQ4**

Training-free visual prompting that marks *the same instance with the same ID
across views*: +20.5% ScanQA, +9.7% OpenEQA episodic, and — critically for us —
a diagnostic set showing it improves perspective-taking, though models still
struggle.

- `EXTRACT:` the cross-view ID marking scheme (our multi-view inspection needs
  exactly this: "face 3 in view 2 is face 3 in view 5"); the count of marked
  instances they found workable.

### `eyevla-robotic-eyeball`
**Look, Zoom, Understand: The Robotic Eyeball for Embodied Perception
(EyeVLA)** — Yang, Han, Xie, Guo, Lian. Nov 2025 (v2 Apr 2026).
[arXiv:2511.15279](https://arxiv.org/abs/2511.15279) · **B · med · RQ4, RQ3**

How to *tokenize* a camera action space: pan/tilt/zoom discretized into action
tokens, integer angle deltas decomposed into decimal digits (L=2 covers >98% of
real actions), jointly modelled with vision and language.

- `EXTRACT:` the action-token encoding and the digit-decomposition trick; the
  argument for discretizing angles rather than emitting continuous poses.
- `VIOLATES:` fine-tuned VLA, not a prompted orchestrator — out of the
  contract's zone; extract the *encoding*, not the method.

### `visual-prompting-survey`
**Visual Prompting in Multimodal Large Language Models: A Survey** — Wu, Zhang,
Xia et al. Sep 2024. [arXiv:2409.15310](https://arxiv.org/abs/2409.15310)
· **B · med · RQ4**

Spine for Family B: taxonomy of visual prompt types, automatic prompt
annotation, and prompt-learning methods.

- `EXTRACT:` the prompt-type taxonomy as our vocabulary for describing encoding
  options; pointers to any encoding-vs-encoding comparisons.

### `visual-structures-binding-problem`
**Visual Structures Help Visual Reasoning: Addressing the Binding Problem in
VLMs** — Jun 2025. [arXiv:2506.22146](https://arxiv.org/abs/2506.22146)
· **B · med · RQ4**

Why overlays work when they work: a lightweight spatial scaffold gives the
model an external traversal plan, with markedly larger gains on 2D (crowded
single-plane) scenes than 3D.

- `EXTRACT:` the 2D-vs-3D gap (our renders of a 3D object sit on the weaker
  side of that split — an argument to take seriously); the scaffold designs
  compared.

---

## Family C — hybrid: text carries structure, render carries what text cannot

Likely the pragmatic answer; the evidence base is the youngest.

### `seeground-query-aligned-rendering`
**SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding**
— Li, Li, Kong, Yang, Liang. CVPR 2025, pp. 3707–3717.
[arXiv:2412.04383](https://arxiv.org/abs/2412.04383) ·
[repo](https://github.com/iris0329/SeeGround) · **C · high · RQ4, RQ3**

The reference hybrid: a 3D scene represented as **query-aligned rendered images
+ spatially enriched text**, with a Perspective Adaptation Module choosing the
render viewpoint and a Fusion Alignment Module tying 2D marks to 3D
descriptions; explicitly argues text conveys positions but "fails to convey
critical visual details".

- `EXTRACT:` the text-description schema *and* the render/mark scheme, plus
  their division of labour; the ablation showing query-aligned viewpoints beat
  static camera strategies and that gains concentrate on **directional
  queries** ("left/right") — ours are directional by construction; the residual
  19% spatial-relationship error rate and viewpoint errors.
- `BENCH:` the error taxonomy (spatial-relation vs. viewpoint errors) is a
  ready failure-mode taxonomy for our benchmark.

### `think3d-3d-manipulation-toolkit`
**Think3D: Thinking with Space for Spatial Reasoning** — Zhang, Wu, Jia, Wang
et al. (DUT / UCSD / Oxford). Jan 2026, v3 Mar 2026.
[arXiv:2601.13029](https://arxiv.org/abs/2601.13029) ·
[repo](https://github.com/zhangzaibin/spagent) · **C · high · RQ4, RQ3**

The fullest published specification of a **discrete camera-control tool
interface for a VLM**: `azimuth_angle` (−180…180, integers only),
`elevation_angle` (−90…90, integers only), a `rotation_reference_camera`
argument, an ego/global-view toggle, all defined relative to an input camera
frame, driven in an observe→manipulate→reflect loop capped at K=3 with an
explicit STOP action.

- `EXTRACT:` the complete tool schema (this is a candidate action-space API we
  could adopt nearly verbatim); the integer-only constraint; the ego/global
  switch; the K-cap and STOP semantics; zero-shot gains (+7.8% BLINK
  multi-view / MindCube, +4.7% VSI-Bench) and what RL added (+0.7% → +10.7% on
  a 4B model — i.e. prompting alone barely moved a small model).
- `CRITIC:` the honest reading is that *the tool interface alone* gave a small
  model almost nothing without RL. Establish which of our candidate models sits
  on which side of that line.

### `thinking-with-novel-views`
**Thinking with Novel Views: A Systematic Analysis of Generative-Augmented
Spatial Intelligence** — Zhang, Wang, Liu et al. May 2026.
[arXiv:2605.10588](https://arxiv.org/abs/2605.10588) · **C · high · RQ4**

Contains the single most directly usable encoding finding in this branch:
**numerical camera-pose specifications yield more reliable view control than
free-form language.** Also studies whether iterative multi-turn view refinement
pays (it does, modestly: +1.3 to +3.9 pp).

- `EXTRACT:` the instruction-format comparison and its numbers — this decides
  whether our action labels should be angles, indices, or names; the
  generation-fidelity coupling result.
- `VIOLATES:` uses generative novel-view synthesis (the contract's deferred
  Cosmos-style branch). Read for the *instruction-format* result only; note
  the deferred-topic overlap for the PI.

### `mv-gel-geometry-conditioned-view-ranking`
**MV-GEL: Language-Driven Multi-View Geometric Entity Localization on Meshes**
— Bali, Aydin (Hereon / TUHH / DFKI). Jun 2026, v2 Jul 2026.
[arXiv:2606.31533](https://arxiv.org/abs/2606.31533) · **C · high · RQ4, RQ3,
RQ2**

The closest paper in the entire branch to our actual problem: grounding
language to **faces, edges and curved surfaces of a CAD-like 3D object** via
prompt-aware view selection, with a geometry-conditioned view-ranking module
(GELviews) that beats CLIP-semantic and random sampling, reaching >90% of oracle
top-1 face/edge localization. Concludes language-driven 3D localization is
"fundamentally a geometry-aware view prioritization problem".

- `EXTRACT:` how faces/edges are addressed and referred to in language (our RQ3
  addressing scheme, solved for a neighbouring task); GELviews' ranking
  features; the explicit finding that **global semantic (CLIP) matching fails
  on homogeneous, textureless industrial parts** — our exact object class;
  the oracle-vs-method gap as an evaluation design.
- `VIOLATES:` meshes/CAD, not defective survey clouds. The whole question is
  whether view ranking survives the swap — hand to Branch 3 for a joint note.

### `seqvlm-multiview-visibility`
**SeqVLM: Proposal-Guided Multi-View Sequences Reasoning via VLM for Zero-Shot
3D Visual Grounding** — Aug 2025.
[arXiv:2508.20758](https://arxiv.org/abs/2508.20758) · **C · med · RQ4, RQ1**

Adds geometric rigour to the render step: world→camera projection with a
**depth-consistency visibility check** before a point is considered visible in
a view, and viewpoint selection by **projected area of the proposal** — a
cheap, implementable view-quality heuristic.

- `EXTRACT:` the visibility test and the projected-area scoring (both directly
  implementable on our PLY); the multi-view sequence prompt format.
- `BENCH:` projected-area ranking is a strong cheap baseline for RQ1's judge.

### `vlm3d-viewpoint-dependency-critique`
**Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?** — Feb
2025. [arXiv:2502.08503](https://arxiv.org/abs/2502.08503) · **C · med · RQ4 ·
BENCH**

Methodological critique plus a minimal render→VLM pipeline (VLM3D), with an
explicit enumeration of viewpoint-dependency failure modes: blind spots outside
the chosen view, occlusion/overlap, single-surface capture missing multifaceted
geometry.

- `EXTRACT:` the failure enumeration (a ready checklist for our render
  interface); their fixed front/side-view convention for object-level clouds
  and why it sufficed; the benchmark-contamination argument.
- `BENCH:` the "is this benchmark actually testing 3D?" test applies verbatim to
  any benchmark we design for view choice.

### `i-perceive-active-perception-foundation`
**I-Perceive: A Foundation Model for Vision-Language Active Perception** — 2026.
[arXiv:2603.00600](https://arxiv.org/abs/2603.00600) · **C · med · RQ4, RQ2 ·
BENCH**

Predicts camera views from open-ended language instructions by fusing a VLM
with a geometric foundation model along two coupled pathways (geometric +
vision-language), closed-loop across steps.

- `EXTRACT:` the two-pathway split — it is the architectural formalisation of
  "let geometry propose, let language dispose", which is our analytical-track
  + orchestrator design; **View Coverage IoU** (volumetric IoU of visible
  voxels) plus translation/rotation pose error as the metric triple.
- `BENCH:` View Coverage IoU is a strong candidate metric for Branch 2.
- `VIOLATES:` emits camera poses (continuous) — outside the contract's discrete
  selection. Extract metrics and architecture, not the output interface.

### `explore-until-confident`
**Explore until Confident: Efficient Exploration for Embodied Question
Answering** — Ren, Clark, Dixit, Itkina, Majumdar, Sadigh. RSS 2024.
[arXiv:2403.15941](https://arxiv.org/abs/2403.15941) ·
[project](https://explore-eqa.github.io/) · **C · high · RQ4, RQ1 · BENCH**

Two things we need and nobody else supplies together: a semantic map built by
**visual prompting a VLM over depth** (hybrid encoding), and **conformal
prediction to calibrate when to stop looking** — the termination criterion our
inspection loop currently lacks.

- `EXTRACT:` the visual-prompt→semantic-value mapping; the conformal
  calibration procedure and its coverage guarantee; the two named failure modes
  (stopping prematurely / over-exploring) — both are ours.
- `BENCH:` HM-EQA protocol and the calibration methodology are directly
  reusable for "how many views until the answer is justified".

---

## CAP — measured limits of VLM spatial reasoning

These bound what *any* encoding may assume. Read them as constraints, not
methods. Every one is a source of benchmark test cases for Branch 2.

### `mmsi-bench`
**MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence** — Yang, Xu,
Xie et al. ICLR 2026. [arXiv:2505.23764](https://arxiv.org/abs/2505.23764) ·
[repo](https://github.com/InternRobotics/MMSI-Bench) · **CAP · high · RQ4 ·
BENCH**

The hardest number in the branch: best open model ~30%, GPT-5 ~40%, humans 97%
— on exactly our task shape (multi-image, camera-camera and camera-object
relations, camera motion), with an automated error analysis naming four failure
modes (grounding, overlap-matching/scene-reconstruction, situation
transformation, spatial logic).

- `EXTRACT:` the camera-camera and camera-object subtask scores specifically —
  that is "can the model relate view 3 to view 7"; the four failure modes as
  our test-case taxonomy.
- `CRITIC:` if camera-camera relations are near chance, a *relational* view
  space (adjacency, "opposite face") is unusable and the encoding must be
  flat/independent. This single result could decide RQ3's topology question.

### `viewspatial-bench`
**ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in
Vision-Language Models** — May 2025.
[arXiv:2505.21500](https://arxiv.org/abs/2505.21500) ·
[project](https://zju-real.github.io/ViewSpatial-Page/) · **CAP · high · RQ4 ·
BENCH**

Allocentric reasoning sits barely above chance (GPT-4o 34.98%, Gemini-2.0-Flash
32.56%, chance 26.33%) while egocentric is fine.

- `EXTRACT:` the ego/allo split numbers and task formats.
- `CRITIC:` "orthogonal to face 3" is an **object-centric (allocentric)** frame.
  This is the most direct measured threat to any object-anchored addressing
  scheme, and the strongest argument for expressing views in the *camera's*
  frame or as opaque labels instead.

### `spinbench`
**SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs** —
Sep 2025. [arXiv:2509.25390](https://arxiv.org/abs/2509.25390) · **CAP · med ·
RQ4 · BENCH**

Names the failure modes we will hit: persistent egocentric bias, difficulty
with rotation and viewpoint change, symmetry inconsistency, and failure of
*linguistic-only* spatial inference — plus the inversion that the best
egocentric models are the worst allocentric ones.

- `EXTRACT:` the linguistic-only-inference result (bears directly on the blind
  orchestrator); the symmetry inconsistency (a box has symmetric faces — this
  is our object class).

### `egocentric-bias-flipset`
**Egocentric Bias in Vision-Language Models (FlipSet)** — Wang, Li, Wang, Zhao,
Ji, Gao, Liu, Deng, Luo. Feb 2026.
[arXiv:2602.15892](https://arxiv.org/abs/2602.15892) · **CAP · med · RQ4**

103 VLMs, most below chance on Level-2 perspective taking, ~75% of errors
literally reproducing the camera viewpoint — and a **compositional deficit**:
theory-of-mind and mental rotation each work in isolation and collapse when
composed.

- `EXTRACT:` the compositional-deficit control design — it suggests our prompt
  should never require composing two spatial operations in one step.

### `vsi-bench-thinking-in-space`
**Thinking in Space: How Multimodal Large Language Models See, Remember, and
Recall Spaces (VSI-Bench)** — Yang et al. CVPR 2025.
[arXiv:2412.14171](https://arxiv.org/abs/2412.14171) · **CAP · med · RQ4 ·
BENCH**

The standard video-based spatial benchmark (8 tasks: relative direction, path
planning, size/distance estimation, appearance order), and the source of the
cognitive-map framing many 2026 methods target.

- `EXTRACT:` relative-direction and path-planning scores (the closest proxies
  for navigating a view graph); the cognitive-map probing methodology.

### `blink-benchmark`
**BLINK: Multimodal Large Language Models Can See but Not Perceive** — Fu et
al. ECCV 2024. [arXiv:2404.12390](https://arxiv.org/abs/2404.12390) · **CAP ·
med · RQ4 · BENCH**

Perception-level (not reasoning-level) failures on tasks humans solve instantly;
its multi-view split is the standard probe for reasoning across distributed
visual evidence and is the benchmark Think3D moves most.

- `EXTRACT:` the multi-view subtask design and human-vs-model gap.

### `robospatial-reference-frames`
**RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language
Models for Robotics** — Song et al. Nov 2024.
[arXiv:2411.16537](https://arxiv.org/abs/2411.16537) · **CAP · med/high · RQ4,
RQ3 · BENCH**

Robotics-specific, and organised around exactly the axis RQ3 must fix: **three
spatial reference frames**, with the documented finding that VLMs default to a
camera-centric interpretation and fail to generalise when an object-centric
frame is required.

- `EXTRACT:` the reference-frame taxonomy and per-frame accuracy; the
  "Context" task type (identify vacant space in a direction relative to a
  reference object) — structurally identical to "the view orthogonal to this
  face".
- `BENCH:` reference-frame disambiguation is a mandatory axis of our benchmark.

### `spatialab-open-ended`
**SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?**
— ICLR 2026. [arXiv:2602.03916](https://arxiv.org/abs/2602.03916) · **CAP ·
med · RQ4 · BENCH**

The multiple-choice-vs-open-ended gap: all models drop ~10–25% when the answer
is not offered as options (GPT-5-mini 40.93% vs. human 64.93%).

- `EXTRACT:` the MC/open-ended delta — direct evidence that *enumerating* the
  action space is doing real work, which is an argument for keeping selection
  discrete regardless of family.

### `picture-worth-thousand-words`
**Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for
Vision Language Models** — Jun 2024.
[arXiv:2406.14852](https://arxiv.org/abs/2406.14852) · **CAP · med · RQ4**

Controlled text-vs-image encodings of the *same* spatial information; text-only
LLMs with scene descriptions match or beat their VLM counterparts (p ≈ 0.0088),
with image-free contamination controls run.

- `EXTRACT:` the paired-encoding methodology (the experiment design our own
  A-vs-B decision needs) and the size of the gap.

### `vlms-see-squares-ocr-pathway`
**Can Vision-Language Models See Squares? Text-Recognition Mediates Spatial
Reasoning Across Three Model Families** — Feb 2026.
[arXiv:2602.15950](https://arxiv.org/abs/2602.15950) · **CAP · med · RQ4**

Same information content, different rendering: a text-symbol condition
dramatically outperformed geometric-square conditions, and **gridlines gave
negligible benefit** — evidence that VLMs route high-fidelity spatial reasoning
through an internal OCR pathway.

- `EXTRACT:` the condition design and result. Implication: alphanumeric labels
  on a render are read, plain geometry is not — which supports Family B's marks
  but undercuts drawing the *structure* (grids, axes) rather than labelling it.

### `embodiedbench-action-granularity`
**EmbodiedBench: Comprehensive Benchmarking MLLMs for Vision-Driven Embodied
Agents** — Yang et al. ICML 2025 (Oral).
[arXiv:2502.09560](https://arxiv.org/abs/2502.09560) ·
[repo](https://github.com/EmbodiedBench/EmbodiedBench) · **CAP · med · RQ4 ·
BENCH**

Organised along the action-abstraction axis (high-level task decomposition vs.
low-level translational/rotational control) with configurable textual *and*
visual designs, detection boxes, and multi-step/multi-view image inputs; best
model (GPT-4o) averages 28.9%, strong at high-level and weak at low-level.

- `EXTRACT:` the high-level/low-level performance split — it is the empirical
  form of "choice of control abstraction dominates capability"; the ablation
  knobs (visual in-context examples, camera resolution, detection boxes,
  multi-view input) as a menu of encoding ablations we can copy.

### `thinking-with-blueprints`
**Thinking with Blueprints: Assisting Vision-Language Models in Spatial
Reasoning via Structured Object Representation** — Ma, Sun, Yu, Wang, Chua,
Bian (MSRA). Jan 2026.
[arXiv:2601.01984](https://arxiv.org/abs/2601.01984) · **CAP/C · med · RQ4**

Has the model emit a **JSON blueprint** of object positions/sizes/attributes,
then reason over that structure — the "make the model build the text encoding
itself" option — with the notable ablation that naive CoT prompting often
*degraded* performance.

- `EXTRACT:` the blueprint JSON schema; the anti-shortcut augmentation; the
  CoT-hurts result (a caution against assuming more reasoning tokens help).

### `spatialvlm-quantitative-limits`
**SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning
Capabilities** — Chen et al. CVPR 2024.
[arXiv:2401.12168](https://arxiv.org/abs/2401.12168) · **CAP · low · RQ4**

Bounds the metric claims an encoding may ask for: even after training on 2B
spatial VQA examples, quantitative estimates land in-range only 37.2% of the
time (qualitative 75.2%).

- `EXTRACT:` the qualitative-vs-quantitative gap. Implication: never ask the
  orchestrator for numbers; ask for choices.

### `sibench-vsr-aggregator`
**SIBench-VSR** — aggregated harness over 3DSRBench, BLINK, MMSI-Bench,
OmniSpatial, SPAR-Bench, SPHERE-VLM, VSI-Bench and others.
[repo](https://github.com/song2yu/SIBench-VSR) · **CAP · low · BENCH**

One harness instead of per-benchmark plumbing, if we ever want to profile a
candidate orchestrator model before committing to an encoding.

---

## ENG — discrete action-set interfaces, and the blind-orchestrator architecture

### `llm-option-order-sensitivity`
**Large Language Models Are Sensitive to the Order of Options in
Multiple-Choice Questions** — Pezeshkpour & Hruschka. NAACL Findings 2024.
[arXiv:2308.11483](https://arxiv.org/abs/2308.11483) · **ENG · high · RQ4, RQ3
· BENCH**

A **13–85% performance swing from reordering options alone** (GPT-4 still 13.1%),
driven by uncertainty × positional bias, with mitigation by option layout and
calibration.

- `EXTRACT:` the magnitude and the mitigation (avoid putting the top-2 first
  and last; calibrate). Our view menu *is* a multiple-choice list; if its
  ordering is derived from geometry, ordering is not free.
- `BENCH:` mandates a **permutation test** in our benchmark — evaluate view
  choice under randomized action orderings, not one fixed listing.

### `llms-not-robust-mcq-selectors`
**Large Language Models Are Not Robust Multiple Choice Selectors** — Zheng et
al. ICLR 2024. [arXiv:2309.03882](https://arxiv.org/abs/2309.03882) · **ENG ·
med · RQ4**

Separates **token bias** (priors over the ID tokens `A`, `C`, …) from position
bias, and offers PriDe debiasing.

- `EXTRACT:` the token-vs-position decomposition — it decides whether our view
  labels should be `A/B/C`, integers, or descriptive names, and whether label
  identity leaks a prior into geometry-neutral choices.

### `how-many-tools-should-an-agent-see`
**How Many Tools Should an LLM Agent See? A Chance-Corrected Answer** — 2026.
[arXiv:2605.24660](https://arxiv.org/abs/2605.24660) · **ENG · high · RQ3, RQ4**

Chance-corrected treatment of action-catalogue size — the quantitative form of
RQ3's "keep the space small". Surrounding literature reports non-linear
degradation (≈50 tools: 84–95%; ≈200: 41–83%; ≈740: 0–20%), lost-in-the-middle
position effects, and that semantic overlap between options matters as much as
raw count.

- `EXTRACT:` the chance-corrected curve and where it inflects; whether the
  penalty is count or *similarity* (our views are near-identical items — the
  worst case for description distinctiveness).
- `CRITIC:` if similarity drives degradation, a geometry-conditioned space of
  20 nearly-identical view descriptions may be *harder* than 36 uniform cells
  despite being smaller. This is a live threat to the branch's premise.

### `anthropic-writing-tools-for-agents`
**Writing effective tools for agents — with agents** — Anthropic Engineering,
Sep 2025.
[anthropic.com/engineering/writing-tools-for-agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
· **ENG · high · RQ4 · blog**

The practitioner rules for exposing a dynamic action set to a Claude-family
orchestrator: search over listing, consolidation, namespacing, response design,
token efficiency, description engineering — plus the prototype→evaluate→optimize
loop for measuring how well the model actually uses the interface.

- `EXTRACT:` naming/description conventions and the evaluation loop; the
  "design for agents, not developers" framing applied to a view-selection tool.

### `anthropic-code-execution-with-mcp`
**Code execution with MCP** — Anthropic Engineering.
[anthropic.com/engineering/code-execution-with-mcp](https://www.anthropic.com/engineering/code-execution-with-mcp)
· **ENG · med · RQ4 · blog**

The "expose a thin surface and let the agent write code against it" pattern
(Cloudflare: ~2,500 endpoints behind two tools, ~1K tokens) — a third option
alongside dumping and querying the view space.

- `EXTRACT:` the thin-surface pattern; whether it is compatible with our
  requirement that selection stay discrete and auditable.

### `faea-demonstration-free-control`
**Demonstration-Free Robotic Control via LLM Agents (FAEA)** — Tsui, Fang, Hwu.
Jan 2026, v2 Jun 2026. [arXiv:2601.20334](https://arxiv.org/abs/2601.20334) ·
**ENG · high · RQ4**

An *unmodified* Claude Agent SDK loop reaching 84.9 / 85.7 / 96% on LIBERO,
ManiSkill3, MetaWorld with privileged state — evidence that the orchestrator
loop is adopt-not-build. Critically, it also reports the failure boundary:
**sub-centimetre precision tasks consistently fail** because the agent
identifies the goal correctly but cannot achieve fine alignment through discrete
action commands in a ReAct loop.

- `EXTRACT:` the tool/action schema used; the precision failure boundary and
  where it sits relative to our orthogonality tolerance (how many degrees of
  angular error does "most orthogonal view" allow?).
- `CRITIC:` privileged state access — the result does not transfer to a
  perception-limited setting without qualification.

### `saycan-affordance-skill-list`
**Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
(SayCan)** — Ahn et al. 2022.
[arXiv:2204.01691](https://arxiv.org/abs/2204.01691) · **ENG · med · RQ4, RQ3**

The origin of the enumerated-skill action space with feasibility scoring: skill
usefulness (LM) × skill feasibility (value function), re-queried step by step.

- `EXTRACT:` the two-factor scoring — the structural analogue of "useful view ×
  reachable view", which is precisely how our IK/collision filter should enter
  the interface rather than silently deleting options.

### `code-as-policies`
**Code as Policies: Language Model Programs for Embodied Control** — Liang et
al. ICRA 2023. [arXiv:2209.07753](https://arxiv.org/abs/2209.07753) · **ENG ·
med · RQ4**

The named critique of fixed skill lists (the abstraction "prevents LLMs from
directly influencing the perception-action feedback loop") and the code-API
alternative, at the cost of hand-crafted API design.

- `EXTRACT:` the argument against enumerated skills, and the API-engineering
  cost it trades for — the two poles our view interface sits between.

### `gemini-robotics-er-2-docs`
**Gemini Robotics-ER 2** — Google AI for Developers.
[docs](https://ai.google.dev/gemini-api/docs/robotics-overview) ·
[model card](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
· **ENG · med · RQ4 · docs**

The only shipped commercial spatial-interface contract: pointing, bounding
boxes, trajectories, image annotation as first-class model capabilities, plus
task orchestration over custom robot APIs with **blocking function calls** for
physical actions.

- `EXTRACT:` the pointing/annotation output formats (a vendor-blessed answer to
  "how does a VLM address a location"); the blocking-call orchestration
  contract; documented limits.
- Note the vendor-churn risk already recorded in our agentic-stack synthesis
  (ER 1.6 shutdown 2026-08-31).

### `anthropic-multi-agent-research-system`
**How we built our multi-agent research system** + **When to use multi-agent
systems (and when not to)** — Anthropic.
[claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)
· **ENG · high · RQ4 · blog**

The pro-case for our current blind-orchestrator + subagent design (orchestrator
+ parallel workers, +90.2% over single agent on their internal eval, ~15×
tokens), *and* the retraction-flavoured follow-up: multi-agent systems are often
used where a single agent would do better; the architecture suits parallel
read-heavy work, not tightly interdependent tasks.

- `EXTRACT:` the four-part subagent contract (objective, output format, tools/
  sources, task boundaries); the "tightly interdependent → single agent"
  criterion, and an honest judgement of which side view selection falls on.

### `cognition-dont-build-multi-agents`
**Don't Build Multi-Agents** — Walden Yan, Cognition.
[cognition.com/blog/dont-build-multi-agents](https://cognition.com/blog/dont-build-multi-agents)
· **ENG · high · RQ4 · blog**

The direct critique of the current design: context isolation between subagents
produces conflicting implicit decisions; principles are *share full agent
traces, not messages* and *actions carry implicit decisions*. Note the author's
later public revision ("we've found some setups that actually work" — one main
loop carries state, subagents are stateless narrow workers), which is close to
the shape we already have.

- `EXTRACT:` the two principles and the revision; map them onto
  orchestrator↔inspect-subagent to identify which decisions are implicitly made
  by the vision subagent and never surfaced.

### `raven-caption-bottleneck`
**RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal
Memory** — 2026. [arXiv:2606.25206](https://arxiv.org/abs/2606.25206) · **ENG ·
med/high · RQ4**

An explicit analysis of the semantic bottleneck our architecture depends on:
compressing observations into captions retains only what the captioner
verbalises, losing shape, spatial layout and implicit relations (their example:
"chicken thighs" degraded to "meat in a refrigerator").

- `EXTRACT:` the caption-bottleneck section and any quantification; whether they
  measure the loss or only argue it.
- `CRITIC:` this is the strongest published argument that the orchestrator
  should see. Weigh against `picture-worth-thousand-words`, which points the
  other way.

### `lohoravens-planner-reporter`
**LoHoRavens: A Long-Horizon Language-Conditioned Benchmark for Robotic
Tabletop Manipulation** — Zhang et al. 2023.
[arXiv:2310.12020](https://arxiv.org/abs/2310.12020) · **ENG · med · RQ4 ·
BENCH**

Directly compares the two architectures we are choosing between — a caption
"reporter" feeding a blind planner vs. an end-to-end multimodal planner — on a
tabletop benchmark, stating the tradeoff plainly (captions are simple but lose
information and amplify training bias).

- `EXTRACT:` the head-to-head numbers and the failure taxonomy.
- `BENCH:` a tabletop, long-horizon, language-conditioned protocol worth
  studying for our own harness design.

### `replan-perceiver-querying`
**RePLan: Robotic Replanning with Perception and Language Models** — Skreta et
al. 2024. [arXiv:2401.04157](https://arxiv.org/abs/2401.04157) · **ENG · med ·
RQ4**

The mitigation pattern for a blind planner: the LLM decides *what to ask* the
perceiver, asks simple specific questions, samples multiple answers, and
consolidates — i.e. the planner controls what gets verbalised, narrowing the
caption bottleneck. Also states the asymmetry that VLMs are much weaker at
language-only reasoning than LLMs.

- `EXTRACT:` the query protocol and consolidation; the LLM-vs-VLM reasoning
  asymmetry (an argument *for* keeping a strong blind reasoner in the loop even
  if something else sees).

### `vlms-cannot-plan-but-can-they-formalize`
**Vision Language Models Cannot Plan, but Can They Formalize?** — Sep 2025.
[arXiv:2509.21576](https://arxiv.org/abs/2509.21576) · **ENG · med · RQ4**

Finds the bottleneck is **vision, not language** — VLMs fail to capture an
exhaustive set of necessary object relations, and intermediate textual
representations (captions, scene graphs) only partially compensate with
inconsistent gains.

- `EXTRACT:` the vision-vs-language attribution and the partial-compensation
  result — the cleanest statement that *neither* family alone closes the gap,
  which is the case for Family C.

### `inspect-robots-eval-harness`
**inspect-robots** — open-source evals for physical AI; a frontier LLM drives
any embodiment through tool calls as a first-class policy, with (policy,
embodiment) pairs validated on action/observation spaces and semantics.
[repo](https://github.com/robocurve/inspect-robots) · **ENG · low · BENCH ·
repo**

- `EXTRACT:` the action/observation-space validation contract — a possible
  skeleton for our benchmark harness rather than rolling our own.

---

## Questions this branch must answer

Ordered so that the first four gate the pass-2 architecture decision. These are
the tasks for the reading threads and for my own later synthesis.

**Capability floor — what may an encoding assume at all?**

1. On the closest measured tasks (MMSI-Bench camera-camera / camera-object,
   ViewSpatial allocentric, SpinBench rotation), **is relating one viewpoint to
   another above chance for the models we would actually deploy?** If not, the
   action space must be flat and independently addressed — no adjacency, no
   "opposite face", no relative moves — and RQ3's topology ambition dies here.
2. Is "orthogonal to face 3" an allocentric query in the sense these benchmarks
   penalise, and **can it be restated egocentrically** (in the camera's current
   frame) without losing the geometry? What does the measured ego/allo gap cost
   us if not?
3. What is the **measured penalty for composing two spatial operations** in one
   step (FlipSet's compositional deficit)? Does that force a
   one-operation-per-turn interface with the geometry pre-composed offline?
4. Which encoding of a camera pose does the model handle best —
   **integers/angles, indices, or names**? `thinking-with-novel-views` claims
   numerical pose beats free-form language; `mapgpt` claims precise coordinates
   confuse the model. Resolve the contradiction and state the conditions under
   which each holds.

**Family choice — blind, seeing, or hybrid?**

5. On paired encodings of identical spatial information, **does text beat image
   or the reverse**, and under what conditions? (`picture-worth-thousand-words`
   and `vlms-see-squares` vs. `raven-caption-bottleneck` and
   `vlms-cannot-plan-but-can-they-formalize`.) Extract the conditions, not the
   verdict.
6. What exactly does a **caption hand-off destroy** that view selection needs —
   and can the orchestrator recover it by *asking better questions* (RePLan) or
   only by seeing?
7. For the seeing option, **what is the render?** A marked render of a partial,
   defective point cloud is not an RGB photo. Which papers render clouds for a
   VLM (GraspView, SPAZER, VLM3D, SeqVLM), what did they do about sparsity,
   holes, and colourless geometry, and does anything survive our ~4 cm
   fused-vs-single-view offset and glossy-surface dropout?
8. Does the **chicken-and-egg** problem have a published answer: to show the
   model a render from which it picks the next view, someone must choose the
   render's viewpoint. What do SeeGround's perspective adaptation, 3D-Mem's
   frontier snapshots, and Manipulate-Anything's tiled contact sheet each do,
   and what does each cost?

**Interface mechanics — if the space is a menu, how is the menu built?**

9. **How many options, and how similar may they be?** Reconcile the tool-count
   degradation curve with the observation that our candidate views are
   near-identical items — is the binding constraint count or description
   distinctiveness?
10. **How much does ordering matter?** Quantify the option-order and token-bias
    effects for a menu whose order is geometrically meaningful (face 1, face 2,
    …), and decide whether the benchmark must permute.
11. How should **infeasible views** (IK/collision-rejected) appear — silently
    removed, listed as unavailable, or scored down à la SayCan's affordance
    factor? Which choice preserves the model's ability to reason about coverage?
12. What is the **hierarchical/on-demand pattern** worth here — SayPlan's
    collapse/expand, Clio/FOUND-IT's task-driven granularity, or a query tool
    (Cypher / Agentic Scene Policies)? Our space is small enough to dump; is
    there any evidence that *dumping a small space* is worse than exposing it
    through a query?
13. What is the **stopping criterion** literature (Explore-until-Confident's
    conformal calibration) and can it give us a defensible "enough views" rule
    that the current design lacks?

**Evidence discipline (the standing critic mandate)**

14. For each system paper, **is the encoding claim ablated or merely
    demonstrated?** Produce a table: paper × (encoding compared against what) ×
    (delta) × (variance reported y/n). Papers with no comparison get flagged,
    not cited as evidence.
15. **What is the random baseline?** SPAZER's +3.6% over a random view and the
    classical result that random motion alone captures most of the active-vision
    gain must be answered before any VLM view-selection claim is believed. What
    random/uniform baseline does our benchmark need (the existing {h,v} sphere
    is the obvious one)?
16. Collect the **reported failure modes of VLM spatial choice** —
    wrong-face selection, left/right and ego/allo confusion, count and order
    sensitivity, mark/OCR dependence, ID hallucination, sub-cm precision
    failure — and hand them to Branch 2 as named benchmark test cases.

**Cross-branch handoffs**

17. To **Branch 5**: does any measured capability limit forbid a structure
    Branch 5 wants to build (e.g. a view adjacency graph)? RQ4 tractability
    pressure flows backwards here.
18. To **Branch 3**: MV-GEL's finding that CLIP-style semantic view ranking
    fails on textureless industrial parts, and its reliance on clean meshes — is
    geometry-conditioned view ranking still available on a defective survey
    cloud?
19. To **Branch 2**: every `BENCH:`-tagged entry above, plus the two metric
    candidates surfaced here (I-Perceive's **View Coverage IoU**; SeqVLM's
    **projected-area** view score) and the permutation-test requirement.

---

## Search trail

Thirty-eight web-search calls (each returning several result sets) plus four
arXiv landing-page checks, 2026-08-27. Abstracts and landing pages only — no
deep reads, per role. The session's web-search budget was exhausted at the end;
two ID verifications were completed via arXiv landing pages instead.

**Queries run** (grouped):

- *Visual prompting / marks*: `Set-of-Mark prompting GPT-4V visual grounding`;
  `PIVOT iterative visual prompting VLM robot action selection`;
  `scaffolding coordinates visual prompting GPT-4V grid coordinate`;
  `Coarse Correspondences visual prompting spatial temporal`;
  `MOKA marks keypoint affordance visual prompting`;
  `limitations set-of-mark prompting failure modes critique`;
  `visual prompting multimodal LLM survey 2409.15310`.
- *VLM active perception / view selection*: `VLM active perception next best
  view 2025 2026`; `AP-VLM 2409.17641`; `VLM chooses among rendered candidate
  viewpoints annotated arrows`; `render point cloud views for VLM to select
  viewpoint`; `VLM viewpoint selection ablation random baseline`;
  `Look Zoom Understand robotic eyeball 2511.15279`; `LLM agent robot
  inspection viewpoint action space eye-in-hand 2026`; `vision language model
  industrial visual inspection agent viewpoint planning defect`.
- *Structure as text*: `SayPlan 3D scene graph collapse expand`; `3D scene
  graph as text context for LLM agent planning survey`; `Structured Interfaces
  for Automated Reasoning with 3D Scene Graphs 2510.16643`; `MapGPT NavGPT
  topological map text prompt`; `ConceptGraphs`; `Clio task-driven 3D scene
  graph granularity`; `3D-Mem snapshot memory`; `Explore until Confident
  2403.15941`; `SeeGround query-aligned rendering`.
- *Capability bounds*: `VLM spatial reasoning benchmark 2025 2026 VSI-Bench
  BLINK 3DSRBench limits`; `VLM mental rotation perspective taking egocentric
  allocentric failure`; `MMSI-Bench`; `RoboSpatial reference frame`;
  `SpatialVLM SpatialRGPT quantitative`; `Can LLMs See Without Pixels
  2601.03590`; `text encoding versus image spatial reasoning blind text-only
  baseline`; `Thinking with Blueprints 2601.01984`; `Think3D 2601.13029`;
  `Thinking with Novel Views 2605.10588`; `EmbodiedBench`.
- *Interface engineering / architecture*: `Anthropic multi-agent research
  system`; `Don't build multi-agents Cognition`; `LLM agent tool count
  degradation action space design`; `Anthropic writing effective tools for
  agents`; `Code as Policies SayCan affordance discrete skill API`;
  `Demonstration-Free Robotic Control via LLM Agents 2601.20334`; `multiple
  choice option ordering position bias LLM`; `GUI agent accessibility tree
  versus screenshot ablation`; `blind LLM planner captions information loss
  vision subagent`; `Gemini Robotics-ER 2 docs`; `agentic visual reasoning
  thinking with images tool use survey 2026`.
- *ID verification via arXiv landing pages*: 2412.14171, 2404.12390,
  2309.16650, 2309.03882 — all four confirmed.

**Dead ends and gaps** (worth recording, per operating agreement):

- **No paper found** that combines text-rendered geometric state → LLM agent →
  viewpoint selection with coverage metrics for eye-in-hand inspection. The
  pieces exist separately. That gap is the project's opening — and also means
  no one has published the answer to RQ4 for us.
- **No survey scoped narrowly to active perception with foundation models** —
  it appears only as a chapter inside broader embodied-AI surveys
  (arXiv:2509.20021 §active perception; arXiv:2510.10991; arXiv:2407.06886).
  If the branch needs a spine, those chapters are it, at low priority.
- **No benchmark measuring whether VLMs can interpret textual azimuth/elevation
  descriptions.** Think3D's tool schema and Thinking-with-Novel-Views'
  instruction-format comparison are the only sources touching it. This is a
  genuine measurement gap that our own benchmark could fill — and arguably
  should, since it is precisely the current {h,v} interface.
- **No published result showing VLM viewpoint selection is statistically
  indistinguishable from random.** The nearest negative evidence is SPAZER's
  +3.6% margin, LogisticsVLN's near-null Choice-VLM ablation, and the classical
  finding that random active motion already yields ~75% of the NBV gain.
  Variance/CIs are absent across the board.
- **GUI-agent literature was probed** as the mature analogue of blind-vs-seeing
  (accessibility tree vs. screenshot) and is genuinely informative — the
  ordering of text-only / vision-only / hybrid **flips by platform and by
  metadata quality**, and compressing the text representation to 22% of tokens
  *improved* success by 5.1 points. Not listed as candidates (out of the
  robotics scope) but flagged here as a source of ablation-design precedent if
  a reading thread wants a fourth arm.

**Deliberately excluded** per branch scope: VLA policy learning and active
perception baked into VLA models (ActiveVLA 2601.08325, SaPaVe 2603.12193,
VG-AVS 2512.13250 — noted only as evidence that *prompting alone was
insufficient* for their authors); geometric NBV without a language interface
(Branch 4); generative novel-view synthesis as a method (deferred topic —
`thinking-with-novel-views` retained for its encoding result only); general
prompt engineering with no spatial action component.

---

## Standing critic notes for round 3

1. **"Is the capability demonstrated or demoed?"** — Family B is the worst
   offender: PIVOT, AP-VLM, GraspView and Manipulate-Anything all show a
   working system; only SPAZER, SeeGround, SCAFFOLD and MV-GEL report an
   encoding-level ablation. Question 14 exists to force that table.
2. **The similarity trap.** The branch's premise is that a smaller,
   geometry-conditioned space is easier for the model. The tool-scaling
   literature suggests degradation tracks *semantic overlap* as much as count —
   and 20 near-identical view descriptions may be harder than 36 uniform cells.
   If true, the whole RQ4 framing needs restating.
3. **The allocentric wall.** ViewSpatial, SpinBench and FlipSet independently
   report that object-centric frames are near chance while camera-centric ones
   are fine. "Orthogonal to face 3" is object-centric. Either the encoding
   restates it egocentrically, or the geometry is pre-composed offline and the
   model never reasons in that frame — the second option quietly reduces RQ4 to
   labelling, which may be the right answer and should be argued, not defaulted
   into.
4. **The chicken-and-egg of rendering.** Any seeing-orchestrator design must
   answer where the render's own viewpoint comes from. Nobody in this list
   solves it cleanly; three papers dodge it three different ways.
5. **Judge–benchmark circularity** (Branch 2's mandate, applied here): if the
   VLM both selects views and judges whether the evidence view is orthogonal,
   the loop is closed and the metric is meaningless. GraspView's
   VLM-as-utility-function is the archetype of that risk.
