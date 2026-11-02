# Pass 1, Round 1 — Branch Plan for the Literature Review

**Date:** 2026-08-27
**Role:** PI planning pass 1. This document decides the branches of the
benchmark-driven literature review on geometry-aware view planning for
AI-driven inspection.
**Contract:** [`../../../geometry-aware-view-planning.md`](../../../geometry-aware-view-planning.md)
— all scoping below derives from its Problem, Goals, RQ1–RQ4, Constraints, and
Candidate Directions sections. `todo.md` was an empty queue at planning time;
this plan supersedes it as the wave-1 structure.

---

## Planning rationale

The four RQs are roughly sequential (judge a view → select views → compose the
space → present to the VLM), and the methodology demands a benchmark before any
solution. The literature that feeds them, however, does not partition along the
RQs one-to-one:

- RQ1 and RQ2 both draw on the classical **view planning problem** literature,
  but with different extraction targets (score functions vs. selection
  strategies). They get separate branches with explicit dedup rules.
- RQ2 silently depends on a capability the contract takes as given: turning a
  **defective survey cloud into exploitable structure** (faces, axes, normals).
  No RQ names it, but every "box → faces, cylinder → orbit" rule assumes it.
  It gets its own branch — it is the highest-risk given in the whole design.
- RQ3 and RQ4 are adjacent but read different literatures: RQ3 is a
  **representation/structure** question (how viewpoint spaces are discretized
  and related — aspect graphs, view graphs), RQ4 is an **interface** question
  (how spatial structure is presented to a language model). Separate branches.
- The **benchmark** is a first-class deliverable, not a byproduct — it gets a
  branch, and every other branch is instructed to tag evaluation material for
  it.

Result: **6 branches.** The two candidate directions (analytical vs.
optimization) are deliberately *not* branches — both tracks consume the same
literature; the split is a solution-design decision for later passes, not a
literature partition. Each branch below notes which track it primarily informs.

## Branch overview

| # | Slug | Title | Feeds | Wave-1 priority |
|---|------|-------|-------|-----------------|
| 1 | `view-quality-scoring` | Scoring a single view: metrics and local optimality | RQ1, benchmark | **P0** |
| 2 | `benchmarking-view-planning` | Benchmarks and evaluation methodology for view planning | Methodology, all RQs | **P0** |
| 3 | `primitive-structure-from-survey-clouds` | Extracting exploitable geometry from defective survey clouds | RQ1, RQ2, constraints | **P1** |
| 4 | `view-set-selection-under-constraints` | Deriving the view set: coverage, selection, reachability | RQ2, RQ3 | **P1** |
| 5 | `structuring-the-viewpoint-space` | Composing and addressing geometry-conditioned view spaces | RQ3, RQ4 | P2 |
| 6 | `vlm-spatial-interfaces` | Presenting spatial action spaces to LLM/VLM agents | RQ4, RQ3 | P2 |

P0 = dispatch first (benchmark-driven methodology says the measuring stick and
the thing it measures come before everything); P1 = same wave if capacity
allows; P2 = may start in wave 1 but tolerate thinner first-round lists.

---

## Branch 1 — `view-quality-scoring`
### Scoring a single view: metrics and local optimality

**Rationale.** RQ1 is the keystone: an algorithmic judge of "how good is this
view for this surface region" unlocks both candidate tracks (analytical: derive
the pose that maximizes it; optimization: climb it). Albert's requirement —
evidence images should be the most orthogonal view of the inspected surface —
is exactly a view-quality criterion. Multiple mature literatures have
formalized pieces of this and nobody in the project has surveyed them yet.

**Feeds:** RQ1 directly; the benchmark (a quality score and a benchmark metric
share DNA — see circularity warning in Branch 2); both candidate tracks.

**In scope:**
- Formal viewpoint-quality criteria: incidence/grazing angle constraints,
  resolution / ground-sample-distance, field-of-view framing, focus/depth of
  field, visibility and self-occlusion — the "generalized viewpoint" sensor
  constraint tradition (Tarabanis-era MVP and descendants).
- Photogrammetry and optical-metrology view constraints: how surface normal vs.
  viewing direction affects measurement quality; RGB-D / structured-light /
  stereo depth error as a function of incidence angle and surface finish
  (directly relevant to the D405 and to the "depth dropout on glossy/dark
  surfaces" given).
- Best-view / viewpoint-selection scoring in graphics and vision: viewpoint
  entropy, view saliency, canonical view preference — for the scoring
  machinery, not the aesthetics motivation.
- NBV information-gain and utility functions — *only the score function
  component* (the selection loop belongs to Branch 4).
- Local viewpoint optimization: gradient-based / differentiable-rendering
  camera pose refinement, sampling-based local search over a view neighbourhood
  — anything that answers "is this view the best within its region, and if not,
  how to move".
- Image-space quality judging of a captured view (sharpness, exposure,
  defect-detectability vs. viewing angle in industrial inspection) — the
  contract allows judging "the captured image itself".

**Out of scope:** multi-view set selection and coverage (Branch 4); benchmark
datasets/protocols (Branch 2, though flag candidates); generative prediction of
unseen views for the judge (deferred — see Deferred topics); aesthetic view
selection for rendering except where the scoring math transfers.

**Key search terms:**
- "viewpoint quality metric", "best view selection", "viewpoint entropy",
  "viewpoint selection survey"
- "view planning sensor constraints", "generalized viewpoint", "MVP sensor
  planning Tarabanis"
- "incidence angle" + photogrammetry / laser scanning / structured light;
  "depth error viewing angle RGB-D"; "RealSense accuracy incidence angle"
- "next best view information gain formulation", "view utility function"
- "camera pose optimization differentiable rendering", "viewpoint optimization
  gradient"
- "defect detectability viewing angle surface inspection", "image quality
  automated optical inspection"

**Guidance for the mapping agent:**
- Spine: 1–2 sensor-planning/view-planning surveys (classic + recent), then
  branch into the metric families. Target 20–30 candidates, ranked.
- For every scored-metric paper, extract the *actual formula or computable
  definition* availability — a metric we cannot implement against a noisy PLY
  + known camera intrinsics is low value. Tag each candidate: PLY-space /
  image-space / both.
- Actively hunt the depth-sensor-error-vs-angle literature (metrology venues,
  sensor characterization papers) — it grounds "orthogonal is better" in
  measurement physics instead of intuition, which the critic will demand.
- Flag every paper whose evaluation section could seed the benchmark → note
  for Branch 2.

---

## Branch 2 — `benchmarking-view-planning`
### Benchmarks and evaluation methodology for view planning

**Rationale.** The contract mandates benchmark-driven research: a quantitative
framework for view quality / view selection before candidate solutions are
built. This branch surveys how the field measures view planning so our
benchmark is designed against precedent, not invented naively. It also owns
the sharpest methodological trap in the project: if the benchmark metric and
the RQ1 judge are the same function, every candidate "wins" by construction.

**Feeds:** the benchmark methodology directly; supplies the measuring
instruments for all RQ1–RQ4 experiments.

**In scope:**
- Evaluation protocols and metrics in NBV / view planning / active
  reconstruction: surface coverage %, reconstruction error vs. number of
  views, view-efficiency curves, per-region visibility metrics.
- Datasets and object sets used to evaluate view planning (ShapeNet-derived
  NBV benchmarks, scanned-object sets, industrial CAD sets) and their
  suitability for primitive-shaped objects with a survey-cloud (not
  ground-truth mesh) input.
- Simulation-based evaluation setups: synthetic ground truth from meshes,
  virtual cameras, noise injection to mimic real sensor defects (pose error,
  dropout, the ~4 cm fused-vs-single-view offset from the contract).
- Ground-truth definition strategies for "the orthogonal/best view" — how
  papers establish what the right answer is when the metric under test is
  itself contested; human expert labels vs. analytical ground truth on
  primitives.
- Benchmark design methodology: ablation protocols, baselines (the current
  uniform {h,v} sphere is our built-in baseline), statistical treatment,
  negative-result reporting.
- Evaluation of *agentic/VLM* view selection (how recent VLM-active-perception
  papers measure success) — shared with Branch 6, extraction here is the
  protocol, not the system.

**Out of scope:** the metric formulas themselves (Branch 1 owns those; this
branch owns how they are *used to compare methods*); 2D anomaly-detection
benchmarks (MVTec-style) except where they define defect-visibility ground
truth transferable to view scoring.

**Key search terms:**
- "next best view benchmark", "view planning evaluation metrics", "NBV
  comparison study"
- "surface coverage metric reconstruction", "reconstruction quality number of
  views"
- "active vision benchmark", "active perception evaluation"
- "viewpoint selection ground truth", "human best view agreement"
- "synthetic benchmark depth sensor noise model", "sensor noise simulation
  point cloud"
- "inspection planning evaluation", "coverage planning benchmark"

**Guidance for the mapping agent:**
- This list will be thinner than Branch 1's — dedicated benchmark papers are
  rare in this field. Expect 12–20 candidates; harvest evaluation *sections*
  of method papers as first-class entries (cite paper + note "evaluation
  protocol only").
- For each candidate, record: metric(s), dataset/objects, ground-truth source,
  baselines used, and whether the protocol survives our constraints (survey
  cloud input, reachability limits, discrete view selection).
- Explicitly collect at least 3 distinct answers to "how do you get ground
  truth for *best view* on a primitive shape" — analytical (face normal
  through centroid), metric-based, and human-labelled. The benchmark design
  will need to choose and defend one.
- Carry a standing critic note: **judge–benchmark circularity** — flag any
  candidate protocol where the evaluated method optimizes the evaluation
  metric directly, and how the paper handles (or ignores) it.

---

## Branch 3 — `primitive-structure-from-survey-clouds`
### Extracting exploitable geometry from defective survey clouds

**Rationale.** Every geometry-conditioned behaviour in the contract — box
scanned face-by-face, cylinder orbited normal to the surface — presupposes
that faces, axes, and normals can be recovered from the input geometry. The
input is not a CAD model: it is a fused multi-view PLY with mm-level pose
error, per-view dropout on glossy/dark surfaces, and a measured ~4 cm offset
between the fused cloud and single-view depth. The contract's non-goals bless
deterministic rules *at the primitive level*, which makes primitive
decomposition the load-bearing step. If robust decomposition of our clouds is
infeasible, both candidate tracks change shape — this branch exists to find
that out early from literature rather than late from failure.

**Feeds:** RQ2 (shape → views needs the shape), RQ1 (orthogonality scoring
needs trustworthy normals), RQ3 (a face/axis decomposition is a natural
skeleton for the action space); grounds all constraint handling.

**In scope:**
- Geometric primitive extraction from point clouds: RANSAC-family plane /
  cylinder / sphere / cone fitting, efficient-RANSAC descendants, region
  growing, and the survey literature around them.
- Shape abstraction: superquadric recovery, cuboid decomposition, learned
  shape abstraction — evaluated for noisy *partial* real scans, not clean
  meshes.
- Robust normal estimation on noisy, unevenly sampled, registration-degraded
  clouds; uncertainty quantification for fitted primitives and normals
  (an orthogonality score is only as good as its normal).
- Effects of the specific defect modes we have: registration/pose error in
  fusion, specular/dark dropout, systematic depth bias — literature on
  characterizing and mitigating them at the geometry-processing level.
- Failure detection: knowing *when* the decomposition is unreliable (residual
  thresholds, fitness scores), because the action-space design needs a
  fallback (plausibly the current uniform sphere).

**Out of scope:** full semantic segmentation / object recognition pipelines;
learned completion or generative repair of clouds (deferred unless a P0/P1
branch surfaces it as necessary); SLAM/reconstruction pipeline improvements
(the survey pipeline is a given, we consume its output); mesh reconstruction
for rendering.

**Key search terms:**
- "primitive fitting point cloud survey", "geometric primitive extraction
  survey", "efficient RANSAC point cloud shapes"
- "superquadric recovery point cloud", "cuboid abstraction", "shape
  abstraction primitives"
- "robust normal estimation noisy point cloud", "normal estimation survey"
- "plane segmentation noisy point cloud", "cylinder fitting partial scan"
- "point cloud registration error effect", "depth dropout specular surface",
  "structured light glossy surface"
- "primitive fitting uncertainty", "goodness of fit RANSAC shape"

**Guidance for the mapping agent:**
- Spine: one primitive-extraction survey + one normal-estimation survey, then
  method papers. Target 15–25 candidates.
- Prioritize papers with experiments on *real scanned* (not synthetic) data,
  and note reported tolerance to noise/incompleteness — we need methods that
  survive our defect profile, and most papers evaluate on clean ShapeNet.
- Tag each method: primitives covered (plane/cyl/sphere/cone/superquadric),
  robustness evidence, off-the-shelf availability (Open3D / CGAL / PCL /
  code release) — the auxiliary goal wants something integrable, not a
  research program.
- Record every quantified claim about normal error vs. point noise — Branch 1
  needs it to propagate uncertainty into the orthogonality score.

---

## Branch 4 — `view-set-selection-under-constraints`
### Deriving the view set: coverage, selection, reachability

**Rationale.** RQ2 asks how the *set* of best views is derived from geometry.
This is the classical model-based view planning problem — three decades of
formulations (set cover over candidate viewpoints, inspection path planning,
photogrammetric network design) that the project must not reinvent. Our twist
is the constraint stack: views must survive IK, collision (object inflation,
gripper+camera hull, table), and a hard minimum shell radius — so an
orthogonal view that exists geometrically may not exist kinematically, and the
selection method must degrade gracefully rather than fail.

**Feeds:** RQ2 directly; RQ3 (the structure of the selected set constrains how
it can be addressed); primarily informs the analytical track, with
optimization-track relevance where selection is posed as an objective.

**In scope:**
- The model-based view planning problem: candidate viewpoint generation from
  known geometry, set-cover / greedy / optimization selection, classical and
  modern surveys.
- Inspection planning for robots: CAD-based inspection planning, structural
  inspection path planning, industrial surface-inspection viewpoint planning
  (CMM, laser scanner, camera-based) — the closest applied neighbours to our
  task.
- Per-primitive deterministic view patterns: face-orthogonal grids, cylinder
  orbits, dome patterns in photogrammetric network design — precedent for the
  contract-blessed "deterministic rules at the primitive level".
- Reachability and kinematic feasibility coupled into view planning:
  capability/reachability maps, IK-filtered viewpoint candidates, eye-in-hand
  constraints, collision-aware viewpoint selection; what the field does when
  the ideal view is unreachable (nearest-feasible projection, re-weighting,
  constraint relaxation).
- Learned NBV / learned view selection — as *selection strategies* (their
  score functions belong to Branch 1); mainly to establish whether learning
  buys anything for primitive shapes, or whether analytical selection
  dominates in our regime.
- Redundancy/minimality trade-offs: how many views a method emits and how
  that is controlled (the VLM needs a *small* discrete set — RQ3/RQ4
  pressure applies backwards onto selection).

**Out of scope:** exploration/NBV for *unknown* scenes as an end in itself
(our geometry is surveyed up front; borrow mechanics only); path/trajectory
optimization between views (sequencing is downstream of this research);
coverage path planning for mobile robots/UAV surveying except where the
viewpoint-generation math transfers; view-quality metric internals (Branch 1).

**Key search terms:**
- "view planning problem survey", "model-based view planning", "viewpoint
  generation known object"
- "inspection path planning", "CAD-based inspection planning", "robotic
  surface inspection viewpoint planning"
- "set cover viewpoint selection", "art gallery problem 3D", "sensor
  placement coverage optimization"
- "photogrammetric network design", "camera network planning"
- "next best view survey", "learned next best view", "NBV deep learning"
- "capability map reachability manipulation", "reachability aware viewpoint",
  "eye-in-hand view planning collision"

**Guidance for the mapping agent:**
- Spine: the classical VPP survey lineage plus the most recent (2020s) NBV /
  inspection-planning surveys; then applied inspection systems. Target 20–30
  candidates.
- Separate candidates into: (a) viewpoint *generation* schemes, (b)
  *selection/optimization* over candidates, (c) *feasibility coupling* —
  the round-2 synthesis will want that structure.
- For applied inspection systems, record the platform (arm vs. UAV vs. CMM)
  and whether reachability was handled up front or by post-filtering — that
  design choice is one Anton will have to make.
- Hunt specifically for precedent on primitive-level view recipes
  (face-orthogonal capture, cylinder orbit) in metrology/photogrammetry
  practice — even standards or industrial guidelines count as sources here.

---

## Branch 5 — `structuring-the-viewpoint-space`
### Composing and addressing geometry-conditioned view spaces

**Rationale.** RQ3 is the bridge between geometry and agent: selected views
must become an *action space* — addressed, related to each other, and
consistent across object shapes. The current {h,v} sphere is trivially
addressable precisely because it ignores geometry; the literature on how
viewpoint spaces are discretized and structured (viewsphere tessellations,
aspect graphs partitioning viewpoint space by qualitative view equivalence,
view graphs in active vision) is the direct precedent for doing better without
losing navigability. This branch is representation-focused: no LLMs yet.

**Feeds:** RQ3 directly; RQ4 (whatever structure is chosen must be
presentable); constrains Branch 4 (set size/topology) from downstream.

**In scope:**
- Viewpoint-space discretizations: viewsphere tessellations (icosahedral,
  geodesic), view sampling strategies, and their addressing schemes.
- Aspect graphs and qualitative viewpoint-space partitions: the classical
  theory, its computability critiques, and modern revivals — the closest
  formal notion of "views structured by object geometry".
- View graphs / topological viewpoint structures in active vision: nodes as
  views, edges as camera moves, and how neighbourhood/adjacency is defined.
- Object-anchored view parameterizations: per-face frames, axis-aligned
  orbits, surface-parameterized camera manifolds — coordinate systems that
  make "orthogonal to face 3" a native address.
- Consistency across shapes: schemes that keep one addressing convention as
  the object changes (the RQ3 hard requirement); hierarchical addressing
  (object → primitive → view).
- Action abstraction in robot learning/planning as analogy: discrete
  skill/option vocabularies, parameterized action spaces — how other fields
  keep discrete action sets small, composable, and semantically meaningful.

**Out of scope:** how the structure is verbalized/rendered for an LLM (Branch
6); view *selection* mechanics (Branch 4); continuous camera control policies
(selection stays discrete per the contract).

**Key search terms:**
- "aspect graph survey", "aspect graph object recognition", "viewpoint space
  partition"
- "view sphere tessellation", "icosahedral viewpoint sampling", "canonical
  views object"
- "view graph active vision", "viewpoint adjacency graph"
- "object-centric camera parameterization", "camera manifold surface"
- "parameterized action space", "discrete action abstraction robotics",
  "skill vocabulary planning"
- "hierarchical action space", "topological map navigation graph"

**Guidance for the mapping agent:**
- This branch has the highest risk of a sprawling, unfocused list — hold it
  to 12–20 candidates and demand each one answers "how are views *addressed*
  and *related*", not merely "how are views used".
- Aspect-graph literature is old (late 80s–90s): take the seminal survey and
  the strongest critique, plus any modern revival — not two dozen variants.
- Actively look for negative evidence: reasons the field abandoned aspect
  graphs (combinatorial explosion) — that failure mode is exactly what RQ4's
  tractability requirement must avoid, and the critic needs the ammunition.
- Cross-tag with Branch 4: when a selection paper also defines an addressing
  scheme, this branch owns the addressing extraction.

---

## Branch 6 — `vlm-spatial-interfaces`
### Presenting spatial action spaces to LLM/VLM agents

**Rationale.** RQ4 is where the project's stated core problem lives: "there is
no reliable method for presenting spatial structure to an LLM." The current
agent navigates {h,v} cells because they are trivially textual. A
geometry-conditioned space needs either a textual/symbolic encoding the
orchestrator can navigate blind, or a redesign where the orchestrator sees
(the contract explicitly permits revisiting this). A fast-moving 2023–2026
literature — LLMs over 3D scene graphs, set-of-marks visual prompting,
VLM-driven active perception and view selection — bears directly on which of
those is viable, and VLM spatial-reasoning benchmarks bound what can be
expected at all.

**Feeds:** RQ4 directly; RQ3 (tractability pressure back onto the
representation); informs the architecture decision (blind orchestrator vs.
seeing orchestrator).

**In scope:**
- LLM/VLM agents operating over structured spatial representations: 3D scene
  graphs as LLM context, semantic maps for language-driven navigation,
  graph-of-rooms / topological abstractions fed as text.
- Visual prompting for spatial choice: set-of-marks, annotated-image action
  menus, "pick the labelled arrow/region" interfaces — the seeing-orchestrator
  alternative where candidate views are rendered and marked.
- VLM-driven active perception and viewpoint selection systems (2024–2026):
  works where a language model chooses the next view/camera action; their
  action-space encodings, success rates, and failure modes.
- Measured limits of VLM spatial reasoning: benchmarks on egocentric/allocentric
  reasoning, 3D relations, mental rotation — to bound what an encoding may
  assume the model can do.
- Discrete tool/action interfaces for LLM agents in embodied settings:
  affordance lists, enumerated skills with arguments — patterns for keeping a
  dynamic action set legible (names, counts, orderings, descriptions).
- Multi-agent architectures where a blind orchestrator consumes vision
  subagent reports — precedent and critiques, since that is the current
  design under review.

**Out of scope:** VLMs emitting metric poses or continuous control (contract
excludes it); general prompt-engineering literature without a spatial action
component; VLA policy learning (different problem class); the geometric
construction of the space itself (Branches 4–5).

**Key search terms:**
- "LLM 3D scene graph planning", "scene graph language agent", "semantic map
  language navigation"
- "set-of-mark prompting", "visual prompting spatial reasoning", "annotated
  image action selection"
- "VLM active perception", "LLM next best view", "language model viewpoint
  selection", "VLM camera control agent"
- "spatial reasoning benchmark VLM", "3D spatial understanding language
  model evaluation"
- "LLM agent discrete action space", "affordance grounding language agent"
- "blind LLM orchestrator vision tools", "multimodal agent orchestration
  perception"

**Guidance for the mapping agent:**
- Recency-weighted: most value is 2024–2026 (arXiv, CoRL/ICRA/RSS/NeurIPS);
  pre-2023 only for foundational grounding work. Target 18–28 candidates.
- Split the list into the three interface families up front: (a) text-encoded
  structure for a blind model, (b) rendered-and-marked views for a seeing
  model, (c) hybrid — every candidate assigned to one. The pass-2 decision is
  a choice among these families.
- For each system paper, extract the *actual encoding* (what the model
  literally received) and any quantitative comparison between encodings —
  claims without ablation get a critic flag, this subfield is demo-heavy.
- Record reported failure modes of VLM spatial choice (wrong-face selection,
  left/right confusion, count sensitivity) — these become benchmark test
  cases in Branch 2.

---

## Cross-branch rules

**Dedup / ownership.** One paper may appear in several branch lists with
different extraction targets; each overlap has a designated owner:
- NBV papers: score function → Branch 1; selection loop → Branch 4;
  evaluation protocol → Branch 2.
- Selection papers that define addressing schemes: addressing → Branch 5.
- VLM active-perception papers: system/encoding → Branch 6; evaluation
  protocol → Branch 2.
- Any paper's noise/robustness characterization of geometry input → Branch 3.

**Benchmark tagging.** Every branch tags candidates whose evaluation setup,
dataset, or ground-truth protocol could seed our benchmark. Branch 2
aggregates these tags in its round-2 synthesis.

**Constraint lens.** Every mapping agent evaluates candidates against the
contract's givens: survey-cloud input (not CAD), reachability limits,
discrete selection, fixed UR5e + D405 platform. A method that assumes clean
meshes, free-flying cameras, or continuous control is not excluded, but must
be tagged with the assumption it violates.

**Critic mandate (all branches).** Per the operating agreement, every thread
runs researcher + critic personas. Standing critic questions per branch:
1. Does the metric survive our sensor's noise? (B1)
2. Is the benchmark circular with the judge? (B2)
3. Does decomposition survive our defect profile? (B3)
4. What happens when the ideal view is unreachable? (B4)
5. Why did aspect graphs die, and are we rebuilding them? (B5)
6. Is the claimed VLM capability demonstrated or demoed? (B6)

## Deferred topics (explicitly not wave 1)

- **Generative unseen-view prediction (world models, e.g. NVIDIA Cosmos)** —
  contract marks it speculative, contingent on image-space judging proving
  necessary. Revisit only if Branch 1 concludes PLY-space judging is
  insufficient.
- **Full coverage of complex geometries** — contract non-goal.
- **View sequencing / trajectory optimization between selected views** —
  downstream of the action space; not needed to answer RQ1–RQ4.
- **Learned end-to-end view policies (RL over views, VLA-style)** — conflicts
  with the discrete-selection given and the benchmark-first methodology;
  reconsider only if Branch 4 shows analytical selection failing.
- **Point-cloud completion / generative repair** — only if Branch 3 shows
  raw decomposition infeasible on our defect profile.

## Round-2 expectations

Each branch's mapping agent delivers a ranked document list (counts above)
with per-candidate: full citation, one-line relevance claim, RQ tags,
extraction target, constraint-violation tags, benchmark tag. Anton
readthrough follows; reading threads (`threads/NN_<topic>/`) are scoped from
his feedback per the operating agreement — no thread dispatch without his go.
