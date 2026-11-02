# Branch 5: task-driven abstractions

**Pass 1, round 2 map. Date:** 2026-10-06.
**Tags:** READ (body read today), ABSTRACT (abstract, landing page or search snippet today), RECALLED (my memory, not checked today).
Numbers marked READ were taken from the HTML body through a fetch tool that summarises pages. Each one has its table or section named so it can be checked by eye.

## 1. Headline findings

**H1. Every formal notion of "sufficient" in this family has one shape: a representation is sufficient when every output the task needs can be computed from it as well as from the full history.**
The information state (Subramanian et al., Def. 3) asks this of reward and of its own next value. Filter equivalence (O'Kane and Shell, Def. 4) asks it of an output "colour" over all observation sequences. Predictive state representations ask it of a core set of "tests". For us, the planner's questions take the place of the reward, the colour or the tests. So our yardstick can be stated formally (section 4).

**H2. Minimality can be defined but not reached. "No bigger" has to be measured as a size, not proven.**
Minimising a combinatorial filter while it still gives the same outputs is NP-hard, and the decision version is NP-complete (O'Kane and Shell, abstract, READ). Information-theoretic map abstraction (Larsson et al.) swaps the hard minimum for a trade-off with weights β, γ, α (eq. 1, READ). So the open choice is a rate and distortion curve (size against questions answered), not a pass or fail test.

**H3. In working manipulation systems, a vision-language model picks what to keep at run time, for one task, and throws the abstraction away afterwards.**
ReKep, MOKA, CoPa, OmniManip and LENS all build their keypoints, constraints or pruned scene from the current instruction and current images. None of them keeps a memory that later questions could read. The persistent members of the family (PDDL state, QIF characteristics, claim memories) all fix their vocabulary in advance. Our questions arrive after the views were taken, which is the "post-hoc query" setting (Vista, ABSTRACT). That pulls against compressing by task alone.

**H4. Task-driven compression helps a language-model planner when the full scene is cluttered or partly irrelevant. It hurts when the scene is only partly observed.**
ViPlan: predicates grounded by a VLM beat direct VLM planning in Blocksworld (46% against 9%) but lose in household tasks (5% against 34%), "hindered by partial observability" (ABSTRACT). Our setting, with faces still unseen, is partly observed. The gains on the other side are real but modest: LENS, CAPruner and PLAHX (section 2G).

**H5. The abstraction, not the planner, is where these systems fail. Each re-grounds in its own narrow way.**
ReKep: VoxPoser 10.0%, ReKep with automatic keypoints 44.3%, ReKep with annotated keypoints 68.6% (Table 1, READ). Point tracking is the largest error source, followed by keypoint proposal (Fig. 4, READ). Ways to re-ground that we found: continuous tracking (ReKep), going back to an earlier stage (ReKep), adding the failure to the prompt and asking again (LENS), and human feedback in language (InterPreT). None of them detects that a stored claim is wrong without help.

**H6. One yardstick from retrieval-augmented generation can be used directly: "sufficient context".**
Definition: (Q, C) is sufficient iff some plausible answer to Q exists given only C. A one-shot LLM rater agreed with human labels 93% of the time (F1 0.935, 115 pairs, Table 1, READ). This gives a test per question that we can run on the 100 logged planner questions without ground truth.

## 2. Approach cards

### 2A. Keypoints and constraints proposed by a VLM (ReKep, MOKA, PIVOT, CoPa, OmniManip, KUDA, GeoManip)
- Atomic unit: a 3D keypoint (ReKep), or a small set of named points such as grasp, function, target and waypoints (MOKA, READ). Also an object part (CoPa) or an interaction primitive in the object's canonical frame (OmniManip, ABSTRACT).
- Kept: point positions plus constraints written as Python cost functions over the points (ReKep, ABSTRACT).
- Discarded: everything else in the image and depth. Proposals are made per instruction.
- Hierarchy: stages of a task (ReKep sub-goal and path constraints). No spatial hierarchy.
- Update, drift, conflicts: ReKep tracks points at 20 Hz and backtracks to an earlier stage when a path constraint breaks (Sec. A.1, READ). OmniManip resamples primitives and checks them with a VLM (ABSTRACT).
- Known vs unknown: not addressed.
- Identity and merging: within a run, tracking handles it. ReKep proposes candidates by clustering DINOv2 features per SAM mask (k=5) and filters candidates closer than 8 cm (READ).
- Links: the point is drawn as a numbered mark on the image, the VLM refers to it by number, and the code holds its 3D position. That is "the model refers, code computes".
- Frame: world for ReKep. Canonical object frame for OmniManip. Image with depth lifting for MOKA and PIVOT.
- Reader: a VLM that sees the marked image. Not a text-only reader.
- Cost: one VLM call per stage. Real-time optimisation (ReKep).
- Training: none for ReKep, MOKA, PIVOT and CoPa. RoboPoint fine-tunes a VLM on 1432K synthetic image-question pairs (ABSTRACT).
- Numbers: ReKep in Table 1, see H5 (7 tasks, 10 trials each, READ). MOKA Table I: zero-shot from 0.5 to 1.0 per subtask, about the same as Code as Policies and VoxPoser. Distilled and in-context variants reach 0.7 to 1.0 (10 trials, READ). CoPa reaches 63% over 10 tasks, ahead of VoxPoser (ABSTRACT). RoboPoint is +21.8% in affordance accuracy and +30.5% in downstream success over GPT-4o and PIVOT (ABSTRACT).
- Used where: papers and research code (ReKep and RoboPoint on GitHub). I found no product use.
- Questions it can answer: where to grasp or touch, and where something must go relative to something else. Questions it gives up: what is written there, what else is on this face, which view showed it.

### 2B. Learned dense descriptors and category keypoints (Dense Object Nets, kPAM, Neural Descriptor Fields, KALM)
- Atomic unit: a per-pixel or per-point descriptor (DON, NDF), or a semantic keypoint per category (kPAM). All RECALLED. NDF is a field over SE(3) that is learned from 5 to 10 demonstrations (ABSTRACT).
- Kept: correspondence across instances and poses. Discarded: appearance detail.
- Identity: correspondence is the whole point, so the same point on two instances matches.
- Frame: category or canonical frame. NDF is SE(3)-equivariant (ABSTRACT).
- Training: needed (self-supervised for DON and NDF). KALM proposes keypoints with an LLM or VLM and checks them against a few demonstrations (ABSTRACT).
- Used where: papers. kPAM-style category keypoints existed in TRI and MIT code (RECALLED).
- Fit for us: weak, because training is needed. The idea of correspondence in a canonical frame is relevant to RQ11 and RQ13.

### 2C. Marks as a shared reference (Set-of-Mark, also the marking step in MOKA and PIVOT)
- Atomic unit: a segmented region with a numeric label drawn on the image.
- Kept: an ID that both the model and the code understand. Discarded: nothing, because the image is still shown.
- Links: the strongest pattern in this family for RQ12. The model answers "mark 3", the code holds the mask and its 3D points.
- Numbers: GPT-4V with SoM, zero-shot, beats the best fully fine-tuned referring-expression model on RefCOCOg (ABSTRACT).
- Reader: needs a vision reader. For our blind planner the mark would be a text ID that the vision agent and the IWM share.
- Used where: an open-source Microsoft repository, and widely copied as a prompting technique (ABSTRACT, RECALLED).

### 2D. Perception APIs and scene text for LLM planners (Code as Policies, SayCan, Inner Monologue, TidyBot)
- Atomic unit: a named object with a few attributes that are fetched on demand.
- Code as Policies, perception functions (READ, paper appendix): `detect_object`, `get_obj_names`, `get_visible_obj_names`, `is_obj_visible`, `get_pos`, `get_bbox`, `get_obj_bbox_xyxy`, `get_obj_bbox_area`, `get_segmask`, `get_color_rgb`, `get_corner_name`, `get_side_name`, `get_loc_names`, `get_robot_pos_and_angle`. The context line is `objs = [...]`.
  This is the de facto list of questions a tabletop manipulator asks: which objects exist, is it visible, where is it, how big, what colour, which side or corner. Note `is_obj_visible`: the API separates "visible now" from "exists".
- Inner Monologue feedback types (Sec. 3.3, READ): success detection, passive scene description, and active scene description, where the LLM asks a VQA question. Simulation (Table 1): object feedback alone 52%, plus success 65%, plus scene 63%. Real kitchen (Table 3): SayCan 30.8%, plus success 48.7%, plus object and success 60.4% (READ).
  The active scene description is our phase-1 design: a question to the vision agent, answered from one image.
- TidyBot: an LLM summarises a few examples into a rule. 91.2% on unseen objects in the benchmark, 85.0% of objects placed in the real world (ABSTRACT).
- Kept: names, positions, success flags. Discarded: everything not in the API.
- Hierarchy, known vs unknown, identity: not addressed, apart from `is_obj_visible`.
- Who chooses what to keep: the designer, through the API.
- Used where: Code as Policies notebooks are open source. The API pattern is common in LLM robotics stacks (RECALLED).

### 2E. Symbolic state and predicates grounded by a VLM (PDDL state, VisualPredicator, pix2pred, InterPreT, ViPlan, LLM-State)
- Atomic unit: a ground predicate, for example `On(a,b)`, with a truth value.
- Kept: the truth values of a compact set of predicates. pix2pred proposes many predicates with a VLM and learns a compact subset from demonstrations (ABSTRACT). Discarded: all geometry and appearance.
- Who chooses what to keep: an objective (pix2pred, VisualPredicator: whatever makes planning on the demonstrations work), a human through language feedback (InterPreT: 73% in simulation and 40% in the real world in the hardest generalisation setting, ABSTRACT), or an LLM (LLM-State: an open set of attributes per object that the LLM extends and updates, ABSTRACT).
- Update: re-evaluated from the current image at each step (pix2pred at test time, ABSTRACT).
- Known vs unknown: a closed-world assumption is the default in PDDL. ViPlan names partial observability as the failure mode of VLM-as-grounder (ABSTRACT).
- Identity: object symbols are given. Not addressed.
- Links: words only. Geometry lives in the skills.
- Reader: a classical planner, or an LLM as in LLM+P, DELTA and PLAHX.
- Training: learned from demonstrations (pix2pred, VisualPredicator), or none (InterPreT, LLM-State).
- Used where: PDDL state in open-source ROS 2 planning (PlanSys2, RECALLED). The predicates grounded by a VLM appear in papers only.
- Questions it can answer: "is X true", "what holds now", goal checks. Questions it gives up: where, which view, what does it say, unless these are coded as predicates.

### 2F. World models written as code (WorldCoder, PoE-World, OCM)
- Atomic unit: a Python program that models transitions (WorldCoder, PoE-World) or object classes (OCM, July 2026).
- OCM keeps object knowledge as Python classes and reveals it step by step: compact summaries first, full source on demand (ABSTRACT). That is a reading interface worth noting for RQ8.
- Relevance: low for us. These programs model dynamics, and our world is static.
- Used where: papers.

### 2G. Task-driven pruning of an existing representation (PLOI, LENS, CAPruner, PLAHX, SayPlan; Clio belongs to branch 3)
- Atomic unit: whatever the underlying representation holds (objects, edges of a scene graph).
- Kept: the subset judged relevant to the current task.
- Who chooses: a trained GNN that scores how much each object matters (PLOI, AAAI 2021, ABSTRACT); a VLM prompted to prune or merge (LENS, GPT-4o, READ); a trained relevance scorer (CAPruner); the LLM itself expanding a collapsed hierarchy (SayPlan "semantic search", ABSTRACT).
- Re-grounding: LENS adds the failure feedback to the prompt and asks again (Algorithm 1, READ). PLOI drops its importance threshold and replans (RECALLED).
- Numbers:
  - LENS (July 2026): VLA success falls from 0.85 to 0.5 with clutter, and LENS brings it back to 0.69. On hardware the VLA reaches 0.5/0.3/0.3/0.7 against a baseline of 0.2/0.0/0.0/0.0 (Table 1). Model-based control succeeds in 39/45 trials against 17/30 (Fig. 5) (READ).
  - CAPruner (April 2026): ScanRefer Acc@0.25 rises from 52.5 to 55.0 (1B model) and from 60.2 to 61.7 (8B), with tokens down 34% (582 against 882) (Table 1, Fig. 4, READ). Its LLMs are fine-tuned.
  - PLAHX: +21.47% success and 13% fewer tokens than baselines (ABSTRACT).
  - Shi et al., ICML 2023: LLM accuracy "dramatically decreased" by irrelevant context (ABSTRACT, no number in the abstract).
- Used where: papers.
- Fit for us: these methods prune for a task that is already known. They do not decide what to store before the question is asked.

### 2H. Shape primitives and object-level landmarks (CubeSLAM, QuadricSLAM, SuperDec, NOCS, Scan2CAD, MoveIt collision objects)
- Atomic unit: a cuboid (CubeSLAM), a dual quadric built from 2D detections in several views (QuadricSLAM), a superquadric with a few parameters (SuperDec, ICCV 2025), a canonical coordinate map with a 6D pose and size per category (NOCS), or a retrieved and aligned CAD model (Scan2CAD, RECALLED). All ABSTRACT unless marked otherwise.
- Kept: pose, extent and shape class. Discarded: surface detail, texture and text.
- Update: factor-graph SLAM jointly with the camera poses (CubeSLAM, QuadricSLAM), so object landmarks correct the camera drift as well (ABSTRACT).
- Known vs unknown: the primitive fills the unseen side with a prior shape. This is one answer to the contract's open item "parts not seen, shape unknown". The fill is not marked as unseen.
- Identity: data association of detections to landmarks (RECALLED).
- Frame: world landmarks, plus the object's canonical frame (NOCS).
- Used where: MoveIt planning scenes use boxes, cylinders, spheres and meshes as collision objects (RECALLED). Cuboid outputs are standard in driving perception (RECALLED). SuperDec was shown on a Spot with an arm (ABSTRACT).
- Questions it can answer: where, how big, which way a face points. Questions it gives up: what is on the face.
- Fit for us: our phase-1 fitted box already belongs to this card.

### 2I. Real-to-sim digital twins as compression into simulator state
- Atomic unit: a simulator asset with pose, joint parameters and physics parameters. Examples: GRS (SAM2 segments, a VLM describes them, assets are matched), Real-is-Sim (a twin kept in sync with measurements), and IKER (rewards from VLM keypoints, 2502.08643). All from search snippets (ABSTRACT).
- Kept: what the simulator needs. Discarded: appearance beyond the matched asset.
- Fit for us: low. The goal is to simulate dynamics, and our world is static.
- Used where: papers. Isaac Sim pipelines (RECALLED).

### 2J. Information-theoretic map abstraction (Q-tree search; Larsson et al., ICRA 2023; Jin and Larsson, Sept 2026)
- Atomic unit: a node of a quadtree or octree at a chosen resolution.
- Kept: the multi-resolution tree that maximises Σ βᵢ I_Yi(T) − Σ γⱼ I_Zj(T) − α I_X(T). Here Yᵢ are semantic classes to keep, Zⱼ classes to remove, and X the finest cells (eq. 1, READ).
- Who chooses: an objective, with the relevant variables named by the designer.
- Hierarchy: explicit, and the resolution is chosen per region.
- Known vs unknown: probabilistic semantics carry classification uncertainty (READ, abstract of the ICRA 2023 paper).
- Cost: exact search is a dynamic-programming recursion. The September 2026 paper replaces it with an anytime sampler in the style of MCTS (ABSTRACT).
- Used where: papers.
- Fit for us: the only working system found that compresses a map by an explicit relevance objective. Its relevant variable is a class label, not a question.

### 2K. Abstractions specific to inspection (QIF characteristics, taught ROIs, claim memories, VISION)
- QIF (ANSI QIF 3.0, ISO 23952:2020): an XML information model that ties measurement plans and results to part geometry and PMI (overview page, ABSTRACT). Its internal pattern is characteristic and feature, each split into Definition, Nominal, Item and Actual (RECALLED, not verified today). This is a shipped "what to verify" list tied to geometry, used by CMM software (RECALLED).
- Taught regions of interest in industrial vision (Cognex, Keyence style): a fixed region in the part frame with a tool and a pass or fail rule (RECALLED).
- INSPECT 2609.20615: already known, not repeated here.
- VISION culvert inspection (2509.21370, quadruped): a VLM proposes ROIs with confidence and a rationale, then the robot moves closer. Agreement with experts went from 61.4% to 80% after repositioning (ABSTRACT).
- Kept: the list of things to check, each with a status. Discarded: everything not on the list.
- Who chooses: the designer or a CAD/PMI author (QIF), or a VLM at run time (VISION).
- Fit for us: the closest to a claim memory. QIF's split of plan against results corresponds to "pending" against "supported or contradicted".

### 2L. Belief state and active perception (theory card)
- Atomic unit: a distribution over hidden state (a POMDP belief), or any information state σ(h) that satisfies P1 and P2 (Subramanian et al., Def. 3, READ).
- Kept: exactly what predicts reward and the state's own next value. Everything else is redundant by definition.
- Bajcsy, Aloimonos and Tsotsos: "An agent is an active perceiver if it knows why it wishes to sense, and then chooses what to perceive, and determines how, when and where to achieve that perception" (Sec. 1, READ). The "why" decides what is worth keeping.
- Majumdar, Mei and Pacelli (IJRR 2023): TRIP, the task-relevant information potential of a sensor, gives an upper bound on achievable reward (ABSTRACT). Pacelli and Theodorou (2026) tighten such bounds in a self-consistent way (ABSTRACT).
- Used where: belief-space planning is used in papers. Kalman and particle filters are information states in every shipped navigation stack (RECALLED).

## 3. Per research question

- **RQ4, atomic block.** The family splits three ways: a point with a meaning (2A, 2B), a predicate with a truth value (2E), and a primitive with pose and extent (2H). Inspection adds a characteristic with a status (2K). Theory says the atom does not matter, only that the set predicts the answers (2L).
- **RQ5, hierarchy.** Mostly silent. The exceptions are task stages (ReKep), multi-resolution trees chosen by an IB objective (2J), and SayPlan's collapsed graph (branch 3).
- **RQ6, drift-free update.** For geometry: tracking (ReKep), and SLAM where object landmarks correct camera drift (CubeSLAM, QuadricSLAM). For symbols: re-evaluating from the current image. Theory adds the δ condition of AIS: an incremental update must match the state rebuilt from the full history (section 4).
- **RQ7, who resolves conflicts.** Variants found: a VLM that re-prompts with failure feedback (LENS), a human in language (InterPreT), an objective (pix2pred, Larsson). Branch 4 owns this.
- **RQ8, contract.** Code as Policies gives the clearest out-contract: a short list of query functions (2D). OCM shows progressive disclosure. ViPlan shows that the choice "symbols out" against "raw image out" changes success by a factor of 4 to 7 in either direction, depending on the domain.
- **RQ9, what to keep.** This branch's main question. Answer from theory: keep whatever leaves the answers to the planner's questions unchanged (P1, filter colour). Answer from practice: keep IDs, positions and status, and leave the pixels outside to fetch on demand. The Inner Monologue "active scene description" is exactly that fetch. Section 4 gives the measurable forms.
- **RQ10, known vs unknown.** Mostly silent. Exceptions: `is_obj_visible` in Code as Policies, partial observability as the named failure in ViPlan, claim status (pending) in INSPECT and QIF, and probabilistic semantics in Larsson. Primitives fill unseen parts with priors and do not mark them.
- **RQ11, identity and merging.** Keypoint and descriptor correspondence (2B), the 8 cm proposal filter (ReKep), landmark data association (2H). No method to undo a wrong merge was found in this family.
- **RQ12, linking geometry, image and words.** Marks: one ID shared by the model, the image and the 3D points (Set-of-Mark, MOKA, ReKep). Predicates: a word evaluated on an image (pix2pred). This is the strongest contribution of the family.
- **RQ13, frame.** Canonical object frame for anything that must survive a pose change (NOCS, NDF, OmniManip). World frame for constraints (ReKep). The side and corner names in Code as Policies are an example of translating into the asker's terms.

## 4. Yardstick candidates (special interest d)

Notation. H is the raw history (images, depth, poses). S = σ(H) is the IWM. Q is the set of planner questions (the 100 logged, grouped into 8 kinds). ans(q, ·) is the answer to q from that input.

**Y1. Information state and approximate information state** (Subramanian, Sinha, Seraj, Mahajan, JMLR 2022, arXiv 2010.08843, READ).
- Exact form (Def. 3). P1: E[R|H=h, A=a] = E[R|Z=σ(h), A=a]. P2: the law of Z_{t+1} given the history equals its law given Z_t (self-prediction).
- Approximate form (Def. 7). AP1: |E[R|h,a] − r̂(σ̂(h),a)| ≤ ε. AP2: d_F(μ, ν) ≤ δ, an integral probability metric between the true and the predicted next state. Theorem 9 bounds the value loss by ε and δ, accumulated over steps.
- How to transfer it: replace R by the answer. ε is the rate at which ans(q, S) disagrees with ans(q, H) (or with ground truth) over Q. δ asks whether S updated with a new view equals S rebuilt from all views, or whether the predicted next IWM matches the actual one.
- To measure: (i) the answer agreement rate per question kind, (ii) the difference between the incrementally updated and the rebuilt structure after replaying each recorded run.

**Y2. Filter equivalence and reduction** (O'Kane and Shell, ICRA 2013, READ).
- Def. 3: a filter is a graph of information states with a colouring c: V → N+. "The coloring describes the task performed by the filter."
- Def. 4: two filters are equivalent on a language L if every observation sequence in L reaches states with the same colour.
- Minimal means the fewest information states while staying equivalent. That problem is NP-hard.
- How to transfer it: the observation sequences are our recorded runs, and the colour is the vector of answers to Q. Two candidate structures are equivalent for our task if they give the same answers on every recorded run.
- To measure: replay all 25 runs into both structures and compare the answer vectors. Size is then compared only between equivalent candidates.

**Y3. Core tests of a predictive state representation** (Littman, Sutton, Singh, NIPS 2001, ABSTRACT, search snippet).
- The state is the vector of predicted outcomes of a core set of tests. It is a sufficient statistic when the core predictions determine the prediction of every other test.
- How to transfer it: find a core question set Q_c (for example "list details", "coverage per face") whose answers determine the answers to all of Q.
- To measure: whether ans(Q_c) determines ans(Q \ Q_c) across the logged runs. This gives a candidate minimal interface.

**Y4. The hierarchy of state abstractions** (Li, Walsh, Littman, ISAIM 2006; landing page ABSTRACT, details RECALLED).
- Five abstractions, from finest to coarsest: model-irrelevance, Q^π, Q*, a*, π*. Finer ones preserve more and suit more uses.
- Value equivalence (Grimm, Barreto, Singh, Silver, NeurIPS 2020, ABSTRACT): two models are equivalent if they give the same Bellman updates for a chosen set of policies and functions.
- How to transfer it: two levels of our yardstick. (a) Answer-irrelevance: merge raw states that give the same answers to all of Q. (b) Decision-irrelevance: merge raw states after which the planner takes the same next action. (b) is coarser and cheaper.
- To measure for (b): the planner's next action given S against given H, on the same run. Phase 1 logs have these decisions.

**Y5. Information bottleneck, rate and distortion** (Pacelli and Majumdar, 1809.07874 and 2002.01428, ABSTRACT; Larsson et al. ICRA 2023, eq. 1, READ).
- Larsson: max over T of Σ βᵢ I(T; Yᵢ) − Σ γⱼ I(T; Zⱼ) − α I(T; X). Pacelli and Majumdar trade expected task cost against the information that the representation keeps about the full state (exact form RECALLED).
- How to transfer it: Y is the answer to each question kind, and X is the raw data. Mutual information needs a probability model we do not have, so the practical stand-in is an empirical rate and distortion curve: size of S (bytes stored, tokens returned per query) on one axis, answer accuracy over Q on the other.
- To measure: tokens and bytes, and accuracy, for several candidate structures. The yardstick is then the knee of the curve, not one threshold.

**Y6. Upper bound from task-relevant information (TRIP)** (Majumdar, Mei, Pacelli, IJRR 2023, ABSTRACT).
- An upper bound on achievable expected reward, set by the sensor and by the best open-loop reward.
- How to transfer it: an upper bound on how many questions can be answered from the views taken. It separates "the IWM lost it" from "the cameras never saw it".
- To measure: the questions answerable from raw H (by a human or an oracle VLM), used as the ceiling for S.

**Y7. Sufficient context** (Joren et al., arXiv 2411.06037, READ).
- Definition, verbatim: "An instance (Q,C) has sufficient context if and only if there exists an answer A′ such that A′ is a plausible answer to the question Q given the information in C." Remarks cover multi-hop questions and ambiguous questions or contexts.
- An automatic rater (Gemini 1.5 Pro, one-shot) reached 93% accuracy on 115 human-labelled pairs (Table 1). Even with sufficient context, models gave wrong answers more often than they abstained.
- How to transfer it: C is what the IWM returns for q. This needs no ground truth, so it works on the logged questions today.
- To measure: the fraction of Q with sufficient returned context, the tokens of C, and whether the planner abstains or invents an answer when C is insufficient. The last one links to RQ10.

**How they compose (for the architect to judge):** Y2 or Y7 test sufficiency per question. Y1's δ tests that updates do not drift. Y5 is the compression axis. Y6 sets the ceiling. Y4(b) is a cheaper, coarser variant. All need the logged Q, replayable runs, and a reference answer, from raw images or from a human.

## 5. What does not transfer, and why

- **Abstractions made per task and thrown away** (2A, 2G): our questions come after the views were taken, so pruning by a task known in advance cannot decide storage. They can still shape a reading view over a fuller store.
- **Readers that see images** (2A, 2C, Set-of-Mark): the planner sees no images. Marks transfer only as shared text IDs between the vision agent and the IWM.
- **Keypoint geometry at about 10 mm cross-view error**: ReKep's 8 cm proposal filter and constraint costs assume consistent 3D. With 10 mm offsets per view (19 mm at p90), two nearby keypoints from different views can swap or merge. Position alone separates features only at about 3 cm (contract, section 7).
- **Methods that need training** (DON, kPAM, NDF, RoboPoint, pix2pred, VisualPredicator, CAPruner's fine-tuned LLMs, PLOI's GNN): excluded by scope. The ideas carry over, the systems do not.
- **Code world models, real-to-sim twins**: built to predict dynamics. Our world is static.
- **The closed world of PDDL**: it contradicts RQ10 ("not there" against "not looked").
- **Information-theoretic objectives in exact form**: they need distributions over states and answers. Only the empirical curve form (Y5) is usable.
- **Single object**: object-level pruning (LENS, PLOI, CAPruner) has little to prune with one object. Its analogue is pruning details or faces.

## 6. Ranked reading list for deep reading

1. Subramanian et al., AIS, https://arxiv.org/html/2010.08843 . Why: the most precise sufficiency definition with an error bound. Extract: Defs. 3 and 7, Theorem 9, the recursive-update version (Def. 4 or 5, not read today), and how they choose the metric d_F.
2. Joren et al., Sufficient Context, https://arxiv.org/html/2411.06037 . Why: a yardstick per question that we can run. Extract: the rater prompt, label guidelines, and how abstention is scored.
3. O'Kane and Shell 2013, https://cse-robotics.engr.tamu.edu/dshell/papers/icra13reduction.pdf . Why: equivalence over observation sequences, colouring as the task. Extract: Algorithm 1 (equivalence check), the conflict-graph construction. Also the later cover-filter work, https://arxiv.org/pdf/2002.07153 , and the FPT algorithm, https://arxiv.org/pdf/2309.06664 (not opened).
4. Larsson et al. ICRA 2023, https://natanaso.github.io/ref/Larsson_OctreeAbstraction_ICRA23.pdf , and Jin and Larsson 2026, https://arxiv.org/abs/2609.27330 . Why: a relevance objective over a hierarchical spatial structure. Extract: how I_Y is computed per node, and β sweeps.
5. ViPlan, https://arxiv.org/abs/2505.13180 . Why: the only matched comparison of a symbolic state against direct VLM planning. Extract: the failure analysis for partial observability.
6. ReKep, https://arxiv.org/html/2409.01652 . Why: the error breakdown. Extract: Fig. 4 fractions, keypoint proposal details (remaining 36k characters not read).
7. LENS, https://arxiv.org/html/2607.19633 . Why: VLM-chosen abstraction with a feedback loop, 2026. Extract: the prompts, and how often feedback rounds fix wrong pruning.
8. Code as Policies, https://arxiv.org/html/2209.07753 , appendix I to K. Why: the de facto query list. Extract: the full API signatures, to set against our 8 question kinds.
9. Li, Walsh, Littman 2006 (dblp: https://dblp.org/rec/conf/isaim/LiWL06.html). Why: the order of abstractions. Extract: the five definitions, verified rather than recalled.
10. QIF 3.0 specification, https://qifstandards.org/download/ . Why: a shipped schema for "characteristics to verify". Extract: Characteristic Definition, Nominal, Item and Actual, and how they link to features and results.
11. Majumdar, Mei, Pacelli, https://arxiv.org/abs/2202.00129 . Why: a ceiling set by the sensor. Extract: whether TRIP can be estimated for a discrete question set.

## 7. Gaps and open questions

- **Not found:** any yardstick defined over a set of natural-language questions for a robot's world representation. The closest are RAG's sufficient context (no robot) and information-state theory (no language). Joining the two looks open.
- **Not found:** a task-driven method that decides what to store before the task is known, with a guarantee. Post-hoc query papers (Vista) handle this with segmentation and retrieval, not with a sufficiency notion.
- **Not found:** direct numbers comparing a task-driven abstraction with a task-agnostic map for a text-only LLM planner on the same robot task. ViPlan is the nearest. MIT 2510.16643 (known) compares interfaces, not compression.
- **Not verified today:** the QIF internal structure, the Li-Walsh-Littman definitions, the exact Pacelli-Majumdar objective, and the DON and kPAM details. All are tagged RECALLED.
- **Not opened:** KITE, pix2pred's numbers, OmniManip's numbers, GeoManip and KUDA beyond their titles, the ScanQA numbers for CAPruner, and SayPlan's token numbers (HTML returned 404).
- **Open question for the architect:** whether "sufficient" should be judged against answers from raw images (Y1, Y2), against plausibility (Y7), or against the planner's next action (Y4b). Each needs different logging.
- **Open question:** cognitive-science items (object files, Kuipers' Spatial Semantic Hierarchy) and LeCun's 2022 position paper were not searched today. From memory, the position paper says only that a representation should drop unpredictable detail (RECALLED). Nothing concrete enough to change the above.
