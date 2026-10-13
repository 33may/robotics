# Branch 3: scene graphs and hierarchy

**Pass 1, round 2 map. Date:** 2026-10-06
**Family:** representations that name things and keep only nodes and relations.
**Tags:** READ = read in the body today. ABSTRACT = abstract or landing page today.
RECALLED = my memory, not checked today. All URLs listed were opened today unless the
tag is RECALLED.

## 1. Headline findings

**H1. Identity in this family is almost always merge-only. The one escape is to keep the raw sightings and treat the merged node as a view over them.**
ConceptGraphs merges greedily and keeps no way back (READ). Hydra merges after loop closure on "same label and one node inside the other's box" (READ). ThinkGraphs (2606.31471) says outright that its corrections are "merge-only" (READ), and NoPA (2607.00529) cannot split after resampling (READ). Khronos keeps every object fragment in memory and recomputes the association after large pose changes. That is the closest thing to undoable merges (READ, but undo is not stated as a goal).

**H2. Our "one rigid shift per view" error is the problem object SLAM and Hydra already solve: the details are landmarks, and each view gets a pose correction.**
Hydra corrects mesh, objects, places and trajectory together with an embedded deformation graph and a robust GNC solver (READ, Sec. IV). Khronos treats fragment-to-object association as translation constraints solved with a truncated least squares loss (READ). Fusion++ keeps per-object TSDFs in a 6-DoF pose graph (ABSTRACT). Our 10 mm shift per view (1.6 mm left after removal) fits this pattern exactly: estimate the per-view shift from repeated details, do not widen the merge radius.

**H3. Part scale is the weak frontier, and the 2026 numbers show it.**
OP3DSG: part recall@3 is 83.6%, but only 25.8% once a 0.10 3D IoU is required (READ). OVIP-SG (2608.17633): open-vocabulary detectors "fragment large surfaces and merge small objects" (ABSTRACT). The closest geometry to ours is RoboSeg (2608.09778: UR5, eye-in-hand D435i, part-level voxel map). It keeps the top 3 instance IDs and their votes in each voxel, and reaches Chamfer 10.7 mm (READ, Table I). Nobody stores text labels, ports or markings as nodes on one object.

**H4. Relations are moving from stored to derived on demand. Only part-of and functional relations stay stored.**
BBQ builds edges "for each user query independently", only between candidate objects (READ). In MIT 2510.16643 the model gets geometric spatial indexing inside Cypher queries (READ). ThinkGraphs derives left and under from axis-aligned boxes (READ). KeySG drops explicit relations and keeps keyframes for a VLM to reason over (ABSTRACT). Functional edges (OpenFunGraph, OP3DSG) and containment (Hydra) cannot be computed from geometry, so they are stored.

**H5. Pruning for a task compresses hard, but only when the task is known in advance.**
Clio keeps 49 objects where the task-free version keeps 1883 (Cubicle, Table I, READ). It merges by information bottleneck against a list of natural-language tasks given at the start, and drops anything below cosine 0.23 for every task (READ). For inspection the "task" is "find everything", so Clio's criterion would prune little. Its merge rule (Jensen-Shannon distance of task relevance) is still useful when deciding how fine a tree to show.

**H6. The only real single-object feature trees are industrial, and they assume a specification exists.**
QIF goes Product, Part, Feature, Characteristic, Measurement. It keeps definition, nominal, item and actual apart, and gives every characteristic an unchanging UUID plus a human tag (READ, QIF MBC page). FRAME (2609.14219) holds an evidence memory per feature: (requirement id, value, admissible flag, provenance). It computes the "uncovered requirements" set with deterministic code before a VLM sees anything (READ). Our case has no CAD or spec, so the tree has to grow from sightings, but the record shape carries over.

## 2. Approach cards

### A. Object-level SLAM (SLAM++, Fusion++, MaskFusion, CubeSLAM, QuadricSLAM)
- Atomic unit: an object instance. SLAM++ uses a known CAD model, Fusion++ a per-object TSDF, CubeSLAM a cuboid, QuadricSLAM a dual quadric.
- Kept: object pose, shape (TSDF or primitive), class label, existence probability in Fusion++ (ABSTRACT).
- Discarded: the background (Fusion++ keeps a coarse background, RECALLED) and raw frames.
- Hierarchy: none, a flat set of objects in a pose graph.
- Update, drift, conflicts: objects are landmarks, and camera poses and object poses are optimised together. Low existence probability removes spurious objects (Fusion++, ABSTRACT).
- Known vs unknown: an existence probability per object (Fusion++). There is no "not looked" state.
- Identity and merging: mask IoU or projection overlap, plus class (RECALLED). Duplicates appear under drift (RECALLED).
- Links: class label only, no free text.
- Frame: **an object frame for the geometry, with the object pose in the world graph**. The only family in this branch that does this.
- Reader: code (SLAM back-end).
- Cost: Fusion++ runs online at 4 to 8 Hz with low memory (ABSTRACT).
- Training: a trained detector (Mask R-CNN), nothing beyond it.
- Used where: research code, not in shipped stacks that I found.
- URL: https://arxiv.org/abs/1808.08378 (Fusion++).

### B. Layered 3D scene graphs (Armeni 2019, Kimera DSG, Hydra, Hydra-Multi, S-Graphs+)
- Atomic unit: Hydra layers are 1 mesh, 2 objects and agents, 3 places, 4 rooms, 5 building (READ, IJRR 2305.07154). An object holds label, centroid and box. A place holds an obstacle-free centroid and radius.
- Kept: the mesh as the bottom layer, plus nodes above it. Nothing is thrown away. The hierarchy is an index over the full geometry.
- Discarded: raw images. Places are sparsified from a GVD.
- Hierarchy: 5 layers, formed bottom-up by clustering. Objects are mesh vertices clustered by label, rooms come from persistent homology on places. A reader moves by inclusion edges. Hydra's own toy example: 355 symbols hierarchical against 1680 flat (READ, Sec. II-B).
- Update, drift, conflicts: one embedded deformation graph corrects every layer at loop closure. Control points are mesh vertices within 2.5 m. GNC rejects bad loop closures as outliers (READ, Sec. IV). Afterwards places within 0.4 m merge, and objects merge when they share a label and one sits inside the other's box (READ).
- Known vs unknown: not addressed at node level. The TSDF holds free and unknown space underneath.
- Identity and merging: see above. No un-merge mechanism is described (READ, not found).
- Links: closed-set labels. Open-set features since Jan 2025 (Hydra README, READ).
- Frame: world (one map frame).
- Reader: code planners. LLM reading came later (SayPlan, MIT 2510.16643).
- Cost: real time on embedded GPU (READ, numbers not extracted).
- Training: segmentation network. GNN loop-closure descriptors are optional. Positives are sub-graphs with at least 40% box IoU (READ).
- Used where: **open source**. MIT-SPARK/Hydra moved to ROS2 Jazzy by default in July 2025, BSD-2 (READ, README). Kimera is open source (ABSTRACT). S-Graphs ships a ROS2 package (ABSTRACT). No shipped product found.
- S-Graphs+ variant: the layers are keyframes, wall planes, rooms and floors, all inside one factor graph. **The hierarchy is part of the estimator, not a summary of it** (ABSTRACT, 2212.11770; S-Graphs 2.0, 2502.18044).

### C. Spatio-temporal fragments (Khronos)
- Atomic unit: **an object fragment**, a partial view built over a short window where pose error and change stay bounded (READ, 2402.13817).
- Kept: per fragment, a mesh or point sequence, AABB, label or feature vector, first and last seen time, and a pose T_WY (READ).
- Discarded: nothing at fragment level. All fragments stay in memory, and the paper names that as a scaling limit (READ).
- Hierarchy: fragments, then objects (reconciled), plus the background mesh.
- Update, drift, conflicts: short-term association by volumetric IoU, kept after 15 observations (READ). After loop closure, candidate edges link fragments with the same label and overlapping boxes. These become translation constraints with a TLS loss, so wrong candidates drop out as outliers (READ). The hash map is "re-computed from scratch" when large pose changes happen (READ).
- Known vs unknown: **yes, through rays**. A library of rays checks whether a fragment's points are seen absent, present or occluded: d_ray 30 cm, and 60% of rays in a 5 s window must agree (READ). This is the clearest "not there" versus "not looked" mechanism in the family.
- Identity: see above. Because fragments are never fused away, re-association is possible.
- Links: label or open-set feature.
- Frame: world, with a per-fragment pose.
- Reader: code.
- Cost: 45.5 ms per frame in the active window. Reconciliation of all fragments under 100 ms (READ, Office scene).
- Training: none of its own.
- Used where: open source, MIT-SPARK/Khronos (READ).

### D. Open-vocabulary object graphs (ConceptGraphs, HOV-SG, BBQ, ThinkGraphs, OVIP-SG, ZING-3D, INHerit-SG, OpenIN, DovSG)
- Atomic unit: an object node with a point cloud, an averaged CLIP feature, a caption and a box.
- Kept: ConceptGraphs gives the LLM JSON with id, bbox_extent, bbox_center, object_tag and caption (READ, Sec. II-C). ThinkGraphs adds material, colour, texture and state from a VLM (READ).
- Discarded: everything but the per-object point cloud and fused feature. Captions come from the 10 best views (ConceptGraphs, READ). The views themselves are not stored as nodes.
- Hierarchy: flat in ConceptGraphs and BBQ. HOV-SG has 3 levels (floor, room, object) and reports 75% less memory than dense open-vocabulary maps (ABSTRACT). KeySG has 4 (floor, room, object, functional element) (ABSTRACT).
- Update, drift, conflicts:
  - ConceptGraphs: greedy assignment. Geometric similarity is the share of points with a neighbour within 2.5 cm, semantic similarity a normalised cosine. Their sum must reach 1.1 (READ, Sec. II-A).
  - ThinkGraphs: score = 0.8 voxel-vote geometry + 0.2 CLIP, threshold 0.4. A node is confirmed after 6 hits. A critic agent proposes duplicates by IoU above 0.01 and SBERT above 0.8, and a VLM confirms "Merge/Keep" on Set-of-Mark image pairs (READ). It reports no duplicate-rate numbers (READ).
  - DovSG: updates the graph locally when the scene changes (ABSTRACT). OpenIN keeps carried-by relations so it can follow objects that move (ABSTRACT).
- Known vs unknown: OVIP-SG decides presence or absence from voxel votes with 0.773 balanced accuracy (ABSTRACT). Elsewhere not addressed.
- Identity: see above. Known: ConceptGraphs node precision 71%, 0 to 5 duplicates per scene. No family member has an un-merge.
- Links: CLIP or SBERT embedding plus caption. Object to image only through the "best views" used for the caption. KeySG keeps keyframes as first-class nodes (ABSTRACT), the only clean geometry-to-image link I found.
- Frame: world.
- Reader: an LLM over JSON (ConceptGraphs, BBQ), an agent with tools (ThinkGraphs), or RAG (KeySG, INHerit-SG).
- Cost: BBQ warns that 100+ objects with all edges exceed 32k symbols (READ).
- Training: none (ConceptGraphs, ThinkGraphs, ZING-3D, BBQ). SceneLM (2609.21400) is the trained outlier. It writes the map as text through ADD, EDIT and REMOVE operations: REMOVE recall goes from about 0% for base VLMs to 72% after fine-tuning (Table 1, READ).
- Used where: research code with public repos (ConceptGraphs, HOV-SG, RECALLED). Widely reused as a baseline, not in products.

### E. Trained relation graphs (3DSSG, SceneGraphFusion, Open3DSG, NoPA)
- Atomic unit: **a geometric segment**, not an object (SceneGraphFusion: "primitive scene components" as nodes, ABSTRACT).
- Kept: segment features and predicted predicates (support, spatial, comparative; Open3DSG, ABSTRACT).
- Update: incremental GNN at 35 Hz (SceneGraphFusion, ABSTRACT). NoPA uses particle sets with a Hellinger pre-filter (0.85 ± 0.05) and an MMD test (0.6 to 0.7). No split (READ).
- Relations: predicted and stored. NoPA propagates them within affinity clusters by majority vote (READ).
- Training: yes (GNN, 2D relation detector). Open3DSG distils from 2D VLMs.
- Used where: papers and benchmarks (3DSSG, 1482 scenes, READ via NoPA).
- Takeaway for us: the segment-as-node idea, not the trained predictor.

### F. Functional and part-level graphs (OpenFunGraph, 2605.15753, OP3DSG, PhysGraph, Search3D, RoboSeg)
- Atomic unit: an object node plus an **interactive element node** (knob, button, handle). OP3DSG adds affordance labels (READ).
- Kept: the label, the affordance, functional edges ("turn on/off", "control flow"), spatial edges including "part of" (READ, OP3DSG).
- Hierarchy: object, then part. 2605.15753 adds "global hierarchy shaping" for multi-level functional relations (ABSTRACT). Search3D searches at part, object and material-region levels (ABSTRACT).
- Update and identity: OP3DSG associates parts greedily on nearest-neighbour geometry, CLIP cosine, and a chi-square distance of colour histograms. Parts are usually one material, so colour is stable (READ). RoboSeg keeps top-3 instance IDs with vote counts in each voxel, and a mask joins an existing same-class ID among its voxels or gets a new one (READ).
- Known vs unknown: not addressed.
- Links: label and affordance text. OP3DSG draws part candidates from an object-to-part knowledge base before detecting (READ). So **the expected parts are a prior, not discovered**.
- Frame: world (OP3DSG back-projects to a global frame, READ).
- Sizes: UniGraph3D has 251 object nodes, 414 part nodes, 357 spatial edges, 425 functional edges and 554 affordances over all scenes (READ).
- Training: OP3DSG and RoboSeg are training-free (READ). OpenFunGraph uses VLMs and LLMs (ABSTRACT; my fetch tool said "not training-free", which I could not confirm).
- Used where: papers only. RoboSeg ran 24 physical trials on a UR5 (READ, Table II).

### G. Single-object part hierarchies (PartNet, PartNeXt, GARField, PartField, SAMPart3D, Find3D, H2G)
- Atomic unit: a part, in a tree from coarse to fine. PartNet has 573,585 part instances over 26,671 models (search snippet only, not read). PartNeXt has 23,000+ models in 50 categories (ABSTRACT).
- Hierarchy: datasets fix it by category. GARField conditions affinity on scale so conflicting masks can live at different scales (ABSTRACT via search). H2G learns a tree in hyperbolic space (ABSTRACT). PartField clusters a feature field into a hierarchy (ABSTRACT via search).
- Update, identity: not addressed. These are offline segmentations of a finished shape, not maps that grow.
- Training: yes (all are learned fields or feed-forward nets), except as off-the-shelf tools.
- Used where: research tools.
- Takeaway: **"granularity as a scale parameter"** (GARField) is a clean way for a reader to move between levels without fixing the layer count.

### H. Task-driven pruning and abstraction (Clio, ASHiTA, Larsson et al., Taskography, SayPlan, EmbodiedRAG, Prior-SG)
- Clio (2404.13696, READ): its primitives are Khronos-style 3D segments and Hydra places. Agglomerative information bottleneck merges pairs by (p_i + p_j) times the JS divergence of p(task | primitive). A threshold δ̄ in [0, 1] sets the compression, and no single experimental value is given. Below cosine α = 0.23 for every task, a primitive goes to the null task and is dropped. Cubicle: 1883 objects become 49 online. 0.29 s per frame on an RTX 3090. Open source (MIT-SPARK/Clio).
- ASHiTA (2504.06553, ABSTRACT): a hierarchical IB that compresses once per level of an LLM-built task hierarchy.
- Larsson, Maity, Tsiotras (Q-tree search, 1910.00063; semantic octrees, 2209.10035, ABSTRACT): choose the multi-resolution tree cut that keeps the most information about a relevant variable within a complexity budget. Prunes octrees with semantic uncertainty. A 2026 sample-based anytime version exists (2609.27330, ABSTRACT). **Fits a voxel octree under a detail tree directly** (see Section 3, RQ5).
- Taskography (2207.05006, ABSTRACT via search): SCRUB prunes a 3DSG to the facts a planner can still use and keeps a satisficing plan if one exists. SEEK scores objects with a GNN.
- SayPlan (2307.06135, ABSTRACT): the LLM sees a collapsed graph and calls expand and contract. Up to 3 floors, 36 rooms, 140 assets and objects. Token numbers could not be read (HTML 404, PDF too large).
- EmbodiedRAG (2410.23968, ABSTRACT): retrieves a task subgraph, "an order of magnitude" fewer tokens, up to 70% less time per planning step.
- Prior-SG (2608.06170, ABSTRACT): an LLM writes a prior graph of the structure it expects, and an MRF aligns that prior with the observed instance graph. **A worked way to hold expected-but-unseen structure.**

### I. Non-robotics precedents (CAD feature trees, STEP AP242 PMI, QIF, BIM/IFC, FRAME)
- QIF (READ, qifstandards.org MBC page): Product, Part, Feature, Characteristic, Measurement. Definition, nominal, item and actual are separate objects. Each characteristic has a human tag (PC007) and a UUID "generated once and never changes", introduced because renumbering broke traceability.
- STEP AP242 (search snippets only): semantic PMI links a tolerance to the exact face it governs. Graphic PMI is presentation only.
- BIM/IFC: an element hierarchy (site, building, storey, space, element), RECALLED. In 2026 LLMs query it through relational plus graph layers (IfcLLM 2605.13236, ABSTRACT via search) and Cypher, the same pattern as MIT 2510.16643.
- FRAME (2609.14219, READ): **Part, then surface region, then requirement.** A record e = (requirement id, value, admissible flag, metadata: scan id, regions, time, failed conditions, sources). An inadmissible record "cannot support PASS" but is kept for audit. The uncovered set comes from a deterministic check before a VLM proposes the next step.
- Used where: QIF and AP242 are ISO/ANSI standards inside commercial metrology and CAD software (RECALLED for named vendors). FRAME is a paper.

## 3. Per research question

- **RQ4, atomic block.** The family has four answers: the object (A, D), the segment or fragment (E, Khronos, Clio primitives), the part or interactive element (F), the place (B). The pattern that survives drift is "fragment or sighting as the stored unit, object as a derived cluster" (Khronos, Clio). For us: the sighting of a detail in one view is the atom, and the detail node is a cluster of sightings.
- **RQ5, hierarchy.** Layers are fixed by type (Hydra 5, HOV-SG 3, KeySG 4) or come out of an optimisation (Clio IB, Larsson Q-tree, GARField scale). Readers move by inclusion edges, by SayPlan expand and contract, or by RAG retrieval per level (KeySG). For a voxel space plus a detail tree, Larsson's tree-cut and Hydra's inclusion edges are the two models.
- **RQ6, update without drift.** Hydra deforms every layer with one embedded deformation graph and re-merges after (READ). Khronos keeps fragments and reconciles them by robust optimisation (READ). S-Graphs puts the hierarchy inside the factor graph (ABSTRACT). For a static object with a rigid shift per view: one pose correction per view, estimated from details seen twice.
- **RQ7, who manages.** Usually a deterministic back-end (Hydra, Khronos). Newer systems add an LLM or VLM critic that only proposes merges (ThinkGraphs, READ) or a trained model that emits ADD, EDIT and REMOVE (SceneLM, READ). Branch 4 owns the details.
- **RQ8, contract.** Inputs: posed RGB-D plus masks, features or captions. Outputs: JSON node lists (ConceptGraphs), a Cypher database (MIT), expand and contract calls (SayPlan). FRAME's output contract is the cleanest: a computed set of uncovered requirements.
- **RQ9, preserve versus discard.** Hydra and Khronos keep all geometry under the graph. ConceptGraphs keeps per-object clouds plus one caption. Clio throws away what is irrelevant to the task. KeySG keeps keyframes instead of relations. Branch 5 owns the rule; this family says "keep the link to the source view", which is cheap and needed for re-reads.
- **RQ10, known versus unknown.** Mostly silent. The exceptions are Khronos ray evidence for absence (READ), OVIP-SG voxel votes for presence or absence (ABSTRACT), and FRAME's admissible flag plus uncovered set (READ). Per-element quality belongs to branch 1.
- **RQ11, identity.** Rules: overlap plus feature similarity with fixed thresholds (ConceptGraphs 2.5 cm and 1.1, ThinkGraphs 0.4, Hydra same label and containment, NoPA Hellinger then MMD, OP3DSG adds colour histograms). Probabilistic: BPDA-GMM uses a Chinese restaurant process prior (2606.04618, ABSTRACT). Measured duplicate rates are rare: only ConceptGraphs has one (already known). Un-merge: none explicit. Khronos's kept fragments are the only basis for it.
- **RQ12, linking geometry, image and words.** A node carries a box or cloud, a CLIP or SBERT vector, and a caption. Image links are weak everywhere except KeySG keyframes and ConceptGraphs "best views". QIF's split (definition, nominal, actual, all under one UUID) is a precedent for keeping one identity across representations.
- **RQ13, frame.** World frame almost everywhere. Object SLAM (Fusion++, SLAM++) keeps geometry in an object frame with one pose in the world graph, which is the right model for "one object now, several later". Khronos gives each fragment its own pose. Translating to the asker's view: silent.

## 4. What does not transfer, and why

- **Room-scale merge rules.** Hydra merges places within 0.4 m and uses box containment for objects. ConceptGraphs sets its neighbour radius to 2.5 cm, close to our 19 mm p90 error. On one object with features 3 cm apart, a 2.5 cm radius plus label similarity would merge neighbouring labels that look alike. The fix in this family is to correct poses first (H2), not to tune thresholds.
- **Label and CLIP similarity as an identity signal.** Two stickers on one box both read as "sticker". OP3DSG's colour histogram helps a little. Text read by OCR is a stronger key than CLIP, and none of these systems use it.
- **Places, rooms and floors layers.** Meaningless on a tabletop object. The usable hierarchy is object, face or region, detail, characteristic (QIF, FRAME).
- **Trained relation predictors** (3DSSG, SceneGraphFusion, NoPA, SceneLM). They break the no-training rule.
- **Clio-style pruning.** It needs a fixed task list, while inspection asks "what is there", so the null-task cut would drop the unexpected details we want.
- **Knowledge-base part priors** (OP3DSG). They assume known categories. An unknown inspected object has no part list, though a "prior graph" à la Prior-SG could still be written by the planner.
- **Pasting whole graphs.** Already known (MIT 2510.16643): 65 objects in the small graph, about 17M tokens for the large one. Our single object will be small enough to paste for a long time, so this limit bites later, at multi-object scale.

## 5. Ranked reading list for deep reading

1. **Khronos**, https://arxiv.org/html/2402.13817. Fragment as atom, reconciliation, ray-based absence. Extract: the fragment data structure, the exact reconciliation loop, whether associations can flip between optimisations.
2. **Hydra IJRR**, https://arxiv.org/html/2305.07154. Sec. IV deformation and merging. Extract: how node positions are interpolated from control points, the order of merge then recompute, any merge failures reported in Sec. VI.
3. **FRAME**, https://arxiv.org/html/2609.14219. Closest inspection memory. Extract: the full record schema, the admissibility conditions per requirement type, the surface correspondence table, what the VLM sees.
4. **RoboSeg**, https://arxiv.org/html/2608.09778. Same arm class, eye-in-hand. Extract: the voxel vote tensor, ID allocation and conflict handling, voxel size, the instance table export format.
5. **ThinkGraphs**, https://arxiv.org/html/2606.31471v1. Agentic merge critic. Extract: the critic prompt, its false merge or keep rate if it exists in the unread last 25k characters, the cost per check.
6. **OP3DSG**, https://arxiv.org/html/2606.29786. Part association. Extract: thresholds, the colour histogram weight, the failure analysis for small parts.
7. **Clio**, https://arxiv.org/html/2404.13696. IB math. Extract: δ̄ values used, and whether "task = list every detail" degenerates.
8. **Larsson et al., semantic octree abstraction**, https://arxiv.org/abs/2209.10035. A tree-cut over voxels. Extract: the objective, runtime per tree, whether the relevant variable can be "coverage" or "detail present".
9. **SceneLM**, https://arxiv.org/html/2609.21400. Text map with ADD, EDIT, REMOVE. Extract: the entry schema and the operation grammar, worth copying even without training.
10. **QIF MBC and the QIF standard**, https://qifstandards.org/model-based-characteristics-enabling-persistent-identification-in-the-digital-thread/. Extract: the exact Feature and Characteristic element fields to borrow names from.

## 6. Gaps and open questions

- **Not found:** any measured wrong-merge rate, or any un-merge or split operation, in the scene graph literature. All numbers are recall, accuracy or mIoU.
- **Not found:** any node type for text, labels or markings read by OCR. Search3D and OP3DSG stop at parts and materials.
- **Not found:** a shipped product that runs a 3D scene graph. Hydra, Khronos, Clio, Kimera and S-Graphs are open-source research stacks with ROS2 packages.
- **Not found:** a study of which relations an LLM planner actually queries. MIT 2510.16643 says the LLM "inconsistently leveraged" distance and containment queries (READ, appendix), nothing more.
- **Could not read:** SayPlan token counts (HTML 404, PDF over the size limit). PartNet depth statistics (search snippet only). The last part of the ThinkGraphs and SceneLM bodies.
- **Resolved by the lead against the raw table, 2026-10-06:** BBQ Table II is the edge ablation on Nr3D with ground-truth objects, Recall@1 overall. Llama3-8B: no edges 36.1, metric edges 43.8, semantic edges 41.4, both 45.5. GPT-4o: no edges 61.8, metric 68.6, semantic 50.5, both 68.4. Metric edges help both models. Semantic edges alone lower GPT-4o by 11 points. The 34.2 is BBQ's own A@0.1 on Sr3D in Table III, a different measurement. The "0.52 to 0.74 on Sr3D" lift given in the brief is not in the paper and is withdrawn.
- **Open for us:** if sightings are the stored atom and detail nodes are derived, does the per-view shift estimate (H2) remove the need for a merge threshold altogether, given 1.6 mm residual against 3 cm feature spacing?
