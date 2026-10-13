# Pass 2 — 3D Scene Graph survey: how semantics is attached to geometry

Paper: Rotondi, Argenziano, Koch, Hughes, ... Carlone, Arras. *3D Scene Graphs: Open Challenges and Future Directions*. arXiv 2606.19383 v1 (2026).
Source read: `papers/md/rotondi-2026-3d-scene-graphs-survey.md`, the whole body (sections 1–6) and the whole reference list. Figure 3 (a vector figure missing from the md) was read from the PDF text layer.

Question for this pass: how can a 3D representation be "coloured" with semantic labels, ideally hierarchical ones (one cell/point belongs to object > face > sticker > text line at once)?

**Reading notes**

- Reference numbers below follow the md reference list. The PDF numbers some early entries one off (e.g. Armeni 2019 is [4] in the md, [5] in the PDF). Check before citing.
- The survey is a map, not a methods handbook. The body is about 6 pages. Per-method details are often one clause long. I flag "thin" wherever the survey gives the name and almost no mechanism.
- The survey never uses the words "voxel", "granularity" (except in the title of FOUND-it [94]) or "multi-resolution". It does not describe how any method stores labels inside a dense map (per voxel / per mesh vertex). That is a real gap for our question.

---

## 1. The survey's own taxonomy of geometry ↔ semantics

### 1.1 The formal definition (§3, Eq. 1)

A 3DSG is a tuple `G = (V, V_g, V_f, E, E_f, [L])`.

- **Grounding `V_g`** is the link from semantics to geometry. "The grounding $\mathcal{V}_{g}:\mathcal{V}\to\mathcal{M}$, with $\mathcal{M}\subseteq\mathbb{R}^{3}$, assigns to each node $v$ the subset of metric map points it occupies." (§3)
- The geometry under it can be anything that reduces to points: "point clouds serve as a common geometric denominator to which other reconstruction modalities, such as meshes [4, 125] and 3D Gaussian splatting models [48], can be reduced." (§3)
- **Node features `V_f`** hold both kinds of attribute. "*geometric attributes*, such as the centroid, enclosing bounding box, or 3D shape, and *semantic attributes*, such as class labels, material properties, color [5], multimodal embeddings [29], or textual descriptions [51]." (§3)
- **Edges `E`, `E_f`** carry typed relations. Types listed: "spatial (*e.g.,* "next to", "contained in"), structural (*e.g.,* "is part of"), semantic (*e.g.,* "smaller than"), or functional". Multigraphs are allowed (§3).
- **Layer labelling `L`** is optional: "organizes nodes into hierarchical *layers*, assigning each node to a level ... a flat 3DSG simply omits it." (§3)
- Any part can be time-indexed. "A 3DSG with any time-indexed component is a *4D Scene Graph*" (§3).

So, in the survey's frame, a semantic label never lives "in" the geometry. It lives on a node. The node points to the subset of geometry it occupies. That pointer (`V_g`) is the only formal coupling.

### 1.2 Hierarchy (§3.1)

- Layers partition the nodes. Layer 1 is the bottom and holds the dense geometry itself: "layer $l=1$ as the layer encoding the lowest-level, often metric information (*e.g.,* meshes or point clouds)." (§3.1)
- Two edge kinds: "$\mathcal{E}_{\mathrm{H}}$ ... are the hierarchical edges connecting adjacent layers, and edges in $\mathcal{E}_{\mathrm{R}}$ ... in practice predominantly relate nodes within the same layer" (§3.1).
- Hierarchical edges are strictly between adjacent layers (`V_l × V_{l+1}`). The survey gives no rule that a parent "summarises" its children. It only says layers organise nodes.
- Usual indoor stack: building → (floor) → rooms → places → objects (§3.1, Fig. 2). Situational Graphs replace objects by "geometric primitives, such as wall and corridor planes" (§3.1).
- Benefit claimed: "*e.g.,* Hughes et al. [62] show that, under certain assumptions, the 3DSGs underlying indoor spaces have low treewidth, enabling fast inference." (§3.1)

### 1.3 Construction design space (§4, Figure 3)

Figure 3 splits construction into five axes: **processing** (offline/online), **nodes** (objects, places, attributes), **edges** (relational, hierarchical), **priors** (knowledge bases, MLLM, tasks), **consistency** (spatial, temporal). "These dimensions are conceptually separable, enabling methods to be interpreted as combinations of design choices across axes" (Fig. 3 caption).

### 1.4 How node semantics are produced (§4.1) — the four mechanisms the survey names

1. **Lift 2D masks into 3D, fuse over views.** "per-frame segmentation masks are back-projected into 3D using depth and camera poses and are subsequently fused across viewpoints. Multi-view aggregation suppresses noisy detections while accumulating complementary evidence" (§4.1). Fusion rules named: "Bayesian updates [125], majority voting [4], or adaptive aggregation mechanisms [82]".
2. **Open-vocabulary lifting.** Tag the image (RAM) or ask a VLM for object mentions, feed the label set to GroundingDINO / YOLO-World, refine with SAM, then extract "multimodal descriptors [111]" (CLIP) (§4.1).
3. **Batch learned prediction on a pre-segmented cloud.** "instance masks are given, while semantic labels must be predicted" with PointNet + GCN + MLP heads (Wald [148]) (§4.1). Needs training.
4. **Language–geometry alignment.** "projecting point cloud features into, for example, the CLIP [111] shared vision-language semantic space" (SGFormer [89], VL-SAT [153], Universal SG [156]) (§4.1). Also trained.

Attributes can come "directly from the available 3D geometry", "from visual observations using VLM-based prompting", or "as textual descriptions (captions) and feature embeddings generated from images" (§4.1, Node Attributes).

### 1.5 The implicit-vs-explicit axis (§1, §4.1, §6)

- §1 frames the gap: classical maps give "precise geometric information but offer limited support for high-level reasoning", while "implicit language-based representations [69, 116] [LERF, LERF-TOGO] capture rich semantic abstractions, yet lack explicit grounding in physical space."
- §4.1 and §6 propose a hybrid as a direction, not an existing method (see §3 below).

---

## 2. Methods the survey describes — what it actually says

Columns that the survey does not state are marked "—". Most methods get one sentence.

| Method | Where the label lives (per survey) | What is stored | Hierarchy | Fusion / update over views | Queries | Survey depth |
|---|---|---|---|---|---|---|
| **Armeni 3DSG [4]** (2019) | Nodes over a mesh ("meshes [4, 125]") | class labels | building/rooms/objects/camera; rooms "aggregates disjoint regions" | "majority voting [4]" | task planning datasets (§5.3) | thin |
| **Kimera [125]** (2021) | Nodes grounded in a mesh | class, human SMPL nodes | rooms by "slicing the 3D Euclidean signed distance field below the ceiling and performing 2D connected-component analysis" (§4.1) | "Bayesian updates [125]" | — | thin on label storage |
| **3D DSG [124]** (2020) | nodes | humans with "skeletal joints and pelvis locations" (§3.2) | places/objects/humans | — | — | thin |
| **Hydra [61], Hughes IJRR [62]** | nodes over metric map | places via free-space clustering "forgoing explicit semantics" (§4.1) | layered; "low treewidth" [62] | greedy/linear assignment; loop closure from "descriptors [61, 62] ... extracted from the 3DSG hierarchy" (§4.3) | — | thin on semantics |
| **Khronos [133]** (2024) | "distinct nodes for different observations or temporal instances of the same entity" (§3.2) | observation history | — | "requires an additional mechanism to reason about which nodes correspond to the same physical entity" (§3.2); reconstructs scene "history" | — | very thin |
| **Clio [93]** (2024) | nodes; "enriching geometrically derived 3D primitives with open-vocabulary embeddings [93]" (§4.1) | open-vocab (CLIP) embedding | primitives/places → objects; listed under "Tasks" priors in Fig. 3 | "explicit object tracking [93, 139]"; assignment by embedding distance (§4.3) | CLIP query vs node feature (§5.2) | thin; "task-driven" only from the title |
| **ConceptGraphs [53]** (2024) | object nodes from lifted masks | CLIP features, VLM captions | flat objects + relations | association by "spatial overlap or feature similarity [53]"; dense graph "pruned before querying an LLM to assign the corresponding predicates" (§4.2) | QA, manipulation grounding | moderate |
| **HOV-SG [154]** (2024) | nodes; floors/rooms/objects | CLIP features | floors "from height distributions and by partitioning rooms through planar spatial segmentation"; floor→room parent edges (§4.1–4.2) | — | "compared CLIP features extracted from the query with those associated with graph nodes [154, 93]" (§5.2); eval "stratified by hierarchical reasoning levels" | moderate |
| **KeySG [155]** (2026) | hierarchy built over keyframes (title) | captions/embeddings "generated from images" (§4.1) | floor→room parent edges | — | "constructing the representation in a RAG-inspired manner to enable fast multimodal indexing [155]" (§5.2) | thin |
| **BBQ [82]** (2025) | object nodes | — | — | "adaptive aggregation mechanisms [82]"; nodes incremental, edges batch | "retrieving smaller, relevant subgraphs using heuristic strategies [82]" | thin |
| **Open3DSG [75]** (2024) | nodes on point cloud | "queryable objects and open-set relationships" (title) | — | VLM run on "a subset of views where the nodes are most clearly visible" (§4.2) | open-vocab | thin, but best-view idea is relevant |
| **FunGraph [126], OpenFunGraph [174]** (2025) | sub-part nodes (knobs, handles) | open-vocab labels, functional edges | "representing interactive sub-parts ... as separate nodes"; "intra-object composition (*e.g.,* a knob is part of an oven)" (§3.3); parts "linked to their parent objects using geometric cues [126, 63, 152]" (§4.2) | VLM predictions "consolidate[d] ... across all images for each node pair [126, 8]" (§4.2) | JSON serialisation for LLM multi-hop queries (§5.2) | moderate — the closest to within-object hierarchy |
| **Articulated SGs: MoMa-SG / Büchner [14], ArtiSG [54], Pandora [168]** | part nodes | "kinematic attributes ... that store the current object state (*e.g.,* the opening angle)" (§3.3) | object → part | from human demonstrations (§5.4) | planning over kinematic states | thin |
| **RoboEXP [63], CuriousBot [152]** | object + action nodes | action effects | parts linked to parents by geometry | interactive exploration | — | thin |
| **DynamicGSG [48], GaussianGraph [150]** | nodes grounded in 3D Gaussians | — | — | change detection "using a Gaussian map [48]" | — | very thin |
| **RelationField [76]** | relations in a radiance field (title) | — | — | — | — | name only (Fig. 3) |
| **FOUND-it [94]** (2026) | — | — | "granularity on demand" (title only) | cited only for "geometric FMs" in mask lifting (§4.1) | — | name only, but title matches our question |
| **T-funs3d [42]** (2026) | — | — | "task-driven hierarchical open-vocabulary 3D functionality segmentation" (title) | — | — | name only |
| **Situational / S-Graphs [9–11, 144]** | planes, rooms, floors in a factor graph | geometric primitives | "maintaining the full 3DSG as an optimizable factor graph, so that structural entities ... directly constrain pose estimation" (§4.3) | joint optimisation | localisation | moderate on consistency |
| **3DSSG / Wald [148]** (2020) | nodes on instance-segmented cloud | class + predicate labels | flat | batch, learned | cross-domain retrieval | moderate, but trained |

What the survey does **not** say for any method: the data type of per-point/per-voxel labels, how a dense mesh stores a label, or a rule for a parent's label from its children's labels.

---

## 3. Part-level hierarchy, granularity, and dense-geometry ↔ graph coupling

### 3.1 Part-level / within-object structure

- Grounding explicitly allows parts: a node is "often an abstraction of low-level scene primitives and geometry (*e.g.,* objects or object parts)" (§3).
- Functional 3DSGs "model the internal functional structure of objects by representing interactive sub-parts (such as knobs and handles) as separate nodes" (§3.3). Edges encode "intra-object composition (*e.g.,* a knob is part of an oven)".
- Fig. 2: "Articulated objects are associated with a motion model that describes part-level motion".
- Linking parts to parents: "when object parts are included in the hierarchy, they can be linked to their parent objects using geometric cues [126, 63, 152]" (§4.2). The cue is not specified.
- Part nodes are produced the same way as object nodes: "A dominant strategy for generating object and object-part nodes ... is to lift 2D instance segmentation masks into a 3D representation" (§4.1).
- Cost warning: "representing all object sub-parts, actions, beliefs, and functional relationships can result in large, computationally expensive graphs" (§3.3).
- **Thin:** nothing on more than one level inside an object. Parts are one level below objects. Nothing like object > face > sticker > text line.

### 3.2 Granularity / multi-scale

- Layers are "often hard-coded and inflexible" (§3.1). ASHiTA [23] has "begun to infer layer definitions automatically in a task-aware manner" (§3.1).
- Wanted: "lightweight parametric nodes with mechanisms that dynamically expand or compress functional detail based on task and context" (§3.3). Repeated in §6: "*Adaptive, task-conditioned graphs* that selectively expand or compress detail based on the active task".
- Ambiguous boundaries: "Node extraction also assumes objects are discrete and well-bounded, whereas real environments often exhibit ambiguous boundaries, articulated structures, and object-scene continua" (§4.1).
- Open-set label evaluation: "label similarity and ambiguity ... *set ranking* or *top-N frequency at category* [67]" (OpenLex3D, §5.1).
- **Thin:** FOUND-it's "granularity on demand" and Clio's task-driven granularity are not explained in the survey.

### 3.3 Coupling dense geometry and the graph

- Formal coupling is only `V_g` (node → set of map points) plus layer 1 being the dense map itself (§3, §3.1).
- Spatial consistency: "ensuring that all graph entities remain geometrically coherent with the underlying metric map as the robot's estimate of the world evolves" (§4.3).
- Data association: "Local detections are scored relative to existing map entities using geometric distance, overlap, appearance similarity, or embedding distance, followed by greedy or linear assignment" (§4.3).
- Tightest coupling named: S-Graphs keep "the full 3DSG as an optimizable factor graph" (§4.3).
- Anchoring: CURB-SG "shows that anchoring this association to persistent geometric structures ... is more reliable than associating through dynamic or semantically ambiguous entities" (§4.3).
- Hybrid direction, the passage closest to our question (§4.1): "This motivates the development of hybrid representations that preserve rich underlying information in a compact implicit representation while maintaining only a limited set of explicit semantic attributes. Additional semantic details can then be retrieved and made explicit on demand". §6 again: "*hybrid representations* that ground explicit graph structure in implicit geometry".
- Grounding back to geometry for use: navigation "projecting object positions onto dense maps for cost-based point navigation" (§5.3); manipulation uses "the bounding box of the object to be retrieved" (§5.4, Chang [20]).
- Best view for reading: Open3DSG [75] first selects "a subset of views where the nodes are most clearly visible before performing inference" (§4.2).

---

## 4. Open challenges that match our problem

| Our concern | Survey challenge (section) | Quote |
|---|---|---|
| Granularity of levels | Defining and Inferring Abstract Layers (§3.1) | "a more rigorous and flexible formulation of abstraction layers is still lacking" |
| Detail expands only when needed | Task Awareness (§3.3); Conclusion (§6) | "dynamically expand or compress functional detail based on task and context" |
| Labels unknown until later | Deferred Semantics (§4.1) | "the information that will ultimately be relevant is not known in advance" |
| Cascaded perception errors | Perception Cascades (§4.1) | "early errors that propagate and result in duplicated, fragmented, or missing nodes" |
| Fuzzy part boundaries | Ambiguous Object Boundaries (§4.1) | "assumes objects are discrete and well-bounded" |
| Semantic consistency across views | Semantic Consistency (§4.3) | "consistency is treated almost exclusively as a geometric problem ... Whether an entity's open-vocabulary embedding, attributes, or relations agree across observations (and how uncertain those semantic estimates are) is rarely modeled explicitly" |
| Identity / merging, soft vs hard | Marginalization (§4.3) | "deciding when data association is certain enough ... to commit to a hard assignment, versus retaining a soft association" |
| Single estimate, no belief | Uncertainty (§3.2) | "most approaches maintain a single estimate of the world state, despite the inherent uncertainty" |
| Edges depend on node quality | Cascade Failures (§4.2) | "segmentation errors, occlusions, or inconsistent identities propagate into incorrect edges" |
| Relations on demand | Scalability (§4.2) | "generate, retrieve, or refine them on demand rather than constructing a fully connected relational graph upfront" |
| Graph grows past LLM context | §5.2 | "the serialized JSON representation expands accordingly, quickly exhausting the LLM's context window" |
| Graph quality vs task effect | Evaluation (§6) | "the effect of perception errors on downstream behavior is rarely quantified directly" |

---

## 5. Fetch-next list (ranked, not fetched)

1. **FOUND-it** — Maggio, Gorlo, Carlone, 2026 [94]. "foundation-model-first task-driven 3D scene graphs with granularity on demand". Title is our question. Survey gives no mechanism.
2. **Clio** — Maggio et al., RA-L 2024 [93]. Task-driven open-set 3DSG; "enriching geometrically derived 3D primitives with open-vocabulary embeddings". Likely shows how primitives below objects are clustered by task.
3. **FunGraph** — Rotondi et al., IROS 2025 [126]. Sub-part nodes, part-to-parent by geometric cues, multi-view VLM consolidation. Closest to within-object hierarchy.
4. **OpenFunGraph** — Zhang et al., CVPR 2025 [174]. "Open-vocabulary functional 3D scene graphs"; part-of + functional edges.
5. **T-funs3d** — Feng & Sabzevari, ICRA 2026 [42]. "task-driven hierarchical open-vocabulary 3D functionality segmentation". Hierarchical segmentation at part level.
6. **Hydra / Hughes IJRR** — 2022 [61], 2024 [62]. The layered design with a mesh at layer 1; need the paper for how mesh labels and layers stay consistent.
7. **HOV-SG** — Werby et al., RSS 2024 [154]. Hierarchical open-vocab features per level; queries "stratified by hierarchical reasoning levels".
8. **KeySG** — Werby et al., ICRA 2026 [155]. Keyframe-based hierarchy with RAG indexing; images stay first-class. Matches "images remain the source of readings".
9. **Open3DSG** — Koch et al., CVPR 2024 [75]. Picks views "where the nodes are most clearly visible" before VLM inference.
10. **ConceptGraphs** — Gu et al., ICRA 2024 [53]. Baseline for lifting, association, captions.
11. **Khronos** — Schmid et al., RSS 2024 [133]. Per-observation nodes plus later merging; relevant to identity and history.
12. **OpenLex3D** — Kassab et al., NeurIPS 2026 [67]. Tiered benchmark for open-vocab label ambiguity.
13. **xFLIE** — Viswanathan et al., 2024 [146]. "actionable hierarchical scene representations for autonomous semantic-aware inspection missions". Only domain match on inspection; survey cites it for partially perceived nodes.
14. **Kimera** — Rosinol et al., IJRR 2021 [125]. Bayesian label fusion on a mesh.
15. **S-Graphs 2.0** — Bavle et al., 2025 [11]. Graph as factor graph; semantics constrain geometry.
16. **GaussianGraph / DynamicGSG** — [150], [48]. Graph grounded on 3D Gaussians.
17. **ASHiTA** — Chang et al., CVPR 2025 [23]. Infers layer definitions task-aware.
18. **LERF** — Kerr et al., ICCV 2023 [69]. The implicit dense-language-field end the survey contrasts against.

---

## 6. Transfer to our case — MY READING, not the survey's

Our case: inward, one object, details down to ~5 mm, ~10 mm per-view pose offset, open-vocabulary VLM labels, no training, images remain the source of readings.

- **The survey's model maps onto our tree directly in form.** Our object > face > sticker/port array > label/port is just `L` with four layers and part-of edges. The survey only shows one level below "object". Deeper within-object nesting is not covered, so the survey gives no precedent for 3–4 nested part levels.
- **"Coloured geometry" in the survey sense = node + `V_g` subset.** A point belongs to several levels at once only implicitly: it sits in the `V_g` of a leaf, and the leaf's ancestors via hierarchical edges. The survey does not say whether a parent's `V_g` must equal the union of its children. That rule would be ours to define.
- **Fusion rules named are all for class labels.** Bayesian, majority vote, adaptive aggregation. With open-vocab VLM text there is no fixed class set. The survey's own semantic-consistency challenge (§4.3) says this is rarely modelled. OpenLex3D [67] is the only pointer for scoring label similarity.
- **Pose offset ~10 mm vs details ~5 mm.** The survey's consistency methods are SLAM-scale (loop closure, factor graphs over rooms). Back-projection of masks with a 10 mm error would put a 5 mm label on the wrong cells. The survey's anchoring remark (associate via "persistent geometric structures", §4.3) and S-Graphs' "structural entities ... constrain pose" hint that larger stable levels (faces) could anchor smaller ones. The survey does not discuss sub-cm precision at all.
- **No-training filter.** The batch learned family (3DSSG, SGFormer, VL-SAT, Lang3DSG) needs training on scene datasets. The lifting family (ConceptGraphs, Clio, HOV-SG, FunGraph, KeySG) uses frozen foundation models. Only the second family fits "no training".
- **"Images remain the source of readings" fits two survey threads.** (a) Deferred semantics / hybrid: keep a few explicit attributes, retrieve details "on demand" (§4.1). (b) KeySG's keyframe + RAG indexing and Open3DSG's "most clearly visible" view selection: the graph points back to the best images rather than holding the reading.
- **Semantics-aware best-view selection** appears in the survey only as Open3DSG's view subset before VLM inference. Pivoting views around a detail is not covered; planning in the survey is navigation-scale.
- **Inward, single object is outside the survey's scenes.** All datasets in Table 1 are rooms, buildings, streets or simulators. The single inspection-like work (xFLIE [146]) is outdoor/infrastructure by its title.

---

## 7. Summary table

| Mechanism | Example methods (survey refs) | Where label lives | How levels relate | Fits us? (my reading) |
|---|---|---|---|---|
| Layered graph over a dense map (layer 1 = mesh/points) | Hydra [61,62], Kimera [125], Armeni [4] | Node; node → subset of map points (`V_g`) | Hierarchical edges between adjacent layers only | Form fits; survey silent on label storage in the mesh and on sub-cm detail |
| 2D mask lifting + multi-view fusion (closed-set) | Kimera [125], Armeni [4], BBQ [82] | Object node built from fused back-projected masks | Flat, or parent by containment | Lifting fits; class-based fusion (Bayes, vote) does not fit open-vocab text as-is |
| Open-vocab lifting with CLIP features / captions | ConceptGraphs [53], Clio [93], HOV-SG [154], KeySG [155] | Node attribute: embedding and/or caption | HOV-SG / KeySG: floor → room → object parent edges | No training needed; levels are room-scale, not within-object |
| Task-driven / on-demand granularity | Clio [93], FOUND-it [94], ASHiTA [23] | Primitives clustered into task-relevant nodes (Clio) | Layers chosen by task | Directly relevant; survey gives titles only — fetch |
| Functional / part nodes | FunGraph [126], OpenFunGraph [174], T-funs3d [42], SceneFun3D [37] | Sub-part nodes (knob, handle) | "is part of" edges; parent by "geometric cues" | Closest to our tree, but only one level below object |
| Articulated part models | MoMa-SG [14], ArtiSG [54], Pandora [168] | Part node + kinematic attribute | object → part | Low; our parts do not move |
| Batch learned 3DSSG prediction | Wald [148], SGFormer [89], VL-SAT [153], Lang3DSG [73] | Node on pre-segmented cloud; predicted class/predicate | Flat | No — needs training |
| Graph as factor graph with geometry | S-Graphs [9–11], VS-Graphs [144] | Structural nodes (planes, rooms) in an optimiser | Floors/rooms constrain pose | Idea of big levels anchoring geometry may transfer; survey shows it at building scale |
| Per-observation nodes + later merge | Khronos [133] | Node per observation | Identity resolved later | Relevant to identity/merging; survey very thin |
| Gaussian / radiance-field grounding | DynamicGSG [48], GaussianGraph [150], RelationField [76], LERF [69] | Gaussians or implicit field; graph on top | — (not described) | Unclear from survey; LERF-type fields usually need per-scene optimisation |
| Hybrid explicit + implicit, details on demand | Proposed direction (§4.1, §6); KeySG RAG [155] | Few explicit attributes; rich info implicit or in images | — | Matches "images stay the source"; survey states it as open, not solved |
| Best-view selection before VLM reading | Open3DSG [75] | — (view subset per node) | — | Relevant to semantics-aware view choice; one sentence only |
