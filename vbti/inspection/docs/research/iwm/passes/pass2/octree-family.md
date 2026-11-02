# Pass 2 — Octree family: how labels "colour" a 3D structure

Question: how can a 3D representation carry semantic labels, ideally hierarchical ones (one cell belongs to object > face > sticker > text line at the same time)?

Corpus: three papers, read in full (markdown + PDF checks for formulas and figures).
- Hornung et al. 2013, *OctoMap* (Autonomous Robots) — `papers/md/hornung-2013-octomap.md`
- Wurm et al. 2011, *Hierarchies of Octrees for Efficient 3D Mapping* (IROS) — `papers/md/wurm-2011-hierarchies-of-octrees.md`
- Larsson et al. 2022, *Information-theoretic Abstraction of Semantic Octree Models* (arXiv 2209.10035) — `papers/md/larsson-2022-semantic-octree-abstraction.md`

Notes on sources:
- OctoMap and Wurm markdown lost formulas and tables. Formulas below were checked against the PDFs.
- Larsson's markdown keeps the LaTeX. Equation numbers (6)–(10) were checked against the PDF.
- Larsson gives **no** map resolution, sensor type or cell size anywhere in the paper.

Each paper section has parts 1–5 (what the paper says) and part 6 (**my reading**, transfer to our case). Our case, for reference: inward inspection, one object on a table, details down to ~5 mm labels, ~10 mm per-view pose offset, labels named open-vocabulary by a VLM, images stay the source of readings.

---

## A. Hornung et al. 2013 — OctoMap

### A1. Where the label lives
- Per octree node (voxel). The core payload is occupancy, stored as one log-odds float per node. §4.1: "To store a per-voxel occupancy probability, a single float value (usually 4 byte) is sufficient to represent the log-odds value."
- The node payload is generic. §4.2: "OcTreeDataNode is templated over data that is stored in the node." Extending nodes is the intended route: "This structure allows for flexible extensions of our framework at different levels, e.g., to extend nodes with custom data".
- §3.5.1 names possible extra payloads: "Voxels could, for example, store terrain information, environmental data such as the temperature, or color information."
- The worked example is colour, not a semantic class. §4.2: "ColorOcTree that uses ColorOcTreeNodes … These nodes store color in addition to an occupancy estimate".
- §6.2 case study: nodes "were extended to store a time stamp attribute".
- No class id, class distribution, instance id, embedding, text or pointer is stored in the paper itself. Semantics is only a downstream use: colour "enables a color-based classification of the environment" (§3.5.1).
- Inner nodes also store the payload. §4.1: "the octree is homogeneous by design, that is, all nodes have the same structure and store occupancy."

### A2. Label form and fusion
- Occupancy: binary Bayes filter in log-odds. Eq. (2): `L(n | z1:t) = L(n | z1:t−1) + L(n | zt)`.
- Clamped. Eq. (4): `L(n | z1:t) = max(min(L(n | z1:t−1) + L(n | zt), lmax), lmin)`. Values used: `lmin = −2`, `lmax = 3.5` (probabilities 0.12 and 0.97); sensor model `locc = 0.85`, `lfree = −0.4` (§5.1).
- Disagreement between views is handled by evidence counting. §3.2: "if a voxel was observed free for k times, then it has to be observed occupied at least k times before it is considered occupied". Clamping bounds this, so "the model can adapt to changes in the environment quickly".
- Extra properties need their own fusion rule. §3.5.1: "Each additional voxel property requires a method that allows several measurements to be fused." For colour: "store the average color of each voxel."
- No conflict rule for non-occupancy data is given beyond "average".
- Same-scan conflict: §5.1, "whenever a voxel is updated as occupied … it is not updated as free in the same measurement update".

### A3. Hierarchy
- The octree is a **spatial** hierarchy. Depth = cell size, not semantic level. §3.1: "the tree can be cut at any level to obtain a coarser subdivision if the inner nodes are maintained accordingly."
- A parent summarises its eight children by a fixed aggregate. §3.3, Eq. (5) average `l̄(n) = 1/8 Σ L(ni)` or Eq. (6) maximum `l̂(n) = max_i L(ni)`. OctoMap uses max: "the maximum occupancy update is used in our system."
- Measurements only update leaves. §3.3: "probabilistic updates are performed only for the leaf nodes in the octree."
- Pruning merges children into the parent when they agree. §3.4: "If all children of an inner tree node are stable leaf nodes with the same occupancy state, then the children can be pruned." Contradicting data later: "its children are regenerated". §3.1 notes the same works for "discrete labels".
- One payload per node. No mechanism for several coexisting labels of different levels.
- §3.5.2 "Octree Hierarchies" points to Wurm 2011 (section B below): "maintains a collection of submaps in a tree-structure, where each node represents a subspace of the environment." Fig. 6: "Background (yellow), table (magenta), and objects (cyan) are represented by individual octree maps of different resolutions." Claim: "about an order of magnitude more compact than a single map".

### A4. Queries
- Point → node: "Individual octree nodes can be accessed by searching for their coordinate" (§4.4.2).
- Region → nodes: iterators over "all nodes, leaf nodes, or leaf nodes in a certain bounding box", which "can be filtered according to further criteria" (§4.4.2). A label filter would be one such criterion; the paper does not show one.
- Level: depth-limited queries (§3.3, §5.5.2): "The depth of a query can be limited during run time which in our data structure is equivalent to a map query in a coarser map."
- Visibility: `castRay(·)`, "casting a ray from an origin into a given direction until it hits an occupied volume … used for visibility checks" (§4.4.2).
- No label → region index. Finding all voxels with a property means traversal.

### A5. Limits and assumptions
- Built for range sensors and robot-scale scenes. Error "in the order of centimeters" (§1). Resolutions in experiments: 2 cm to 80 cm (Table 2).
- Poses are given by an external method (scan matching, visual odometry, RGB-D SLAM; §5.2, §3.5.1). Pose error is not modelled.
- Max depth 16 levels; at 1 cm this covers 655.36 m per axis (§4.2).
- Max-likelihood file format loses extra node data; extra data needs the full format, "considerably larger files" (§4.3).
- Clamping loses information near 0 and 1 (§3.2, §5.6).
- The paper offers no semantic label model of its own.

### A6. Transfer to our case — MY READING (not the paper)
- Directly reusable: the idea that a node is a **slot for an arbitrary payload** plus a fusion rule per payload type. A label distribution, an instance id, or a list of image/observation ids could live there.
- Reusable: inner node = aggregate of children, recomputed bottom-up. This gives "what is inside this region" cheaply at a coarser level.
- Reusable: `castRay` on the same structure that holds labels. This is the hook for visibility of a labelled region from a candidate view.
- Breaks: depth is cell size, not semantic level. A 5 mm sticker and the face it sits on are both just leaves. Nothing says "this leaf is part of sticker S, which is part of face F".
- Breaks: one payload per node. Co-membership (object and face and sticker at once) is not modelled. You could pack several fields per node, but the paper does not do or discuss it.
- Risk: a ~10 mm per-view pose offset is larger than a 5 mm cell. Fusing labels per voxel across views would put the same sticker into different voxels. OctoMap assumes accurate poses and has no answer to this.
- Risk: OctoMap fuses range data. Our readings come from images. The structure has no notion of "which image produced this value".

---

## B. Wurm et al. 2011 — Hierarchies of Octrees

### B1. Where the label lives
- Per **sub-map** (map node), not per voxel. §III-A, Eq. (2): `ni = (mi, ti, pi, Ci, li)`. "Each node stores a 3D map mi. Additionally, it holds a reference pi to its parent node and a set of references to child nodes". "A semantic label li is stored in each map node."
- `mi` is an OctoMap (§IV-B). `ti` is a 6D transform relative to the parent: "The 6D transform ti denotes the relative transform from the parent's reference frame to the node's local reference frame."
- Fig. 2 shows the example value `li = table`.
- Each node also has an existence probability (§III-D, Eq. 8).
- What the label is: a symbol from a labelling function. §III-E: "we assume that a labeling function L(z) exists that returns a label given a point cloud z."
- In the implementation the label set is closed and three-valued. Eq. (10): `L′(z)` returns `table`, `object` or `background`.

### B2. Label form and fusion
- The label comes from one segment of one measurement. There is no per-observation label fusion rule in the paper.
- Fusion happens at the node level through association. §III-C: segment the cloud, then "Associate each segment zi to an existing map node using fA", then "Update submaps … or create a new map node if no corresponding node could be found."
- The label gates association. §IV-C: "We then use its semantic label L′(z) to find existing map nodes in the model M′ with the same label." Then oriented bounding boxes are tested for intersection.
- Consequence (stated by the mechanism, not discussed by the authors): a segment with a different label never merges with an existing node. The paper does not describe what happens when two views label the same structure differently.
- Existence is fused with a binary Bayes filter. §III-D: "we estimate a probability of the existence for each node n … using a binary Bayes filter". Eq. (8) is the same form as OctoMap Eq. (1). "The inverse sensor model P(n | zt) is specific to the sensor and the segmentation used for mapping."
- Removal: §VI-B, the hierarchy can "remove the corresponding map node completely after its existence probability falls below a threshold of 0.5".
- Geometry inside each sub-map is fused as in OctoMap, in the node's local frame. Eq. (12): measurements are transformed by `(t_i^global)^−1` before ray casting.
- Node frame: set once from the first measurement. §IV-A: "The translational component is computed from the centroid of the measurement and we use principal component analysis to determine its orientation."
- Object movement: §III-D, "a perceived relative 6D-transformation t … of a submap ni is integrated by updating the corresponding node transform ti ← ti ◦ t."

### B3. Hierarchy
- A tree of labelled sub-maps. §III-A, Eq. (1): "a map M consists of a tree of map nodes".
- Parent–child is a user-given **spatial relation**. §III-B: "This relation r(ni, nj) ⊆ M × M determines whether node ni is at a higher level in the hierarchy than nj. Real-world examples of such relations include the relations supports or contains."
- When several ancestors qualify, the deepest wins. Eq. (7): `a* = argmax_{a∈A} depth(a)`. Example: "a cup standing on a table is supported by the table but also by the floor".
- Several relations may be combined if mutually exclusive. §III-B: "objects are supported by a table and that this table is contained in a specific room. Note that an object supported by the table could not be contained in the table at the same time."
- In the implementation the relation is defined **on labels**. Eq. (11): `(li = table ∧ lj = object) ∨ (li = background ∧ lj = table) ∨ (li = background ∧ lj = object)`.
- Parent and child labels are different symbols. The parent does not summarise the child's label. The parent holds its own geometry; the child's geometry is not in the parent's octree.
- Pose chains down the tree. Eq. (4): `t_i^global = t_{p_i}^global ∘ t_i`. §I: objects on a table "are moved along when the table is moved."
- **Semantic level is tied to geometric resolution.** §IV-B: "An important advantage of the proposed map hierarchy is the ability to adapt the mapping resolution based on the semantic class." Example: objects "very fine resolutions (e.g., millimeters)", table "coarser resolution (e.g., centimeters)", walls and floor "even coarser".
- Values used: §VI-A, table 0.01 m, background 0.05 m, objects at "1 mm, 2 mm, 4 mm, 8 mm, and 16 mm". Exploration (Fig. 7): "Objects are modeled at a resolution of 5 mm, the table top is modeled at 10 mm."
- Tree only. §III-A: "There are certain situations which cannot be modeled using a tree structure (e.g., one object resting on two tables)."

### B4. Queries
- Point/segment → node: the association function "returns the lowest node n ∈ M in the tree that z falls into" (§III-C).
- Label → nodes: filter nodes by label, then OBB test (§IV-C).
- Node → global pose: walk parent references to the root (Eq. 4). "The tree can also be used to locate map nodes more efficiently" (§III-A).
- Node → geometry: its own OctoMap, with all OctoMap queries.
- Information gain for a viewpoint (§V, Eq. 13–15): ray casting from a candidate view, "considering the number of unknown cell along each ray together with the change in entropy ∆H". Assumption: "only previously unknown cells contribute to a change of entropy." Semantics enters only by choosing the area of interest ("findNearestUnexploredTable", Alg. 1).

### B5. Limits and assumptions
- Accurate poses assumed. §III: "We assume that the robot always has an accurate pose estimate." §I: "Estimating the poses of the sensor based on appropriate 3D SLAM methods is beyond the scope of this paper."
- Needs a segmentation `S(z)` and a labelling `L(z)` given from outside (§III-B, §III-E).
- Implementation is specific to "a supporting planes assumption" (§IV-A); objects are clusters above a table plane, cluster threshold "e.g. 0.01 m".
- Closed label set of three. Hierarchy depth in practice: background > table > object (Fig. 1).
- One label per node. No probability over labels.
- Tree only; relations must be mutually exclusive.
- Each node frame is fixed from its first measurement by centroid + PCA; "this cannot guarantee an optimal orientation" (§IV-A).
- Data: PR2 stereo with texture projector, "20 overlapping dense stereo images each" (§VI-A).
- Result: "At very fine resolutions the monolithic map is about one order of magnitude bigger and it takes about one order of magnitude longer". A monolithic 1 mm map "would require in the order of 10 GB of memory" (§VI-A).

### B6. Transfer to our case — MY READING (not the paper)
- This is the only paper of the three with a real **semantic** hierarchy: a tree of nodes, each with a label, its own geometry, its own resolution and its own frame.
- Directly reusable pattern: object > face > sticker > line as nested map nodes, with a relation like "contains" or "lies on". Finer semantic level gets a finer cell size, exactly as the paper ties resolution to class.
- Reusable: the node frame `ti` relative to the parent. A sticker's geometry could live in the face's frame. A per-node transform update (`ti ← ti ∘ t`) is a place where a correction could go. The paper uses it for object motion, not for pose error.
- Reusable: label-gated association ("same label, overlapping box"). With open-vocabulary VLM names this gate breaks: two views may name the same sticker differently. The paper has no rule for that.
- Breaks: the parent–child relation in the implementation is defined on a fixed label set (Eq. 11). Open-vocabulary names would need the relation to come from geometry or from the VLM, not from a label table.
- Breaks: the hierarchy is a tree, relations mutually exclusive. A port array that spans two faces, or a label wrapping an edge, is the "object on two tables" case the authors exclude.
- Breaks: membership is exclusive by construction. A point goes into exactly one sub-map (the lowest that contains it). So "this cell belongs to object and face and sticker" is expressed by the chain of ancestors, not by multiple labels on the cell. That may be enough, or not — it is a structural fact, not a recommendation.
- Gap: no label fusion over views, no label confidence, no link to source images.
- Gap: 10 mm per-view pose offset. The paper assumes accurate poses. Its mm-level object maps came from a static stereo sweep, not from views with independent offsets.
- Note: the exploration utility (unknown cells along rays) is purely geometric. Semantics only selects which table to look at.

---

## C. Larsson et al. 2022 — Semantic octree abstraction

### C1. Where the label lives
- Per **leaf** voxel of the finest octree `T_W`: a truncated class distribution. §3.A: the algorithm "maintains a truncated probability distribution over semantic classes, represented by a random variable S : Ω → K for each leaf node".
- What is stored: the 3 most likely classes, free space, and one "other" bucket. §3.A: "if k ∈ K3(x) ∪ {0} then p(S = k|x) is provided by the octree … The octree also stores a forth entry, corresponding to p(S ∈ K \ (K3(x) ∪ {0})|x)".
- Class set is closed: `K = {0, 1, …, K}`, 0 = free space (§2). Experiment: FCHarDNet pre-trained on RUGD, "providing 24 possible categorizations (i.e., K = {0, …, 24})" (§4).
- Inner nodes hold derived distributions and a "G-value" used for pruning (§3.C, Alg. 1).
- No instance id, no embedding, no text, no pointer to source data.

### C2. Label form and fusion
- Leaf update is taken from earlier work. §3.A: "We adopt the semantic model proposed in [11] to build a hierarchical Bayesian multi-class octree representation of the world." The update rule itself is **not given** in this paper. ([11] = Asgharivaskasi & Atanasov 2021.)
- Input is a semantic point cloud (Alg. 1). Fig. 1: "Semantic octree update from a new observation indicating cell D is blue and C is free space."
- Disagreement between views is absorbed inside the per-cell distribution (Bayesian multi-class). How exactly is in [11], not here.
- Truncation side effect: to recover a full distribution, the "other" mass is spread evenly. §3.B: "uniformly distribute the probability p(S ∈ K \ (K3(x) ∪ {0})|x) among the K − 3 outstanding classes."
- Unobserved children get a uniform distribution. §3.B: "we account for absent children by instantiating a maximum entropy distribution (i.e., uniform over K)".
- Build-time pruning: "the algorithm will prune nodes whose children all have identical multi-class probability distributions" (§3.A).
- Incremental: §3.D, "at each time instance for which point cloud information is available, the semantic tree-building algorithm inserts (or updates) exactly one leaf node". Updates walk up to the root.

### C3. Hierarchy
- Spatial octree. Depth = cell size. The class set is **flat**: building, vegetation, road, etc. No part-of relation between classes.
- Parent = weighted mixture of children. Eq. (3): `p(n) = Σ_{n′∈C(n)} p(n′)`. Eq. (9): `p(yi|n′u) = Σ_{n″∈C(n′u)} Π_{n″} p(yi|n″)` with `Π_{n″} = p(n″)/p(n′u)`.
- Each class is treated as a binary variable. §2: "Yi : Ω → {0, 1}"; §3.B: "p(yi = 1|x) = p(S = i|x)".
- **Semantics decides where geometry stays fine.** The compressed tree keeps cells fine where children disagree about task-relevant classes. Eq. (6): `∆IYi(n) = p(n) JSΠ(p(yi|n′1), …, p(yi|n′|C(n)|))`. Objective Eq. (1): `max Σ βi IYi(T) − Σ γj IZj(T) − α IX(T)`.
- Task weights choose which classes matter: `β` relevant, `γ` irrelevant, `α` compression (§2). Result §4.A: "regions of the map that contain paved road … retain high resolution as compared with areas that contain … grass and trees". Classes neither relevant nor irrelevant "are aggregated whenever doing so does not contribute to a loss of relevant information."
- Several labels per cell exist only as **alternatives** (a probability distribution summing to 1), not as co-membership.
- Contrast with build-time pruning, §4.A: the ad-hoc rule "prunes nodes based on thresholds placed individual elements of the distributions"; G-tree search uses JS divergence and "is able to exploit these redundancies".

### C4. Queries
- Region → class distribution at any node, via Eq. (9)–(10).
- Task → compressed multi-resolution tree: G-tree search returns which nodes are leaves (§3). Called "provably optimal tree abstractions" (§1).
- Region → colour for planning: §4.B, "Graph vertices are classified (i.e., given a color) according to the semantic information contained in the (semantic) octree node corresponding to the volumetric region containing the (x, y)-coordinate". Used with Class-Ordered A* (COA*) on Reeds-Shepp edges.
- No label → region index described.
- Result reported: optimal path "about 10% percent faster on average"; planning-time standard deviation "reduced by 60% percent", over 50 search instances (§4.B).

### C5. Limits and assumptions
- Closed class set from a pre-trained segmenter (24 RUGD classes).
- Top-3 truncation; full distribution is reconstructed approximately (uniform spread of "other").
- Assumes a prior `p(x)` over leaves: "we assume that a distribution p(x) over leaf nodes is provided" (§3). How it is chosen in the experiment is not stated.
- Large outdoor scene, ground robot, 2D (x, y, θ) planning use.
- No resolution, sensor or pose source stated in the paper.
- Pose accuracy is not discussed.
- Hierarchy is spatial only; no notion of parts.

### C6. Transfer to our case — MY READING (not the paper)
- Directly reusable idea: a per-cell **distribution** instead of a hard label, with parents as weighted mixtures. This gives "how much of class X is in this region" at any level for free.
- Directly reusable idea: **semantic relevance drives geometric resolution**. With weights per class, the tree stays fine around what matters (e.g. a sticker) and coarse elsewhere. This is the clearest "semantics informs geometry" mechanism in the corpus.
- The JS-divergence term measures "children disagree about class X". In our terms that is "this region has a boundary of a relevant detail". My reading only; the paper does not use it for viewing.
- Breaks: fixed class index set and top-3 truncation. Open-vocabulary VLM names have no fixed index. The "other" bucket and uniform redistribution assume a known K.
- Breaks: the class set is flat. object > face > sticker > line are not alternatives; a cell is all of them. Putting them in one distribution would make them compete for probability mass.
- Breaks: no instance identity. Two stickers of the same class are one class here.
- Gap: the leaf update rule is in [11], not in this paper. Fusion under pose error is not discussed.
- Gap: no link from a cell back to the images that produced it.

---

## D. Cross-paper: mechanisms for hierarchical semantics

| Mechanism | Paper | Where label lives | How levels relate | Fits us? (my reading) |
|---|---|---|---|---|
| Extensible node payload + per-property fusion rule | OctoMap §3.5.1, §4.2 | Every octree node | Inner node = max/avg of children (Eq. 5–6); depth = cell size, not semantics | Partly. A slot for label/obs-ids exists. No semantic levels, one payload per node. |
| Octree pruning of agreeing children | OctoMap §3.1, §3.4; Larsson §3.A | Leaves; parent replaces 8 identical children | Coarse cell = "all children say the same" | Partly. Compact for uniform faces. Says nothing about part-of. |
| Tree of labelled sub-maps, relation-defined parent | Wurm §III-A/B, Eq. 2, 6–7, 11 | Per sub-map (`li`), one symbol | Parent–child by spatial relation (supports/contains); deepest candidate wins; tree only | Closest to object > face > sticker. Closed label set, tree-only, no label fusion. |
| Resolution chosen by semantic class | Wurm §IV-B, §VI-A | Per sub-map | Finer class → finer cells (objects 1–16 mm, table 10 mm, background 50 mm) | Yes as an idea; our details are ~5 mm. Pose offset ~10 mm is larger than such cells. |
| Frame chain (pose relative to parent) | Wurm Eq. 4, §III-D | Per sub-map (`ti`) | Child pose = parent pose ∘ local transform | Yes as an idea. Used for object motion, not for pose error. |
| Label-gated association | Wurm §IV-C | Per segment → node | Same label + OBB overlap | Breaks with open-vocabulary names. |
| Truncated class distribution per leaf | Larsson §3.A–B | Per leaf: top-3 + free + other | Alternatives in one distribution; flat class set | Partly. Confidence yes. Hierarchy and open vocabulary no. |
| Parent = weighted mixture of child distributions | Larsson Eq. 3, 9–10 | Every node (derived) | Same class set, coarser region | Yes for "what is in this region". Not a different semantic level. |
| Task-weighted, information-theoretic pruning | Larsson Eq. 1, 4–8 | Compressed tree leaves | Relevant classes keep fine cells; irrelevant ones get merged | Yes as a "semantics informs geometry" mechanism. Needs a class index. |
| Existence probability per sub-map | Wurm §III-D, Eq. 8 | Per sub-map | Independent of level | Possibly, for "is this detail really there". |

What none of the three does:
- Store several **co-existing** labels of different semantic levels on one cell.
- Fuse labels across views when poses are off by more than a cell.
- Handle open-vocabulary names.
- Keep a link from a cell or node back to source images.
- Use semantics to choose views (Wurm's view utility is geometric; Larsson's use is 2D path planning).

## E. Cited references worth fetching later (not fetched)

- Asgharivaskasi & Atanasov 2021, *Semantic OcTree Mapping and Shannon Mutual Information Computation for Robot Exploration* (arXiv 2112.04063) — Larsson [11]. Holds the actual Bayesian multi-class leaf update, and mutual information over a semantic octree (view selection).
- Larsson, Maity, Tsiotras 2022, *A generalized information-theoretic framework for the emergence of hierarchical abstractions in resource-limited systems* (Entropy) — Larsson [35]. Full G-tree search and the definitions of `IY`, `IZ`, `IX`.
- Galindo et al. 2005, *Multi-hierarchical semantic maps for mobile robotics* (IROS) — Wurm [4]. Explicit semantic + spatial hierarchies side by side.
- Anguelov et al. 2002, *Learning hierarchical object maps of non-stationary environments* (UAI) — Wurm [1]. Hierarchical object maps.
- Kraetzschmar et al. 2004, *Probabilistic quadtrees for variable-resolution mapping* — cited by both OctoMap §3.3 and Larsson [19]. Strategies for aggregating children into a parent.
- Nüchter & Hertzberg 2008, *Towards semantic maps for mobile robots* (RAS) — Wurm [12]. Semantic labelling on 3D data.
- Blodow et al. 2011, *Autonomous semantic mapping for robots performing everyday manipulation tasks in kitchen environments* (IROS) — OctoMap §6 case study "semantic mapping".
- Rosinol et al. 2020, *Kimera* (ICRA) — Larsson [12]. Metric-semantic mapping at scale.
- Lim & Tsiotras 2021, *A generalized A\* algorithm … in weighted colored graphs* (ICRA) — Larsson [42]. How region colours are consumed by a planner.
