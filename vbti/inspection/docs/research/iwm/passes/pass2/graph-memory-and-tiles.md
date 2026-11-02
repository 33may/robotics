# Pass 2: graph memories and 3D Tiles. How labels get attached to 3D content

**Question.** How can a 3D representation be "coloured" with semantic labels, ideally hierarchical ones, so that one cell or point belongs to object > face > sticker > text line all at once?

**Sources read in full:**
- `papers/md/ali-2025-graphpad.md` (GraphPad, arXiv 2506.01174 v1). The PDF has 8 pages and the md matches it.
- `papers/md/lee-2026-obsgraph.md` (ObsGraph, arXiv 2606.24068 v1). The PDF has 17 pages: main text plus references. **The supplementary material is not in the corpus.** The paper sends the reader there for the incremental update, the prompts and the retrieval details.
- `papers/md/zoomeye-2024.md` (ZoomEye, arXiv 2411.16044 v4). The prompt templates are vector figures in the md, so I read them from the PDF (App. D.1/D.2).
- `papers/md/3d-tiles-spec.md`. This is the 3D Tiles **core README only**. It *references* the Batch Table, the feature table, `EXT_structural_metadata`, the 3D Metadata Specification and the Implicit Tiling spec, but **does not contain them**. Anything said here about per-feature or per-vertex tables comes only from the README's short mentions.

**Conventions.** Quotes are verbatim, and § numbers are the source's own. Sections marked **[my reading]** are my interpretation for our case, not claims from the source.

---

## 1. GraphPad (Ali et al., 2025)

### 1.1 Where the label lives, and how it links to geometry
- The label lives **on the object node**, together with the geometry. §3.2: "Each node $n_i$ stores a point cloud $P_i$, pooled visual embedding $V_i$, pooled language embedding $L_i$, caption $C_i$, room/floor ID, and a list of keyframes in which the object is visible."
- So one record holds four things: geometry (a point cloud), semantics (caption text plus two embeddings), a coarse-place label (room/floor ID), and **pointers back to frames** (the keyframe list).
- A second, parallel structure holds task notes. §3.2: "Graphical Scratch-Pad: Mirrors $\mathcal{N}$ but adds a free-form notes field initialized empty; the APIs write task-specific information here during reasoning."
- The **reverse index** (frame to labels) lives in the Navigation Log. §3.2: "For each keyframe: room, textual field-of-view tag, egocentric motion label (from pose deltas), and the set of visible node IDs."
- Relations live on edges. §3.2: "Edges encode four view-invariant spatial relations … on top of, subpart of, contained in, and attached to." §3.2 (Edge discovery): "Each predicted edge is stored with a subject-ID, object-ID, relation label, and the VLM's free-form justification string."
- How geometry is produced. §3.2: "Each box is passed to SAM to obtain a mask; the mask is back-projected with depth into a point cloud and voxel-downsampled to 0.02 m. Noise is removed by keeping only the largest DBSCAN cluster."
- API outputs carry evidence. §3.3: "Each returns a JSON patch containing new nodes, edges, or scratch-pad notes plus evidence pointers that are incorporated into the SSM."

### 1.2 Hierarchy
- There is **no explicit tree of levels**. Hierarchy shows up in three places:
  - the `subpart of` edge type (§3.2), a part-whole relation between object nodes;
  - room/floor as a node **attribute**, not as a node layer. §3.2: "floors from height-histogram modes and rooms via watershed segmentation on wall skeletons; room labels derive from CLIP similarity to a fixed set of class names" (adopted from HOV-SG);
  - caption consolidation, which works within one node over time, not across levels. §3.2: "Accumulated captions on a track are periodically compressed by prompting the VLM with the list and asking for a single sentence that faithfully paraphrases all entries."
- Geometric resolution is **one fixed level** (0.02 m voxels). The semantic level of a node (object vs. part) is not tied to any change in resolution.

### 1.3 Who writes, and how merges and identity work
- **Initial build is pipeline code plus a VLM.** §3.2: "For every $k$-th RGB image … we run a VLM detector that outputs bounding boxes and captions."
- **Identity is decided by code, using a 2-of-3 vote.** Eq. (1): $S_{ij}=\mathbf{1}[V_i\cdot V_j>0.7]+\mathbf{1}[L_i\cdot L_j>0.8]+\mathbf{1}[G_{ij}>0.4]$. A match needs the vote to "exceed 2", "where $G_{ij}$ is the fraction of points in $D_i$ within $\delta_g$ of $T_j$ (we use $\delta_g=5$ cm)". "This voting scheme requires at least two of the following conditions … visual feature similarity, caption semantic similarity, or spatial overlap."
  - Note: the text says "exceeds 2" and also "at least two of". Read literally, these disagree. I did not resolve this.
- Merge rule: "Visual and language embeddings of the matched track are updated by an exponential moving average with $\alpha=0.5$; unmatched detections start new tracks."
- **Edges are written by the VLM.** §3.2: "Every three frames we prompt the VLM with the current frame plus the JSON list of visible ⟨bbox, caption⟩ pairs; the model predicts all pairwise relations."
- **At inference, the reasoning VLM writes through three APIs** (§3.3). Each takes "a Frame ID and a natural-language query":
  - `find_objects` adds new detections and fuses them "by associating them with existing tracks or creating new ones";
  - `analyze_objects` stores per-node answers "in the corresponding nodes' scratch-pad entries";
  - `analyze_frame` "jointly discovers undetected objects and annotates existing ones".
- **No conflict handling.** §5: "Our current implementation lacks a verification mechanism for object detection quality. When the VLM misidentifies an object or relation, this error becomes part of the scene graph."

### 1.4 Queries, and what the VLM sees
- §3.2: "the Scene Graph, Scratch-Pad, and Navigation Log are serialized to JSON, while Frame Memory is supplied as interleaved images with their Frame IDs."
- Label to place goes through the node's keyframe list and point cloud. Place to labels goes through the Navigation Log ("visible node IDs" per keyframe). Granularity is object-level only.
- An answer must cite both kinds of evidence. §3.1: "we prompt the VLM that it must support its final answer with dual evidence—visual evidence from frames in the Frame Memory and semantic evidence from notes in the Scratch-Pad."
- Ablation (Table 2, 184-question subset): adding the Navigation Log to frames gives "+9.6 percentage points over frame-only". The static scene graph alone gives "+1.7". The frame-level API beat the node-level API (50.5 vs 47.1).

### 1.5 Limits and assumptions
- Episodic memory, no new sensing. §3: "no new sensing is possible after deployment."
- §5 lists: error propagation; an API set "not rigorously evaluated"; latency "typically 2-3 seconds per call"; tested only on "static scenes"; "As scene graphs grow larger, both the prompt size and reasoning complexity increase."
- Small search depths can hurt. §4.3: "limited search ($m=1$ or $m=2$) sometimes performs worse than no search … preliminary updates without follow-up refinement can introduce misleading information."

### 1.6 Transfer to our case **[my reading]**
- GraphPad's node is already "one record = geometry + text + embeddings + frame pointers". That is the closest of the four sources to "semantics and geometry in one structure".
- Its hierarchy is weak: `subpart of` edges plus a room attribute. Object > face > sticker > line could be written as a chain of `subpart of` edges. The paper does not test deep chains, and it gives each level no geometric support of its own beyond each node's own point cloud.
- Its scale numbers are room-scale: 0.02 m voxels and a 5 cm association radius. Both are larger than our ~5 mm details and our ~10 mm pose offset. A purely geometric overlap test would be ambiguous at our scale. In the 2-of-3 vote, the visual and caption terms could still decide.
- The Navigation Log maps one-to-one onto "which view saw which detail". It was the biggest single gain in their ablation.
- The VLM-writes-JSON-patch pattern fits "open vocabulary, no training". Their stated weak point, no verification of VLM writes, applies to us directly.

---

## 2. ObsGraph (Lee et al., 2026)

### 2.1 Where the label lives, and how it links to geometry
- **Observation-centric.** Nodes are images or crops, not geometry. §3.1: "we retain and hierarchically organize the visual observations themselves across multiple abstraction levels."
- **Object node = an image crop.** §3.1: "Each object node is defined as the object-centric crop from the image that best captures the object." The object set itself is updated "following [10]" (ConceptGraphs).
- **The object-to-view link is a visibility matrix.** §3.1: "a visibility mask $\mathcal{M}$ is updated, where $\mathcal{M}_{i,j}=1$ if the $j$-th object $o_j$ is detected in the $i$-th view $v_i$ … Edges between the object and the view layer are defined by $\mathcal{M}$."
- **Room node = a semantic room type, with probabilistic membership.** §3.1: "we request VLM to output room-wise probabilities supported by the view $P(v_j|r_i=k)$ and accumulate across views via Bayesian update."
- **Geometry is kept outside the graph.** §3.3: "In addition to the scene graph, we maintain several geometric representations that support option construction. We build an occupancy map … We also maintain a room segmentation map, obtained by propagating SAM3 predictions from view nodes to the top-down map."

### 2.2 Hierarchy
- Three layers. §3.1: "a room layer that serves as a coarse spatial anchor for lower layers, a view layer that preserves object co-visibility as contextual information, and an object layer that provides fine-grained detail for each object."
- **The view layer is itself a tree**, built from an agglomerative clustering of objects by co-visibility. §3.1: $\mathrm{Sim}(C_i,C_j)=|V(C_i\cup C_j)|/|V(C_i)\cup V(C_j)|$, where the view set of a cluster is "the view [that] observes at least one object from each child cluster".
- Each cluster gets one representative image. §3.1: "we traverse it in inverse merge order and assign each cluster $C$ a representative view $v_C\in V(C)$ … selected as the view most dissimilar to previously selected views based on CLIP similarity computed over objects in $C$. If no sufficiently distinct view exists, the cluster is pruned, and its children are reattached to the parent."
- **How a parent relates to its children.** A parent cluster is a *set of objects*, and its image shows at least one object from each child. It is an image that covers the children, not a text summary of them.
- Each level maps to a *scale of observation*, not a geometric resolution. Rooms are coarse. Views give context. Crops give "fine-grained visual evidence that cannot be easily captured by the view layer" (§1).

### 2.3 Who writes, and how merges and identity work
- Object identity and visibility come from the ConceptGraphs pipeline (code) (§3.1).
- The clustering and view assignment are code, run incrementally. §3.1: "Newly observed objects are locally integrated into the hierarchy, preserving stable higher-level abstractions while refining lower-level clusters." §4.1: "cluster nodes with height $H\geq 3$ … are fixed during updates." The full update procedure is "in the supplementary material" (not in the corpus).
- **The room layer is written by a VLM** (GPT-4o-mini, §4.1) and fused conservatively. §3.1: "For room-view assignments, we combine two sources: (1) direct room predictions, and (2) object-level cues … This conservative fusion improves robustness against noisy VLM predictions."

### 2.4 Queries, and what the VLM sees
- **Coarse-to-fine LLM retrieval with a fixed budget.** §3.2: "At each layer, we query the LLM once to select up to $k$ nodes … The candidate space for the next layer is restricted to nodes structurally connected to the selected nodes." §4.1 sets $k=5$.
- What is shown at each layer (§3.2):
  - room: "room types and the object classes assigned to each room";
  - view, in two stages: first "identify task-relevant object classes", then "greedily select the minimum number of views to cover these classes";
  - object: crops, "when focused observation of the object is assumed to be important".
- Why view selection has two stages. §3.2: "When we queried LLM to select views in a single step, we found that the relevance reasoning was interrupted by object co-visibility information."
- **Semantics drives the geometry, via the hierarchy level.** §3.3: "if a room is retrieved, unexplored regions within that room become exploration candidates. If a view is retrieved, adjusting the viewpoint around that view may complement the missing information."
- **View refinement** (§3.3): "closer view, wider context, and different angle". "If the VLM selects this option, it outputs the index of the view to refine, a description of the task-relevant region being observed, and the appropriate refinement strategy. We apply SAM3 to obtain a mask corresponding to the described region, and a preset function associated with the chosen strategy computes the target position."
- **The VLM is kept out of coordinates.** §3.3: "We restrict the VLM to output high-level exploration selection, decoupling semantic reasoning from computation of precise target coordinates, as VLMs lack accurate coordinate-level reasoning ability."
- Ablation (Table 4, 300-episode subset): replacing their view construction with 3D-Mem's (R2) gave "the largest performance drop despite using the fewest images" (54.2 vs 63.5).

### 2.5 Limits and assumptions
- §5: "our approach relies on VLM reasoning to interpret retrieved observations and select exploration options, which may be sensitive to prediction errors in perception or language understanding. Future work could investigate more robust integration of geometric reasoning."
- The method assumes a planner. §4.1: "we assume a collision-free trajectory planner navigates the agent to that location."
- Room-scale, navigation setting (Habitat). Implementation details are in a supplement that is not in the corpus.

### 2.6 Transfer to our case **[my reading]**
- ObsGraph is the clearest example of "the images remain the source of readings". Nodes *are* images or crops, and labels are attached to them.
- Its three layers map loosely onto ours: room ≈ face, view ≈ a captured frame, object ≈ a sticker or port crop. Its levels, though, are tied to observations, not to a 3D part tree. A text line would be a crop of a crop, and the paper has no such level.
- The three view-refinement strategies (closer, wider, different angle) are close to "pivot views around a detail". The division of labour is explicit: the VLM names the region and the strategy, SAM gives a mask, and code computes the pose.
- The object-by-view visibility matrix $\mathcal{M}$ is a simple, explicit form of "which view saw which detail". That holds no matter what geometry sits underneath.
- Geometry lives *outside* the graph (occupancy and room maps). That is not the "one data structure" Anton has decided on. It is a data point that this split worked for them.

---

## 3. ZoomEye (Shen et al., 2024)

### 3.1 Where the label lives, and how it links to geometry
- **2D only.** Each node is an image region. §3.2: "A specific node, denoted as $n_t$, represents an image patch view $\{\mathbf{I},\mathbf{b}_t\}$, where $\mathbf{I}$ is the image and $\mathbf{b}_t=(x_{1,t},y_{1,t},x_{2,t},y_{2,t})$ is the normalized bounding box coordinates."
- **No persistent labels.** A node holds query-specific scores, computed lazily (Alg. 2: "if $n_t$.priority is None then …"). §3.1 lists generic node attributes: "id", "depth", "value … numeric or textual data", "children", "Other custom attributes".
- Scores are VLM Yes/No probabilities. §3.3: "existing confidence and latent confidence, denoted as $c_e$ and $c_l$". The prompts are "Is there a {o} in the zoomed-in view?" and "is it possible to find a {o} by further zooming in the current view?" (App. D.2 prompt templates, from the PDF).

### 3.2 Hierarchy
- **A fixed geometric quadtree.** §3.2: "If the size of $n_t$'s image patch exceeds the predefined resolution by the image encoder, it can be further divided into four equal-sized sub-patches, serving as its children … Nodes are recursively divided until they meet the resolution limit."
- Depth decides which score counts more. §3.3: "When the depth is shallow … assign more weight $c_l$. As depth increases, shift more weight to $c_e$." With $\mathcal{W}=\frac{1-b}{D^2}d^2+b$ (§4.1).
- **Semantic and geometric levels do not line up**, and the authors say so. §6: "the image is partitioned into fixed-size patches … which may not align well with the semantic regions of the image. As a result, some visual cues may be fragmented or overlooked."
- Split factor matters little. Table 5: 4, 9 and 16 sub-regions give similar accuracy, with fewer search steps for larger splits.
- A cited alternative in which **parents summarise children** (App. B.1, DC²): "The MLLM then generates textual descriptions for each leaf patch. These descriptions are then relayed to the parent nodes, which create combined descriptions by synthesizing the contents from their child nodes with their own." ZoomEye contrasts itself: DC² is "question-agnostic … may lead to unfocused textual content".

### 3.3 Who writes, and how merges and identity work
- Nothing is persisted. For each question, the MLLM first produces "visual cues" from in-context examples (§3.5). Example from the App. in-context set: "Tell me the number on the black board … I need the information about the following objects: black board". The tree is then searched for each cue.

### 3.4 Queries, and what the VLM sees
- Cue to region. "Type 1" cues stop at the first node where the answering confidence $c_a\ge\tau$. "Type 2" ("all …") cues collect every node with $c_e\ge\tau_2$, up to depth 2 (§3.5, App. D.3).
- Final input is the union box. §3.5: "we union the bounding-box coordinates of all nodes in $L$ to create a union bounding-box." In Global+Local mode, the VLM sees the full image plus the crop, with "the visual prompt with a red rectangle to emphasize the local focus" (§3.2).
- Zoom success matters. Table 4: 93.45 when "the searched box covers at least 50% of the target object" vs 54.55 when it does not.

### 3.5 Limits and assumptions
- §6: heuristic ranking and stopping; fixed patches; "less applicable to document understanding tasks, where layout, reading order, and structured information (e.g., tables, forms) are central."
- §4.3: on RS/Position, "the model struggles to link positional relationships between the full image and sub-images". Single-image only. No 3D, no memory.

### 3.6 Transfer to our case **[my reading]**
- ZoomEye shows how to *search* a geometric hierarchy with a VLM, using a "is it here / could it be found deeper" pair of scores. It says nothing about how to *store* labels.
- Its own admission matters for us: fixed geometric splits do not line up with semantic parts. Our details (stickers, port arrays) are semantic regions. A pure octree or quadtree over the object would face the same mismatch.
- The "document understanding" limit (layout, reading order) touches our sticker and text-line level directly.
- DC² (the cited child-to-parent text summary) is the only explicit "parent summarises children" mechanism in this set of sources.

---

## 4. OGC 3D Tiles (core spec)

### 4.1 Where the label lives, and how it links to geometry
- **Metadata can attach at five granularities.** Metadata §: "Metadata may be associated with high-level entities like tilesets, tiles, contents, or features, or with individual vertices and texels." The list:
  - "**Tileset** … year of collection, author details";
  - "**Tile** … the timestamp when a tile was last updated or the maximum height of the tile, or spatial hints";
  - "**Groups** … Each group definition represents a metadata entity that can be assigned to the tile contents by specifying the index within this list as the `group` property";
  - "**Content** … such as a list of attribution strings";
  - "**Features** glTF 2.0 assets with feature metadata … The `EXT_structural_metadata` extension allows associating metadata with vertices or texels."
- **What a feature is.** Introduction: "The content references a set of _features_, such as 3D models representing buildings or trees, or points in a point cloud. Each feature has position and appearance properties and additional application-specific properties."
- **Typed schema plus instances.** Metadata Schema §: "It contains a definition of the metadata classes, which are templates for the metadata instances." Properties have types (`SCALAR`/`FLOAT32`, `STRING` arrays, `ENUM`). Assigning Metadata §: "Each metadata entity contains the name of the class that it is an instance of, as well as a dictionary of property values … An entity may provide values for only a subset of the properties of its class, but class properties marked `required: true` shall not be omitted." Example: `"metadata": {"class": "building", "properties": {"height": 16.8, …}}`.
- **An untyped escape hatch.** Extras §: "The `extras` property allows application specific metadata to be added to any 3D Tiles JSON object" (example: `"extras": {"name": "Empire State Building"}`).
- **Per-class statistics** (Metadata Statistics §): "count", min/max/mean/…, and "occurrences" (enum value counts), plus app-specific `_*` statistics.
- **The binary per-feature tables are not in this copy.** The text only points to them: "(see Batch Table)", `EXT_structural_metadata`, and implicit-tiling metadata "stored in a more compact binary form. See Implicit Tiling."
- **No pointer back to source images or frames is defined.** Content points to renderable files (`content.uri`).

### 4.2 Hierarchy
- **A tree with spatial nesting.** Introduction: "Each tile has a _bounding volume_ … The tree has spatial coherence; the content for child tiles are completely inside the parent's bounding volume." Bounding-volume § adds: "This does not imply that a child's bounding volume is completely inside its parent's bounding volume."
- **Resolution per level.** Geometric error §: "Generally, the root tile will have the largest geometric error, and each successive level of children will have a smaller geometric error than its parent, with leaf tiles having a geometric error of or close to 0." A client refines when screen-space error exceeds a maximum.
- **Parent and children can coexist or replace each other.** Refinement §: "If the tile has replacement refinement, the children tiles are rendered in place of the parent … If the tile has additive refinement, the children are rendered in addition to the parent tile." "A tileset can use … any combination of additive and replacement refinement." "When omitted, a tile inherits the refinement type of its parent."
- **Inheritance and override.** I found only these rules in this copy:
  - `refine` is inherited (quoted above);
  - transforms compose top-down ("post-multiplying a child's `transform` with its parent's `transform` like a traditional scene graph");
  - `tilesetVersion` in external tilesets: "If an external tileset defines `asset.tilesetVersion`, this overrides the value from the parent tileset. If the external tileset does not define `asset.tilesetVersion`, the value is inherited from the parent tileset";
  - schemas: "Any class that is defined in the schema of an external tileset shall also be defined in the top-level tileset schema."
  - **I found no rule in this copy that inherits metadata *property values* from parent tile to child tile.**
- **A tree of trees.** External tilesets §: "a tile's `content.uri` can point to an external tileset … This enables, for example, storing each city in a tileset and then having a global tileset of tilesets."
- **Several representations, or overlapping categories, per tile.** Tile Content §: "a single tile may contain multiple representations of the same geometry data", e.g. "one as a triangle mesh and one as a point cloud". `content.group` puts contents into categories that can span tiles ("Contents of different tiles or the contents of a single tile can be assigned to groups").
- **Structure-only nodes.** Grids §: "empty tiles: those tiles that have a bounding volume, but no content."
- **Avoiding split features.** Quadtrees §: "loose quadtrees, where child tiles overlap but spatial coherence is still preserved … useful to avoid splitting features, such as 3D models, across tiles."
- **The spatial tree can be anything with spatial nesting.** Spatial data structures §: "k-d trees, quadtrees, octrees, and grids". Non-uniform subdivision and "tight bounding volumes" are allowed.
- **Local frames.** Transforms §: "a building tileset inside a city tileset can be defined in its own coordinate system, and a point cloud tileset inside the building could, again, be defined in its own coordinate system." The CRS § notes that a tileset need not be geospatial: "a power plant may be defined fully in its local coordinate system". Units are meters.

### 4.3 Who writes, and how merges and identity work
- The spec defines a file format, not an update process. The relevant hooks:
  - `tilesetVersion`, "for when an existing tileset is updated", usable "as a query parameter … to avoid using outdated content from a cache";
  - implicit tiling allows "random access of tiles based on their tile coordinates which enables … efficient updates of tile content".
- There are no merge, conflict or identity rules for features in this copy.

### 4.4 Queries
- Introduction: "A client may choose to select features at runtime and retrieve their properties for visualization or analysis."
- Metadata use cases: "**Inspection:** … click or hover over specific tiles or tile contents, showing informative metadata"; "**Collections:** Tile content groups may be used to define collections (similar to map layers)".
- Styling evaluates expressions over feature properties, e.g. `"(${Height} > 90) ? color('red') : color('white')"`.
- Geometric queries are built into traversal: bounding-volume culling, `viewerRequestVolume` ("the viewer shall be inside of before the tile's content will be requested"), and screen-space-error refinement.
- There is no label-to-place index in the core spec. Finding "all features of class X" relies on traversal plus property lookup. Only aggregate statistics are precomputed.

### 4.5 Limits and assumptions
- The purpose is streaming and rendering. Introduction: "3D Tiles does not define explicit rules for visualization of the content."
- The spec is about geometry and display. Nothing ties a label to an observation, a confidence or a source image.
- The parts that matter most for per-point or per-feature labels (Batch Table, `EXT_structural_metadata`, 3D Metadata Spec, Implicit Tiling) are **outside the corpus copy**.

### 4.6 Transfer to our case **[my reading]**
- 3D Tiles is the only source where **every level of a spatial tree is a first-class metadata carrier**: tile, content, group and feature, each with a typed class.
- Object > face > sticker > line maps naturally onto nested tiles with shrinking bounding volumes. A deep feature is "inside" all its ancestors by spatial nesting, so its label path is the chain of ancestor tiles. That is one way to get "belongs to all levels at the same time".
- `ADD` refinement is the mode where parent and child are both present at once. `REPLACE` is the mode where the child stands in for the parent.
- `content.group` gives a second, non-spatial way to group across the tree (like layers). For example, "all serial-number lines" could be one group spanning faces.
- Per-tile `transform` and local frames fit a single object in its own frame, with faces in sub-frames.
- `geometricError` is a per-level resolution number in meters. It is the closest concept here to "the semantic level relates to geometric resolution". The spec ties it to rendering error, not to semantics.
- Gaps for us: no links to frames or pixels, no confidence, no update or merge semantics, and no inheritance of property values (in this copy). These would have to come from elsewhere, e.g. GraphPad's keyframe lists or ObsGraph's visibility matrix.
- Its labels are typed class schemas with enums. Open-vocabulary VLM text would go into `STRING` properties or `extras`, not into enums.

---

## 5. Cross-source table

| mechanism | source | where label lives | how levels relate | fits us? **[my reading]** |
|---|---|---|---|---|
| Object node = point cloud + caption + CLIP/BGE embeddings + keyframe list | GraphPad §3.2 | on the object node, with its geometry | flat; part-whole only through `subpart of` edges; room/floor is an attribute | one record links geometry, text and frames; hierarchy would be edge chains; 2 cm / 5 cm numbers are coarser than our details |
| Navigation Log: frame → visible node IDs | GraphPad §3.2 | per keyframe | none | direct "view ↔ detail" index; largest gain in their ablation |
| Scratch-pad notes written by VLM API (JSON patch + evidence pointers) | GraphPad §3.3 | per node, parallel to the graph | none | matches VLM-writes, no training; no verification of writes |
| 2-of-3 identity vote (visual, caption, spatial overlap) + EMA | GraphPad §3.2 Eq. 1 | n/a (merge rule) | n/a | semantic terms can carry identity when geometry is ambiguous at ~10 mm offset |
| Room–view–object observation graph; visibility matrix $\mathcal{M}$ | ObsGraph §3.1 | on images or crops; room type with Bayesian probabilities | layers; the view layer is a co-visibility cluster tree with one image per cluster | images stay the evidence; levels are observation scales, not 3D parts; geometry kept outside |
| Coarse-to-fine LLM retrieval, ≤k per layer | ObsGraph §3.2 | n/a (query) | each layer restricts the next | pattern for walking object → face → detail |
| View refinement (closer / wider / different angle): VLM names region and strategy, SAM masks, code computes pose | ObsGraph §3.3 | region text from the VLM | from a retrieved view to a new viewpoint | close to "pivot around a detail"; VLM kept out of coordinates |
| Image quadtree + exist/latent confidence | ZoomEye §3.2–3.3 | not stored; per-query scores on 2D patches | fixed 4-way split; depth-weighted scores | search pattern only; authors note splits do not line up with semantics |
| Child → parent text synthesis (DC²) | ZoomEye App. B.1 (cited) | text per patch node | parent combines children's descriptions | only explicit "parent summarises children" here; question-agnostic |
| Tile tree with bounding volumes, `geometricError`, REPLACE/ADD | 3D Tiles core | tile and content `metadata` | spatial nesting; `refine` inherited; transforms compose | nested BVs could carry object > face > sticker > line; no frame links or confidence |
| Typed metadata classes at tileset / tile / group / content / feature (vertex/texel via glTF ext) | 3D Tiles Metadata § | at any level; class + properties | per-level entities; no value inheritance found in this copy | the industrial "label at every level" pattern; open vocabulary would go into STRING/extras |
| `content.group` | 3D Tiles Tile Content § | group metadata entity, referenced by index | cuts across tiles | cross-tree categories, e.g. all text lines |

---

## 6. Cited references worth fetching later (not fetched)

- **HOV-SG.** Werby et al., "Hierarchical open-vocabulary 3D scene graphs for language-grounded robot navigation" (GraphPad [26]). An explicit floor/room/object hierarchy with open-vocabulary labels; GraphPad borrows its room/floor step.
- **Search3D.** Takmaz et al., "Hierarchical open-vocabulary 3D segmentation" (GraphPad [23]). The title says it does hierarchical labels on 3D segments, which is our exact question.
- **ConceptGraphs.** Gu et al. (GraphPad [7], ObsGraph [10]). ObsGraph's object layer is built from it, and both papers depend on it. It explains how per-object point clouds and captions are fused.
- **DC².** Wang et al., "Divide, conquer and combine" (ZoomEye ref). A tree where parents synthesise their children's descriptions.
- **Cai et al. 2025.** "Vision to geometry: 3D spatial memory for sequential embodied MLLM reasoning" (ObsGraph [6]). Scene graph plus keyframes "selected by geometric coverage".
- **3D-Mem.** Yang et al. (both papers). The baseline both compare against: informative image snapshots kept alongside structure.
- **Clio.** Maggio et al., "Task-driven open-set 3D scene graphs" (GraphPad [13]). Task-conditioned granularity of graph nodes.
- **OpenScene / LERF / OpenMask3D** (ObsGraph [22], [17], [28]). Dense per-point or per-field language features, the "colour every point" end of the spectrum.
- **3D Tiles companions (not in corpus):** 3D Metadata Specification; glTF `EXT_structural_metadata`; Batch Table / Feature Table (legacy); Implicit Tiling spec. These hold the actual per-feature, per-vertex and per-texel property-table mechanics.
  - From my own knowledge, not cited in this copy: glTF `EXT_mesh_features` is where feature IDs per vertex or texture are defined.
