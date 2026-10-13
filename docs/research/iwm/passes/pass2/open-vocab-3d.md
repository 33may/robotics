# Pass 2: colouring a 3D representation with (hierarchical) semantic labels

Corpus pass over three papers. All three were read in full: md, plus the PDF for the tables and thresholds quoted here.
Question: how do you attach semantic labels to a 3D representation, ideally hierarchical ones (object > face > sticker > text line at the same time)?

- Wang et al. 2025, *Open-Vocabulary Octree-Graph for 3D Scene Understanding* (`wang-2025-octree-graph.md`)
- Ying et al. 2023, *OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning* (`ying-2023-omniseg3d.md`)
- Dan et al. 2026, *Octrees as an Explicit 3D Language* (OctLLM) (`dan-2026-octllm.md`)

Each paper has sections 1–5 on what the paper says. Section 6 is **my reading** for our case and is marked as such.

---

## A. Open-Vocabulary Octree-Graph (Wang et al. 2025)

### A1. Where the label lives
- The label lives on an **object (instance) node** of a graph. It does not live on octree cells.
- §3.5: "The node $N_i$ consists of correlated semantics $n_i^s$ (*e.g.*, captions and features), center $n_i^c$, and adaptive-octree $n_i^o$."
- What is stored: one fused instance feature $f_i^*$. It is the average of a fused visual feature (CLIP ViT-H or OVSeg ViT-L) and a fused caption feature (TAP captions) (§3.4, §4.1). Captions are stored too.
- The adaptive-octree stores **occupancy only**. §3.5: the adaptive-octree is "to depict the occupancy information of each object". The abstract says the adaptive-octree "stores semantics and depicts the occupancy". The method text, though, puts the semantics on the node ($n_i^s$), not on octree cells.
- Edges store relations. §3.5: "$E_{i,j}$ comprises the semantic relation $e^s_{i,j}$, spatial distance $e^d_{i,j}$ and the 3D vector $e^v_{i,j}$". Example relation: "right".

### A2. Lifting from 2D and fusing across views
- Per frame (§3.2): CropFormer gives class-agnostic 2D masks. "We then filter out tiny and marginal masks". Each mask gets a visual feature and a caption feature. Each mask is "project[ed] ... into the 3D space as a point cloud segment" using depth, then denoised with DBSCAN. Filters: "masks with pixels less than 25 and segments with points less than 50" (§4.1).
- Merging (CGSM, §3.3): frames are split "into several groups in time order with interval $I$" ($I$=200). Merging is iterative: $\{M_{k-1}, G_k\}$ → $M_k$.
- Merge score: $\phi=\phi^{iou}_{geo}+\phi^{ior}_{geo}+\phi^{v}_{sem}+\phi^{c}_{sem}$. Two segments are merged when $\phi(m,n)\ge\theta_i$. The threshold is "linearly decay[ed] ... at each step" so that partial views and over-segments still merge.
- Containment cue: $\phi^{ior}_{geo}(m,n)=|S_m\cap S_n|/|S_n|$, the share of $S_n$ that lies inside $S_m$.
- Conflict handling, called "semantic-guided under-segment filtering": collect the segments that $S_m$ contains ($\phi^{ior}\ge 0.8$). "If the semantic feature variance of these contained segments exceeds a threshold $\tau_u$, it indicates that $S_m$ is probably an under-segment containing different objects, and $S_m$ will be filtered out." ($\tau_u$=0.02.)
- Feature fusion (IFA, §3.4, Eq. 1): a softmax-weighted average of the per-view features. Weight $a_{i,j}=\cos(f_{i,j},\bar f_i)-\sum_{O_k\in N_i}\cos(f_{i,j},\bar f_k)$. "a feature gets a larger weight if it is closer to its own cluster center and farther from neighboring instances". Neighbours are defined by $\cos\ge\tau_d$=0.7.
- There is no per-view confidence and no viewing-angle weighting. View disagreement is handled only by the distinctiveness weighting above.

### A3. Hierarchy
- **No semantic hierarchy.** There is one level of semantics: the object.
- The octree hierarchy is purely geometric. Eq. 2: node size $d_l=(b_{max}-b_{min})/2^l$, built per instance from its bounding box. A non-cubic bounding box gives non-cubic cells ("varying voxels that adaptively adjust their sizes and shapes"). $L_{max}$=4 "considering the dimensions of indoor objects" (§4.1).
- The design actively removes nested segments. A segment that contains semantically different sub-segments is treated as an error and deleted (A2). A "part-of" parent would be discarded by this rule.
- Related work points to hierarchy elsewhere: "HOV-SG [44] proposes a hierarchical 3D scene graph to enable scene representation of different granularities" (§2). "OctreeOcc [26] ... use[s] the octree structure to store semantic class" (§2).

### A4. Queries
- `Query(target)`: "directly locating an object by comparing the similarity between queries and stored semantics" (§3.5).
- `Query(reference, relation, target)`: locate the reference, then the matching edge, then the target.
- LLM decomposition for more complex queries. Occupancy queries for A*/JPS path planning.
- Granularity: object only.

### A5. Limits and assumptions
- Input: "posed RGB-D" sequence (Fig. 2), with poses taken as given. Projection to points needs depth.
- Scale: indoor rooms (Replica, ScanNet, ScanNet200, HM3DSem). The parameters are tuned for that ($L_{max}$=4, $I$=200 frames).
- Training-free: "we devise a training-free pipeline" (§1). It uses frozen CropFormer, CLIP/OVSeg and TAP.
- Open vocabulary through CLIP and caption features.
- Accuracy is modest even at object level. Zero-shot ScanNet200 AP is 14.3 (Table 2). Replica mIoU is 0.320 with OVSeg (Table 1).
- Occupancy fidelity is low. mEOR is 0.0108 on Replica for the adaptive-octree vs 0.0057 for the plain octree (Table 8). EOR is measured against the point cloud dilated by 0.005.

### A6. Transfer to our case — MY READING
- **Directly reusable:**
  - The **node record** shape: semantics + centre + geometry + typed edges. It is one structure that holds both semantics and geometry.
  - The **$\phi^{ior}$ containment ratio** is a geometric "is-inside" test. For us it could serve as a parent/child cue: sticker inside face. The paper uses it the opposite way (to delete parents).
  - The IFA idea of weighting views by distinctiveness is training-free and simple.
- **Breaks:**
  - Semantics live only at object level. Our tree needs face/sticker/line levels.
  - The under-segment filter would delete a "face" segment that contains several distinct "sticker" segments. That is exactly the parent we want to keep.
  - The IoU/IoR merging relies on 3D overlap of back-projected points. With a ~10 mm per-view pose offset and ~5 mm labels, two views of the same label may barely overlap, or not at all. The semantic terms alone would then carry the merge. Two identical-looking stickers would also merge wrongly.
  - CLIP/caption features are a weak fit for reading text at 5 mm. Our readings stay in images, so here the feature would only be an index, not the reading.
  - The octree is per instance and occupancy-only. It carries no label per cell, so it cannot localise a sub-part inside a node.

---

## B. OmniSeg3D (Ying et al. 2023)

### B1. Where the label lives
- The label lives at **every 3D point of a NeRF-style field**, as a learned **class-agnostic "segmentation identity feature"**.
- §3.2: "for each point $x_i$ ... we define a segmentation identity feature $f_i\in\mathbb{R}^D$". An MLP outputs $(\sigma_i,f_i)$ from position only (Eq. 2). Features are rendered like colour: $f(r)=\sum T_i\alpha_i f_i$ (Eq. 4).
- $D$=16. "the average mIoU first increases with $D$, then nearly saturates beyond $D=16$" (Table 4). It is stored in "the semantic feature volume $V_s$" plus the MLP (§3.3).
- No class id, no text, no instance id. Features are pushed onto the unit sphere by $L_{norm}$ (Eq. 9).
- The paper claims the field is representation-agnostic. It "can be formulated on point cloud, mesh, NeRF" (§1), and "a lightweight plug-in which can be easily integrated into reconstruction methods based on ... NeRF, mesh, and point cloud" (§3.3). The experiments train a NeRF. The mesh GUI case uses region growing on the mesh (§3.4).

### B2. Lifting from 2D and fusing across views
- Per image (§3.1): SAM with a 32×32 point grid gives a set of masks $M_{segs}$. Overlapping masks are **not** flattened. The paper says "directly overlapping masks, as done in SAM, may destroy the rich hierarchical information".
- The image is split into **patches**: "the smallest collection of pixels that share the same one-hot vector" of mask membership. This gives a patch index map $I_p$. "$N_p$ typically ranges from 200 to 500".
- Correlation by voting (Eq. 1): $C_{hi}(p_i,p_j)=\sum_k \mathbb{1}(p_i\subseteq m_k)\mathbb{1}(p_j\subseteq m_k)$, the number of masks that contain both patches.
- Per-anchor tree: "sort the patches according to the vote counts in vector $v_i$ and construct a hierarchical tree for anchor patch $p_i$ ... Patches located at shallower depths in the tree has stronger relevance".
- Fusion across views happens through **optimisation, not an update rule**. The contrastive loss is applied **per image** to features rendered from the shared 3D field. The 3D field must satisfy all images at once. The paper says: "We hypothesize that the 3D contrastive learning implicitly aggregates and averages the voting-based correlations from multi-view inputs" (§4.1).
- There is no explicit confidence and no explicit conflict rule. Inconsistent 2D masks are "averaged" by the field.

### B3. Hierarchy
- **Levels are not stored as labels. They are encoded as feature distance.** Points that are "closer" in the per-image patch trees must end up with more similar features.
- Loss (Eq. 6): $L_H=\sum_i\sum_{d}\frac{\lambda^{d-1}}{NL}\sum_j\sum_{s\in S^i_d}\max(L^{i,j}(s),L^{i,j}_{max}(d-1))$.
  - The $\lambda^{d-1}$ decay down-weights deeper (weaker) levels. $\lambda$=0.5 (§3.3).
  - The $\max(\cdot, L_{max}(d-1))$ term enforces **order**: "a patch higher in the hierarchy tree (smaller $d$) exhibits a higher feature similarity with the anchor patch than patches at lower levels".
  - Borrowed from Zhang et al. 2022 [50] (hierarchical multi-label contrastive).
- Parent vs child: a child is a **tighter similarity ball** around a point. A parent is a **looser ball** around the same point. There is no explicit parent object.
- Link to geometric resolution: none. The hierarchy is independent of field resolution.
- Trade-off (Table 3): with no hierarchy, Lv.1/Lv.2 = 93.1/80.4. At $\lambda$=0.5 it is 91.3/88.9. At $\lambda$=1 it is 89.2/89.2. Pushing coarse levels costs fine-level accuracy.
- Evaluated with **only two levels** ($M_{L1}\subset M_{L2}$, part vs instance; 281 annotated images, Replica, §4.1).

### B4. Queries
- Click → region at any level. "With a single click ... our model generates a score field based on feature similarities. By adjusting the binarization threshold, the segmentation can seamlessly traverse the scene hierarchy from atomic components to entire objects" (§3.4).
- **The level is a threshold**, not a named level. Eq. 11: $\exists\, th_i$ such that $M_{L_i}=\{p\,|\,score(p)>th_i\}$.
- Multi-click multi-object selection. Global discretisation by feature clustering (Fig. 6).
- **No text queries**. The features are class-agnostic. "text-aligned hierarchical segmentation may be a future direction" (§5).

### B5. Limits and assumptions
- "Given a set of calibrated input images" (§3). Poses are assumed accurate. Pose noise is not studied.
- **Per-scene training**: "We train our model for 50000 iterations for each scene", 8192 rays per batch (§3.3). It is trained once per scene and then reused for all instances (§4.2).
- Level ambiguity: "Due to the absence of a clear definition for hierarchy levels, there is no assurance that the objects will be segmented at the same level by simply clustering features" (§5).
- Cross-image blind spot: "since the contrastive learning is applied on single images, two objects that have never appeared in the same image may have similar semantic feature" (§5).
- Scene scale: indoor rooms (Replica) and NVOS/MVSeg forward-facing scenes.

### B6. Transfer to our case — MY READING
- **Directly reusable (training-free part):**
  - §3.1 runs per image with no training: SAM masks → disjoint patches → vote matrix $C_{hi}$ → containment tree.
  - This gives a 2D object > part > sub-part nesting per view. It could produce our face > sticker > line nesting **in each image**.
  - The patch definition (pixels with the same mask-membership vector) is a clean way to keep overlapping masks without flattening them.
- **Breaks:**
  - §3.2 (how the hierarchy lands in 3D) needs a per-scene optimised feature field (50k iterations). Anton wants no training.
  - The levels come out as unnamed similarity thresholds. The paper itself says the level is not consistent across objects. Our tree needs named, stable levels (face, sticker, line) that the VLM labels.
  - The features carry no text. Open-vocabulary names would have to be attached elsewhere.
  - Fusion is implicit averaging under "calibrated" poses. With a ~10 mm pose offset and ~5 mm labels, a 5 mm sticker seen in two views would be pulled to two places. The paper gives no mechanism for that.
  - The cross-image blind spot matters inward: two faces never seen in one image could get similar features.
- **Open question for Anton (not answered by the paper):** what is the 3D analogue of the patch/vote construction without a learned field? For example, voting on surface elements instead of rendered rays. The paper only does it via contrastive learning.

---

## C. OctLLM — Octrees as an Explicit 3D Language (Dan et al. 2026)

### C1. Where the label lives
- **No semantic labels in the 3D structure at all.** The octree stores occupancy only.
- §3.1: "pack every 8 consecutive binary occupancy bits into one byte-level mesh token whose value lies in [0,255]". The octree is serialised in Z-order.
- Each token is tied to a place and a scale. §3.2: "its three channels carry the 3D center coordinates of the node it encodes, rescaled per octree level". There is also "a learnable depth embedding that tells the model the scale at which a token should be interpreted".
- Semantics sit in the LLM's text output (captions), about the whole object. "the S-Octree encodes geometry alone, so OctLLM describes shape and part structure and says nothing about appearance" (§4.2).

### C2. Lifting from 2D and fusing across views
- None. The input is a mesh, voxelised to an octree: "100K surface points with face normals are used to build a depth-6 O-CNN octree" (App. B.1).
- Images enter only as a generation condition (image-to-3D) and as Blender renders for captioning: "four fixed views are captioned by Qwen3.5-9B" (§4.1).
- Captions are per object and become training targets. They are not projected onto the geometry.

### C3. Hierarchy
- Geometric only. It is a coarse-to-fine octree, depths 3 to 6 shown (Fig. 1). The finest grid is $64^3$ on an object "isotropically scaled to $[-1,1]^3$" (§4.1, App. B.1).
- Parent → child is plain octree subdivision. Each token's depth embedding means "the same spatial neighborhood can be read differently at coarse and fine levels" (§3.2).
- S-Octree drops fine detail on purpose: "randomly setting a fixed fraction of nodes at the second-to-last level to empty" (p=0.5), later restored by a 3D U-Net. "The LLM focuses on global 3D layout and nontrivial shape parts while a geometry network handles dense local completion" (§3.1).
- The text side describes parts ("two stacked wings joined by struts", §4.3). Those part words are **not bound** to octree nodes.

### C4. Queries
- 3D → text (describe a shape). Text/image → 3D (generate). That is all.
- No region queries, no click → label, no text → region. "OctLLM does not support 3D understanding of appearance or 3D editing" (App. E).

### C5. Limits and assumptions
- Heavy training. Qwen2.5-VL-7B is frozen, and 12 routed branches add "about 2.80B trainable parameters" (§4.1). Training ran "69K steps on 16 H20 GPUs".
- Resolution is $64^3$ per normalised object.
- Synthetic assets (Objaverse-XL, ShapeNet, ABO, HSSD), mesh input, geometry-only.
- Error compounding: "upstream occupancy errors can propagate or be amplified, producing holes" (App. E).

### C6. Transfer to our case — MY READING
- **Directly reusable as an idea, not as code:**
  - The **address of a region is (centre coordinate, depth)**. One octree node is at once a place and a granularity. This is a natural key for "a label lives at node X at level L".
  - The paper shows that an LLM can read such addressed tokens.
  - Z-order serialisation keeps spatial neighbours close in a sequence. That helps if an octree region is ever handed to a VLM/LLM as text.
- **Breaks:**
  - There are no semantics in the octree and no 2D → 3D lifting. It answers "how to give geometry to an LLM", not "how to colour geometry with labels".
  - It needs large-scale training. That is ruled out for us.
  - Resolution: $64^3$ over the object's extent. Cell size ≈ extent / 64. For example, ≈4.7 mm for a 300 mm object (my arithmetic, assumed size). That is about the size of our smallest labels, so a line of text would be 1–2 cells.
  - Geometry-only. Stickers and printed text are flat appearance on a face. They have no occupancy signature, so a pure occupancy octree cannot see them.

---

## D. Cross-paper summary

| Mechanism | Paper | Where the label lives | How levels relate | Fits us? (my reading) |
|---|---|---|---|---|
| Instance node with fused CLIP+caption feature; occupancy octree inside the node; relation edges | Octree-Graph §3.4–3.5 | Graph node (object); octree = occupancy only | Single semantic level; octree depth is geometric ($d_l=(b_{max}-b_{min})/2^l$) | Node record shape yes; one level only; overlap-based merging fragile under 10 mm offset |
| Containment ratio $\phi^{ior}=\|S_m\cap S_n\|/\|S_n\|$ | Octree-Graph §3.3 | Between 3D segments | Detects "n inside m". Used to **delete** parents with mixed content | Usable as a geometric part-of test if kept, not filtered |
| Distinctiveness-weighted view fusion (IFA) | Octree-Graph §3.4 | Per-instance feature | n/a | Training-free; fits per-node feature fusion |
| Disjoint patches + vote matrix $C_{hi}$ + per-anchor tree from SAM masks | OmniSeg3D §3.1 | Per image (2D patches) | Nesting by vote count (more shared masks = closer) | Training-free; gives 2D nesting per view; does not by itself reach 3D |
| Hierarchical contrastive feature field ($\lambda^{d-1}$ decay + order constraint) | OmniSeg3D §3.2 | Per 3D point, D=16 class-agnostic feature | Level = similarity threshold around a click; no named levels | Needs per-scene training; unnamed and inconsistent levels; no text |
| Coordinate + depth anchored octree tokens | OctLLM §3.1–3.2 | Octree node (occupancy byte), no semantics | Octree depth = scale; depth embedding | Addressing idea only; no labels; training-heavy; geometry-blind to flat stickers |

What none of the three provides: a single 3D structure where each cell carries **several named labels at once** (object > face > sticker > line). The closest pieces are listed below. They are not combined in any paper.
- Octree-Graph's node-with-semantics.
- OmniSeg3D's 2D containment tree.
- OctLLM's (coordinate, depth) addressing.

---

## E. Cited references worth fetching later (not fetched)

- **HOV-SG**, Werby et al. 2024, *Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation* (Octree-Graph [44]). An explicit multi-level open-vocabulary graph. Check how levels are built and how features fuse per level.
- **Clio**, Maggio et al. 2024, *Real-time task-driven open-set 3D scene graphs* (Octree-Graph [27]). Chooses granularity by task. Relevant to "which level does this detail live at".
- **OctreeOcc**, Lu et al. 2024, *multi-granularity occupancy prediction using octree queries* (Octree-Graph [26]). Semantic class stored in octree nodes at several granularities. This is the one paper here that puts labels on octree cells.
- **MaskClustering**, Yan et al. 2024, *view consensus based mask graph clustering* (Octree-Graph [48]). Multi-view consensus for merging 2D masks in 3D. Relevant to view disagreement.
- **ConceptGraphs**, Gu et al. 2023 (Octree-Graph [7]). The object-node + caption baseline that Octree-Graph builds on.
- **Zhang et al. 2022**, *Use all the labels: A hierarchical multi-label contrastive learning framework* (OmniSeg3D [50]). The source of the hierarchy loss.
- **Contrastive Lift**, Bhalgat et al. 2023, and **Panoptic Lifting**, Siddiqui et al. 2023 (OmniSeg3D [2], [39]). Alternatives for lifting inconsistent 2D instance ids into 3D.
- **SA3D**, Cen et al. 2023 (OmniSeg3D [5]). Propagates one SAM mask across views into 3D, training-light per object.
- **Part-X-MLLM**, Wang et al. 2025, *Part-aware 3D multimodal LLM* (OctLLM ref). Part-level 3D–language binding, the piece OctLLM lacks.
- **OctoMap**, Hornung et al. 2013 (Octree-Graph [11]). Probabilistic per-node update rule. Possible template for a per-node label distribution.
