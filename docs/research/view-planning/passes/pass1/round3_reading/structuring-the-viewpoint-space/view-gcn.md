# Wei, Yu, Sun — "View-GCN: View-based Graph Convolutional Network for 3D Shape Analysis"

CVPR 2020, pp. 1850–1859. Full text obtained (10-page PDF, CVF Open Access,
`openaccess.thecvf.com/content_CVPR_2020/papers/Wei_View-GCN_...pdf`), read in
full. **No arXiv preprint exists** for this paper — CVF/IEEE Xplore is the only
version; there is no separate HTML full text either, only the CVF abstract page.
Checked the code repo (`github.com/weixmath/view-GCN`) for implementation
details beyond the paper text; the fetched README gave no additional detail
(no license stated, no camera-coordinate generation code surfaced by the
fetch), so all claims below are from the paper text only.

**Framing note up front, because it changes how every branch question should be
read against this paper: this is a recognition/retrieval paper, not an action-
space or view-planning paper.** The "view-graph" exists solely to aggregate
a *fixed, pre-supplied* set of multi-view images into a single global shape
descriptor for classification (`ModelNet40`) and retrieval (`ShapeNet Core55`).
There is no sequential decision process, no next-best-view step, no notion of
an agent standing at node *i* and moving to node *j*. All N views are consumed
simultaneously in one forward pass; the output is one class label (Fig. 1,
Fig. 2, p.1850/1852). This is the single biggest transfer caveat and is
threaded through every section below.

## Core method

**View-graph construction (Sec. 4, p.1852).** A directed graph *G* with the
*i*-th node = the *i*-th view, carrying explicit 3D camera coordinates *v_i*.
Edge weight (learned, soft):

- `S_ij = Φ(g_ij; θ_s)` (Eq. 1), where `g_ij = [v_i, v_j, v_i − v_j, ‖v_i − v_j‖_2] ∈ R^10`
  is the concatenated relative-pose descriptor of the two cameras, and Φ is a
  3-layer MLP (LeakyReLU, 10 hidden units in the first two layers) outputting
  a scalar.
- Sparsification by k-nearest-neighbour on camera-coordinate Euclidean distance:
  `A_ij = S_ij · 𝟙{v_j ∈ N(v_i)}` (Eq. 2). Only kNN edges are kept; k = 2 for
  the 12-view circular configuration, k = 3 for the 20-view dodecahedral
  configuration (Sec. 5.5, p.1854).
- Camera positions are **fixed a priori by the chosen configuration, not
  derived from the individual object's geometry**: 12-view circular (cameras
  on a circle, 30° elevation) or 20-view dodecahedral (cameras at
  dodecahedron vertices), and these exact same positions are used for every
  object in the dataset (Fig. 4a/b, p.1853; Sec. 5.5). A third, "irregular"
  configuration is also tested: 12 views randomly subsampled from the
  dodecahedral 20 and each coordinate randomly perturbed (Fig. 4c) — still a
  single fixed-per-experiment set of positions applied uniformly, not a
  per-object adaptive placement.

**View-GCN architecture (Sec. 5, Fig. 5, p.1852–1854).** A hierarchy of
progressively **coarsened** (not refined) view-graphs `G^0 → G^1 → ... → G^{L−1}`
(20→10→5 nodes, or 12→6→3). At each level:
1. **Local graph convolution** (Eq. 3): diffuse node features through the
   learned adjacency `A^l`, then a per-node linear transform `W^l`, then
   BatchNorm+LeakyReLU.
2. **Non-local message passing** (Eqs. 4–5): a 3-layer-MLP relation function
   Γ computes a message for *every* node pair (not just kNN neighbours),
   messages are summed per target node and fused with the node's own feature
   by a 1-layer MLP Ω — this is what lets far-apart views (e.g. the
   symmetric/opposite view of a chair, Fig. 3b) influence each other.
3. **Selective view-sampling for graph coarsening** (Sec. 5.4, Eq. 6): start
   from a Farthest-Point-Sampling (FPS) subset (diverse but not necessarily
   discriminative), then for each FPS seed replace it with whichever of its
   kNN neighbours gets the highest score from a learned per-position
   "view-selector" `V` (a 2-layer MLP outputting per-shape-class
   probabilities). Sampling rate `s = 0.5` throughout; `N_{l+1} = round(N_l·s)`.

Node features at every level are max-pooled (Eq. 7) and **all levels'
pooled features are concatenated** into the final global descriptor
(Sec. 5.5, p.1854) — nothing from the fine level is discarded once coarsened,
unlike a strict coarse-to-fine search that abandons rejected branches.

**Training (Sec. 5.6, p.1854–1855).** Two-stage: (1) fine-tune a 2D backbone
(ResNet-18/AlexNet/ResNet-50, ImageNet-pretrained) on shuffled multi-view
images for classification to get per-view features `f_i^0`; (2) end-to-end
fine-tune backbone + view-GCN with a combined loss `L = L_shape(C(F), y) +
Σ_l Σ_j Σ_{v_q∈N(v_j)} L_view(V(f̂_{v_q}^l; θ_v^{l,j}), y)` (Eq. 8) — a global
shape cross-entropy plus a per-view-selector cross-entropy that pushes each
selector to be individually discriminative of shape class within its local
neighbourhood. **73.4M trainable parameters for the 20-view model** (44.8M
backbone + 28.6M view-GCN); trained with SGD, batch of 20 shapes × 20 views
= 400 views/batch (or 32×12=384 for the 12-view model), 30 epochs for stage
1, 15 for stage 2, cosine-decayed LR with 1-epoch warmup (p.1855).

## Concrete results and numbers

- **ModelNet40 classification** (Table 1, p.1855): View-GCN (ResNet-18)
  **96.5% per-class / 97.6% per-instance**, vs. MVCNN 90.1/90.1, MVCNN-new
  92.4/95.0, GVCNN 90.7/93.1, RCPCNN –/93.8, 3D2SeqViews 91.5/93.4, VERAM
  92.1/93.7, EMV 92.6/94.7, RN 92.3/94.3. Beats best prior view-pooling
  method by >3.4 pt / >2.6 pt.
- **vs. RotationNet, same backbone, 20 views** (Table 2, p.1855): AlexNet
  97.2 vs. 96.4; ResNet-50 97.3 vs. 96.9 (per-instance). Margins are small
  (0.4–0.8 pt) once backbone is controlled for.
- **ShapeNet Core55 retrieval** (Table 3, p.1856), micro-averaged: P@N 81.8,
  R@N 80.9, F1@N 80.6, **mAP 78.4**, NDCG 85.2; macro-averaged mAP 60.2.
  Beats RotationNet (mAP 77.2 micro / 58.3 macro) on every metric except
  micro-NDCG (RotationNet 86.5 vs. 85.2).
- **RGBD real-sensor dataset** (Table 4, p.1856), 12 views/object, 45°
  elevation, 10-fold CV: View-GCN (ResNet-18) **94.3%** per-instance, vs.
  RotationNet(AlexNet) 89.3%, MVCNN(AlexNet) 86.1%. View-GCN(AlexNet) 91.9%
  — 5.8 pt over MVCNN(AlexNet), 2.6 pt over RotationNet(AlexNet) with the
  same backbone.
- **Training cost**: 3 h (stage-1 backbone fine-tune) + 6 h (stage-2
  end-to-end) on ModelNet40, single GTX 1080 Ti (Sec. 6.1, p.1855). This is
  network-training cost, not per-object graph-construction cost — the
  view-graph itself (Eqs. 1–2, a tiny 3-layer/10-hidden-unit MLP over camera
  coordinates) is a cheap forward pass computed once per fixed configuration,
  reused unchanged across every object.
- **Ablations, ShapeNet Core55** (Table 5, p.1857; all ResNet-18 backbone):
  - Baseline (MVCNN-new max-pool): 76.7 / 88.9.
  - Full view-GCN: **79.8 / 90.9**.
  - w/o local graph conv: 78.8 / 90.6 (−1.0 / −0.3).
  - w/o non-local message passing: 77.7 / 90.5 (−2.1 / −0.4) — the larger
    single-component effect.
  - Selective view-sampling vs. plain FPS: 79.8/90.9 vs. 78.2/90.3
    (+1.6/+0.6 pt); on unaligned ModelNet40, +0.9/+0.8 pt.
  - Learned adjacency (Eq. 1) vs. binary kNN indicator (view-GCN-A1,
    `A_ij=𝟙{v_j∈N(v_i)}`) vs. Gaussian-kernel kNN (view-GCN-A2,
    `A_ij = e^{-‖v_i-v_j‖_2}·𝟙{...}`): learned wins by (0.6, 0.2) and
    (0.7, 0.4) pt respectively — a small, consistent but not large margin.
    **A simple distance-thresholded kNN adjacency gets ~90% of the benefit
    of the learned one.**
  - Hierarchy depth: 1-level (L1, 20 nodes only) 78.5/89.9 → 2-level (L2,
    20+10) 78.6/90.6 → full 3-level (20+10+5) 79.8/90.9. Deeper still
    (s=0.6→4 levels [20,12,7,4]; s=0.7→5 levels [20,14,10,7,5]) is *worse*:
    −0.1/−0.2 and −0.3/−0.3 pt vs. the 3-level default — **diminishing,
    then negative, returns from more hierarchy levels.**
  - View-selector supervision (`L_view`) ablation: removing it costs only
    0.1/0.2 pt — the selectors learn useful positions almost entirely from
    the global shape-loss gradient; explicit per-view supervision is nearly
    redundant.
  - Irregular configuration (Fig. 4c, perturbed 12-of-20 subsample):
    85.3% per-class / 89.5% per-instance, 4.2/1.9 pt above MVCNN-new on the
    same configuration (Sec. 6.4, p.1857) — accuracy doesn't collapse when
    camera placement is irregular/perturbed, evidence the kNN-graph
    mechanism tolerates non-uniform view layouts.

## Load-bearing assumptions

- **Every object in a given experiment shares the identical set of camera
  positions.** The "flexible view configurations" claim (title, abstract,
  Sec. 1) means the *architecture* can be instantiated over different
  topologies (circular / dodecahedral / irregular-perturbed) chosen once by
  the practitioner — it does **not** mean camera positions are adapted
  per-object to that object's own geometry. Confirmed directly: Sec. 5.5
  states the 12- and 20-view configurations as fixed geometric layouts with
  no object-dependence in how `v_i` is chosen.
- **Objects are pre-aligned to a canonical frame by the dataset, not by the
  method.** ShapeNet Core55 evaluation explicitly uses "the 'normal' dataset,
  i.e., all of the shapes are consistently aligned and normalized to a unit
  length cube" (Sec. 6, p.1855). ModelNet40 is likewise a pre-aligned CAD
  benchmark. Cross-shape comparability of node *i* (`"view 7 of a chair"`
  vs. `"view 7 of a cup"`) is therefore inherited entirely from external
  dataset curation — the paper supplies no canonicalization machinery of
  its own.
- **Node identity = explicit metric camera coordinate, known at construction
  time**, not a purely relational/graph-defined identity. Sec. 4: "We build
  a directed graph *G* with *i*-th node as the *i*-th view **with camera
  coordinates v_i**." Pose is the primitive datum from the start; the graph
  (adjacency `A`) is a *derived*, learned structure computed from those
  coordinates for feature-aggregation purposes, not the thing that defines
  what the address means.
- **Whole-set input, not sequential/incremental.** All N views are always
  available and processed together (Sec. 5.1, "Given N views {I_i}, features
  {f_i^0} are extracted..."); the architecture has no mechanism for
  operating on a partial or growing view set, and no cost model for
  acquiring views one at a time.
- **Adjacency's job is feature aggregation for a recognition loss, not
  semantic naming.** Nothing in the graph construction (Eqs. 1–2) or the
  ablations (Table 5, view-GCN-A1/A2) attaches any interpretable label
  (e.g. "opposite," "adjacent face") to an edge — kNN edges are just
  "the k nearest cameras by Euclidean coordinate distance," with a learned
  scalar weight on top.

## What transfers to our setting

- **A concrete, computable answer to "how is adjacency defined when there is
  no {h,v}±1": k-NN on camera-coordinate Euclidean distance, optionally
  weighted by a small MLP over `[v_i, v_j, v_i−v_j, ‖v_i−v_j‖]` (Eqs. 1–2).**
  This is directly liftable: given any candidate address set with associated
  3D camera poses (regardless of whether those poses came from a uniform
  sphere or a primitive-conditioned scheme), `k`-NN in R³ is computable with
  `scipy.spatial.cKDTree` or `sklearn.NearestNeighbors` in closed form, no
  training required to get the ablation-tested "80–90%-as-good" version
  (view-GCN-A1/A2, Table 5). This answers the branch's "modern engineering
  answer to defining adjacency" framing literally.
- **The ablation that learned-vs-simple adjacency only differs by 0.2–0.7 pt
  (Table 5) is evidence that a hand-specified kNN/threshold rule is a
  reasonable stand-in for a learned affinity** — relevant to the
  primitive-level deterministic-rule track this project has already
  committed to (box-face-by-face, cylinder-orbit), since it suggests we
  don't need to learn adjacency weights to get most of the achievable
  benefit from *a* notion of neighbouring views.
- **Empirical tolerance of irregular/perturbed camera layouts** (Sec. 6.4
  irregular-configuration result, 85.3/89.5%, only ~2–4 pt below the
  clean-dodecahedral numbers and still above MVCNN-new) is evidence — on a
  recognition task, not ours, but still evidence — that kNN-graph-based
  adjacency degrades gracefully rather than catastrophically when view
  positions are non-uniform. That is a relevant existence proof for the
  branch's RQ6 ("does the agent need adjacency at all... uniform sphere vs.
  geometry-conditioned"): a coordinate-proximity adjacency computed post hoc
  over *whatever* addresses a geometry-conditioned scheme produces is at
  least not obviously broken by non-uniformity.
- **The hierarchy-depth ablation (Table 5: 1→2→3 levels improves, 3→4→5
  degrades) is a concrete, checkable data point for "does hierarchy pay for
  itself," relevant to RQ10/RQ11 tractability arguments** — more structure
  is not free even in a domain (ModelNet40/ShapeNet, thousands of training
  shapes, full gradient supervision) far more favourable to hierarchy than
  ours; for our ~6–20-node primitive-level address sets, this is a soft
  argument against building deep multi-level hierarchies at all.
- **Directional confirmation of the aspect-graph-panel finding (companion
  read, `faugeras-why-aspect-graphs-not-practical.md`) that the "indexing
  problem" is scoped to the inverse direction.** View-GCN's whole graph
  machinery exists to solve image-set → label (classification), i.e.
  exactly the inverse/recognition direction Faugeras worried about — and it
  is expensive to build (73.4M-parameter learned network, hours of GPU
  training) precisely *because* it targets that direction. This is a second,
  independent piece of evidence (different paper, different decade) that
  cheap adjacency (kNN) is available for the *forward* direction our
  project needs (label → pose) without needing anything like view-GCN's
  learned apparatus, which is solving a harder, different problem.

## What does NOT transfer, and why (negative findings)

- **This is not an action space and view-GCN offers no view-selection,
  next-best-view, or planning mechanism at all.** All N views are consumed
  in one batch forward pass to emit a single class label (Fig. 1, Fig. 2,
  Sec. 3.2, Sec. 5.1). There is no concept of an agent occupying a node and
  choosing an edge to traverse, no cost or budget over views, no incremental
  acquisition. This paper cannot answer the branch's central RQ ("does the
  agent need adjacency at all... can it navigate {h,v} by incrementing
  bins") because it never poses a navigation problem — the "graph" adjacency
  exists only to let non-local and local feature-passing operations know
  which node's feature should influence which other node's feature during
  offline descriptor pooling.
- **Camera/view positions are identical across every object in a dataset,
  not derived from individual object geometry** (Sec. 5.5, Fig. 4a/b). This
  directly negates reading view-GCN as a "geometry-conditioned view space"
  precedent: it is a uniform, fixed-position multi-view scheme (three flavors:
  circular, dodecahedral, irregular-but-still-fixed-per-experiment) evaluated
  against *other uniform, fixed-position schemes*, never against a
  per-object, geometry-adaptive placement. This is a real instance of the
  branch's RQ ("does any found work compare a geometry-conditioned view
  space against a uniform sphere on the same task with a metric") — **the
  answer for this paper is explicitly no**: every comparison in Tables 1–5
  is uniform-sphere-vs-uniform-sphere (different topologies/backbones), and
  it reinforces rather than fills the identified literature gap.
- **Node addresses are not purely relational, despite the branch's own
  candidate classification listing "view-GCN" under "(c) a node ID whose
  meaning is purely relational."** This needs correcting from the primary
  text: node `i` carries an explicit metric camera coordinate `v_i` from
  construction (Sec. 4), and the graph/adjacency is a *derived*, learned
  structure computed from those coordinates for a downstream recognition
  loss — closer to "(a) index into a fixed enumeration, with coordinates
  attached" than to a graph whose *meaning* is defined purely by its edges
  (which is closer to how a classical aspect graph's visual-event topology
  works). Any synthesis citing view-GCN as the "(c)" exemplar should flag
  this correction.
- **Cross-shape addressing consistency is not solved by the method — it is
  outsourced to dataset curation (canonical pre-alignment).** ShapeNet
  Core55's "normal" split ("all shapes are consistently aligned and
  normalized to a unit length cube," Sec. 6) is what makes "node 7" mean
  roughly the same relative direction for a chair and a cup, not anything
  in the view-graph or view-GCN architecture. This paper therefore gives
  **no evidence at all** on the branch's cross-shape naming-consistency RQ
  (RQ12/RQ19) beyond the negative point that it assumes the problem away.
  Confirms, from a second independent angle, the branch's suspected
  RQ19 finding that cross-shape *view*-addressing (as opposed to cross-shape
  *part*-addressing) has no real precedent to lean on.
- **No seam treatment.** With 20-vertex-dodecahedron or 12-point-circle
  configurations there is no analogue of a box's edges/corners or a
  primitive-decomposition boundary — nothing in the paper discusses
  degenerate/boundary addresses, so it is silent on the branch's RQ7 (seams).
- **No ordering semantics, and this is by design, not oversight.** Unlike
  RotationNet, which needs "assumption of homogeneous space (e.g.,
  icosahedron)" and treats the view index as a latent variable to search
  over consistent rotations (Sec. 2.1, p.1851; Sec. 6.1 comparison, Table 2),
  view-GCN explicitly avoids requiring any canonical/consistent index
  ordering across shapes — it substitutes coordinate-space kNN for
  order-based neighbourhood. Useful as a documented example of the
  "no-ordering" alternative for RQ4, but the paper offers no evidence either
  way on whether *ordering itself* (next/previous/opposite) is useful for
  an *agent*, since — again — there is no agent here.
- **73.4M-parameter learned system, hours of GPU training on tens of
  thousands of labeled 3D shapes (ModelNet40: 12,311 shapes/40 classes;
  ShapeNet Core55: 51,162 shapes/55 classes) is not a lightweight,
  liftable component.** For the branch's "what is liftable" question: the
  *recipe* (kNN over camera coordinates, Eqs. 1–2) is liftable in an
  afternoon with `scipy`/`sklearn`; the *trained network* (backbone +
  local/non-local graph conv + selective view-sampling + view-selectors) is
  not liftable at all to a single-object, no-category-label, primitive-shape
  setting like ours — there is no classification target to train it against.
- **No mesh/point-cloud geometry is consumed anywhere** — inputs are
  rendered 2D images from pre-chosen camera poses (Sec. 5.1, "features
  extracted by a fine-tuned 2D image classification network... e.g.,
  ResNet-18"); the method never touches the underlying 3D surface (mesh or
  point cloud) except to render the fixed views. It therefore gives zero
  evidence about the branch's RQ2 ("does it require CAD features or
  tolerate primitives fitted to a noisy cloud") — the "geometry" input
  question doesn't arise because geometry is only ever a rendering source,
  not an addressing input.

## Citations (for checkability)

- Abstract, method summary: p.1850.
- Fig. 1 (illustration, whole-set-in/label-out pipeline): p.1850.
- Related work, RotationNet/EMV homogeneous-space dependence: Sec. 2.1,
  p.1851.
- Motivation (Fig. 3, neighbouring/symmetric views): Sec. 3.1, p.1852.
- View-graph definition, `S_ij` (Eq. 1), kNN sparsification `A_ij` (Eq. 2):
  Sec. 4, p.1852.
- Fig. 4 (circular/dodecahedral/irregular configurations): p.1853.
- Local graph convolution (Eq. 3): Sec. 5.2, p.1853.
- Non-local message passing (Eqs. 4–5): Sec. 5.3, p.1853.
- Selective view-sampling, FPS + view-selector (Eq. 6): Sec. 5.4, p.1853–1854.
- Fig. 5 (one level of view-GCN, selective-sampling detail): p.1854.
- Global feature fusion across levels (Eq. 7), 12-/20-view configuration
  specs, k=2/k=3: Sec. 5.5, p.1854.
- Training loss (Eq. 8), 73.4M parameters (44.8M backbone + 28.6M
  view-GCN): Sec. 5.6, p.1854–1855.
- Dataset descriptions (ModelNet40, ShapeNet Core55 "normal"/aligned split,
  RGBD): Sec. 6, p.1855.
- ModelNet40 classification results, training time (3h+6h, GTX 1080 Ti):
  Table 1, Sec. 6.1, p.1855.
- RotationNet same-backbone comparison: Table 2, p.1855.
- ShapeNet Core55 retrieval: Table 3, Sec. 6.2, p.1856.
- RGBD real-sensor classification: Table 4, Sec. 6.3, p.1856.
- Ablations (LGC/NLMP removal, FPS vs. selective sampling, adjacency
  variants A1/A2, hierarchy depth L1/L2/full/deeper, view-loss removal,
  irregular configuration): Table 5, Sec. 6.4, p.1857.
- Conclusion, future work (multi-modality fusion, not action spaces):
  Sec. 7, p.1857.

## Assessment for the branch

View-GCN answers exactly one of the branch's questions cleanly and in closed
form: **the "modern engineering answer to adjacency without {h,v}±1" is
k-nearest-neighbour on camera-coordinate Euclidean distance** (Eqs. 1–2),
and the paper's own ablation shows a hand-specified (untrained) version of
this rule captures most of the benefit of the learned version (Table 5,
A1/A2 within 0.2–0.7 pt of the full model) — this is directly liftable to
our PLY-derived candidate view poses with off-the-shelf nearest-neighbour
code, no training required. Everything else about the paper is a scope
mismatch that should be stated plainly rather than stretched: this is a
whole-set-in/label-out recognition-and-retrieval architecture with **no**
sequential decision process, **no** per-object-geometry-adaptive view
placement (all objects in an experiment share identical camera positions,
Sec. 5.5), and **no** solved cross-shape addressing-consistency problem
(comparability across shapes comes from dataset pre-alignment, not from the
method, Sec. 6). It also corrects the branch's own candidate classification:
view-GCN's node identity is an explicit metric coordinate attached at
construction, not a "purely relational" node ID — the graph is a derived,
learned, task-specific structure layered on top of already-meaningful
addresses, not the source of their meaning. Net effect for RQ3/RQ4/RQ5: a
clean, cheap, precedented computational recipe for one narrow sub-question
(neighbour lookup for a scattered address set), wrapped inside a paper whose
central contribution (learned hierarchical feature aggregation for shape
recognition) doesn't touch view-planning, action-addressing, or
geometry-conditioned view placement at all.
