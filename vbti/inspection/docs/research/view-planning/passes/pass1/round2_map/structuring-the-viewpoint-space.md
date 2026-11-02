# Branch 5 — `structuring-the-viewpoint-space`
## Composing and addressing geometry-conditioned view spaces

**Date:** 2026-08-27
**Role:** literature mapper (judgment tier), pass 1 round 2.
**Branch plan:** [`../round1_plan.md`](../round1_plan.md) § Branch 5
**Contract:** [`../../../../geometry-aware-view-planning.md`](../../../../geometry-aware-view-planning.md)
**Feeds:** RQ3 (primary), RQ4 (secondary); constrains Branch 4 from downstream. Priority P2.

---

## What this branch is actually asking

Not "which views are good" (Branch 1), not "which set of views to take"
(Branch 4), not "how to phrase it for the model" (Branch 6). This branch owns
one question with three parts:

1. **Addressing** — what is the *name* of a view? Today it is `(h, v)`, an
   index into a uniform bin grid. If views become geometry-conditioned, what
   replaces that index, and is it still a name a blind orchestrator can say?
2. **Relating** — what is the *neighbour* of a view? The `{h,v}` grid gives
   adjacency for free (h±1, v±1). Geometry-conditioned views have no such
   built-in topology; someone has to define one, or prove the agent does not
   need one.
3. **Consistency across shapes** — the same convention must survive a box, a
   cylinder, and a two-primitive assembly. Per-object naming is the ADAM
   regression; per-primitive naming is contract-blessed.

The literature that has actually done this is older than the field remembers:
**aspect graphs** are the only formal tradition that partitioned viewpoint
space *by object geometry* and gave the partition an adjacency structure. They
were abandoned. The most valuable thing this branch can produce is a precise
account of **why** — because the plan's standing critic question for this
branch is "why did aspect graphs die, and are we rebuilding them?", and the
honest answer determines whether RQ3 is a solved-and-discarded problem or an
open one.

Ranked list below is 27 documents. The 10 marked **high** are the mandatory
spine and alone satisfy the plan's 12–20 target if capacity is tight; med/low
are for the second reading wave.

---

## Document map

### Group A — Aspect graphs: the structure, its complexity, and its death
The direct formal precedent for "views partitioned by object geometry", plus
the negative evidence RQ4's tractability requirement needs.

| # | Slug | Document | Type | Pri |
|---|------|----------|------|-----|
| 1 | `faugeras-why-aspect-graphs-not-practical` | Faugeras, Mundy, Ahuja, Dyer, Pentland, Jain, Ikeuchi, Bowyer — *Why Aspect Graphs Are Not (Yet) Practical for Computer Vision*, CVGIP:IU 55(2):212–218, 1992. [PDF](https://www.ri.cmu.edu/pub_files/pub3/faugeras_olivier_1992_1/faugeras_olivier_1992_1.pdf) | paper | **high** |

The obituary, written as an adversarial panel with the field's own architects
responding. Read for the *taxonomy of objections* (Dyer splits them into
definitional vs. usage/indexing), and for Ikeuchi's concession that indexing —
"given this image, which node am I in?" — was never solved. Critical for us:
our use is **addressing an action**, not indexing an observation, so the
usage objection may not transfer while the definitional one does. This is the
single document that decides whether RQ3 is rebuilding a known failure.
Constraint tag: pre-noise-era, all polyhedral/CAD assumptions. Critic ammo.

| 2 | `gigus-canny-seidel-aspect-graph-complexity` | Gigus, Canny, Seidel — *Visibility, Occlusion, and the Aspect Graph*, IJCV. [Springer](https://link.springer.com/article/10.1007/BF00054919) | paper | **high** |

The numbers behind the explosion: viewpoint-space-partition and aspect-graph
size bounds Θ(n²)/Θ(n⁶) under orthographic and Θ(n³)/Θ(n⁹) under perspective
projection, plus the "asp" representation. Extraction target: **evaluate these
bounds for our objects.** A box is n=6 faces; a cylinder is not polyhedral at
all. If a primitive-level aspect graph is 30 nodes rather than 30,000, the
combinatorial-explosion argument does not apply to us and the critic loses its
strongest card — that arithmetic must be done, not asserted.

| 3 | `bowyer-dyer-aspect-graph-survey` | Bowyer & Dyer — *Aspect Graphs: An Introduction and Survey of Recent Results*, Int. J. Imaging Systems & Technology 2(4), 1990. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.1850020407) | paper | **high** |

The seminal survey; defines viewpoint space partition, aspect, visual event,
and the VSP↔aspect-graph duality cleanly. Take the definitions verbatim — the
vocabulary of "the partition and its dual graph" is exactly the vocabulary
RQ3 lacks. One survey only; do not chase the 1988–1995 variant papers.

| 4 | `cyr-kimia-similarity-based-aspect-graph` | Cyr & Kimia — *A Similarity-Based Aspect-Graph Approach to 3D Object Recognition*, IJCV 57:5–22, 2004. [Springer](https://link.springer.com/article/10.1023/B:VISI.0000013088.59081.4c) | paper | **high** |

The reformulation that survived the critique, and structurally the closest
thing in the literature to what we want: **sample the view sphere densely (5°),
define a similarity metric between views, then region-grow the samples into
aspects.** No exact visual-event computation, no polyhedral assumption, hence
no explosion — the partition is a *segmentation* of a sampled sphere. That is
a directly implementable recipe on a noisy PLY if the similarity metric is
replaced by something geometric (shared visible surface, normal agreement).
Highest-value single method paper in this branch.

| 5 | `entity-based-aspect-graphs` | Shimshoni & Ponce (et al.) — *Entity-Based Aspect Graphs: Making Viewer Centered Representations More Efficient*, Pattern Recognition Letters, 1998. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167865598000063) | paper | med |

The other escape from the explosion: do not compute all aspects — compute only
the visibility structure of the *entities you care about*. A non-convex "step"
has 71 characteristic views; restricting to the entity of interest collapses
that. Maps one-to-one onto our situation, where the agent inspects **one
surface region at a time**, not the whole object. Read for the pruning
criterion and the reported size reductions.

| 6 | `seibert-waxman-learning-aspect-graphs` | Seibert & Waxman — *Learning Aspect Graph Representations from View Sequences*, NeurIPS 1989. [PDF](https://proceedings.neurips.cc/paper/1989/file/6da9003b743b65f4c0ccd295cc484e57-Paper.pdf) | paper | low |

Adjacency *learned* from observed view transitions instead of computed from
geometry. Cheap read; relevant only as an existence proof that the edge set of
a view graph can come from experience. Skip if time-pressed.

---

### Group B — Discretizing and addressing a sphere properly
The `{h,v}` grid is a naive spherical discretization. If part of the answer is
"keep a sphere but do it right", these are the schemes that already solved
equal-area, hierarchical refinement, and explicit neighbour lookup.

| 7 | `yershova-lavalle-hopf-so3-grids` | Yershova, Jain, LaValle, Mitchell — *Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration*, IJRR 29(7):801–812, 2010 (WAFR 2008). [PDF](https://lavalle.pl/papers/YerJaiLavMit10.pdf) | paper | **high** |

The rigorous answer to "how do I discretize a view space and still know what
the neighbours are". Four properties they claim explicitly: low metric
distortion on neighbour edges, optimal dispersion reduction per added sample,
**explicit neighbourhood structure**, equivolumetric cells. Uses HEALPix on the
S² base and the S¹ fiber for camera roll — i.e. it addresses *full camera
orientation*, not just position, which matters because our views are currently
all forced vertical. Robotics-native (motion planning), so the vocabulary
transfers. Extraction: the addressing scheme and the neighbour operator.

| 8 | `gorski-healpix` | Górski et al. — *HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere*, ApJ 622:759, 2005. [IOP](https://iopscience.iop.org/article/10.1086/427976/pdf) | paper | med |

The concrete addressing scheme: 12 base pixels, recursive quartering, N_pix =
12·N_side², equal area, and a **nested numbering where a child's ID is the
parent's ID plus two bits**. That is hierarchical addressing that a language
model can literally read off as a path. Mature implementation (`healpy`).
Read the addressing/indexing section only; skip the spherical-harmonics half.
Note: geometry-agnostic — this is the *good* version of our current baseline,
not the geometry-conditioned answer. Useful as an upgraded baseline.

| 9 | `h3-hierarchical-spatial-index` | Uber H3 — hexagonal hierarchical geospatial index, core library overview. [h3geo.org](https://h3geo.org/docs/core-library/overview/) | docs | low |

An addressing system that shipped. Read for the engineering realities of
hierarchical cell IDs: parent/child containment, `kRing` neighbour traversal,
uniform 6-neighbour adjacency, and the **12 pentagon defects** where the
neighbour count breaks — the practical lesson being that no sphere tiling has
uniform adjacency everywhere, and a real system just names the exceptions.
Cheap, concrete, and directly answers "what does a usable address look like".

| 10 | `multiview-3d-recognition-review` | *Deep Models for Multi-View 3D Object Recognition: A Review*, 2024. [arXiv:2404.15224](https://arxiv.org/abs/2404.15224) | paper | low |

One document that catalogues the de-facto viewpoint conventions the vision
community standardized on — 12-view azimuth ring at 30° elevation (MVCNN),
20-view dodecahedral vertices (RotationNet), spherical spreads (MVTN) — and
what breaks when you change them. Our `{h,v}` grid is literally the MVCNN ring
with 3 elevations. Read this instead of the individual method papers. Also
note RotationNet's ordered-view-sequence trick: **view order carries meaning**,
which is a free lesson for presenting an addressed set.

---

### Group C — Object-anchored view parameterizations
Coordinate systems in which "orthogonal to face 3" is a native address rather
than a nearest-neighbour search over a global grid. The core of RQ3.

| 11 | `magana-feature-based-constrained-spaces` | Magaña, Dirr, Bauer, Reinhart — *Viewpoint Generation Using Feature-Based Constrained Spaces for Robot Vision Systems*, Robotics 12(4):108, 2023. [arXiv:2306.06969](https://arxiv.org/abs/2306.06969) | paper | **high** |

The strongest object-anchored parameterization found. Each viewpoint constraint
(standoff, incidence angle, FOV, resolution) is formulated as a **feature-based
constrained space (C-space): the topological space of sensor placements that
acquire a given feature while satisfying that constraint**, with geometric,
deterministic, closed-form solutions, intersected across constraints. Verified
on a real robot with two sensors. This is exactly "per-face frame + the set of
poses that see this face acceptably", which is both the address *and* the
region an optimizer could climb inside. TUM group, industrial inspection —
same problem class as ours. Read in full; check whether the closed forms
assume CAD features or tolerate fitted primitives.

| 12 | `magana-feature-cluster-constrained-spaces` | Magaña et al. — *Viewpoint Planning for Range Sensors Using Feature Cluster Constrained Spaces for Robot Vision Systems*. [Semantic Scholar](https://www.semanticscholar.org/paper/f77289136a51cafd718600ec6e3c570a463cae46) | paper | med |

The multi-feature extension of #11 — how single-feature C-spaces compose when
one view must cover several regions, plus model-uncertainty handling. That
composition rule is the object→primitive→view hierarchy in concrete form.
Read only after #11.

| 13 | `mosbach-feature-driven-viewpoint-placement` | Mosbach, Gospodnetić, Rauhut, Hamann, Hagen — *Feature-Driven Viewpoint Placement for Model-Based Surface Inspection*, Machine Vision & Applications 32, 2020. [PDF](https://web.cs.ucdavis.edu/~hamann/MosbachGospodneticRauhutHamannHagenMachineVisionAndAppplicationsPaperRevision2AsPublished10292020.pdf) | paper | med |

Views parameterized by the **surface itself**: the model is approximated by
B-spline patches, geometric feature functionals drive adaptive non-uniform
sampling (sparse where flat, dense where complex), and viewpoint generation
becomes independent of mesh resolution. That is a surface-parameterized camera
manifold with a built-in level-of-detail — the natural home for "box face gets
one view, curved region gets several". Extraction: the parameterization and the
density functional, not the selection stage.

| 14 | `gospodnetic-viewpoint-placement-inspection-survey` | Gospodnetić et al. — *Viewpoint Placement for Inspection Planning*, Machine Vision & Applications 32, 2021. [Springer](https://link.springer.com/article/10.1007/s00138-021-01252-z) | paper | **high** |

The spine for Group C: the best modern survey of *how candidate view spaces
are constructed*, taxonomized as (a) space sampling independent of the model —
uniform/adaptive tessellated spheres, spherical codes, Fibonacci spheres; (b)
**dilated/offset surface sampling** — displace the surface by the standoff
distance and sample the shell, "the tessellated sphere but correlated with
model geometry"; (c) direct surface/patch sampling. That three-way split *is*
the RQ3 design space, already named by someone else. Read first in this group.
Cross-tag Branch 4 (it also covers selection).

| 15 | `scott-view-planning-survey` | Scott, Roth, Rivest — *View Planning for Automated Three-Dimensional Object Reconstruction and Inspection*, ACM Computing Surveys 35(1):64–96, 2003. [ACM](https://dl.acm.org/doi/10.1145/641865.641868) | paper | **high** |

The classical spine, and the one place where aspect graphs, the **visibility
matrix** (Tarbox & Gottschlich: the viewpoint × surface-patch bipartite
structure), and the art-gallery formulation are set side by side as competing
*representations of the view space*. The visibility matrix matters here even
though selection is Branch 4's: it is the structure that *relates* views to
geometry, and a row/column addressing over it is a candidate answer to RQ3.
Extraction target for this branch is the representation taxonomy only.

---

### Group D — View graphs and shape-adaptive view sets in modern vision
Where the aspect-graph idea actually survived: nodes-as-views graphs, and view
configurations regressed from the shape.

| 16 | `mvtn-multi-view-transformations` | Hamdi, Giancola, Ghanem — *MVTN: Learning Multi-View Transformations for 3D Understanding*, IJCV 2024 (ICCV 2021). [arXiv:2212.13462](https://arxiv.org/abs/2212.13462) | paper | med |

The clean existence proof for a **geometry-conditioned view set of fixed
cardinality**: a point encoder reads coarse shape features and regresses the
azimuth/elevation of N cameras, end-to-end through a differentiable renderer.
Cardinality stays constant while the *placement* adapts to the shape — which is
precisely the property RQ3 needs (the agent always sees N addressable slots;
what changes is where they point). Extraction: does view *identity* stay stable
across shapes, or does slot 3 mean something different per object? That
question decides whether this template is usable. Code: `ajhamdi/MVTN`.

| 17 | `view-gcn` | Wei, Yu, Sun — *View-GCN: View-Based Graph Convolutional Network for 3D Shape Analysis*, CVPR 2020. [CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_View-GCN_View-Based_Graph_Convolutional_Network_for_3D_Shape_Analysis_CVPR_2020_paper.pdf) | paper | med |

A literal view graph: nodes are views, edges are nearest-neighbour relations on
the view sphere, with coarse-to-fine hierarchical view sampling over that
graph. The modern engineering answer to "how do you define adjacency over a
set of views" when there is no `h±1`. Read for the graph construction and the
hierarchical pooling, not the classification results.

| 18 | `viewactive` | *ViewActive: Active Viewpoint Optimization from a Single Image*, 2024. [arXiv:2409.09997](https://arxiv.org/abs/2409.09997) | paper | low |

Self-described as drawing inspiration from aspect graphs; replaces the discrete
partition with a continuous **3D Viewpoint Quality Field** (self-occlusion
ratio, occupancy-aware surface-normal entropy, visual entropy). Relevant here
as the "field instead of graph" alternative to structuring view space — and as
evidence of what a 2024 revival actually looks like. Score internals belong to
Branch 1; this branch takes the structural claim only.

| 19 | `gennbv` | Chen et al. — *GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction*, CVPR 2024. [arXiv:2402.16174](https://arxiv.org/abs/2402.16174) | paper | med |

The best-argued **counter-position** to this entire branch: they claim
hand-crafted or object-specific candidate sets (~100 views, or a constrained
hemisphere) are the bottleneck and move to a free 5D action space. Read for the
argument, not the method — our contract fixes discrete selection, so we need
the strongest available statement of why that is a limitation, and their
answer to "with a large action space you need an informative state
representation" is the trade we would be making. Constraint tag: free-flying
camera, no IK/collision, RL.

| 20 | `nbvnet-supervised-nbv` | Mendoza et al. — *Supervised Learning of the Next-Best-View for 3D Object Reconstruction*, 2019. [arXiv:1905.05833](https://arxiv.org/abs/1905.05833) | paper | med |

Our current baseline's academic twin: NBV posed as **classification over 14
discretized poses on an upper hemisphere**, chosen because the object sits on a
table. Read for the honest engineering of a tiny fixed discrete view space —
how the 14 were chosen, how ground-truth labels were generated, and what the
authors say the discretization costs them. Benchmark tag → Branch 2.

| 21 | `hestia-hierarchical-nbv` | *Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction*, 2025. [arXiv:2508.01014](https://arxiv.org/abs/2508.01014) | paper | low |

Recent, and structurally on-topic: a **hierarchical decomposition of a 5-DoF
view space** plus a face-aware term "to avoid overlooking geometry". Abstract
does not reveal the hierarchy's levels — read only to extract whether the
hierarchy is spatial coarse-to-fine or object-anchored. Cheap to check, drop if
it is the former.

---

### Group E — Cross-shape addressing consistency
The RQ3 hard requirement: one convention that survives a change of object.

| 22 | `structurenet` | Mo, Guerrero, Yi, Su, Wonka, Mitra, Guibas — *StructureNet: Hierarchical Graph Networks for 3D Shape Generation*, SIGGRAPH Asia 2019. [arXiv:1908.00575](https://arxiv.org/abs/1908.00575) | paper | med |

The object→part hierarchy done properly: shapes as hierarchies of n-ary part
graphs, with **semantic-label-indexed child slots** and an order-invariant
encoder over siblings. That pair — a stable semantic slot for each part plus
explicit invariance to sibling ordering — is the mechanism that keeps part
identity consistent across shapes with different part counts. Read the
structure representation, ignore the generative machinery. Directly informs
"object → primitive → view" addressing.

| 23 | `fu-upright-orientation` | Fu, Cohen-Or, Dror, Sheffer — *Upright Orientation of Man-Made Objects*, ACM TOG (SIGGRAPH) 2008. [PDF](https://www.cs.tau.ac.il/~dcor/articles/2008/upright_orientation.pdf) | paper | med |

The prerequisite nobody budgets for: before any addressing convention can
transfer between objects, the objects need a **canonical frame**. Key negative
result to carry: PCA principal axes are non-robust for pose normalization and
only reduce the candidates to six; symmetry axes help but do not close it. Their
answer is the natural supporting base — which for tabletop inspection is
already given by the table, a genuine simplification for us. Read the critique
of PCA and the base-finding criterion.

| 24 | `partfield` | *PartField: Learning 3D Feature Fields for Part Segmentation and Beyond*, 2025. [arXiv:2504.11451](https://arxiv.org/abs/2504.11451) | paper | low |

Modern learned cross-shape part consistency: feature fields where correspondence
emerges across differing topologies and part counts, used for co-segmentation
by seeding target clustering with source part centroids. Read for the honest
failure mode the authors admit — the seeding trick is fragile when the target
has a different part count, which is exactly the box-vs-cylinder case. Overlaps
Branch 3 (extraction); this branch takes the *labelling consistency* only.

| 25 | `name-that-part` | *Name That Part: 3D Part Segmentation and Naming*, 2025. [arXiv:2512.18003](https://arxiv.org/abs/2512.18003) | paper | low |

Segmentation *and naming* — producing human-legible part identifiers rather
than indices. A name is an address, and this is the only found work that treats
producing the name as the task. Bridge to Branch 6; take the naming protocol
and any evidence about cross-shape name stability.

| 26 | `armeni-3d-scene-graph` | Armeni et al. — *3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera*, ICCV 2019. [arXiv:1910.02527](https://arxiv.org/abs/1910.02527) | paper | low |

Layered graph over spatial concepts where **the camera is itself a layer/node**
alongside objects and rooms — i.e. viewpoints and geometry addressed in one
structure. Read for the layer schema and the inclusion/adjacency edge
semantics; hand the LLM-consumption angle to Branch 6.

---

### Group F — Action abstraction analogies from robot learning
How other fields keep a discrete action set small, composable, and meaningful.
Borrow the *formalism*, not the learning.

| 27 | `hacman-plus-plus` | *HACMan++: Spatially-Grounded Motion Primitives for Manipulation*, RSS 2024. [arXiv:2407.08585](https://arxiv.org/abs/2407.08585) | paper | **high** |

The direct template for a view address. Their action is a **triple: which
primitive type, where it is grounded on the object, and how it is parameterized**
— i.e. `(what, where-on-the-geometry, how)`. Map straight onto ours:
`(face-orthogonal | cylinder-orbit | ..., primitive_id / surface location,
standoff + roll)`. Read the action representation section; the RL is not our
problem. Best single analogy in the branch.

| 28 | `masson-parameterized-actions-pamdp` | Masson, Ranchod, Konidaris — *Reinforcement Learning with Parameterized Actions*, AAAI 2016. [arXiv:1509.01644](https://arxiv.org/abs/1509.01644) | paper | **high** |

The formalism that names our structure: a **PAMDP**, A = {(k, x_k)} — pick a
discrete action k, then continuous parameters x_k from that action's own
parameter set. Our contract says the VLM picks discretely; a PAMDP framing lets
the VLM own k while a deterministic rule or optimizer owns x_k, which is
exactly the analytical/optimization split in the contract's Candidate
Directions. Read the formalism and the hierarchy argument; skip the algorithms.

| 29 | `omnimanip` | *OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints*, 2025. [arXiv:2501.03841](https://arxiv.org/abs/2501.03841) | paper | med |

Object-anchored addressing that a VLM actually consumes: the object's canonical
space, defined by functional affordances, provides **semantically meaningful
interaction primitives (points and directions)** that translate VLM
commonsense into 3D spatial constraints, with a resample-render-and-check loop.
The closest working system to "geometry-derived, named, model-selectable
spatial actions". Extraction here is the primitive definition and canonical
space; the VLM interface belongs to Branch 6.

| 30 | `sutton-precup-singh-options` | Sutton, Precup, Singh — *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*, AI 112:181–211, 1999. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370299000521) | paper | med |

The canonical account of a discrete vocabulary of extended actions: an option
is ⟨I, π, β⟩ — where it may be initiated, what it does, when it ends. The
initiation set I is the piece this branch needs: it is the formal way to say
"this view action is only addressable when the object has a planar face here",
which is how the action space stays consistent while its *contents* change with
shape. Read §1–3 and the interruption result; skip intra-option learning.

| 31 | `dalal-raps-parameterized-action-primitives` | Dalal, Pathak, Salakhutdinov — *Accelerating Robotic RL via Parameterized Action Primitives*, NeurIPS 2021. [arXiv:2110.15360](https://arxiv.org/abs/2110.15360) | paper | low |

A hand-specified library of parameterized primitives, and the reported cost:
the augmented discrete-continuous space imposes a real exploration burden. Read
only for that negative — it is the quantitative version of "a bigger action
space is harder to navigate", which is RQ4's whole worry. Companion:
`nasiriany-maple` ([arXiv:2110.03655](https://arxiv.org/abs/2110.03655)),
same lesson, skip unless the first is thin.

---

### Second-tier bench (found, catalogued, not worth a read slot now)

- **Zeng, Wen, Zhao, Liu — *View Planning in Robot Active Vision: A Survey*, Computational Visual Media 6(3), 2020** ([sciopen](https://www.sciopen.com/article/10.1007/s41095-020-0179-3)) — open access, overlaps #14/#15 on view-space representation. Read only if Branch 4's mapper does not already own it.
- ***A Structured Review and Taxonomy of Next-Best-View Strategies for 3D Reconstruction*, 2025** ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667393225000171)) — freshest taxonomy; surface-based vs. volume-based split. Branch 4 owns it.
- **Hydra / Foundations of Spatial Perception** ([arXiv:2201.13360](https://arxiv.org/abs/2201.13360)) — hierarchical layered spatial graphs built online. Excellent engineering, wrong scale (rooms/buildings, not object faces).
- ***A Solution to Efficient Viewpoint Space Partition in 3D Object Recognition*** (IEEE, 2010) — prunes nonexistent EEE visual events. Only matters if we actually compute an exact aspect graph, which #4 suggests we should not.
- **Koenderink & van Doorn, *The Internal Representation of Solid Shape with Respect to Vision*, Biol. Cybernetics 32:211–216, 1979** ([Springer](https://link.springer.com/article/10.1007/BF00337644)) — origin of the visual potential. Historical citation; #3 carries the content.
- **View sphere partitioning via flux graphs** ([Frontiers ICT 2015](https://www.frontiersin.org/articles/10.3389/fict.2015.00024/full)) — skeletal-similarity partition of the view sphere. Same idea as #4 with a different metric; read only if #4's metric does not transfer to PLY space.
- **Canonical-view perception literature** (Blanz/Tarr/Bülthoff 1999 *Perception* 28:575–599; Mezuman & Weiss NeurIPS 2012) — human view preference and the negative result that stability- and frequency-based formal theories fail to predict it. Interesting, but view *quality* is Branch 1's and the aesthetics motivation is out of scope.
- **HyAR / P-DQN / MP-DQN / H-PPO** — algorithmic PAMDP solvers. #28 gives the formalism; the solvers assume RL we are not doing.
- **Topological navigation maps** (osmAG [2309.04791](https://arxiv.org/abs/2309.04791), FHT-Map [2310.13899](https://arxiv.org/abs/2310.13899)) — hierarchical topometric graphs with places-as-nodes. Right shape of idea, but the navigability problem there is metric traversal, not view addressing.

---

## Questions this branch must answer

Tasks for the readers and for the round-3 synthesis. Grouped by what they
unblock.

### Addressing (RQ3 core)

1. **What is an address, formally, in each scheme?** Classify every candidate
   into one of three: (a) an index into a global tessellation
   (`{h,v}`, HEALPix, H3), (b) an object-part reference plus a local offset
   (`face_3 / orthogonal`, C-spaces, HACMan++'s `(what, where, how)`), or
   (c) a node ID in a graph whose meaning is only relational (aspect graph,
   view-GCN). Which of the three lets a *blind* orchestrator predict what it
   will get before it moves?
2. **What is the minimal scheme in which "the orthogonal view of face 3" is a
   native address** rather than a nearest-neighbour lookup into a grid? Does
   #11's C-space formulation actually deliver this in closed form, and what
   does it require as input — CAD features, or fitted primitives from a noisy
   cloud?
3. **Does the address need to encode the pose at all,** or can it be purely
   symbolic with pose resolved deterministically downstream? Test the
   PAMDP split (#28): VLM emits `k`, geometry emits `x_k`.
4. **Is there a stable ordering over addresses,** and does it carry meaning
   (next/previous, opposite, adjacent-face)? RotationNet's ordered view
   sequences (#10) and HEALPix's ring vs. nested numbering (#8) are the two
   precedents — collect what ordering buys and what it costs.

### Relating / adjacency

5. **How is adjacency defined once views leave a uniform sphere?** Enumerate
   the found definitions: visual-event boundary (aspect graph), k-NN on the
   view sphere (view-GCN), similarity below threshold (Cyr–Kimia), cell
   neighbour in a tiling (HEALPix/H3), shared visible surface region
   (visibility matrix). For each: is it computable on our PLY, and is it
   *nameable*?
6. **Does the agent need adjacency at all?** Argue both sides with evidence:
   the current agent navigates `{h,v}` by incrementing bins, but a flat menu of
   8–20 named views may need no topology whatsoever. If adjacency is
   unnecessary, most of this branch's aspect-graph inheritance is decoration —
   say so explicitly.
7. **What happens at the seams?** Every sphere tiling has defects (H3's 12
   pentagons, #9). Every primitive decomposition has edges and corners where
   two faces meet and neither orthogonal view is right. How do the found
   schemes name the seam, and does the seam get an address of its own?

### Tractability — the aspect-graph failure mode (RQ4 pressure, critic mandate)

8. **Do the aspect-graph size bounds actually bite for us?** Compute
   Θ(n²)/Θ(n⁶) orthographic and Θ(n³)/Θ(n⁹) perspective (#2) for: a 6-face
   box, a cylinder, a box+cylinder assembly. Report absolute node counts. This
   is arithmetic, not judgment, and it must be in the round-3 synthesis.
9. **Which of the recorded causes of death applies to us?** From #1, separate
   (i) size, (ii) non-computability on real curved/noisy input, (iii) the
   indexing problem "which node am I in", (iv) no compelling use case. State
   for each whether it transfers when the graph is used for **action
   addressing** rather than **observation indexing** — the indexing objection
   plausibly does not, and that is the branch's headline claim if it holds.
10. **What is the smallest structure that expresses the contract's target
    behaviour** — "box scanned face-by-face, cylinder orbited normal to the
    surface"? Give the node/edge count. If it is under ~20 nodes for our
    object class, the explosion argument is irrelevant to us and RQ4's
    tractability requirement is satisfied by construction.
11. **Which pruning mechanisms survive?** Entity-based restriction (#5),
    similarity merging (#4), hierarchical refinement (#8, #17). Which of these
    maps onto "we only care about the surface region currently under
    inspection"?

### Consistency across shapes (RQ3 hard requirement)

12. **How do the schemes handle variable arity** — a box has 6 faces, a
    cylinder has 2 caps plus 1 lateral orbit, an assembly has both? Collect the
    answers: semantic slots (#22), co-segmentation seeding (#24), fixed
    cardinality with adaptive placement (#16). Which keeps a *stable* meaning
    for a given address across objects, and which merely keeps the *count*
    stable?
13. **What canonical-frame machinery is presupposed,** and does it survive a
    survey cloud rather than a clean mesh? #23 says PCA is not robust and
    reduces only to six candidates; the table gives us the up-axis for free —
    confirm what is still needed (front/side disambiguation) and whether we
    need it at all for an inspection action space.
14. **Is a bounded-cardinality guarantee available?** Can we promise the agent
    "always between N and M addressed views regardless of shape"? #16 is the
    only found mechanism that gives this while remaining geometry-conditioned —
    verify whether slot identity is stable or arbitrary.

### Interface with the rest of the project

15. **Where does the discrete/continuous boundary sit?** The contract fixes
    discrete selection; #19 argues discretization is the bottleneck. Is the
    middle path — a discrete address naming a *region*, with a deterministic
    or optimized continuous refinement inside it — supported by the options
    framework (#30, the initiation set + policy) and PAMDP (#28)? This
    directly serves the contract's optimization track.
16. **Does the addressing scheme survive infeasibility?** When the ideal
    orthogonal view fails IK/collision/min-shell-radius, does the address
    persist with a degraded pose, disappear from the menu, or get flagged?
    Collect any precedent for an addressed "nearest feasible" fallback.
    → hand findings to Branch 4.
17. **What does it cost to build the structure, and when?** Offline per-object
    compilation (aspect graphs), online incremental, or amortized? Our loop is
    survey-then-inspect in one session — record every reported build time and
    whether it assumed a preprocessing phase.
18. **What is liftable?** For each candidate, note input requirements (mesh /
    cloud / fitted primitives / CAD features) and available implementation
    (`healpy`, `h3`, `MVTorch`, Open3D). The auxiliary goal wants something
    integrable, not a research program.
19. **Benchmark tag → Branch 2:** does any found work compare a
    geometry-conditioned view space against a uniform sphere *on the same task
    with a metric*? If nothing does, that absence is itself a finding and
    strengthens the case for building our own benchmark.

---

## Search trail

Executed 2026-08-27 via `WebSearch`; abstracts and landing pages only, no deep
reads. Identifiers cross-checked against the arXiv API where a slug was
uncertain (`1905.05833`, `1910.02527`, `1509.01644`, `1511.04143`,
`2110.15360`, `2110.03655` confirmed; View-GCN confirmed absent from arXiv,
cited via CVF).

Queries run:

- `aspect graph survey object recognition viewpoint space partition`
- `"aspect graph" criticism combinatorial explosion "why aspect graphs" not used`
- `viewsphere tessellation icosahedral viewpoint sampling uniform sphere discretization view planning`
- `HEALPix spherical grid hierarchical pixelization addressing scheme sphere`
- `"view graph" active vision viewpoint adjacency nodes views edges camera moves object recognition`
- `aspect graph revival deep learning modern 2020 2024 viewpoint space partition neural`
- `parameterized action space reinforcement learning discrete-continuous hybrid action PAMDP`
- `object-centric action space skill vocabulary robot manipulation discrete primitives library`
- `"canonical views" object recognition psychology preferred viewpoint stability computational`
- `topological map navigation graph nodes places edges robot navigability hierarchical abstraction survey`
- `viewpoint generation offset surface dilated surface model-based inspection per-face camera frame patch normal`
- `"view space" representation next best view action space discretization learned NBV policy sphere candidate views`
- `Scott Roth Rivest "View planning for automated three-dimensional object reconstruction and inspection" ACM Computing Surveys viewpoint space taxonomy`
- `consistent part labeling across shapes co-segmentation canonical object frame semantic part correspondence 3D shapes`
- `Yershova LaValle deterministic sampling SO(3) Hopf fibration uniform incremental grids sphere multiresolution`
- `options framework temporal abstraction hierarchical reinforcement learning Sutton Precup Singh semi-MDP`
- `StructureNet hierarchical graph networks shape structure part hierarchy consistent addressing 3D`
- `3D scene graph hierarchical representation Armeni Hydra layered spatial abstraction robotics`
- `Cyr Kimia similarity-based aspect graph 3D object recognition viewing sphere segmentation homogeneous regions`
- `multi-view CNN camera setup 12 views dodecahedral RotationNet viewpoint arrangement convention 3D shape recognition`
- `"view planning in robot active vision" survey 2020 Zeng viewpoint space representation classification`
- `upright orientation man-made objects canonical pose alignment 3D shape collection consistent frame`
- `discrete global grid system H3 hexagonal hierarchical index S2 geometry cell addressing neighbor traversal`
- `Koenderink van Doorn "internal representation of solid shape" visual potential aspect 1979`
- `"Viewpoint Generation Using Feature-Based Constrained Spaces for Robot Vision Systems" Gomez Robotics 2023 abstract`
- `aspect graph sensor planning vision algorithm compiler Ikeuchi Hutchinson Kak sensing strategy viewpoint`
- `MVTN multi-view transformation network learned viewpoint placement adaptive to shape 3D recognition`
- `geometry-conditioned viewpoint action space robot inspection 2026 structured view space agent`

### Gaps and dead ends

- **No 2020s "aspect graph revival" exists as a named movement.** Searches for
  it return either view-based deep 3D analysis (view-GCN, multi-view GCNs,
  ECCV'24 multi-part recognition) that inherits the *spirit* without computing
  a viewpoint-space partition, or unrelated neural graph-partitioning work.
  Treat any claim of a revival with suspicion; #4 (2004) and #18 (2024) are
  the only genuine descendants found.
- **No paper found that poses exactly our problem** — a geometry-conditioned,
  addressed, VLM-navigable view space for manipulator inspection. The 2026
  "geometry-conditioned" hits are manipulation VLA/world-model papers where
  geometry conditions end-effector actions, not view selection. This gap is
  the branch's main positive finding and should be stated as such in round 3.
- **Cross-shape *view* addressing consistency is essentially unstudied.**
  Cross-shape *part* consistency is a mature literature (Group E); nobody
  found has carried it through to naming views. RQ3's hard requirement may be
  genuinely novel — which also means no precedent to lean on.
- **Hutchinson & Kak's evidential sensor-planning line did not surface** and
  was not chased; flag for a later targeted search if aspect-graph-based
  sensing strategies turn out to matter.
- **Web search budget exhausted** before running two planned queries:
  view-GCN's graph construction details, and coarse-to-fine/multi-resolution
  viewpoint sampling. Both are covered indirectly by #17, #8 and #14 but merit
  a fresh query in round 3.

### Cross-branch handoffs

- **→ Branch 1:** ViewActive's Viewpoint Quality Field components
  (self-occlusion ratio, occupancy-aware normal entropy, visual entropy); the
  per-face "best view = inverse surface normal" formulation recorded in #14.
- **→ Branch 2 (benchmark):** #20's ground-truth generation procedure for
  discrete NBV labels; the finding that no head-to-head
  geometry-conditioned-vs-uniform-sphere comparison was located.
- **→ Branch 3:** #24's admitted failure mode on differing part counts; #23's
  critique of PCA for pose normalization — both bear on primitive
  decomposition reliability.
- **→ Branch 4:** #14 and #15 are shared spines (they own selection, we own
  the representation taxonomy); #19's argument about limited action spaces;
  question 16's infeasibility findings.
- **→ Branch 6:** #29's VLM-consumable object-centric primitives; #25's part
  naming; #10's view-ordering conventions. This branch stops at the structure;
  verbalization is theirs.
