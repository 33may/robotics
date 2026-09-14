# Mo, Guerrero, Yi, Su, Wonka, Mitra, Guibas — "StructureNet: Hierarchical Graph Networks for 3D Shape Generation"

SIGGRAPH Asia 2019 (ACM TOG 38(6)). arXiv:1908.00575v1. Full HTML text read (ar5iv-style rendering of the arXiv source), not just the abstract. Selected for this branch as the "object-to-part hierarchy done properly" candidate — semantic-label-indexed child slots plus an order-invariant sibling encoder, i.e. a candidate for the object → primitive → view skeleton.

## What the paper actually is

StructureNet is a **generative model for 3D shape geometry**, trained on PartNet, that learns to encode/decode/generate/interpolate/edit hierarchically-structured shapes (chairs, tables, cabinets, etc.). It is not a viewpoint, camera, or action-space paper. There is **no mention of cameras, views, viewpoints, aspect graphs, or observation planning anywhere in the paper** — this needs to be stated up front because the branch selection blurb frames it as an "object → primitive → view skeleton" candidate, but the paper only goes as far as "object → primitive." Any transfer to view-addressing is an extrapolation I make below, not something the paper claims.

## Core representation (§4, "Hierarchy of Graphs for Shape Structure")

A shape is `S = (P, H, R)`:
- `P` = set of N parts, each `Pi = (Gi, li)` — geometry `Gi` plus a semantic label `li` drawn from a fixed, category-specific label set (36–82 labels/category in PartNet).
- Geometry `Gi` is either a minimum oriented bounding box `Bi = (ci, qi, ri)` (center, quaternion orientation, size) or a point cloud `Ai = {x1...xk}` (1,000 uniformly sampled surface points per part).
- `H` = a **tree**: root = whole shape, recursively decomposed into parts, terminating in leaf parts. Nodes have variable numbers of children (unbalanced tree); a hard cap `np = 10` children per node is enforced (shapes with any node exceeding this are filtered out of training/eval).
- `R` = **sibling-only** relationship edges `(Ci, Ri)`, four typed edges: adjacency (τa), translational symmetry (τt), reflective symmetry (τr), rotational symmetry (τo). Restricting relationships to siblings (not all pairwise, avoiding Θ(N²)) is an explicit design choice, justified by compute cost plus an empirical claim that the relationships that matter occur between siblings. Symmetry detection is precomputed with Wang et al.'s method; adjacency is precomputed with a fixed geometric rule: **two leaf parts are adjacent iff their closest-point distance < 0.05 × the shape's average bounding-sphere radius.**

## Encoder/decoder mechanism (§5)

VAE with 256-d latent `z`. Bottom-up encoder: leaf geometry encoded (SLP for boxes; PointNet + SLP for point clouds), then a graph encoder does 2 rounds of message passing along the typed sibling edges (GIN/DGCNN-style: per-edge MLP message, **max-pool aggregation** across neighbors — Eq. 1–3), producing a permutation-invariant feature per node; recursed up the tree to a single root feature = `z`.

Top-down decoder: given a parent feature, an MLP (`gparts`) always produces a **fixed np=10 slate of child feature vectors**; separate small networks predict, per slot, a part-existence probability, all np² pairwise edge-existence probabilities (typed), the semantic label, and a leaf/non-leaf flag. Two more rounds of message passing refine features using the *predicted* edges before recursing further.

Critically: **the np=10 slots are not hard-bound to fixed semantic labels.** A slot's label is predicted dynamically per shape by the label-decoder head; the encoder is permutation-invariant (max-pool discards sibling order); and slot↔ground-truth-part correspondence for the reconstruction loss is computed via a **geometry-based (Chamfer-distance) linear assignment**, matched per child-graph, not a label lookup. Semantic labels enter only as one soft supervised loss term (Eq. 16), not as an identity/index constraint. I verified this precisely because the branch's selection blurb implies a stronger "semantic-label-indexed" mechanism than the paper delivers — see Negative findings below.

## Losses (§5.3)

`Ltotal = E_S[Lr(S) + Lsc(S) + β·Lv(S)]`, β=0.05, `Lv` = standard VAE KL term. `Lr` (reconstruction, Eq. 18) sums, over the matched-part assignment: geometry Chamfer loss (α=20), a normal-orientation loss over box faces (γ=10), part-existence and edge-existence cross-entropy, semantic-label cross-entropy (λ=0.1), leaf-classification cross-entropy. `Lsc` (structure-consistency, Eq. 22) = symmetry loss + adjacency loss, applied to the *generated* structure to keep sibling relationships geometrically valid (constrains whole subtrees, not just the parent).

## Data and numbers (§6)

- PartNet: 26,671 shapes, 24 categories. Primary categories after np=10 filtering: chairs 4,871, tables 5,099, cabinets 862 (also beds/vases/trashcans in supplementary). 7:1:2 train/val/test split.
- Per-shape averages: **16.94 parts, hierarchy depth 3.59, 29.93 relationship edges** — depth is a mean only, no variance reported.
- Table 1 (chair reconstruction): geometry error `EP`=0.061 (full model, non-squared Chamfer, range [0,2]), hierarchy error `EH`=0.200, edge error `ER`=0.246, reconstructed-consistency `Erc`=0.018, ground-truth-consistency `Egc`=0.023. Ablation: dropping edges raises `EP` to 0.065. GRASS baseline: 0.103 (original), 0.082 (with PartNet semantic guidance) — StructureNet beats it, but GRASS also fails outright on 840/4,871 chairs (binary-tree scaling failure) while StructureNet doesn't.
- Table 2 (generation quality/coverage, relative to StructureNet=1.0): GRASS original 0.714/0.818, GRASS+PartNet 0.788/0.818, StructureNet-no-edges 0.984/0.989 — edges measurably help generation fidelity, not just reconstruction.
- Model cost: `np`=10 cap makes decoder memory quadratic in `np` (~1–2 GB, called "manageable"), 2 message-passing rounds encoder + decoder.

## Applications (§6.2–6.5)

Generation, interpolation (including *partial*, single-part interpolation via latent-space projection), and structure-aware editing (Eq. 26: latent-space optimization balancing proximity to original + satisfying box edits + structure-consistency loss) are all shape-synthesis operations — none touch cameras or observation. The one application worth flagging for "does it tolerate noisy real input": **§6.4, shape abstraction** — a ResNet18 image encoder or PointNet++ point-cloud encoder is trained to regress into the same latent space as the shape encoder (losses `LI`, `LO`), letting the network infer a labeled part hierarchy from a single rendered image or from a point cloud, including **real-world ScanNet scans** (partial, noisy, no CAD features). The paper reports this recovers "plausible" structure despite missing/noisy regions — qualitative only, no metric given for the noisy-input case.

## Load-bearing assumptions

- Requires a large, per-category-labeled training corpus (PartNet, thousands of shapes with consistent semantic part annotations) to learn both the geometry decoder and the (soft) label-to-slot regularity. This is a learned statistical system, not a closed-form rule.
- Presupposes shapes are pre-aligned to a canonical, category-consistent frame (PartNet convention: upright, front-facing) before any part/box geometry is computed — the paper never discusses recovering this canonical frame from an arbitrary-pose scan; it's inherited as a dataset property.
- The `np=10` sibling cap and quadratic decoder memory are architectural limits, not requirements of the underlying idea — but they mean the mechanism as shipped assumes small-to-moderate branching factors (fits box/cylinder-scale primitive counts easily; irrelevant ceiling for us).
- Sibling-only relationships (no cross-branch edges) is justified for object part structure; whether the same restriction is defensible for a view graph (where non-sibling-face adjacency, e.g. two lateral faces of a box, matters) is untested here.

## What transfers to our setting

1. **A concrete, computable adjacency rule for primitives fitted from geometry**: closest-point distance < 0.05 × average bounding-sphere radius. This is directly liftable as a definition of "these two primitives/faces are adjacent" computed on our fitted-primitive geometry (relevant to RQ5/RQ17 — nameable, computable on our PLY-derived primitives, no learning required).
2. **Point-cloud-native part representation with real, noisy input tolerance** (§6.4, ScanNet abstraction): parts/geometry are represented and encoded directly from sampled points (PointNet), and the pipeline is demonstrated — qualitatively — on real partial/noisy scans, not just clean meshes or CAD features. This is evidence, from a different application, that the general idea of "geometry-derived local frames indexed by semantic role" does not intrinsically require CAD features. It does not evidence that our deterministic primitive-fit approach will be robust — that's untested here.
3. **A typed, sibling-relative relational vocabulary** (adjacency / translational / reflective / rotational symmetry) as a small, nameable edge-type set — a plausible naming scheme if we ever wanted to label relations between primitive faces (e.g. "opposite face" ≈ reflective symmetry, "adjacent face" ≈ adjacency), though the paper applies it to parts, never to views.
4. **The order-invariant sibling encoder (typed-edge message passing + max-pool)** is a real mechanism for handling a variable number of children without hand-indexing, useful in principle for an object with a variable number of faces/primitives — but it is a *learned, continuous-vector* mechanism (256-d features via a trained VAE), not a symbolic addressing scheme a blind LLM orchestrator could consume. It does not produce human/agent-nameable slot IDs.

## What does NOT transfer, and why (negative findings)

- **No view/camera/observation content whatsoever.** Zero relevance to viewsphere tessellation, aspect graphs, view adjacency, or any of the branch's core "addressing a view" questions. Any use of this paper for RQ1 (formal address classification), RQ4 (ordering), RQ7 (seams), RQ13 (arity/meaning stability *of views*), or RQ14 (canonical frame *for inspection*) is an analogy the branch has to construct itself, not something the paper supports.
- **"Semantic-label-indexed child slots" is not what the architecture does — this corrects the branch's framing of the paper.** Decoder slots are a fixed-size (np=10) array whose semantic label is predicted per-slot per-shape by the network, not a stable lookup table (slot 3 ≠ "always chair leg"). Cross-shape identity/correspondence is recovered only via a **geometry-based linear assignment computed for the training loss**, and the closest the paper comes to a stable-identity claim is a category-level empirical regularity ("most chairs... naturally decompose into backrest, seat, and base at the top level") — a soft, learned statistical pattern over thousands of annotated examples, not an architectural or formal guarantee of address stability. Directly relevant to RQ12 (stable MEANING vs. stable COUNT): StructureNet gives neither a proof nor even a strong claim of per-slot semantic stability; what stability exists is emergent from training-set scale and would need an equivalent-sized labeled corpus per object class to reproduce — something our deterministic, rule-based, no-training-corpus primitive fitting (box/cylinder/assembly) neither has nor wants.
- **This is a heavyweight learned system (VAE + GNN, trained per category on thousands of PartNet shapes), the opposite of the deterministic primitive-level rules our framing doc explicitly prefers.** Nothing here is closed-form; nothing is offered as a lightweight, inspectable rule a small number of primitive types (box, cylinder) could use directly. Not liftable as a library/tool for our pipeline (no camera code, no available integration point — the closest usable artifact is the adjacency-threshold *idea*, not code).
- **No canonical-frame recovery from an unposed/noisy scan is addressed** — canonicalization is a dataset precondition (PartNet ships pre-aligned shapes), not solved or discussed by the method. This paper cannot answer RQ14's "does front/side disambiguation survive a survey cloud" question; it assumes the answer is already given.
- **No benchmark, metric, or comparison against a uniform/geometry-agnostic baseline of any kind related to observation/view planning** — reinforces the branch's cross-cutting finding (RQ18) that this comparison is absent from the literature generally; this paper simply isn't in that space.
- **No discussion of feasibility/reachability/degraded fallback** (RQ16) — irrelevant, since there's no robot or camera in the loop at all.

## Citations (for checkability)

- Shape representation `S=(P,H,R)`, part definition, bounding-box/point-cloud geometry: §4.
- Sibling-only relationship restriction, four edge types, adjacency threshold (0.05× avg bounding-sphere radius), symmetry via Wang et al.: §4, §6 (Data Preparation).
- Encoder/decoder architecture, Eq. 1–6 (message passing, max-pool aggregation, part/edge existence prediction): §5.1–5.2.
- Losses, Eq. 7, 11–18, 20–22 (total loss, Chamfer/normal/existence/semantic/leaf losses, structure-consistency): §5.3.
- Dataset stats (26,671 shapes, 24 categories; chairs/tables/cabinets counts; 16.94 parts, depth 3.59, 29.93 edges, 36–82 labels/category), np=10 filtering, 7:1:2 split: §6 (Data Preparation).
- Table 1 (reconstruction errors, GRASS comparison incl. 840/4,031 chair failures), Table 2 (generation quality/coverage vs. GRASS): §6.1–6.2.
- Shape abstraction from images/point clouds/ScanNet, encoder losses `LI`/`LO`: §6.4.
- Shape editing optimization, Eq. 26: §6.5.
- Limitations (data bias, np=10 hard cap, quadratic decoder memory ~1–2GB, noise manifests as structural error not blur): §6.6 / discussion.

## Assessment for the branch

StructureNet is a strong "object → primitive" hierarchy paper — order-invariant sibling GNN encoding, typed sibling relations, category-consistent semantic labels — but it is a **learned shape-generation model with no view/camera content**, and its part-addressing mechanism is weaker than the branch's selection blurb implies: slots are not semantic-label-indexed at inference, only statistically regularized toward category-consistent labels by training on a large annotated corpus. The one directly liftable, checkable artifact is the adjacency rule (distance threshold on bounding-sphere radius) and the general point that a point-cloud-native, no-CAD-features part/geometry encoding survives real noisy scans (ScanNet) in a different application. Everything else — the "→ view" half of the object → primitive → view skeleton the branch is looking for — has to come from elsewhere; this paper stops at primitives.
