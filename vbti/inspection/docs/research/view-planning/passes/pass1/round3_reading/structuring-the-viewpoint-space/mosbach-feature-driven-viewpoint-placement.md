# Mosbach, Gospodnetić, Rauhut, Hamann, Hagen — "Feature-Driven Viewpoint Placement for Model-Based Surface Inspection"

Machine Vision and Applications 32(1):8, 2021. https://doi.org/10.1007/s00138-020-01116-y
PDF fetched: https://web.cs.ucdavis.edu/~hamann/MosbachGospodneticRauhutHamannHagenMachineVisionAndAppplicationsPaperRevision2AsPublished10292020.pdf
(21 pages, open access CC-BY 4.0). Read in full (all 8 sections + references). Extracted text cached at `/tmp/mosbach2021.txt` (not committed).

This paper is also read for the `view-set-selection-under-constraints` and `view-quality-scoring` branches (see
`../view-quality-scoring/mosbach-feature-driven-viewpoint-placement.md` for the full method/results/quality-functional
writeup — not duplicated here). **This note re-reads the same paper strictly through the addressing/structuring lens**:
what an "address" is in this scheme, whether it is object-anchored or global, how (or whether) adjacency/ordering/seams
are handled, and whether the scheme composes across shapes.

---

## Core method, in address terms (Sect. 3, pp. 4–11)

Pipeline: mesh → **B-spline surface network** (Eck & Hoppe [7] reconstruction: mesh → surface Voronoi tessellation →
dual Delaunay-like complex → quad merge → per-quad least-squares B-spline fit with thin-plate fairing) → **recursive
quadtree subdivision in parameter space**, driven by a feature functional `E(S,Ω)` and threshold `t` (Algorithm 1,
p. 6–7) → **one viewpoint candidate per leaf**, placed by a closed-form normal offset (Eq. 6, p. 10):

```
C_{S,Ω} = S(u_c, v_c) + n(u_c, v_c) · d
```

`(u_c,v_c)` = center of the leaf's parameter region `Ω`; `n` = analytic surface normal at that point; `d` = a
constant, manually-chosen standoff. View direction is the reversed normal.

## What is the address, formally? (branch Q1)

**Full address = `(surface_id, Ω_leaf)`** — a two-level, object-anchored, hierarchical address:

1. **`surface_id`** — which B-spline surface in the network (the paper's closest analogue to a "face" or "primitive").
   Determined at model-construction time by the Eck–Hoppe quad layout, not by any semantic face definition.
2. **`Ω_leaf`** — a node in that surface's recursive parameter-space quadtree (Algorithm 1, `split()` divides
   `[u0,u1]×[v0,v1]` into 4 equal quadrants, p. 7), i.e. a **quadtree path** (a bit-string of "which quadrant, which
   quadrant, …") within `[0,1]²`.

Pose is then a **deterministic closed-form function of the address** (Eq. 6) — no lookup, no nearest-neighbour search,
no optimization. This is squarely the branch's **type (b): object-part reference + local offset**, and it is the
cleanest example found in this pass of that type: the "offset" here isn't a residual on top of a discretized cell
(as in HACMan++'s "where"), it's the literal continuous parametric coordinate `(u,v)` on an object-anchored surface —
the address *is* a point in the object's own surface parameterization, not an index into any externally-imposed grid.

It is explicitly **not** type (a) (no global tessellation index — the "grid," such as it is, is a per-surface, per-object
quadtree built from that surface's own feature distribution, not a shared discretization like HEALPix/H3/`{h,v}`) and
**not** type (c) (the address carries no purely-relational/node-ID semantics — nothing here is defined by *what it is
adjacent to*; it is defined entirely by *where it sits on the surface*).

**Consequence for "predict what it will get before it moves"**: for this scheme, yes — trivially. Because pose is a
closed-form deterministic function of the address (Eq. 6), a blind orchestrator that is only told the address can have
the pose computed for it (or could, in principle, be told "this address = orthogonal view of surface 3, standoff `d`")
without any search. This is the strongest form of predictability among schemes read so far, because the address
*chose itself to be predictable* — the whole point of Eq. 6 is to bake "orthogonal, fixed offset" into the pose formula
so that nothing about the pose is left to be inferred.

## Is "the orthogonal view of face 3" a native address here, in closed form? (branch Q2)

**Yes, in closed form, by construction — Eq. 6 literally *is* "give me the orthogonal view of this surface region."**
There is no scoring, ranking, or search over candidate orientations: the view direction is defined as the reversed
surface normal at the address's parameter-center, full stop. This is the most direct affirmative evidence found in this
pass for the branch's premise that a surface-parameterized camera manifold can make "the orthogonal view of face 3" a
native address rather than a nearest-neighbour lookup into a pre-built grid.

**But it does not tolerate primitives fitted to a noisy cloud — it requires CAD-quality (or CAD-level-clean) surface
data, not our PLY.** Two separate load-bearing requirements stack up against our setting:

- **Requires a watertight triangle mesh**, not a raw point cloud. The paper's own aside ("a conversion to a triangle
  mesh can easily be achieved by a variety of freely available tools," p. 5) treats mesh construction as an
  already-solved preliminary, never revisited or evaluated.
- **Requires the Eck–Hoppe B-spline network fit to succeed**, and the paper states explicitly (Sect. 3.1, p. 5–6) that
  "the input data — containing a low or high degree of noise — have direct impact on the shape quality of the computed
  B-spline surface model," concluding that "it will be essential to ensure highly accurate scanning with a low degree
  of noise" for high-quality output. **No quantitative noise-tolerance bound is given anywhere in the paper** — this
  is asserted, not measured.

Our stated input (fused PLY, pose error of a few mm, per-view depth dropout, ~4 cm fused-vs-single-view offset) is
exactly the regime this paper flags as degrading its method, with no fallback offered. **The mechanism (closed-form
normal-offset addressing on a parametric surface) transfers conceptually to our primitive-level rules (box face →
normal offset; cylinder mantle → radial-normal offset); the machinery that produces the address space (Eck–Hoppe
B-spline network reconstruction from a mesh) does not run on our PLY and is not validated on noisy data anywhere in
this paper.**

## Adjacency: does this scheme define, need, or use it? (branch Q4, Q5, Q6)

**Adjacency between *surfaces* exists but only at model-construction time, as a fitting constraint, not as an
addressing/navigation structure.** The B-spline network is made C⁰-continuous by adding linear constraints that force
"control points of neighboring surfaces along their shared boundary curve to coincide" (p. 6, immediately before
Sect. 3.2). This is a real, computable, nameable relation ("surface *i* borders surface *j* along curve *C*") — but it
is used exclusively to keep the reconstructed geometry watertight, and **is never exposed to, or used by, the
subdivision, candidate-placement, or selection steps that follow.**

**Adjacency between *leaf segments* (within or across surfaces) is never defined at all.** The quadtree split (`split()`,
p. 7) partitions a parameter domain into 4 quadrants recursively; nothing in Algorithm 1 or Algorithm 2 records or
queries which quadrants/leaves are neighbours.

**Adjacency between *viewpoint candidates* is never defined or used.** The downstream selection step (Sect. 5, Eq. 11,
p. 10) treats the candidate set purely as a flat, unordered set `V = {v_1, ..., v_n}` indexed only for bookkeeping in a
binary visibility matrix `A ∈ {0,1}^{n_F × n_vpc}` (built by ray-tracing against the mesh) and solved as a Set-Cover ILP
via greedy NBV. **No graph structure, no neighbour relation, no notion of "next" or "opposite" candidate appears
anywhere in the selection formalism.** This is a strong, citable data point on the "flat menu needs no topology" side
of branch Q6: a full, published, evaluated view-planning pipeline gets from "geometry" to "a working ranked/selected
viewpoint set" without ever constructing or consulting an adjacency graph over candidates.

Contrast with prior work the paper itself cites (Sect. 2, p. 4): Gronle & Osten [11] *do* group candidates via an
**octree**, where "each leaf contains a group of neighboring candidates from which random viewpoints get selected
while others with similar orientation get deleted" — i.e. a genuine adjacency-based redundancy-pruning mechanism, but
in a different paper, applied post-hoc to already-generated candidates in world space (octree over R³ positions), not
to this paper's own parameter-space method. Worth carrying to branch Q11 (pruning mechanisms): **octree spatial
grouping in R³ = similarity/proximity merging**, one of the few concretely-named pruning mechanisms found across this
pass, precedented in [11] but not implemented or evaluated in this paper itself.

## Ordering (branch Q4)

**None.** Leaves are visited only implicitly via the recursion in Algorithm 1/2; no stable traversal order, ring
numbering, or "next/previous" relation over either leaves or candidates is defined, named, or used downstream. The
final candidate set is an unordered set `V` for the ILP formulation (Eq. 11).

## Seams (branch Q7)

**Two distinct seam types exist in this scheme's geometry, and neither gets an address, a name, or special handling:**

1. **Within a surface**, the quadtree partition is exact and exhaustive (axis-aligned recursive quadrant split of
   `[0,1]²`) — there is no leftover/undefined region, so there is no internal seam problem *by construction*.
2. **Between surfaces**, at the shared boundary curves, and in general at concave/complex topology, the paper's own
   stated limitation (Sect. 7.4, p. 17) is explicit: **"the algorithm does not deal with self-occlusions and collisions
   of viewpoint candidates with the object's geometry. Holes or tight cavities still pose a challenge."** This is
   exactly a seam-type failure (regions where no single surface's orthogonal-offset formula produces a valid, unoccluded
   view) and the paper offers **no addressing mechanism for it at all** — no flagged/degraded address, no fallback rule,
   no coverage guarantee. Sect. 7.4 also states plainly: "the subdivision approach does not explicitly address the total
   coverage of an object" — coverage is empirical (it happened to work for the 3 test models), not guaranteed by the
   addressing scheme. This is a clean negative-finding data point for branch Q16 (infeasibility handling): **no
   precedent here for a "nearest feasible" fallback** — the gap is acknowledged, not solved.

## Variable arity and cross-shape address stability (branch Q12, and RQ3 novelty check, branch Q20)

The three test objects decompose into very different, **automatically-determined** surface counts (Table 1, p. 14):
cylinder → 6 surfaces, spring → 14 surfaces, tangle cube → 452 surfaces. Critically, **this count is an artifact of the
Eck–Hoppe mesh-Voronoi/Delaunay reconstruction algorithm, not a semantic decomposition tied to object identity.** The
paper's own description of the cylinder confirms this (p. 12–13): "Top and bottom are each realized by a single
B-spline surface while the mantle is modeled by **four** surfaces" — not the 2-cap-+-1-mantle decomposition a human (or
our primitive-fitting pipeline) would assign; the mantle is split into 4 quarter-patches purely because the object is
"stretched in one dimension" (cross-section is elongated, not circular) and because of how the quad-layout algorithm
merges the input mesh's Voronoi cells. **`surface_id = 3` on this cylinder has no portable meaning even to a second run
of the same algorithm on a differently-meshed version of the same object**, let alone across object classes — the
paper gives no mechanism, discussion, or claim of a stable per-object-class addressing convention. This directly
supports the branch's Q20 hypothesis: **this paper is one more confirmed data point that cross-shape VIEW-addressing
consistency is unaddressed in the literature** — surface/patch layout is a side effect of mesh reconstruction here, not
a first-class, portable naming scheme, and the paper never claims otherwise (contrast with cross-shape *part*-consistency
work, which this paper doesn't touch at all — no shape correspondence, no canonical part labels, no cross-object
comparison of surface identities anywhere in the text).

Net for Q12: this scheme **keeps neither count nor meaning stable across shapes** — count varies with mesh
complexity/layout algorithm output (6 → 14 → 452), and meaning (which physical part `surface_id=k` denotes) is not
even guaranteed stable for the *same* object re-run through the pipeline.

## Canonical frame (branch Q13)

**No discussion anywhere in the paper.** Confirmed by full-text search: no mention of "canonical," "up-axis," "front,"
"side," "orientation of the object," or any disambiguation convention. The parameter domain `[0,1]²` per surface is
purely local to that surface's own fit; nothing establishes a shared reference frame across surfaces or across objects.
This is a clean negative finding — this paper presupposes and needs no canonical-frame machinery because it never
compares or aligns addresses across different surfaces or different objects; it only ever resolves an address to a
pose *within* one already-fit model.

## Bounded cardinality (branch Q14)

**Two regimes, one bounded, one not:**

- **Recursive/threshold-driven (Algorithm 1, the paper's main mode)**: candidate count `n_vpc` is an *emergent* function
  of the feature functional and threshold, not a guarantee — Table 1 shows `n_vpc` ranging from 36 (cylinder, curvature)
  to 1571 (tangle cube, normal deviation) even *within* one object, purely from functional/threshold choice. No
  cross-object bound is claimed or given.
- **Iterative/budget-driven (Algorithm 2, Sect. 3.2.3, p. 8)**: explicitly designed to give a **hard cardinality
  guarantee** — globally split only the single highest-`E` segment, repeated "until the desired amount of segments is
  [reached]," governed by a user-specified target `t_n`. This is a real, precedented mechanism for "always exactly `N`
  candidates regardless of shape," directly answering the mechanical half of Q14. It says nothing, however, about
  whether those `N` slots carry **stable identity/meaning** across shapes (per Q12's distinction) — nothing in the
  paper suggests they would, since which segments get chosen up to budget `t_n` depends entirely on each object's own
  `E` distribution.

## Discrete/continuous boundary (branch Q15)

This is the cleanest example in this pass of "discrete address names a region, pose is resolved **deterministically**
(not optimized) inside it": the leaf `Ω` is the discrete unit, and Eq. 6 is a closed-form, non-iterative function of
that leaf's center — there is no gradient step, no refinement loop, no optimization anywhere in candidate placement
(confirmed also in the view-quality-scoring note's negative-finding #5: "no local/gradient optimization of pose... the
paper's own future work is precisely to add camera-position optimization"). This is evidence for a variant of the
branch's PAMDP split that is **stronger than "VLM emits `k`, geometry emits optimized `x_k`"**: here, geometry emits a
**deterministic, non-optimized** `x_k` — the middle path the branch asks about (discrete region + continuous
refinement) is realized here in its simplest form, with the continuous part reduced to one closed-form evaluation, not
a search.

## Build cost and when (branch Q17)

Strictly **offline, per-object, one-shot compilation**, assumed as a separate preprocessing phase before selection and
path planning (Fig. 1 pipeline: Virtual Model → Viewpoint Candidates → Selected Viewpoints → Path Planning →
Inspection). Table 1 timings (i7-4790 @ 3.6 GHz, 16 GB RAM, single-core, no GPU, 2020-era):

| Object | Surfaces | SD (subdivision) | RT (visibility raytrace) | NBV (greedy select) |
|---|---|---|---|---|
| Cylinder | 6 | 0.001–12.5 s | 41–62 s | 0.42–1.0 s |
| Spring | 14 | 0.005–79.9 s | 268–647 s | 181–610 s |
| Tangle cube | 452 | 0.008–16.7 s | 1772–3103 s | 235–290 s |

**Ray-traced visibility (RT) dominates cost** and scales with both mesh/candidate complexity, not the subdivision step
itself. No incremental or amortized-online variant is discussed anywhere — the whole pipeline (model fit through
selection) is a batch, offline job per object.

## Liftable? (branch Q18)

**Not directly.** Nothing in this pipeline is available as a named, off-the-shelf library the paper points to for reuse
— the closest is the citation of Eck & Hoppe [7] for the B-spline network reconstruction step, which the paper treats
as prior art to be re-implemented, not a package to import. No mention of Open3D, CGAL, MVTorch, healpy, h3, or any
comparable tool. **Input requirement is a watertight mesh** (or one converted from a point cloud by unnamed "freely
available tools," p. 5 — not validated for our noise regime). This entire technique sits in the "needs machinery we
would have to build" tier, not the "integrable today" tier, for our PLY-based pipeline.

## Benchmark against a uniform sphere? (branch Q19)

**No.** The paper's only comparisons are among its own 5 feature functionals against each other (Table 1, Table 2's
qualitative rubric), never against a uniform/tessellated viewsphere baseline on a shared task or metric. Sect. 2
(Related Work, pp. 3–4) explicitly discusses tessellated-sphere approaches (Tarabanis MVP's spherical parameter space;
Jing's mesh-dilation-then-sample method, "resembles the tessellated sphere") as prior art the paper is positioned
*against*, but no head-to-head quantitative comparison (coverage, candidate count, or viewpoint quality on the same
model) is run. This paper adds one more confirmed absence to the branch's running tally on Q19.

## Aspect graphs / view graphs — engagement level

**None.** Full-text search confirms zero occurrences of "aspect graph," "view graph," "topological," or "visual event"
anywhere in the paper. This is a purely metric/parametric-surface method with no relational/graph-theoretic framing at
all — not in scope, not cited, not compared against. Nothing here bears on branch Q8/Q9/Q10's aspect-graph-specific
arithmetic; those must come from the dedicated aspect-graph sources (e.g. `faugeras-why-aspect-graphs-not-practical.md`
in this same folder).

## Concrete numbers worth citing verbatim (cross-reference with view-quality-scoring note)

- Cylinder: 6 surfaces (2 caps + **4** mantle quarter-patches, not 2+1 — object cross-section is "stretched," non-circular).
  Incidence-angle functional (t=60°): 48 candidates → 22 selected. Area (t=0.3·t_avg): 48 → 24. These sit right at the
  branch Q10 "under ~20 nodes" tractability boundary — close, not comfortably under, and this is for an *elongated*
  cylinder-like shape with a non-round cross-section, not a true circular cylinder.
- Spring: 14 surfaces, 194–482 candidates → 9–13 selected.
- Tangle cube: 452 surfaces, 1325–1571 candidates → 32–38 selected.

## What transfers to our setting

- **The address-as-closed-form-pose-function pattern (Eq. 6)** is the strongest transferable idea for this branch:
  define the address as `(primitive_id, local_surface_coordinate)`, resolve pose deterministically. Directly matches
  Albert's target behaviour (box face → orthogonal offset; cylinder mantle → radial-normal offset) and is a concrete,
  citable precedent for making "orthogonal view of face 3" a first-class, closed-form address rather than a lookup.
- **The recursive-subdivision-into-a-budget (Algorithm 2) mechanism** is a directly reusable pattern for guaranteeing
  bounded cardinality per object regardless of shape complexity, if we need that guarantee for RQ4/Q14.
- **The finding that adjacency, ordering, and seam-handling are all simply absent from a full, working, published
  pipeline** is itself transferable as methodological license: a functioning view-generation-and-selection system does
  not require a topology over its candidates. Strengthens the "flat menu may need no topology" side of branch Q6.

## What does NOT transfer, and why

- **The address-space-*construction* machinery (Eck–Hoppe B-spline network fit) requires a watertight mesh and degrades
  under noise, by the authors' own admission (Sect. 3.1) — untested, unquantified, and not built for our fused,
  defect-laden PLY.** The *addressing pattern* transfers; the *pipeline that produces the address space* does not.
- **No cross-shape address stability is offered or attempted** — surface/patch identity here is an artifact of
  automatic mesh reconstruction, not a portable convention; cannot be cited as precedent *for* RQ3, only as one more
  confirmation that RQ3 is open (see Q20 above).
- **No adjacency, ordering, seam, or infeasibility-fallback machinery exists to borrow** — every one of those branch
  questions gets a negative answer from this paper specifically (as opposed to "not discussed," this paper positively
  demonstrates a working system that never needed any of them).
- **No off-the-shelf implementation** — nothing here is "liftable" in the sense of an integrable library; it is a
  from-scratch research pipeline.

## Section/page map for citation checking

- Abstract, Sect. 1 (Intro) — pp. 1–4.
- Sect. 2 (Related work — Tarabanis MVP, Jing mesh-dilation, Scott per-vertex candidates, **Gronle & Osten octree
  grouping [11]**) — pp. 3–4.
- Sect. 3.1 (B-spline network via Eck–Hoppe [7], noise-sensitivity caveat) — pp. 5–6.
- Sect. 3.2 (Recursive subdivision, Algorithm 1, additivity requirement, shared-boundary-curve C⁰ continuity
  constraint) — pp. 6–8.
- Sect. 3.2.1–3.2.3 (threshold selection Eq. 5, backward-merge modification, **iterative/budget subdivision
  Algorithm 2**) — pp. 7–8.
- Sect. 3.3 (Viewpoint candidate placement, Eq. 6, Fig. 5) — pp. 10–11.
- Sect. 4 (Feature functionals) — pp. 8–11.
- Sect. 5 (Selection formalism, visibility matrix `A`, Eq. 11 set-cover ILP, greedy NBV) — pp. 10–11.
- Sect. 6 (Results, Table 1) — pp. 11–14.
- Sect. 7.4 (Limitations: self-occlusion/collision unaddressed, no coverage guarantee) — p. 17.
- Sect. 8 (Conclusions/future work) — pp. 18–19.

## Assessment for the branch

This paper is the single best-fit precedent found so far for the branch's central positive claim (a surface-parameterized
address can make "the orthogonal view of face 3" native and closed-form, Eq. 6) — but it is equally sharp negative
evidence on almost every structuring question beyond that core pose formula: no adjacency, no ordering, no seam
handling, no cross-shape stability, no canonical frame, no infeasibility fallback, and no off-the-shelf liftability. It
functions well as a **working counter-example to the aspect-graph tradition**: it builds a fully functional
object-anchored addressing-and-selection pipeline with zero relational/graph machinery, which is direct evidence for
the "the agent may not need adjacency at all" side of Q6. Its hardest-hitting limitation for us is structural, not
incidental: the entire address space is generated by fitting a watertight B-spline surface network to a clean mesh,
which is precisely the representation our survey-cloud pipeline does not have and which the authors themselves flag as
degrading under noise.
