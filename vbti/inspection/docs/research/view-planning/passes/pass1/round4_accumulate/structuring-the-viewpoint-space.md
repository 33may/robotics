# Branch 5 — Accumulated Synthesis

## Composing and addressing geometry-conditioned view spaces

**Date:** 2026-08-27
**Role:** branch synthesizer, pass 1 round 4.
**Feeds:** RQ3 (primary), RQ4 (secondary); constrains Branch 4 downstream.
**Inputs:** [`../round1_plan.md`](../round1_plan.md) § Branch 5 ·
[`../round2_map/structuring-the-viewpoint-space.md`](../round2_map/structuring-the-viewpoint-space.md) ·
20 reading notes in [`../round3_reading/structuring-the-viewpoint-space/`](../round3_reading/structuring-the-viewpoint-space/) ·
contract [`../../../geometry-aware-view-planning.md`](../../../geometry-aware-view-planning.md)

---

## 0. Headline

Six claims survive the reading, ranked by how much they change the project's direction.

1. **We are not rebuilding aspect graphs — but not for the reason the field
   remembers.** The size-explosion objection, computed rather than asserted,
   *does not bite* a box or a cylinder. The indexing objection is directional and
   does not apply to forward addressing. What kills the aspect graph for us is a
   fifth objection, from Scott et al. 2003 and absent from the famous 1992 panel:
   **visibility equivalence is not measurement equivalence.** Two viewpoints in
   the same aspect cell can differ arbitrarily in grazing angle. Orthogonality is
   exactly the property an image-topology aspect definition is blind to.
2. **In the classical formalism, "the orthogonal view of face 3" is not a region —
   it is a measure-zero boundary.** Worked arithmetic on a cube: 8 open corner
   regions (positive area), 12 edge arcs, 6 face-on points. The views we want are
   the seams. This is why the framing doc's complaint is a *category* property,
   not a grid-resolution problem, and it dictates the construction direction:
   start from the surface and derive the view; never start from the view space
   and partition it.
3. **"Orthogonal to face 3" as a closed-form native address is precedented twice,
   independently, in industrial inspection** — Magaña et al. 2023 (frustum
   reflected through the feature frame → 8-vertex polytope of valid positions,
   4.1 ms) and Mosbach et al. 2021 (`C = S(u_c,v_c) + n(u_c,v_c)·d`, one
   equation). Both need a clean modeled surface as input. **Neither has ever been
   run on a noisy fused cloud.** That input requirement is the branch's single
   largest inherited risk and it lands squarely on Branch 3.
4. **Adjacency is probably decoration.** Six independent working systems —
   spanning inspection planning, NBV, recognition, RL manipulation and
   VLM-driven manipulation — get from geometry to a used view/action set with
   *zero* topology over candidates. What the systems that need *something*
   actually use is a **priority ranking** ("what to try next if this fails"), not
   a spatial relation ("what is next to this"). Caveat kept visible: none of them
   had a blind orchestrator.
5. **Cross-shape view-addressing consistency has no precedent anywhere in the
   corpus.** Cross-shape *part* consistency is mature. Nobody carries it through
   to naming views. Every one of the 20 documents that processes multiple objects
   through one pipeline still gives each object its own private, meaningless
   indices. RQ3's hard requirement should be deliberately weakened (see §5.3) or
   accepted as genuinely novel work with nothing to lean on.
6. **The uniform-vs-geometry-conditioned benchmark exists — three times — and
   consistently favours geometry-conditioning.** The map's hypothesis that
   "nothing does this" is wrong. What is genuinely unbuilt is the version we need:
   orthogonality as the metric, a noisy survey cloud as input, named per-primitive
   addresses as the space. That is a *stronger* position for building our own,
   because the protocol is precedented and only the objective needs swapping.

---

## 1. Addressing — what is the name of a view?

### 1.1 The taxonomy, populated

The map proposed three classes. The reading confirms them, populates them, and
forces two corrections.

**(a) Index into a global tessellation.** Geometry-agnostic; address is a cell ID.

| Scheme | Address | Address→pose | Size |
|---|---|---|---|
| `{h,v}` (our baseline) | `(h,v)` bin pair, roll fixed | table lookup | 12×3 = 36 |
| HEALPix (Górski 2005) | RING index, or NESTED `base_pixel ∈ [0,11]` + 2 bits per level | closed form, Eqs. 2–18 | `12·N_side²` exactly |
| Hopf/SO(3) (Yershova 2010) | sequence index `i` → (1-of-72 base region, recursive cube path) | `O(log i)` | `72·8^l`; covers **roll** |
| NBV-Net (Mendoza 2019) | one of 14 softmax classes, upper hemisphere, one fixed radius | lookup | 14 |
| Tarbox/Gottschlich (via Scott 2003) | icosahedron subdivision + bistatic point pair | lookup | 5–6 boresight orientations per axis |
| Gospodnetić "space sampling" | sphere sample index — **discarded after repositioning** | n/a | 10/100/500/1000 tested |

**(b) Object-part reference + local offset.** Address names a piece of geometry;
pose is a function of it.

| Scheme | Address | Address→pose |
|---|---|---|
| Magaña C-space (2023) | `(feature, r_s^fix)` | reflect 8-vertex frustum polytope through feature frame → **region** of valid positions, closed form, 4.1 ms ± 2.4 ms |
| Mosbach (2021) | `(surface_id, quadtree_leaf_path)` | `C = S(u_c,v_c) + n(u_c,v_c)·d`, Eq. 6 — deterministic, non-optimized, **one evaluation** |
| Gospodnetić vertex sampling | vertex id | vertex + fixed offset along normal |
| HACMan++ (2024) | `(a_prim, a_loc, a_m)` = what/where/how | learned; `a_loc` is a raw per-episode point index (opaque) |
| OmniManip (2025) | canonical-space `(p, v)`, `v` from principal axes | constrained optimization + bounded local resample |

**(c) Relational node ID.** Meaning defined only by what the node borders.

Only the classical aspect graph / VSP genuinely occupies this class. **Correction
to the map: view-GCN is not class (c).** Its node identity is an explicit metric
camera coordinate assigned at construction; the graph is a *derived*, learned
structure over already-meaningful addresses. Reading note verified this against
the primary text (Sec. 4).

**(d) A fourth axis the reading surfaced.** Masson et al.'s PAMDP type signature
`A = ⋃_{a∈A_d}{(a,x) | x ∈ X_a}` is orthogonal to (a)/(b)/(c): it is a *shape of
address*, not a source of meaning, and its distinguishing feature is **per-action
variable parameter dimensionality** (explicitly contrasted against Hauskrecht's
factored hybrid actions in §6). A face address carrying a 2D in-plane offset and
a cylinder address carrying a 1D orbit angle coexist in one formalism, natively.

**MVTN fits none of them.** Its only address is an unconstrained array index into
a regression output; nothing in the architecture or training constrains slot `i`
to carry stable geometric meaning, and the paper never tests or claims it does.

### 1.2 The discriminating test — can a blind orchestrator predict what it gets?

- **(a) yes, mechanically.** HEALPix resolves cell→direction by closed-form
  equations at billion-pixel scale. But the prediction is "a direction on an
  abstract sphere," carrying no statement about the object.
- **(b) yes, semantically.** `face_3/orthogonal` predicts *both* the pose and its
  relationship to the surface. Mosbach's note calls this "the strongest form of
  predictability among schemes read" — because the address was designed to be
  predictable, with orthogonality baked into the pose formula.
- **(c) no.** An aspect node names "the set of viewpoints from which the same
  features are visible." That does not name a pose, and — per §3.2 below — the
  pose we want is not even inside a node.

### 1.3 The upgraded baseline nobody in the project has considered

HEALPix + Hopf is the mathematically correct version of `{h,v}`: equal-area cells,
no metric distortion, provable dispersion bounds, closed-form `O((log i)/d)`
neighbour queries, **and it covers roll**, which `{h,v}` today does not (all views
forced vertical per the framing doc). Build cost is effectively zero; `healpy` and
Yershova's own C++ library are off the shelf.

It does not solve RQ3 — it is object-blind, so "orthogonal to face 3" remains a
nearest-neighbour lookup. But it sharpens what RQ3 must beat. **Benchmarking
against naive `{h,v}` proves nothing; benchmarking against a properly constructed
uniform grid proves the geometry-conditioning claim.** This is the single most
actionable methodological finding in the branch (see §8).

### 1.4 Does the address need to encode pose? (No.)

Four independent registers of evidence for a symbolic address with deterministic
downstream pose resolution:

- **HEALPix**: closed-form `g(k) → x_k`, no lookup table, two decades of
  production use at billion-cell scale.
- **Mosbach Eq. 6**: geometry emits a *deterministic, non-optimized* `x_k` — one
  closed-form evaluation, no gradient step, no search. This is a *stronger* claim
  than PAMDP's "optimized `x_k`."
- **Ikeuchi 1992** (inside the Faugeras panel): the Aspect-Classification /
  Linear-shape-Change split, with two design rules — "aspects should be defined
  based on the features that will be used in the corresponding LC function" and
  "the resolution of aspects should be based on the robustness of the LC
  function" — plus the extreme case: *"in the extreme case in which the LC
  function is continuous in all viewing directions, it is not necessary to have
  an AC phase at all."* A 1992 precedent for placing the discrete boundary at the
  geometric discontinuities rather than on a default grid. Consequence in §5.4.
- **OmniManip**: a working system where the VLM commits to one of ≤6
  geometry-derived candidates and constrained optimization resolves the pose.
  68.3% closed-loop vs. 51.7% open-loop on 12 real tasks.

**Two corrections that must not be smoothed.** Masson's PAMDP licenses the *type
signature*, not our architecture: in PAMDP both `π^d(a|s)` and `π^a(x|s)` belong
to the **same learned policy**, and Theorems 4.1–4.4 certify an *iterative
optimization* `P-UPDATE`, not a hand-coded deterministic rule. And **HACMan++
does not test the split either** — all three tuple components come from one
network forward pass; the only deterministic-geometry precedent in the paper is
Poke's pre-contact standoff *direction* along the estimated surface normal
(`d1 = 0.04 m`), while the in-plane orientation stays learned.

Net: the split is well-precedented as an *engineering pattern* and formally named,
but nowhere formally certified in the heterogeneous form (blind agent picks `k`,
geometry module computes `x_k`) our contract proposes.

### 1.5 Ordering — three different things called by one name

1. **Prefix-quality ordering** (Hopf/HEALPix): the sequence is ordered so any
   *prefix* is well-dispersed — stop early, still have coverage. Consecutive
   indices are **not** spatially adjacent. Yershova's base 72-point ordering was
   hand-picked and optimal ordering is left open (§5.3, §7).
2. **Hierarchical path ordering** (HEALPix NESTED): child ID = parent ID + 2 bits;
   refine/coarsen is one bit-shift. Genuinely path-like and LLM-legible. Caveat:
   crossing a base-pixel boundary needs a separate algorithm (Wandelt/Hivon/Górski
   1998) that HEALPix's own paper cites but does not give.
3. **Relevance ranking** (OmniManip): an LLM scores VLM-generated captions of
   candidate axes, producing an ordered candidate list the RRC loop walks.
   **This is the only ordering in the entire corpus that an agent actually
   consumed, and its meaning is task relevance, not adjacency.**

Negative evidence is strong. Gospodnetić: "automatically created viewpoints
exhibit no particular pattern in spatial distribution" (§9.4); NBV output order is
greedy-coverage order, an artifact of the heuristic. Only *human experts* produced
spatially patterned sets — larger (44 vs. 23 candidates on the Gear) but organized
into turntable-replication clusters, because experts implicitly optimize for
mountability and traversal. No automated method in the corpus recovers that. The
aspect-graph literature has **zero** ordering convention (confirmed independently
in two documents).

---

## 2. Relating — what is the neighbour of a view?

### 2.1 Complete inventory of adjacency definitions found

| # | Definition | Source | Computable on our PLY? | Nameable? |
|---|---|---|---|---|
| 1 | Share a visual-event boundary in the VSP | Plantinga & Dyer §3 | No — needs exact polyhedral equations | Yes in principle |
| 2 | k-NN on camera-coordinate Euclidean distance | view-GCN Eqs. 1–2 (k=2 for 12 views, k=3 for 20) | **Yes**, `cKDTree`, afternoon's work | Weakly |
| 3 | View-similarity below a threshold (region-grow, distinctness-preserving stop) | Cyr & Kimia | Yes, given any view-similarity function | Weakly |
| 4 | Cell neighbour in a multiresolution tiling | HEALPix NESTED; Hopf `O((log i)/d)` | Yes, closed form, no stored graph | Yes |
| 5 | Octree-leaf co-membership + orientation angle < φ | Gronle & Osten via Gospodnetić §4.2.2 | **Yes**, any point set | Weakly |
| 6 | Shared visible surface region (visibility/measurability matrix) | Tarbox & Gottschlich via Scott §5.1 | Yes but expensive (3.81 s/candidate) | Yes |
| 7 | Closest-point distance < 0.05 × avg bounding-sphere radius | StructureNet §4 | **Yes**, directly liftable to fitted primitives | Yes |
| 8 | C⁰ shared boundary curve between surfaces | Mosbach §3.1 | Only at model-fit time; never exposed | Yes |
| 9 | Learned from observed view transitions | Seibert & Waxman 1989 | **Not read** — catalogued only | — |

Two findings inside this table matter more than the table.

**The exact definition is a trap.** Plantinga & Dyer's worked tetrahedron: region
`A` (1 face visible) and region `ABC` (3 faces) **are** adjacent — connected
through the point P — while `AB` and `AC` (2 faces each) are **not** adjacent,
even though P naively looks like it borders both, because faces are never visible
edge-on. Even a fully worked-out, exact adjacency definition produces non-obvious,
easy-to-get-wrong graphs. That is a direct warning against handing an LLM a
"next/adjacent" relation derived from first principles.

**Cheap beats learned.** view-GCN's own ablation: learned edge affinity beats
binary-kNN and Gaussian-kNN by only **0.2–0.7 pt**. A hand-specified rule gets
~90% of the benefit. And kNN adjacency degrades gracefully on irregular/perturbed
layouts (85.3/89.5% vs. clean-configuration numbers), which is the existence proof
that a proximity relation computed *post hoc* over whatever addresses a
geometry-conditioned scheme produces is at least not obviously broken.

### 2.2 Does the agent need adjacency at all?

**Evidence for "no" (six independent systems, five subfields):**

- **Mosbach** — a full, published, evaluated view-planning pipeline gets from
  geometry to a selected viewpoint set with a flat unordered candidate set `V`, a
  binary visibility matrix and a set-cover ILP. No graph, no neighbour relation,
  no "next." The paper positively demonstrates a working system that never needed
  any of them.
- **Gospodnetić** — same across all five surveyed families.
- **NBV-Net** — 14 one-hot classes, no modeled neighbour relation of any kind,
  drives unseen-object reconstructions to 80–93% coverage.
- **PAMDP** — `A_d` is a flat unordered finite set throughout; two working RL
  domains never needed any relation between `kick-to`/`shoot-goal` or
  `run`/`hop`/`leap`.
- **HACMan++** — permutation-invariant point set by construction, 73% real-robot
  success over 6 geometrically unrelated objects.
- **OmniManip** — the ordering that mattered was a priority ranking over a flat
  candidate set; the agent never traverses anything.

**Evidence for "yes":** our current agent navigates `{h,v}` by incrementing bins.
That is an artifact of a uniform grid being the only structure available, not a
demonstrated requirement. view-GCN's ablations show adjacency helps *feature
aggregation for recognition*, a different job.

**Verdict, stated plainly as the branch was instructed to:** most of this branch's
aspect-graph inheritance on adjacency is decoration for our use case. What the
successful systems substitute is a **priority ordering** or a **coverage-driven
selection loop**. The useful relation over addresses is *"what to try next if this
fails,"* not *"what is next to this in space."*

**Caveat that must stay visible:** not one of these systems had a *blind*
orchestrator that must build a mental model of a space it cannot see. The evidence
says systems work without adjacency; it does not say a blind LLM navigating a menu
works without adjacency. That question is Branch 6's, and this branch cannot close
it.

### 2.3 Seams

- **One positive precedent.** HEALPix bakes the seam into the address: *"All pixel
  indices carry a prefix… which identifies which base-resolution pixel they belong
  to."* The 12-base-pixel boundary is not hidden, it is the leading field. A
  seam-aware scheme can still be a single flat namespace as long as the seam ID is
  first-class.
- **Counter-evidence to "every tiling has a bad seam."** HEALPix is equal-area by
  construction; Yershova explicitly contrasts it against Layered Sukharev's
  cube-projection, which produces up to **4× cell-volume disparity**. H3's
  12 pentagons are a property of *that* tiling, not a law.
- **The hard negative.** For a cube: 8 open corner regions (positive area, a real
  camera can sit inside one), 12 edge arcs, 6 face-on points — total 26. **The
  views we want are the measure-zero elements.** Plantinga & Dyer are ambiguous
  about whether those count as nodes at all. Either reading is bad: a knife-edge
  node, or not a node.
- **Universal silence on primitive-boundary seams.** Mosbach states it plainly
  (§7.4): "the algorithm does not deal with self-occlusions and collisions… Holes
  or tight cavities still pose a challenge," and "the subdivision approach does not
  explicitly address the total coverage of an object." No addressing mechanism, no
  flagged/degraded address, no fallback. Magaña, the aspect-graph corpus, view-GCN,
  MVTN, HACMan++, NBV-Net and OmniManip are all silent.

**For our design:** edges, corners and the box↔cylinder junction need an explicit
decision — either they get an address (an `edge_(i,j)` class with a
bisector-normal view) or they are declared out of the action space. **No precedent
tells us which.** This is ours to decide and to defend.

---

## 3. Tractability — did aspect graphs die of something we would catch?

### 3.1 The arithmetic, done (Q8)

**Provenance correction, first-class.** The paper at DOI `10.1007/BF00054919` is
**Plantinga & Dyer, "Visibility, Occlusion, and the Aspect Graph," IJCV
5(2):137–160, 1990** — not Gigus, Canny & Seidel, who wrote a different paper
(TPAMI 13(6):542–551, 1991, `O(n⁶ log n)` construction). The bounds cited in the
map are correct; the attribution was not. Canny is genuinely in the lineage: the
Θ(n⁶)/Θ(n⁹) adversarial lower-bound constructions are credited to him
(1987, personal communication) and he caught an earlier `O(n⁴)` error.

| | convex | non-convex |
|---|---|---|
| orthographic | Θ(n²) | Θ(n⁶) |
| perspective | Θ(n³) | Θ(n⁹) |

**The explosion is a non-convex-only phenomenon.** Edge-edge-edge occlusion —
the mechanism behind Θ(n⁶)/Θ(n⁹) — *cannot occur on a convex solid at all*
(§4 intro). A 6-face box is convex. A prism-approximated cylinder is convex.
Both are categorically excluded.

- **Box.** Generic convex hexahedron: `2n(n−1)` intersection points →
  `F = n²−n+2 = 32` regions. But a box's antiparallel face pairs collapse 6 normals
  to **3** great circles (the paper explicitly calls parallel faces "highly
  degenerate" and size-reducing) → octahedral subdivision, `V=6, E=12, F=8`.
  Counting every dimension: **8 corner regions + 12 edge arcs + 6 face points = 26**,
  matching the independently known "26 topologically distinct views of a cube."
  Under perspective, the 3 parallel plane-pairs cut ℝ³ into **27** cells (3×3×3),
  far under the generic `O(n³)`.
- **Cylinder.** Plantinga & Dyer explicitly do not treat curved surfaces (flagged
  verbatim as future work). Their only cylinder-adjacent result is their own
  degenerate case: *"the class of n-sided approximations to a cylinder; for
  polyhedra in this class the VSP has size Θ(n)"* — **linear, the framework's best
  case, not its worst.** A 36-facet prism → O(36), not 1296, and nowhere near
  36⁶ ≈ 2.2×10⁹. **The genuinely authoritative curved-object paper is one DOI away
  (`10.1007/BF00054918`, "Computing exact aspect graphs of curved objects: Solids
  of revolution," same IJCV issue) and was not read.** Flagged as follow-up.
- **Assembly — and here the story turns.** The first non-convex case among our
  three targets. Plantinga & Dyer give no assembly example. But Eggert, Bowyer &
  Dyer (1992 ISPRS, read as substitute for the unobtainable Bowyer & Dyer 1990
  survey) do: a **two-block articulated assembly** (base 2×4×2, appendage 1×2×1,
  one translational DOF) has **206–461 aspect-graph nodes** per configuration
  interval (reported: 449, 461, 461, 451, 459, 206, 319, 319, 315, 319, 311 across
  8 intervals). **A trivial two-primitive assembly already produces several hundred
  nodes under the classical global definition.**

Supporting figures from the same source: a non-convex plate-with-hole gives 24
sphere regions (4 visually distinct) vs. **81** regions in full 3-D space — and the
3-D model finds views *through the hole* the sphere model cannot reach, which is
independent 1992 evidence that a sphere-only action space provably misses views.
A squash-shaped implicit surface gives 11 regions. Yang/Marefat/Johnson report a
simple non-convex "step" has **71** characteristic views.

One more load-bearing line: *"The reason for the larger size is the higher
dimensionality of viewpoint space and not the change from orthographic to
perspective projection"* (§4.2). Our camera lives on a fixed working-distance
shell — a 2-D viewpoint space — so the Θ(n²)/Θ(n⁶) regime is the relevant one even
if the object weren't convex.

**Contradiction to keep visible:** the box/cylinder arithmetic exonerates the
bound; the assembly datum indicts the construction. Both are correct — they measure
different things (per-primitive vs. global whole-object). The resolution both
sources independently propose is the one we already chose: **hierarchical,
per-primitive decomposition.** Plantinga & Dyer's own 1987 conclusion:
*"construct a hierarchical VSP by constructing the VSP for each level of a
hierarchical representation of the object."* Pentland, 1992: restrict to
"component parts, subassemblies or critical features" — "the resulting aspect
graphs are extremely simple." Dyer, 1992: abandon the global whole-object catalog
for feature-centric local cells.

### 3.2 Which causes of death transfer? (Q9)

| Cause | Transfers to action-addressing? | Evidence |
|---|---|---|
| (i) **Size** | **No** for individual convex primitives; **partly** for assemblies *under the global definition*; fully mitigated by per-primitive decomposition | §3.1 arithmetic; three independent sources converge on the same fix |
| (ii) **Non-computability on curved / noisy input** | **Yes — the most serious inherited risk** | Plantinga & Dyer explicitly exclude curved surfaces; Scott §5.2: the quantization "strictly fails" when the object goes discrete→continuous; every event surface in the corpus comes from exact equations of an exact object |
| (iii) **Indexing** ("which node am I in?") | **No — and this is the branch's headline** | Faugeras p.213 states the inverse direction explicitly: observed image → match against stored node silhouettes, worse when object identity is unknown. Our address is forward-only: label → deterministic pose. We never infer an address from pixels |
| (iv) **No compelling use case** | **Inverted** — the use case is the whole point | — |
| (v) **Visibility ≠ mensuration** (*new, Scott §5.2, absent from the 1992 panel*) | **Yes, and it is fatal independent of (i)–(iii)** | *"viewpoints with the same visibility (i.e., the same aspect) are by no means equivalent for sensing purposes."* Two viewpoints inside one aspect cell differ arbitrarily in grazing angle and standoff — the exact quantities our task optimizes |

Corroboration for (iii) from three angles: every application in Eggert/Bowyer/Dyer
§7 is recognition, pose estimation, or pre-scripted rendering-path replay — never
"where should the camera go next"; view-GCN spends 73.4M parameters and 9 GPU-hours
precisely on the inverse direction; HACMan++'s flat `K×N` action space shows no
combinatorial blowup across arbitrarily varied real geometry, because it names a
*place to act*, not an *appearance to recognize*.

**So: the size argument alone would not have stopped us. Scott's equivalence-class
objection does.** What we should build is closer to Pentland's part-restricted,
face-based, range-native structure — the fix the panel itself converged on — but
with equivalence classes defined by *measurement quality*, not image topology.

### 3.3 The smallest structure expressing the target behaviour (Q10)

- Box: **6** face-orthogonal addresses (+12 edges +8 corners = 26 if seams are
  addressed — the coincidence with the aspect-graph count is not accidental; the
  VSP elements of a box *are* its face/edge/corner views).
- Cylinder: 2 caps + K orbit stations. Mosbach's real (elongated) cylinder was
  decomposed by the automatic quad layout into **6** B-spline surfaces —
  2 caps + **4** mantle quarter-patches, not the 2+1 a human would assign — and
  yielded 48 candidates → **22–24 selected**. That sits *at* the ~20-node
  tractability boundary, not comfortably under it. OmniManip's principal-axis menu
  is ≤6 directions + 6 refinements.
- Assembly: 6 + (2+K) + seam addresses under per-primitive addressing;
  **206–461** under the classical global definition (do not imitate).

**A per-primitive scheme lands at roughly 8–20 addresses for our object class** —
exactly the framing doc's target band — provided we address per primitive, not
globally, and choose the cylinder's K deliberately. Two hard-cap mechanisms exist
(Mosbach's Algorithm 2 budget `t_n`; MVTN's fixed `M`); both give stable *count*,
neither gives stable *meaning*.

### 3.4 Pruning mechanisms that survive (Q11)

| Mechanism | Source | Number attached | Maps onto "one region at a time"? |
|---|---|---|---|
| Entity-based restriction to entities of interest | Yang/Marefat/Johnson 1998 (abstract only) | — (thesis confirmed, mechanism inaccessible) | **Directly** |
| Feature-type restriction (occluding contour only) | Seales & Dyer rim-appearance model | captures **~25% of all events** | By analogy: keep only orthogonality-relevant structure |
| Similarity merging / region growing | Cyr & Kimia | node count capped by sample count, never by object complexity | Yes |
| Octree-leaf orientation pruning | Gronle & Osten | — | Yes |
| **Scale-space merging** | Eggert/Bowyer/Dyer §8 | worked before/after at 10° vs. 52° visual arc | **Yes — the "viewer size" reading is literally "merge views a camera at this standoff cannot distinguish"** |
| Hierarchical refinement | HEALPix, Hopf, Mosbach quadtree, view-GCN coarsening | view-GCN: 1→2→3 levels improves, **3→4→5 degrades** | Weakly — deep hierarchy is contraindicated at our scale |
| Feasibility pruning | Tarbox morphological erosion; Gospodnetić pre-filter | — | Yes (see §6.1) |

The scale-space mechanism is the most mechanistic and the least known: three
named interpretations of a blur parameter σ — viewer size (a finite-radius
observer sphere in viewpoint space merges sub-sphere cells), image feature size
(sub-pixel features unrepresentable → cells merge), object detail (smooth the 3-D
surface directly). It is the clearest primary-literature answer to "merge views
that are indistinguishable at the resolution that matters."

Also worth carrying: Yang/Marefat/Johnson's *"combine and contract previously
constructed EAGs in a database to generate EAGs for new objects"* is a named 1998
precedent for compositional per-primitive reuse — build box-EAG and cylinder-EAG
once, compose for the assembly. Cite hedged: the full text was unobtainable and
the mechanism is inaccessible, including exactly how composition handles the seam.

---

## 4. Consistency across shapes

### 4.1 Variable arity — stable count vs. stable meaning (Q12)

The corpus draws this distinction repeatedly and lands almost entirely on one side.

**Stable count, arbitrary meaning:**
- **MVTN** — `M` fixed at design time (4/6/8/10/12/20 tested); geometry moves only
  *where* the M cameras point. Slot identity never claimed or tested stable. And
  `MVTN-direct` (unanchored absolute-angle regression) **collapses views onto each
  other at M > 4** — only the variants anchored to a pre-spread canonical base
  avoid it. Direct evidence for anchoring every address to a fixed canonical role.
- **HACMan++** — N=400 object points + 1000 background, K=5, identical for a bowl,
  a cup, a tennis ball, cardboard, a toy car and a Rubik's cube. Point 137 of a
  bowl has *no* relation to point 137 of a cup. Generalization (95.2% unseen
  instance, 84.9% unseen category) comes from the **network**, not the address.
- **Mosbach Algorithm 2** — budget `t_n`, globally split the highest-`E` segment
  until you have exactly N. Which segments get chosen depends entirely on each
  object's own `E` distribution.

**Explicitly anti-bounded:** Gospodnetić's requirements R2/R4 want redundancy to
*scale with* object complexity — and the numbers deliver it: the same method gives
~33 optimal viewpoints on the Gear, ~12 on the Hirth, ~23 on the Crankshaft.
Cardinality is a dependent variable of geometry **by design**. Cyr & Kimia's
aspect count is an emergent output of a threshold, actively cutting against any
guarantee.

**Learned soft consistency — and a correction to the map.** StructureNet's decoder
slots are **not** semantic-label-indexed at inference. The `np=10` slate's labels
are predicted per-slot per-shape; the encoder is permutation-invariant (max-pool);
slot↔ground-truth correspondence for the loss is a **Chamfer-distance linear
assignment**, not a label lookup. What stability exists is an emergent statistical
regularity over thousands of PartNet annotations, reproducible only with an
equivalently labeled corpus per object class — which we neither have nor want,
given the contract blesses deterministic primitive-level rules.

**The most useful calibration number in the branch** comes from MVTN §10.3:
per-instance views 91.0%, **class-averaged views 90.6%**, generic fixed views
89.0%. Most of the benefit of geometry conditioning is captured at *category*
level, not per instance — direct support for primitive-class deterministic rules
(box → face normals, cylinder → orbit) over per-instance continuous optimization.

### 4.2 Canonical frame (Q13)

- **The PCA negative is confirmed and quotable.** Fu et al. §2: *"principal axes
  are not robust for pose normalization of many models… PCA does not fully resolve
  the orientation problem, but simply reduces the orientation candidate set to six
  candidates."* Symmetry axes are "more robust" but still only narrow. Both are
  related-work claims citing Kazhdan 2003 — **not measured in the paper** (no PCA
  baseline appears in Table 1).
- **Fu solves 1 of 2 rotational DOF**, stated twice ("two to one degrees of
  freedom"). **Rotation about the up-axis — front/side disambiguation — is never
  addressed anywhere in the paper built specifically for canonicalization.**
- For us the up-axis is free: the object rests on the table. NBV-Net makes exactly
  the same argument for its upper hemisphere ("objects to be reconstructed are
  usually placed over a surface, for example a table").
- **Two of Fu's failure modes are ours.** ~20% of failures are multiply-plausible
  orientations, *"mostly box-like shapes (refrigerator, cabinet)"* — precisely our
  object class. And 5% are attributed to mesh defects (holes, noise, one-sided
  faces) **on curated repository models**; our fused PLY is a harder input class,
  and every step of their pipeline (convex hull, coplanar merging, 32-view
  visibility rendering) is mesh-native with no point-cloud variant offered.
- **OmniManip silently assumes the problem away.** Its canonical frame comes from a
  trained pose network; rotational-symmetry ambiguity (a cylinder's free rotation
  about its own axis — exactly our case) is never surfaced, because their objects
  (teapots, drawers, jars with handles) are asymmetric enough for the training
  distribution to resolve it implicitly.

### 4.3 Bounded cardinality (Q14) and the cross-shape gap

Mechanisms delivering stable count: MVTN, Mosbach Alg. 2, HACMan++, HEALPix
(`12·N_side²`, exact, by picking one integer). **None delivers stable per-slot
meaning, and HACMan++ shows the two are in tension, not both free.**

**The cross-shape view-addressing gap is now confirmed across the entire corpus,
independently, ten times over.** Aspect graphs (every example is its own private
numbering); Cyr & Kimia (18- and 64-object databases pipelined identically, each
object's aspects unrelated to any other's); Gospodnetić (three objects planned
independently); Mosbach (`surface_id=3` on a cylinder has no portable meaning even
to a second run on a differently-meshed version of the *same* object); Magaña
(single-feature scoped); view-GCN (comparability outsourced to dataset
pre-alignment); MVTN (slot semantics untested); HACMan++ (opaque positional
addresses, naming sidestepped entirely); StructureNet (statistical, corpus-derived);
Fu (single-object-instance scoped).

**Cross-shape *part* consistency is a mature literature. Nobody has carried it
through to naming *views*.** RQ3's hard requirement is genuinely novel — which
also means there is nothing to lean on.

---

## 5. Interface with the rest of the project

### 5.1 Where the discrete/continuous boundary sits (Q15)

Supported and precedented, with hard operational limits.

- **Strongest form** (Mosbach): continuous part reduced to one closed-form
  evaluation, no search at all.
- **Region form** (Magaña): the address names an 8-vertex polytope an optimizer
  could climb inside — **but only in translation.** Orientation is fixed or
  discretely sampled, with an explicit "Discretization without Interpolation"
  caveat (§VI-B3): arbitrary orientations *between* sampled ones are **not
  guaranteed** to lie inside the resulting manifold.
- **Cost cliff, measured.** CSG-intersecting discretized orientations: 27 combos →
  15 s; 75 → 40 s; 363 → **720 s**. Count grows 13.4×, time grows **48×**. Volume
  gain from the finest discretization: 0.9995–0.9999. **Keep orientation fixed or
  coarse; never buy orientation tolerance with CSG.**
- **Ikeuchi's design rule gives a non-obvious, actionable consequence.** Put the
  discrete boundary where the downstream pose function stops being smooth. For a
  box that means 6 addresses (one per face, discontinuous at the edges). For a
  cylinder's lateral surface the function is smooth all the way round — so **by
  Ikeuchi's own extreme case the cylinder orbit arguably needs no discrete
  addressing at all**, only a continuous (or coarsely binned) angle parameter.
  That is exactly PAMDP's per-action variable parameter dimensionality.

**The counter-position, read against its own numbers.** GenNBV claims hand-crafted
discrete action spaces are the bottleneck and moves to free 5D. Three things blunt
it: (1) its own **non-learned Uniform Hemisphere** baseline reaches 89.71%/92.90%
FCR vs. GenNBV's 98.26%/97.12%; (2) decomposing Table 1's half-swap rows, the free
action space alone buys **+2.90 pp AUC / +4.02 pp FCR** while the state
representation buys +1.99/+0.59 — comparable, not dominant; (3) its reward has
**no orthogonality term at all**, so it never tests what we care about. And its own
text supplies the strongest pro-discreteness argument in the corpus: prior
limited-space agents *"do not consider collision avoidance since their search
space, like hemisphere, is by-design safe."* **A bounded discrete address space is
a free feasibility guarantee** — one GenNBV had to re-earn with a learned collision
penalty. Its uni-source ablation adds the kicker: action-history-alone is the
*weakest* input (69.53% AUC). What the agent knows about the surface dominates how
big its action space is.

**One live caution for the optimization track.** MVTN §4.5: directly SGD-optimizing
azimuth/elevation against the task loss produced 74.95–76.89 — **below the
random-noise baseline of 78.52** — and far below the learned regressor's 82.8.
Naive direct optimization of view parameters is fragile. Magaña's formulation (a
closed-form *region* to optimize inside) is a better-posed alternative than
optimizing a free camera angle against a surrogate.

### 5.2 Does the address survive infeasibility? (Q16 → Branch 4)

Four distinct behaviours found. **None of them is "flagged, degraded, still
addressable."**

1. **Repair** — Gospodnetić §4.1.2: ray-cast along the direction to the surface or
   convex hull, move the focal point there, optionally reorient to `d = −n`
   (functionally "snap to the orthogonal view"). If the ray misses, *"the viewpoint
   remains unchanged"* — left in the list, uncorrected, to fail later.
2. **Delete** — Gospodnetić §5.1 pre-filters candidates failing focusing-distance
   or incidence-angle before ray tracing; Tarbox erodes pose-error-fragile
   viewpoints out of the address space morphologically.
3. **Ordered fallback then terminal failure** — OmniManip Algorithm 1: walk the
   ranked list; one bounded local resample (6 perturbations) on "refine"; then
   `Task Failed`.
4. **Constrain the ground truth upstream** — NBV-Net bakes collision-freedom into
   the label-generation argmax; at test time a repeated prediction is used only as
   a stopping signal, with no fallback at all.

Explicit gaps: Magaña defers collision entirely to a downstream per-pose check
(§VII-E3, *"an explicit proof for collision must be performed in the last step"*);
Mosbach admits self-occlusion/collision are unhandled (§7.4); HACMan++, PAMDP,
MVTN, view-GCN say nothing.

**For a blind orchestrator, deletion is the dangerous option** — the menu changes
shape between turns, which is exactly the instability a language model handles
badly. There is no literature cover for the alternative. Hand to Branch 4 with that
flag attached.

### 5.3 Build cost — a real spectrum with numbers (Q17)

| Profile | Cost | Sources |
|---|---|---|
| Closed-form, online, free | address↔direction by equation | HEALPix Eqs. 2–18; Hopf `O(log i)` gen, `O((log i)/d)` neighbour |
| Milliseconds per address | **4.1 ms ± 2.4** (𝒞₁), **5.8 ms ± 3.2** (𝒞₃), 0.7–1.32 s occlusion ray-cast, 200 ms multisensor | Magaña |
| Seconds per candidate set | "magnitude of seconds" for every family | Gospodnetić |
| **Hours, offline, per object** | **3.81 s per candidate** for ray-traced visibility → **~1–1.6 h/object** at 1000–1500 candidates; NBV select <10 s | Gospodnetić; Mosbach RT 41–3103 s |
| Hundreds of CPU-hours per dataset | **~200 h** exhaustive-search label generation → 1.9 s inference | NBV-Net |
| GPU-days per object class | 32M PPO iters / ~24 h / V100 / 256 envs / 256 objects | GenNBV; view-GCN 3 h + 6 h; MVTN per-dataset training |
| Per-inference recomputation | one forward pass over ~1400 points, every timestep | HACMan++; OmniManip (multiple VLM calls, no timings reported) |

Our loop is survey-then-inspect in one session. The profiles that fit are the
closed-form/millisecond end and per-inference recomputation. **The ray-traced
visibility matrix — the component everyone uses to score coverage — does not fit,
at ~1–1.6 h/object.** If coverage scoring is needed in-session we must replace it
with an analytic primitive-level check (plausible for planes and cylinders,
unprecedented in the corpus).

Scott §1 gives the structural precedent for the split without numbers: inspection
is *"an easier problem, having the advantage of a preexisting model… collected
off-line in a suitable data structure, allowing different tasks to be planned
quickly online."*

### 5.4 Liftability (Q18)

**Available today:** `healpy` (note: named in common knowledge, *not* in the 2005
paper); Yershova's C++ SO(3) sampling library; `scipy.spatial.cKDTree` /
`sklearn.NearestNeighbors` for view-GCN-style adjacency; Open3D for octree/normals;
MVTorch (`github.com/ajhamdi/mvtorch`); Gospodnetić's own tool
(`gitlab.itwm.fraunhofer.de/gospodnetic/vob`, CC BY 4.0).

**Reimplementable in an afternoon:** Magaña's C-space reflection (pure
translate/reflect/scale on 8 vertices); StructureNet's adjacency threshold; the
octree-leaf orientation prune; Mosbach's Eq. 6 applied to fitted primitives; Fu's
32-view color-ID visibility score.

**Real projects:** Eck–Hoppe quad remesh + B-spline network + fairing; Magaña's
full CSG multi-constraint stack.

**Not liftable at all:** view-GCN (73.4M params, needs class labels), MVTN (needs
a labeled corpus of similar shapes), GenNBV (Isaac Gym + 256 GT meshes + 24 GPU-h),
StructureNet (PartNet-scale annotation), Fu (345 hand-labeled meshes), HACMan++
(no named backbone; full RL loop).

**The load-bearing pattern in this table:** *every closed-form addressing scheme in
the corpus takes a clean modeled surface as input* — a CAD feature frame, a
watertight mesh, a B-spline network, a convex hull. The only schemes consuming raw
noisy clouds directly are the **learned** ones (HACMan++ on 4 fused Kinects,
StructureNet's ScanNet abstraction, MVTN on ScanObjectNN) — and they give up
nameable addresses in exchange. **Nameable address vs. noisy-cloud-native input is
the sharpest unresolved tension in the branch**, and closing it is Branch 3's job.

### 5.5 The benchmark question, corrected (Q19)

**The map's hypothesis was wrong.** Three genuine head-to-head comparisons exist,
all favouring geometry-conditioning.

1. **Gospodnetić et al. 2021 — the closest thing to what we want.** Uniform sphere
   sampling vs. vertex sampling vs. patch sampling, ONE shared downstream selection
   (greedy NBV over a ray-traced visibility matrix), ONE metric family (% coverage
   + viewpoint count + inadmissible-candidate rate), three CAD objects, **292
   candidate sets**. Space sampling loses on every axis: inadmissible-candidate
   rates **61.9% (Gear), 75.3–77.0% (Hirth), 26.5–33.3% (Crankshaft)** vs.
   **15.9–26.6%** for patch sampling; worst run-to-run consistency (Fig. 10). The
   paper's verdict: *"Space Sampling methods show no benefits and their use is not
   recommended."* **Our `{h,v}` sphere is structurally exactly that family.**
2. **MVTN** — matched renderer and backbone: fixed spherical ViewGCN* 93.0% vs.
   learned MVTN 93.8% at 12 views (Δ+0.8); 93.3 vs. 93.5 at 20 (Δ+0.2). Margin
   shrinks as view count grows. Metric is classification accuracy.
3. **OmniManip Table 3** — geometry-conditioned (principal-axis) vs. uniform SO(3)
   sampling, same task, two metrics: Recycle Battery **80% / 1.7 iter** vs.
   **50% / 1.8**; Pour Tea **70% / 1.8** vs. **30% / 3.4**. Off-domain
   (manipulation directions), but structurally the exact comparison.

Partial: GenNBV Table 1 compares Uniform Hemisphere (82.91/89.71) against free 5D
learned (91.19/98.26) — uniform vs. *free continuous*, not uniform vs.
geometry-conditioned discrete. Nobody there compares a geometry-conditioned discrete
space against either.

**Confirmed absences.** No comparison anywhere uses **orthogonality-to-surface** as
the metric. None runs on a **noisy survey cloud** (Gear/Hirth/Crankshaft,
ModelNet40, Houses3K — all clean CAD or synthetic). None compares a
**per-named-primitive** address space against anything. And Scott's Table III shows
"model quality specification" is `N` for **all 10** surveyed algorithms — no
explicit pass/fail criterion tying view selection to inspection quality existed as
of 2003, and nothing in this corpus has fixed it since.

---

## 6. Contested — contradictions kept visible

1. **Does the explosion bite?** Box/cylinder arithmetic: no (convex; 26/27
   elements; Θ(n) for a cylinder prism). Two-block assembly: **206–461 nodes.**
   Both correct; they measure different constructions.
2. **Is aspect-graph adjacency computable for us?** Plantinga & Dyer give an
   exhaustive closed-form event catalog — *for exact polyhedra*. Scott says the
   representation *"strictly fails"* for curved objects. Cyr & Kimia sidestep both
   by sampling and clustering, but were built and evaluated on clean rendered
   models. Three positions, not reconcilable in general.
3. **Does the agent need adjacency?** Six systems say no. Our current agent's
   bin-incrementing says something like yes. But no system in the corpus had a
   blind orchestrator, so the evidence does not actually address our question.
4. **Bounded cardinality: goal or anti-goal?** MVTN / HACMan++ / Mosbach-Alg2
   deliver it. Gospodnetić's R2/R4 explicitly want the opposite. Cyr & Kimia's
   mechanism cannot deliver it at all.
5. **What does discretization cost?** GenNBV: it's the bottleneck. GenNBV's own
   uniform baseline: ~8 pp on a coverage-only metric. Gospodnetić: the uniform
   sphere is the worst family. NBV-Net: a 14-way flat menu reaches 80–93% coverage.
   GenNBV again: bounded spaces are "by-design safe."
6. **Is direct optimization of view parameters viable?** MVTN's controlled internal
   experiment says it underperforms a *noise baseline*. Magaña's C-space says
   optimize inside a closed-form region instead. These are compatible but point at
   different implementations of our "optimization track."
7. **Is the orthogonal view even a thing in view-space terms?** Classical
   formalism: a measure-zero boundary element. C-space / surface-parameterized
   formalism: a positive-volume region by construction. Same physical view,
   opposite structural status — because the two formalisms define equivalence
   differently (image topology vs. constraint satisfaction). **This is the deepest
   finding in the branch, and it is Scott's objection restated constructively.**
8. **Provenance.** Three of 20 documents could not be read in full: Bowyer & Dyer
   1990 (substituted by Eggert/Bowyer/Dyer 1992 ISPRS — same author team, two
   years later, all numbers sourced to the substitute); Cyr & Kimia 2004 (metadata
   + convergent secondary description only — the 64-object / top-3-recall figures
   are **single-source and unverified**, do not propagate); Yang/Marefat/Johnson
   1998 (abstract only — absences there are "unknown," not "confirmed negative").
   Plus one authorship correction (§3.1) and one unread sibling paper that is the
   actual authority for the cylinder case.

---

## 7. What is missing

- **Curved-object aspect graphs** (`10.1007/BF00054918`, solids of revolution) —
  the only authority for a cylinder's exact structure. Unread.
- **Any closed-form scheme for a cylinder-orbit address.** Magaña does flat and
  pocket features only ("cylinder", "orbit", "primitive" appear zero times).
  Mosbach's cylinder was split into 4 mantle quarter-patches by accident of the
  quad layout. **This is a genuine hole for RQ2/RQ3.**
- **Primitive-boundary seams** — edges, corners, the box↔cylinder junction — as
  addressable or explicitly excluded. Universal silence.
- **An address that survives infeasibility** in a flagged/degraded form. The
  literature repairs, deletes, or fails.
- **Cross-shape view-addressing consistency.** Confirmed absent, ten independent
  confirmations.
- **Any evaluation with orthogonality as the metric.**
- **Any addressing scheme validated on a noisy fused point cloud.**
- **Front/side (azimuthal) canonicalization** — open even in the paper built for
  canonicalization.
- **Seibert & Waxman 1989** (adjacency learned from view transitions) — catalogued,
  not read. **Hutchinson & Kak** evidential sensor planning — never surfaced.

---

## 8. Implications

### RQ1 — Judging a view

- **Adopt Scott's framing as the foundation:** our equivalence classes must be
  defined by *measurement quality*, not visibility. This is what makes RQ1 and RQ2
  the same object, and it is the reason the aspect-graph tradition cannot be
  reused even where it is computable.
- **Adopt the measurability-matrix substrate** (Tarbox & Gottschlich via Scott
  §5.1): a bipartite viewpoint × surface-patch matrix whose entries are a *quality
  score*, not a visibility bit. Sensor-agnostic and discretization-agnostic — it
  survives whatever addressing scheme we pick.
- **Orthogonality already has an operational definition in the field**: Magaña's
  incidence angle `φ = arccos(n_f·e_s^z / |n_f||e_s^z|)` with a `φ_max` threshold,
  and the C-space is the region where it holds. That gives RQ1 both a continuous
  score *and* a feasible set in one construction.
- **Cost warning:** ray-traced visibility runs at 3.81 s/candidate. An analytic
  primitive-level substitute (plane and cylinder visibility in closed form) is
  needed for the session loop and has no precedent in the corpus.
- **Cheap ready-made self-occlusion score:** Fu's 32-uniform-view render with
  color-ID per face, `<5%` visibility ⇒ occluded.

### RQ2 — Selecting the view set

- **Face-orthogonal placement is solved twice in closed form** and both reduce to
  the same formula: pose = surface point + `n·d`, look along `−n`. Use it directly
  on fitted planes; the literature's heavier machinery (frustum polytopes, B-spline
  networks) is what you need only if you want the *feasible region* rather than the
  *point*.
- **Cylinder orbit must be derived — there is no precedent.** The natural
  derivation is Ikeuchi's rule: the lateral surface's pose function is smooth, so
  it wants a **continuous or coarsely binned angle parameter**, not a bank of
  discrete addresses. Design the cylinder address as `(cylinder, cyl_i, {θ, d})`.
- **Primitive-class rules are licensed by evidence, not just by the contract.**
  MVTN: class-averaged 90.6% vs. per-instance 91.0% vs. generic 89.0%. Most of the
  win is category-level.
- **Caution the optimization track** with MVTN §4.5 (direct scene-parameter
  optimization underperformed a noise baseline) and steer it toward Magaña's
  better-posed "optimize inside a closed-form region" formulation.
- **Expect to trade the last 1–2% of coverage.** Gospodnetić's greedy NBV uses a
  2%-of-best-contribution stalling threshold, yielding 98–99% not 100%, with
  remaining gaps found manually via a visualization tool they had to build.
- **Bound orientation search hard.** 27→75→363 CSG orientation combinations cost
  15 s → 40 s → 720 s for a volume gain of 0.9995. Keep orientation fixed per
  address.

### RQ3 — Composing the action space (this branch's own question)

- **Adopt category (b), shaped as HACMan++'s tuple with the "where" replaced
  wholesale:** `(primitive_type, primitive_id, parameters)` —
  `(plane, face_3, {standoff, roll})`, `(cylinder, cyl_1, {θ, standoff})`. The
  tuple shape is precedented; HACMan++'s raw-point instantiation is the *opposite*
  of what we need and must be discarded. Per-action variable parameter
  dimensionality is native to PAMDP.
- **Address→pose must be closed-form, not a nearest-neighbour lookup.** That single
  property is the whole difference from the `{h,v}` baseline, and it is what makes
  the address predictable to a blind orchestrator.
- **Do not build an aspect graph.** Reasons in priority order: (1) Scott's
  equivalence-class mismatch, fatal regardless of size; (2) the views we want are
  measure-zero boundary elements in that formalism; (3) curved primitives break the
  representation ("strictly fails"); (4) the assembly node count. **The size
  argument alone would not have stopped us** — that reframing is worth stating
  explicitly against the standing critic question.
- **Build no adjacency initially.** If it is later needed, kNN on camera coordinates
  (view-GCN Eqs. 1–2) captures ~90% of a learned relation's benefit and is an
  afternoon's work over whatever addresses exist. Prefer a **relevance ranking**
  (OmniManip's pattern) as the ordering the agent actually consumes.
- **Decide the seam question explicitly.** Precedent exists only for HEALPix's
  "name the exception in the address prefix." Either add an `edge_(i,j)` address
  class with a bisector-normal view, or state that edges are out of scope. There is
  no cover in the literature either way.
- **Weaken the cross-shape requirement deliberately.** Require stable meaning
  *within a session and within a primitive type* — every `plane` address means
  "orthogonal to that plane at standoff d" — rather than stable slot identity across
  objects. That is achievable by construction and is exactly what the corpus's
  successful systems deliver: a stable **rule**, not a stable **index**. The strong
  version (slot 3 means the same thing on a box and a cylinder) has no precedent
  anywhere and would be novel work with nothing to lean on.
- **Cardinality:** 8–20 per object falls out of per-primitive addressing. Do not
  chase a global bound across shapes; there is no mechanism and probably no need,
  since the VLM sees the menu each turn.
- **Anchor every address to a fixed canonical role.** MVTN's view-collapse at M>4
  without an anchor is evidence, not intuition — and primitive-face addressing
  provides the anchor structurally.

### RQ4 — Presenting to the VLM

- **Tractability is satisfied by construction** at 8–20 named addresses. The
  aspect-graph explosion is not the risk it was framed as; the risk is elsewhere
  (input quality, seams, infeasibility).
- **The address should be a name, not an index** (`face_3/orthogonal`), and pose
  resolution should be closed-form so the orchestrator can be told what it will get
  *before* it moves. Both (a) and (b) support mechanical prediction; only (b)
  supports semantic prediction.
- **HEALPix's NESTED prefix+path is the one concrete precedent for a hierarchically
  legible address.** Our analog — `object → primitive → view` — is shorter and
  simpler, and inherits the same property: the coarse level is a readable prefix.
- **Prefer "flagged infeasible" over "silently removed."** A menu that changes shape
  between turns is exactly what a blind language model handles badly. No literature
  cover — every corpus precedent (repair / delete / ordered-fallback-then-fail)
  assumes a non-blind consumer.
- **OmniManip is the closest working precedent for the whole pattern** (VLM picks
  from a small geometry-derived ordered menu; geometry resolves the pose; a check
  loop refines), and it quantifies what the check loop buys: **68.3% closed-loop
  vs. 51.7% open-loop, a >15 pp gap.** Budget for a verification step.
- **Hand to Branch 6, explicitly:** whether a blind orchestrator needs adjacency is
  genuinely untested. Every "no adjacency needed" datum in this branch comes from a
  system whose consumer was a network, an optimizer, or a *seeing* VLM.

### Benchmark methodology

- **The protocol is precedented — lift Gospodnetić's design wholesale and swap the
  objective.** Several candidate-generation families → ONE shared downstream
  selection algorithm → ONE metric, across multiple objects, reporting coverage
  **and** viewpoint count **and** inadmissible-candidate rate **and** the
  run-to-run envelope. 292 candidate sets across 3 objects is the scale of a
  credible study.
- **Add a second, upgraded baseline: HEALPix/Hopf.** Beating naive `{h,v}` proves
  nothing — it is a known-bad discretization by 2003 standards (Scott §9.2.3: *"to
  the degree that object shape is not spherical, viewpoint position is
  nonoptimal"*). Beating an equal-area, distortion-free, roll-covering,
  zero-build-cost uniform grid is what proves the geometry-conditioning claim.
  **This is the branch's single most valuable methodological implication.**
- **Report inadmissible-candidate rate as a first-class metric.** It was
  Gospodnetić's sharpest discriminator between families (61.9–77.0% vs. 15.9–26.6%)
  and it maps directly onto our IK / collision / min-shell-radius stack.
- **Manage the circularity risk concretely.** If the address space is *constructed*
  to be orthogonal and the metric *is* orthogonality, every geometry-conditioned
  candidate wins by construction. Two mitigations from the corpus: (i) score
  orthogonality achieved on the object's *actual surface patches* (measurability-
  matrix rows) rather than at the addressed points; (ii) keep generator and judge
  on different data — ground truth from the **fitted primitive**, the candidate's
  score from the **raw cloud's local normals**. Report coverage and count jointly
  so a scheme cannot win on orthogonality by covering less.
- **Scott's Table III is the citable justification for the benchmark's existence:**
  "model quality specification" is `N` for all 10 surveyed algorithms. The field has
  never tied view selection to an explicit quality pass/fail. We are filling a
  documented, twenty-year-old hole, not inventing a niche.
- **Our distinguishing contribution is the input.** Every benchmark object in the
  corpus is clean CAD or synthetic. Running on real survey clouds with the measured
  defect profile (few-mm pose error, dropout, ~4 cm fused-vs-single-view offset) is
  what makes our benchmark new — and Branch 3 owns whether the primitive fit
  survives that.
- **Budget realistically.** Ray-traced visibility costs ~1–1.6 h/object. Acceptable
  as an offline benchmark precompute; unacceptable in the session loop. Plan an
  analytic primitive-level replacement for deployment and keep the ray-traced
  version as the benchmark's ground truth.
- **Compare against the human baseline too.** Gospodnetić's experts produced
  *larger* view sets (44 vs. 23) that were nonetheless better structured for
  physical implementation. If our benchmark only scores coverage and orthogonality
  it will systematically reward sets no operator would choose.

---

## 9. Cross-branch handoffs

- **→ Branch 1 (view quality):** Magaña's incidence-angle formula and its `φ_max`
  constraint as the operational orthogonality definition; the measurability-matrix
  substrate; Fu's 32-view color-ID visibility score; the 3.81 s/candidate ray-trace
  cost that motivates an analytic substitute.
- **→ Branch 2 (benchmark):** §5.5 in full — the corrected finding that three
  head-to-head comparisons exist, Gospodnetić's protocol as a liftable template,
  the HEALPix upgraded-baseline recommendation, the inadmissible-rate metric,
  Scott's Table III, and the generator/judge separation for circularity control.
- **→ Branch 3 (primitive structure):** the load-bearing dependency — *every*
  closed-form addressing scheme in this corpus requires a clean modeled surface;
  Mosbach's authors state B-spline quality "strongly decreases" under noise with no
  quantitative bound; Magaña explicitly neglects random sensor noise and models only
  *pose* uncertainty, never *geometry* uncertainty; Fu attributes 5% of failures to
  mesh defects on curated models; Fu's ~20%-of-failures mode is box-like shapes.
- **→ Branch 4 (selection/feasibility):** §5.2's four infeasibility behaviours and
  the absence of a flagged-degraded option; Gospodnetić's repair rule (ray-cast to
  surface, reorient to `−n`); Tarbox's morphological erosion; the 2% stalling
  threshold and its 98–99%-not-100% consequence; the set-cover NP-completeness
  framing.
- **→ Branch 6 (VLM interface):** the untested question of whether a blind
  orchestrator needs adjacency; OmniManip's ordered-relevance menu and its
  >15 pp closed-loop gain; HEALPix's prefix+path address as the legibility
  precedent; MVTN's view-collapse argument for anchoring addresses to canonical
  roles; the menu-stability argument for flagging rather than deleting.

---

## 10. Follow-ups worth a read slot in pass 2

1. **`10.1007/BF00054918`** — "Computing exact aspect graphs of curved objects:
   Solids of revolution," same IJCV issue as Plantinga & Dyer. The only authority
   for a cylinder's exact structure; one DOI away and unread.
2. **Bowyer & Dyer 1990** with institutional access — currently substituted; if the
   vocabulary or bounds become load-bearing for RQ3, close the dependency.
3. **Cyr & Kimia 2004** with institutional access — the most directly implementable
   non-classical recipe in the branch, read only through convergent secondary
   description.
4. **Yang/Marefat/Johnson 1998** — the "combine and contract EAGs from a database"
   composition mechanism is exactly our assembly case and is abstract-only.
5. **Seibert & Waxman 1989** — adjacency learned from view transitions; the only
   unread member of the adjacency inventory.
6. A targeted search for **closed-form orbit/sweep view generation on curved
   primitives** — the confirmed hole in RQ2/RQ3, which nothing in this branch
   fills.
