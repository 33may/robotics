# Bowyer & Dyer — "Aspect Graphs: An Introduction and Survey of Recent Results" (1990)

Int. J. Imaging Systems and Technology 2(4):315–328, 1990. Also presented as Bowyer & Dyer,
SPIE Proc. Vol. 1395 (*Close-Range Photogrammetry Meets Machine Vision*), 13950R, 1990
(same title, same authors, conference version of the same content).

## Access note — read carefully before trusting any claim below

**I could not obtain the full text of the actual 1990 paper.** Both the Wiley journal DOI
(`10.1002/ima.1850020407`) and the SPIE conference DOI (`10.1117/12.2294270`) are confirmed
closed-access with **no OA location** via Unpaywall (`oa_status: closed`, `oa_locations: []`
for both) and via the Semantic Scholar API (`openAccessPdf.status: CLOSED`). I additionally
tried and exhausted:
- Wiley abstract/epdf pages directly (403 Forbidden both times).
- The author's own page, Kevin Bowyer's Notre Dame site (`academicweb.nd.edu/~kwb/aspects.htm`
  and `/publications.htm`) — lists the citation, links to six *related* aspect-graph papers,
  **no PDF for this one**.
- University of Wisconsin CS technical-report FTP/MINDS@UW archive — has other Plantinga/Dyer
  aspect-graph TRs (e.g. TR627, TR683, TR736) freely as PDF, but not this survey.
- CiteSeerX / CORE OAI record — record exists but is Cloudflare-gated / returns a mismatched
  (medical, unrelated) document when actually fetched.
- **Internet Archive** — the exact scanned journal issue exists
  (`sim_international-journal-of-imaging-systems-and-technology_winter-1990_2_4`, confirmed via
  metadata to contain this exact page range) but the item is flagged
  `"access-restricted-item": true` in collection `printdisabled` — borrow-only, not
  downloadable or full-text-searchable without a lending account I don't have.
- ResearchGate (403), Google Books API (quota exhausted), Internet Archive Scholar (0 hits).

**Substitute used:** I read in full (13 pages, all sections and figures) the freely available
**Eggert, Bowyer & Dyer (1992), "Aspect Graphs: State-of-the-Art and Applications in Digital
Photogrammetry,"** ISPRS Congress Proc. Vol. XXIX Part B5, pp.633–645
(`https://www.isprs.org/proceedings/xxix/congress/part5/633_xxix-part5.pdf`, legitimately open,
no paywall). This is **not** the requested document, but it is the closest legitimate substitute
available: two of the three authors are Bowyer and Dyer themselves, it explicitly cites the 1990
survey as a direct predecessor in its own reference list (`Bowyer, K.W. and Dyer, C.R., 1990.
Aspect Graphs: An Introduction and Survey of Recent Results...`), its abstract's structure
("history of the evolution... state of the art in algorithms... applications... current
limitations") mirrors the phrasing that appears in the 1990 paper's Semantic-Scholar-derived TLDR
almost sentence-for-sentence, and it is explicitly framed as extending the same tutorial two years
on. **Every claim below is sourced to this 1992 document unless marked otherwise**; where a claim
is attributed inside it to *other* named papers (e.g. Plantinga & Dyer 1990's Θ(n⁶)/Θ(n⁹) bound),
I say so — those are the primary sources for that specific number, not this substitute, and
**not** the 1990 Bowyer & Dyer survey itself. I flag every place where I am inferring what the
1990 survey probably said rather than reading it directly. Bibliographic metadata for the actual
1990 paper (title, venue, authors' affiliations, DBLP key `journals/imst/BowyerD90`) is confirmed
via Unpaywall/Semantic Scholar and is reliable; its *content* beyond the TLDR sentence is not
independently confirmed here.

Given this, treat this note as informative but **not as a substitute for the primary source** —
if the vocabulary/bounds below turn out load-bearing for RQ3, the actual 1990 text should be
obtained through an institutional subscription before being cited as a settled fact.

## Core content (from the 1992 substitute; presumed close to the 1990 original's scope)

### Aspect graph definition (§2, p.633)
Two load-bearing elements, explicitly flagged as "commonly agreed":
- **A node for each *general view*** — the view seen from some maximal connected cell of
  **viewpoint space**.
- **An arc for each possible transition, called a *visual event*, between two neighboring
  general views.**

The "not so commonly agreed" elements — left as design choices by every implementation — are
**(a) the model of viewpoint space** and **(b) what is meant by a "general view."** This is
exactly the branch's Q1 vocabulary gap: the paper is explicit that "aspect graph" is a *family*
of representations parameterized by these two choices, not one fixed structure.

### Two models of viewpoint space (§2.1, p.634)
1. **Viewing sphere**: unit sphere centered on the object; a point on the sphere = a line-of-sight
   direction; normally paired with **orthographic** projection (2-D parameter space).
2. **Full 3-D space**: any 3-D position is a viewpoint; strictly this needs a 7-D parameter space
   for perspective (position + line-of-sight direction + focal length), but the standard
   simplifying assumption is a 360°-FOV viewer, reducing it to **3 parameters** (position only).

### General view vs. accidental viewpoint (§2.2–2.3, p.635)
- **General viewpoint**: infinitesimal movement in *any* direction leaves the view's symbolic
  description isomorphic to the original.
- **Accidental viewpoint**: some direction of infinitesimal movement changes the isomorphism
  class.
- **Visual event** = the boundary crossed going from one region of general viewpoints, through an
  accidental viewpoint, into another region.
- The symbolic description used for polyhedra is the ***image structure graph***: lines +
  junctions of the idealized line drawing, explicitly **qualitative** — "omitting any
  quantitative data such as the lengths of lines or the size of the angles between lines" (p.635).
  A junction catalog (Fig.2, after Malik 1987 / Eggert 1991) gives 10 named types (arrow, Y,
  multi, L, T, pseudo, termination, three-tangent, curvature-L, transition-S); **only the first
  five occur for polyhedra**, all ten for curved surfaces.
- For curved objects the description additionally needs **contour generators / rims / limbs**:
  loci where the line of sight is tangent to the surface.

### Visual events, concretely
- **Polyhedra** (§4.1, p.636): only two fundamental event types exist — **edge/vertex alignment**
  (a vertex and a point on an edge project to the same image point; marks the start/end of
  occlusion, appears as a T-junction) and **edge-triplet alignment** (three edges project through
  a common image point; changes the *ordering* of T-junction relations). Edge/vertex-alignment
  event boundaries are **planes** in 3-D viewpoint space / **great circles** on the viewing
  sphere; edge-triplet events lie on a **quadric surface** in 3-D space / a **quadric curve** on
  the sphere (Gigus & Malik 1987; Stewman & Bowyer 1988).
- **Curved objects** (§5.1, p.638): the event catalog comes from singularity theory (Kergosien
  1981; Rieger 1987, generalizing to piecewise-smooth surfaces): beak-to-beak, lip,
  curvature-L / three-tangent junctions, cusp (swallowtail), tangent-crossing. Event surfaces
  are **ruled, tangent-developable surfaces**; their intersection with the sphere/3-D space
  yields **polynomial curves**, and finding them is explicitly called "the most difficult stage
  of the algorithm" (p.639).

### Two named data structures that generalize the graph
- **`asp`** (Plantinga & Dyer 1986/87/90): a **4-D structure** (2-D image-plane location × 2-D
  sphere viewpoint; would be 5-D for 3-D viewpoint space) that tracks *every individual feature's*
  image-plane location continuously as a function of viewpoint. Occlusion of one feature by
  another = intersection of their two `asp` sub-structures. This is a **feature-centric**
  alternative to the whole-object aspect graph — matches the same mitigation the Faugeras-panel
  sibling note already found (Dyer's 1992 panel response), here shown to predate the panel by two
  years and to be a shipped, implemented structure, not just a proposal.
- **Rim appearance model** (Seales & Dyer 1990): `asp` restricted to only the events relevant to
  the **occluding contour** — explicitly reported to capture **"roughly 25% of all actual
  events"** (p.638) — a concrete, numeric precedent for entity-based pruning (RQ11): keeping only
  the feature type you actually care about (here: occluding silhouette) removes ~75% of the
  full event catalog with named justification, not guesswork.

### Concrete node/region counts given in worked examples (Figs.4, 5, 7, 8, 10)
- **Tetrahedron** (Fig.4): a full worked aspect graph is shown (small, not counted numerically in
  text, but visibly on the order of a dozen nodes on the sphere).
- **Nonconvex triangular-plate-with-hole polyhedron** (Fig.5, p.637): under the viewing-sphere
  model, the sphere's upper quadrant partitions into **6 regions**, giving **24 total regions**
  over the whole sphere by symmetry, but **only 4 visually distinct views** among them (heavy
  symmetry collapse). Under the full 3-D-space model, there are **81 total regions**, revealing
  additional views (e.g. an "inside" view through the hole) never seen from the sphere at all —
  concrete, numeric evidence that **viewing-sphere vs. full-3-D-space are not interchangeable
  models**: the richer model finds qualitatively different views the impoverished one cannot
  reach.
- **Solid of revolution** (Fig.7, p.639): a 2-D cross-section of the 3-D parcellation is shown
  with 12 labeled regions, collapsing by horizontal symmetry to a smaller distinct set.
- **Squash-shaped implicit-surface object** (Fig.8, p.640, adapted from Petitjean/Kriegman/Ponce
  1992): explicit degree-4 implicit equation given
  (`4y⁴+3xy²-5y²+4z²+5x²-2xy+2x+3y-1=0`); its viewing-sphere parcellation has **11 regions**
  bounded by beak-to-beak, lip, swallowtail, and tangent-crossing event curves.
- **Two-block articulated assembly, 1-D translational joint** (Fig.10, p.641): base 2×4×2,
  appendage 1×2×1, configuration parameter c∈[0,2]. The configuration range splits into **8
  general-configuration intervals** (7 accidental configuration values). **The aspect-graph node
  count per interval ranges from 206 to 461** (exact reported values: 449, 461, 461, 451, 459,
  206, 319, 319, 315, 319, 311 across the boundary/interval sequence shown). This is the single
  most load-bearing number in the whole document for our branch: **a trivial two-primitive
  assembly under the classical full topological definition already produces several hundred
  nodes**, which is strong quantitative support for why the classical, unpruned aspect graph is
  the wrong structure to imitate directly for our box+cylinder assembly (RQ8/RQ10), and why every
  mitigation this document and the Faugeras-panel sibling document independently converge on
  (restrict to parts, restrict to one feature type, add scale) exists.

### Bounds (feeds RQ7 directly, but attributed to a different primary source)
> "Upper bounds on the number of nodes in the aspect graph have been determined to be **Θ(n⁶)**
> and **Θ(n⁹)** for a general polyhedron of n faces under the viewing sphere and 3-D space
> models, respectively (Plantinga and Dyer, 1990)." — §4.2, p.638.

This is cited *inside* the 1992 substitute to **Plantinga & Dyer 1990** (`Visibility, Occlusion,
and the Aspect Graph`, IJCV 5:137–160) — a different paper from our target, published the same
year. I cannot confirm from this substitute whether the Bowyer & Dyer 1990 survey itself restates
these bounds (very likely, since it is a survey of exactly this result, published the same year
by one of the same two authors, Dyer) or gives the finer-grained pair the branch brief asks about
(Θ(n²) orthographic / Θ(n³) perspective *node counts* as distinct from Θ(n⁶)/Θ(n⁹) *total
complexity*). **The Θ(n⁶)/Θ(n⁹) split found here matches the branch brief's perspective-vs-
orthographic asymmetry in spirit** (richer 3-D-space/perspective model → higher exponent) but I
cannot verify the exact n²/n³ vs n⁶/n⁹ two-tier claim from this document — that requires either
the primary Plantinga & Dyer 1990 IJCV paper or the actual Bowyer & Dyer 1990 survey text, neither
of which I was able to open. **Flag this as unresolved, not fabricated-and-confirmed.**

### The scale-space aspect graph — a direct, numbered pruning mechanism (§8, p.642)
Introduced specifically as a fix for "the size can become quite large as object complexity
increases (Faugeras et al., 1992)" — explicitly cites the same panel piece already read for this
branch. Three named interpretations of a blur parameter σ, each with a distinct mechanism:
1. **Viewer size** — model a finite-radius observer as a sphere in viewpoint space; cells smaller
   than that sphere merge/disappear as σ increases.
2. **Image feature size** — a feature below one pixel of visual angle is unrepresentable; two
   cells differing only by such a feature merge.
3. **Object detail** — smooth the 3-D surface itself (not the image), reducing the visual event
   set directly at the geometry level.
Figure 11 gives a **concrete before/after count** for a six-sided nonconvex polygon: viewpoint-
space parcellation at 10° visual-arc resolution vs. 52° visual-arc resolution — cells present at
the fine resolution disappear at the coarse one, with an explicit claim that "several different
scale events occur between depicted scale values," i.e. cell count is demonstrably, numerically
resolution-dependent. This is the clearest primary-literature precedent found so far for "prune
by merging similar/indistinguishable views" (RQ11), with an actual mechanism (not just "similarity
below threshold" asserted informally) and a worked example.

### Applications section (§7, p.640–642) — confirms the sibling note's negative finding
Every named computer-vision application is **recognition/indexing**: CAD-based feature-saliency
prediction for interpretation trees (Ikeuchi 1987; Hansen & Henderson 1989), aspect-indexed
nonlinear-optimization pose matching (Stark, Eggert & Bowyer 1988), part-based recognition via
aspect graphs of primitive geons (Dickinson, Pentland & Rosenfeld 1992). **No application anywhere
in this 13-page document names or implies using the aspect graph, `asp`, or rim-appearance model
to select *where to move the camera next***. The only place viewpoint *motion* appears at all is
graphics rendering (§7.2): precomputing hidden-line removal once via the aspect graph / `asp`,
then doing cheap incremental updates as a **known, pre-specified camera path** crosses visual-event
boundaries (Stewman 1991; Eggert 1991; Plantinga, Dyer & Seales 1990). This is the closest thing
in the whole document to "adjacency used for camera motion" — but the path is externally given
(rendering an animation), not chosen by any agent, and the graph is not being *addressed* — it's
being *walked*, silently, by pre-scripted continuity. Confirms the sibling Faugeras-panel note's
finding that action-addressing has no precedent in the classical aspect-graph literature.

## Load-bearing assumptions (as with the sibling document, transferred from the substitute)

- Every worked example and every event catalog assumes the object is either an exact polyhedron
  (planar faces, sharp edges) or an exact smooth/piecewise-smooth analytic surface (solids of
  revolution, algebraic/implicit surfaces, quadrics) — **never a noisy point cloud**. Nothing in
  this document tests robustness to sensor noise, pose error, or dropout; visual-event surfaces
  are derived from exact object equations throughout.
- Perspective-projection construction is reported as only recently, partially achieved even for
  the *simplest* curved case (solids of revolution: "very recently achieved a complete
  implementation (Petitjean, Ponce and Kriegman, 1992)... assuming orthographic projection" —
  §5.2, p.639 — note even that "complete implementation" is qualified as orthographic-only in the
  surrounding sentence). For general polyhedra under the full perspective/3-D-space model, the
  **geometric incidence lattice** construction is stated to be "constructible for polyhedra"
  (Stewman 1991) but curved-object equivalents "has so far proved too difficult to construct"
  (p.639).
- The `asp`/rim-appearance and geometric-incidence-lattice machinery all presuppose exact CAD-like
  feature identities (edges, vertices, faces) available in closed form — nowhere is fitting these
  structures to a segmented/noisy point cloud attempted or discussed.

## What transfers to our setting

- **The two-part definition (viewpoint-space model + general-view definition are separately
  chosen) is exactly the vocabulary gap the branch needs**, and it is explicit that these are
  independent design axes, not baked into "aspect graph" as a monolith — directly usable
  scaffolding for classifying our own candidate addressing schemes (RQ1: is our address a global
  tessellation index, a part+offset, or a relational node ID — this document's "model of viewpoint
  space" is exactly the first two, and "general view" is the third).
- **The rim-appearance model's ~25% figure is a concrete, citable precedent for restricting to
  the feature type that matters** (occluding contour there; "orthogonal-to-surface visibility"
  for us) rather than cataloging every possible visual event — directly supports the branch's
  entity-based-restriction pruning question (RQ11) with an actual number rather than an assertion.
- **The 206–461-node range for a two-block articulated assembly is the strongest quantitative
  argument found anywhere so far (across both documents read for this branch) for why the
  classical full aspect graph is the wrong target to imitate**, and it directly motivates why
  RQ8/RQ10's "smallest structure that expresses box-face-by-face / cylinder-orbit behaviour" must
  be a *deliberately different, much smaller, task-specific* structure rather than a pruned
  version of the classical one — the classical one doesn't get small even for two boxes.
- **The scale-space mechanism (§8) is a second, independent, numbered precedent (beyond the
  Faugeras-panel note's Pentland/Dyer citations) for pruning by resolution/similarity merging**,
  with an actual before/after example (Fig.11) rather than just a proposal — reusable evidence for
  RQ11.
- **The viewing-sphere-vs-3-D-space count gap (24/4-distinct vs 81 regions, Fig.5) is concrete
  evidence that a uniform-sphere-only action space provably misses views a richer model would
  find** — direct support for the framing doc's own diagnosis (uniform {h,v} sphere can't express
  face-orthogonal views) from an independent 1992 worked example, not just our own project's
  intuition.

## What does NOT transfer, and why (negative findings)

- **No cross-shape addressing scheme of any kind appears.** Every example (tetrahedron, plate-
  with-hole, solid of revolution, squash blob, two-block assembly) is treated as its own
  self-contained graph with its own node numbering; there is no discussion of what it would mean
  for "node 3" to mean the same thing across two different objects. This is independent
  confirmation (second document in a row, after the Faugeras panel) that **RQ19's cross-shape
  view-addressing question has no precedent in the classical aspect-graph literature** — the gap
  is real, not an artifact of only reading one document.
- **No ordering, "next/previous," or numbering convention over nodes is discussed anywhere.**
  Nodes are identified only by position in the partition/graph traversal, never by an assigned
  index with directional meaning. RQ4 (stable ordering, opposite/adjacent-face semantics) gets
  zero support from either aspect-graph document read so far.
- **Adjacency is defined, once, purely geometrically** — two aspects/cells are adjacent iff their
  viewpoint-space regions share a visual-event boundary — but this is **never tied to a name or
  action semantics** (no "opposite face," no "next ring") beyond the raw geometric fact of sharing
  a boundary. Directly answers RQ5's "collect every found definition" with one clean, computable
  entry (shared visual-event-surface boundary) but it comes with zero vocabulary for what that
  adjacency *means* to a user of the graph — that has to be supplied by us, not inherited.
  Whether it's computable on our noisy fused PLY is **not addressed anywhere in this document** —
  every event surface here is derived from exact equations of an exact object, never from a fitted
  primitive on real sensor data. This is the same gap the sibling Faugeras-panel note already
  flagged and hands to the primitive-structure-from-survey-clouds branch.
- **No benchmark, comparison to a uniform sphere, or task-based metric appears anywhere.**
  Reinforces (third time now, across two aspect-graph documents plus the project's own prior
  finding) that a geometry-conditioned-vs-uniform-sphere comparison on a stated task is absent
  from this literature — the benchmarking-view-planning branch's gap claim gets a third
  independent confirmation.
- **The document never once frames the aspect graph as an action space.** Every single use case in
  §7 is recognition, pose estimation, or pre-scripted rendering-path replay — none is "an agent
  choosing where to look next." This matches the sibling note's finding exactly and is worth
  restating because it comes from an independent, non-panel, primary-technique survey rather than
  a discussion piece — the forward/inverse asymmetry the sibling note argues for (our address is
  used forward: label → pose; classical aspect graphs are used inverse: image → node) is not
  contradicted anywhere in this second document either.

## Citations (for checkability — all page numbers refer to the 1992 ISPRS substitute, NOT the
1990 original, whose pagination I never had access to)

- Aspect graph definition (node = general view of maximal viewpoint-space cell; arc = visual
  event): §2, p.633.
- Two viewpoint-space models (viewing sphere vs. full 3-D space): §2.1, p.634.
- Image structure graph, junction catalog (Fig.2): §2.2, p.635.
- General/accidental viewpoint definitions, visual event definition: §2.3, p.635.
- Polyhedral visual events (edge/vertex alignment, edge-triplet alignment) and their surface types
  (plane/great-circle, quadric/quadric-curve): §4.1, p.636–637.
- Nonconvex-polyhedron worked example, 24/4-distinct regions (sphere) vs. 81 regions (3-D space):
  Fig.5, p.637.
- `asp` representation (4-D feature-centric structure), rim appearance model, ~25% event coverage:
  §4.2–4.3, p.637–638.
- Θ(n⁶) / Θ(n⁹) node-count upper bounds, attributed to Plantinga & Dyer 1990: §4.2, p.638.
- Curved-object visual event catalog (Kergosien 1981; Rieger 1987), event-surface geometry: §5.1,
  p.638.
- Solid-of-revolution worked example: Fig.7, p.639.
- Squash-shaped implicit-surface worked example, 11 regions: Fig.8, p.640.
- Applications — recognition/indexing use cases: §7.1, p.640–641.
- Applications — graphics rendering, incremental update along a known path: §7.2, p.641–642.
- Scale-space aspect graph, three interpretations of σ, Fig.11 resolution example: §8, p.642–643.
- Two-block articulated-assembly worked example, 206–461 nodes across 8 configuration intervals:
  Fig.10, p.641.
- Faugeras et al. 1992 panel cited as the motivation for §8 (size explosion): p.642, and in the
  reference list, p.644.
- 1990 Bowyer & Dyer survey cited as direct predecessor in the reference list: p.644.

## Assessment for the branch

I did not get to read the requested document; I read its closest legitimate, same-author-team
2-years-later extension instead, in full, and cross-checked every load-bearing claim against page
and section numbers within *that* document. The vocabulary gap (viewpoint-space model × general-
view definition, aspect, visual event, general/accidental viewpoint) is now filled and directly
maps onto the branch's addressing taxonomy (RQ1). Two new, concrete numbers earn their place in
the synthesis: the **206–461-node count for a trivial two-block assembly** (strong evidence
against imitating the classical unpruned structure, feeds RQ8/RQ10) and the **rim-appearance
model's ~25%-of-events figure** (concrete precedent for entity-based pruning, feeds RQ11). The
Θ(n⁶)/Θ(n⁹) bound is present but attributed inside the substitute to a different 1990 paper
(Plantinga & Dyer), and I could not verify whether it matches the finer Θ(n²)/Θ(n³) vs Θ(n⁶)/Θ(n⁹)
two-tier claim in the branch brief — that arithmetic (RQ7) needs the primary Plantinga & Dyer 1990
IJCV paper, not this document. Every negative finding from the Faugeras-panel sibling note (no
cross-shape addressing, no ordering scheme, no benchmark, no action-space use case) is
independently reconfirmed here from a second, non-overlapping document — these gaps are now
corroborated twice, not asserted once. **If RQ3 or RQ7 end up depending on the exact wording or
numbers of the actual 1990 Bowyer & Dyer survey rather than this substitute, that dependency
should be flagged for follow-up with real institutional access** — I was not able to close it.
