# Plantinga & Dyer — "Visibility, Occlusion, and the Aspect Graph"

IJCV 5(2):137–160, Nov. 1990. DOI `10.1007/BF00054919` (the URL/DOI given in the task
brief). Full text obtained via the underlying UW–Madison CS Dept. tech report
(TR#736, Dec. 1987, "Visibility, Occlusion and the Aspect Graph," Harry Plantinga
and Charles R. Dyer — 60 pp., identical content to the IJCV version per the paper's
own citation of itself as "[Plantinga and Dyer, 1986]" for the earlier conference
version), OCR'd in full (Springer's page was paywalled; the tech report is open at
`minds.wisc.edu/handle/1793/58908` → bitstream PDF, 60 scanned pages, OCR'd with
tesseract at 300dpi, all pages read).

## ⚠ Authorship correction (first-class finding, not a nitpick)

**The task brief's attribution to "Gigus, Canny, Seidel" is wrong.** The paper at
the given DOI/URL (`10.1007/BF00054919`, title "Visibility, occlusion, and the
aspect graph," IJCV 5(2):137–160, 1990) is by **W. Harry Plantinga and Charles R.
Dyer** — confirmed from the paper's own title page and abstract byline, both
OCR'd directly ("Visibility, Occlusion, and the Aspect Graph* / W. Harry Plantinga
— Charles R. Dyer / Department of Computer Sciences / University of Wisconsin -
Madison"). Gigus, Canny & Seidel wrote a different, related paper — "Efficiently
Computing and Representing Aspect Graphs of Polyhedral Objects," IEEE TPAMI
13(6):542–551, 1991 (not IJCV, not this title) — which gives an O(n⁶ log n)
*construction-time* algorithm for the polyhedral orthographic case but is not what
sits at this URL. I did not fetch that paper; everything below is from the
Plantinga & Dyer paper actually at the cited URL.

Despite the wrong author names, **the DOI/URL and the numbers in the task's
selection rationale are correct** — Θ(n²)/Θ(n⁶) orthographic and Θ(n³)/Θ(n⁹)
perspective are exactly the bounds this paper proves (Abstract, Table 1, Table 2,
Conclusion). So the *document* was correctly selected; only its authors were
mislabeled. Likely source of the mix-up: the IJCV DOI immediately adjacent to this
one, `10.1007/BF00054918`, is a different 1990 IJCV paper, "Computing exact aspect
graphs of curved objects: Solids of revolution" — i.e. the paper that would
actually cover a *cylinder's* aspect graph is one DOI number away in the same
issue. I could not confirm its authors (web-search budget exhausted this
session) — **flagging as a concrete follow-up read for the cylinder side of this
branch**, since Plantinga & Dyer explicitly do not cover curved objects (see
below).

## Core method / idea

The paper studies how the *topology* of a polyhedron's line-drawing image (edges
visible, hidden-line-removed) changes as viewpoint varies, for two viewing models:
**orthographic** (viewpoint restricted to directions on a sphere — a 2-D viewpoint
space) and **perspective** (viewpoint anywhere in ℝ³ — a 3-D viewpoint space, "as
if you could stand inside a house"). It defines:

- **Aspect / viewing region**: a maximal connected region of viewpoints from which
  the image has the same topological structure (§3, p.13–14), formalized as
  equivalence classes of an equivalence relation `v1 ~ v2` iff a path of
  constant-topology viewpoints connects them.
- **VSP (viewpoint space partition)**: the partition of viewpoint space into these
  equivalence classes, plus their boundaries — the primary data structure the
  paper builds and recommends (richer than the aspect graph: "the space it
  requires is only a constant factor larger," §3 p.17).
- **Aspect graph**: the dual of the VSP — one vertex per region, edges between
  adjacent regions (§3, p.15).
- **Events**: the ways aspect can change — for convex polyhedra only the
  *horizon effect* (a face turning away from the viewer, §4 intro, p.18–19); for
  non-convex polyhedra additionally **edge-vertex occlusion** (4 sub-cases: region
  appear/disappear, merge/split — §5.1.1.1, p.19–20, Figs 13–16) and
  **edge-edge-edge occlusion** (3 unconnected edges appear to intersect at one
  image point — §5.1.1.2, p.21–22, Figs 17–18). The paper proves these are a
  *complete* catalog: "If more than three edges meet at [an image] point, then
  either they are degenerate and some subset of three generates the same event,
  or they are general and hence do not generate an event with viewpoint-space
  extent" (p.29).
- **Asp (aspect representation)**: a continuous, viewer-centered 4-D (orthographic)
  or 5-D (perspective) representation — viewpoint × image-plane coordinates — used
  as the practical intermediate structure the construction algorithms build before
  taking the final viewpoint-space subdivision (§5.1.4.1, p.27–39).

## Concrete results and numbers

**Table 1 (max size of VSP / aspect graph), p.8, and stated again identically in
the Conclusion (p.56 OCR / paper p.50):**

| | convex | non-convex |
|---|---|---|
| orthographic | Θ(n²) | Θ(n⁶) |
| perspective | Θ(n³) | Θ(n⁹) |

**Table 2 (construction time), same page:**

| | convex | non-convex |
|---|---|---|
| orthographic | O(n²) | O(n⁶ log n) |
| perspective | O(n³) | O(n⁹ log n) |

`n` is used as a generic "size" measure of the polyhedron. For the **convex
orthographic bound** the paper ties `n` explicitly to **face count** — it is
derived directly from face normals cutting great circles on the viewsphere
(§4.1.1, p.19: "a polyhedron with n faces… the corresponding n great circles…").
For the **non-convex bound** `n` is tied to **edge count** — the O(n³)
edge-triples that each generate a boundary curve (§5.1.1.2, p.22: "There are O(n³)
sets of three edges in a polyhedron"). The paper treats these as interchangeable
("O(n) vertices and O(n) edges in a polyhedron," p.27) — valid for polyhedra with
bounded vertex/face degree (Euler-relation regime), which box and cylinder both
satisfy, but worth flagging as a load-bearing assumption (see below).

**Key exact formulas (not just asymptotics) that I plugged numbers into for our
objects:**

- Convex orthographic upper bound, exact: "the total number of intersection
  points is at most `2n(n−1)`" (great circles in general position, no antiparallel
  normals) — §4.1.1, p.19. For a generic (no parallel faces) convex hexahedron,
  `n=6` faces: this plus Euler's relation on the sphere (`V−E+F=2`, with
  `V=n(n−1)`, `E=2n(n−1)` for circles in general position) gives
  **F = n²−n+2 = 32 regions** for an *asymmetric* 6-faced convex solid.
- Non-convex orthographic: "there are O(n³) curves… O(n³) quadratic curves on the
  sphere intersect in O(n⁶) points" (§5.1.2, p.29) — this is edge-edge-edge
  occlusion, which **only exists for non-convex polyhedra** (§4 intro, p.18: "In
  the convex case, the only sort of event that can change the aspect is the
  horizon effect").
- Convex-case "highly degenerate" note, verbatim (§4.1.2, p.20–21): *"An example
  of a highly degenerate class of polyhedra is the class of n-sided approximations
  to a cylinder; for polyhedra in this class the VSP has size Θ(n)."* — This is
  the paper's own, explicit, stated result for a cylinder-*like* object (a prism
  approximation), giving **linear**, not quadratic, growth.

### Worked arithmetic for our objects (RQ7's task — done here, checkable)

**6-face box (cube):** The box is **convex**, so the Θ(n⁶)/Θ(n⁹) non-convex
bounds are a **category error** — edge-edge-edge occlusion cannot occur on a
convex solid at all (§4 intro). Only the convex bound applies, and even that is
loose because opposite faces have antiparallel normals — a "highly degenerate"
configuration the paper explicitly calls out as reducing VSP size (§4.1.2, p.19:
"If two faces of the polyhedron are parallel, the corresponding great circles
coincide, and the number of vertices in the VSP is reduced"). Working the
degenerate case by hand using the paper's own machinery (their formulas, my
arithmetic — not a number the paper itself states): a box's 6 antiparallel-paired
face normals collapse to **3** distinct great circles (mutually orthogonal, since
each pair of opposite faces shares one great circle). Three orthogonal great
circles subdivide the sphere exactly like an octahedron's edge skeleton: `V=6`
poles, `E=12` arcs, `F=8` open spherical-triangle regions (`V−E+F=6−12+8=2` ✓,
`V=n(n−1)=3·2=6` ✓ for `n=3` circles in general position). Applying the paper's
own equivalence-relation definition of "aspect" (§3) to *every* dimension of this
subdivision, not just the top-dimensional regions, gives:
- **8** open regions = the classic "3 faces visible" corner views (positive area
  — a real camera can sit inside one),
- **12** open arcs = "2 faces visible" edge-on views (measure-zero curves),
- **6** isolated points = "1 face visible, dead-on orthogonal" views (measure-zero
  points) — **these are exactly the views our task wants** ("the orthogonal view
  of face 3").
- **Total 26** — matching the well-known "26 topologically distinct views of a
  cube" figure from the vision/robotics literature (6 face + 12 edge + 8 corner).
  The paper never computes this example itself; this is my derivation from its
  stated definitions, cross-checked against that independently-known figure.

  **Caveat that matters for RQ2**: the paper's own prose is ambiguous about
  whether the 12 arcs / 6 points count as first-class aspect-graph nodes or are
  merely "boundary points" of the open regions ("the regions are open and their
  complements are closed," p.15, discussed only informally). Either reading is
  bad news for treating "orthogonal to face 3" as a native address: under the
  strict/generous reading it is one of 26 nodes but a **zero-measure** one (no
  real camera pose is exactly it); under the stricter "only open regions count"
  reading it is **not a VSP region at all** — folded into the boundary between
  two of the 8 real corner regions. Either way, the classical aspect-graph
  formalism does not hand you a stable, positive-volume "orthogonal-to-face-3"
  region — it structurally treats that viewpoint as a knife-edge.

- **Perspective, bonus check**: under perspective (§4.2, p.22–24), the box's 6
  faces (3 parallel pairs) cut ℝ³ into the familiar tic-tac-toe **3×3×3 = 27
  cells**, again vastly smaller than the generic convex bound Θ(n³) (which for
  n=6 asymmetric faces would suggest O(n³)=216-ish, matching the paper's derived
  worst-case constants in §4.2.1–4.2.2).

**Cylinder:** Plantinga & Dyer give **no** treatment of curved surfaces at all —
explicitly flagged as future work in the Conclusion (verbatim, p.51/57): *"It
would also be interesting to construct the VSP for objects with some curved
faces, such as cylinders."* The only cylinder-adjacent result is the "highly
degenerate… n-sided approximations to a cylinder… VSP has size Θ(n)" aside quoted
above (§4.1.2) — an n-gon **prism** (flat facets), not a smooth cylinder, and
still convex, so again the Θ(n⁶)/Θ(n⁹) bounds never apply. For, say, a 36-facet
prism approximation, this predicts an aspect graph of size **O(36)**, not
O(36²)=1296 and nowhere near O(36⁶)≈2.2×10⁹. **This document cannot supply a
number for a true smooth-cylinder aspect graph** — that would need the sibling
curved-object paper (DOI `10.1007/BF00054918`, same IJCV issue, unconfirmed
authorship this session — recommend as a direct follow-up read).

**Box + cylinder assembly:** Not treated in this paper at all (no discussion of
multi-object scenes beyond the paper's own adversarial lower-bound *constructions*
— see next paragraph). Structurally, an assembly is where the story changes:
gluing a cylinder to a box generally produces a **non-convex compound**, so
edge-edge-edge occlusion between the two primitives becomes possible for the
first time (box edges vs. cylinder-silhouette edges), and the paper's Θ(n⁶)
machinery is no longer categorically excluded the way it was for the box or
cylinder alone. I cannot derive a real number for this from the paper — it gives
no assembly example — but I can bound the *worry*: the paper's own Θ(n⁶)/Θ(n⁹)
lower-bound constructions (Figs 19 and 29, next paragraph) are deliberately
adversarial — dense grids of near-parallel, near-coincident repeating features
engineered to make every triple of edges generate a distinct event. A box next to
a cylinder is the opposite: few features, still mostly axis-aligned/axisymmetric.
Expect real growth close to what the box and cylinder show individually (low
tens, driven by the smaller primitive's discretization) plus a modest number of
genuinely new cross-object events at the join — not the astronomical number the
raw Θ(n⁶) formula would produce if naively plugged in (e.g. combined n≈120 →
120⁶≈3×10¹²). This is reasoned extrapolation, not a number the paper supports —
flagging the gap explicitly rather than asserting a figure.

**Adversarial lower-bound constructions (why the worst case is what it is,
§5.1.3 p.24–25 Fig 19–21, §5.2.3 p.42–43 Fig 29):** Θ(n⁶) is achieved by "two
grids of m strips each" behind "two screens, each with m slits," arranged so
sweeping the viewpoint in one direction cycles through m nearly-parallel grid
elements appearing/disappearing one at a time (Q(m³) boundaries per screen,
orthogonal to the other screen's Q(m³) boundaries, giving Q(m⁶) intersections).
Θ(n⁹) uses the 3-D analogue: three such screen/grid pairs on three faces of a
cube (Fig 29). **Both examples are attributed to John Canny [1987, personal
communication]** — worth noting, since this is presumably the actual link to a
"Canny" in this space, distinct from the mislabeled "Gigus, Canny, Seidel"
authorship in the task brief. Neither construction remotely resembles a box or a
cylinder — they are purpose-built adversarial grids. This is direct textual
support for the branch's expectation that the explosion bound doesn't bite for
primitive shapes.

## Load-bearing assumptions

- Objects are **solid, opaque polyhedra** with only flat faces; visibility model
  is hidden-line-removed line drawings, infinite image resolution (§2, p.9–10).
  Curved surfaces are explicitly out of scope (Conclusion, quoted above).
- `n` is used loosely as "number of faces" (convex/orthographic derivation) and
  "number of edges" (non-convex derivation), justified only informally by
  assuming edges/faces/vertices are all O(n) of each other — true for
  bounded-degree polyhedra (Euler's formula regime), which box and cylinder-prism
  satisfy but which breaks for e.g. a very high-facet curved-surface
  approximation.
- The convex bound assumes **general position** (no antiparallel face normals) to
  hit the true Θ(n²); the paper itself flags that parallel/antiparallel faces are
  "highly degenerate" and shrink the graph — i.e. the worst-case bound is
  explicitly *not* meant to describe symmetric primitives like a box.
- The "aspect" defining feature set is **all visible edges of the whole object at
  once** (a global definition) — the same diagnosis Dyer gives independently in
  the companion Faugeras-panel reading (`faugeras-why-aspect-graphs-not-practical.md`
  in this same directory): "aspect is a global property... a visual event is
  produced by a change in any one of [all visible features]." This paper is the
  primary technical source that panel discussion was reacting to on the size
  question.
- Offline construction is assumed throughout — no discussion of incremental
  updates, real-time construction, or noisy/point-cloud input; the object is a
  known, exact polyhedral model (§2, p.9: "we assume that the resolution of the
  image is infinite").

## What transfers to our setting

- **The Θ(n⁶)/Θ(n⁹) "explosion" is a *non-convex-only* phenomenon that a 6-face
  box and a prism-approximated cylinder never trigger**, because both are convex.
  This is the single most load-bearing, checkable fact for RQ7/RQ8: the
  combinatorial-explosion argument, as this paper defines and proves it, is
  categorically inapplicable to our two primitive shapes taken individually. It
  only becomes potentially relevant for the **box+cylinder assembly**, where
  non-convexity first appears — and even there the paper's own adversarial
  lower-bound examples show what it actually takes to hit the bound (dense,
  engineered, repeating near-parallel structure), which a simple two-primitive
  assembly does not resemble.
- **The paper's own "highly degenerate ⇒ Θ(n)" cylinder-prism aside (§4.1.2) is
  direct, if narrow, precedent that our target behaviour (orbit the cylinder,
  one address per angular increment) is exactly the linear-size regime the
  aspect-graph framework itself identifies as its best case, not its worst.**
- **The paper's own suggested fix for a VSP that's too large is hierarchical
  decomposition by object structure** — Conclusion, verbatim: *"construct a
  hierarchical VSP by constructing the VSP for each level of a hierarchical
  representation of the object."* This is 1987-vintage, paper-native precedent
  for our primitive-level (box-face / cylinder-orbit) decomposition strategy —
  independent of, and earlier than, the similar convergent recommendation in the
  Faugeras-panel responses (Pentland, Dyer) already logged in the sibling note.
- **Adjacency in this formalism is precisely "VSP regions share a boundary
  curve"**, and the paper gives a genuinely useful, concrete, checkable example
  of why that's subtler than it looks: for a tetrahedron, region `A` (1 face) and
  region `ABC` (3 faces) *are* adjacent (share the single point P where paths can
  connect through the open region `A`), while `AB` and `AC` (each 2 faces) are
  *not* adjacent, even though naively "the same point P" looks like it borders
  both — because faces are never visible edge-on, so P belongs to `A` but not to
  `AB`, `AC`, or `ABC` (§3, p.17, Fig. 8). Directly answers part of RQ5: even a
  fully worked-out, exact adjacency definition ("share a boundary in the VSP") can
  produce non-obvious, easy-to-get-wrong adjacency graphs — a warning for any
  scheme that wants to hand an LLM orchestrator a simple "next/adjacent" relation
  over primitive-face addresses.
- **Visual-event boundary types are enumerated exhaustively and are each
  individually nameable and computable from exact geometry**: horizon (great
  circle, from a face normal), edge-vertex (great-circle arc, from a point +
  edge), edge-edge-edge (quadratic curve, from 3 edges, closed-form in Eq. 1–4,
  §5.1.1.2). This is a complete, checkable answer to RQ5's "enumerate every found
  definition of adjacency" for the aspect-graph family specifically — useful even
  though (per the box arithmetic above) most of these events are irrelevant to
  primitive shapes taken individually.
- **The paper is explicit that perspective vs. orthographic is not what drives
  the size difference — dimensionality of viewpoint space is**: "The reason for
  the larger size is the higher dimensionality of viewpoint space and not the
  change from orthographic to perspective projection" (§4.2, p.22). Relevant to
  RQ13/RQ16: our actual camera constraint (fixed working-distance shell, not
  free-roaming ℝ³) is much closer to the orthographic model's assumptions
  (2-D viewpoint space) than the perspective model's, which is a load-bearing
  reason the smaller Θ(n²)/Θ(n⁶) bounds — not the Θ(n³)/Θ(n⁹) ones — are the
  relevant regime for us even if the object weren't convex.

## What does NOT transfer, and why (negative findings)

- **No curved-object treatment at all** — the paper explicitly nominates this as
  future work it did not do. Any Θ(·) number this document supplies for "a
  cylinder" is necessarily for a flat-faceted *approximation*, not the smooth
  object our system actually surveys as a point cloud. The genuinely relevant
  curved-object aspect-graph paper is one DOI away in the same IJCV issue and
  unread this session.
- **No handling of noisy or point-cloud input** — visibility/occlusion boundaries
  are derived from exact polyhedral equations (edges as exact line segments,
  §5.1.4.1 Eq. 5–10). Nothing here says whether primitive-fit boundaries computed
  from a noisy fused PLY (per the framing doc's ~4 cm fused/single-view offset)
  would perturb these boundary curves enough to matter — that's the
  primitive-structure-from-survey-clouds branch's territory, not answerable from
  this document.
- **No assembly / multi-primitive-object treatment** — the only multi-feature
  "scenes" in the paper are the adversarial worst-case constructions (grids
  behind screens), not anything resembling a box+cylinder inspection object. The
  RQ7 assembly number cannot be derived from this paper; see the reasoned
  extrapolation above, clearly marked as such.
- **No node/edge count for any real object** — the paper proves asymptotic
  bounds and gives one exact upper-bound formula (`2n(n−1)`, convex orthographic)
  but never works a concrete example numerically (no tetrahedron count, no cube
  count, no cylinder-prism count with an actual `n`). The 26-node box figure and
  the 8/12/6 breakdown above are entirely my own derivation from the paper's
  stated definitions and formulas, cross-checked against an independently known
  figure — not a number stated in the text. Flagging this so it's checkable
  and not mistaken for a quoted result.
- **No benchmark, no comparison to a uniform/geometry-agnostic tessellation on any
  task** — reinforces the same absence already logged from the
  `benchmarking-view-planning` branch and the Faugeras-panel note: this remains
  unbuilt in the literature surveyed so far.
- **No addressing-for-action discussion** — like the Faugeras panel, this paper's
  entire framing is recognition/characteristic-views (§1, p.1: "3-D object
  recognition… storing several characteristic views"). It never considers naming
  a region so a robot can be commanded there; the aspect graph here is a
  passive catalog for image matching, not an action space. Reinforces (does not
  newly establish) the sibling note's finding that the size-explosion critique
  and the action-addressing use case are different problems.
- **No seam-naming machinery** — the paper's boundaries (great-circle arcs,
  quadratic curves) are exactly the "seams" RQ5/RQ7's aspect-graph-death question
  cares about, and per the box arithmetic above they are treated ambiguously (see
  the 8-vs-26-node discussion) — the paper does not resolve whether a seam gets
  its own stable name/address or is merely a discarded zero-measure boundary.
  This ambiguity is itself the finding, not a gap in my reading.

## Citations (for checkability)

- Title page / byline (Plantinga & Dyer, not Gigus/Canny/Seidel): p.1 (OCR
  `page-03.txt`).
- Abstract, exact Θ(n²)/Θ(n⁶) orthographic and Θ(n³)/Θ(n⁹) perspective bounds:
  p.1/Abstract (`page-03.txt`).
- Table 1 (VSP/aspect-graph max size) and Table 2 (construction time): p.2 of
  paper (`page-08.txt`).
- "In an earlier paper… incorrectly claims… O(n⁴)… overlooking edge-edge-edge
  occlusion… We thank John Canny [1987] for pointing out the error": p.2
  (`page-08.txt`) — direct evidence a "Canny" is genuinely present in this
  paper's lineage, as an acknowledged error-catcher and (per below) originator of
  the worst-case lower-bound examples, plausibly the true source of the task
  brief's author confusion.
- VSP/aspect graph formal definitions, tetrahedron example (Figs 4–7), A/ABC vs.
  AB/AC adjacency subtlety (Fig. 8): §3, p.7–11 (`page-13.txt`–`page-17.txt`).
- Convex case restricted to horizon effect only: §4 intro, p.12
  (`page-18.txt`–`page-19.txt`).
- Convex orthographic upper bound, exact `2n(n−1)` formula, O(n²): §4.1.1, p.13
  (`page-19.txt`).
- "Highly degenerate… n-sided approximations to a cylinder… VSP has size Θ(n)":
  §4.1.2, p.14–15 (`page-20.txt`–`page-21.txt`).
- Convex perspective bound, O(n³): §4.2–4.2.2, p.16–17 (`page-22.txt`–`page-23.txt`).
- Edge-vertex occlusion, 4 sub-cases, Figs 13–16: §5.1.1.1, p.19–20
  (`page-25.txt`–`page-26.txt`).
- Edge-edge-edge occlusion, Eq. 1–4, complete-catalog argument: §5.1.1.2, p.21–23
  (`page-27.txt`–`page-29.txt`).
- Non-convex orthographic upper bound O(n⁶): §5.1.2, p.23 (`page-29.txt`).
- Non-convex orthographic lower bound Θ(n⁶), grids-behind-screens construction
  "due to John Canny [1987]", Figs 19–21: §5.1.3, p.23–25
  (`page-29.txt`–`page-31.txt`).
- Non-convex orthographic algorithm, brute-force Ω(n⁷) vs. asp-based O(n⁶ log n),
  convex-case-via-brute-force-still-Ω(n⁷) caveat: §5.1.4, p.25–26
  (`page-31.txt`–`page-32.txt`).
- Non-convex perspective upper bound O(n⁹): §5.2.2, p.41–42
  (`page-47.txt`–`page-48.txt`).
- Non-convex perspective lower bound Θ(n⁹), three-screens-on-a-cube construction,
  Fig. 29: §5.2.3, p.42–43 (`page-48.txt`–`page-49.txt`).
- "Reason for the larger size is the higher dimensionality… not… orthographic or
  perspective": §4.2, p.16 (`page-22.txt`).
- Related work — Koenderink & van Doorn's "visual potential graph," their own
  quote that cell decomposition "has not yet been solved by geometers" [1976,
  p.57]: §6, p.48 (`page-54.txt`).
- Related work — characteristic-view counts in practice, "ranging from a small or
  moderate constant (up to about 300)… to every topologically distinct view":
  §6, p.49 (`page-55.txt`).
- Conclusion — restated Θ bounds, "algorithms for object recognition that work on
  relatively complex objects probably cannot use all topologically-distinct
  views": p.50 (`page-56.txt`).
- Conclusion — hierarchical-VSP suggestion, curved-object (cylinder) future work:
  p.51 (`page-57.txt`).
- References list, incl. Canny's unpublished 1984 IBM report and 1987 personal
  communication (both cited as sources for the lower-bound constructions):
  p.52–53 (`page-58.txt`–`page-59.txt`).

## Assessment for the branch

This paper is the actual primary source for the Θ(n²)/Θ(n⁶)/Θ(n³)/Θ(n⁹) numbers
the task brief asked to be evaluated — but is by Plantinga & Dyer, not Gigus,
Canny & Seidel (flagged above, first-class correction, not decoration). Worked
through for our objects, the headline result is unambiguous and arithmetic, not
judgment, exactly as RQ7 demanded: **a 6-face box is convex, so the entire
Θ(n⁶)/Θ(n⁹) explosion apparatus (edge-edge-edge occlusion) cannot fire on it at
all** — only the Θ(n²) convex bound applies, and the box's own antiparallel-face
symmetry shrinks that further, to an exact, hand-derivable **26 total
topologically-distinct viewpoint-space elements** (8 real, positive-area "corner"
regions; 18 zero-measure edge/point boundaries — including, awkwardly, the 6
single-face orthogonal views our task actually wants to name). A cylinder,
treated as the paper's own explicitly-flagged "highly degenerate" n-gon-prism
case, gets Θ(n) — linear, not quadratic, and again categorically excluded from
the non-convex bound because it too is convex. The one place the explosion
argument could genuinely start to bite is the **box+cylinder assembly**, the
first non-convex case among our three targets — and this paper supplies no
assembly example, only a reasoned (clearly-flagged, not paper-sourced) expectation
that a two-primitive assembly looks nothing like the paper's own adversarial
worst-case constructions. Net effect on the branch: the size-explosion objection,
computed honestly rather than asserted, does not bite our box or our cylinder
individually, and the paper's own stated remedy for when it might ("hierarchical
VSP… for each level of a hierarchical representation of the object") is
independent, 1987-vintage precedent for exactly the primitive-decomposition
strategy the project has already chosen. The sharper and less comfortable finding
is RQ2's: even for the box, the classical formalism does not deliver "orthogonal
view of face 3" as a stable, positive-measure address — it is either absent or a
knife-edge, which is textual, structural support for why the framing doc's
complaint (no view in a uniform grid is ever exactly orthogonal) is not a
grid-resolution problem but a category property of treating aspect purely as
image topology.
