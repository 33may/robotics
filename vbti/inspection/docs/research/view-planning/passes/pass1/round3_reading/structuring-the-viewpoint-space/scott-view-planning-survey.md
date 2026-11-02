# Scott, Roth, Rivest — "View Planning for Automated Three-Dimensional Object Reconstruction and Inspection"

ACM Computing Surveys 35(1):64–96, March 2003. DOI 10.1145/641865.641868 (ACM DL 403s on
direct fetch — full text obtained via the authors' NRC accepted-manuscript archive,
https://nrc-publications.canada.ca/eng/view/accepted/?id=c9a6149f-27d0-4269-8122-80dd6104c0a1,
33-page PDF, read in full including the reference list). This is the canonical survey of the
pre-2003 view-planning field for active triangulation range scanners; it is not itself a new
algorithm but a requirements framework (§3), a categorized literature survey (§5–6), a
comparison table across 10 algorithms (§7, Table III), and a critique (§8–10). Authors are NRC
Canada (Scott, Roth) and U. Ottawa (Rivest); Scott's own PhD work (measurability-matrix set
covering, cited throughout) is folded in as one of the surveyed methods.

## Core content relevant to the branch

**Formal problem setup (§2.1, p.65–67).** A "generalized viewpoint" is defined as a pair
`(v, λs)` — sensor pose `v` (up to 6D+3 rotation) plus a vector of controllable sensor
parameters `λs` (laser power, scan length). Viewpoint space `V` is the set of all such pairs;
object surface space `S` is the set of sampled 3D surface points; imaging workspace `I` is the
volume visible over the full range of the positioning system. The View Planning Problem (VPP,
§2.3, p.68) is: find a short view plan `N ⊂ V` satisfying reconstruction/inspection goals in
acceptable time. **The VPP is stated as NP-complete** (citing Tarbox and Gottschlich [1995] and
Scott [2002]), which is why the paper insists throughout that no algorithm should chase a
provably minimal view plan — heuristics/near-optimal solutions are the accepted target. This is
the formal, citable basis for treating our RQ2/RQ3 tractability requirement as a design
constraint, not a nice-to-have.

**Tarbox and Gottschlich's measurability matrix — the bipartite structure named in the task
brief (§5.1, p.74–75; revisited §9.1, p.88–89).** Their method:
- Discretizes viewpoint space by **recursive subdivision of an icosahedron**, constrained to a
  sphere fully enclosing the object, viewing direction always toward the sphere center.
- Because the sensor is bistatic (laser source + optical receiver separated by a baseline),
  viewpoint space is redefined as **the set of all ordered pairs of sphere points separated by
  the sensor's optical baseline distance** — for a tessellated icosahedron this gives "five or
  six orientations about the sensor boresight per viewing axis" (§9.2.3, p.90).
- Builds a **measurability matrix `M(i,j)`** — rows = discretized surface points, columns =
  discretized viewpoints, entries = a computed measurement-quality score (not just a
  visibility bit) from a complete visibility analysis over all surface points × all admissible
  viewpoints. This is exactly the bipartite viewpoint↔surface-patch structure the branch was
  told to expect.
- **Computational cost was "prohibitive"** — stated as "a fundamental limitation regarding
  direct applicability of the work to object reconstruction" (p.75). No number is given beyond
  "prohibitive"; the paper does not quantify it further.
- Three algorithms tested: two irrevocable-selection variants differing in how they handled
  grazing-angle constraints, and a third **revocable-selection strategy using randomized search
  with simulated annealing** for viewpoint-set minimization (p.74–75).
- Pose-error robustness: viewpoints not robust to pose variation were removed by **morphological
  erosion on the viewpoint set** associated with each surface point (p.75, §9.3 p.91) — i.e.
  addresses near a feasibility/robustness boundary get pruned from the address space itself
  rather than flagged, a concrete precedent for the branch's "does the address survive
  infeasibility" question (though it applies to pose-error robustness, not IK/collision).
- **Scott et al. [2001a] later reformalized the same matrix as a set-covering problem**: "the
  view planning problem can be expressed as the problem of covering the rows of a binary
  measurability matrix by a minimal subset of its columns," with an **integer-programming
  formulation including a registration constraint** (§9.1, p.88–89). This is the formal
  optimization framing behind the matrix, independent of the discretization scheme used to
  build the matrix's columns — the matrix itself is discretization-agnostic; icosahedron
  tessellation is just the specific column-generator Tarbox and Gottschlich chose.

**Aspect graphs — rejected, with a reason not present in the Faugeras panel piece (§5.2, p.75).**
The paper defines an aspect graph exactly as the classical literature does — nodes = qualitatively
equivalent viewpoint regions, arcs = adjacency in viewpoint space — then gives **two independent
grounds for rejection specific to reconstruction/inspection**, not recognition:
1. **Discrete→continuous breakdown for curved objects.** "When dealing with objects as simple
   as polyhedra... the criterion 'qualitatively the same' simply means that the same vertices,
   edges, and faces are visible... When the object is changed from discrete to continuous, the
   quantization of viewpoint space inherent to an aspect graph representation **strictly
   fails**." The paper notes silhouette-topology aspect graphs remain definable for curved
   objects but calls this "not useful for object reconstruction" (p.75). **This is a new,
   explicit statement that aspect graphs are well-defined for a box (polyhedral) but break down
   for a cylinder (curved) under the feature-topology definition** — directly on point for the
   branch's box-vs-cylinder test case.
2. **Visibility equivalence ≠ measurement equivalence — the paper's own core objection, not
   found in the Faugeras et al. panel discussion.** "As we are dealing with a sensing problem
   which involves not only visibility but mensuration, **viewpoints with the same visibility
   (i.e., the same aspect) are by no means equivalent for sensing purposes**" (p.75). Two
   viewpoints inside the same aspect-graph cell can differ arbitrarily in grazing angle,
   stand-off distance, and therefore measurement precision/sampling density — quantities the
   aspect graph node carries zero information about. This is a distinct, additional failure
   mode beyond the Faugeras panel's size/indexing objections: **even a computable, correctly
   sized aspect graph answers the wrong question for view planning**, because its equivalence
   classes are defined by feature-topology, not by the metric quality the task actually cares
   about (orthogonality/grazing angle in our case).
3. Restates the size objection with the same source as the prior pass note: "aspect graphs for
   even moderately complex objects quickly become huge and unwieldy... aspect graph theoretical
   development is not sufficiently mature [Faugeras et al. 1992] to be a suitable representation
   basis for performance-oriented view planning" (p.75) — no new numbers beyond the panel piece
   already read.

**Art gallery problem (§5.3, p.75).** Given only two sentences. Classic 2D result: minimum guard
placement to see all walls of polygon `P`. The paper's only substantive claim: object
reconstruction is "somewhat related" but has **three added complexities art-gallery formulations
don't have** — three dimensions (not 2D), **bistatic visibility** (source and receiver must both
see the point, not a single omnidirectional guard), and **mensuration in lieu of visibility**
(again: seeing a point isn't sufficient, measuring it to spec is the actual requirement). No
algorithm, bound, or citation to a 3D/bistatic art-gallery variant is given — the comparison is
asserted, not developed. This is a negative finding: **the art-gallery formulation is not
actually pursued anywhere in this survey's 30-algorithm review**; it appears only as a
one-paragraph aside noting why it doesn't directly transfer.

**The view-sphere critique (§9.2.3, p.89–90; §9.2.5, p.91) is a direct, pre-existing diagnosis of
our project's current `{h,v}` scheme.** The paper states the *majority* of view-planning
literature reduces viewpoint space to a sphere/circle around the object (cites Connolly, Banta,
Garcia, Massios & Fisher, Morooka, Reed, Tarbox and Gottschlich, Yi — 8 independent papers using
this stratagem), then gives the exact failure mode our framing doc describes:
- "**To the degree that object shape is not spherical, viewpoint position is nonoptimal**"
  (p.90) — this is a verbatim restatement of our framing doc's problem statement, from a
  2003 survey, about generic reconstruction, independent of any VLM/LLM concern.
- "the view sphere approach... **fails to optimize position** or some aspects of orientation...
  there is more to achieving good viewing directions than merely selecting the orientation of
  the sensor boresight. The position from which the image is taken is also important" (p.90).
- "the view sphere approach **says nothing about the orientation of the camera around its
  boresight** (the twist angle)" (p.90) — a dimension our `{h,v}` scheme also doesn't address
  (all views vertical, per the framing doc), consistent with this being the field's default
  simplification rather than a project-specific oversight.
- Discretization variants surveyed: parallels/meridians (simple, nonuniform), tessellated
  icosahedron (near-uniform, most popular, "complex computations regarding cell occupancy"),
  spherical distribution maps / SDM (intermediate, fast mapping, inferior uniformity to
  icosahedron) — this is a useful three-way trade-off table for RQ13/RQ17 if a uniform-sphere
  baseline is still wanted for comparison, but the paper's verdict is that **all three are the
  same category of geometry-blind discretization** and share the position/orientation
  optimality failure above regardless of which one is used.

**Table III (§7, p.86) — the systematic version of RQ20's absence finding.** The paper builds an
explicit `Y`/`N`/`P`/`?`/`—` comparison table across 10 algorithms (Maver & Bajcsy, Connolly,
Yuan, Banta et al., Massios & Fisher, Tarbox & Gottschlich, Reed et al., Whaite & Ferrie, Pito,
Papadopoulos-Orfanos) against 16 requirements. **"Model quality specification" is `N` for all 10
algorithms without exception** — no reviewed method has any explicit, quantified pass/fail
criterion tying view selection to reconstruction/inspection quality (§8.1.1, p.85; §8.5.4, p.87:
"No current method includes explicit pass/fail quality criteria on the outcome. Generally, there
is a lack of clarity with respect to the objective."). **6D pose is `Y` only for Pito**; every
other algorithm restricts to 1–2D viewpoint space. **Efficiency is mostly `?` or `N`** — almost
no algorithm reports computational complexity or timing in a form comparable across methods.
This is a hard, table-form confirmation (not just an assertion) that **no benchmark comparing
view-planning approaches on a shared quantitative task/metric existed as of 2003** — direct,
citable support for RQ20's absence finding, from a source independent of whatever the
benchmarking-view-planning branch already found in more recent literature.

**Inspection vs. reconstruction — precedent for our survey-then-inspect split (§1, p.65–66).**
"While dealing mainly with object reconstruction, the techniques examined here also apply to
inspection. The latter is an **easier problem, having the advantage of a preexisting model**.
Essential information can be collected **off-line in a suitable data structure**, allowing
different tasks to be planned **quickly online**." This is a direct, if generic, precedent for
building an addressing/view-planning structure once from a completed survey (our fused PLY) and
querying it cheaply during the inspection phase — matching our project's survey-then-inspect
session structure. The paper gives no build-time numbers for this split, though — it's a framing
claim, not a measurement (relevant to RQ17: no numeric answer here, only structural precedent).

## Concrete results and numbers

- VPP is NP-complete (asserted, citing Tarbox and Gottschlich [1995]; Scott [2002] — proof not
  reproduced in this survey).
- Tarbox and Gottschlich's icosahedron + bistatic-pair scheme yields **5–6 boresight
  orientations per viewing axis** (p.90) — the only concrete cardinality number in the paper
  related to a discretized viewpoint scheme.
- Optical baseline subtends **35°–50°** at a sensor's optimal stand-off range for mid-to-high
  performance triangulation cameras (p.90) — relevant only as domain context for grazing-angle
  sensitivity, not to addressing per se.
- Table III: 10 algorithms × 16 requirements, virtually all `N`/`?` — no positive quantitative
  result, this is the paper's central negative finding, presented as a table not a narrative.
- No node/edge counts, no build-time figures, no Θ(·) bounds for aspect graphs or any other
  structure are given anywhere in this paper (defers entirely to Faugeras et al. [1992] for
  aspect-graph scale claims, already covered in the sibling reading-notes file in this folder).

## Load-bearing assumptions

- The entire survey assumes **active, triangulation-based laser range scanners** — bistatic
  sensors with a nonnegligible optical baseline (§2.1, p.66–67). Our D405 is a stereo/structured
  RGB-D sensor, not a triangulation laser scanner; the specific bistatic-shadow and
  baseline-separation mechanics (source shadow ≠ receiver shadow, Tarbox's paired-point scheme)
  do not transfer directly — our camera is effectively monostatic for addressing purposes, so
  the exact viewpoint-pairing construction Tarbox uses to build the measurability matrix's
  columns is not needed for us, though the matrix concept itself (viewpoint × surface-patch,
  entries = quality not just visibility) is sensor-agnostic and does transfer.
- §3.1 (p.69) assumes calibrated sensor + positioning system, workspace compatible with object
  size, and "reasonable" object shape/topology and "benign" reflectance — explicitly excludes
  deep cavities, multiple protrusions, highly specular or volume-scattering materials. Consistent
  with our own primitive-shape, non-goal-excluded-complex-geometry scoping.
- Virtually every surveyed method (all except Pito) constrains viewpoint space to 1–2D
  (view sphere/circle) with boresight fixed toward an object-centered origin (Table III, "6D
  pose" column) — meaning the field's default *is* a uniform-sphere-style reduction, which
  matters for calibrating how radical our geometry-conditioned departure actually is relative to
  field norms (it's a departure from the field's own historical default, not merely from our own
  earlier prototype).
- The paper's object-reconstruction framing generally assumes the object shape is **unknown a
  priori** (a next-best-view / autonomous-exploration setting) — this differs from our setting,
  where the object has already been surveyed into a known (if noisy) point cloud before view
  planning begins. This is the single most important framing mismatch for how far any given
  method here transfers (see below).

## What transfers to our setting

- **The measurability-matrix formalism (bipartite viewpoint × surface-patch, entries = quality
  not just visibility bit) is a legitimate general blueprint for RQ1/RQ2**, independent of
  Tarbox's specific icosahedron/bistatic-pair column generator. Building `M(i,j)` over our own
  addressed views (primitive faces/orbit angles) × surface patches, with entries scored by
  orthogonality/grazing-angle quality (exactly the "quality" factor Massios, Tarbox, and Pito all
  used, per §8.1.5, p.86), is a direct, off-the-shelf way to formalize RQ1's "how orthogonal, how
  well framed" scoring and RQ2's view-selection-as-set-covering framing (Scott et al. [2001a]'s
  integer-programming reformulation, §9.1).
- **The aspect graph's mensuration-vs-visibility objection is new and load-bearing for RQ9.**
  Beyond size/computability/indexing (already covered by the Faugeras panel notes), this paper
  adds a fourth, independent reason aspect graphs are the wrong tool: their equivalence classes
  are defined by *what's visible*, not by *how well it can be measured* — and orthogonality is
  precisely a measurement-quality property invisible to a feature-topology aspect definition.
  This applies even if node-count and indexing were both solved. Strengthens the branch's likely
  conclusion that aspect graphs are dead for us regardless of which failure mode is invoked.
- **The discrete/continuous breakdown claim (aspect graphs "strictly fail" for continuous/curved
  objects) is the most concrete literature statement found so far that the aspect-graph
  representation itself is polyhedron-shaped** — it degrades specifically for our cylinder case,
  not just at scale. Directly answers part of RQ9's "does it transfer when curved primitives are
  involved" question: no, by this paper's own explicit statement.
- **The view-sphere critique (§9.2.3/9.2.5) is independently-derived, pre-VLM confirmation of
  the entire branch's premise** — a uniform sphere is position/orientation-suboptimal whenever
  object shape isn't spherical, from a source with zero interest in LLM orchestration. This is
  useful as an appeal to a broader, older, non-LLM-motivated literature consensus when writing
  the synthesis, rather than relying only on our own framing doc's assertion.
- **Table III's systematic `N` on model-quality-specification is strong, citable support for
  RQ20** — no comparative benchmark existed across 10 major algorithms as of 2003; the field's
  own internal comparison table shows this as a structural absence, not an oversight of any
  single paper.
- **Offline-build/online-query for inspection (vs. reconstruction) is direct textual precedent**
  for our survey-then-inspect session split (§1, p.65–66) — even though no build-time numbers are
  given, the structural claim ("collected off-line... planned quickly online") supports treating
  our primitive-fit-then-address-generation step as a legitimate, expected architecture rather
  than an ad hoc simplification.
- Pose-error-based address pruning (Tarbox's morphological erosion of viewpoints near the
  boundary of their feasible set, p.75/p.91) is a concrete, if narrow, precedent for one of the
  three fallback behaviors the branch asks about for infeasible ideal views (vanish from the
  menu) — though scoped to pose *error* robustness, not IK/collision infeasibility.

## What does NOT transfer, and why (negative findings)

- **No treatment anywhere in this paper of addressing views for an agent/orchestrator to select
  from.** Every method surveyed is a closed-loop autonomous NBV algorithm: the system itself
  computes and executes the next view with no external selector in the loop. There is no
  concept of a discrete menu handed to a separate decision-maker, no adjacency semantics for
  navigation purposes, no naming/ordering convention beyond raw discretization indices. This
  paper cannot be cited for RQ4 (ordering), RQ5 (adjacency definitions beyond "aspect-graph arc
  = adjacency," already known), RQ6 (does the agent need adjacency), RQ7 (bound arithmetic — no
  numbers given), RQ11 (variable arity across object classes), RQ13 (canonical-frame machinery),
  or RQ14 (bounded-cardinality guarantees) — these questions are simply out of scope for a 2003
  autonomous-NBV survey and the paper does not gesture toward them even indirectly.
- **The field's dominant setting (unknown object shape, explore-to-reconstruct) does not match
  ours (known/surveyed shape, plan-to-inspect).** Most surveyed algorithms (Connolly, Banta,
  Massios & Fisher, Whaite & Ferrie, Papadopoulos-Orfanos) are next-best-view exploration methods
  reasoning over partial/unknown volumetric occupancy — this entire machinery (voxel occupancy
  grids, octrees, occlusion-edge cuing, space carving) is upstream of our problem and largely
  irrelevant: we already have the fused cloud before view planning starts. Only the *inspection*
  framing (§1, "easier problem... preexisting model") and the measurability-matrix/set-covering
  formalism (which is agnostic to whether the model was pre-existing or built online) are the
  parts of this paper that actually apply.
- **No primitive/part-based addressing precedent here**, unlike the Faugeras panel piece's
  Pentland/Dyer responses. This survey's methods are either fully global (whole-object view
  sphere) or locally cued by occlusion edges of the *current partial model* (not by fitted
  primitive faces of a known shape) — nothing here anticipates our "box face-by-face, cylinder
  orbit" primitive-level addressing scheme. That precedent remains solely in the
  faugeras-why-aspect-graphs-not-practical.md notes in this same folder, not here.
- **Art-gallery formulation is a one-paragraph aside, not a developed method** — no algorithm,
  bound, or worked example anywhere in the paper's 30-algorithm survey uses it. It cannot be
  cited as a competing implementable scheme, only as a two-sentence "related but insufficiently
  different" note from the authors themselves. This confirms the branch's premise that the
  art-gallery framing is aspirational/comparative rather than a real candidate to build on.
- **No adjacency, ordering, or seam treatment for any discretization scheme.** Icosahedron
  tessellation, parallels/meridians, and SDM are compared only on uniformity and cell-mapping
  speed (§9.2.3, p.90) — never on adjacency semantics, seam handling, or degenerate cells (the
  icosahedron's own irregular vertices, analogous to H3's 12 pentagons, are not discussed at
  all). This document is silent on RQ7's seam-naming question.
- **No benchmark comparing a geometry-conditioned view space against a uniform sphere** — the
  paper's own comparison (Table III) is across different *selection algorithms*, all of which
  use view-sphere-style or occlusion-cued discretizations; no method in the 30-paper survey
  conditions the discretization itself on object geometry the way our target design requires.
  Reinforces the absence finding from RQ20 rather than filling it.

## Citations (for checkability)

- Formal VPP definition, `(v, λs)`, NP-completeness: §2.1 p.65–67; §2.3 p.68.
- Tarbox and Gottschlich measurability matrix, icosahedron + bistatic-pair viewpoint space,
  "prohibitive" complexity, three algorithms (2 irrevocable + 1 revocable/simulated-annealing):
  §5.1 p.74–75.
- Set-covering / integer-programming reformulation with registration constraint (Scott et al.
  2001a): §9.1 p.88–89.
- Aspect graph definition, discrete→continuous "strictly fails" for curved objects, visibility ≠
  mensuration objection, size/maturity objection citing Faugeras et al. [1992]: §5.2 p.75.
- Art gallery problem, 3D/bistatic/mensuration gap: §5.3 p.75.
- Inspection as "easier problem... preexisting model... off-line... quickly online": §1 p.65–66.
- View-sphere critique ("to the degree that object shape is not spherical, viewpoint position is
  nonoptimal"; position vs. orientation; boresight twist angle not addressed): §9.2.3 p.89–90;
  §9.2.5 p.91.
- Icosahedron/parallels-meridians/SDM discretization comparison: §9.2.3 p.90.
- Table III comparison of 10 algorithms × 16 requirements, all `N` on model quality
  specification: §7 p.86; discussion §8.1.1 p.85, §8.5.4 p.87.
- Grazing angle used as subjective quality factor by Massios, Tarbox, Pito: §8.1.5 p.86.
- Pose-error viewpoint pruning by morphological erosion (Tarbox and Gottschlich): §5.1 p.75;
  §9.3 p.91.
- Optical baseline subtends 35°–50° at optimal stand-off: §9.2.3 p.90.

## Assessment for the branch

This is a foundational-but-generic survey, not a proposal for our problem. Its direct value to
the branch is threefold: (1) it independently confirms, from 2003 non-LLM literature, that the
uniform-view-sphere failure mode our framing doc describes is a known, named, field-wide problem
(§9.2.3), which is useful ballast when writing the synthesis; (2) it gives the branch its
clearest evidence yet for *why* aspect graphs are wrong for us beyond size/indexing — their
equivalence classes track visibility, not measurement quality, which is a category mismatch with
an orthogonality-driven inspection objective, and this mismatch is stated to be terminal even for
polyhedra, and explicitly worse (representation "strictly fails") for curved primitives like our
cylinder; (3) it hands RQ1/RQ2 a ready-made formalism — a viewpoint × surface-patch
measurability matrix scored by measurement quality, later reducible to set covering — that is
sensor- and discretization-agnostic and could underlie the benchmark the project's own framing
doc calls for. It contributes nothing to RQ4–RQ7, RQ11, RQ13, RQ14 (agent-facing addressing,
ordering, seams, cross-shape stability) since no surveyed method here ever hands a discrete menu
to an external selector — those questions remain to be answered by literature closer to the
active-vision / robot-learning side of the branch's scope, not the classical view-planning side.
