# Magaña, Dirr, Bauer, Reinhart — "Viewpoint Generation using Feature-Based Constrained Spaces for Robot Vision Systems"

Robotics 12(4):108, 2023 (MDPI). arXiv:2306.06969v1 [cs.RO], 12 Jun 2023, CC BY 4.0.
Authors: Institute for Machine Tools and Industrial Management, TU Munich. Funded by the
"CyMePro" project with AUDI AG and Carl Zeiss Optotechnik GmbH as industry partners.

Full text obtained from the arXiv HTML rendering (`arxiv.org/html/2306.06969v1`, raw HTML
downloaded and parsed locally — MDPI's own page returned HTTP 403). All 9 sections read in
full, including all equations, both algorithms-heavy sections (VI, VII), and the evaluation
(VIII) and outlook (IX). Section/equation/table numbers below match the paper.

## Core idea

The paper solves the **Viewpoint Generation Problem (VGP)**: given one feature to acquire
and a set of viewpoint constraints, find valid sensor poses. It reframes VGP as a
**Configuration-Space problem in SE(3)** (§IV-E, borrowing directly from motion-planning
C-space theory, quoting Latombe: "once the configuration space is clearly understood, many
motion planning problems... can be solved by the same planning algorithms"). Instead of
optimizing toward one ideal pose, each viewpoint constraint c_i is given a closed-form
geometric region **𝒞_i ⊆ SE(3)** — the set of *all* sensor poses that satisfy that one
constraint (Eq. 11: 𝒞_i = 𝒯_s × ℛ_s). The joint feasible region is the intersection of all
per-constraint spaces:

    𝒞(C̃) = ⋂ 𝒞_i(c_i)   (Eq. 12)

and a viewpoint exists iff 𝒞 ≠ ∅ (any pose in 𝒞 is valid — this is the paper's central
"robustness" pitch: infinitely many valid solutions, not one optimum). Nine constraint types
are given closed-form 𝒞-space formulations (Table I): (1) frustum containment, (2) sensor
orientation/incidence angle, (3) feature geometry (whole-feature visibility), (4) kinematic
error, (5) sensor accuracy, (6) occlusion, (7) bistatic/multisensor, (8) robot workspace,
(9) multi-feature.

**The mechanism that matters most for us (§VI-A):** the core constrained space 𝒞₁ is built
by taking the sensor's frustum (a hard-coded 8-vertex polytope — the pyramid/frustum shape
of its FOV × depth-of-field, Eq. 4) and **reflecting it point-wise through the feature's
frame** (a homeomorphism h, Eq. 16, "reflection... over the geometric center of the frustum
space"). The result is a homeomorphic (topologically identical) copy of the frustum,
positioned at the feature, for one **fixed sensor orientation** r_s^fix. Any pose inside this
polytope acquires the feature satisfying the imaging constraint — the polytope itself is the
"region an optimizer could climb inside." Two equivalent construction methods are given
(Algorithm 1 "Extreme Viewpoints," Algorithm 2 "Homeomorphism") — both are pure translate/
reflect/scale operations on 8 vertices, not search or sampling.

**Orthogonality is exactly the incidence-angle constraint (§III-E5, Eq. 5):**
φ_s^f = arccos(n_f·e_s^z / |n_f||e_s^z|), required to satisfy φ_max > |φ_s^f|. φ=0° is the
"orthogonal to face" address. There is no dedicated named "orthogonal" constraint — you get
it by fixing r_s^fix so the sensor's optical axis is anti-parallel to the feature normal,
then §VI-A builds 𝒞₁ for that fixed orientation. **"The orthogonal view of face 3" is
literally: pick r_s^fix = feature normal, run Algorithm 2, get an 8-vertex polytope of valid
positions — closed form, no search.** This directly matches the branch's framing-doc premise
("orthogonal to face 3 as a native address plus the region an optimizer could climb inside").

**Range of orientations (§VI-B) — and its sharp limitation.** If you don't want a single
fixed orientation but a tolerance band [r_min, r_max], the paper's only mechanism is to
sample discrete orientations r_s,m within the band, build 𝒞₁ for each, and CSG-intersect them
(Eq. 18: 𝒞₂(R_s) = ⋂ 𝒞₁(r_s,m)). Critical negative property, stated explicitly (§VI-B3,
"Discretization without Interpolation"): **𝒞₂(R_s) is only proven valid for the sampled
orientations in R_s — arbitrary orientations strictly between r_min and r_max are NOT
guaranteed to be inside the resulting manifold.** They show empirically (Eq. 20, Fig. 8) that
coarse-vs-fine discretization gives "almost identical" manifolds for their case, and verify
two extra untested orientations happen to work (§VI-B4), but state this "cannot be
guaranteed for any other arbitrary orientation" and "must be individually assessed for each
application" (§VI-B5). So the address does *not* get a free continuous interior over
orientation — only over translation, for one fixed rotation.

**Constraint composition, second mechanism (§VII-B) — scaling vectors, not always CSG.**
Several constraints (feature geometry Δ in Eq. 22-23, kinematic error ε, sensor accuracy a_s)
are formulated as **per-vertex scaling/offset vectors** applied directly to 𝒞₁'s 8 vertices
(Eq. 27-28), rather than as independent manifolds requiring CSG intersection. This is
explicitly the fast path: "the complexity of CSG operations is at least proportional to the
number of vertices... the complexity of the sum [Eq. 28] is just proportional to the number
of viewpoint constraints" (§VII-B1). The paper recommends prioritizing scaling-vector
formulations over CSG whenever a constraint admits one, and reserves CSG intersection for
constraints that can't be reduced to a vertex offset (occlusion, workspace, orientation
range, multi-device, multi-feature).

## Concrete results and numbers

**Timing (all on a Lenovo W530: i7-4810MQ @2.80GHz, 32GB RAM, single-threaded Python/ROS
Noetic, PyMesh for CSG, trimesh+Embree for ray-casting):**

- Single 𝒞₁ (frustum reflected to fixed orientation, 8 vertices, no CSG): **4.1 ms ± 2.4 ms**
  (30 repetitions) (§VI-A5).
- 𝒞₃ (feature-geometry-scaled, still pure vertex arithmetic, no CSG): **5.8 ms ± 3.2 ms**
  (§VII-A6) — confirms the scaling-vector path stays ~constant-time regardless of the extra
  constraint.
- 𝒞₂, range of 3 orientations per axis (3³=27 combinations, CSG-intersected): **15 s**
  (§VI-B4).
- 𝒞₂, finer discretization, 5×5×3=75 combinations: **40 s**. 11×11×3=363 combinations:
  **720 s** (§VI-B4). **This is a superlinear/roughly-quadratic-or-worse blowup in CSG
  intersection count** (27→15s, 75→40s, 363→720s: count grows 13.4×, time grows 48×) —
  the paper's own general-notes caveat (§VI-B5, IX-B) that CSG cost scales badly is borne
  out by its own numbers.
- Volumetric payoff for that 48× time cost is negligible: Vol(𝒞₂ at 1°)/Vol(𝒞₂ at 2.5°) =
  **0.9999**; Vol(𝒞₂ at 1°)/Vol(𝒞₂ at 5°) = **0.9995** (§VI-B4) — finer discretization buys
  almost nothing here, an explicit argument for staying coarse.
- Occlusion space for one occluding body (icosahedron), one feature: ray-casting step
  **0.7 s**; total occlusion-space characterization **1.32 s** (§VII-C3).
- Multisensor (2 imaging devices) joint 𝒞-space: **200 ms** total, of which the CSG
  intersection step is **100 ms** (§VII-D3) — but flagged as not generalizable ("depends
  decisively on the number of vertices of the manifolds of the occluding objects").
- Full academic scenario (2 sensors, 4 imaging devices, 3 features, all 9 constraint types,
  Table III): **≈50 s total**, but explicitly caveated as including IK/ROS-service overhead,
  not a clean C-space-only number (§VIII-A2: "this time comprises other computation steps...
  which distort the effective computation time... a proper analysis... remains to be further
  investigated").

**No accuracy, success-rate, or coverage percentage numbers anywhere in the paper.** Both
the simulation study (§VIII-A) and the real robot validation (§VIII-B, Fanuc M-20ia +
ZEISS COMET Pro AE + Roboception rc_visard 65, on a real car door in the AIBox industrial
cell) report results as qualitative confirmations only: "the depth images demonstrate that
all imaging devices can successfully acquire all features without occlusion" / "the 2D
images and point clouds prove that both features can be successfully acquired." A handful of
extreme (vertex) viewpoints per scenario were checked by rendering or real capture; there is
no N-trial success rate, no comparison against a baseline (e.g., a uniform sphere), and no
statistical claim of any kind about the C-space's coverage or optimality. **This is a
first-class negative finding for RQ-benchmark: the paper that most closely matches our
target behavior reports zero quantitative validation of the thing it's proposing** — it
validates that the geometry is *correct* (points lie in the claimed polytope) not that the
resulting viewpoints are *good* by any external metric.

## Load-bearing assumptions

- **Features are known a priori, in closed form, before any C-space is built.** A feature is
  `f := (B_f, G_f(L_f))` (Eq. 2): a known frame B_f (position + normal) plus a set of
  geometric parameters L_f. §III-D3 explicitly generalizes every feature topology (circle,
  slot, star, "any feature topology," Fig. 3) to **a flat square of side length l_f with 5
  surface points** (center + 4 corners) purely for tractability of the trigonometric
  derivation. This is a modeled/specified feature, not something the pipeline extracts or
  fits from raw sensor data — it is the *input*, analogous to a CAD annotation or hand-placed
  keypoint, not an output of a segmentation/fitting stage.
- **Point clouds appear only as sensor *output* for verification, never as input geometry.**
  Grepping the full text: every "point cloud" mention (12 occurrences) is a rendered or
  captured depth output used to *check* a computed C-space post hoc (e.g., "the depth images
  and point clouds prove that both features can be successfully acquired"). The paper never
  discusses deriving a feature frame/geometry from a fused or noisy point cloud.
- **Random sensor noise is explicitly neglected** (§III-H, general assumptions list): "random
  sensor noise" is named as a Range-Image-Quality effect that is "neglected." The only
  uncertainty the framework internalizes is *pose* uncertainty — kinematic alignment error
  ε_e and robot absolute-position error ε_r (§III-B1, §III-F2) — folded into a single scalar
  ε = |ε_e|+|ε_s|+|ε_r| (§VII-B2) that uniformly shrinks the C-space's vertices. There is no
  model anywhere for feature-geometry uncertainty (e.g., an uncertain normal direction or
  uncertain face boundary from a fitted primitive) — only for where-the-sensor-actually-ends-
  up uncertainty.
- **Frustum is fixed at 8 vertices and assumed watertight/linear-edged** (§VI-A3 "General
  Notes" — three explicit existence conditions: known vertex frames, watertight manifold,
  straight edges between adjacent vertices). This assumption is load-bearing for every
  downstream construction (feature-scaling, error-scaling, occlusion) because they all
  operate on those same 8 vertices.
- **One feature is acquirable in a single static image** (§III-H): no motion-during-capture,
  no acquisition-parameter (exposure/gain) optimization, no robot dynamics constraints
  (velocity/acceleration/jerk) — the sensor pose is treated as instantaneously reachable and
  static once chosen.
- **Robot workspace 𝒲_r is assumed already known as a manifold** (§VII-E1, Eq. 32) and simply
  intersected (𝒞₈ = 𝒲_r). Explicit disclaimer (§VII-E3): **collision-freedom is *not* checked
  by the framework** — "an explicit proof for collision must be performed in the last step
  for a selected sensor pose within the 𝒞-space." IK feasibility and collision-checking are
  deferred entirely to a downstream, per-selected-pose check, not baked into the C-space
  construction.
- **Which features can be jointly captured, and at what shared orientation, is assumed given
  — not solved.** §VII-F (multi-feature) requires *a single fixed r_s^fix* shared by all
  features in the set, chosen by the user in advance; the question of "which features can be
  acquired simultaneously and which is an adequate sensor orientation" is explicitly named
  and explicitly punted: "This question falls outside the scope of this paper... motivation
  of our ongoing research" (§VII-F3). This is precisely the Set Covering Problem the paper
  separates out in §IV-C (Problem 1) as future work, never solved here.
- **Occlusion-space characterization is an approximation, not exact** (§VII-C4): built from a
  discretized grid of view rays (step size d^ς chosen from robot accuracy via arc-length,
  e.g. 0.3° for 1mm accuracy at 200mm), can miss corner points (Fig. 16a explicitly shows a
  missed corner), and is "more conservative" (over-restrictive) for concave/hollow occluders
  like a torus. The occlusion space can also produce a non-watertight manifold, so the paper
  recommends always computing it *last* in the integration order.

## What transfers to our setting

- **"Orthogonal to face 3" as a native, closed-form address is directly validated by this
  paper's core mechanism**, not merely analogous to it: fix the sensor orientation to the
  face's outward normal, reflect the frustum through the face-center frame, and you have a
  closed-form polytope of positions — exactly the branch's target primitive ("each viewpoint
  constraint becomes a closed-form geometric C-space... 'orthogonal to face 3' as a native
  address plus the region an optimizer could climb inside"). This is category (b) in RQ1's
  taxonomy — object-part reference (face + its known frame) plus a local offset region — not
  a lookup into a global tessellation and not a purely relational node ID. A blind
  orchestrator that names "face_3/orthogonal" can predict, deterministically and without
  sampling, that it will get a sensor pose normal to that face within a known-shape region.
- **The discrete/continuous split this paper embodies is a precedent for the PAMDP-style
  split the branch is investigating**: the discrete choice is *which feature* (which face,
  in our terms) and *which fixed orientation regime* (r_s^fix); the continuous part is the
  translation t_s, which is left as an entire feasible region (not resolved to a point) for a
  downstream optimizer/IK solver to pick within. This matches "discrete address names a
  region, continuous refinement resolves the exact pose inside it" almost exactly — with the
  caveat (see below) that orientation itself is *not* left continuous by default; it must be
  fixed or discretely sampled.
- **Scaling-vector composition (§VII-B) is a cheap, generalizable pattern for stacking
  constraints onto a primitive-level address**, and its cost profile (§VII-B1: O(#constraints)
  vs. O(#vertices) for CSG) is a concrete argument for preferring analytic offsets over CSG
  wherever a constraint (e.g., a minimum standoff distance, a kinematic-error margin) can be
  expressed as a vertex shift on a per-face polytope, rather than reaching for CSG
  intersection by default.
- **The C-space-per-feature, then-intersect-for-a-set pattern (§VII-F, Algorithm 5) is a
  workable template for our "views on this primitive face."** For a flat face this collapses
  cleanly to one 𝒞₃ region. It does not, however, offer anything for orbiting a curved
  surface (see negative findings).
- **Timing numbers give a concrete cost floor for "how cheap is it to compute one address's
  region": low-single-digit milliseconds for vertex-arithmetic constraints (position,
  feature-geometry, kinematic-error, sensor-accuracy scaling), 1-2 seconds for one
  ray-cast-based occlusion check, and tens of seconds only if you insist on CSG-intersecting
  many discretized orientations.** For our survey-then-inspect single-session loop (no
  offline/online split needed — see the framing doc), these numbers say per-primitive-face
  address computation is cheap enough to do online, but orientation *ranges* (if we ever
  wanted a tolerance band rather than a single fixed orthogonal direction) should be avoided
  or kept coarse.
- **The occlusion-space's explicit tie between discretization step size and robot absolute
  accuracy** (§VII-C4: d^ς ≈ arc-length(accuracy/working-distance)) is a reusable, generic
  rule of thumb for setting any discretization parameter relative to known kinematic error —
  applicable to any downstream sampling we do (e.g., placing candidate positions inside a
  primitive's addressed region).
- **The paper's own related-work classification (§II-A1) is a clean vocabulary check for
  RQ1's taxonomy**: it explicitly separates "tessellated spheres... fixed working distance"
  (Reed et al., Tarbox & Gottschlich's visibility matrices, Scott's measurability matrix —
  RQ1's category (a)) from synthesis-based continuous solution spaces (this paper's own
  category), and states the tessellated-sphere line's "major shortcoming... require[s] a
  fixed working distance, which limits their applicability... and reduces efficiency for
  multi-feature acquisition" (§II-A1) — an explicit, citable critique of the uniform-sphere
  approach our project is trying to replace.

## What does NOT transfer, and why (negative findings)

- **The method never touches whole-surface / curved-primitive coverage — it has no answer
  for "orbit the cylinder normal to the surface."** Every worked example is a small, flat,
  locally-planar feature (circle, slot, square, half-sphere pocket, at most a 3D convex/
  concave bump with height h_f) sitting at one known frame. There is no mechanism for a
  continuously-varying normal direction over a curved face — grep confirms zero occurrences
  of "cylinder," "orbit," or "primitive" (shape sense) anywhere in the paper. Covering a
  cylinder's lateral surface orthogonally would require *many* adjacent 𝒞₁ regions (one per
  small patch, each with its own r_s^fix following the locally-varying normal), stitched
  together — a construction this paper does not attempt or discuss. This is the sharpest gap
  against our RQ2 ("cylinder orbited with the camera normal to the surface").
  **[Contradicts the selection blurb]**: the paper was selected partly for validating
  cylinder-orbit-style behavior; it does not — it validates flat/pocket features only.
- **Orientation is not free inside the address by default — only translation is.** Any
  "region an optimizer could climb inside" from this paper is, strictly, a *translation*
  region for one fixed (or discretely intersected, non-interpolable) orientation. If our
  addressing scheme wants "orthogonal ± some tolerance angle" as a single continuously
  climbable region, this paper does not deliver it in closed form — §VI-B's own stated
  limitation ("Discretization without Interpolation," §VI-B3) says exactly that arbitrary
  intermediate orientations are unverified.
- **No cross-shape addressing consistency is discussed or attempted.** The paper is entirely
  single-feature / single-object-instance scoped; there is no notion of "face_3" meaning
  something comparable across a box vs. a cylinder vs. an assembly, no canonical frame
  machinery, no discussion of stable slot identity across shapes. RQ3's cross-shape
  consistency requirement gets zero precedent from this document — this reinforces (does not
  merely fail to contradict) the branch's suspicion that cross-shape *view*-addressing
  consistency is unaddressed in the literature the branch has sampled so far.
- **Feature geometry (l_f, B_f) is never derived from a noisy fused point cloud, and the
  paper offers no evidence the trigonometric scaling-vector formulas (Eq. 22-26) are robust
  to an uncertain or approximately-fitted face boundary.** Our setting's actual input (a
  fused PLY with ~4cm cross-view offset, per-view depth dropout, real defects) is a
  qualitatively harder problem than anything this paper's "a priori model" assumption covers.
  Random sensor noise being explicitly "neglected" (§III-H) means the paper's uncertainty
  handling (the scalar ε folded into vertex scaling) is a *pose* uncertainty model, not a
  *geometry* uncertainty model — it would need to be extended, not reused as-is, to tolerate
  primitives fitted to a noisy survey cloud.
- **Set-Covering / multi-feature-with-free-orientation is explicitly unsolved here.** Two
  separate deferrals converge on the same gap: (1) §IV-C names the SCP (how many viewpoints
  are needed to cover a feature set) as a distinct problem, explicitly out of scope; (2)
  §VII-F3 names "which features can be acquired simultaneously and which is an adequate
  sensor orientation" as unsolved, ongoing work. **Composing multiple per-primitive C-spaces
  into a coherent whole-object action space (our RQ3) is therefore not answered by this
  paper — only the atomic per-feature/per-face building block is.** The paper's own outlook
  (§IX-C) confirms the "holistic strategy" for the full VPP is a *separate future
  publication*, not covered here.
- **No collision-checking or IK-feasibility guarantee is built into the C-space** (§VII-E3,
  discussed above under load-bearing assumptions) — it is asserted, not demonstrated, that
  restricting the search space this way "contributes substantially" to making downstream
  collision-free IK easier; no numbers back that claim. This means the paper gives us zero
  evidence for the branch's "does the address survive infeasibility" question (RQ16) — no
  discussion anywhere of what happens to an addressed region when IK/collision fails inside
  it, no "nearest feasible" fallback of any kind is proposed.
- **Adjacency, ordering, and seams are entirely absent as concepts.** There is no discussion
  anywhere in the paper of neighboring features, an ordering over addresses, or how two
  adjacent C-spaces (e.g., two faces of a box) relate topologically to each other — each
  feature's C-space is computed and consumed independently. RQ4/RQ5/RQ7's adjacency and seam
  questions get no material from this document, mirroring the same gap already found in the
  Faugeras et al. aspect-graph panel discussion.
- **Build cost is not "offline vs. online" framed at all** — there's no separate compilation
  phase; every number above is a from-scratch computation given known geometry. This is
  actually a mild positive-by-omission for us (nothing here argues against on-the-fly
  per-session computation), but it means the paper gives no data point on amortized/offline
  compilation costs the branch's RQ17 asks about.
- **No comparison to a uniform/tessellated sphere baseline on any task with any metric.**
  Reinforces the cross-branch finding (also true of the Faugeras panel) that this comparison
  appears to be unbuilt in the sampled literature — direct support for the branch's own
  benchmark-gap finding (RQ19).

## Citations (for checkability)

- Abstract — "Feature-Based Constrained Spaces," nine constraints, closed-form geometric
  solutions: Abstract.
- C-space framing, Latombe quote: §IV-E, citing [51].
- Viewpoint definition v:=(f, p_s, C̃): §IV-A, Eq. 7.
- Nine viewpoint constraints, Table I: §IV-B.
- VGP formulation (Problem 2), Eq. 8: §IV-D.
- Ideal C-space 𝒞*=ℝ³×SO(3): §IV-F, Eq. 9.
- Joint C-space definition, existence conditions 1-4: §IV-G2, Eq. 10-12.
- Feature model f:=(B_f, G_f(L_f)), square generalization with 5 points: §III-D, Eq. 2-3,
  Fig. 3.
- Frustum space ℐ_s, Eq. 4: §III-E2, Fig. 4.
- Incidence angle φ_s^f, Eq. 5: §III-E5.
- Core constraint c1, Eq. 13-15: §VI-A1-A3.
- Algorithm 1 (Extreme Viewpoints), Algorithm 2 (Homeomorphism): §VI-A4, Fig. 6.
- 4.1ms±2.4ms timing for 𝒞1: §VI-A5.
- Range-of-orientations formulation, Eq. 17-18: §VI-B2.
- "Discretization without Interpolation," Eq. 19-20: §VI-B3, Fig. 8.
- 27→15s, 75→40s, 363→720s timings, volume ratios 0.9999/0.9995: §VI-B4.
- Feature-geometry C-space (C3), scaling vector Δ, Eq. 21-26: §VII-A1-A5.
- 5.8ms±3.2ms timing for C3, "efficient... analytical solution": §VII-A6.
- Scaling-vector generic formulation, cost comparison to CSG, Eq. 27-28: §VII-B1.
- Kinematic error ε=|ε_e|+|ε_s|+|ε_r|: §VII-B2.
- Occlusion space formulation, Algorithm 3, Eq. 29-30: §VII-C1-C2.
- Occlusion timings 0.7s/1.32s, discretization-vs-accuracy rule: §VII-C3-C4.
- Multisensor formulation, Algorithm 4, Eq. 31: §VII-D1-D2.
- Multisensor timing 200ms/100ms: §VII-D3.
- Robot workspace C8, Eq. 32, "explicit proof for collision must be performed in the last
  step": §VII-E1-E3.
- Multi-feature C-space, Algorithm 5: §VII-F1.
- "Which features can be acquired simultaneously... falls outside the scope": §VII-F3.
- Integration strategy, Algorithm 6: §VII-G.
- Academic evaluation setup, Table III: §VIII-A1.
- Academic evaluation results, ≈50s caveat: §VIII-A2.
- Real robot (AIBox, Fanuc M-20ia, ZEISS COMET Pro AE, Roboception rc_visard 65) setup:
  §VIII-B1.
- Real robot results, qualitative-only language: §VIII-B3.
- General assumptions/limitations list (single-image capture, noise neglected, no dynamics):
  §III-H.
- General Requirements table: Table IV.
- Limitations: CSG cost, "break-even point... remains to be further investigated": §IX-B.
- Outlook: holistic VPP strategy is future work, "already in progress": §IX-C.
- Related-work critique of tessellated-sphere/fixed-working-distance approaches: §II-A1.

## Assessment for the branch

This is the strongest hit in the branch so far for RQ1/RQ2's "native address" question: the
paper hands us, in closed form, exactly the primitive our framing doc names — "orthogonal to
face 3" resolves deterministically to an 8-vertex polytope of valid sensor *positions* via
one reflection operation, at millisecond cost, requiring only a known feature frame and
size. That is category (b) addressing (object-part reference + local offset region), not a
tessellation lookup and not a relational node ID, and it is exactly the kind of address a
blind orchestrator could name and get a predictable result from. But three gaps matter more
than the paper's selection blurb implies. First, it validates flat/pocket features only —
it has no mechanism for continuously orbiting a curved primitive (our cylinder case), so
RQ2's "cylinder orbited normal to the surface" gets no support here; that has to come from
elsewhere or be derived by us as a stitched sequence of this paper's atomic construction.
Second, orientation is fixed or discretely-sampled-and-CSG-intersected, never left as a free
continuous dimension inside one address the way translation is — so "the region an optimizer
could climb inside" is real for position but not, without extra machinery, for orientation
tolerance. Third, and most consequential for RQ3: composing per-feature C-spaces into a
consistent whole-object action space (which faces, in what order, sharing what orientation,
addressed how across shapes) is the exact problem the authors twice name and twice defer to
future work (§IV-C's Set Covering Problem, §VII-F3's "which features... simultaneously").
The paper gives us a validated atomic address primitive and a real, if unglamorous, cost
profile for building it; it does not give us the composition layer RQ3 is actually asking
about, and it gives zero quantitative validation (no success rate, no coverage metric, no
baseline comparison) of the geometry it proposes — only qualitative photographic confirmation
that a handful of extreme viewpoints work.
