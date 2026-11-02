# Spensieri, Åblad, Salman, Carlson — "A unified sampling method for optimal feature coverage and robot placement" (Robotics and Computer-Integrated Manufacturing 93 (2025) 102932)

DOI: 10.1016/j.rcim.2024.102932 · Open access, CC BY 4.0 · Received 1 Jul
2024, accepted 8 Dec 2024, online 19 Dec 2024. Authors: Fraunhofer-Chalmers
Centre, Geometry and Motion Planning Group, Gothenburg. ScienceDirect page
(`S0736584524002199`) returned HTTP 403; fetched the full-text PDF instead
from the Chalmers open-access repository:
`https://research.chalmers.se/publication/544793/file/544793_Fulltext.pdf`
(13 pages, version of record). Read in full, including all figures,
equations, and pseudocode.

## What this paper actually is

Not a view-planning paper in our sense — it is an **exact, general-purpose
divide-and-conquer + ILP algorithm for the geometric minimum-piercing-set
problem** (dual of minimum unit-disk cover), applied to four industrial
placement problems that all reduce to the same abstract form: *given a set of
"features" (points/tasks that must be observed or operated on), and for each
feature a geometric region of space from which a single candidate location can
satisfy it, find the minimum number of candidate locations such that every
feature is satisfied by at least one of them.* Sensor/camera placement,
inspection-viewpoint placement, and multi-robot/AGV base placement are all
treated as **instances of the same problem** by swapping what the "candidate
location" and "admissible region" mean. Central relevance to us: it is a
concrete demonstration that (a) minimum set cover over admissible-view regions
can be solved **exactly** rather than greedily at industrially relevant scale,
and (b) reachability/collision constraints can be folded into the *candidate
generation* step itself rather than bolted on afterward as a post-filter.

## Problem formulation (Sect. 3, 3.1)

Starts from **MGUDC** (Minimum Geometric Unit Disk Cover): given N points in
the plane, find the minimum number of unit disks needed to cover them —
NP-hard (cite [63], Gonzalez). The paper adopts the **dual**: given a set `F`
of `|F|=N` unit disks (or general convex shapes), find the smallest set `P`
of points such that every disk `f ∈ F` contains at least one point of `P`
(Sect. 3, Fig. 2(a)). This dual is the frame used throughout: a "feature" (an
inspection point, a welding task, a cleaning task) is mapped to a convex
region `f` in candidate-location space (the set of positions from which that
feature can be validly observed/reached); a "candidate location" is a point;
minimum coverage = **minimum piercing set** of the feature-shapes.

**Rationale (Sect. 3.1, load-bearing):** "The idea is to find all subsets
`F_i` of `F` where the corresponding disks have non-empty intersection, i.e.
`⋂_{f∈F_i} f ≠ ∅`. While the number of such subsets can grow exponentially,
we show that, in our applications, this growth is manageable." This is an
**empirical, not theoretical, claim** — validated only by the paper's own
computational results (Sect. 4), not proven in general. Directly the same
"hunch that this is optimization" our framing doc states, from an independent
industrial source.

**Convexity is a hard structural requirement**, stated explicitly: "The
convexity assumption is needed in a step of the proposed algorithm, see
Section 3.2.3." Every feature-shape used in the paper (disk, sphere, cone) is
convex. This matters for transfer — see "What does NOT transfer" below.

## The base algorithm — Algorithm 1, "Feature Coverage" (Sect. 3.2)

A cell-based divide-and-conquer search over candidate-location space (2D or
3D box cells, generalizes to any convex shape):

1. Initialize a coarse grid queue `Q` (max 1000 cells at start; Sect. 3.2.1).
   Each cell computes its **covering set** `C` = the subset of shapes in `F`
   that intersect the cell; cells with `C = ∅` are dropped.
2. Pop a cell `q`, subdivide it (`divide(q)`, Sect. 3.2.2) into subcells along
   its **largest axis** (empirically the best-performing subdivision rule —
   "our computational experience shows dividing the cell along its largest
   axis leads to very good performance").
3. For each subcell `s`: find `s.C` (`findCovering`, Sect. 3.2.3, cheap under
   convexity). If `s.C ≠ ∅` and `s` is not **dominated** by an existing leaf
   (dominated = `s.C ⊆ q.C` for some already-found leaf `q` — `s` covers a
   subset of what's already achieved by one candidate point, so it's
   redundant and pruned), try `findIntersection(s.C)` (Sect. 3.2.4) — find a
   single point `p ∈ s` lying in the intersection of *every* shape in `s.C`.
4. If such a point is found → `s` becomes a **leaf** = a candidate location
   `p` that simultaneously covers the whole feature subset `s.C`. If not
   found → push `s` back onto the queue for further subdivision.
5. Terminate when `Q` is empty or a minimum cell volume `v_min` is reached
   (resolution parameter that trades solution quality for compute).

**`findIntersection` — two implementations (Sect. 3.2.4, Eq. 1–2):**
- **Exact convex-optimization formulation**: `min g(x) s.t. ‖x−c_f‖² ≤ r_f² ∀f∈F, l≤x≤u, x∈ℝ³` (Eq. 2, generalizing the single-sphere feasibility problem Eq. 1). Accurate but slow — "requires analytical formulation for convex shapes... too slow in many settings."
- **Fast GJK-style separating-direction algorithm** (ref. [66]), used for *all* reported computational experiments — decision-only (do the shapes intersect? yes/no), not an exact point; when they intersect, a point is obtained by sampling within the cell. If no point is confirmed, the cell is subdivided rather than solved analytically. Quote: *"the difference between the mathematical programming approach and the approximation scheme is that in the first case computational time is spent to solve the mathematical problem, whereas the second case copes with approximation by refining the cell."*

**Queue ordering**: LIFO/stack "performs very well in practice" (Sect. 3.2.1)
— i.e. depth-first refinement, not breadth-first uniform sampling.

## Set covering — the selection stage (Sect. 3.4, Eq. 3)

Once the queue empties, all leaves `L` (candidate locations, each covering a
subset `C(f)` per feature) are fed to a standard **binary ILP set cover**:

```
minimize   Σ_{i∈L} x_i
s.t.       Σ_{i∈C(f)} x_i ≥ 1   ∀ f ∈ F
           x_i ∈ {0,1}          ∀ i ∈ L
```

Solved with **HiGHS** [67], an off-the-shelf open-source ILP solver. Quote:
*"the problem above is not very difficult to solve by general Integer Linear
Programming solvers... If this is not the case, one can relax the variables
to `x_i ∈ [0,1]`."* No greedy heuristic, no submodularity discussion anywhere
in the paper — the selection stage is **always exact**, not approximate.

## Enhanced algorithm — reachability folded into generation (Sect. 3.3, Algorithm 2)

**Algorithm 2 ("Enhanced Feature Coverage")** adds a `findAllVisible`
application-specific routine at the *cell* level, *before* `findIntersection`
runs — i.e. reachability/collision constraints prune candidates during
generation, not as a post-hoc filter on an already-enumerated candidate set.
`findAllVisible` is customized per application:
- **Laser sensor (Sect. 4.2)**: checks the sensor itself is collision-free
  with the environment *and* the laser beam (line-of-sight ray) is
  collision-free — ray-casting only, no robot kinematics.
- **Robot manipulator (Sect. 4.3)**: solves inverse kinematics and computes
  distance-to-collision w.r.t. the rest of the scene, via the commercial
  robotics simulation package **Industrial Path Solutions** [68].

**Algorithm 3 ("Lazy Enhanced Feature Coverage")** defers the expensive
distance/IK queries when they dominate compute time: it tracks an **upper
bound `n̄`** (best confirmed-feasible solution count) and a **lower bound
`n̲`** via a relaxed ILP that optimistically includes not-yet-fully-verified
queue cells (Eq. 4a–4c, using an index set `Q` for queue cells and a function
`G(f)` retrieving queue-cell indices covering feature `f`). The algorithm
stops when `n̲ = n̄` (proven optimal) or a time limit is hit, returning the
best solution found with a known gap. This is a genuine **anytime,
gap-bounded stopping criterion** — not a learned or heuristic budget.

**Speed-ups (Sect. 3.5)**: parallel multi-threaded queue processing (each
cell processed independently); distance computation done first on the
*children* of a cell that already collided as a parent (skip re-testing since
the parent must be subdivided anyway regardless of the children's status).

## Reachability modeled as a conservative sphere (Sect. 4.3, load-bearing)

For the two robot-base-placement scenarios (cleaning AGV, stud welding),
reachability is folded into the *shape definition itself*, not just the
distance-query filter: *"for typical industrial manipulators with six degrees
of freedom... their workspace can be conservatively overestimated by a
sphere. Consequently, following the dual approach... the convex sets `F` are
represented as spheres positioned at the locations where the robot faceplate
must be situated to perform the operations."* I.e. each task point induces a
**solid ball** of admissible robot-base positions (candidates that are
plausibly within reach), and the exact IK/collision distance query (Algorithm
2/3) only runs on the reduced candidate set that survives this coarse
geometric admissibility check. This is a genuine **two-tier reachability
filter**: cheap analytic over-approximation (a convex ball) narrows the
search, expensive exact query (IK + collision distance) validates what
remains. The paper also explicitly notes the feature *shape* here is not an
intrinsic property of the feature alone — it depends on the robot's
workspace and the tool geometry ("the feature positioning depends on the
tool used since it defines the position and orientation of the robot
faceplate").

**Important caveat, not flagged by the authors**: the over-approximation used
is a **solid ball** — no inner exclusion radius. Real six-axis arms
(including ours) typically have a *minimum* reachable radius too (near-base
singularities, self-collision, gripper/camera hull clearance), making the
true reachable set a spherical **shell** (ball minus a smaller concentric
ball) rather than a solid ball. A shell is **not convex**, and the paper's
own stated structural requirement (Sect. 3, "the convexity assumption is
needed") means their exact `findIntersection` machinery cannot directly
encode a shell as one of the feature-shapes `F` without either (a)
convex-decomposing it (e.g. as a union of thin convex slabs) or (b) falling
back to using only the expensive per-candidate distance/IK query (which
needs no convexity, since it's a black-box test, not a shape). This is a
concrete, checkable gap in what transfers to our UR5e setting, whose framing
doc explicitly notes "the reachable shell radius has a hard minimum."

## Computational results (Sect. 4, all four scenarios)

**4.1.1 — 2D MGUDC, random unit disks (Fig. 2, 5, 6).** N = 50…2000 randomly
placed unit disks in a 10×10 square, 100 random instances per size,
`v_min = 1e-5`. Candidate-generation time (Algorithm 1) scales from ~0.01 s
avg (N=50) to ~25–40 s avg / ~60 s max (N=2000) (Fig. 5). ILP solve time
(HiGHS) scales separately from ~0 s (N=50–200) to **avg ~210 s, worst-case
~600 s (10 min)** at N=2000 (Fig. 6b). Total worst case for the largest
random instance tested: roughly 11 minutes on a desktop (AMD Ryzen 9 3900X,
32 GB RAM).

**4.1.2 — 3D cones, no collision test (Fig. 7, 8).** 427 laser-scanner
inspection points on a car frame, each with "a given maximum angle between
the beam and the surface normal" → 427 cone-shaped admissible regions. **No
explicit numeric incidence-angle threshold is stated anywhere in the main
text** (unlike Gospodnetić et al.'s 75° — see sibling note
`view-quality-scoring/gospodnetic-viewpoint-placement-inspection-planning.md`);
it may only be in the paper's supplementary material, which was not fetched.
Algorithm 1 (D&C) vs. uniform-grid sampling baseline, across 7 resolutions
(`v_min` from 1.1e-3 down to 1.7e-8): both converge to the **same optimum,
~80 locations**, but D&C reaches it at coarser resolution (91 vs. 97 points
at the coarsest tested resolution) and stays cheap throughout (<20 s);
uniform sampling's time explodes to ~800 s at `v_min=1.3e-7` and then
**runs out of memory** at the finest resolution tested (negative bars in
Fig. 8b = "out of memory"/no optimum found).

**4.2.1 — Industrial ray-casting case, Saab Aeronautics (Fig. 9).** 413
real inspection points on an aerospace artefact, cone-shaped admissibility
regions again, this time with Algorithm 2 (Enhanced) handling laser-beam
line-of-sight collision. **Minimum found: 4 sensor locations**, consistent
across resolutions for D&C; uniform sampling agrees on the same optimum at
coarser resolutions but **runs out of memory at the finest tested resolution
(`v_min=1.2e-7`)** while D&C completes. This is the paper's closest analog to
"place a small number of cameras/sensors to see everything" — directly on
point for our RQ2, though for a *static* sensor rig, not an eye-in-hand arm.

**4.3.1 — Cleaning station, AGV + manipulator (Fig. 10, 11).** 35 cleaning
task locations; candidates are **AGV base positions** (with height as an
extra DOF). **Minimum found: 4 locations** (converges from 5 at the coarsest
resolution). D&C stays far cheaper at fine resolution (~25 s vs. ~210 s for
uniform sampling at `v_min=1.1e-6`, Fig. 11b) — same optimum either way.

**4.3.2 — Stud welding, automotive body-in-white (Fig. 12, 13).** 77 welding
tasks; candidates are **robot base positions** (multi-robot placement).
**Minimum found: 3 robots** (converges from 4 at the coarsest resolution).
This is the paper's own flagged hard case: *"challenging due to
computationally heavy distance queries because of the cluttered
environment."* At the finest tested resolution (`v_min=1.2e-6`), D&C takes
**~1.02×10⁴ s (≈2.8 h)** vs. uniform sampling's **~1.55×10⁴ s (≈4.3 h)**
(Fig. 13b) — both converge to the same optimum (3 robots), but the D&C speed
advantage **shrinks dramatically** here versus the other three scenarios.
Explicit stated reason: *"the distance function is not caught or tried to be
reconstructed along the proposed algorithms"* — the geometric shape-
intersection speedup only accelerates *which candidate points are
geometrically distinct*; it does **not** reduce the number of expensive
IK/collision distance queries once those, not shape geometry, are the
bottleneck. Also flagged: *"the coexistence of multiple robots requires more
work to handle mutual collisions"* — inter-robot collision not fully solved.

## Discussion / future work (Sect. 5)

Explicitly unsolved and left as future work: **joint optimization of
location count *and* cycle-time/load-balancing** across the found locations
— *"simultaneously optimizing robot positions and load balancing is
challenging and is a topic for future research."* The algorithm minimizes
*count* only; it says nothing about how work should be distributed once
locations are fixed. Also: "more sophisticated [stopping] strategies may be
devised... in the near future" — the bound-gap stopping criterion is
presented as adequate but not final. No dataset was released ("No data was
used for the research described in the article" — presumably meaning no
external dataset; their own CAD/industrial data is proprietary and not
published, so results are not independently reproducible from public data).

## What transfers to our setting

- **Exact ILP set cover is trivially fast at our scale.** Their smallest ILP
  instances (tens to hundreds of features, e.g. the 413-point Saab case → 4
  locations, the 77-task stud-welding case → 3 robots) solve near-instantly;
  even their *largest* random abstract instances (2000 disks) solve in
  minutes with off-the-shelf HiGHS. Our instance (tens of candidate views ×
  tens of surface regions) is smaller than every scenario in this paper.
  **Direct, quantified, load-bearing support for skipping greedy/submodular
  apparatus entirely and solving exactly** — this is an independent
  industrial-optimization group reaching the same conclusion implicitly (they
  never mention greedy or submodularity anywhere in the paper).
- **A genuine, principled, gap-bounded stopping/minimality criterion**
  (Algorithm 3, Eq. 4): track ILP upper/lower bounds during search, stop when
  they meet (proven global optimum) or report the gap. This is a cleaner
  answer to "is there a principled minimality criterion" than any
  learned-budget approach (SCVP, PRVNet) — but it presupposes the exact-ILP
  framing is available, which in turn presupposes convex admissible-view
  regions per feature/region (plausible for our box faces / cylinder shells
  under an incidence-angle tolerance, since a normal-cone around a planar
  patch or an angular sector around a cylinder axis is naturally convex or
  near-convex in the right local coordinates).
- **Reachability folded into candidate generation, not bolted on after
  (Algorithm 2/3, Sect. 4.3)**: a concrete, published precedent for "never
  leave the feasible set" — collision/IK distance queries prune candidate
  cells *during* the divide-and-conquer search, and a fast analytic
  over-approximation (a convex ball proxy for the manipulator's reach) is
  applied *first*, with the expensive exact query reserved for the reduced
  candidate set that survives it. This is a direct, two-tier answer to the
  branch's "capability map up front vs. post-filter IK" question: **the
  answer this paper gives is "both, hierarchically"** — cheap conservative
  geometric filter first, exact per-candidate query second, because the
  conservative filter alone is known to admit false positives (that's
  precisely why the exact query still has to run afterward).
- **"Generate candidates from constraint intersections, then still run an
  explicit selection stage" is the paper's actual pipeline**, not
  "enumeration vs. intersection-derivation" as an either/or. Candidate
  *locations* are derived directly from the geometric intersection structure
  of per-feature admissible regions (divide-and-conquer refinement toward
  intersection points), which is empirically shown to strictly dominate
  naive uniform-grid enumeration (same or better optimum, far less time and
  memory — uniform sampling runs out of memory in 3 of 4 industrial
  scenarios at fine resolution, Fig. 8, 9). But the intersection-derived
  candidates are still redundant/overlapping and require a final exact
  set-cover selection pass to reach the true minimum. Useful nuance for the
  branch's RQ4: the "generate then select" pipeline earns its keep, but the
  *generation* step should be constraint-driven, not grid-driven — echoes
  and gives quantified industrial backing to the Gospodnetić finding
  (`view-quality-scoring/gospodnetic-viewpoint-placement-inspection-planning.md`)
  that geometry-native candidate generation beats space/sphere sampling —
  here from an *exact-optimization* angle rather than a heuristic-NBV angle.
- **Exact selection immunizes against redundancy regardless of candidate-
  generation method.** Because the final stage is always an exact minimum
  set cover, the paper's own uniform-sampling *baseline* still reaches the
  same true optimum as the geometry-native method in every scenario reported
  — the D&C method's advantage is purely in *time/memory to get there*, not
  in the quality of the final answer. This nuances the branch's redundancy
  concern (RQ re: Bircher et al.): redundancy is a property of *greedy or
  incomplete* selection, not of candidate-generation strategy, as long as an
  exact selector sits on top.
- **View/location count is emergent from geometry + tolerance, not a fixed
  per-primitive recipe** — compare 4 locations for 413 points (wide-coverage
  static sensor) vs. ~80 locations for 427 narrow-incidence-angle cones (same
  order of feature count, 20× different location count) purely because of
  the admissible-cone width and sensor range. Suggests our RQ7 ("how many
  views does a primitive deserve") may not need a closed-form formula at
  all — it can instead be *computed exactly*, per primitive and per
  incidence-tolerance, by running this class of algorithm on our own
  geometry, since it is cheap at our scale.

## What does NOT transfer, and why (first-class negative findings)

- **Requires known, exact feature geometry (CAD-derived or process-defined
  points), never a defective survey point cloud.** All four scenarios use
  either random synthetic points, laser-scanner target points on a known car
  frame, or pre-specified process points (weld studs, cleaning locations).
  No pose error, no depth dropout, no fused-vs-single-view offset — none of
  our PLY-specific noise model is addressed. **Confirms the "no paper plans
  from a defective self-acquired survey cloud" gap** (adds one more negative
  data point for that branch question).
- **No primitive decomposition anywhere.** Features are given directly by
  the industrial process (weld stud locations, cleaning task points,
  inspection points), never derived by fitting/segmenting a scanned shape
  into boxes/cylinders/etc. **Confirms the "primitive-decomposition ×
  view-planning fusion appears absent" claim** — though weakly, since this
  paper was never a candidate for that fusion to begin with (it doesn't even
  start from a scanned/unknown shape).
- **No per-view/per-region quality score, no orthogonality-as-a-continuum.**
  Admissibility is a hard binary gate baked into the shape definition
  (inside the cone/sphere = admissible, outside = not); there is no
  soft-weighted "how orthogonal" or "how well-framed" term anywhere, and no
  discussion of a quality floor distinct from a coverage requirement. Does
  **not** answer the branch's "is coverage even the right objective, or is
  it per-region quality subject to a coverage floor" question directly — it
  only shows the industrial-standard move of collapsing quality into a hard
  admissibility threshold (same move as Gospodnetić's 75° incidence
  threshold), reinforcing that pattern without adding a new number.
- **No max-k-coverage variant.** The objective is always "cover everything,
  minimize count" — there is no "given a fixed budget k, cover as much as
  possible" formulation anywhere, so this paper cannot directly resolve the
  branch's min-set-cover-vs-max-k-coverage framing question; it is firmly
  and only a minimum-set-cover paper.
- **Convexity is structurally required and our reachable workspace is
  probably not convex.** See the load-bearing caveat above (spherical shell
  vs. solid ball) — this is a concrete technical obstacle to directly
  reusing Algorithm 1/2's shape-intersection machinery for our reachability
  constraint, not just a simplification to note in passing.
- **No IK-multiplicity discussion, no Cartesian-vs-joint-space "what is a
  view" question.** IK and collision distance are treated as an opaque
  feasibility oracle (via commercial software), never as a multi-solution
  space with different collision/joint-limit behavior per solution. Silent
  on the branch's IK-multiplicity RQ.
- **No percentages on how much of a surface/workspace becomes unreachable.**
  Every reported scenario *achieves full coverage* (a feasible covering set
  is always found) — there is no reported case, number, or discussion of
  partial/impossible coverage under reachability constraints, so this paper
  gives no equivalent to ObjView-Bench's ~45/128 reachable-views or
  0.13–0.17 coverage-loss numbers.
- **No VLM, no learned selection, no image-space judging anywhere.** Purely
  classical computational geometry + exact ILP. Silent on branch RQs about
  learned NBV vs. analytical selection.
- **Two of the four scenarios (cleaning AGV, stud welding) solve a
  different-level problem than ours**: candidates there are *robot/AGV base
  mounting positions* (where to park a whole manipulator), not *individual
  end-effector viewpoints* (which pose the eye-in-hand camera should strike).
  The mathematical machinery is the same, but the physical interpretation is
  one level higher than our per-view selection problem — treat these two
  scenarios as weaker/indirect transfer, and the 3D-cone and Saab-sensor
  scenarios (Sect. 4.1.2, 4.2.1) as the closer analogs to "select camera
  viewpoints."

## Branch-question coverage summary

Directly and strongly informs: exact-ILP-at-our-scale (the "should we just
solve it exactly" RQ — yes, with quantified precedent far above our instance
size); the fallback-ladder / capability-map-vs-post-filter RQs (this paper's
answer is "both, hierarchically," with a stated reason the coarse filter
alone isn't enough); "generate then select" pipeline justification (RQ4,
nuanced: keep both stages, make generation constraint-driven); a genuine
gap-bounded stopping/minimality criterion (RQ re: SCVP/PRVNet alternatives);
redundancy-immunity-via-exact-selection (nuances the Bircher et al.
redundancy concern). Partially informs: per-primitive view-count RQ (no
closed form given, but demonstrates count is cheaply computable exactly per
geometry+tolerance, suggesting we may not need a closed form at all).
Explicitly does NOT inform: min-set-cover-vs-max-k-coverage framing (paper is
only ever min-set-cover); the orthogonality-tolerance numeric threshold (no
degree value stated in the main text); submodularity (never invoked, exact
ILP used instead, which sidesteps the question rather than answering it);
IK-multiplicity / Cartesian-vs-joint "what is a view" question (silent);
reachable-shell-fraction numbers (every scenario reports full coverage
achieved, no partial-coverage numbers); learned-vs-analytical selection
(no learning anywhere); defective-survey-cloud planning (confirms absence,
does not fill it); primitive-decomposition × view-planning fusion (confirms
absence, does not fill it — features are process-given, not shape-derived).
The one genuinely new technical caveat this paper surfaces for us and that no
sibling note yet covers: **our reachable workspace, with its stated hard
minimum shell radius, is a non-convex spherical shell, which conflicts with
this method's explicit convexity requirement** — worth carrying forward as a
concrete design constraint if we ever want to encode reachability as a
"feature shape" rather than only as an opaque per-candidate query.
