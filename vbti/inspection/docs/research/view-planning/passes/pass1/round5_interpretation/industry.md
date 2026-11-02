# Round 5 — Interpretation: senior robotics engineer (industry)

**Persona:** I ship inspection systems. I care about what survives contact with real
hardware and real point clouds, what the integration and maintenance bill is, and what
teams actually deploy versus what papers claim. I have watched projects die because
someone trusted a fitted primitive on a cloud the sensor never actually delivered.

**Read:** the architect's round-4 feedback (`round4_accumulate/feedback.md` — binding;
overrides the framing doc where they conflict), the original contract
(`geometry-aware-view-planning.md`), and all six round-4 branch syntheses.

**The design as I now understand it:** inspection = answering questions about objects
(QA), not defect detection. Planning is an incremental loop — scan, fit primitives to
the partial cloud, derive views, present a small menu, the blind LLM picks, execute,
refit, replan. No complete survey up front. The "intelligent viewsphere": geometry
computes the orthogonal, reachable, well-framed views internally and hands the LLM a
finished finite menu (view number + label); the LLM never sees primitives, coordinates
or knobs. A separate VLM image agent reads the captured images. Month goal: a working
system, no thesis; one comparative claim — geometry-conditioned view spaces beat the
uniform viewsphere — proved by the 3-tier component benchmark. Demo v2: a box with
codes on one side; the evidence image is the orthogonal view of that side, robust to
the box being moved and rotated. Out of scope: VLA, RL, defect detection, deep
sensor-noise modeling.

---

## 1. Interpretation against the updated design

### 1.1 Where the feedback fixed things the literature was flagging

1. **Scoping inspection to QA dissolves the review's most uncomfortable tension.**
   The reading established that orthogonality is provably best for measurement
   quality, sampling density and evidence legibility — and *unproven* for defect
   contrast (raking light points the other way; the canonical inspection survey never
   says "defect"). For reading codes, labels and text, orthogonality is exactly the
   proven case: foreshortening costs resolution as 1/cos θ (measured 38,500 points
   head-on vs 4,900 at steep tilt — 8× loss), and framing/legibility arguments are
   uncontested. The updated scope sits squarely on the defensible ground. Keep one
   line on record that the defect-visibility question is deferred, not answered —
   whoever picks up defect detection later inherits an open question, not a settled one.

2. **The intelligent viewsphere is the published consensus pattern, not a bet.**
   Six of six working systems read in Branch 6 do exactly this: geometry proposes a
   small labelled candidate set deterministically; the model disposes by naming a
   label; the model never constructs the set and never sees raw geometry (GraspView,
   MOKA, PIVOT, SeeGround, AP-VLM, MV-GEL). Selection-over-emission carries the
   biggest number in the whole review (GPT-4V: 25.7% emitting a bounding box vs 86.4%
   naming a mark). Dropping the "expose primitive tuples to the LLM" idea was right —
   it had no precedent and every measured failure mode (multi-axis confusion,
   composition collapse, frame-implicit collapse) stacked against it.

3. **The incremental loop is what deployed systems actually do.** Survey-first was
   the fantasy; nobody gets a complete clean model before planning in production
   either. The correction also un-dismisses the useful half of the NBV literature:
   our loop IS iterative NBV with primitive fitting in the middle and no learned
   selector. Pan 2025's treat-prior-as-fallible mechanism (union the transferred
   prior with the current scan plus their divergences — the corpus's only measured
   technique for fallible priors) moves from footnote to core reference.

4. **Two-stage selection (which surface → which view of it) is backed by the largest
   measured effect in the visual-prompting corpus.** MOKA: removing the high-level
   decomposition step collapses success from 0.6–1.0 to 0.0–0.2 — an order of
   magnitude more than any encoding choice. And small menus are measurably better
   (K≈2 beats K=5 by 6 pp, 30 pp on hard cases). The 2-stage structure the architect
   liked is the single best-supported interface decision available.

5. **Making the benchmark the arbiter of representation choices fills a real gap.**
   Nobody in the field has ever ablated angle labels vs directional words vs indices
   for a camera menu — Think3D deploys integers and never varies them, ViewSpatial
   uses compass words and never varies them, MOKA uses chess notation and never
   varies it. Our A/B is cheap, novel, and decision-relevant. Correct call.

### 1.2 What the literature confirms about the build

1. **The claim we want to make is the field's own 20-year-old conclusion.** Sphere
   sampling discards 33–77% of candidates as inadmissible vs 0.08–32% for
   surface-anchored generation; Gospodnetić's verdict is verbatim "no benefits and
   their use is not recommended"; Scott condemned the view sphere in 2003. Three
   head-to-head comparisons exist and all favour geometry-conditioning. We are not
   chasing a phantom — but see §1.3.4 on what control it must beat.

2. **Orthogonal-by-construction is mainstream, 30 years old, and shipped.** Pose =
   surface point + n·d, look along −n (Mosbach Eq. 6, one closed-form evaluation;
   Scott's Optimal Scanning Zone; Roberts & Marshall). The full pipeline — decompose
   into primitives, grid each bounded primitive by camera FOV, one orthogonal
   viewpoint per cell — is a published system on a real 6-DOF arm (Wang et al.,
   T-Mech 2024). Deterministic per-primitive rules are also licensed by evidence,
   not just by the contract: MVTN shows category-level view rules capture 90.6% vs
   91.0% for per-instance optimization. Boring, precedented, buildable.

3. **Per-iteration replanning is computationally free.** Exact ILP set cover solves
   in ~1 s at 800 candidates (Gurobi); our instances are tens of candidates. The
   compute sink is the visibility predicate — ray-traced matrices run ~1–1.6 h/object
   and cannot live in a session loop, so the occlusion/visibility check must be
   analytic at primitive level (closed-form for planes and cylinders) with SEE's
   mesh-free point-proximity test as the pre-fit fallback. Menu sizes come out right
   by construction: minimal covering sets for tabletop objects cluster in single
   digits to mid-teens across ten independent sources, and 1–2 extra views capture
   essentially all achievable gain (four systems converge on this).

4. **Reachability is a first-class menu property, with numbers.** Roughly two-thirds
   of a naive candidate sphere is kinematically unavailable on a tabletop rig
   (45.4/128 average, range 26–74, across placements). And the documented worst case
   is exactly our architecture's shape: a discrete selector whose action space
   shrinks underneath it fails catastrophically, not gracefully (NBV-Net 0.988 →
   0.472 when the view space was restricted). Every menu item must be pre-verified
   with exact IK + collision in the kinematic sandbox before the LLM ever sees it,
   and the menu must be able to say "unavailable, next best" rather than silently
   changing shape.

5. **The blind orchestrator is not refuted.** Its measured cost is decisiveness and
   path efficiency, not coverage (MapGPT blind arm: −3.2 SR, −5.8 SPL, +5.8 oracle
   success); symbolic verify-and-correct recovered 13.3% → 86.6% executability with
   zero vision. Vision-pretrained backbones should still be used even blind. The
   architecture holds — with the burden of proof noted in §1.3.3.

### 1.3 Where I push back

1. **Sensor deprioritization: agreed on scope, one carve-out I will not drop.** No
   metrology campaign, no σ(range, θ) research — agreed; once we're at the orthogonal
   view only the RGB matters, and the camera is swappable anyway. But the *planner's
   input is depth*, every iteration of the loop. The D405 is passive stereo — no
   projector (the review's load-bearing datasheet catch) — and its documented worst
   case is a textureless matte surface: correspondence loss independent of angle.
   Our demo object is a cardboard box. If depth fill collapses on the actual demo
   objects, every downstream component stalls, and "camera choice flexible" only
   helps if we find out in week 1. A half-day timeboxed smoke test — fill rate and
   plane-fit quality on the real objects at operating standoffs — is not research;
   it's checking the forklift starts before loading it. Also two hard datasheet
   numbers bind the standoff band regardless: Min-Z is 100 mm at full resolution
   (7 cm is only true at 848×480), and the stereo invalid-depth band eats ~10% of
   image width at 10 cm. Bake both into the framing math and move on.

2. **"Do we need to go that deep on view quality?" — No. Ship the shallow judge.**
   One incidence gate (threshold picked from the smoke test or defaulted to the
   literature's 60–70° band and tuned once), one standoff band, one framing check.
   That is the entire per-view judge this month. The field never combines more than
   two quality terms across 25 surveyed systems; Secord showed more than five
   attributes stop helping; term redundancy is unmeasured everywhere. Uncertainty
   propagation, image-space scoring, learned judges — all research programs, all out
   of a build month. The benchmark will tell us if the shallow judge is insufficient;
   my experience says for boxes and cylinders it won't be.

3. **The LLM-picking-views step still carries an unmet burden of proof, so keep the
   auto-select arm alive.** No published result shows a *prompted* VLM beating random
   view selection, with variance — both positive random comparisons use trained
   selectors. The cheapest credible system is: LLM decides *what* to inspect (which
   surface, when to stop, what to ask the image agent); geometry auto-selects the
   view of that target. That arm costs nearly nothing to keep, is what I'd bet ships
   for demo v2, and is the ablation that makes the full menu-choosing design
   credible when it wins — or saves the month when it doesn't. This is not hostility
   to the design; it's the control the design needs.

4. **The headline claim needs a stronger control than our own old grid.** Beating
   the in-house 12×30°×3 az/el sphere convinces nobody who has read this literature —
   naive az/el is the *weak* uniform construction; the field's benchmarks use Tammes
   packings, and an equal-area HEALPix grid costs zero to build. Add one
   budget-matched strong-uniform arm. Equally: do not let the claim be scored on
   coverage — the uniform sphere saturates coverage metrics at 98%+ and we would
   lose vacuously on a construct that isn't ours. Score it on grounded QA outcome
   and per-region orthogonality. The metric choice decides whether our claim is even
   expressible.

5. **Circularity discipline, because we are the textbook N=1 case.** One team
   designs the benchmark, the judge, and the method, and iterates all three. If the
   space is constructed to be orthogonal and the metric is orthogonality, geometry
   wins by construction. Mitigations that survive scrutiny: score achieved
   orthogonality against the *raw cloud's locally-refit normals*, not the fitted
   primitive that generated the view; report the QA outcome layer as co-headline;
   pre-register the losing condition before iterating; keep an untouched holdout of
   captures from day one.

### 1.4 What the updated design makes newly hard

1. **Partial-cloud fitting is now the load-bearing wall — and it is the least
   supported case in twenty papers.** Nobody tests a half-scanned cylinder with
   one-sided dropout. Best anchors: EMS holds 86.4% global-optimality at 20% surface
   coverage (superquadrics, contiguous synthetic cap); region growing degrades under
   missing data; Efficient RANSAC's axis error is 7°+ even on clean CAD; every
   library returns infinite primitives (bounded extents are ours to write); every
   off-the-shelf confidence signal is residual-family and structurally blind to
   systematic bias. One thing works in our favour: early-iteration fits only need to
   aim the *next scan*, not the final evidence view. So the loop needs an explicit
   fit-maturity gate — coarse fits drive exploration; evidence capture only from a
   fit that has passed a measured stability check. That gate's threshold cannot be
   cited; it must be calibrated (E4 below), because the score→error mapping does not
   exist anywhere in the literature.

2. **Menu stability across refits is a new hard requirement with zero literature
   cover.** The loop refits every iteration; Efficient RANSAC is randomized with no
   seed parameter in its public API; slot identity across runs is untested in every
   learned method; no re-scan repeatability experiment exists in the field. A blind
   LLM mid-episode facing a menu that silently renumbers itself is the NBV-Net
   collapse in slow motion. Hard requirements: address views by geometric content
   (normal direction in the table frame + area + position), never by extraction
   index; a cross-refit matching layer (Hungarian on primitive parameters — we write
   it, the libraries don't ship it); hysteresis so a menu entry keeps its label
   unless the underlying fit moved beyond a threshold; vanished or infeasible
   entries flagged, never silently removed.

3. **Labels must be scene-anchored, not object-canonical — the demo forces this.**
   The box gets moved and rotated between runs; azimuthal canonicalization is
   unsolved even in the paper built for canonicalization, and box-like shapes are
   its dominant failure class. So: whatever the representation A/B picks, the frame
   is table/robot-anchored, stated explicitly in every prompt (frame-implicit
   collapses to ~0% for 6 of 8 models), stable within an episode, with no promise
   across episodes. That is fine for demo v2 because the code face is found by
   *content* (the image agent sees codes), not by a persistent name.

4. **The loop inherits NBV's stopping problem, cheaply solvable.** Scan phase: map
   stabilization (MS@δ — stop when relative point gain <1% for 3 steps) is the
   validated generic rule. Agent phase: an always-present explicit stop action plus
   an anti-repeat instruction — both free, both precedented. Do not build conformal
   stopping this month.

### 1.5 Component verdicts

- **Incremental reconstruction:** standard machinery; keep per-view clouds alongside
  the fusion so evidence-view scoring can use live-frame normals (the ~4 cm
  fused-vs-single-view offset makes the fused frame untrustworthy for scoring — a
  design response with direct precedent in Nguyen's pipeline).
- **Primitive fitting:** Open3D `detect_planar_patches` (bounded boxes come out
  nearly for free: center + extent + normal) + CGAL/PCL cylinder + our extent
  extraction + our matching/identity layer + the curvature guard against the
  documented RSPD failure of tiling large cylinders with fake planes. Two hours of
  running it on our real PLYs beats any further reading.
- **Menu crafting:** closed-form per-primitive recipes, gate stack, per-iteration
  set-cover dedup to ~6–12 items, deterministic and seedable, flagged infeasibles,
  stable ordering, explicit stop. This is deliberately boring engineering — which is
  the compliment, and the property that makes it maintainable.
- **Benchmark harness:** the 3-tier design is right and Tier 1 (replay from dense
  capture) is the workhorse — it makes every LLM-side A/B a dataset query instead
  of a robot session, which is the only way we run seeds and permutations at all.
  The unattended-runs idea (every application run records benchmark-grade data) is
  precisely how real deployments accumulate eval sets; wire the recorder into the
  loop skeleton from day one, not later.

---

## 2. Methodology for the build month

Principle: **wire the loop end-to-end first with the dumbest internals, spend the
real effort on the two things no paper measures (partial-fit behaviour, menu
stability), keep a deployable auto-select fallback at every stage, and make every
add-on buy its way in on Tier-1 numbers against a pre-registered kill criterion.**

### Part A — Loop skeleton (`loop-skeleton`)
Week-1 target: scan → fit → derive → menu → pick → execute → refit runs end-to-end
in the kinematic sandbox on the demo box, with planes only and a scripted auto-picker
in place of the LLM. Integration risk dies first; every later part swaps a component
into a working loop instead of assembling at the end. Includes the scan-phase stop
rule (map stabilization), the agent stop action, and the benchmark-grade recorder
(RGB, depth, pose, intrinsics per view) wired in from the start.

### Part B — Partial fitting and identity (`partial-fit-identity`)
The two-hour library shakedown on real fused PLYs first (Open3D patches + CGAL/PCL
cylinder + EMS). Then the three measurements the literature cannot supply:
(1) coverage sweep — fit box/cylinder from 1..N views, normal/axis error vs coverage
fraction, including the half-cylinder worst case; (2) refit churn — run the loop,
measure identity flip rate and label survival across iterations, with the matching
layer and hysteresis on/off; (3) fit-maturity calibration — for every partial fit,
compute available confidence signals, derive the orthogonal viewpoint, measure its
angular error against reference; output is the score→error curve and a maturity gate
threshold with measured false-accept/false-reject rates. Plus the cylinder-tiling
curvature guard.

### Part C — Intelligent-viewsphere machinery (`intelligent-viewsphere`)
The menu factory. Per-primitive closed-form recipes: face → centroid + normal at a
standoff derived from FOV footprint, Min-Z at operating resolution, invalid-band and
the Tier-0 legibility floor (Part E); cylinder → orbit stations binned by the
incidence functional; caps as faces. Gate stack at construction: incidence ≤ τ,
standoff band, analytic/mesh-free occlusion, exact IK + collision on every survivor.
Per-iteration set-cover dedup to a 6–12-item menu. Two-stage decision structure
(surface first, view second), flat leaf menu, flagged infeasibles with next-best,
stable geometric ordering, explicit stop, anti-repeat. Deterministic and seedable
throughout — the properties I need to debug it at 2 a.m.

### Part D — Menu representation A/B (`menu-representation-ab`)
The architect's open question, answered by numbers on Tier-1 replay: angle-style
labels vs plain directional words vs short semantic descriptors vs bare numbered
items (and label+descriptor combos). Frame stated explicitly in every variant; order
permutations as a test condition (canonical, reversed, two shuffles — instability is
a failure signal, not noise); ≥3 seeds; one frontier and one deployable-open model
tier, because nearly every interface result in the review changes sign across that
boundary. Also runs the two-stage vs flat ablation. My prior, for the record: the
deltas between labelled schemes will be small (having *any* explicit scheme is what
matters — SCAFFOLD), and the frame statement plus menu stability will matter more
than the label style. Cheap to prove me wrong.

### Part E — Benchmark harness (`benchmark-harness`)
The 3-tier framework (detail in §5): Tier 0 single-image QA floor, Tier 1
dense-capture replay episodes, Tier 2 full-scale runs. Built before Parts C/D
iterate; QA pairs LLM-written under the architect's review; pre-registered metric,
arms, and losing condition; capture holdout untouched by design iteration.

### Part F — Demo v2 acceptance (`demo-v2-acceptance`)
The box with codes, hardened. Acceptance protocol written before the demo is built:
K placements × M rotations of the box; score = fraction of episodes ending with a
readable, orthogonal evidence image of the code side; plus moves-to-answer, wall
time, and a failure narrative per miss. Robustness to pose comes by construction —
per-episode refit, content-based target finding — so the protocol is a check on the
whole loop, not on a memorized recipe. This is also the ADAM-regression guard: no
per-object scripts anywhere; only per-primitive rules.

### Part G — Sensor smoke test (`sensor-smoke-test`)
The timeboxed carve-out from §1.3.1: half a day. Depth fill rate and plane-fit
quality on the actual demo objects (cardboard, tape, printed codes) across the
operating standoff band and incidence angles; verify Min-Z and invalid-band at the
chosen resolution. Output is a go/no-go on the camera and a default incidence gate
τ. If fill collapses on textureless faces, request a different camera immediately —
the option exists precisely so we can exercise it early.

---

## 3. Proposed solutions

- **S0 — Uniform baselines** (current 12×30°×3 grid + budget-matched Tammes/HEALPix):
  permanent controls, and the fallback action space when the fit fails its maturity
  gate. Never leaves the results table.
- **S1 — Geometry-selects / LLM-consumes:** LLM (blind) picks the target surface and
  when to stop; geometry auto-selects the view; image agent reads the evidence. The
  deployable default and my candidate for what actually runs demo v2.
- **S2 — Intelligent viewsphere with blind LLM menu selection:** the architect's
  design — two-stage choice over the precomputed menu. Must beat S1 and the random
  floor on Tier-1 grounded metrics to earn its extra surface area.
- **S2a/b/c — Representation variants as swappable components:** angle labels /
  directional words / semantic descriptors, decided by Part D numbers, not taste.
- **S3 — Seeing orchestrator over marked renders (gated, likely deferred):** only
  enters if the cheap mark-legibility protocol passes on renders of *our* defective
  clouds — nobody has ever rendered one for a VLM, and the measured failure regime
  (marks on hard geometry: 0–44%) sits uncomfortably close.

## 4. Experiments — prove or disprove

1. **E1 loop smoke test** (A): end-to-end episode on the demo box in sandbox,
   auto-picker. Disproves: the architecture integrates at all.
2. **E2 partial-fit coverage sweep** (B): normal/axis error vs coverage fraction,
   incl. half-cylinder. Disproves: partial fits are good enough to aim the next scan.
3. **E3 refit churn test** (B): identity flip rate across loop iterations, matching
   layer on/off. Disproves: a stable menu is achievable across refits.
4. **E4 fit-maturity calibration** (B): fit-score → derived-view angular error curve;
   gate threshold with false-accept/false-reject. Disproves: any usable gate exists —
   if no score separates good from bad fits, the fallback tier triggers blind and the
   design needs rethink.
5. **E5 reachability audit** (C): fraction of constructed menu items surviving exact
   IK+collision per placement. Calibrates menu sizing and the flagged-entry design.
6. **E6 headline A/B** (E, Tier 1/2): geometry-conditioned menu vs uniform spheres at
   matched budget, same LLM, grounded QA + per-region orthogonality + moves.
   Pre-registered losing condition — this is the month's one claim, provable or not.
7. **E7 selector ablation** (E, Tier 1): LLM picker vs random picker vs auto-select
   oracle over the *identical* menu. Disproves: the LLM adds anything to view
   selection — the field has never shown it and neither have we, yet.
8. **E8 representation A/B + order permutations** (D): angle vs words vs descriptors
   vs index; instability across permutations reported as failure.
9. **E9 two-stage vs flat menu** (D): MOKA's effect replicated on our task or not.
10. **E10 demo v2 acceptance** (F): K placements × M rotations; readable-orthogonal-
    evidence rate; failure narratives.
11. **E11 sensor smoke test** (G): fill-rate go/no-go; sets τ default and standoff band.
12. **E12 Tier-0 legibility floor** (E): can the image agent read the demo codes as a
    function of pixels-per-code (GSD)? Back-propagates the required standoff/framing
    into Part C's recipes — the cheapest experiment with the most direct effect on
    the geometry math.

## 5. The benchmark

**Component-based by construction:** swap exactly one component per comparison —
action space (uniform vs geometry-conditioned), selector (random / LLM / auto-select
oracle), menu representation, budget regime — holding all else fixed. That is the
field's own hold-one-stage-fixed discipline, and it is the only structure that
separates "the space is better" from "the agent used it better."

**Tier 0 — single-image floor.** One image, one decision (read the code, spot the
logo), images sampled from existing captures. Separates reader failure from planner
failure and yields the GSD-vs-legibility curve (E12) that feeds the standoff math.
Cheap, runs first.

**Tier 1 — replay from dense capture (the workhorse).** Per object × placement,
drive the robot once through a fixed shared candidate superset — the uniform grids
*and* the geometry-derived poses — recording RGB, depth, pose, intrinsics per view
(nothing else; the architect's per-view data decision is right and keeps capture
cheap). Episodes then replay offline: a "move" is a dataset fetch; travel cost is
computed in the kinematic sandbox from poses. This is where all A/Bs, seeds, order
permutations and model-tier sweeps run, at zero robot cost. Every real application
run records the same schema — the unattended-run flywheel that turns operations into
future benchmark data and training material.

**Tier 2 — full-scale quantitative.** N objects × M runs (real robot, replay, or
sim), for the headline claim and the demo acceptance numbers: % correct, moves,
duration, per stratum (pure primitive / compound / near-primitive-ambiguous — the
last is where correct fallback to S0 is a *scored outcome*, not a footnote).

**Scoring — two layers, never blended.** Layer 1: per-(object, surface-region) view
quality — angular deviation from the region normal, computed judge-free against the
raw cloud's locally-refit normals (not the fitted primitive that proposed the view —
the circularity guard), [0,1]-normalized on the Dutagaci VSE skeleton, min over a
per-region symmetry-equivalence set, worst case reported alongside the mean (one
grazing evidence image is exactly the complaint that started this project; a mean
hides it). Layer 2: per-(object, question) episodes with view-level labels, reported
as the 2×2 view-correct × answer-correct grid plus an abstention cell — the field's
own numbers show raw end-to-end accuracy overstates grounded competence by up to 70
points, and 26% of trials can be wrong-view/right-answer. The rank correlation
between the layers is itself a reported result. Orthogonality stays one pluggable
scoring mode of the general framework, per the architect — but it is the mode the
month's claim runs on.

**Mandatory arms in every table:** the current {h,v} grid (permanent in-house
control — it is the thing being obsoleted and never leaves the table), a
budget-matched Tammes/HEALPix uniform (the strong control), a random floor, the
auto-select oracle ceiling. Budget matched by construction across compared spaces,
with the number justified from geometry (set cover on the fitted primitives), not
from a tuned constant.

**Discipline:** the finding to hunt in every sweep is a ranking flip, not a score
drop. Paired design over matched cases; Wilcoxon + paired t; WSR intervals on
bounded scores; ≥3 seeds for any LLM arm; per-stratum reporting, no pooled headline;
honest sizing statement up front (at n≲100 episodes we resolve 15–20-point
differences, not 5). Pre-registered before iteration: the metric, the equivalence
threshold, the arms, the statistical test and α, and the explicit losing condition
for the geometry-conditioned space. Ship the artifact: captures, object list,
protocol version, evaluation code, with the untouched holdout enforced from day one.

## 6. Bottom line

The feedback moved the project onto the ground the literature actually supports: QA
inspection where orthogonality is provably right, a geometry-proposes/model-disposes
interface that six working systems already use, and an incremental loop that matches
how real systems operate. The trunk build is boring and precedented — that is the
best possible news. The month's genuine unknowns are exactly three: does the
primitive fit behave on *partial* defective clouds (E2–E4 — the literature is
silent), does the menu stay stable across refits (E3 — zero precedent), and does a
prompted LLM add anything to view selection over auto-select (E7 — never shown by
anyone). Spend the effort there, keep S1 as the deployable fallback, and let Tier 1
arbitrate the representation question the architect deliberately left open. And run
the half-day sensor smoke test in week 1 — the camera is flexible only while there
is still time to swap it.
