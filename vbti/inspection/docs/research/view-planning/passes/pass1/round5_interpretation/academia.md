# Round 5 — Interpretation through the academic persona

**Persona:** research scientist, robot perception / active vision. Criteria applied
throughout: construct validity, novelty against prior art, whether every load-bearing
claim is actually supported, and suspicion of engineering shortcuts that dodge the hard
question rather than answer it.
**Date:** 2026-09-01 · Pass 1, round 5 — **rewritten against the round-4 feedback**
(`round4_accumulate/feedback.md`), which overrides the framing doc where they conflict.
**Inputs:** the design contract (`geometry-aware-view-planning.md`), the
architect's round-4 feedback, and all six round-4 branch syntheses.

The setting this interpretation now answers to, restated so no claim below can hide
behind the older framing: inspection = **answering questions about objects (QA)**, not
defect detection. Planning is an **incremental loop** — scan, fit primitives to the
partial cloud, derive views, pick, execute, refit, replan; there is no complete survey
up front. The decider is a **text-only LLM choosing from a small menu of precomputed
views** (the "intelligent viewsphere"): geometry computes orthogonal, reachable,
well-framed views internally; the LLM never sees primitives, coordinates or parameter
knobs — only the finished menu. A separate VLM agent reads captured images. Month goal:
a working system, no thesis; the one comparative claim is *geometry-conditioned view
spaces beat the uniform viewsphere*, proved by the component-based, three-tier
benchmark, which is auxiliary but is the arbiter for open representation choices.
Demo v2: a box with codes on one side; after inspection the evidence image is the
orthogonal view of that side, robust to the box being moved and rotated.

---

## 0. Standpoint and evidential hygiene

A reviewer reads provenance first. Five facts bound what may be claimed:

1. **The closest published precedent — Wang et al., T-Mech 2024 (decompose into
   primitives → per-primitive orthogonal viewpoints, real 6-DoF arm) — has never been
   read from its own tables** (Branch 3 §4.3: abstract + third-party walkthrough +
   companion paper). Structure: medium confidence. Numbers: low. Any novelty statement
   that leans on its details is unsafe until the primary text is obtained.
2. **Variance reporting is near-absent in every literature we inherit from** (16% of
   445 benchmarks report any uncertainty; 4 of 20 VLM-interface sources). Deltas under
   ~5 points anywhere in this corpus are noise until replicated. Our own harness must
   not reproduce this — three seeds and paired designs are the floor, not aspiration.
3. Several round-2 cautions remain **unverified** (SPAZER's +3.6%-over-random for
   agent-selected views; TagaVLM's +0.94% text-topology ablation). They are the two
   most direct published threats to a text-only decider and have no substitute in the
   read corpus. Do not cite them; do verify them.
4. **We are an N=1 team designing the benchmark, the geometry judge, and the method,
   iterating all three together** — the worst case of `benchmark-lottery`'s "rigging
   the lottery," worse than its own examples. The only usable mitigations are
   procedural: pre-registration of metric, baselines, statistical test and the losing
   condition before iterating; a design-iteration-untouched holdout object set from
   day one.
5. The contract's sensor story was wrong (D405 is passive stereo, no projector —
   Branch 1's datasheet read) and the feedback has since made the camera choice
   flexible and scoped deep sensor modeling out. Fine. But note §1.4 below: the
   *loop's own perception* still runs on depth, and one cheap measurement survives
   the scope cut.

None of this is fatal. It sets the tone: this project's strongest academic position is
that it is placed to *measure several things nobody has measured* — and the feedback's
scope corrections have, mostly, moved the project onto firmer ground, with two new
validity problems created in exchange (§1.5, §1.6).

---

## 1. The updated design against the literature

### 1.1 The QA construct swap: a legitimate retreat that must be declared

The move from defect detection to question-answering is, from where I sit, the single
best decision in the feedback — and I want it stated as a *construct change*, not
absorbed silently.

Under the old construct, the orthogonality requirement was unsupported at its core:
the entire inspection-planning field defines inspection as dimensional conformance
(Scott/Roth/Rivest contains zero occurrences of "defect"); the one defect-detectability
paper read has no incidence-angle term; and raking illumination — industrial practice
for shallow flaws — points the opposite way (Branch 1 §2.4). That tension is now
**dissolved, not resolved**: for QA over object surfaces (read the codes, spot the
logo, name the label), orthogonality is supported by exactly the evidence that exists —
resolution/foreshortening (38,500 → 4,900 points from straight-on to steep tilt, an 8×
loss), framing, and evidence-image legibility. A code on a face is a resolving-power
problem, and resolving power is the one angle→outcome link the literature actually
quantifies.

Two obligations follow:

- **Never retro-claim defect visibility.** The system we build this month supports the
  claim "orthogonal views maximize evidence legibility for surface QA." It does not
  support "orthogonal views maximize defect detectability," and the feedback's own
  "maybe move later" should stay exactly that cautious.
- **The QA construct buys us a judge-free benchmark, and we should take the gift with
  both hands.** Demo v2's payload — codes on a box face — is *mechanically verifiable*:
  a barcode/QR decoder either decodes or it does not; an OCR string either matches or
  it does not. That is the `ycb-protocols` pattern (every score a physical
  measurement) and the `planning-with-the-views` pattern (no judge anywhere)
  transplanted into our task. A benchmark whose primary outcomes are decode-success
  and exact-string-match never confronts LLM-judge circularity at all (Branch 2 Q3).
  QA items should be authored to maximize the mechanically-scorable fraction; the
  LLM-written QA pairs the feedback endorses should be constrained to closed-form
  answers wherever possible, with free-form items quarantined into a separately
  reported, judge-scored slice with disclosed judge identity.

### 1.2 The incremental loop: we have rejoined the field's mainstream — cite it that way

The feedback's correction ("we do NOT have the scanned object up front; it is a loop")
inverts Branch 4's anti-NBV argument, and the honest reading is that this is a
*strengthening*: the iterative structure — scan, update model, plan next view — is the
canonical NBV loop with thirty years of precedent (Kriegel's next-best-scan with its
explore/quality blend; Wu's quality-driven autoscanning; SEE's mesh-free frontier
machinery). Our variant inserts primitive fitting between scan and plan and replaces
the learned/IG selector with a deterministic menu plus an LLM chooser. Three
consequences the build must absorb:

1. **Pan 2025's treat-prior-as-fallible pattern is now load-bearing, not an edge
   case.** Every iteration's fitted primitives are a fallible prior over the next
   iteration's reality. Conservative inflation (union the prior with the current
   partial evidence, distrust where they diverge) is the only measured technique in
   the corpus for this, and it is cheap.
2. **Partial-shape fitting is the common case, not the stress test.** Branch 3's
   verdict stands and now sits on the critical path: nobody in 20 documents tests a
   half-scanned cylinder with one-sided dropout; EMS's 86.4% global-optimality at 20%
   surface coverage (superquadrics, synthetic contiguous cap) is the closest number
   in existence. The first iteration of every episode will fit primitives to
   something like 15–40% coverage. **The angular error of the derived orthogonal view
   as a function of coverage fraction is the single number the whole design turns on,
   it exists nowhere in the literature, and it must be Experiment 1** (§3, E1).
3. **The exact set-cover solve per iteration is a non-issue** — ~1 s at instance
   sizes an order of magnitude above ours (Pan 2025, MA-SCVP, Glorieux). Re-solving
   every round is free; say so and move on. View budget falls out of the solve
   (plan-length-as-stop, validated in MA-SCVP), not from a tuned constant.

And one consequence the feedback has not yet priced:

4. **The loop makes the menu non-stationary by design, and menu instability is a
   named LLM failure mode.** Branch 5 §5.2 flagged deletion-between-turns as "exactly
   the instability a language model handles badly," with no literature cover for the
   alternative. In the incremental design the menu is *recomputed from a refit* every
   iteration — views will appear, vanish, and renumber unless we prevent it. The fix
   is the one Branch 3 derived from CGAL's documented nondeterminism: **address views
   by geometric content in a stable frame (table plane + gravity + face
   normal/area/position), never by extraction index; run a cross-refit matching layer
   (SPFN's Relaxed-IoU + Hungarian pattern); prefer deterministic detectors.** And
   because no re-scan/re-fit repeatability protocol exists anywhere in the
   literature, identity stability across loop iterations is our measurement to make
   (E2). A menu whose "view 3" silently becomes a different physical view mid-episode
   is a correctness bug in the interface contract, not UX polish.

### 1.3 The intelligent viewsphere: the literature's strongest convergence with this design

The feedback's interface correction — geometry computes the good views; the LLM sees
only a finished finite menu; primitives and parameters stay internal — is not merely
compatible with Branch 6; it is what Branch 6's eight findings *prescribe*:

- **Geometry proposes, the model disposes, and the disposal is a label** — six of six
  working systems, zero exceptions [strong]. Selecting a label beats emitting a value
  by +60.7 points (Set-of-Mark). The contract's "VLMs cannot emit metric poses" is
  validated at the output layer with real numbers.
- **Quality by construction, not by scoring.** Mosbach's architecture (every candidate
  placed on the surface normal; orthogonality a property of how a candidate exists at
  all) is route (b) in Branch 4 §2.1 and is precisely the internal recipe the
  intelligent viewsphere should use. It also cuts inadmissible-candidate rates by
  2–4× versus sphere sampling (Gospodnetić Table 4) — normal-oriented generation is
  better engineering before any quality argument.
- **The two-stage decision the architect liked (which surface → which view of it) is
  the largest ablated effect in the visual-prompting literature**: MOKA's hierarchy
  ablation, 0.6–1.0 → 0.0–0.2 when decomposition is removed, versus ~0.2 for
  everything else. Note carefully what MOKA's hierarchy is: decomposing *what is
  decided*, with a flat leaf menu. Two turns, each with a small flat list — never one
  turn with a structured space.
- **One spatial operation per turn; orientation baked into each candidate** (AP-VLM's
  `3Dxy` collapse 0.8→0.5, 0.5→0.1 from adding one rotation axis; PIVOT explicitly
  dropped orientation as too hard). The menu's entries must be complete poses
  internally; the LLM never composes position with orientation.
- **Menu sizes: low single digits per decision** (How-Many-Tools: K≈2 beats K=5 by
  6 pp, 30 pp on hard cases). Per-primitive recipes land at 6–20 addresses per object
  (Branch 5 §3.3), which two-stage decomposition brings under budget per turn.
- **Free, mandatory interface furniture:** an explicit stop action (MapGPT), an
  anti-repeat instruction (Think3D), explicit reference-frame statement in every
  prompt (RoboSpatial's frame-implicit collapse to ~0%), and — this one is ours to
  design, since no paper compares presentations of infeasibility — an explicit
  statement of what was removed as unreachable and why. Silent removal (GraspView)
  plus a blind decider means the model cannot know a surface is uninspectable;
  ObjView-Bench's NBV-Net collapse (0.988 → 0.472 when the action space shrank
  underneath a discrete selector) is the measured catastrophe at the end of that
  road. The menu must be able to express "that one is unavailable; here is next
  best."

Two Branch 6 results temper enthusiasm, and both convert into experiments rather than
objections:

- **No published result shows a *prompted* VLM beating random view selection, with
  variance** (Branch 6 Q15). The positive random-comparisons use trained selectors.
  For us this is double-edged: with a well-crafted menu, most decision weight moves
  into the menu itself, and the LLM's residual job is task-relevance (which face
  carries the codes?) fed by VLM reports from prior captures. That is a feature — but
  it means the benchmark must include a **random-from-menu floor and a scripted
  geometric selector** alongside the LLM, or we cannot say what the LLM contributes
  (E5). If the LLM fails to beat random over the intelligent menu, the correct
  conclusion is that the menu is doing the work — a publishable and useful finding,
  and a different claim than the one we would have been making by accident.
- **Blindness costs decisiveness, not coverage** (MapGPT: blind arm −3.2 SR, −5.8
  SPL, +5.8 OSR). The text-only decider is defensible — SayPlan reaches 86.6%
  executability with zero vision through symbolic verify-and-correct — but the
  measured failure shape is an agent that keeps exploring instead of committing. The
  loop's stopping rule therefore cannot be left to the LLM's own judgement alone:
  hard caps plus the benchmark measuring stop quality separately (Branch 2 Q6
  pattern: hold stopping fixed when comparing spaces; measure agent stopping as its
  own number).

### 1.4 The sensor scope cut: agree on evidence images, dissent on the loop's own eyes

The feedback: once in a good view we care only about the RGB image; deep sensor-noise
modeling is out of scope. For the *evidence* leg I agree without reservation — the QA
construct makes RGB legibility the outcome, and the benchmark measures it directly
(decode success), so no σ(ρ,θ) curve is needed.

The dissent, narrowly: **the loop's own perception still runs on depth.** Primitive
fitting consumes the partial cloud; the partial cloud's quality is incidence- and
material-dependent; and the field's unanimous two-tier practice (smooth model + hard
gate near the knee, six independent groups, knee contested across 45–75°) exists
precisely because grazing captures poison reconstruction. We do not need the curve; we
need **one hard gate value for our sensor** to build into the viewsphere's internal
admissibility test, and the literature cannot supply it (every published knee belongs
to a different sensor class). A half-day dropout-vs-angle check on one plate (E8) is
not "deep sensor-noise modeling"; it is reading the one dial the loop depends on.
Until it is run, adopt 60° as the gate (the majority report across Soudarissanane,
Lichti, Johnson, Mehdi-Souzani) and record it as provisional.

### 1.5 The Tier-1 replay design: sound precedent, one structural validity threat

The three-tier benchmark vision is methodologically the strongest part of the
feedback. Tier 0 (single image, one decision) is the cheapest possible instrument for
the representation A/Bs. Tier 1 (dense pre-capture; a "move" = fetch from dataset) has
direct precedent — MA-SCVP's exhaustive 32-view capture as GT, and every
fixed-candidate-set benchmark in Branch 2 §3.2 — and the plan to compute travel costs
in the kinematic sandbox from poses is exactly how the field does it. Recording every
future application run as benchmark-grade data is the `ycb-protocols` ship-the-artifact
discipline done one better.

But there is a structural problem the feedback does not name, and it strikes at the
headline claim: **a replay dataset captured on a (dense) uniform grid can only serve
views that exist on that grid — and the intelligent viewsphere's entire selling point
is that it produces views that fall *between* grid views.** The contract's founding
complaint (the true orthogonal direction falls between neighbouring bins) applies to
our own Tier-1 dataset. Run naively, Tier 1 is biased *against* the treatment arm: the
geometry-conditioned menu's chosen view gets snapped to the nearest captured view, and
the snap eats precisely the angular advantage being measured. Three mitigations, to be
used together (E6):

1. **Capture the union**: for every benchmark object, capture the dense uniform set
   *plus* the geometry-derived candidate views (computed from a reference fit at
   capture time). The replay executor serves exact frames for both arms.
2. **Report snap error as a first-class column** wherever snapping is unavoidable:
   the angular distance between requested and served view, per episode, per arm. If
   the arms' snap-error distributions differ, Tier-1 comparisons are confounded and
   only Tier 2 can referee.
3. **Validate Tier 1 against Tier 2 once**: rank correlation of method ordering on a
   small overlapping slice (the `embodied-iqa` discipline of reporting each proxy
   link as its own number). A benchmark tier whose ranking does not survive contact
   with the real loop is a development tool, not an arbiter — fine, but then the
   arbiter role the feedback assigns it must be scoped to the choices it can
   actually referee.

Same discipline for the architect's ambition to grow the framework (view-count
optimization, prompt testing, path statistics, simulation, RL, prices): as
architecture, yes — pluggable metric modes over a shared episode format, with
orthogonality one mode among several, exactly as the feedback says. As *deliverables
this month*, no. The build month needs the minimal harness that arbitrates the two
open choices actually blocking the build (menu representation; uniform vs
geometry-conditioned). Benchmark sprawl is the failure mode of benchmark-first
methodology, and the field's graveyard of unmaintained protocol suites (32 of 33
object sets unavailable, per YCB's own survey) says maintenance is the scarce
resource.

### 1.6 The comparative claim, and where it can silently go wrong

"Geometry-conditioned view spaces beat the uniform viewsphere" is provable this month
— but only if three known traps are pre-empted, because each has a published instance
of eating exactly this claim:

- **The metric decides the answer** (Branch 2 F3/C1): on coverage metrics the uniform
  sphere near-saturates (98.41% from a generic 32-view sphere); on
  fidelity/legibility metrics it is clearly beatable. Our construct is per-region
  evidence quality and QA outcome — state up front that coverage was rejected as the
  primary metric and why, or a reviewer will ask why we chose the one family we can
  win on. Reporting the (object, surface region) unit is itself original — no paper
  in the corpus uses it.
- **The baseline must be the strong uniform, not just ours** (Branch 2 Q5): beating
  our naive 12×30°×3 az/el grid is a weaker claim than beating a budget-matched
  Tammes-packed sphere. Report three uniform variants — the {h,v} grid as the
  permanent in-house control (it is the thing being obsoleted; it never leaves the
  table), a budget-matched uniform, and Tammes at matched budget — plus random floor
  and oracle ceiling. Budget justified from geometry (the set-cover count), not
  tuned.
- **"Smaller and more legible" is a confound, not a virtue** (Branch 2 Q14): an
  8-item named menu may beat a 36-cell grid because it is easier for the LLM, not
  because its views are better. The 2×2 — {uniform, geometry-conditioned} × {scripted
  oracle selector, LLM selector} — is the one experiment that separates "the space is
  better" from "the agent uses it better," and it is the single highest-value
  experiment in the month (E4). Without the oracle row, a win is uninterpretable.

Additionally: the grounded-success discipline is mandatory for the demo claim.
Pooled across six models, 26.2% of trials in the one benchmark that measures it were
wrong-view/right-answer; raw answer accuracy overstated grounded accuracy by up to 70
points. Demo v2's claim is *specifically* view-grounded ("the evidence image is the
orthogonal view of the code side"), so every end-to-end number ships as the 2×2 grid
(view-correct × answer-correct) plus an abstention cell — never as a single accuracy.

### 1.7 Novelty, honestly assessed under the new scope

Not novel (and should be cited as adopted): decompose-into-primitives → per-primitive
orthogonal views (Wang T-Mech 2024; the metrology accessibility-cone chain for the
principle); iterative NBV structure; exact set-cover selection; label-selection
interfaces for VLMs; dense-capture replay evaluation.

Novel, each with a confirmed absence behind it: **(a)** primitive fitting inside the
loop on defective partial self-acquired clouds, with measured angular error versus
coverage and an identity-stable addressing layer (no angular-error benchmark, no
re-scan repeatability protocol, no partial-cylinder-with-dropout test exists);
**(b)** the intelligent-viewsphere interface — a geometry-computed, reachability-
filtered, finished-view menu navigated by a text-only LLM, with the representation
choice settled by measurement (nobody has ablated angle labels vs directional words vs
semantic labels for a camera-pose menu; nobody has measured whether VLMs interpret
textual az/el at all); **(c)** the benchmark's (object, surface region) unit with
mechanically-verifiable QA outcomes and a view-grounding grid (no precedent unit; the
2×2 exists once, in an adjacent domain). That is a defensible contribution triple for
a systems paper later, and none of it requires more than the build month to
instantiate.

---

## 2. Methodology for the build month

Decomposed into seven named parts. Each part lists what gets built, the open choices
inside it, and which experiment (§3) settles each choice. The dependency order is
roughly as listed; parts 5–6 run concurrently with 2–4 once part 1 exists.

### 2.1 `loop-scaffold` — the incremental perception–planning loop

The skeleton: capture → integrate into the partial cloud → fit/refit primitives →
derive candidate views → filter (gate, reachability) → craft menu → LLM picks → execute
→ repeat; stop on plan exhaustion, LLM stop action, or hard cap. Design rules imported
with citations: treat the previous iteration's fit as fallible (Pan 2025's
conservative-inflation pattern); re-solve selection exactly every round (~1 s; greedy's
measured tax is 12.5–40% extra views); compute incidence against locally-refit normals
in the live frame, not the fused frame (Nguyen's pattern — this sidesteps most of the
fused-offset hazard by construction); reachability exhaustion is a valid stop with an
explicit report, not an error (Vasquez-Gomez). Open choice: none fundamental — this
part is engineering with strong precedent.

### 2.2 `partial-primitive-fitting` — geometry from defective partial clouds

Build: Open3D `detect_planar_patches` (bounded faces: centroid, extent, normal — the
action tuple directly) + CGAL/PCL cylinder + our own cylinder-extent extraction +
Schnabel-style wider-band refit + a curvature-consistency guard against RSPD's
documented plane-tiling-on-cylinders failure. On top: the **identity layer** — address
primitives by geometric content in the table/gravity frame; Hungarian matching across
refits; deterministic detectors preferred (CGAL Efficient RANSAC is seedless and
non-reproducible by API construction — verified, avoid where identity matters).
Fallback tier: when no primitive fit clears the acceptance bar, degrade to the uniform
sphere for that region and *say so to the decider* (abstention as a scored outcome).
Open choices: acceptance bar for a fit (settled by E1's score→error mapping — no
residual-family score can be trusted a priori, since residual statistics are
structurally blind to systematic bias); cylinder fitter (settled by E1 run per
fitter); whether table-plane and symmetry priors (find-superquadric pattern, unread)
are needed for first-iteration fits (settled by E1's low-coverage end).

### 2.3 `view-crafting` — the internal recipe of the intelligent viewsphere

Per-primitive deterministic rules, quality by construction (Mosbach route b): face →
normal through centroid at a standoff derived from FOV/Min-Z/framing; cylinder → cap
views + K orbit stations with K emergent from the incidence tolerance (Mosbach's
Fig. 6d precedent: rings emerge from a 60° functional, not hand counts); edges/corners
explicitly out of the action space v1 (a decision, recorded, revisitable). Then the
constraint stack: hard incidence gate (60° provisional, E8); IK + collision filter
with **defer-to-next-best composed with filtering** (the fallback ladder's rungs i+iv
— Yu & Gupta's implemented proof they compose), never a bare filter; action
representation `(position, approach direction)` with roll and IK branch resolved at
execution (Zacharias and Wang converge on this from opposite ends; it also shrinks
the menu). Every menu entry carries its full pose internally; the LLM sees none of it.

### 2.4 `menu-interface` — what the text-only LLM actually sees

Two-stage decision (which surface/primitive → which view of it), flat leaf menus of
≤~8 entries, stable content-derived view names, explicit stop, anti-repeat, explicit
frame statement, explicit unavailable-view disclosure ("the underside cannot be
inspected: table contact"). Label content is the open choice the feedback flags —
angle-style labels (architect's tentative preference) vs directional words vs
semantic/face-anchored labels vs bare indices — and it is settled by E3, not by taste.
Literature priors going in: having *some* explicit scheme matters more than which
(SCAFFOLD: +5.7–10.3 baseline→any, 4.6 best→worst); numbers are fine as labels of an
enumerated set and bad as quantities to reason over; TwNV's "Discrete" condition —
the closest to our interface — is the least-documented result in its paper. So the
A/B is cheap insurance, not a research program. Menu order is a variable (oracle
best-vs-worst ordering swings up to 73 points; no correlation with option count) —
E3 permutes it.

### 2.5 `benchmark-harness` — the three tiers as instruments

Tier 0: single image → single decision (read/decode/locate), sampled from existing
data; the instrument for E3 and for VLM-agent regression tests. Tier 1: dense
pre-capture replay with the union-capture + snap-error discipline of §1.5; episodes
score view choice, moves, travel (kinematic sandbox), and QA outcome. Tier 2: N
objects × M runs, real robot / replay / sim, full statistics. Cross-cutting: episode
format is shared and versioned; per-view record is RGB, depth, pose, intrinsics
(feedback's ruling); QA items authored for mechanical verification first (decode
success, exact match), judge-scored free-form quarantined; orthogonality is one
pluggable scoring mode of the framework, per the feedback. Statistics: paired designs
over matched cases (object, placement, question), Wilcoxon + paired t, ≥3 seeds for
anything with an LLM in it, WSR intervals for bounded scores, per-stratum reporting
(pure primitive / compound / near-primitive-ambiguous), no pooled headlines, random
floor and oracle ceiling in every table, honest sizing statement (at n≲100 we resolve
15–20-point differences, not 5).

### 2.6 `comparative-experiments` — the pre-registered claim

The 2×2 of E4 plus the selector-floor row of E5, run at matched budget under the
three uniform controls, with the losing condition written down before the first run
(§4). This part owns the month's one claim.

### 2.7 `demo-v2` — the box with codes, as a grounded-success artifact

The demo is also an experiment (E9): box moved and rotated across trials; success =
the 2×2 grid's grounded cell (evidence image within the angular tolerance band of the
code face's normal AND codes decoded), plus abstention scored when the code face is
placed unreachable (face-down). The demo inherits everything: stable face identity
across re-placements (part 2), reachability disclosure (parts 3–4), mechanical
verification (part 5).

---

## 3. Experiments — each designed to be able to fail

- **E1 — Partial-fit fidelity curve** (parts 2, 5). For primitives with known ground
  truth (CAD or metrology-grade reference scan of the calibration subset): fit at
  swept surface-coverage fractions and defect levels (per-view rigid registration
  jitter injected *before* fusion; incidence-keyed dropout; systematic per-view depth
  offset — our triad, absent from every published corruption suite); report **angular
  error of the derived orthogonal viewpoint** vs coverage, per fitter, plus the
  score→error mapping for every available confidence signal. *Disproves:* if the
  first-iteration error exceeds the evidence tolerance band, primitive fitting cannot
  drive view crafting from early partial clouds, and the loop must lead with a
  coarse-scan stage or the fallback tier — a design change, made on a number.
- **E2 — Identity stability across the loop** (part 2). N repeated episodes per
  object with re-placements: primitive count stability, parameter dispersion,
  identity flip rate after matching, menu-entry churn per iteration. No published
  protocol exists; this is original by construction. *Disproves:* stable menu
  addressing — if flip rate is material, the interface contract needs redesign
  (coarser addresses, or menu freezing after iteration k).
- **E3 — Menu-representation A/B** (parts 4, 5; the feedback's named use of the
  benchmark). Tier 0 and Tier 1: {angle labels, directional words, semantic
  face-anchored labels, bare indices} × {canonical, reversed, 2 random orders,
  adversarial top-2-at-extremes} × ≥2 model tiers (frontier closed + deployable
  open), ≥3 seeds, SCAFFOLD's template. Report per-tier; expect the winner to change
  across model tiers (Branch 6 §1.1). *Disproves:* the architect's angle-label
  preference — or shows the choice is noise-level, which licenses picking by
  convenience.
- **E4 — The headline 2×2** (part 6). {uniform sphere, intelligent viewsphere} ×
  {scripted oracle selector, LLM selector}, matched budget, three uniform controls,
  Tier 1 then a Tier-2 slice. Primary metrics: per-region angular evidence score and
  QA grounded-success grid; moves and travel as secondary. *Disproves:* the month's
  claim, in the oracle row (space quality) separately from the LLM row (usability).
- **E5 — Selector floor** (part 6). Random-from-menu vs LLM-from-menu vs scripted
  geometric selector, same menus, with the chance-corrected formalism. *Disproves:*
  the LLM's contribution over the menu — a sign-flip here reframes the system claim,
  it does not sink the system.
- **E6 — Replay validity** (part 5). Snap-error distributions per arm on union
  captures; Tier-1 vs Tier-2 rank correlation on an overlapping slice. *Disproves:*
  Tier 1's fitness as arbiter for pose-sensitive comparisons.
- **E7 — Decision-framing ablation** (part 4). Two-stage vs single-turn flat menu
  (MOKA's ablation transplanted). Cheap; predicts a large effect; if absent at our
  scale, the interface simplifies.
- **E8 — Incidence gate check** (part 3). One plate, one half-day sweep:
  depth-fill/noise vs incidence at operating standoffs, separated matte vs glossy.
  Sets the gate; does not model the sensor. *Disproves:* the provisional 60° gate.
- **E9 — Demo robustness protocol** (part 7). Box re-placed and rotated per trial,
  including adversarial placements (code face away from the arm, near-table,
  face-down → abstention expected); report the grounded grid + abstention, per
  placement stratum, with every failure narrated (`ycb-protocols`' mandatory field).
  *Disproves:* the demo claim as a robustness claim rather than a lucky recording.

Deliberately absent: sensor-noise modeling beyond E8, coverage-metric leaderboards,
learned selectors, RL, VLA — per the feedback's scope, and I concur with all of it
this month.

---

## 4. Pre-registration — written before the first comparative run

Per `benchmark-lottery` and Branch 2 §8.5, decided in advance, versioned with the
harness: the operational definition of "orthogonal enough" (the tolerance band, from
E1/E8, with exclusions named); the candidate sets and budgets and their geometric
justification; the symmetry-equivalence relation per (object, region) (Dutagaci's Σ
construction; the addressing scheme must be invariant under it — a "no" is a
benchmark-detectable defect); the defect conditions; the statistical protocol
(test, replicates, α, resolvable effect size); which arms our metrics advantage and
why that is acceptable; the holdout object set, untouched during design iteration;
and **the losing condition**: the intelligent viewsphere is reported as *not better*
if, at matched budget against the budget-matched uniform controls, the oracle-row
per-region evidence score and the grounded-QA rate are not better by more than the
pre-stated resolvable effect, or if the advantage vanishes in execution space, or if
it does not survive the defect conditions. Committing to that sentence now is what
makes the eventual win citable.

## 5. Reading debts that survive the scope change

Reordered by the new design: **(1)** Wang et al. T-Mech 2024 primary text — the
closest precedent must be read from its own tables before any novelty claim ships;
**(2)** SPAZER + TagaVLM — the two unverified direct threats to the text-only decider;
**(3)** find-superquadric (partial cloud + table prior — the closest deployment match
for first-iteration fits) and M3C2-EP (registration covariance into geometry);
**(4)** the solids-of-revolution aspect-graph paper (one DOI from Branch 5's read) for
the cylinder's view-space structure. Demoted by the QA swap: M2AD/Real-IAD-MVN
(defect-visibility GT — now relevant only to the "maybe move later" future) and
arXiv:2201.11924 (dropout simulation — relevant only if E1's synthetic corruption
needs upgrading). The metrology standards (VDI/VDE 2634, ISO 10360-13) and
photogrammetric network design remain the two unsearched lines that could hand Tier 2
an artefact-based ground truth; low urgency, high option value.

---

*One closing sentence in persona: the round-4 feedback traded an unsupported premise
(defect visibility) for a supported one (evidence legibility), moved the architecture
onto the field's best-precedented skeleton (NBV + quality-by-construction candidates +
label-selection interface), and in exchange created two new obligations that are ours
alone — proving the fit survives partial defective clouds inside the loop (E1/E2), and
keeping the replay benchmark honest about views that live between its own samples
(E6). Meet those two and the month's claim stands on evidence; skip them and it stands
on taste.*
