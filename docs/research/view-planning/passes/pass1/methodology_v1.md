# Methodology v1 — the build month

**Status:** consolidated proposal for the architect. Supersedes the three round-5
interpretations (`round5_interpretation/{anton,academia,industry}.md`) as the working plan;
they remain the source material.
**Inputs:** round-4 feedback (binding, overrides the framing doc) · framing doc · all three
round-5 interpretations · all 17 round-6 part verifications · round-7 adversary + completeness.
**Date:** 2026-09-01.

**What this document is.** One canonical part list, one canonical experiment list with
non-colliding IDs, the benchmark design, and a register in which **every** round-6 and
round-7 issue is either resolved (with the resolution stated) or carried explicitly as an
open question for the architect (§9). Nothing here changes the architect's scope: the loop
is incremental, the viewsphere is intelligent, the LLM decides from a finished menu, the
benchmark is auxiliary and is the arbiter for open representation choices, demo v2 is the
milestone, VLA/RL/defect-detection/deep-sensor-modeling stay out.

---

## 1. The claim, in one sentence — and the floor if it fails

The round-7 adversary's first rejection was that no single sentence exists saying what the
month claims, measured how, against what. Proposed sentence (needs the architect's ruling on
the comparison target — **OQ-1**):

> **Claim.** At matched view budget, the geometry-conditioned menu (intelligent viewsphere)
> produces evidence images closer to the surface normal and higher grounded-QA success than
> the deployed 12×30°×3 uniform viewsphere, on the same incremental loop, same reader, same
> objects — measured system-level, co-headlined by (a) grounded QA (view-correct ×
> answer-correct + abstention) and (b) achieved evidence incidence scored against
> system-independent ground truth.

Two deliberate scopings, both from the adversary's error budget (§4, X0):

- **Comparison target is the deployed grid** (the thing being obsoleted), *binding*.
  Tammes/HEALPix at matched budget stays in every table as an **exploratory strong control**,
  not as the claim-bearing comparison. Reason: the assembled error budget says the expected
  difference against a budget-matched Tammes sphere is single-digit degrees ⇒ ~2 % GSD, which
  no instrument in this month resolves. Claiming against it manufactures a null.
- **System-level, not factor-level.** In a loop the menu at round *k* depends on the
  selector's earlier picks; the cells are policy-in-the-loop comparisons. A matched-cloud-state
  variant (frozen partial cloud after a fixed scripted scan prefix) is run as the
  factorial-clean companion (X12c) and reported beside, not instead.

**Null-world deliverable (pre-committed, so the kill switches are safe to leave armed).** If
X3 comes back flat, X12 ties, X13 ties and X10 ties — the *expected* outcome on the design's
own arithmetic — the month still ships:

1. a working incremental scan→fit→derive→menu→pick→execute→refit system on real hardware;
2. demo v2 passing its acceptance protocol (or failing it with narrated causes);
3. a measured win over the deployed grid, or the honest absence of one;
4. five curves nobody has published: derived-viewpoint angular error vs coverage (X4);
   fit-score → viewpoint-error calibration with FA/FR (X5); identity flip/churn across refits
   (X6) and the flip-tolerance curve (X7); RGB legibility vs incidence and GSD (X3);
   menu-label ablation for a camera-pose menu (X10);
5. the 3-tier harness with a reusable capture set.

That list is the floor, written before the runs, so a null month is a deliverable and not a
scramble that invites post-hoc item tuning.

---

## 2. Canonical parts (the three interpretations merged)

| ID | Part | Absorbs (round-5/6 names) | One-line contract |
|---|---|---|---|
| **P0** | `bench` — sensor + legibility bench | `sensor-smoke-test`; anton's slim depth check | Two numbers the design otherwise guesses: τ_scan (depth admissibility) and the RGB legibility/GSD curve. Half a day each, week 1. |
| **P1** | `loop` — incremental scan→fit→derive→menu→pick→execute→refit | `loop-skeleton`, `loop-scaffold` | The running skeleton every later part swaps into. Owns registration, the accumulating cloud, stop arbitration, the recorder. |
| **P2** | `fit` — primitives from partial, growing clouds | `partial-fit`, `partial-primitive-fitting`, fitting half of `partial-fit-identity` | Bounded faces + cylinders from the round-*k* cloud, with a calibrated maturity gate and an abstention output. The load-bearing wall. |
| **P3** | `identity` — stable addresses across refits | `identity-persistence`, identity half of `partial-fit-identity` | Content-addressed primitives + matcher + hysteresis so "the code face" survives refits within a session. |
| **P4** | `factory` — the intelligent viewsphere (view construction) | `menu-crafting`, `view-crafting`, `intelligent-viewsphere` | Closed-form per-primitive view derivation, two-layer scoring (gate + continuous rank), full constraint stack, witness configurations, abstain/fallback state. |
| **P5** | `interface` — what the text-only LLM sees | `menu-interface`, `menu-representation-ab` | Two-stage menu, stable names, stop/abstain/anti-repeat/frame statement/diffs, serializer as a swappable component. |
| **P6** | `harness` — the 3-tier component benchmark | `benchmark-harness` | Recorder + Tier 0/1/2 + scoring + arms + statistics. Auxiliary in effort, arbiter in role. |
| **P7** | `comparison` — the one claim | `headline-comparison`, `comparative-experiments` | Spaces × selectors at matched budget with the validity gates attached. |
| **P8** | `demo` — demo v2 acceptance | `demo-v2`, `demo-v2-acceptance` | Box with codes, moved and rotated, scored on the grounded grid, with a generalization probe. |

Round-6 verified 17 documents that were 8 parts wearing three sets of names. This table is
the mapping; from here on only P0–P8 and X0–X18 are used.

---

## 3. Proposed solutions, part by part

Each part states: what gets built, the design decisions that were contested and how they are
resolved, and which experiment can disprove it.

### P0 — bench (week 1, one day total)

**Build.** (a) Depth bench: single-view fill rate and plane-fit residual/normal error vs
incidence and standoff, on real demo materials (cardboard, tape, printed codes) plus one
textured control plate; matte and glossy separated; tilt about **both** axes (the D405 has a
stereo baseline — Scott's 60°/70° anisotropy is a warning, not a curiosity); run in the demo
cell's lighting, with the exact stream profile recorded; verify Min-Z (100 mm @ 1280×720) and
the invalid band (~10 % of image width at 10 cm) at that profile. (b) RGB legibility bench:
decode/read accuracy vs incidence at matched distance, and vs GSD (pixels per code feature),
including **glossy code material under the demo cell's own lighting** — this is where the
specular-at-normal worry lives now that evidence is RGB, and it is the only place the pass
ever closes it.

**Resolved decisions.** Pass bar written *before* running (X2): no-go on the depth axis if a
single-view plane fit on the worst demo face at operating standoff, sub-τ incidence, errs by
more than the menu's angular tolerance or finds no plane. τ_scan is derived from the
**textured** control (where correspondence is valid and a knee can exist); go/no-go is judged
on the **textureless** demo face; if no knee appears, τ_scan = 60° recorded as
literature-default, not measured. The camera verdict is split by axis: a depth-axis "no-go"
is a week-1 swap trigger; a depth-axis "go" does **not** clear the camera — RGB evidence
quality remains the architect's demo-v2 call.

**Disproof.** X2 can veto the sensor in week 1 while swapping is still free. X3 can soften the
whole orthogonality premise (if reading is flat to 40–50° off-normal, menu density relaxes) —
the bravest disproof condition in the month.

### P1 — loop

**Build.** capture → register/integrate → refit primitives → derive views → gate/verify →
craft menu → selector picks → execute → repeat. Week-1 target: end-to-end in the kinematic
sandbox on the demo box, planes only, scripted picker, recorder wired in.

**Resolved decisions.**
- **Stop arbitration** (fork closed): primary stop = question answered ∧ menu/fit stability ∧
  hard cap. Any bootstrap scan is a **fixed tiny count (1–3 views)** to seed the first fit —
  never run-to-stabilization; MS@δ is demoted to a fallback termination guard with its
  difficulty-variance caveat attached. Rationale: a run-to-stabilization scan phase is a
  mini-survey and re-imports the premise the architect overrode, and at MS@δ's published
  operating point (~9 of 128 views) it would hand every matched-budget comparison to the
  baseline. Stop policy is an **injectable component** so the harness can hold it fixed.
- **Scripted picker is a generic rule over the menu** (highest-ranked unvisited item for the
  queried target; stop when answered), recomputed from the fit each round, zero object-specific
  constants. This is the one place a per-object recipe could leak in and then ship; stated as a
  rule, it doubles as P7's scripted-selector row.
- **Visibility is analytic at primitive level** (closed form for planes/cylinders, SEE-style
  mesh-free point-proximity as pre-fit fallback). The "~1 s" number is the ILP *solve*; the
  field's per-round sink is the visibility matrix (41–62 s ray-traced per cylinder; 250–2000 s
  in Pan). Analytic-by-construction is a design obligation of this part, and X9 measures the
  whole-round wall clock, not the solve.
- **Set cover is menu pruning subordinate to the question**, not a coverage plan: exact solve
  because it costs ~1 s, not because of the greedy tax (that number measures executed views in
  full-coverage plans — a different construct). Question-relevant redundancy is exempt from dedup.
- **Fallible prior operationalized**: each primitive carries observed vs extrapolated support
  tags. Extrapolated geometry may drive **exploration** views ("go see the far side"); it may
  never carry an **evidence** claim. Pan's conservative union is imported as cheap hygiene and
  cited as pattern (~1 pp, sim-only), not as de-risking.
- **Registration named and logged**: the method (kinematics+hand-eye, ICP refinement, or
  σ-weighted fusion) is stated in the code and its per-round residual is logged by the recorder
  from day one. The ~4 cm fused-vs-single-view offset is a pipeline defect, not extrinsics
  (RGB and depth share the left imager); live-frame locally-refit normals **detect** it at
  arrival, Tarbox-style margin in the admissible set **prevents** it — both rungs, not one.
- **Recorder claim corrected**: the loop and the dense-capture tool share one writer and one
  schema (RGB, depth, pose, intrinsics). Sparse episode logs feed Tier-0 sampling and archives;
  they cannot serve Tier-1 replay for other arms. The first **dense** capture of the demo box
  is an explicit week-1 deliverable.

**Disproof.** X9: if the incremental loop needs as many moves as a one-shot dense survey to
produce a usable menu, the premise buys nothing and the architecture reverts to
survey-then-inspect. Sandbox runs are falsification-only (a sandbox pass proves integration,
not the premise) and are re-run on replayed real captures once P2 lands.

### P2 — fit (the load-bearing wall)

**Build.** Open3D `detect_planar_patches` (verified on this machine: returns
`OrientedBoundingBox` with `.center`, `.extent`, z-column of `.R` as normal — the bounded-face
action tuple) + CGAL region-growing / PCL cylinder + our cylinder-extent extraction + EMS as a
sanity vote; wider-band refit; curvature-consistency guard against RSPD's documented
cylinder-tiling; normals re-fit locally near the inspected region.

**Resolved decisions.**
- **Kill condition recalibrated.** Not "menu angular resolution" (30° bin spacing) but a
  **paired comparison of distributions**: per placement, derived-view angular error vs the
  *snap error* of (a) the deployed grid and (b) the budget-matched strong uniform. Report
  median and 90th percentile. Two bars, not one: an **exploration bar** (error small enough
  that the next scan lands on the intended surface, derived from the FOV footprint) and an
  **evidence bar** (beats the control's snap error). The verdict is read at the coverage the
  loop actually reaches by menu-derivation time — X4 consumes X9's coverage trajectory and
  reports *the round at which the evidence bar is first met*, per object class.
- **Partials generated three ways, all labelled**: real per-round loop accumulations
  (primary), real single views, synthetic sweeps corrupted with the defect triad (per-view
  rigid jitter injected **before** fusion, incidence-keyed dropout, systematic per-view offset).
  Never plain geometric masking — that reproduces the exact hole-punching error the corpus was
  condemned for. The per-view data the triad needs exists on disk (27 runs under
  `inspection/data/runs/*/` with per-view capture directories preserved).
- **Ground truth is system-independent, stated per surface class**: gravity + calibrated table
  plane for horizontal faces; printed fiducials or robot touch-off (the freedrive-tap machinery
  already in the repo) for verticals and the cylinder axis; CAD/machined calibration objects for
  the sweep, with stated GT uncertainty ≪ effect size. Never "fit the full cloud with the same
  stack and call it truth" — that instrument is blind to exactly the systematic bias this
  experiment exists to catch. Cardboard is kept for the depth-adequacy check only (a bulged
  cardboard face's true normal is ill-defined at the effect-size scale).
- **Maturity-gate features are not residual-only.** The residual family is *predicted* to fail
  (SHREC cone: parameter error 1e11 at low residual), so a flat mapping would be a
  re-derivation, not a finding. X5's candidate feature set explicitly adds the two non-residual
  signals the system already computes: **cross-fitter disagreement** (Open3D patch vs CGAL/PCL
  cylinder vs EMS) and **cross-round parameter dispersion** (X6's own metric). Only then does
  X5 have a live hypothesis and a real kill ("nothing separates good fits from bad ⇒ the
  fallback triggers blind and the design needs a rethink").
- **Gate features must be runtime-computable.** Plane "surface-coverage fraction" is circular
  at runtime (the patch extent is defined by the observed points); cylinder arc coverage is
  computable from the fitted radius. Calibration uses coverage as the sweep axis; the deployed
  gate uses only runtime-available features.
- **EMS role pinned**: sanity vote; disagreement ⇒ abstain to the fallback tier; agreement is
  never evidence of absence of bias (a one-sided partial plausibly biases both fitters the same
  way), and EMS's family excludes compounds.
- **Curvature guard gets a two-sided test** inside X4: large-radius cylinder ⇒ zero spurious
  planar faces (true-positive suppression), and a bowed real cardboard face ⇒ **not** suppressed
  (false-suppression rate). Over-suppression is the worse failure: tiling produces visibly
  absurd menus, suppression produces quietly incomplete ones.
- **Claim hygiene**: the wider-band refit's 5–25× precedent is position/scale, not orientation
  — X4 runs it with/without arms and measures the angular effect instead of citing the number.
  All absence claims are phrased "absent from the corpus we read (20/39 documents; sections E/F
  unread; Wang T-Mech never read primary)".
- **Effort cut, honestly**: calibrate only in the coverage regime the loop actually visits
  (first-round partials ≈ 15–40 %) on the two demo-class shapes; the full 10–100 % × 3-defect-axes
  × per-fitter cross product is a research program and is not the month's job.

**Disproof.** X4: if derived-view angular error at loop-realistic coverage is not
stochastically better than the control's snap error, the geometry-conditioned menu cannot win
by construction and the claim dies in week 2 — the cheapest place to find out.

### P3 — identity

**Build.** Address primitives by geometric content — normal bucket + area + position — in the
**robot base frame** (table calibrated in the cell, gravity known), never by extraction index;
a cross-refit matching layer; label-level hysteresis; nearest-centroid re-anchoring of the
task-relevant referent.

**Resolved decisions.**
- **Matcher specified**: parameter-space matching (normal angle + centroid distance + area
  ratio) with explicit dummy rows for **unbalanced assignment** (primitive count is supposed to
  grow every round), stated tolerance to monotone area growth and centroid drift (round 1 sees
  half the top face; that is the point of scanning), thresholds written down and swept. RIoU
  over point membership is not used directly — the cloud is re-fused and re-voxelized every
  round, so membership indices are not stable.
- **Scope sentence**: within-session identity is **promised and gated**; cross-placement
  dispersion is **reported and unpromised**. Demo v2 does not need cross-episode identity — the
  code face is re-found by content each episode — so the demo's robustness claim does not
  secretly depend on the harder property.
- **Correctness beside stability**: flip rate alone is gameable by aggressive hysteresis
  (stable-and-wrong is worse than unstable for demo v2 — an orthogonal evidence image of the
  wrong face, scored confident). X6 therefore reports **wrong-attachment rate** against
  independent correspondence GT (dense reference decomposition or CAD), **target-referent flip
  rate** above global flip rate, flips **anchored to round 1** (not round-to-round, which permits
  drift), and churn **decomposed** into {adds from new geometry, drops from reachability,
  identity flips, jitter relabels}.
- **Cylinder azimuth identity** is defined by a scene-fixed convention (robot-base direction),
  not by the fit's arbitrary azimuth origin, or orbit-station identity is not even well-posed.
- **Determinism is not an identity mechanism.** Deterministic detectors buy debugging
  reproducibility; the loop's input changes every round by construction, so identity rests
  entirely on the matcher + hysteresis. The ablation is {matching, hysteresis} on/off.
- **Half-day prior-art check on plane-landmark data association in plane-based SLAM** before
  writing a bespoke matcher; the novelty claim is re-scoped to the *protocol*, hedged to the
  read corpus.

**Disproof.** X6 measures the flip rate; X7 measures what a flip *costs* (injected flips at
0/5/20/50/100 % × {announced diffs, silent}, grounded-QA degradation). The acceptance bar is
"measured flip rate below the knee of X7's curve". The 100 % cell **is** the anonymous-menu
fallback, measured rather than asserted — which also settles whether the identity layer's build
cost is justified at all. The NBV-Net 0.988→0.472 collapse is downgraded from equation to
analogy: it is a trained selector emitting an index into a shrunken space and terminating; our
LLM cannot pick an infeasible view by construction.

### P4 — factory (the intelligent viewsphere)

**Build.** Per-primitive closed-form derivation → constraint stack → two-layer scoring →
per-round dedup → menu state (serve / abstain).

**Resolved decisions.**
- **Two-layer scoring, not gates alone** (the most severe round-6 finding). Boolean gates
  decide admissibility; a **continuous incidence/framing rank** picks the per-region
  representative, and the dedup selects *among region representatives* and never drops a
  region. Without this layer, orthogonality is a floor rather than an objective, and a cylinder
  region can be "covered" by a neighbouring station at ~τ off-normal — Albert's original
  complaint recreated inside our own machinery, on the curved half of the design.
- **Three tolerances, three owners** (replacing one overloaded τ):
  - **τ_scan** — depth admissibility for fitting. Owner: X2.
  - **τ_menu** — menu density / orbit-station spacing; guarantees every inspectable region has
    a candidate inside the legible band. Owner: X3 (RGB legibility curve).
  - **τ_evidence** — the "orthogonal enough" success band for the grounded view-correct cell.
    Owner: **the architect**, as a requirement (recommendation: ≤15° off-normal), with X3 able
    to *tighten* it but never loosen it. Rationale: if the band is set from the decode curve,
    view-correct collapses into answer-correct, the wrong-view/right-answer cell empties by
    definition, and the demo can ship a 45° oblique image that decodes — passing the grid while
    violating the requirement that started the project (**OQ-3**).
- **Fit-maturity gate is inside the factory's contract**: input = X5's calibrated threshold;
  output = per-primitive serve/abstain state; abstain ⇒ fallback-tier item, disclosed in the
  menu. Without this the factory launders a 20°-wrong axis into six plausible views orbiting a
  hallucinated cylinder.
- **Standoff is clamped, not merely lower-bounded**: `clamp(FOV-fit, Min-Z + margin, GSD
  ceiling)`. When the interval is empty (large face, small code — the demo's own shape), the
  recipe emits a deterministic **ROI close-up** (same normal, centred on the VLM-reported code
  region) and/or FOV-grid tiling of the bounded face with sub-face cells addressed under the
  parent face. Still closed-form, still per-primitive-class, still not a per-object recipe.
- **Witness configuration per menu entry**: verification computes and caches a specific roll +
  IK branch + a motion-plan query from the current (or canonical home) pose; execution uses the
  witness or re-verifies before commitment. "An IK solution exists" does not imply the executed
  branch is collision-free — the corpus never engages this, and the flagship joint-space paper
  still fails 8/100 end-to-end. Execution-time failure routes through the flagged/next-best
  machinery, and execution-failure rate of pre-verified items is a standing logged metric.
- **Roll is not free where it matters**: for elongated faces roll is constrained at build time
  (folded into the witness) so the FOV-derived standoff actually frames the face; the
  invalid-depth band's scene location is a function of roll and is recorded in the witness policy.
- **Collision world stated**: inflated union(accumulated cloud, fitted primitives, prior) —
  checking against fitted primitives alone passes poses that collide with unseen geometry.
  Occlusion is in the stack (analytic at primitive level), with the partial-cloud hole ambiguity
  ("occluded" vs "never sensed") named in the spec rather than discovered at 2 a.m.
- **Reachable-but-degraded is disclosed, never silently served**: if all within-band stations of
  a region are blocked and only a 50°-off-normal station survives, the menu carries the quality
  band on the item ("best remaining view of face 3: ~50° off-normal") or routes to abstention.
  Deferring to a *different region* is a scanning-phase mechanism and does not answer an
  evidence question.
- **Dedup honestly demoted**: exact set cover is a tidy pruning pass with an emergent size
  (expected 6–12); the leaf menu is kept small by the two-stage structure, not by an enforced
  budget (max-k-coverage has essentially no precedent and is not used).
- **Cold start declared**: round 0 has no fit ⇒ no trust ⇒ uniform bootstrap tier.
- **Cylinder-orbit recipe ships as ours, derived** (Mosbach Fig. 6d is behavioural precedent for
  rings emerging from a functional, not a recipe); long cylinders get multiple rings along the
  axis from the axial FOV footprint. Read GS-NBV and the solids-of-revolution aspect-graph paper
  before any external claim.

**Disproof.** X8 (construction audit, half a day): achieved vs planned incidence on executed
captures (scored against locally-refit **raw-cloud** normals, never the primitive that proposed
the view), framing verified on real images, worst-case evidence angle across the mantle for the
chosen K, per-region menu completeness, reachability survival rate, execution-failure rate of
pre-verified items. Without this, "quality by construction" is an assertion.

### P5 — interface

**Build.** Two-stage decision (which surface → which view of it), flat leaf menus ≤ ~8, stable
content-derived names, always-listed stop, explicit abstain/fallback item, anti-repeat,
reference frame stated in every prompt, unavailable items flagged with reason, menu diffs
announced between rounds, serializer swappable.

**Resolved decisions.**
- **The serializer is decided by X10, not by taste** — the architect's angle-label preference is
  on the record as a hypothesis. Variants are **information-matched** (same facts, different
  encodings); content-derived descriptors that leak task information ("the labelled side") are
  either excluded or run as a separately-labelled arm, or the finding degenerates into "more
  task information helps."
- **Frame is deconfounded from format**: `{frame: camera-relative, table/base-anchored} ×
  {format: angles, words, hybrid, index}`. "Frame stated explicitly" equalizes disclosure, not
  identity; without factorizing, a win for either variant is unattributable.
- **Instruction-following pre-check restored** as a gate on the composite prompt before spending
  seeds on a below-threshold open model (prescribed by the reading, dropped by all three
  interpretations). The composite (two-stage + stop + anti-repeat + frame + disclosure + diffs)
  is exactly the "added machinery" that is net-negative below the capability threshold.
- **Anti-repeat carve-out**: repeats are allowed when the diff announces the underlying fit
  changed — otherwise the rule blocks the legitimate recapture the maturity-gate design requires.
- **Stage-1 overflow rule**: above ~8 surfaces, group by primitive. The ≤8 discipline previously
  governed only the leaf.
- **Two invented mechanisms get arms instead of assumptions**: disclosure-vs-silent-removal and
  menu-diff-announcement (X11) — both are confirmed field gaps, both are one serializer flag on
  replay episodes, and both are scored on **decisiveness** (moves, repeat-hunting of unavailable
  surfaces) as well as accuracy, since that is where blindness is measured to hurt.

**Disproof.** X10 with a stated MDE and a pre-stated decision rule (winner = best grounded
per-decision accuracy; order-stability as lexicographic tie-break; deltas below MDE ⇒ cheapest
serializer). X11 can drop the second turn if flat menus match two-stage.

### P6 — harness

**Build.** Recorder (day one, shared writer/schema with the loop) → Tier 0 single-image items →
Tier 1 dense-capture replay → Tier 2 full runs. Everything a swappable component: view-space
generator, selector, serializer, stop policy, scoring mode.

**Resolved decisions** (design detail in §5).

**Disproof.** X16 self-checks: replay determinism, travel-cost sanity against real executions,
Tier-0 item audit (manually verify every scored-wrong item — the precedent found benchmark GT
itself wrong), Tier-1↔Tier-2 **winner-agreement** per comparison (a Spearman ρ over 5 arms
cannot distinguish 0.9 from 0.4 and must not be reported as a validity claim).

### P7 — comparison

**Build.** Spaces × selectors, all cells reported.

- **Space column (a family, not a cell):** deployed 12×30°×3 grid (permanent, claim-bearing per
  OQ-1) · budget-matched uniform · Tammes/HEALPix at matched budget (exploratory strong control)
  · **standoff/framing-parity arm** (uniform directions + the treatment's standoff/framing/GSD
  logic) · intelligent viewsphere.
- **Selector rows (four, named — the "scripted oracle" conflation resolved):** random-from-menu
  (chance-corrected floor) · scripted geometric auto-select (deployable null, no GT access) ·
  LLM · **outcome oracle** (best achievable grounded QA per space by exhaustion over captured
  candidates, identical GT access both arms). The outcome oracle carries the losing condition;
  an angular-metric oracle is secondary and is scored against locally-refit raw-cloud normals,
  never against the primitive that proposed the view — otherwise the oracle row scores the space
  with the functional that built it.
- **Matched budget defined in four currencies**: executed views per episode; menu cardinality per
  decision (the LLM row must not compare an 8-item menu with a 36-cell grid); reachability —
  identical IK+collision filter for every arm with budgets matched **post-filter**; LLM calls
  (bounded by the matched-budget deliberation control, whose purpose here is stated as budget
  bounding). **The scan tax is charged to the treatment arm** and said out loud — the comparison
  is conservative by choice.
- **Label parity**: one serializer over all spaces in the LLM row, or the row is scoped
  explicitly as a whole-system comparison.
- **Budget regimes**: small-fixed and large-fixed at minimum, with regime dependence reported as
  a result rather than a single treatment-derived number carrying the claim.
- **Validity gates attached to the losing conditions, symmetrically**: a losing condition counts
  only when its gates passed (item set demonstrably inside the measured angle-sensitive band;
  snap error accounted; budgets matched in all four currencies; oracle named), and equally, a win
  under failed gates does not count.
- **The mechanism attribution arm is mandatory** (adversary §1.3): the intelligent viewsphere
  bundles direction + standoff + framing; without the parity arm, even a win teaches nothing
  about orthogonality.

**Disproof.** X12 kills the space claim in the oracle row; X13 kills LLM-selects. The
LLM-vs-random kill is pre-registered as chance-corrected **and efficiency-inclusive** (moves,
travel under the cap) with a stated MDE, and the fallback is phrased as *"geometry-selects ships
by default unless LLM-selects shows ≥MDE advantage"* — an engineering default on current
evidence, not a demonstrated null.

### P8 — demo

**Build.** Box with codes on one side; moved and rotated between runs; the loop finds the code
face and delivers the orthogonal evidence image.

**Resolved decisions.**
- **GT normal per grid cell, never the system's own fit**: PnP from the decoded code's own
  corner geometry (code corners + known size + intrinsics ⇒ face normal in camera frame, purely
  from RGB, independent of depth/fitting/registration) wherever decode succeeds; manual pose
  annotation otherwise (tens of trials — cheap). **This GT is promoted from demo-only to the
  benchmark's angular scoring mode wherever decode succeeds** — it is the only scoring path fully
  independent of the system's sensor and registration stack.
- **Scored on the grid, never a single fraction**: view-correct × answer-correct + abstention;
  raw counts and narratives, never stratum percentages at K×M ≈ tens.
- **Two answer criteria reported separately**: (a) the evidence *image* decodes offline
  (artifact quality — the third-party-verifiable construct Albert asked for); (b) the *system's*
  reported string matches (QA outcome). They can disagree; collapsing them silently kills one of
  the two claims.
- **Adversarial placements are in the protocol**, with the expected outcome pre-registered per
  stratum: code face away → grounded success expected (orbit and find it); near-table
  (orthogonal unreachable, oblique reachable) → best-reachable-with-disclosure as its own named
  outcome; face-down → **abstention is the scored success**. Abstention is scored in both
  directions (false abstention on inspectable placements too, or always-abstain games the strata).
- **Discovery and capture stratified**: episodes split by whether the code face is visible from
  the first scan. A hard-cap-before-discovery miss is a search failure, not a view-planning
  failure, and pooling makes the score's variance a function of the placement draw.
- **Generalization probe is a named deliverable, not a suggestion** (X18): the identical
  unmodified pipeline on an object the code never names — a differently-dimensioned box at
  minimum, ideally the cylinder with a label. Any required code change is an ADAM regression,
  caught mechanically. Without it, "no per-object recipes" is enforced by code review, which is
  not a protocol.
- **Cell conditions are first-class recorded state**: lighting and stream profile pinned for
  both legs (depth adequacy and RGB glare).
- **Pass bar and sizing stated before the first scored run**; thresholds frozen; protocol
  parameters versioned with the harness.

**Disproof.** X17 can fail on any stratum; X18 can catch the regression the framing doc's
non-goal forbids.

---

## 4. Canonical experiments

Old E-numbers collided across the three interpretations ("E1" was three different experiments).
This is the single numbering; the mapping column retires the old ones.

| ID | Experiment | Part | Wk | Disproof condition | Was |
|---|---|---|---|---|---|
| **X0** | **Error-budget memo** — measured/assumed fit error × computed snap distributions of all uniform controls × the 4 cm offset's angular contribution × X3's legibility curve, assembled on one page | P7 | 1 | If the distributions overlap for the strong control, **re-scope the claim to the deployed grid now** and demote Tammes to exploratory. If they overlap for the deployed grid too, the month's comparative claim is unprovable and the null-world deliverable becomes the plan | new (adversary §5.1) |
| **X1** | Two-hour fit shakedown on real fused PLYs (Open3D + CGAL/PCL + EMS) | P2 | 1 | Library reality check before any more reading | anton E1 |
| **X2** | Depth bench: single-view fill + plane-fit vs incidence/standoff, matte/glossy/textured, two-axis tilt, cell lighting, stream profile | P0 | 1 | Written no-go bar ⇒ camera swap while it is still free; τ_scan or 60°-provisional | acad E8 / ind E11 |
| **X3** | RGB legibility: decode/read accuracy vs incidence and GSD, incl. glossy under demo-cell lighting (glare rider) | P0 | 1 | Flat to 40–50° ⇒ the orthogonality requirement softens to a band and menu density relaxes; a dip at 0° on glossy ⇒ the menu needs a small off-normal option | anton E6 / ind E12 |
| **X4** | Partial-fit fidelity: derived-view angular error vs coverage, per fitter, per stratum (box face / half-cylinder-with-dropout / compound), 3 partial sources, independent GT, refit on/off, curvature guard two-sided | P2 | 2 | Not stochastically better than the control's snap error at loop-realistic coverage ⇒ the geometry menu cannot win by construction | anton E2 / acad E1 / ind E2 |
| **X5** | Maturity-gate calibration: score→derived-view-error mapping incl. cross-fitter disagreement and cross-round dispersion; FA/FR; calibrate/report on disjoint object sets | P2 | 2 | No signal separates good from bad fits ⇒ the fallback triggers blind and the abstention design needs a rethink | anton E4 / ind E4 |
| **X6** | Identity & churn: flip rate vs round-1 anchor, target-referent flip rate, wrong-attachment rate, decomposed churn, {matching, hysteresis} ablation | P3 | 2 | Flip/wrong-attachment above X7's knee ⇒ menus degrade to per-round anonymous | anton E3 / acad E2 / ind E3 |
| **X7** | Flip-tolerance + churn-presentation curve (Tier 1): injected flips 0/5/20/50/100 % × {diffs announced, silent} | P3/P5 | 3 | Sets X6's bar from downstream damage; the 100 % cell prices the anonymous-menu fallback and the identity layer's build cost | new (round-6 identity §3) |
| **X8** | Construction audit: achieved vs planned incidence, framing on real images, worst-case mantle angle vs K, per-region completeness, reachability survival, execution-failure rate of pre-verified items | P4 | 2 | "Quality by construction" falsified where it fails; also the rendered-evidence artifact | new (round-6 view-crafting §2.7) + ind E5 |
| **X9** | Loop behaviour: rounds/moves/travel to a usable menu vs one-shot dense survey at matched budget; per-round wall clock; stop quality vs oracle stop on replay; fallible-prior on/off | P1 | 1–3 | Loop needs as many moves as a survey ⇒ revert to survey-then-inspect; per-round budget blown ⇒ analytic-visibility obligation failed | anton E5 / ind E1 |
| **X10** | Menu-representation A/B on **teacher-forced frozen decision points** (500–1000 items): {frame} × {format} × order permutations × model tiers, with instruction-following pre-check | P5/P6 | 3 | Angle labels lose ⇒ ship words; all tie within MDE ⇒ ship the cheapest serializer | anton E7 / acad E3 / ind E8 |
| **X11** | Decision-structure and presentation arms: two-stage vs flat (stimuli must include cylinders/compounds), disclosure vs silent removal, diff announcement on/off | P5 | 3 | Flat matches two-stage ⇒ drop the turn; disclosure/diffs neutral or negative ⇒ drop them | anton E9 / acad E7 / ind E9 |
| **X12** | **Headline**: space family × 4 selector rows, four budget currencies, parity arm, outcome oracle, snap accounting; Tier-2 slice binding, matched-cloud-state variant (X12c) beside | P7 | 4 | Oracle row not better at matched budget vs the deployed grid ⇒ the geometry claim is dead regardless of interface | anton E8 / acad E4 / ind E6 |
| **X13** | Selector floor: random / scripted / LLM / outcome-oracle over identical menus, chance-corrected and efficiency-inclusive | P7 | 4 | LLM shows no ≥MDE advantage ⇒ ship geometry-selects / LLM-reads-and-answers | acad E5 / ind E7 |
| **X14** | Discovery recall (Tier 0): "is there a code/label on this face?" from oblique, partial, distant captures | P6/P8 | 3 | If recall from bad views is poor, discovery cost dominates and the hard cap fires before discovery — a loop-design finding, not a reader finding | new (completeness §3.2) |
| **X15** | Scorer noise floor: error of the episode-time view-correct label vs independent GT (PnP/fiducial) | P6 | 3 | Near-threshold episodes unscoreable ⇒ widen the reported band or drop the cell | new (completeness §3.3) |
| **X16** | Harness self-checks: replay determinism, travel-cost sanity vs real executions, Tier-0 item audit, Tier-1↔Tier-2 winner agreement | P6 | 3–4 | Tier 1 disagrees with Tier 2 on winners ⇒ Tier 1 is a development tool, not an arbiter, for that comparison | anton E11 / acad E6 |
| **X17** | Demo v2 acceptance: K×M placements incl. adversarial strata, grounded grid + two-way abstention, artifact-decode vs system-answer, counts + narratives, frozen thresholds, stated pass bar | P8 | 4 | Any stratum can fail; face-down abstention can fail in either direction | anton E12 / acad E9 / ind E10 |
| **X18** | Generalization probe: unmodified pipeline on a second object (different box; cylinder with label) — includes the one real-robot cylinder episode | P8 | 4 | Any code change required ⇒ ADAM regression, mechanically caught | new (round-6 demo-acceptance §2.1 + completeness §6) |

**Sequencing and gates.**

- **Week 1 — the cheap kills.** X1 (2 h) → X2, X3 (half a day each) → X0 (half a day, consumes
  X2/X3/X4-pilot) → P1 skeleton in the sandbox on replayed real captures + first dense capture.
  **Gate G1:** X0 decides the claim target (OQ-1) and whether any comparative claim is
  resolvable; X2 decides the camera on the depth axis.
- **Week 2 — the wall.** X4, X5, X6 (P2/P3), P4 factory + X8. **Gate G2:** the evidence bar is
  met at some reachable round (X4); a usable maturity signal exists (X5); flips are below X7's
  provisional knee (X6). A failure here re-scopes the month to the null-world deliverable
  *before* any harness effort is spent on the headline.
- **Week 3 — the interface and the instrument.** P5, P6 core, X7, X10, X11, X14, X15, X16.
  **Gate G3:** the frozen-decisions file is complete (thresholds, arms, budgets in all four
  currencies, oracle definition, losing conditions, null-world deliverable) — no comparative run
  before it is versioned with the harness.
- **Week 4 — the claim and the demo.** X12, X13, X17, X18, report.

**Triage of the ~90 round-6 fixes** (adversary §5.5). Claim-protecting, must land: X0; the
tolerance split τ_scan/τ_menu/τ_evidence; independent GT (X4 GT protocol, PnP promotion);
matched-budget currencies; outcome-oracle definition; snap accounting + Tier-2 binding; the
parity arm; the two-layer scoring in P4; the maturity gate in the factory contract; the
frozen-decisions file. Hygiene, run only with week-3/4 slack: X11's disclosure and diff arms,
X7's full five-cell sweep (three cells suffice for a knee), X10's hybrid serializer variant,
X16's Tier-1↔Tier-2 agreement beyond one comparison, the two-axis anisotropy sweep in X2.
Explicitly deferred with reason: injected-defect ranking-stability sweeps; simulation as a Tier-2
mode; coverage-metric leaderboards; edge/corner addresses.

---

## 5. Benchmark design

**Role.** Auxiliary in effort, arbiter in role — and only for the choices it can actually
referee. Component-based by construction: swap exactly one component per comparison (view-space
generator, selector, serializer, stop policy, scoring mode), hold everything else fixed.

**Tier 0 — single image, one decision.** Read/decode/locate/presence items sampled from
captures. Instruments X3 (legibility vs angle and GSD), X10 (frozen decision points), X14
(discovery recall). **Sampling must be angle-stratified or capture-source-balanced**: pools
drawn only from old uniform-viewsphere captures under-represent exactly the near-orthogonal view
class the treatment produces.

**Tier 1 — dense-capture replay (workhorse, scoped).** Per object × placement, drive the arm
once through a shared candidate superset — the uniform grids **and** the geometry-derived poses
(union capture), each derived pose captured with a **local bracket** (± one tolerance step in the
two view-sphere directions) so recipe drift within the band still serves near-exact frames.
Recipe hash version-stamped into every capture set; recapture policy stated in advance. A "move"
is a fetch; travel is computed in the kinematic sandbox from poses. Per-view schema: RGB, depth,
pose, intrinsics — nothing else.

Tier 1's arbiter role is **scoped in the spec**:

- **Referees**: same-menu comparisons — serializer, order, model tier, two-stage vs flat,
  disclosure, diffs, churn, selector floor. Snap error is common-mode and cancels; seeds and
  permutations are affordable only here.
- **Does not referee**: space-vs-space. The uniform arm's requests are on-grid (snap ≈ 0); the
  treatment's in-loop requests derive from partial-cloud fits and cannot coincide even with
  union captures. Union capture makes the unfairness *measurable*, not absent. The headline
  number therefore comes from Tier 2, or from Tier-1 episodes whose per-episode snap error falls
  inside the measured legibility flat band (the snap column's **decision rule** — without a
  threshold, a reported column absorbs any inconvenient value).
- **Named, not assumed**: replay refits on **served** depth, so snapping perturbs the whole
  trajectory, not just the evidence view. Replay of an incremental loop is a slightly different
  episode than the robot would run; the snap column carries it.

**Tier 2 — full runs** (real robot / replay / sim). Carries the headline and the demo numbers.
**Sim is deferred with reason**: simulating from our input needs a fitted mesh plus synthetic
materials (a raw cloud has no material channel), and glossy dropout is the case even PBR
simulation has not closed. Recorded as a plug-in slot, not a build item.

**Scoring — two layers, never blended.**

1. **Per-(object, surface region) evidence quality**: angular deviation from the region normal,
   scored against **system-independent GT where available** (PnP-from-code / fiducials) and
   otherwise against the raw cloud's **locally-refit** normals — never against the fitted
   primitive that proposed the view. Min over a per-(object, region) **symmetry-equivalence set**
   (undefined equivalence makes the grounded metric noise on cylinders and near-cubes, and
   excluding ambiguous items is unavailable to us because symmetric primitives are the common
   case). Worst case reported beside the mean — one grazing evidence image is the complaint that
   started the project, and a mean hides it.
2. **Per-(object, question) episodes**: the grounded 2×2 (view-correct × answer-correct) plus an
   abstention cell. Raw answer accuracy is never reported alone.

The rank correlation between the two layers is itself a reported result.

**QA items.** Mechanical-first (decode success, exact match with a **normalization rule written
into the item schema** — otherwise "mechanical" quietly becomes "someone eyeballs the near
misses"); free-form quarantined into a separately reported judge-scored slice with disclosed
judge identity. **Stratify by question type** — angle-sensitive reading vs angle-insensitive
(counts, presence/absence, colour — all exact-match scorable) — because mechanical-first
authoring otherwise selects the question class where the treatment wins by physics. Expect the
margin to concentrate in the first stratum; if it does, the headline ships scoped to
surface-reading QA, which is also demo v2's construct, so the scope costs nothing. The item set
is authored and **frozen before** the headline runs, with X3's curve disclosed as the
justification for its difficulty placement, and an easy stratum reported alongside so saturation
is visible rather than hidden. This *discloses* rather than removes the item-difficulty knob —
see OQ-6.

**Arms in every table.** Deployed 12×30°×3 grid (permanent) · budget-matched uniform ·
Tammes/HEALPix at matched budget · standoff/framing-parity arm · intelligent viewsphere ×
{random floor, scripted null, LLM, outcome oracle}. Budget matched in four currencies (§P7).

**Statistics.** Paired comparisons over matched cases (object, placement, question); ≥3 seeds on
anything with a model in it (API "seeds" are sampled repeats — say so); per-stratum reporting,
no pooled headline; medians beside means; WSR intervals on bounded scores; the honest sizing
statement up front (at n ≲ 100 episodes we resolve 15–20-point differences, not 5) — which is
exactly why X10's primary instrument is teacher-forced decision points, where n scales at the
cost of API calls only. No α rituals, no significance stars.

**The frozen-decisions file** (the minimal replacement for the pre-registration fork, **OQ-2**):
one versioned file, written before the first comparative run, containing only — τ_evidence and
its source; the arms and budgets in all four currencies; the oracle definition and its
information set; the item set hash and its difficulty justification; the holdout designation;
the losing conditions with their validity gates; the null-world deliverable. No statistical
ceremony. The adversary's objection stands and is recorded honestly: several of these values are
outputs of mid-month experiments, so the file freezes **sources and decision rules in week 1**
and **values at gate G3** — a partial mitigation, not a cure.

**Holdout, priced.** Objects are designated as holdout on day one and captured with the same
recorder version as the rest at the same time (designated-day-one, captured-with-the-set), so
the recipe-version confound does not eat them. State the total object count and the holdout
fraction in the frozen file; at our resolution, reserving objects is expensive and silently
skipping it is the failure mode.

**Sprawl control.** Must-have core: recorder, Tier-1 replay executor for same-menu A/Bs,
mechanical Tier-0 items, the grounded grid, floors/ceilings, paired design. Gated on reaching
weeks 3–4 with slack: bracket capture, Tier-2 validation slice, extra scoring modes,
view-count/prompt/path-statistics modes of the general framework the architect wants eventually.

---

## 6. Verifier issue register

Every round-6 and round-7 issue, with its disposition. **R** = resolved in this document (with
the mechanism); **O** = carried as an open question (§9); **A** = accepted as a residual risk,
stated and not mitigated.

### P0 — bench (`sensor-smoke-test`)

| # | Issue | Disposition |
|---|---|---|
| 1 | No pass bar for the go/no-go | **R** — bar written before running: single-view plane fit on the worst demo face at operating standoff, sub-τ, normal error within the menu's angular tolerance, else depth-axis no-go |
| 2 | τ may be unidentifiable (texture vs incidence dropout conflated) | **R** — protocol split: go/no-go on textureless demo faces, τ_scan from a textured control; no knee ⇒ τ_scan = 60° recorded as literature-default |
| 3 | Single-view vs fused unspecified | **R** — single-view is the loop-realistic measurement and is mandated; fused clouds hide single-view collapse |
| 4 | Camera authority overlaps the architect's reserved decision | **R** — split by axis: depth "no-go" is a week-1 veto; depth "go" does not clear the camera; RGB remains demo-v2's call |
| 5 | Scalar τ assumes isotropy the corpus warns against | **R** (hygiene tier) — two-axis tilt at 2–3 angles; if asymmetric, τ becomes a pair and the factory takes the tighter value |
| 6 | Uncontrolled conditions (ambient light, stream profile) | **R** — run in the demo cell, record profile/exposure; "go" is valid for that profile only |
| 7 | "Documented worst case" overstates provenance | **R** — phrased as a live hypothesis being tested, not an established D405 result |

### P1 — loop (`loop-skeleton`, `loop-scaffold`)

| # | Issue | Disposition |
|---|---|---|
| 1 | Two-phase MS@δ scan phase re-imports survey-first | **R** — bootstrap is a fixed 1–3 views; MS@δ demoted to fallback guard |
| 2 | Three stop machines across interpretations | **R** — answered ∧ stability ∧ hard cap; injectable; benchmark holds it fixed for space comparisons and measures agent stopping separately (X9) |
| 3 | "Every run feeds Tier 1" overstated | **R** — shared writer/schema; sparse episode logs feed Tier 0/archives; first dense capture is a week-1 deliverable |
| 4 | Sandbox E5 is asymmetric evidence; scan source unspecified | **R** — sandbox verdict is falsification-only; scan source is replayed real captures where they exist; re-run on real clouds after P2 |
| 5 | Scripted picker unspecified — the ADAM vector | **R** — generic menu rule, no object constants; doubles as P7's scripted row |
| 6 | Part declared itself unfalsifiable | **R** — X9 attached with the revert-to-survey disproof |
| 7 | "~1 s per round" is the solver, not the round | **R** — analytic-visibility obligation stated; X9 measures whole-round wall clock |
| 8 | Pan import has no operational definition | **R** — observed vs extrapolated support tags; extrapolated ⇒ exploration only, never evidence; cited as pattern (~1 pp, sim-only) |
| 9 | "Sidesteps the fused-offset hazard" overstated | **R** — live-frame normals *detect*, Tarbox margin *prevents*; both imported; Nguyen cited as pattern-level analogy |
| 10 | Set-cover category slip (greedy tax measures another construct) | **R** — reframed as question-subordinate pruning; question-relevant redundancy exempt |
| 11 | Registration method unnamed | **R** — named in code + per-round residual logged from day one |
| 12 | Vasquez-Gomez read via Peuzin-Jubert; Scott 2001 objective-switch dropped silently | **R** — cited "via"; the drop is recorded as a decision |
| 13 | Missing falsifiers: stop quality, fallible-prior value | **R** — both folded into X9 (stop quality vs oracle stop on replay; prior on/off as one table row) |

### P2 — fit (`partial-fit`, `partial-primitive-fitting`, fitting half of `partial-fit-identity`)

| # | Issue | Disposition |
|---|---|---|
| 1 | Kill threshold ~2× too lenient | **R** — paired distribution comparison vs the control's snap error (median + p90) |
| 2 | Exploration vs evidence regimes conflated | **R** — two bars; verdict read at the coverage the loop reaches; X4 consumes X9's trajectory |
| 3 | Partial generation unstated | **R** — three labelled sources, defect triad injected pre-fusion, never plain masking |
| 4 | GT unaddressed / circular / cardboard floor | **R** — per-surface-class GT protocol (gravity+table, fiducials, touch-off, CAD); cardboard restricted to the depth check |
| 5 | Gate features are residual-family and pre-doomed | **R** — cross-fitter disagreement and cross-round dispersion added as first-class candidates |
| 6 | Gate calibrated and tested on the same distribution | **R** — object-disjoint calibrate/report split; recalibrate from recorder logs once real episodes exist |
| 7 | Wider-band refit precedent misapplied (position/scale ≠ angular) | **R** — with/without arms in X4; caveat carried |
| 8 | EMS independence overstated, semantics undefined | **R** — sanity vote; disagreement ⇒ abstain; never a bias detector |
| 9 | Curvature guard undesigned and untested | **R** — two-sided test inside X4 (cylinder suppression + bowed-face false suppression) |
| 10 | Depth-adequacy check missing incidence dimension and a threshold | **R** — folded into X2 with a written bar |
| 11 | "Exists nowhere in the literature" overstates provenance | **R** — all three absence claims re-phrased with corpus scope |
| 12 | Runtime-computable features (plane coverage is circular at runtime) | **R** — sweep axis ≠ deployed feature set; stated |
| 13 | Effort sizing (mini research program inside an auxiliary month) | **R** — calibrate only in the 15–40 % regime on two shapes |
| 14 | Rendered evidence not mandated | **R** — per-round overlays (cloud + primitives + derived-view arrows + GT normal) are required artifacts |
| 15 | Camera swap invalidates the curves | **R** — a swap re-runs X4/X5; stated in the frozen file |
| 16 | Fallback granularity fork (per-region vs whole-menu vs action-space) | **O — OQ-4** (recommendation: per-region, with gate-trigger rate reported per arm so hybrid contamination is visible) |
| 17 | Cylinder half has no demo usage story; cut line unnamed | **O — OQ-8** (recommendation: one real cylinder episode is a named deliverable in X18, first to be cut, and if cut the claim ships scoped to plane-dominated objects) |
| 18 | Disproof threshold depends on an experiment scheduled later | **R** — X3 pulled to week 1 |

### P3 — identity (`identity-persistence`, identity half of `partial-fit-identity`)

| # | Issue | Disposition |
|---|---|---|
| 1 | No acceptance bar; harm link asserted, not measured | **R** — X7 flip-tolerance curve sets the bar; 100 % cell prices the anonymous fallback |
| 2 | Requirement broader than the load-bearing need | **R** — target-referent flip rate reported above global; nearest-centroid re-anchor as a standing recovery step |
| 3 | Content addresses non-stationary (area/centroid grow by design); matcher unspecified | **R** — parameter-space matcher, dummy rows for unbalanced assignment, thresholds stated and swept |
| 4 | RIoU assumes stable point membership | **R** — parameter-space matching named as the build |
| 5 | Flip GT is the matcher's own output | **R** — independent correspondence GT from dense capture / CAD |
| 6 | Metric gameable under hysteresis (stable-and-wrong) | **R** — wrong-attachment rate reported beside flip rate |
| 7 | Drift across many small refits | **R** — anchored to round 1, not round-to-round |
| 8 | Cylinder azimuth identity ill-posed | **R** — scene-fixed azimuth convention |
| 9 | Determinism oversold as an identity mechanism | **R** — restated; ablation is matcher/hysteresis on/off |
| 10 | Frame anchor jitter (fitted table plane as coordinate system) | **R** — robot base frame; the table plane becomes one more primitive |
| 11 | Churn undecomposed | **R** — four-way decomposition; churn also reported conditioned on gate state |
| 12 | Scope fork (within-session vs cross-placement) | **R** — within-session promised and gated; cross-placement reported, unpromised |
| 13 | Novelty overclaim; plane-SLAM data association unread | **R** — half-day prior-art pass before writing the matcher; claim re-scoped to the protocol |
| 14 | Hysteresis concentrates churn at the gate crossing | **R** — churn reported conditioned on gate state |
| 15 | Expected outcome is "confirm" | **A** — accepted; the unpublished number is still worth having, and X7 is the half that can surprise |

### P4 — factory (`intelligent-viewsphere`, `menu-crafting`, `view-crafting`)

| # | Issue | Disposition |
|---|---|---|
| 1 | Gates + cover cannot express "most orthogonal" | **R** — two-layer scoring; dedup selects among region representatives, never drops a region |
| 2 | One τ doing two (three) jobs | **R** — τ_scan / τ_menu / τ_evidence with named owners; τ_evidence is **OQ-3** |
| 3 | Fit-maturity gate outside the factory's contract | **R** — pulled in as a named gate with defined input/output and a disclosed abstain state |
| 4 | Empty standoff band undeclared | **R** — clamp + FOV tiling + deterministic ROI close-up as a declared branch |
| 5 | Set-cover overstated / 6–12 presented as enforced | **R** — demoted to emergent-size pruning; leaf size is the real constraint |
| 6 | Flagged-never-deleted adopted on argument alone | **R** — X11 disclosure arm (hygiene tier); NBV-Net citation downgraded to analogy |
| 7 | Pose verification ≠ executable action | **R** — witness configuration (roll + IK branch + motion plan) cached and bound to execution; execution-failure rate logged |
| 8 | Free roll breaks framing and the invalid-band promise | **R** — roll constrained for elongated faces in the witness; band location recorded |
| 9 | Collision world on partial clouds unstated; occlusion missing from one spec | **R** — inflated union stated; analytic occlusion in the stack; hole-ambiguity named |
| 10 | Orbit spacing sourced from a refused number | **R** — τ_menu from X3, pulled to week 1; 60° quarantined and never shipped in a reported number |
| 11 | Defer-to-next-best is a scanning mechanism in an evidence job | **R** — reachable-but-degraded is disclosed with its quality band or routed to abstention |
| 12 | Standoff has no GSD ceiling — direct demo failure mode | **R** — clamp + ROI close-up action |
| 13 | Menu diffs invented with no falsification path | **R** — X11 diff arm (hygiene tier) |
| 14 | Two-stage degenerates on plane-only objects | **R** — X11 stimuli must include cylinders/compounds; the box-case coincidence with the geometry-selects architecture is stated, not hidden |
| 15 | Fallback-tier presentation undefined in the two-stage menu | **R** — fallback surfaces as one stage-1 item ("overview — shape not yet trusted") with sphere views as its leaf |
| 16 | Cold start unstated | **R** — round 0 ⇒ uniform bootstrap tier |
| 17 | Gate generalization to novel objects/materials | **R** — gate thresholds frozen before holdout contact; mapping reported per primitive type |
| 18 | No experiment falsifies the construction itself | **R** — X8 construction audit |
| 19 | Yu & Gupta rung mis-citation; long cylinders need multiple rings; "seedable" vacuous | **R** — cite Spensieri + ranked retry; axial rings from the FOV footprint; determinism claim scoped to P2's detectors |

### P5 — interface (`menu-interface`, `menu-representation-ab`)

| # | Issue | Disposition |
|---|---|---|
| 1 | Disclosure borrows NBV-Net's number; no test | **R** — analogy downgrade + X11 arm |
| 2 | E3 power: catastrophe detector, not ranker | **R** — teacher-forced decision points (500–1000), MDE stated |
| 3 | MOKA transfer confound (visual clutter vs text menu) | **R** — stated in the write-up; two-stage default-on pending X11; a null is the expected result of the confound |
| 4 | Value of stable naming assumed | **R** — X7 churn arm prices it |
| 5 | Instruction-following pre-check dropped by all three | **R** — restored as a gate on the composite prompt per model tier |
| 6 | Stage-1 menu unbounded | **R** — group by primitive above ~8 surfaces |
| 7 | Anti-repeat blocks legitimate post-refit recapture | **R** — carve-out tied to the diff announcement |
| 8 | Closed-loop episodes cannot be paired across serializers | **R** — teacher-forced probes are primary; closed-loop is a secondary check |
| 9 | Frame and format move together | **R** — factorized; frame-implicit control added |
| 10 | Semantic descriptors leak task information | **R** — information-matched variants; content-derived naming quarantined |
| 11 | A/B cannot touch the engineer's own prior | **R** — frame-implicit and churn arms added, which is where the literature predicts a large effect |
| 12 | Two-stage × serializer interaction unspecified | **R** — X11 runs under the top-2 serializers |
| 13 | Order condition / "seeds" on closed APIs | **R** — canonical, reversed, two shuffles; adversarial-extremes dropped; API sampling labelled as such |
| 14 | No tie-break rule | **R** — accuracy ≻ order-stability ≻ cheapest |
| 15 | "30 pp on hard cases" mis-attributed (embedding scorer, not LLM chooser) | **R** — corrected wherever quoted |

### P6 — harness

| # | Issue | Disposition |
|---|---|---|
| 1 | Union capture cannot serve the treatment exactly | **R** — Tier-1 arbiter role scoped in the spec; headline bound to Tier 2 or snap-tolerance-passing episodes |
| 2 | Snapped depth perturbs the whole trajectory | **R** — named; replay refits on served depth; the snap column carries it |
| 3 | Dataset coupled to the capture-day recipe | **R** — bracket capture + recipe-hash stamping + stated recapture policy |
| 4 | Snap column has no decision rule | **R** — tolerance = X3's legibility flat band; exceed-fraction reported per arm |
| 5 | Mechanical-first authoring selects on the dependent variable | **R** — question-type stratification + deliberate angle-insensitive mechanical items; headline scoped to surface-reading QA if the margin concentrates there |
| 6 | OCR exact-match brittleness | **R** — normalization rule in the item schema |
| 7 | Circularity guard and symmetry equivalence missing from the harness spec | **R** — both moved into the scoring-code requirements |
| 8 | Holdout unpriced | **R** — designated day one, captured with the set, size stated in the frozen file |
| 9 | Tier-1↔Tier-2 rank correlation underpowered | **R** — downgraded to per-comparison winner agreement |
| 10 | "Defect conditions" clause inconsistent with the scope cut | **R** — scoped to naturally occurring defect strata; injected sweeps deferred with reason |
| 11 | Benchmark sprawl under the auxiliary label | **R** — must-have core vs gated extras (§5) |
| 12 | Tier-0 sampling bias toward old uniform captures | **R** — angle-stratified / source-balanced pools |
| 13 | Sim mode unscoped | **R** — deferred with reason, recorded as a plug-in slot |
| 14 | Episode-time scorer has no noise floor | **R** — X15 |

### P7 — comparison

| # | Issue | Disposition |
|---|---|---|
| 1 | QA-item saturation ⇒ false kill; difficulty tuning ⇒ false win | **R (partial)** — X3 first, band disclosure, easy stratum, frozen item set; residual risk **A/OQ-6** |
| 2 | Arm-asymmetric snapping | **R** — Tier-2 binding + snap columns + winner-agreement gate |
| 3 | "Matched budget" is 3–4 currencies | **R** — all four defined; scan tax charged to the treatment and stated |
| 4 | Oracle under-specified / re-imports circularity | **R** — outcome oracle carries the losing condition; angular oracle secondary and independently scored |
| 5 | "Blind no-geometry baseline" unoperationalized | **R** — instantiated as content-free labels over the same menu (the only variant testing something the other arms don't) |
| 6 | LLM-vs-random underpowered; tie ≠ loss | **R** — chance-corrected + efficiency-inclusive + stated MDE; fallback phrased as default-on-current-evidence |
| 7 | Validity gates not attached to losing conditions | **R** — attached symmetrically for wins and losses |
| 8 | Uniform column is a family, not a cell | **R** — four space arms + the parity arm |
| 9 | Loop breaks the clean factorial | **R** — claim declared system-level; matched-cloud-state variant X12c reported beside |
| 10 | Label parity across arms | **R** — one serializer over all spaces in the LLM row |
| 11 | Reachability treatment of uniform arms | **R** — identical filter, budgets matched post-filter |
| 12 | Budget derived from the treatment's own machinery selects a favourable regime | **R** — small-fixed and large-fixed regimes reported; regime dependence is a result |
| 13 | Losing condition has no numbers to be written with | **R** — sources frozen week 1, values at G3; residual honesty note kept |
| 14 | Error budget never assembled; margin may be inside the noise | **R** — X0 as a week-1 gate; claim re-scoped to the deployed grid (**OQ-1**) |
| 15 | Mechanism unattributed (direction vs standoff vs framing) | **R** — standoff/framing-parity arm |
| 16 | Angular scorer shares the system's pipeline | **R** — PnP/fiducial GT promoted from demo-only to the benchmark's angular mode where decode succeeds |
| 17 | Pre-registration cannot bear its assigned load | **A / OQ-2** — minimal frozen-decisions file, sources frozen early, values at G3; the residual is stated, not cured |

### P8 — demo

| # | Issue | Disposition |
|---|---|---|
| 1 | Reference normal circular | **R** — PnP-from-code, manual annotation otherwise, never the system's fit |
| 2 | Tolerance band has three owners; two drift the construct | **R/O** — single owner; τ_evidence is an architect-set requirement (**OQ-3**), diagnostics may tighten but not loosen it |
| 3 | Placement strata lack expected outcomes | **R** — pre-registered per stratum (away ⇒ success; near-table ⇒ best-reachable-with-disclosure; face-down ⇒ abstention) |
| 4 | Abstention scored one-way | **R** — both directions |
| 5 | "Mechanically decoded" conflates artifact and system | **R** — both reported |
| 6 | Cross-episode identity inflation | **R** — within-episode suffices; the stronger requirement dropped |
| 7 | Stratum rates at tiny N | **R** — counts + narratives, never rates |
| 8 | ADAM guard unsupported by K×M on one box | **R** — X18 generalization probe as a named deliverable |
| 9 | Single-fraction score | **R** — grounded grid |
| 10 | No unwinnable placements ⇒ confirmation instrument | **R** — adversarial strata mandatory |
| 11 | "Readable"/"orthogonal" not operationalized; thresholds float | **R** — mechanical scorer named; thresholds frozen before scored runs |
| 12 | No acceptance bar, no sizing | **R** — bar and K×M stated before running; else the artifact is a robustness report, not an acceptance test |
| 13 | Discovery vs capture conflated | **R** — stratified; X14 measures discovery recall |
| 14 | Demo v2 has no paper trail in the binding record | **R (proposed)** — written into the framing doc by `design_update_proposal.md` §Context/§Goals |
| 15 | Sensor smoke test is an unstated prerequisite | **R** — X2 gates the acceptance runs |
| 16 | N=1 overfitting to the single demo box | **A** — mitigated by X18 + holdout; residual stated |

### Round 7 — whole-proposal issues

| # | Issue | Disposition |
|---|---|---|
| A1 | No error budget; margin vs the strong control may be ~2 % GSD | **R** — X0, week 1, gating; claim re-scoped (**OQ-1**) |
| A2 | Valid instrument underpowered, powered instrument invalid | **R (partial)** — Tier-1 role scoped to same-menu A/Bs; headline on Tier 2 + the outcome oracle on union captures; the deployed-grid target is where the effect is large enough to see. Residual: the strong-control comparison stays exploratory |
| A3 | Mechanism unattributed | **R** — parity arm |
| A4 | Item difficulty is a free knob | **A / OQ-6** — disclosed, stratified, frozen, holdout — but the N=1 team still authors the items |
| A5 | Scorer shares the pipeline | **R** — PnP/fiducial GT promoted |
| A6 | Pre-registration unfillable | **A / OQ-2** — minimal file, staged freezing |
| A7 | Six unresolved persona forks | **R/O** — stop rule (R), fallback granularity (**OQ-4**), tolerance-band owner (**OQ-3**), identity scope (R), ship architecture (R — four rows + MDE rule), strong-control necessity (**OQ-1**) |
| A8 | ~90 fixes, no triage, no owner | **R** — §4 triage into claim-protecting / hygiene / deferred |
| A9 | No null-world deliverable | **R** — §1 |
| A10 | No single claim sentence | **R** — §1, pending OQ-1 |
| C1 | Three colliding experiment numberings | **R** — §4 table |
| C2 | Contract stale; demo v2 absent from the binding record | **R (proposed)** — `design_update_proposal.md` |
| C3 | Reading debt larger than recorded | **R** — §8 triage |
| C4 | Glare-at-orthogonal moved, not closed | **R** — X3 glossy + cell-lighting rider, with the off-normal menu option as the response if a dip appears |
| C5 | Discovery capability has no experiment | **R** — X14 |
| C6 | Scoring instrument has no error bar | **R** — X15 |
| C7 | Menu diffs, stop quality, per-round wall clock, curvature guard unexperimented | **R** — X11, X9, X9, X4 respectively |
| C8 | Absence claims exceed the read corpus; measure-zero argument unstated formally | **R** — corpus-scoped phrasing; the two-line statement is in §7 |
| C9 | Tier-0 sampling bias; sim unscoped | **R** — §5 |
| C10 | Cylinder never meets the real robot | **O — OQ-8** |
| C11 | Demo fixes exist as commentary only | **R** — folded into P8's protocol |

---

## 7. Claim hygiene (carried forward, unchanged in spirit)

- **The measure-zero argument, stated precisely so it cannot be quoted past its validity:** for a
  convex polyhedron in generic orientation relative to a fixed sphere partition, the set of view
  directions exactly aligned with a face normal has measure zero in the direction sphere; a
  partition into finitely many cells therefore contains a face-normal direction only by
  coincidence, and refining the partition shrinks the *worst-case* deviation without making the
  exact direction a cell member. It is an argument about construction order (surface → view vs
  view-space → partition), not a claim that fine grids are useless.
- **Not built on:** Wang T-Mech 2024's numbers (read third-hand); any incidence-knee value from
  the 45–75° band; SPAZER's +3.6 % and TagaVLM's +0.94 % (unverified, and the two most direct
  threats to the blind design — verify before leaning either way); human best-view labels as GT
  for our construct; residual-only fit confidence as the fallback gate; "just run CGAL Efficient
  RANSAC" as the fitting plan; any assumption that the LLM can infer view relations, frames or
  topology.
- **Absence claims** ship as "absent from the corpus we read (20/39 in the fitting branch; 20/58
  in the benchmarking branch; sections E/F unread; Wang T-Mech not read primary)".
- **Never retro-claim defect visibility.** This month's system supports "orthogonal views
  maximize evidence legibility for surface QA". It does not support "orthogonal views maximize
  defect detectability" — deferred, not answered.

## 8. Reading debt, triaged

- **Read before claiming** (blocks novelty statements only): Wang et al. T-Mech 2024 primary
  text; SPAZER + TagaVLM.
- **Read before building** (blocks a design decision): Branch 3 map section F (failure
  detection / introspection — the literature behind P2's maturity gate, which currently has zero
  sources); GS-NBV (the only candidate cylinder-orbit precedent, while the orbit is half the
  factory); robust extrinsic symmetry estimation (the only partial-cylinder axis-completion
  candidate); Katz HPR primary + Open3D `hidden_point_removal` parameter semantics (the mesh-free
  visibility hazard sitting in P4's stack); plane-based-SLAM data association (P3's matcher,
  half a day).
- **Read if time**: find-superquadric; M3C2-EP; solids-of-revolution aspect graph; Branch 2
  group H (`cpp-measurement-uncertainty`, `quoniam`, Karaszewski 2016); Branch 1 tier 7
  (image-space quality) top 2–3; Vasquez-Gomez expected-utility-under-pose-uncertainty.
- **Retired with reason**: defect-visibility group (construct out of scope); dropout-simulation
  paper (only if X4's synthetic corruption needs upgrading); Branch 3 section E beyond what X2
  measures ourselves.

## 9. Open questions for the architect

Numbered; each states the recommendation and what changes with each answer. These are the
decisions the integrator cannot make without overriding architect-level scope.

**OQ-1 — What must the claim beat?** Recommendation: the deployed 12×30°×3 grid, binding;
Tammes/HEALPix at matched budget reported as exploratory. Consequence of choosing Tammes as
binding: on the assembled error budget the expected margin is single-digit degrees and the month
most likely reports a null that is an artifact of instrument resolution, not evidence about
spaces. Consequence of the recommendation: the win is real but is the one the academic persona
called "the claim we are embarrassed to beat" — which is also exactly the system being obsoleted.

**OQ-2 — Frozen decisions: minimal file, or nothing?** The architect dropped "pre-registration
theatre"; the round-6 verifiers lean on pre-registered gates ~35 times and the losing conditions
formally depend on them. Recommendation: one versioned file with sources and decision rules
frozen in week 1 and values frozen at gate G3, no α rituals. Consequence of "nothing": the
losing conditions become unenforceable and every kill switch is negotiable at week 4.

**OQ-3 — Who owns τ_evidence (the "orthogonal enough" success band)?** Recommendation: the
architect sets it as a requirement (≤15° off-normal proposed); X3's decode curve may tighten it,
never loosen it. Consequence of sourcing it from the decode curve instead: view-correct collapses
into answer-correct, the wrong-view/right-answer cell empties by definition, and a 45° oblique
image that decodes passes the demo — satisfying the grid while violating the requirement Albert
flagged twice.

**OQ-4 — Fallback granularity when the maturity gate trips.** Per-region (the failing surface
falls back to uniform views), whole-menu (the whole round falls back), or whole-episode action
space. Recommendation: per-region, with gate-trigger rate reported per arm. Consequence:
per-region produces a hybrid menu that partially contains the control, so the treatment arm
converges toward the baseline exactly when the gate trips often — visible in the tables, but it
dilutes the claim; whole-menu keeps the arms clean and wastes good fits.

**OQ-5 — Does the LLM keep the selection role if X13 ties?** Recommendation: ship
geometry-selects / LLM-reads-and-answers as the deployable default, keep LLM-selects as the
design under test, and phrase the outcome as "no ≥MDE advantage measured", not "the LLM adds
nothing". Consequence of a stricter reading: the month converts an absence of evidence into
evidence of absence under our own name, on an underpowered cell.

**OQ-6 — Item authoring under N=1.** We hold both ends of the effect size: we author the
questions and set the difficulty band. Mitigations in place: disclosure of the legibility curve,
type stratification, easy stratum, frozen item set, holdout. Recommendation: accept the residual
and state it in the report. Alternative: source a slice of items from someone outside the loop
(Albert), which costs a conversation and buys real independence for the headline table.

**OQ-7 — Angle labels vs directional words.** On the record as the architect's hypothesis;
X10 decides; expected outcome is a tie within MDE, in which case the cheapest serializer ships.
Confirm the tie rule now so the A/B does not become a fishing expedition later.

**OQ-8 — Cylinder scope.** Recommendation: one real-robot cylinder episode inside X18 as a named
deliverable; it is the first thing cut if week 4 compresses, and if cut, the claim ships scoped
to plane-dominated objects. Consequence of cutting silently: "geometry-conditioned spaces" is
claimed on box faces alone, which is a narrower claim than the framing doc's target behaviour.

**OQ-9 — Edges, corners, junctions** stay out of the action space for v1 (recorded decision, no
literature either way, revisitable). Confirm.

**OQ-10 — Camera.** X2 can veto on the depth axis in week 1; the RGB-evidence axis remains the
architect's demo-v2 call. Confirm the split so a week-1 "go" is not read as settling the camera.
