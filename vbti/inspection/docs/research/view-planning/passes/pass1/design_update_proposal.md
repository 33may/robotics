# Proposed updates to the framing doc — PROPOSAL ONLY

**Target file:** `docs/research/geometry-aware-view-planning.md` — **not edited**.
Everything below is proposed text for the architect to accept, amend or reject.
**Why now:** the round-4 feedback overrides the framing doc in four places, and the round-7
completeness pass found that a later reader citing "the contract" would cite dead premises
(a falsified sensor story, the survey-first given, the defect-flavoured inspection framing) and
would find demo v2 nowhere in the binding record at all.
**Companion:** `methodology_v1.md` (the build plan these edits describe).

**How to read:** each item gives the current text, why it is stale, the proposed replacement
verbatim, and where the replacement comes from. Items marked **[decision]** encode one of the
open questions in `methodology_v1.md` §9 and should not be applied before that ruling.

---

## 1. `## Context` — three corrections

**Current (lines 5–18):** "a VLM-orchestrated robot (UR5e, eye-in-hand camera)… Albert flagged
that evidence images should be the most orthogonal view of the inspected surface…"

**Stale because:** the doc never says what "inspection" means, and the round-4 feedback
narrowed it to QA about objects, not defect detection. The demo target is absent. Nothing
records that the camera is now a flexible choice.

**Proposed replacement for the last paragraph of Context, and a new paragraph after it:**

> Albert (VBTI, 2026-08-27) flagged — twice already in communication — that evidence images
> should be the most orthogonal view of the inspected surface, which the geometry-agnostic
> sphere cannot guarantee. He sketched the target behaviour: a box should be scanned face-by-face
> orthogonally; a cylinder orbited with the camera normal to the surface — the robot recognizing
> the shape and planning coverage from it.
>
> **Inspection here means answering questions about the object (QA)** — read the codes, name the
> label, report what is on this face — not defect detection. This scoping is deliberate: the
> orthogonality requirement is supported for QA by the one angle effect the literature actually
> quantifies (resolution/foreshortening, measured 38,500 → 4,900 points from straight-on to
> steep tilt), and is *unsupported* for defect contrast, where industrial practice for shallow
> flaws uses grazing light. The defect-visibility question is **deferred, not answered**.
>
> **Demo v2 is the milestone:** a box with codes on one side. After inspection, the evidence
> image is the orthogonal view of that side, and the behaviour is robust to the box being moved
> and rotated between runs.

**Provenance:** round-4 feedback (`benchmarking-view-planning`, `view-quality-scoring` sections);
round-7 completeness §1.3 (the demo has no paper trail); round-6 `demo-v2-acceptance` §1.

---

## 2. `## Problem` — keep the argument, state it precisely, add the loop

**Current (lines 22–42):** the uniform sphere is geometry-isolated; the orthogonal direction
falls between bins; navigability for a blind agent is the second half.

**Stale because:** the structural argument is the single most-cited justification in the whole
pass and is currently phrased loosely enough to be quoted past its validity; and the problem
statement still reads as if a complete object model were available when planning.

**Proposed addition after the second paragraph:**

> Stated precisely: for a convex polyhedron in generic orientation relative to a fixed sphere
> partition, the set of view directions exactly aligned with a face normal has measure zero in
> the direction sphere. A partition into finitely many cells therefore contains a face-normal
> direction only by coincidence, and refining the partition shrinks the worst-case deviation
> without ever making the exact direction a member. The fix is a change of construction order —
> surface → view, rather than view-space → partition — not a finer grid.

**Proposed addition to the navigability paragraph:**

> The resolution of the navigability half is the **intelligent viewsphere**: geometry computes
> the good views internally (orthogonal, reachable, well-framed) and the agent receives a small
> finished menu of them. The agent never sees primitives, coordinates or parameter knobs. This
> is the published consensus pattern — geometry proposes, the model disposes, and the disposal
> is always a label — stated independently by six deployed systems; selecting a label rather
> than emitting a value is worth +60.7 points in the one within-paper ablation that measures it.

**Provenance:** round-4 feedback (`structuring-the-viewpoint-space` correction); round-4 branch 6
synthesis; round-7 completeness §4 (make the measure-zero claim formal).

---

## 3. `## Goals` — restate against the loop and the menu

**Current (lines 46–56):** action space conditioned on geometry, simple enough for the VLM;
auxiliary view-quality algorithm producing best views "given the object's PLY geometry".

**Stale because:** "given the object's PLY geometry" presumes a survey-first model, which the
feedback overrode; and the view-quality goal is now a two-layer judge inside the factory, not a
standalone optimizer.

**Proposed replacement:**

> - **Action space:** an action space conditioned on the inspected object's geometry as it is
>   known *at that round* — views oriented by the object's surfaces (orthogonal where a surface
>   exists), covering what is worth seeing on that shape — rather than by uniform bins around it.
>   The space is presented to the decider as a small finished menu; the geometry stays internal.
> - **Incremental loop:** the planner runs scan → fit primitives to the partial cloud → derive
>   views → present the menu → pick → execute → refit → replan. There is no complete
>   reconstruction up front, and every part of the design must work on a partial, growing cloud.
> - **View judgement (internal, two layers):** a boolean admissibility gate (incidence, standoff
>   band, occlusion, reachability) that decides what may enter the menu, **plus** a continuous
>   orthogonality/framing rank that decides which candidate represents a surface region. The gate
>   alone makes orthogonality a floor rather than an objective — which is not what was asked for.
> - **Demo v2 (acceptance):** the box-with-codes behaviour above, scored on a grounded grid
>   (view-correct × answer-correct + abstention), robust to placement and rotation.

**Provenance:** round-4 feedback (loop correction, intelligent-viewsphere correction); round-6
`intelligent-viewsphere` issue 1 (two-layer scoring is the branch's own prescription).

---

## 4. `## Non-Goals` — add the four scope cuts

**Current (lines 59–64):** full coverage of complex geometries; hand-programmed recipes
(partially).

**Proposed additions (keep both existing items verbatim):**

> - **Defect detection.** Inspection in this project is QA. The defect-visibility question is
>   recorded as open, not closed, and no result here may be retro-claimed for it.
> - **Deep sensor-noise modeling.** No σ(range, θ) campaign. Depth serves the loop's own
>   perception; RGB serves the evidence. The only sensor work in scope is a half-day bench that
>   produces two numbers the design otherwise guesses: the depth admissibility angle for fitting,
>   and the RGB legibility/GSD curve for evidence.
> - **VLA and RL.** Out for this month, in every part, including as benchmark modes.
> - **Learned selectors and learned stopping.** The selector is a prompted model over a
>   deterministic menu; stopping is answer-driven plus a hard cap.

**Provenance:** round-4 feedback; orchestrator scope statement.

---

## 5. `## Research Questions` — rewrite for the loop, add two

**Current (lines 70–84):** RQ1 judging a view (from PLY or image), RQ2 selecting best views,
RQ3 composing the action space, RQ4 presenting to the VLM.

**Stale because:** RQ1/RQ2 presume the surveyed geometry; the two hardest questions the pass
surfaced (identity across refits; what the LLM actually contributes) are not asked at all.

**Proposed replacement:**

> - **RQ1 — Fitting under partiality:** how accurately can primitives be fitted to a partial,
>   growing, defective cloud, and what is the angular error of the orthogonal viewpoint derived
>   from such a fit as a function of surface coverage? This number decides everything downstream
>   and exists nowhere in the corpus we read.
> - **RQ2 — Deriving views:** how do we derive, per primitive class, the set of orthogonal,
>   well-framed, reachable views — including the standoff band (framing, minimum range, and the
>   pixels-per-feature floor) and the cylinder orbit spacing?
> - **RQ3 — Trusting the fit:** what computable signal tells us a fit is good enough to derive
>   evidence views from, and what does the system do when it is not (abstain, fall back, disclose)?
> - **RQ4 — Composing the action space:** how are views addressed, related and kept consistent
>   across object shapes and across rounds of the same session?
> - **RQ5 — Identity across refits:** can a view keep the same name after the geometry is refit,
>   and what does a name change cost the decider? Both halves are unmeasured in the field.
> - **RQ6 — Presenting to the decider:** what does the finished menu look like — labels, order,
>   decision structure, disclosure of what is unavailable — and does a prompted model over that
>   menu do better than random or than geometry auto-selecting? No published result shows a
>   prompted model beating random view selection with variance; this is the design's open burden
>   of proof, not an assumption.

**Provenance:** round-4 feedback; round-6 `partial-fit`, `identity-persistence`,
`headline-comparison`; round-4 branch 6 Q15.

---

## 6. `## Constraints & Givens` — three corrections and two additions

**Current (lines 88–100):** survey cloud with a ~4 cm fused-vs-single-view offset; reachability;
VLMs cannot emit metric poses; "Platform is fixed — UR5e, RG2 + D405 eye-in-hand".

**Stale because:** the first bullet encodes the survey-first premise; the platform bullet says
"fixed" while the feedback made the camera flexible; and the doc's implicit active-stereo
assumption was falsified by the datasheet read.

**Proposed replacement:**

> - **The geometry is a partial cloud that grows during the episode** — not a model, and not a
>   completed survey. Every round fits primitives to what has been seen so far. Real defects
>   apply: pose error of a few mm, per-view depth dropout on glossy/dark and textureless
>   surfaces, and a measured ~4 cm offset between the fused cloud and what a single view's own
>   depth reports. That offset is a registration-pipeline defect, not sensor extrinsics — the
>   D405's depth and colour share the left imager.
> - **Not every desired view is reachable** — views must survive IK and collision checking, and
>   verification must bind a specific configuration (roll and IK branch) through to execution: an
>   IK solution existing does not mean the executed branch is collision-free.
> - **VLMs cannot emit metric poses** — working systems have the model pick from pre-enumerated
>   views; selection stays discrete. (Now cited rather than asserted.)
> - **The camera is a passive stereo D405 — there is no projector.** Its documented worst case is
>   a textureless matte surface (correspondence loss independent of incidence angle), and the demo
>   object is cardboard. Two datasheet numbers bind the standoff band: minimum range is 100 mm at
>   1280×720 (not the marketed 70 mm), and the stereo invalid-depth band eats about 10 % of image
>   width at 10 cm.
> - **The camera choice is flexible.** If depth adequacy fails the week-1 bench, or if RGB
>   evidence quality fails demo v2, we request a different camera. The arm, gripper and kinematic
>   sandbox are fixed.

**Provenance:** round-4 feedback; round-4 branch 1 datasheet read; round-6 `view-crafting` §2.2,
`sensor-smoke-test`, `loop-scaffold` §3.4.

---

## 7. `## Candidate Directions` — resolve the two-branch question

**Current (lines 104–120):** analytical vs optimization track, "shared hunch that this task is
most probably optimization"; benchmark-driven methodology.

**Stale because:** the reading settled it. The field constructs and selects; it does not climb.
Direct view-parameter optimization lost to a noise baseline in the one paper that tried it, and
quality-by-construction (place the candidate on the surface normal) is thirty years old and
shipped.

**Proposed replacement:**

> **Resolved by the pass-1 reading: the analytical/constructive track ships.** Views are
> constructed per primitive class in closed form — face → centroid + normal at a clamped
> standoff; cylinder → caps + orbit stations spaced by the evidence tolerance — and then filtered
> and ranked. Surface-anchored generation is also better engineering before any quality argument:
> sphere sampling discards 33–77 % of candidates as inadmissible versus 0.08–32 % for
> surface-anchored generation, and the canonical survey's verdict on space sampling is verbatim
> "no benefits and their use is not recommended".
>
> **The optimization track is demoted, not deleted.** Local refinement of a constructed view
> enters only if it is ever shown to buy more than the fit uncertainty it would be correcting.
> Generative world models for predicting unseen views are out of scope this month.
>
> **Benchmark-driven, with the role scoped.** The benchmark is a three-tier component framework
> (Tier 0: single image, one decision. Tier 1: replay episodes against a dense capture — a "move"
> is a dataset fetch, travel costed in the kinematic sandbox. Tier 2: full-scale runs, real or
> replayed). It is auxiliary in effort and is the arbiter for open representation choices.
> Orthogonality is one pluggable scoring mode of a general view-planning framework, not the
> framework's purpose. One limit is recorded up front: Tier 1 can referee comparisons where both
> arms share a menu, and cannot cleanly referee space-versus-space comparisons, because a replay
> dataset can only serve views it contains and the geometry-derived views are precisely the ones
> that fall between the grid's samples.

**Provenance:** round-4 feedback (benchmark tiers, orthogonality as a mode); round-4 branches 4–6;
round-6 `benchmark-harness` §2.1.

---

## 8. New section — `## The Claim` **[decision: OQ-1, OQ-2]**

The framing doc has goals and RQs but no falsifiable claim, and the round-7 adversary's first
rejection was that the experiment set is a shotgun pattern around three different targets.
Proposed new section, placed after Goals:

> **Claim.** At matched view budget, the geometry-conditioned menu produces evidence images
> closer to the surface normal and higher grounded-QA success than the deployed 12×30°×3 uniform
> viewsphere, on the same incremental loop, same reader, same objects — measured system-level,
> co-headlined by grounded QA (view-correct × answer-correct + abstention) and by achieved
> evidence incidence scored against system-independent ground truth.
>
> **Comparison targets.** The deployed grid is the binding target — it is the system being
> obsoleted. A budget-matched equal-area (Tammes/HEALPix) sphere is reported as an exploratory
> strong control; on the assembled error budget the expected margin against it is inside this
> month's resolution, so it does not carry the claim.
>
> **Losing conditions, with their validity gates.** The claim is reported as *not proven* if the
> outcome-oracle row is not better at matched budget, or if the advantage does not survive
> execution. A losing condition counts only when its validity gates passed (item set inside the
> measured angle-sensitive band; snap error accounted; budgets matched in view count, menu
> cardinality, reachability filtering and model calls; the oracle's information set declared) —
> and symmetrically, a win under failed gates does not count.
>
> **If everything ties** — the likely outcome on the design's own arithmetic — the month still
> ships a working incremental system, demo v2's acceptance result, a measured comparison against
> the deployed grid, five curves nobody has published, and the harness. That floor is written
> down in advance so the kill switches are safe to leave armed.

**Provenance:** round-7 adversary §4.2, §4.5, §5.1–5.2, §5.6; round-6 `headline-comparison`
issues 1–8.

---

## 9. `## Sources` — currently empty

**Proposed:** point it at the workspace rather than duplicating it.

> The pass-1 corpus lives in `view-planning/passes/pass1/` — round-2 maps, round-3 per-document
> reads, round-4 branch syntheses, round-5 interpretations, round-6/7 verification. The
> consolidated build plan is `passes/pass1/methodology_v1.md`; the open questions this document
> does not settle are listed there in §9.

---

## 10. What deliberately does not change

- The ADAM non-goal, verbatim: fixed recipes per whole object or class are a regression;
  deterministic rules at the primitive level are fine. Every part of the build plan is checked
  against it, and the demo now carries a mechanical guard (an unmodified run on an object the
  code never names).
- The blind orchestrator. The eyes tier reads images, the orchestrator navigates symbols.
  Revisit only if the comparative numbers demand it; a marked render of a defective cloud has
  never been shown to work for a model, and the fidelity-floor evidence says it can be
  net-negative.
- "Full coverage of complex geometries" stays a non-goal; edges, corners and junctions stay out
  of the action space for v1 as a recorded, revisitable decision.
- The framing doc's role as the contract. This proposal keeps it a short contract; the machinery
  lives in the methodology document.

## 11. Application order (if accepted)

1. Items 1, 4, 6 (Context, Non-Goals, Constraints) — pure fact corrections, no decision needed.
2. Items 2, 3, 5, 7 (Problem, Goals, RQs, Directions) — encode the round-4 feedback; the
   architect should read the replacement text as written, since it changes what the doc commits to.
3. Item 8 (The Claim) — hold until OQ-1 and OQ-2 are ruled.
4. Item 9 (Sources) — trivial.
