# Round 6 — Verification: `intelligent-viewsphere` (the menu factory)

**Part under review:** closed-form per-primitive recipes (face → centroid + normal at a
standoff derived from FOV, Min-Z at operating resolution, invalid band, Tier-0
legibility floor; cylinder → incidence-binned orbit stations; caps as faces), gate
stack (incidence ≤ τ, standoff band, analytic/mesh-free occlusion, exact IK+collision
on every survivor), per-iteration set-cover dedup to a 6–12-item menu, two-stage
decision (surface first, view second), flat leaf menu, flagged-never-deleted
infeasibles with next-best, stable ordering, anti-repeat, deterministic and seedable.

**Role:** adversarial verifier. **Read:** round-4 feedback (binding), all three
round-5 interpretations, Branches 1/4/5/6 syntheses, the framing doc.

**Verdict: NEEDS-WORK.** The trunk is genuinely grounded — this is the
best-precedented part of the whole design, and the three interpretations converge on
it independently. But the spec as written omits three things its own source
syntheses say are mandatory (a continuous orthogonality ranking layer, the
fit-maturity gate inside the factory's contract, a declared behaviour when the
standoff band is empty), conflates two different incidence thresholds, and adopts
two design choices on argument alone with no test that could falsify them. All
fixable at spec level; nothing requires re-architecting.

---

## 1. Grounding audit — round-5 invention or reviewed literature?

Component by component, against the branch syntheses. This part is mostly *assembly*
of read evidence, not invention — but the assembly has seams.

| Component | Status | Source trail |
|---|---|---|
| Face → centroid + normal | **Grounded, twice, primary** | Mosbach Eq. 6 (`C = S + n·d`, one closed-form evaluation); Scott 2009 Optimal Scanning Zone (`f_d·R_o` normal dilation); Wang T-Mech 2024 as the full-pipeline precedent (structure trusted, tables not — the architect's own refusal list) |
| Standoff from FOV/density | **Grounded with a named caveat** | SEE Alg 2 density↔standoff↔FOV closed form — derived **fronto-parallel, no cos θ term** (Branch 1 flags this). Fine for face views (orthogonal by construction); do not reuse it unmodified for oblique framing checks |
| Min-Z @ resolution, invalid band | **Grounded, primary datasheet** | D405 Tbl 4-11 (100 mm @ 1280×720, not the marketed 70); `band = B·F/Z`, ~10 % of image width at 10 cm (Branch 1 §4.5). Parametric, so survives a camera swap |
| Tier-0 legibility floor in the standoff math | **Invented in round 5 — honestly, as a measurement** | No literature source; it is industry-E12's GSD-vs-legibility curve fed back into the recipe. Acceptable *because* it is self-measured, but it creates a bootstrap and a conflation (issues 2, 4 below) |
| Cylinder → incidence-binned orbit stations | **Invented, with an emergent-behaviour precedent only** | Branch 5 §7: "Any closed-form scheme for a cylinder-orbit address" is a **confirmed hole** — no paper names the recipe. Mosbach Fig. 6d (rings emerge from a 60° functional) is behaviour, not a recipe; GS-NBV, the one candidate orbit precedent, is **unread**. The derivation from Ikeuchi's smoothness rule is the branches' own construction. Defensible, but it must ship as "ours, derived" — not as precedented |
| Caps as faces | Trivially grounded | Mosbach's cylinder = caps + mantle; cap = plane face, recipe reduces to the face case |
| Incidence gate ≤ τ | **Grounded as a pattern; the number is not** | Classical lineage bakes quality gates into admissibility (Scott, Glorieux, Gospodnetić). Published τ spans **8.6°–75°** — no transferable value; every source ties it to its own sensor (Branch 4 §3.3). The 60° provisional + smoke-test plan is the right handling. But see issue 3: *which* τ? |
| Analytic / mesh-free occlusion | **Invented, acknowledged as unprecedented, well-motivated** | Ray-traced visibility is ~1–1.6 h/object (Gospodnetić) and cannot live in the loop; "analytic primitive-level check… plausible for planes and cylinders, **unprecedented in the corpus**" (Branch 5 §5.3). SEE Alg 8 point-proximity is the mesh-free fallback, primary-sourced |
| Exact IK+collision on survivors | **Grounded, strong** | Spensieri's hierarchical filter (cheap conservative → exact on survivors, explicitly because the cheap filter admits false positives); Scott 2001's filter-at-generation; the action-space-integrity argument priced by ObjView-Bench's NBV-Net collapse (0.988 → 0.472) |
| Per-iteration exact set cover | **Grounded — the field's canonical formulation** | Unanimous across 30 years (Branch 4 §2.1); exact ILP ~1 s at instances 10–100× ours (Pan 2025, MA-SCVP, Glorieux); per-round re-solve blessed by the feedback's loop correction. But see issue 5: what job is it doing here? |
| 6–12-item menu budget | **Weakly grounded** | Empirical anchors: minimal covers for tabletop objects are "single digits to mid-teens" (SCVP 9.6–13.1, Mosbach cylinder 22–24, Gospodnetić 12–34). The count is an **output** of the solve, not a knob (Branch 4 §3.2, MA-SCVP uses it as a complexity *metric*). Max-k-coverage — the formulation you'd need to *enforce* a budget — has "essentially no precedent" (one fallback sentence in Scott 2001). See issue 5 |
| Two-stage decision, flat leaf menu | **Grounded (largest measured interface effect), transfer assumed** | MOKA hierarchy ablation 0.6–1.0 → 0.0–0.2; flat-dump-at-our-scale from Structured-Interfaces small-graph column + SayPlan scale sweep. Transfer from seeing-VQA to a blind text menu is untested — but E9/E7 exists with a stated drop condition. Correct handling |
| Flagged-never-deleted + next-best | **Invented — the literature's four behaviours are repair/delete/fallback-then-fail/constrain-upstream; "flagged, degraded, still addressable" exists nowhere** (Branch 5 §5.2) | Motivated by real evidence (NBV-Net collapse; menu-shape instability for a blind LLM). Honest novelty. **No experiment anywhere tests it** — issue 6 |
| Stop action, anti-repeat, explicit frame | Grounded, free | MapGPT's always-listed stop; Think3D's anti-repeat; RoboSpatial frame-implicit collapse to ~0 % |
| Stable ordering | Grounded as risk management | Pezeshkpour order sensitivity (oracle bound 13.1–73.4; realized 0–8); benchmark permutes. Consistent |
| Deterministic & seedable | Grounded via a negative | CGAL Efficient RANSAC's seedless nondeterminism (Branch 3) is the anti-example; determinism of the factory itself is trivial given the fit (closed forms + ILP). Fine |

**Net:** not a round-5 fabrication. The genuinely novel composites (orbit recipe,
flagged infeasibles, analytic occlusion, the whole menu-factory-for-a-choosing-agent)
sit exactly on gaps the branches confirmed as gaps (enumeration-for-presentability
has no precedent — Branch 4 §8 item 10). The interpretations mostly *say so*. Where
they don't, I flag it below.

---

## 2. Scope compliance

- **Incremental loop:** respected — per-iteration recompute, cheap by evidence, prior
  treated as fallible upstream. ✔
- **Real sensor defects:** Min-Z/invalid-band/τ-from-smoke-test are in. One unpriced
  hole: mesh-free occlusion on a *partial* cloud cannot distinguish "occluded" from
  "never sensed" (SeeGround's named, unmeasured risk) — an unscanned occluder passes
  the gate. Low severity on a single-object tabletop, self-correcting in the loop,
  but it should be named in the spec rather than discovered at 2 a.m.
- **Kinematic reachability:** exact per-survivor check + flagged infeasibles — the
  strongest treatment in the design. But see issue 7 (pose ≠ trajectory).
- **Discrete LLM menu:** fully compliant. Geometry never surfaces; labels are Part
  F's open question; frame stated explicitly. ✔
- **ADAM regression:** no. Rules are per-primitive, sanctioned explicitly by the
  framing doc's non-goal carve-out and licensed by evidence (MVTN category-level
  90.6 % vs per-instance 91.0 %). The legibility floor is parametrized by target
  feature size, a task input, not an object script. Demo protocol independently
  guards this (industry Part F). ✔
- **No VLA/RL/defect detection; geometry hidden:** ✔ throughout.
- **Benchmark auxiliary:** ✔ — the factory *consumes* one benchmark number (the
  legibility floor), which is the feedback's sanctioned arbiter role, not sprawl.

---

## 3. Issues — ranked

### Issue 1 — The gate+cover machinery cannot express "most orthogonal view," which is the founding requirement. **[most severe]**
The part's selection logic is boolean gates + minimum set cover. Branch 4 states the
trap in its own contradiction table (§7.1): folding quality into a hard admissibility
gate makes orthogonality **a floor, not an objective — "which is not what Albert
asked for."** Its RQ1 implication explicitly prescribes the fix: a **two-layer
score** — boolean gate for the cover, plus a **continuous orthogonality/framing rank
used to pick the per-region representative**. That second layer is absent from this
part's spec. Failure scenario: cylinder mantle, stations binned at τ = 60°; the cover
drops a station because its neighbour still "covers" the region within the gate; the
served evidence view of that region is up to ~60° off-normal — Albert's original
complaint, recreated inside our own machinery, on the curved half of the design where
the framing doc's target behaviour ("cylinder orbited with the camera normal to the
surface") is most explicit. Faces don't bite (one orthogonal candidate each, the
cover must keep it); cylinders and compounds do.

### Issue 2 — One τ is doing two jobs; the part conflates depth-scan admissibility with RGB-evidence quality.
Under the QA reframe (feedback: "once we are in a good view we care only about the
RGB"), there are two distinct tolerances: **τ_depth** — incidence beyond which depth
fill collapses and the capture is useless *for fitting* (measured by the sensor smoke
test, E8/E11); and **τ_evidence** — incidence beyond which the VLM can no longer
*read* the surface (measured by E6/E12, Tier-0 reading-accuracy vs angle; anton's own
disproof condition anticipates it may be flat to 40–50°). These will not be the same
number, they gate different view roles (scan views vs evidence views), and
τ_evidence — not τ_depth — is what should set orbit-station spacing and menu density.
The part's single "incidence ≤ τ" hides this. Failure scenario: τ set from the depth
smoke test at 60° while code-reading dies at 35° → menu admits evidence views that
pass every gate and produce unreadable evidence images; or the reverse — reading is
flat to 50°, stations are packed at depth-τ density, and the menu carries redundant
near-identical orbit items (the untested similarity trap, Branch 6 Q9).

### Issue 3 — The fit-maturity/abstention gate is not in the factory's gate stack, and without it the factory launders garbage fits into confident menus.
The part lists four gates: incidence, standoff, occlusion, IK+collision. All four
check the candidate **against the fitted primitive** — none checks **whether the fit
deserves candidates at all**. The residual-blindness evidence (SHREC cone: parameter
error 1e11, residual fine) and the calibrated gate (E4's score→angular-error mapping)
live in Part B in all three interpretations; but the *consumer* of that gate is the
menu factory — it decides when the factory abstains and the uniform Tammes tier is
served instead. As specced, the interface between the parts is undefined. Failure
scenario: round 1, 15 % coverage, half-cylinder with one-sided dropout (the case
nobody in 20 papers tests, now the *common* case per the feedback); axis is wrong by
20°; all four gates pass happily against the wrong primitive; the LLM is served six
plausible-looking views orbiting a hallucinated cylinder. The fallback tier exists in
prose; it must be a named gate in this part's contract with a defined input (the E4
threshold) and a defined output (the abstain/fallback menu state, disclosed to the
LLM).

### Issue 4 — The face standoff is over-constrained with no declared resolution when the band is empty.
The recipe intersects: frame-the-face (standoff up from FOV footprint) ∧ Min-Z@res
(standoff ≥ 100 mm) ∧ invalid band (argues standoff up) ∧ legibility floor (standoff
**down**, GSD ≥ floor). For a large face carrying a small feature the band is empty —
frame the whole face and the code drops below the legibility floor; satisfy the floor
and the face overflows FOV. Nothing in the part says what happens then. The fix is
already in the read corpus and even cited as adopted precedent elsewhere in round 5:
FOV-grid the bounded primitive, one orthogonal viewpoint per cell (Wang T-Mech's
pipeline; Mosbach's area functional `E_A`, motivated verbatim by narrow-frustum
framing). Sub-face tiling must be a declared branch of the face recipe, not an
emergency patch. (Demo v2's box probably fits in one view — which is exactly why
this would otherwise ship untested.)

### Issue 5 — "Set-cover dedup to a 6–12-item menu" quietly mixes two formulations and overstates what the ILP does here.
(a) Min set cover emits an *emergent* count; if the min cover is 15, there is no
principled way to reach 9 — the budgeted formulation (max-k-coverage) has essentially
no precedent (Branch 4 finding 1), and two-stage decomposition already makes the
global budget mostly irrelevant (only the leaf size faces the LLM). (b) The cover's
objective is whole-object coverage economy — a survey-planning objective — while the
loop's menu serves per-surface QA; mid-episode, coverage-completeness is not what the
decision needs. (c) At 8–20 per-primitive candidates that are already
one-per-face + incidence-spaced stations, the ILP is plausibly a no-op — the greedy
tax and the ~1 s solve numbers are inherited prestige from instances where selection
had real work to do. None of this breaks anything (the solve is free), but the spec
leans on "set cover" as load-bearing where it is at most a tidy dedup pass, and the
6–12 number should be stated as an *expected emergent range with the leaf size as the
real constraint*, not as a guarantee the machinery enforces.

### Issue 6 — Flagged-never-deleted is adopted on argument alone; nothing in the experiment suite can falsify it.
The design directly contradicts the only deployed pattern with a real-robot record
(GraspView's silent removal) on the strength of an analogy (NBV-Net's collapse is a
*trained* selector, not a prompted LLM) and a gap ("no comparative study of
infeasible-option presentation" — Branch 6 gap 7). It is probably right, it is cheap,
and it doubles as the abstention vehicle E10 scores. But listing unavailable items
also *costs* something the corpus measures: distractor count degrades choice accuracy
−6 pp at K=5 vs K≈2 (30 pp on hard cases, How-Many-Tools), and a flagged item is a
distractor the model must not pick. Under the architect's falsification standard this
is an untested load-bearing interface choice with a known countervailing effect —
and it is one Tier-1 arm away from being tested.

### Issue 7 — "Exact IK+collision on every survivor" verifies a pose, not an executable action; the literature names this exact hole and the part doesn't close it.
Branch 4 gap 7: `g_kin ≥ 1` nowhere implies the *executed* IK branch is
collision-free — unaddressed in the entire corpus; and Wang 2026, the flagship
joint-space paper, still fails 8/100 end-to-end because path planning is a post-hoc
retry. Failure scenario: the checked pose has a collision-free elbow-up solution; the
executor's planner picks elbow-down, or no collision-free *path* to the (valid) pose
exists from the current configuration; execution fails on a "pre-verified" menu
item — the action-space-integrity failure this part's own rationale says is the one
thing that must never happen. The spec must either bind the verified IK branch
through to execution (check a specific configuration, hand that configuration to the
executor) or explicitly extend the flagged/next-best machinery to execution-time
failures (it composes — Yu & Gupta is the precedent that rungs compose) and say so.

### Minor
- Occlusion gate hole-ambiguity on partial clouds (§2 above) — name it.
- The legibility floor creates a bootstrap: Part C (week 2) consumes an E12 number
  produced by Part E from captured images. Existing captures make this workable, but
  the recipe needs a declared default standoff for rounds before E12 exists.
- "Seedable" is vacuous for closed-form recipes + ILP; the nondeterminism it guards
  against lives in Part B's detectors. Harmless, but don't let it imply the factory
  is where reproducibility is won.

---

## 4. Do the experiments prove/disprove — or only confirm?

Genuinely falsifying, kept: **E4** (flat score→error mapping kills residual gating —
and with it this part's abstention trigger), **E9/E7** (two-stage dropped if flat
matches — stated), **E6/E12** (flat reading-vs-angle curve *softens the orthogonality
requirement itself* — the bravest disproof condition in the month, and it belongs to
this part because it sets menu density), **E10** (abstention scored, not assumed).
The true kill switch for the whole factory is upstream and correctly placed: Part B's
E2 (fitted-normal error vs menu angular resolution — "the claim dies there").

Confirmatory-only, as specced: **E5** (reachability audit *calibrates* menu sizing;
it cannot fail the design), and nothing tests flagged-vs-silent (issue 6) or whether
the set-cover pass earns its place (issue 5). The τ conflation (issue 2) is invisible
to the suite because E8 and E6 measure the two thresholds separately and no
experiment checks that the menu uses each where it belongs.

Rendered evidence + usage story: strong — every factory output is an inspectable
menu + evidence image; demo v2 is a concrete, adversarially re-posed usage story with
narrated failures. This part would survive that half of the standard today.

---

## 5. Suggestions

1. **Add the second scoring layer** (Branch 4's own RQ1 prescription): after the
   gates, rank admissible candidates per region by continuous incidence, and let the
   cover/dedup select among *region-representatives*, never drop them. One paragraph
   of spec, closes issue 1.
2. **Split τ into τ_depth (scan admissibility, from E8) and τ_evidence (menu density
   and evidence gating, from E6/E12)** and state which gate uses which.
3. **Pull the fit-maturity gate into the factory's contract**: input = Part B's
   calibrated threshold; output = per-primitive serve/abstain state; abstain ⇒
   uniform-tier fallback item, disclosed in the menu. The factory owns the abstention
   behaviour it currently only implies.
4. **Declare the empty-standoff-band branch**: face → FOV/GSD tiling (Wang/Mosbach
   precedent), sub-face cells addressed under the parent face so the two-stage
   interface is untouched.
5. **Bind the verified IK branch to execution** (or explicitly route execution-time
   failure through flagged/next-best) and record that pose-level verification is not
   trajectory-level — cite Wang's 8/100 as the reason the next-best rung stays hot.
6. **Spend one Tier-1 arm on flagged-vs-silent removal** (piggyback on E7's replay
   episodes — same menus, one bit flipped). It is the cheapest genuinely novel
   ablation in the design (Branch 6 gap 7) and it converts issue 6 from taste to
   measurement.
7. **Demote the ILP honestly**: "exact set-cover dedup, expected emergent size 6–12,
   leaf menus ≤ ~8 by two-stage construction; if the emergent cover exceeds the band,
   the two-stage structure — not a budget constraint — is what keeps decisions
   small." Drop any implication that a 6–12 budget is enforced.
8. **Ship the cylinder-orbit recipe as a stated novelty** with Mosbach Fig. 6d as
   behavioural precedent only; read GS-NBV (and the solids-of-revolution aspect-graph
   paper, one DOI away) before any external claim about the orbit construction.
