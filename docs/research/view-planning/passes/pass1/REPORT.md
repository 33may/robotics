# Pass 1 — Report

**Date:** 2026-09-01. **Reader:** Anton. **Scope:** the month goal — working incremental
inspection system; one claim (geometry-conditioned view spaces beat the uniform viewsphere);
milestone demo v2 (box with codes, orthogonal evidence image, robust to move/rotate).
**Paths** below are relative to `docs/research/view-planning/passes/pass1/` unless absolute.
The binding review is your `round4_accumulate/feedback.md`; it overrides the framing doc
(`docs/research/2026-08-27-geometry-aware-view-planning.md`) everywhere they conflict.

Pass structure: round 1 plan (`round1_plan.md`) → round 2 maps (`round2_map/`) → round 3
per-document reads (`round3_reading/`, 120 files) → round 4 branch syntheses
(`round4_accumulate/`) → your feedback → rounds 5–7 re-run against the feedback
(`round5_interpretation/`, `round6_verify/`, `round7_verify/`) → consolidation
(`methodology_v1.md`, `design_update_proposal.md`).

---

## 1. What was researched — six branches

Branch structure per `round1_plan.md`. Per branch: documents mapped (round 2) vs read to
depth (round 3), and what the branch established.

**B1 — view-quality-scoring** (40 mapped / 20 read; `round2_map/view-quality-scoring.md`,
`round4_accumulate/view-quality-scoring.md`). What makes a single view good. The one proven
angle effect is resolution/foreshortening: pixel density on the surface falls as 1/cos θ,
measured 38,500 → 4,900 points from straight-on to steep tilt (8× loss). Every published
incidence knee (45–75°) belongs to a different sensor class — no transferable τ exists; the
field's practice is a hard gate + smooth model, and we must measure our own gate. The
load-bearing correction of the whole pass came from the datasheet read: **the D405 is passive
stereo, no projector** — the framing doc's specular-projector worry is void; the real worst
case is a textureless matte surface (the demo cardboard), Min-Z is 100 mm at 1280×720 (not
the marketed 70), and the stereo invalid-depth band eats ~10 % of image width at 10 cm. The
defect-visibility question (is orthogonal best for spotting flaws) is unanswerable from the
literature — irrelevant now that inspection = QA, per your feedback (`feedback.md` lines 14–18).
Tier 7 of the map (image-space quality scoring) went entirely unread
(`round7_verify/completeness.md` §2.7).

**B2 — benchmarking-view-planning** (58 mapped / 20 read;
`round4_accumulate/benchmarking-view-planning.md`). How to measure without fooling ourselves.
Core yields: the grounded 2×2 (view-correct × answer-correct) — in the one benchmark that
measures it, 26.2 % of trials were wrong-view/right-answer and raw accuracy overstates
grounded by up to 70 points; coverage metrics saturate for the uniform sphere (~98 % from a
generic 32-view sphere), so a coverage-primary benchmark loses to the baseline being obsoleted
for metric reasons; **no published result shows a prompted VLM beating random view selection
with variance**; the strong uniform control is Tammes/HEALPix at matched budget, not our
{h,v} grid; we are the benchmark-lottery N=1 worst case (one team authors benchmark, judge and
method); honest sizing — at n≲100 episodes we resolve 15–20-point differences, not 5. Your
3-tier benchmark vision was recorded here (`feedback.md` lines 31–43).

**B3 — primitive-structure-from-survey-clouds** (39 mapped / 20 read — sections E (sensor
defects) and F (failure detection) unread, the largest hole;
`round4_accumulate/primitive-structure-from-survey-clouds.md` §0). Fitting is off-the-shelf
and fast: Open3D `detect_planar_patches` returns bounded faces (center/extent/normal) —
verified against the installed library (`round6_verify/partial-primitive-fitting.md` §1.1);
CGAL/PCL return infinite primitives (extents are ours to write); CGAL Efficient RANSAC is
seedless and non-reproducible run-to-run; EMS superquadrics hold 86.4 % global-optimality at
20 % synthetic-cap coverage. But every published robustness number uses the wrong corruption
model (isotropic noise / hole-punching, never registration jitter + one-sided dropout +
systematic offset); residual-family fit scores are structurally blind to systematic bias
(SHREC cone: parameter error 1e11 at low residual); RSPD tiles large cylinders with fake
planes (acknowledged, unfixed); **nobody tests a half-scanned cylinder with one-sided dropout,
and no benchmark anywhere reports the angular error of a derived viewpoint** — the number our
design turns on. Six confirmed absences, all promoted to our own measurements.

**B4 — view-set-selection-under-constraints** (20 read, all of the reading list;
`round4_accumulate/view-set-selection-under-constraints.md`). The classical view-planning
problem. Sphere sampling discards 33–77 % of candidates as inadmissible vs 0.08–32 % for
surface-anchored generation; the canonical survey's verdict on space sampling is verbatim "no
benefits and their use is not recommended" — our claim is the field's own 20-year-old
conclusion. Orthogonal-by-construction is 30 years old and shipped (Mosbach Eq. 6: pose =
surface point + n·d; Wang T-Mech 2024 is the closest full precedent — never read primary,
structure trusted, tables refused). Exact ILP set cover solves in ~1 s at instances 10–100×
ours; the real per-round sink is the visibility matrix (ray-traced: 41 s–2000 s), forcing
analytic primitive-level visibility. Reachability: ~45/128 of a naive sphere reachable on a
tabletop rig; NBV-Net collapses 0.988 → 0.472 when its action space shrinks underneath it.
Pan 2025's treat-prior-as-fallible union is the corpus's only measured fallible-prior
technique. Your loop correction (`feedback.md` lines 45–49) inverted this branch's premise:
the iterative NBV structure IS our structure, with primitive fitting in the middle.

**B5 — structuring-the-viewpoint-space** (27 mapped / 20 read;
`round4_accumulate/structuring-the-viewpoint-space.md`). How view spaces are addressed. The
keystone structural argument: **the orthogonal view of a face is a measure-zero boundary
element of any sphere partition** — no grid refinement ever contains it; the fix is
construction order (surface → view), not a finer grid (formal statement:
`methodology_v1.md` §7). Aspect graphs died of combinatorial explosion — a warning, not a
recipe. Confirmed holes: no closed-form cylinder-orbit addressing scheme exists in any paper
(the orbit recipe ships as ours, derived); azimuthal canonicalization is unsolved, box-like
shapes its dominant failure class — which forces scene-anchored, content-found addressing
(demo v2's move/rotate robustness depends on this). Your intelligent-viewsphere correction
was recorded against this branch (`feedback.md` lines 51–55).

**B6 — vlm-spatial-interfaces** (63 mapped / 20 read;
`round4_accumulate/vlm-spatial-interfaces.md`). What the decider can be given. Six of six
deployed systems use the same pattern: geometry proposes a small labelled candidate set, the
model disposes by naming a label, never constructing or seeing raw geometry —
selection-over-emission is worth +60.7 points in the one within-paper ablation. Two-stage
decision (which surface → which view) is the largest ablated effect in the corpus (MOKA:
0.6–1.0 → 0.0–0.2 without decomposition) — the design you endorsed (`feedback.md` line 59).
Small menus (K≈2 beats K=5 by 6 pp); frame-implicit prompts collapse to ~0 % for 6/8 models;
menu order swings up to 73 points (oracle bound); models cannot infer view relations — hand
everything over flat. The gap our A/B fills: nobody has ever ablated label format for a
camera-pose menu. Unverified threats: SPAZER (+3.6 % over random) and TagaVLM — read before
leaning either way.

---

## 2. Key findings that matter for the build

Ranked by build impact, with sources.

1. **The claim is winnable only against the deployed grid.** Assembled error budget
   (`round7_verify/adversary.md` §1.1): treatment-arm achieved incidence error = fit error
   (7°+ on clean CAD, worse on partials) + registration (~4 cm ⇒ up to ~9° at 25 cm standoff);
   strong-uniform snap error ≈ 8–15°. Expected difference ~0–10° ⇒ ~2 % GSD — below anything
   this month's instruments resolve. Against the deployed 12×30°×3 grid (worst-case bins 30°+,
   all-vertical views) the win is real. `methodology_v1.md` §1 re-scopes accordingly (OQ-1).

2. **The advertised mechanism is not the one that wins.** The intelligent viewsphere bundles
   direction + standoff + framing; standoff/framing (a fixed-shell view of a small code face
   wastes most pixels) can be worth multiples of the ~2 % direction effect. Without a
   standoff/framing-parity arm, even a win says nothing about orthogonality
   (`round7_verify/adversary.md` §1.3). The parity arm is now mandatory in P7
   (`methodology_v1.md` §3-P7).

3. **Partial-fit fidelity is the load-bearing number and it exists nowhere.** Angular error of
   the derived orthogonal viewpoint vs surface coverage, on loop-realistic partials with the
   real defect triad — X4, week 2, with the recalibrated kill condition (paired distributions
   vs the control's snap error, not "menu resolution" — `round6_verify/partial-fit.md` §2.1).
   If it fails at loop-realistic coverage, the claim dies in week 2, cheaply.

4. **The camera can be vetoed in week 1, on the depth axis only.** Passive stereo + textureless
   cardboard is a live failure hypothesis; the smoke test (X2) got a written no-go bar
   (`round6_verify/sensor-smoke-test.md` §4-Issue 1). Depth "go" does not clear the camera —
   RGB evidence quality stays your demo-v2 call (OQ-10).

5. **One τ was doing three jobs.** τ_scan (depth admissibility, X2) ≠ τ_menu (menu density,
   from X3's RGB legibility curve) ≠ τ_evidence (the "orthogonal enough" success band — yours
   to set, OQ-3). Conflating them either serves 60°-off-normal "evidence" views
   (`round6_verify/view-crafting.md` §2.1) or lets the success criterion drift to whatever
   decodes (`round6_verify/demo-v2.md` Issue 2).

6. **Gates alone can't express "most orthogonal".** Boolean admissibility + set cover makes
   orthogonality a floor, not an objective — a cylinder region can be "covered" at ~τ
   off-normal, recreating Albert's complaint inside our machinery. Two-layer scoring
   (gate + continuous rank picking the per-region representative) is now in P4's contract
   (`round6_verify/intelligent-viewsphere.md` Issue 1; `methodology_v1.md` §3-P4).

7. **X3 (RGB legibility vs incidence + GSD, glossy under demo-cell lighting) is the bravest
   experiment in the month.** Flat to 40–50° ⇒ the orthogonality premise itself softens and
   menu density relaxes; a dip at 0° on glossy ⇒ the menu needs a small off-normal option.
   It also closes the glare-at-orthogonal worry your feedback left open (`feedback.md` line 10;
   `round7_verify/completeness.md` §3.1). Pulled to week 1 because three other experiments
   consume its curve.

8. **The only fully system-independent ground truth is PnP from the decoded code's own
   corners** (code corners + known size + intrinsics ⇒ face normal, pure RGB, independent of
   depth/fitting/registration). Discovered by the demo verifier (`round6_verify/demo-v2.md`
   Issue 1), promoted to the benchmark's angular scoring mode wherever decode succeeds
   (`methodology_v1.md` §3-P8). Everything else shares the pipeline with the system under test.

9. **Tier 1 cannot referee space-vs-space.** The uniform arm's requests are on-grid (snap≈0);
   the treatment's in-loop requests derive from partial fits and never coincide with captures —
   snap error is arm-asymmetric by construction, and snapped depth perturbs the whole replayed
   trajectory (`round6_verify/benchmark-harness.md` §2.1). Tier 1 referees same-menu A/Bs
   (serializer, order, selector floor, churn); the headline binds to Tier 2
   (`methodology_v1.md` §5).

10. **Identity across refits: stability is gameable, correctness is what matters.** Flip rate
    under hysteresis is minimized by freezing labels — stable-and-wrong (orthogonal evidence
    image of the wrong face, scored confident) is the demo-killing failure. X6 reports
    wrong-attachment rate against independent GT beside flip rate; X7's injected-flip curve
    sets the acceptance bar and its 100 %-cell prices the anonymous-menu fallback — i.e.
    whether the identity layer is worth building at all
    (`round6_verify/partial-fit-identity.md` Issue 2; `round6_verify/identity-persistence.md` §3).

11. **The stop rule nearly re-imported survey-first.** The two-phase MS@δ scan phase is a
    mini-survey that would also hand every matched-budget comparison to the baseline (~9 views
    burned before the first menu). Resolved: bootstrap = fixed 1–3 views; stop = answered ∧
    stability ∧ hard cap; MS@δ demoted to fallback guard; stop policy injectable
    (`round6_verify/loop-skeleton.md` Issue 1; `methodology_v1.md` §3-P1).

12. **Pre-verified ≠ executable.** An IK solution existing does not make the executed branch
    collision-free; the flagship joint-space paper fails 8/100 end-to-end on exactly this.
    Every menu entry now carries a witness configuration (roll + IK branch + motion-plan
    query), and execution-failure rate of pre-verified items is a standing metric
    (`round6_verify/view-crafting.md` §2.2; `round6_verify/menu-crafting.md` §2.2).

13. **The demo's ADAM guard was fake.** K×M placements of one box cannot distinguish
    per-primitive rules from a per-object recipe with a pose front-end. The mechanical guard is
    X18: the identical unmodified pipeline on an object the code never names; any required code
    change = ADAM regression, caught (`round6_verify/demo-v2-acceptance.md` Issue 1).

14. **Empty standoff band is a demo-shaped failure.** Large face + small code: frame the face
    and the code drops below the legibility floor; nothing in v1 had a closer-look action.
    Fixed: standoff = clamp(FOV-fit, Min-Z+margin, GSD ceiling); empty interval ⇒ deterministic
    ROI close-up / sub-face tiling (`round6_verify/view-crafting.md` §2.6).

---

## 3. The consolidated methodology (`methodology_v1.md`), compressed

Round 5 produced three parallel plans (your `round5_interpretation/anton.md`, plus
`academia.md` and `industry.md`) with colliding part names and three colliding E-numberings
("E1" was three different experiments — `round7_verify/completeness.md` §1.1).
`methodology_v1.md` merges them into 9 parts (P0–P8), 19 experiments (X0–X18), one benchmark
design, and an issue register (§6) in which **every** round-6/7 issue is marked resolved (R),
open for you (O → §9 OQs), or accepted residual (A).

**Parts** (§2–3), each with its round-6 verifier and verdict — all 17 part-documents returned
**needs-work**; what was wrong and how v1 resolved it:

- **P0 bench** (`sensor-smoke-test.md` — needs-work): go/no-go had no pass bar; τ possibly
  unidentifiable (texture vs incidence conflated); single-view unstated; camera authority
  overlapped your reserved decision. → Bar written before running; τ_scan from the textured
  control, go/no-go judged on textureless demo faces, 60°-provisional fallback; single-view
  mandated; verdict split by axis (depth veto ≠ camera cleared).
- **P1 loop** (`loop-skeleton.md`, `loop-scaffold.md` — both needs-work): MS@δ scan phase
  (survey-first relapse); three stop machines; "every run feeds Tier 1" overstated (sparse
  episode logs can't serve replay); scripted picker unspecified — the one ADAM vector;
  "~1 s/round" was the solver, not the round; part declared itself unfalsifiable; registration
  method unnamed. → Stop arbitration fixed and injectable; recorder claim corrected + first
  dense capture is a week-1 deliverable; picker = generic menu rule, zero object constants
  (doubles as P7's scripted row); analytic-visibility obligation + X9 measures wall clock;
  X9's revert-to-survey disproof attached; registration named and residual logged per round.
- **P2 fit** (`partial-fit.md`, `partial-primitive-fitting.md` — needs-work): kill threshold
  ~2× too lenient; exploration/evidence regimes conflated; partial generation unstated (the
  field's own hole-punching error); GT circular or cardboard-floored; gate features
  residual-only (pre-doomed per the branch's own reading); curvature guard untested; effort
  sized like a research program. → Paired-distribution kill vs snap error (median+p90), two
  bars, verdict read at loop-reached coverage; three labelled partial sources with the defect
  triad injected pre-fusion; per-surface-class independent GT (gravity/table, fiducials,
  freedrive touch-off, CAD); cross-fitter disagreement + cross-round dispersion added as gate
  candidates; guard gets a two-sided test; calibration cut to the 15–40 % regime on two shapes.
- **P3 identity** (`identity-persistence.md`, identity half of `partial-fit-identity.md` —
  needs-work): no acceptance bar; harm link asserted (NBV-Net analogy over-stretched); matcher
  unspecified on non-stationary addresses; flip GT = the matcher's own output; gameable under
  hysteresis; cylinder azimuth identity ill-posed; plane-SLAM data association unread. →
  X7 flip-tolerance curve sets the bar (100 %-cell = anonymous-menu price); parameter-space
  matcher with dummy rows, thresholds swept; independent correspondence GT; wrong-attachment
  rate beside flip rate, anchored to round 1, churn decomposed; scene-fixed azimuth convention;
  robot-base frame; half-day prior-art pass before writing the matcher; within-session promised,
  cross-placement reported-unpromised.
- **P4 factory** (`intelligent-viewsphere.md`, `menu-crafting.md`, `view-crafting.md` —
  needs-work): no continuous orthogonality layer (finding 6 above); one τ doing three jobs;
  fit-maturity gate outside the factory's contract (garbage fits laundered into confident
  menus); empty standoff band undeclared; set cover overstated; witness/roll defects; collision
  world on partial clouds unstated; orbit spacing sourced from a number on your refuse-list;
  no experiment falsified the construction. → Two-layer scoring; τ split; maturity gate pulled
  into the contract with a disclosed abstain state; clamp + ROI close-up; ILP demoted to
  pruning with emergent size; witness configurations bound to execution; inflated-union
  collision world; τ_menu from X3 (week 1); X8 construction audit (achieved vs planned
  incidence against locally-refit raw-cloud normals, framing on real images, worst-case mantle
  angle, per-region completeness).
- **P5 interface** (`menu-interface.md`, `menu-representation-ab.md` — needs-work): disclosure
  borrowed NBV-Net's number for a failure the design already precludes and had no test; the A/B
  couldn't pair episodes across serializers in a closed loop and its expected effect sat below
  its own detection floor; frame confounded with format; semantic labels leak task info;
  instruction-following pre-check dropped by all three interpretations; anti-repeat blocked
  legitimate post-refit recapture. → Teacher-forced frozen decision points (500–1000) as the
  primary instrument with stated MDE; frame × format factorized; information-matched variants;
  pre-check restored as a gate; anti-repeat carve-out tied to diff announcements; disclosure and
  diff arms (X11) scored on decisiveness; tie rule: accuracy ≻ order-stability ≻ cheapest.
- **P6 harness** (`benchmark-harness.md` — needs-work): union capture oversold as making Tier 1
  fair (it makes unfairness measurable, not absent); snap column had no decision rule;
  mechanical-first QA authoring selects on the dependent variable (reading items are exactly
  the angle-sensitive class); circularity guard and symmetry equivalence missing from the spec;
  holdout unpriced; Tier1↔Tier2 rank correlation unsupportable at n=5 arms. → Tier-1 arbiter
  role scoped to same-menu A/Bs; snap tolerance = X3's legibility flat band with exceed-fraction
  per arm; question-type stratification + deliberate angle-insensitive items (headline ships
  scoped to surface-reading QA if the margin concentrates there — which is demo v2's construct
  anyway); guards moved into scoring-code requirements; bracket capture + recipe-hash stamping;
  holdout designated day one, size in the frozen file; winner-agreement per comparison.
- **P7 comparison** (`headline-comparison.md`, `comparative-experiments.md` — needs-work):
  QA-item saturation ⇒ false kill / difficulty tuning ⇒ false win; "matched budget" was 3–4
  different currencies; "scripted oracle" conflated a deployable null with a GT ceiling; the
  obvious angular oracle re-imports circularity; the loop breaks the clean factorial; label
  parity, reachability filtering, and the scan tax unstated; the uniform column is a family. →
  Four named selector rows (random / scripted / LLM / **outcome oracle** — best achievable
  grounded QA by exhaustion, carries the losing condition); budget matched in four currencies
  (executed views, menu cardinality, post-filter reachability, LLM calls), scan tax charged to
  the treatment and said out loud; one serializer across spaces; five space arms incl. the
  deployed grid (permanent), Tammes (exploratory) and the standoff/framing-parity arm; validity
  gates attached symmetrically to losses and wins; system-level claim declared, matched-cloud-
  state variant (X12c) beside; small-fixed + large-fixed budget regimes.
- **P8 demo** (`demo-v2.md`, `demo-v2-acceptance.md` — needs-work): reference normal circular
  by default; tolerance band had three owners, two of which drift the construct; single-fraction
  score collapses the grounded grid; no unwinnable placements ⇒ confirmation instrument;
  abstention scored one-way; artifact-decodes vs system-answer conflated; discovery vs capture
  pooled; no acceptance bar or sizing; the ADAM-guard sentence unsupported; demo v2 absent from
  any binding document. → PnP-from-code GT (never the system's fit); single owner for
  τ_evidence (you — OQ-3); grounded grid + two-way abstention; adversarial strata with
  pre-registered expected outcome per stratum (away ⇒ success; near-table ⇒ best-reachable-
  with-disclosure; face-down ⇒ abstention = success); both answer criteria reported; discovery
  stratified (X14 measures discovery recall); bar + K×M stated before scored runs; X18
  generalization probe; demo v2 written into the contract via `design_update_proposal.md` §1.

**Experiments** (§4): X0 error-budget memo (week 1, gates the claim target) · X1 2-h fit
shakedown · X2 depth bench · X3 RGB legibility (+glare rider) · X4 partial-fit fidelity ·
X5 maturity-gate calibration · X6 identity/churn · X7 flip-tolerance curve · X8 construction
audit · X9 loop vs survey + wall clock + stop quality · X10 menu-representation A/B ·
X11 decision-structure/disclosure/diff arms · X12 headline (+X12c) · X13 selector floor ·
X14 discovery recall · X15 scorer noise floor · X16 harness self-checks · X17 demo acceptance ·
X18 generalization probe. Sequenced with gates G1–G3 (week 1 cheap kills → week 2 the wall →
week 3 interface+instrument, frozen-decisions file → week 4 claim+demo). The ~90 round-6 fixes
are triaged (claim-protecting / hygiene / deferred-with-reason) — the adversary's §3
meta-problem ("the fix list is a second project") answered in §4.

**Benchmark** (§5): three tiers, everything a swappable component; Tier 0 angle-stratified /
source-balanced; Tier 1 union capture + local brackets + recipe-hash, referees same-menu
comparisons only; Tier 2 carries the headline; sim deferred with reason (a raw cloud has no
material channel). Two scoring layers never blended: per-region evidence incidence against
system-independent GT (min over a symmetry-equivalence set, worst case beside the mean) and
grounded-QA episodes. Mechanical-first items with a normalization rule in the schema; type
stratification; item set frozen before headline runs. Paired stats, ≥3 sampled repeats on
model arms, per-stratum reporting, no α rituals. The frozen-decisions file replaces
pre-registration theatre: sources + decision rules frozen week 1, values at G3 (OQ-2). A
pre-committed **null-world deliverable** (§1) makes a flat month a result, not a scramble:
working system + demo v2 outcome + measured comparison vs the deployed grid + five unpublished
curves + the harness.

**Framing-doc updates** are drafted verbatim in `design_update_proposal.md` — nothing applied;
items 1/4/6 are pure fact corrections (QA scope, demo v2 into the record, passive-stereo +
loop + flexible-camera constraints), items 2/3/5/7 encode your feedback, item 8 (The Claim)
waits on OQ-1/OQ-2.

---

## 4. What the verifiers killed or flagged hardest

**Round 6: 17/17 needs-work.** No part was invented out of thin air — provenance audits traced
nearly every element to round-4 reading or your feedback — but every part as specified could
pass while wrong. Hardest hits: the partial-fit kill condition mis-calibrated ~2× lenient
(`round6_verify/partial-fit.md`); circular GT in three places (fit reference, flip reference,
demo normal); orthogonality inexpressible by the factory's own machinery
(`round6_verify/intelligent-viewsphere.md` Issue 1); the demo's ADAM guard unable to catch an
ADAM regression (`round6_verify/demo-v2-acceptance.md` Issue 1); Tier 1 structurally biased
against the treatment on the one comparison that matters (`round6_verify/benchmark-harness.md`
§2.1); the stop rule contradicting the binding loop premise (`round6_verify/loop-skeleton.md`
Issue 1).

**Round 7 adversary (`round7_verify/adversary.md`) — needs-work.** Three structural holes
living *between* parts, closable by decisions, not machinery:
1. **No error budget existed**, and the fragments implied the strong-control margin is ~2 %
   GSD — the claim as aimed was likely unprovable by construction; the claim it could win was
   the one it declared weak (§1.1–1.2). → X0 + re-scope (OQ-1).
2. **Gameable where nobody owned it**: item difficulty is a free knob that sets the effect
   size (N=1 team holds both ends); the angular scorer shared the pipeline; "pre-registration"
   was unfillable because its parameters are mid-month experiment outputs (§2). → disclosure +
   stratification + frozen file + PnP promotion; residuals carried as OQ-2/OQ-6.
3. **Six persona forks papered over, ~90 fixes with no triage or owner** — under deadline the
   claim-protecting fixes are exactly the ones that look skippable (§3). → §6 register + §4
   triage + the OQ list; the forks are now either resolved (stop rule, identity scope, ship
   architecture) or yours (OQ-1/2/3/4).
Also §1.3: without the parity arm, even a win is mis-attributed to orthogonality — "the one
claim is unattributable even if it wins."

**Round 7 completeness (`round7_verify/completeness.md`) — needs-work.** The pass had no spine
(three colliding numberings — fixed by `methodology_v1.md`); the contract is stale and **demo
v2 appeared in no binding document on disk** (fixed by `design_update_proposal.md`, pending
your acceptance); the real reading debt is ~3× what round 5 recorded — most dangerous: Branch 3
section F (failure detection — the literature behind the maturity gate, zero sources), GS-NBV
(only cylinder-orbit precedent, while the orbit is half the factory), Wang T-Mech primary,
SPAZER/TagaVLM, Katz HPR semantics, plane-SLAM data association (triaged in
`methodology_v1.md` §8); three design elements had no experiment (discovery recall → X14,
scorer noise floor → X15, glare-at-orthogonal RGB leg → X3 rider); the cylinder never met the
real robot → one real cylinder episode inside X18 (OQ-8).

---

## 5. Open questions needing your decision

Full statements with consequences in `methodology_v1.md` §9. Recommendations are the
integrator's, on the record, not decisions.

1. **OQ-1 — What must the claim beat?** Rec: the deployed 12×30°×3 grid binding;
   Tammes/HEALPix reported as exploratory strong control. Choosing Tammes as binding likely
   buys an instrument-resolution null, not evidence.
2. **OQ-2 — Frozen-decisions file: minimal, or nothing?** You dropped pre-registration
   theatre; the verifiers lean on pre-registered gates ~35 times and the losing conditions
   depend on them. Rec: one versioned file — sources/rules frozen week 1, values at G3, no α
   rituals. "Nothing" makes every kill switch negotiable in week 4.
3. **OQ-3 — Who owns τ_evidence (the "orthogonal enough" band)?** Rec: you set it as a
   requirement (≤15° proposed); X3 may tighten, never loosen. Sourcing it from the decode
   curve lets a 45° oblique image that decodes pass the demo.
4. **OQ-4 — Fallback granularity when the maturity gate trips.** Per-region (rec, with
   gate-trigger rate reported per arm) vs whole-menu vs whole-episode. Per-region dilutes the
   treatment arm toward the control exactly when the gate fires often; whole-menu wastes good
   fits.
5. **OQ-5 — Does the LLM keep the selection role if X13 ties?** Rec: geometry-selects /
   LLM-reads-and-answers ships as the deployable default; phrase the outcome as "no ≥MDE
   advantage measured", never "the LLM adds nothing" (underpowered cell).
6. **OQ-6 — Item authoring under N=1.** We hold both ends of the effect size. Rec: accept the
   disclosed residual (curve disclosure, stratification, frozen set, holdout) — or source a
   slice of items from Albert, which buys real independence for one conversation.
7. **OQ-7 — Angle labels vs directional words.** Your hypothesis is on the record; X10
   decides; expected outcome is a tie within MDE ⇒ cheapest serializer ships. Confirm the tie
   rule now.
8. **OQ-8 — Cylinder scope.** Rec: one real-robot cylinder episode inside X18, first thing cut
   if week 4 compresses — and if cut, the claim ships scoped to plane-dominated objects rather
   than silently claiming "geometry-conditioned spaces" on box faces alone.
9. **OQ-9 — Edges/corners/junctions stay out of the v1 action space** (recorded, revisitable,
   no literature either way). Confirm.
10. **OQ-10 — Camera decision split.** X2 can veto on depth in week 1; RGB evidence remains
    your demo-v2 call. Confirm the split so a week-1 depth "go" is not read as settling the
    camera.

Plus one non-OQ acceptance: `design_update_proposal.md` — items 1/4/6 are fact corrections
ready to apply; 2/3/5/7 change what the contract commits to (read the replacement text as
written); 8 waits on OQ-1/OQ-2.
