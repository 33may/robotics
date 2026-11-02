# Round 7 — Methodology adversary: the consolidated picture

**Role:** adversary over the WHOLE proposal — not the parts. The parts were verified in
round 6 (17 files, 17× "needs-work"); this pass asks whether the assembled experiment set
can discriminate at all, whether the benchmark is gameable, whether the three personas'
disagreements are resolved or papered over, and where a sceptical architect rejects
wholesale.
**Inputs:** round-4 feedback (binding) · framing doc · all three round-5 interpretations ·
all 17 round-6 verdicts · spot-checks against the round-4 syntheses (fit-error, Tammes,
1/cos θ, measure-zero numbers verified present at source).
**Date:** 2026-09-01. **Verdict: needs-work** — the build plan is executable and the
per-part fixes are real, but the consolidated experiment set has three structural holes
that no round-6 part-fix closes, because each lives *between* parts.

---

## 1. Discrimination: would the experiment set separate anything, or would everything pass?

### 1.1 Nobody has added up the error budget — and the fragments in hand say the margin is inside the noise

The month's claim is an *angular* claim: the intelligent viewsphere puts the camera closer
to orthogonal than the uniform sphere does. Every round-6 verifier priced one link of the
chain; no document anywhere assembles it. Assembled from the project's own numbers:

**Treatment arm — achieved incidence error at the evidence view:**
- fit-normal error: 7.0–7.5° for Efficient RANSAC on *clean complete* CAD (SPFN Table 1);
  1.5° floor only when type-conditioned. Partial clouds with the real defect triad:
  unmeasured (E1/E2's whole purpose), plausibly ≥7–15° in early/mid rounds.
- fused-frame registration: ~4 cm measured offset; at a 25 cm standoff that is up to ~9°
  of pointing error if uncorrected. The local-refit guard *detects* it at arrival; the
  Tarbox margin *mitigates*; neither zeroes it.
- execution error: negligible (UR5e).

**Strong uniform control — snap error to the true normal:** budget-matched Tammes/HEALPix
at ~10–20 reachable views has covering radius ~15–26°, *expected* snap ~8–15° (the
partial-fit verifier's own arithmetic, Issue 1).

So the expected *difference* in achieved incidence between the arms is ~0–10°. On the one
proven mechanism (GSD ∝ cos θ), that is cos 10°/cos 15° ≈ a **2% resolution difference**.
A mechanically-decoded QA item flips on 2% GSD only if it sits exactly at the legibility
cliff. Conclusion the consolidated plan never states: **against the strong uniform control
at matched budget, the headline experiment saturates on any naturally-difficult item set.**
E6's own disproof condition (reading flat to 40–50° off-normal) is not a tail risk — on
this arithmetic it is the *expected* outcome, and with it losing condition (a) of Part G
fires as an artifact of item easiness, not as evidence about spaces (exactly
headline-comparison Issue 1, now derivable in advance from the design's own numbers).

Where the treatment *does* clearly win: against the deployed 12×30°×3 {h,v} grid with its
all-vertical views (worst-case bins 30°+, no top-down view at all for some placements) —
the weak claim every persona agreed to be embarrassed about. The consolidated picture has
therefore engineered itself into: **the claim it can win is the one it declared weak; the
claim it declared strong it probably cannot resolve.**

### 1.2 The valid instrument is underpowered and the powered instrument is invalid

Round 6 established both halves separately; together they are a bind:
- Tier 1 (the workhorse, where seeds/permutations/n are affordable) is structurally biased
  on exactly the space-vs-space comparison — snap error is arm-asymmetric *by
  construction*, and the honesty clause ("if snap distributions differ, only Tier 2
  referees") triggers always (benchmark-harness §2.1).
- Tier 2 (the referee) resolves 15–20-point differences at n≲100 — while §1.1 says the
  true effect against the strong control is single digits.

So the headline number must come from the tier that cannot detect it. No round-6 fix
closes this, because each fix was scoped to its part. The only instruments that escape the
bind are the oracle-row measurements on union captures (outcome-oracle per space; angular
score vs independent GT) — which is precisely where the circularity guards are thinnest
(§2). The experiment set as consolidated does not discriminate; it defers.

### 1.3 The advertised mechanism is not the one that would produce a win

The intelligent viewsphere bundles three advantages: (i) orthogonal direction, (ii)
derived standoff (GSD/framing floor), (iii) face-centred framing. The uniform arms sit on
a fixed shell, object-centred. §1.1 says (i) is worth ~2% GSD against the strong control;
(ii) and (iii) can be worth *multiples* (a fixed-shell view of a small code face wastes
most of its pixels). **No experiment in any document separates direction from standoff and
framing.** A headline win would be real but mis-attributed: the orthogonality story — the
founding complaint, the thing every scoring mode measures — would be decoration on a
framing-and-zoom result. There is no standoff-parity control anywhere in the 2×2's
uniform column. This is a new consolidated finding: every round-6 verifier audited budget
currencies, label parity, snap parity — none asked for *distance/framing parity*, and
without it the one claim is unattributable even if it wins.

## 2. Gameability

The task's specific question — is the address space constructed to be orthogonal and then
scored on orthogonality — has named guards (raw-cloud locally-refit normals, grounded-QA
co-headline, outcome oracle, pre-registration, holdout). The guards are genuine but the
consolidated system remains gameable along three axes the guards don't cover:

1. **Item difficulty is a free knob that directly sets the effect size.** Per §1.1, the
   treatment beats strong uniform on grounded QA only for items inside the angle-sensitive
   legibility band. QA items are LLM-written under the architect's guidance; the
   headline-comparison fix ("require the item set to sit demonstrably inside the measured
   angle-sensitive band, curve disclosed") *discloses* the tuning, it does not remove it.
   The resulting claim is conditional and near-tautological: given 1/cos θ physics, items
   in the band favour the more orthogonal arm by construction. The QA layer, sold as the
   independent outcome check on the angular metric, re-derives the angular metric through
   decode physics on a curated item distribution. Combined with mechanical-first authoring
   selecting reading tasks (benchmark-harness §3), the N=1 team holds both ends of the
   effect size.
2. **The angular scorer still shares a pipeline with the system under test at benchmark
   scale.** Locally-refit raw-cloud normals come from the same sensor, registration, and
   normal-estimation stack that built the views; systematic bias common to construction
   and scoring is invisible (the exact SHREC/Araújo circularity Branch 3 documented). The
   one fully independent GT in the whole plan — PnP from the decoded code's own corner
   geometry — was discovered by the demo-v2 verifier and left *confined to the demo*. The
   benchmark's angular mode never inherits it.
3. **Pre-registration cannot bear the load it is assigned.** Every load-bearing threshold
   in the pre-registration (tolerance band, τ_evidence, MDE, budgets, gate values) is an
   *output of mid-month experiments run and interpreted by the same team on the same
   instruments*, with an explicit licence to soften the construct if E6 comes back flat
   (demo-v2 Issue 2's drift path). A "pre"-registration whose parameters are filled in
   after pilot data from the same chain, at week 3 of 4, under a ship deadline, is the
   benchmark-lottery rigging case with paperwork. The round-6 fix lists make this worse,
   not better: they add dozens of decision points (arms, thresholds, strata, gates) to be
   resolved mid-month — a garden of forking paths formally covered by an unfillable
   pre-registration.

Verdict on gameability: **not gameable on the axis everyone guarded (angular metric vs
fitted primitive); gameable on the two axes nobody owns (item difficulty, band
ownership) — and the wins and kills are symmetric, so the same holes can also produce a
false null.**

## 3. Persona disagreements: resolved or papered over?

Round 6 *caught* most forks; none is yet resolved, and the catching created a meta-problem.

| Fork | Positions | Status |
|---|---|---|
| Stop policy | anton: menu-stability+answered+cap · academia: exhaustion/LLM-stop/cap · industry: two-phase MS@δ scan then agent stop | **Open, and one branch violates the binding premise** (MS@δ scan phase = mini-survey-first; loop-skeleton Issue 1). Three stop machines = three different loops = the matched-budget comparisons mean different things per implementer. |
| Fallback granularity | per-region (academia) vs whole-menu (anton) vs fallback action-space S0 (industry) | **Open**; decides whether the treatment arm converges toward the control mid-experiment (hybrid contamination of the headline — partial-primitive-fitting 3.3). |
| Tolerance-band owner | E1/E8 (academia pre-reg) vs E6 decode curve (anton) vs GSD/E12 (industry) | **Open, construct-level**: this number is simultaneously the kill threshold for E2, the success criterion for demo v2, and the menu density knob. Three owners = the construct is whatever the last editor meant. |
| Identity scope | within-episode only (industry) vs cross-placement (academia E2) vs straddling (anton E3) | Open; E3 can fail on a property never promised. |
| Ship architecture | LLM-selects (anton's design) vs geometry-selects S1 (industry's "what actually ships") | Nominally resolved by the 2×2 — but the LLM-vs-random cell is underpowered and the tie rule defaults to S1 (headline Issue 6), so this fork will be resolved *by the power budget*, i.e. by industry's taste wearing an experiment. |
| Strong control necessity | all three personas import Tammes-at-matched-budget without dissent | **The disagreement that should have happened and didn't.** The month goal is to obsolete the deployed naive grid; the academic control maximizes null risk (§1.1). Nobody asked the architect whether the month's claim needs to beat Tammes or the thing actually being replaced. |

**The meta-problem:** 17 of 17 parts returned needs-work with ~90 required changes, new
arms (churn, disclosure, frame-implicit, flag-vs-silent, injected flips ×5 rates,
matched-cloud-state 2×2, four selector rows × three uniform variants), new GT protocols
(fiducials, touch-off, CAD objects, PnP), and several fixes in mutual tension
(matched-cloud-state scan prefix vs the no-survey premise; charging the scan tax vs claim
viability; teacher-forced probes vs a system-level claim). No triage exists, no owner, no
statement of which fixes protect the one claim vs which are hygiene. The "auxiliary"
benchmark plus its verification debt is now the largest workstream in the month. As
consolidated, the plan is not executable in four weeks — and the failure mode is silent:
under deadline, exactly the claim-protecting fixes (error budget, band ownership,
independent GT) are the ones that look skippable.

## 4. Where the sceptical architect rejects wholesale

1. **"Show me the error budget."** No document sums fit error + registration + snap vs
   the legibility band. The fragments in hand (§1.1) suggest the strong-control margin is
   ~2% GSD — i.e. the month's claim may be unprovable *by construction* at natural item
   difficulty. This is a week-1 back-of-envelope plus E1; running weeks of harness work
   before it exists is building the courtroom before checking there is a case.
2. **"Which claim is it?"** Space claim, system claim, or interface claim — the loop
   entangles them (comparative-experiments I3), the budget currencies decide who wins
   (headline I3), and the mechanism is unattributed (§1.3). Until one sentence exists —
   *the month's claim is X, measured by Y, against Z* — the experiment set is a shotgun
   pattern around three different targets.
3. **"Your arbiter arbitrates nothing."** Both open choices the benchmark exists to settle
   are ties-by-construction: the representation A/B expects ≤5-point deltas on an
   instrument that resolves 15–20 (menu-representation-ab Issue 3); the headline
   saturates or is tuned (§1.1, §2.1). A benchmark whose every planned decision lands on
   the pre-declared default ("cheapest serializer," "S1 ships") is taste with
   instrumentation — worse than admitted taste, because it launders.
4. **"The fix list is a second project."** §3's meta-problem. Without triage, either the
   month ships the build and silently drops the validity gates, or ships the gates and
   drops the build.
5. **"What do we ship in the probable-null world?"** E6 flat, E8 null vs Tammes, E7 tie,
   LLM-row tie — every one of these is the *likely* outcome per the design's own numbers,
   each has a local fallback, but no document states what the month delivers if they all
   fire together. (The honest answer is respectable: a working system, beats-the-deployed-
   grid, plus four measured curves nobody has published. It should be written down as the
   floor, in advance, so a null month is a deliverable and not a scramble that invites
   post-hoc item tuning.)

## 5. What would change the verdict

1. **Week-1 error-budget memo** (half a day + E1): measured fit-error distributions ×
   computed snap distributions of all three uniform controls × the 4 cm offset's angular
   contribution, laid against the E6 legibility curve. If the distributions overlap for
   the strong control, re-scope the claim to the deployed grid *now* and demote Tammes to
   an exploratory arm — on the numbers, not in week 4.
2. **One claim sentence from the architect**, fixing: the comparison target (deployed
   {h,v} grid as binding; Tammes exploratory), the level (system-level, say so), the
   co-headline metrics (grounded QA + outcome-oracle), and the single owner of the
   tolerance band (E6's RGB curve; E1/E8 are diagnostics that may fail it, never define it).
3. **A standoff/framing-parity arm**: uniform directions + the treatment's standoff logic.
   One arm, attributes the win to the advertised mechanism or exposes that framing did the
   work. Without it, even a win teaches nothing about orthogonality.
4. **Promote PnP-from-code (or fiducial) GT** from demo-only to the benchmark's angular
   scorer wherever decode succeeds — the only scoring path fully independent of the
   system's sensor/registration stack.
5. **Triage the ~90 round-6 fixes** into claim-protecting (error budget, band owner,
   snap accounting, budget currencies, oracle definition, independent GT — must land) vs
   hygiene (disclosure/churn/flag-vs-silent ablations — run only if weeks 3–4 have slack);
   name an owner per fork in §3 and resolve all six in one decision memo before the
   skeleton locks interfaces.
6. **Pre-commit the null-world deliverable** (§4.5) in the same memo as the losing
   conditions, so the kill switches are safe to leave armed.

**Bottom line.** The parts are grounded, the falsification culture is real, and the build
skeleton is right. But assembled, the proposal currently discriminates only against the
baseline it is embarrassed to beat, cannot attribute a win to its advertised mechanism,
holds both knobs that set its own effect size, and has scheduled more validity work than
the month contains. Every repair is cheap; all of them are decisions, not machinery; none
of them belongs to any single part — which is exactly why, after seventeen needs-works,
nobody owns them yet.
