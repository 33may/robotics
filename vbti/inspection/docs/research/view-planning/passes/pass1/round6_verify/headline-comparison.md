# Round 6 — Verification: `headline-comparison`

**Part under test:** the one claim and the selector null (anton Part G–E8 / academia §2.6 E4+E5 /
industry E6+E7) — the full 2×2 at matched budget, {uniform viewsphere, intelligent viewsphere} ×
{scripted/oracle selector, LLM selector}, all four cells reported; mandatory arms: random-from-same-
menu floor, blind no-geometry baseline, matched-budget deliberation control; two pre-registered
losing conditions (oracle-row loss kills the geometry claim; LLM-vs-random loss kills LLM-selects
and ships geometry-selects/LLM-reads-and-answers).
**Role:** adversarial verifier. **Date:** 2026-09-01.
**Inputs:** round-4 feedback (binding) · all three round-5 interpretations · Branch 2
(`benchmarking-view-planning`) and Branch 6 (`vlm-spatial-interfaces`) syntheses · Branch 4/5 spot
checks · framing doc.

**Verdict: needs-work.** This is the best-grounded and most honestly falsifiable part of the whole
methodology — nothing here is round-5 invention, and both losing conditions are real, consequential,
and have shippable fallbacks. But as specified, the experiment can produce a false kill (or a false
win) through at least four harness artifacts that no interpretation fully closes: QA-item
saturation, arm-asymmetric replay snapping, an undefined "matched budget," and an under-specified
oracle that can re-import the circularity the 2×2 exists to remove. Two of the three mandatory arms
are transplanted from the literature without being operationalized for our architecture. All fixes
are specification work, not new machinery.

---

## 1. Provenance — grounded or invented in round 5?

**Grounded, element by element:**

- **The 2×2 itself** is Branch 2 Q14's explicit recommendation ("Run the full 2×2 and report all
  four cells... the single highest-value experiment the branch can specify"), assembled from
  measured precedents: `vin-nbv`'s fitness-swap, `gennbv`'s cross-implementation 2×2,
  `e3vs-bench`'s VQA-at-goal isolation. Q14 also records "**Missing:** Nobody runs the full 2×2" —
  so the part is original in execution but literature-shaped, exactly what round 5 was supposed to
  produce.
- **The legibility-confound rationale for the oracle row** is Q14 verbatim: "a geometry-conditioned
  space with 8 well-named views may win because it is smaller and more legible, not because its
  views are more orthogonal. The oracle-selector row settles that."
- **The random-from-same-menu floor** is Branch 6 Q15 ("no published result shows a *prompted* VLM
  beating a random view-selection baseline, with variance" — both positive comparisons use trained
  selectors) plus Branch 6's mandatory-arm list #1, including the chance-corrected BoR formalism.
  The SPAZER +3.6% caveat (unverified, nearest thing to a negative result) is correctly carried in
  anton §4 as a verify-before-leaning item.
- **The deliberation control** is Branch 6 mandatory arm #5, sourced from TwNV Table 2 (text
  self-reflection never beats R=0; +3.0 only from a new view at matched 2-call budget) and Think3D
  §5.6.
- **The blind baseline** is Branch 6 mandatory arm #2 (MMSI's Blind GPT-4o row) — but see issue 5:
  the transplant is nominal, not operational.
- **The losing conditions** descend from the feedback's own line ("the losing condition is real —
  if the geometry-conditioned menu does not beat the uniform sphere at matched budget on grounded
  QA, that ships as the result") and Branch 6's MV-GEL observation that geometry-selects/VLM-reads
  "is arguably the *default* against which the VLM-selects designs must justify themselves."
  Industry S1 is the same architecture named as the deployable fallback.

**One provenance nit:** the part's framing sometimes reads as if the oracle row *proves* view
quality directly. Branch 2's own caveat (Q14 Contested) is that field oracles are "oracle-ish" —
an oracle that re-optimizes on the test object is a different construct from one with GT access to
a fixed quantity. The part inherits the 2×2's shape but not yet this discipline (issue 4).

## 2. Scope compliance

- **Incremental partial-cloud fitting:** respected in intent — the intelligent arm runs the loop.
  But note issue 3(c): whether the loop's *scanning* views are charged against the matched budget
  is nowhere stated, and the answer changes who wins.
- **Real sensor defects:** Tier 1 replays real captures, so defects are present in-band. The
  injected-defect *ranking-stability* sweep (Branch 2 Q9's "the finding to hunt is a rank flip")
  appears only in academia's pre-registration ("or if it does not survive the defect conditions") —
  it should be named in the part or explicitly deferred.
- **Kinematic reachability:** intelligent-arm menus are pre-verified. The uniform arms' treatment
  is unstated — see issue 3(b); ObjView's 45.4/128 makes this a first-order fairness question, not
  a detail.
- **Discrete LLM menu selection / geometry never exposed:** respected in all cells; every arm is
  label-selection over finished menus.
- **Benchmark-as-auxiliary:** respected — the harness exists to arbitrate exactly this comparison
  and the representation A/B; no sprawl inside this part.
- **No VLA / RL / defect detection:** none present.
- **ADAM regression:** none. The scripted selector is a benchmark instrument; the losing-condition-
  (b) fallback (geometry-selects / LLM-reads-and-answers) is built from primitive-level
  deterministic rules, which the feedback explicitly sanctions. Shipping it is a scoped outcome,
  not a regression.

## 3. Where it breaks — issues

**Issue 1 (validity precondition): QA-item saturation produces a false kill, and the corpus
predicts it.** The oracle-row losing condition fires on "no difference in grounded QA at matched
budget." But if the demo-class QA items are answerable from mediocre views — large codes, a
legibility curve flat out to 40–50° off-normal (exactly the possibility anton's E6 disproof
condition entertains) — then all four cells saturate and "no difference" is an artifact of item
easiness, not evidence about spaces. This is Branch 2 F3 ("the metric decides the answer") wearing
a new coat: coverage saturated for the uniform sphere; easy QA saturates for everyone. The mirror
risk is worse for credibility: an N=1 team that *tunes* item difficulty into the exact
discriminative band manufactures the win (`benchmark-lottery` §6). Fix, cheap and pre-registerable:
(a) run E6/E12's Tier-0 legibility-vs-angle and GSD curves *first*; (b) require the headline item
set to sit demonstrably inside the measured angle-sensitive band, with the curve disclosed as the
justification; (c) report an easy stratum alongside so saturation is visible rather than hidden;
(d) source the view-correct tolerance band from the E6 RGB legibility curve — academia's
pre-registration currently derives "orthogonal enough" from E1/E8 (fit error and *depth* incidence
gate), which is the wrong construct for an RGB evidence claim.

**Issue 2 (harness artifact): Tier-1 snapping is arm-asymmetric and lands exactly on the measured
quantity.** Academia §1.5 names it and the part does not: the uniform arm's requested views exist
on the captured grid (snap ≈ 0); the intelligent arm's live-loop views are derived from *per-round
partial-cloud refits* and will not coincide even with union-captured, reference-fit-derived
candidate views. Snap error is therefore systematically nonzero for the treatment arm only — and
it can bias both ways: serving the nearest grid view eats the angular advantage (false loss);
serving the pre-captured reference-derived view substitutes a better view than the live loop would
have earned (false win, because the reference fit is cleaner than a round-2 partial fit). Anton's
Part G says "Tier 1/2" without making the Tier-2 slice load-bearing. Fix: the headline verdict is
not allowed to rest on Tier 1 alone — union capture + per-arm snap-error columns + the Tier-1↔Tier-2
rank-correlation check (academia E6) are preconditions, and the pre-registration states that if
snap-error distributions differ materially between arms, only Tier 2 referees.

**Issue 3 (undefined term): "matched budget" is three different currencies, and each unmatched one
is a documented confound.**
- **(a) Menu cardinality per decision.** If the LLM-row uniform arm presents the 36-cell grid while
  the intelligent arm presents ~8 items, the row is confounded by menu size alone (How-Many-Tools:
  K≈2 beats K=5 by 6 pp, 30 pp on hard cases — a measured, large effect). Branch 2 Q5's own
  recommendation (budget-matched uniform, Tammes at the same budget) implies the uniform arm should
  be served as an equally small menu; nobody says so explicitly in the 2×2's spec.
- **(b) Reachability-filtered budget.** Roughly two-thirds of a naive candidate sphere is
  kinematically unavailable (ObjView 45.4/128). If the uniform arm's budget counts unreachable
  cells, geometry wins partly because uniform wasted budget — a real deployment fact, but a
  different claim than "the views are better." Both arms must be IK/collision-filtered identically,
  budgets matched post-filter, with the uniform arm inheriting the same flagged-not-silently-removed
  discipline (NBV-Net's 0.988→0.472 collapse is what an unfiltered discrete arm does).
- **(c) The scan tax.** The intelligent arm must spend early views scanning to fit primitives; the
  uniform arm needs no fitting. If those scanning views are charged to the budget the comparison is
  conservative (fine — say so); if they are excluded, the comparison is rigged in the treatment's
  favor. Unstated in all three interpretations.
- **(d) LLM calls.** Two-stage selection roughly doubles calls per decision. This is exactly what
  the matched-budget deliberation control can bound — but only if the part *says* that is its
  purpose here (its literature purpose, TwNV's view-vs-thinking, is a different claim that belongs
  to the loop premise, not the space comparison). One sentence fixes the aim.

**Issue 4 (circularity re-entry): the oracle selector is under-specified, and the obvious
implementation restates the construction objective.** If "oracle" means "pick the view minimizing
angular deviation from the fitted normal," then the oracle row scores the intelligent space with
the same functional that built it — geometry wins the view-quality axis by definition (Branch 5's
measure-zero argument guarantees the uniform grid can essentially never contain the exact
orthogonal view). The 2×2 was commissioned to remove a confound, not to launder one. The clean
instrument exists and is affordable precisely because of Tier 1: an **outcome oracle** — for each
space, evaluate the VLM reader on every candidate view (they are all captured) and take the best
achievable grounded-QA per space. That measures "what the space makes possible" on the outcome
construct, shares no assumption with menu construction, and sidesteps Branch 2's oracle-ish
warning if defined identically for both arms. Keep the angular-metric oracle as a secondary mode
if desired, scored against locally-refit raw-cloud normals (industry's circularity guard), never
against the fitted primitive that proposed the view. The part must name which oracle carries the
losing condition.

**Issue 5 (unoperationalized arm): "blind no-geometry baseline" is three words with no referent in
our architecture.** Branch 6's mandatory arm #2 is blind/*no-image* — meaningful for VLM
benchmarks, already true of our selector by design. The phrase appears once (anton Part G) and is
defined nowhere. Three inequivalent candidates, each falsifying something different: (i) same menu,
content-free labels ("View 1..8") — tests whether the geometry-derived label semantics carry the
selection signal; (ii) answer with no new views at all — tests whether the task needs views
(largely duplicates the deliberation control); (iii) selector stripped of VLM-report feedback from
prior captures — tests whether selection runs on accumulated evidence or on menu priors. Pick one,
state what a null there would mean, or drop the arm. As written it is a checkbox, and a reviewer
will notice.

**Issue 6 (power + kill-rule asymmetry): the LLM-vs-random cell is likely underpowered, and a tie
is not a loss.** Over a small menu of by-construction-good views, most decision weight lives in
the menu (Branch 6's own reading); the residual LLM contribution on *terminal accuracy at matched
budget* is plausibly below the 15–20-point effect the harness resolves at n≲100 — and random with
budget will stumble onto the code face, so the LLM's real value should surface as **moves-to-answer
and travel under the cap**, not final accuracy. As specified, the kill rule can fire on an
underpowered null while the efficiency channel — the one the literature predicts (MapGPT: blindness
costs decisiveness/efficiency, not coverage) — goes unexamined. Fix: pre-register the kill metric
as chance-corrected (BoR, since P_rand over an 8-item menu is high) *and* efficiency-inclusive;
state the MDE; and phrase the ship decision honestly — "geometry-selects ships by default unless
LLM-selects shows ≥MDE advantage" is an engineering default on current evidence, not a measured
null. The distinction matters because Branch 6's "no published result" line is an absence of
evidence the project would otherwise convert into evidence of absence under its own name.

**Issue 7 (minor, framing): the losing conditions need their manipulation checks pre-registered
alongside them.** Both kills should stay — they are the part's spine. But an experiment that can
kill the claim through issues 1–3 can also fake a win through them; symmetry demands the validity
gates (item angle-sensitivity shown, snap-error accounted, budgets matched in all three currencies,
oracle named) ship inside the pre-registration, so that whichever way E8 lands, the result is
attributable to the claim and not the harness. Academia's §4 pre-registration is 80% of this;
what's missing is the explicit statement that a losing condition only *counts* when its validity
gates passed — and equally, that a win under failed gates does not.

**Issue 8 (minor): the uniform column is a family, not a cell.** Branch 2 Q5 mandates three uniform
variants (in-house 12×30°×3 as permanent control; budget-matched uniform; Tammes/HEALPix at matched
budget) — academia and industry carry all three, and "beating naive az/el is a weak claim" is on
record. The part's one-line "2×2" framing quietly compresses this to one uniform arm. Cheap fix:
state the design as 2×2 with the uniform column instantiated three ways, oracle row runs against
the strongest (Tammes) as the claim-bearing comparison.

## 4. Does it prove/disprove rather than confirm?

Structurally, yes — this is the only part of the methodology whose *failure* is written into the
plan with a named consequence, and losing condition (b) has a fallback that is arguably the
literature's default architecture anyway. The 2×2 shape genuinely separates space quality from
interface advantage, which no published work has done. But as currently specified the experiment
has multiple paths to an artifactual verdict in *both* directions: saturation and snapping can
kill a true claim; an angular-metric oracle, uncardinality-matched menus, unfiltered uniform arms,
and a free scan phase can confirm a false one. The falsification attempt is real; the instrument
is not yet clean enough to trust its answer either way. Every fix is a sentence in the
pre-registration plus modest harness discipline, not new scope.

## 5. Architect's standard (falsification + rendered evidence + usage story)

- **Falsification attempt:** the strongest in the whole methodology — two pre-registered kills,
  both with consequences the team has committed to shipping. Weakened only by issues 1–4's
  artifact paths and issue 6's tie/loss conflation.
- **Rendered evidence:** four cells + floors/ceilings + grounded 2×2 grids + evidence images per
  episode; the worst-case-per-object reporting (one grazing evidence image is Albert's original
  complaint) is carried by Part E. Meets the standard once snap-error and stratum columns are in
  the tables.
- **Usage story:** direct — a win ships the intelligent viewsphere with the LLM selector; an
  oracle-row-only win ships geometry's menus regardless of selector; a selector-null ships
  industry's S1. Every outcome has a deployable system attached. This is the part's best property.

## 6. Summary of required changes

1. Sequence E6/E12 (legibility-vs-angle, GSD) before E8; require the headline item set to sit in
   the measured angle-sensitive band; derive the view-correct tolerance band from the E6 RGB curve,
   not E1/E8; report an easy stratum. (Issue 1)
2. Make the Tier-2 slice load-bearing for the headline; mandate union capture, per-arm snap-error
   columns, and the Tier-1↔Tier-2 rank-correlation gate. (Issue 2)
3. Define matched budget in all four currencies: post-reachability-filter view budget, matched
   menu cardinality per decision in the LLM row, scan views charged to the budget (stated), call
   counts bounded by the deliberation control with that purpose named. (Issue 3)
4. Name the oracle: outcome-oracle (best achievable grounded QA per space, identical GT access
   both arms) carries the losing condition; angular oracle secondary, scored against locally-refit
   raw-cloud normals. (Issue 4)
5. Operationalize or drop the blind no-geometry arm; if kept, content-free-labels over the same
   menu is the variant that tests something the other arms don't. (Issue 5)
6. Pre-register the LLM-vs-random kill as chance-corrected + efficiency-inclusive with a stated
   MDE; phrase the fallback as a default-on-current-evidence, not a demonstrated null. (Issue 6)
7. Attach the validity gates to the losing conditions in the pre-registration, symmetric for wins
   and losses; state the uniform column as the three-variant family with Tammes as the
   claim-bearing control. (Issues 7, 8)

None of this changes the part's identity, cost class, or week-3–4 slot. All of it changes whether
E8's four cells can carry the month's one claim against a sceptical reader — including the one
writing this.
