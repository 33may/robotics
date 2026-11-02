# Round 6 — Adversarial verification: `comparative-experiments` (the pre-registered headline claim)

**Part under test:** the 2×2 {uniform sphere, intelligent viewsphere} × {scripted oracle
selector, LLM selector} at matched budget, under three uniform controls (permanent {h,v}
grid, budget-matched uniform, Tammes at matched budget), plus a random-from-menu selector
floor; losing condition written before the first run.
**Stance:** default scepticism. **Date:** 2026-09-01.
**Verdict: needs-work** — the best-grounded part of the whole round-5 plan, with real
disproof paths, but five specification gaps would let the headline experiment be run in a
way that is either confirmatory-only or confounded, which is exactly the failure this part
exists to prevent.

---

## 1. Provenance — grounded in round 4, not invented in round 5

This part survives the invention check cleanly. Every element traces to the accumulated
literature, not to round-5 improvisation:

- **The 2×2 itself** is Branch 2 Q14's explicit recommendation, verbatim: *"Run the full
  2×2 and report all four cells. It is the single highest-value experiment the branch can
  specify, because without it a win is uninterpretable: a geometry-conditioned space with
  8 well-named views may win because it is smaller and more legible, not because its views
  are more orthogonal. The oracle-selector row settles that."* (`benchmarking-view-planning.md`
  Q14, with mechanics from `vin-nbv`'s fitness-swap and `gennbv`'s cross-implementation
  2×2.) Branch 2 also honestly records: *"Nobody runs the full 2×2"* — so the design is
  literature-motivated but the full factorial is our construction. That is fine for an
  experiment design; it is not a citable claim.
- **The three uniform controls** are Branch 2 Q5's recommendation verbatim (permanent
  12×30°×3 {h,v} grid as in-house control that "never disappears from a table";
  budget-matched uniform per `gennbv`'s sized-to-budget recipe; Tammes at matched budget
  because `pose-noise-robustness-plant-vp` treats naive az/el as the weak construction).
- **The random-from-menu floor** rests on Branch 6 Q15's verified negative: *"no published
  result shows a prompted VLM beating a random view-selection baseline, with variance"*
  (both positive comparisons — Think3D +10.43, MV-GEL 0.598 vs 0.250 — use *trained*
  selectors). The floor is therefore not decoration; it is a live possibility that the LLM
  ties it.
- **The losing condition** is Branch 2 Q13's recommendation plus §8.5's pre-registration
  items, motivated by `benchmark-lottery`'s N=1 "rigging" case, which we are.
- **Convergence:** all three round-5 interpretations carry the same design independently
  (academia E4/E5 + §4; anton Part G/E8; industry E6/E7 + §5-arms), and the architect's
  Part G names both disproof conditions explicitly. No drift between interpretations on
  the core shape.

**Scope compliance:** the oracle/scripted selectors are primitive-level deterministic
rules (explicitly permitted; not per-object ADAM recipes); the LLM rows see only the
finished menu (geometry never exposed); no VLA, RL, or defect detection anywhere in the
part; and it is consistent with benchmark-as-auxiliary *provided* it stays the minimal
harness arbitrating the two open choices — academia §1.5's sprawl warning applies to the
surrounding framework, not to this part.

**Falsifiability:** genuine, on four independent kill-switches. (a) Oracle-row loss at
matched budget kills the space claim regardless of interface (anton Part G-a). (b) LLM-row
tie with random-from-menu kills the LLM-selects design and ships geometry-selects
(Part G-b). (c) Upstream, Part B's E2: if fitted-normal error on loop-realistic partial
clouds exceeds the menu's angular resolution, the claim dies before the 2×2 runs. (d)
Anton's E6: if RGB reading accuracy is flat out to 40–50° off-normal, any matched-budget
Tammes sphere almost surely contains a view inside the flat band and the intelligent
space cannot beat it on grounded QA — the claim honestly dies on the metric that matters.
Note that (d) is what saves the oracle row from being confirmatory-only: on the
*angular* metric the intelligent space wins by construction up to fit error (its views
are built to maximize exactly that score), so the informative oracle-row outcomes are
(i) angular score against **raw-cloud locally-refit normals** (tests the fit-error chain,
per industry §1.3.5's circularity guard) and (ii) **grounded QA** (tests whether
orthogonality pays at all). An oracle-row comparison on the angular score against the
fitted primitives that generated the views would be a tautology; the interpretations
mostly say this, but the pre-registration must make it binding.

It would survive the architect's standard: falsification attempts exist and are named;
rendered evidence is the grounded 2×2 grid plus evidence images; the concrete usage story
is demo v2. The remaining problems are specification gaps, listed in severity order.

---

## 2. Issues

### I1 — "Matched budget" is two different quantities, and the part conflates them
`gennbv`-style budget matching concerns **executed views per episode** (a 30-view grid
for a 30-view budget). Q14's legibility confound concerns **candidate-menu cardinality**
(8 well-named views vs a 36-cell grid). Matching the first does not match the second: a
budget-matched uniform arm could still present the LLM a larger/denser menu than the
intelligent arm, and the LLM-row difference would again be menu-size confounded — the
exact confound the part exists to kill. Additionally, in an incremental loop with an LLM
stop action, executed-view budget is dynamic; Branch 2 Q6's recommendation (hold the
stopping decision fixed in the action-space comparison, measure agent stopping
separately) is stated in academia §1.3 but absent from the part's own definition. The
pre-registration must define both budgets (menu cardinality per decision; executed views
per episode), state which is matched in which row, and fix the stopping rule for the
space comparison.

### I2 — "Scripted oracle selector" conflates two different constructs with different jobs
The interpretations use one label for two rows: (a) the **deployable scripted geometric
auto-selector** (industry S1 — picks the menu view most orthogonal to the queried region;
no GT access; this is the null architecture the LLM must beat), and (b) a **GT-access
oracle ceiling** (`vin-nbv` true-RRI / `objview-bench` greedy-GT rollout / Oracle-QA-by-
exhaustion in replay). These answer different questions: (a) is an architecture ablation,
(b) is the space-quality measurement the 2×2's oracle row requires. Branch 2 Q14's
contested note warns precisely about "oracle-ish" constructs whose information set is
undeclared. None of the three round-5 docs specifies the oracle's information set (GT
target region? reference-scan normals? exhaustive per-view decode in Tier-1 replay —
implementable and clean?). Un-specified, the oracle row is unreproducible and the
headline number means whatever the implementation happened to mean.

### I3 — The incremental loop breaks the clean factorial reading imported from static-set precedents
Every precedent 2×2 (`gennbv`, `vin-nbv`) had a **fixed candidate set**. In our loop the
intelligent menu at round k is a function of the partial cloud, hence of the selector's
earlier picks — the "space" factor is trajectory-dependent and entangled with the
selector factor, while the uniform arms' menus are static. The oracle row therefore
measures "space + fitting-loop under an oracle-driven trajectory," not the space factor
alone; the cells are policy-in-the-loop system comparisons, not a factor decomposition.
This is acceptable — the month's claim is about the system — but then the part's own
tagline ("the oracle row separates 'the space is better'") over-reads the design. Either
add a matched-cloud-state variant (freeze the partial cloud after a fixed scripted scan
prefix; compare all spaces × selectors from identical cloud state — cheap in Tier-1
replay) to recover the factorial, or pre-register the claim as system-level and say so.

### I4 — Label parity across arms is unaddressed; the legibility confound reappears inside the labels
If intelligent-menu entries read "top face, straight down" while the uniform arms' read
"az 210°, el 30°" (or bare indices), the LLM row confounds label informativeness with
view quality even at matched cardinality — SCAFFOLD's +5.7–10.3 for *any* explicit
scheme is large enough to manufacture a fake win. The oracle row is immune, so the space
claim survives; but the "LLM uses it better" cell needs a stated label policy: either run
the same serializer over both spaces (describe uniform views relative to the same fitted
geometry) or scope the LLM row explicitly as a whole-system comparison. No round-5 doc
addresses this.

### I5 — Reachability treatment of the uniform controls is unspecified
The intelligent menu is IK+collision pre-verified by construction. Are the uniform grids
filtered too, and is the budget matched before or after filtering? Unfiltered, the
uniform arms waste budget on unreachable views (~2/3 of a naive sphere per ObjView's
45.4/128) and lose for reasons that have nothing to do with view quality; filtered
pre-match, the effective budgets differ. Branch 2 F5 says reachability changes method
rankings, so this choice can decide the experiment. Must be pre-registered:
filter-then-match, same kinematic sandbox, for every arm.

### I6 — Tier-1 replay is structurally biased against the treatment; the headline cannot ship from Tier 1 alone
Academia §1.5 names it: a replay dataset serves only views that exist in it, and the
treatment's entire selling point is views that fall *between* grid views — snapping eats
precisely the measured advantage. The union-capture mitigation only partially fixes this,
because in-loop menus derive from *partial-cloud* fits, which will not coincide with the
capture-time reference-fit views (that discrepancy is E1's own subject). So even under
union capture, snap error returns for the treatment arm mid-loop. Consequence: the
headline number must come from the Tier-2 slice, or from Tier 1 with per-arm snap-error
columns demonstrating equal, negligible snap. Academia states this as advice ("only
Tier 2 can referee"); it must be binding in the pre-registration, or a Tier-1-only "no
better" result (or worse, a Tier-1-only win) will be reported on a confounded instrument.

### I7 — The matched budget derived from the treatment's set-cover count selects the regime where uniform is weakest
Budget justified "from geometry" means the number comes from the treatment's own
machinery, and small budgets are where a sparse uniform sphere is most likely to lack a
near-orthogonal view. `objview-bench` showed winners flip across budget regimes
(K=5 / K=30 / auto). The permanent large {h,v} grid control partially covers this, but
the pre-registration should adopt Q6's regime discipline explicitly (small-fixed +
large-fixed at minimum) and name budget-regime dependence as a reported result rather
than letting a single treatment-derived number carry the claim. Note `ma-scvp-longtail`'s
counter-warning cuts the other way too: the budget number needs a defensible source, and
set-cover-on-fitted-primitives is defensible — it just isn't neutral.

### I8 — Sequencing: the losing condition currently has no numbers to be written down with
The losing condition needs the "orthogonal enough" tolerance band (E1/E8/E6 outputs) and
the resolvable effect size (n≲100 episodes ⇒ 15–20 points, per all three docs). Those
experiments must complete before the first comparative run or the pre-registration is
unfillable except with placeholders — and a losing condition retro-filled after seeing
pilot comparative data is `benchmark-lottery`'s rigging case with extra steps. The
architect's sequencing (F+G in weeks 3–4, gated on B) is compatible; make the gate
explicit: E6's legibility-vs-incidence curve is the linchpin that turns the 2×2 from
confirmation into a test, and it is a cheap Tier-0 measurement.

Minor notes, not blocking: the random floor must be chance-corrected when compared across
menus of different cardinality (academia E5 already says "chance-corrected formalism" —
keep it); the two round-5 losing-condition formulations (anton: grounded QA at matched
budget; academia: oracle-row evidence score AND grounded QA, plus execution-space and
defect survival) differ in strictness — anton's is binding as architect, but pick one
sentence and version it; industry's "matched-budget deliberation control" (same calls,
more thinking, no new view) appears only in anton §1.5/Part G and industry — it is cheap
and worth keeping to attribute gains to the view rather than the extra turn.

---

## 3. What would convince me (suggested pre-registration deltas)

1. Define **budget twice**: menu cardinality per decision (matched in the LLM row across
   intelligent / budget-matched-uniform / Tammes arms) and executed views per episode
   (matched across all arms; stopping rule fixed benchmark-side for the space
   comparison, LLM stopping measured separately as its own number).
2. Split the selector axis into **four named rows**: random-from-menu (floor), scripted
   geometric auto-select (deployable null, no GT), LLM, GT-oracle (ceiling; information
   set declared — recommend oracle-by-exhaustion in Tier-1 replay: score every menu item's
   actual QA outcome, pick the best; it is implementable, reproducible, and construct-true).
3. Add the **matched-cloud-state 2×2 variant** (frozen partial cloud after a scripted
   scan prefix) as the factorial-clean companion to the full-loop system comparison;
   report both.
4. State the **label policy**: one serializer over all spaces in the LLM row, or scope
   the LLM row as system-level in the claim's wording.
5. State **reachability handling**: identical IK+collision filter for every arm,
   budget matched post-filter.
6. Bind the **headline to Tier 2** (or Tier 1 + demonstrated snap parity), with
   snap-error columns mandatory in any Tier-1 table.
7. Gate the first comparative run on **E6 + E1 outputs** filling the tolerance band and
   effect-size blanks in the losing-condition sentence; version that sentence with the
   harness before the run.

None of this is a redesign. The part's skeleton — 2×2, three uniform controls, random
floor, pre-written losing condition — is the strongest-precedented piece of the month and
should be kept exactly as scoped. The gaps above are the difference between "we ran the
highest-value experiment" and "we ran something shaped like it."
