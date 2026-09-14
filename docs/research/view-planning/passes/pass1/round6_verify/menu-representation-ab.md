# Round 6 — Verification: `menu-representation-ab`

**Part under test:** the menu representation A/B (industry Part D / anton Part F–E7 / academia E3) —
angle-style labels vs plain directional words vs short semantic descriptors vs bare indices (and
combos), frame stated explicitly in every variant, order permutations as a test condition, ≥3 seeds,
frontier + deployable-open tiers, plus two-stage vs flat menu; decided on Tier-1 replay numbers.
**Role:** adversarial verifier. **Date:** 2026-09-01.
**Inputs:** round-4 feedback (binding) · all three round-5 interpretations · Branch 6
(`vlm-spatial-interfaces`) and Branch 2 (`benchmarking-view-planning`) syntheses · framing doc.

**Verdict: needs-work.** The part is genuinely grounded, in scope, and carries real disproof
conditions — but the experimental protocol as written has four design flaws that would let it
produce numbers that don't answer the question it claims to answer, and its most probable outcome
(a tie) is pre-baked by its own power budget. All fixes are cheap.

---

## 1. Provenance — grounded or invented in round 5?

**Grounded.** Every load-bearing element traces to round-4 material, not round-5 invention:

- The gap claim ("nobody has ever ablated label format for a discrete camera-pose menu") is stated
  verbatim in Branch 6 §Q4 and again in its §5 gap list (items 4, 5): Think3D deploys integer
  angles and never varies them; MOKA uses chess notation and never varies it; ViewSpatial uses
  compass words and never varies them; textual az/el comprehension — the current in-house
  interface — has never been measured by anyone. The synthesis explicitly nominates both as "gaps
  our own benchmark could fill cheaply."
- The architect's round-4 feedback *names this exact use* as why the benchmark exists: "open design
  choices like the view representation (angle labels vs directional words vs anything else) are
  exactly why we need the benchmark framework — evaluate the different options as swappable
  components and select the best by numbers." This is the one part of the methodology the binding
  scope document explicitly commissions.
- Each design element has a measured antecedent: explicit frame in every variant (RoboSpatial
  frame-implicit collapse to ~0% for 6/8 models); order permutation as a condition (Pezeshkpour
  13.1–73.4-pt oracle swing, no correlation with option count, instability-as-failure-signal is
  Branch 6 mandate 4); ≥3 seeds (4-of-20-sources variance shame table); frontier + open tier
  (Branch 6 §1.1 — capability gating, results change sign across the boundary; mandate 6);
  two-stage vs flat (MOKA 0.6–1.0 → 0.0–0.2, the largest ablated effect in the corpus).
- The engineer's stated prior (small deltas between labelled schemes) is SCAFFOLD's own shape:
  baseline→any-scheme +5.7–10.3, best→worst-labelled only 4.6.

One novelty phrasing to keep honest: TwNV *does* contain a "Discrete" (categorical named-view)
condition — it is just reported without numbers. The claim should be "no documented ablation
exists," not "nobody has ever tested it." Both syntheses already carry this caveat; the part's
one-line summary drops it.

## 2. Scope compliance

- **Discrete LLM menu selection:** yes — all variants are label-selection over identical
  precomputed views. Angle-style labels are labels *of* finished views, not parameter knobs, so
  they do not violate "the LLM never sees primitives, coordinates or knobs" (and they are the
  architect's on-record tentative preference).
- **No VLA / RL / defect detection:** none present.
- **ADAM regression:** none — serializers are generic per-view label generators; nothing
  per-object. (Watch one drift: "semantic descriptors" sourced from VLM content reports rather
  than from geometry would start naming object-specific content; see issue 4.)
- **Benchmark-as-auxiliary:** respected — this is the architect's named use of the benchmark, and
  the industry framing ("cheap insurance, not a research program") keeps it contained.
- **Incremental loop / real defects / reachability:** Tier-1 replay serves real captures; menus
  pre-verified in the kinematic sandbox. The A/B itself runs on stable precomputed menus, which is
  a deliberate control, not a violation — but note issue 5: it thereby holds fixed the exact
  variables its own prior says are decisive.

## 3. Where it breaks — issues

**Issue 1 (protocol incoherence): "same episodes, serializer swapped" is impossible in a closed
loop.** The moment two serializer arms elicit different picks at decision 1, their episodes
diverge — states, menus, and remaining decisions are no longer matched, and the promised pairing
(same object, placement, QA item) degrades to episode-outcome pairing over different trajectories.
Neither interpretation says how this is handled. The fix exists and is cheaper than the bug:
score serializers primarily on **teacher-forced frozen decision points** — a bank of (state,
menu, oracle-best view) items where every variant sees the identical state and the single choice
is scored against geometry-known ground truth. Full closed-loop episodes then become a secondary,
smaller check. Without this, "paired items" in the design is partly fiction.

**Issue 2 (confound): the angle and word variants differ in reference frame, not just label
style.** Variant 1 is camera-relative ("45° right, 30° up **from current**"); variant 2 is
table/world-anchored ("top face, straight down"). RoboSpatial and the ViewSpatial/RoboSpatial
contradiction (Branch 6 §Q2 — the field's sharpest live dispute) say frame choice carries large,
model-dependent effects of its own. "Frame stated explicitly in every variant" equalizes frame
*disclosure*, not frame *identity*. As written, a win for either variant is unattributable —
label style and frame move together. For the ship-a-serializer decision this is tolerable (the
deployed thing is the package); for the claimed field-gap contribution ("first ablation of label
style") it is fatal to the attribution. Fix: factorize — {frame: camera-relative, table-anchored}
× {format: angles, words, hybrid, index} — or scope the claim down to "first ablation of menu
*presentation packages*."

**Issue 3 (power): the design's expected effect size is below its own detection floor.** The
part predicts SCAFFOLD-sized deltas (≤~4.6 pts between labelled schemes) while both
interpretations state that at n≲100 episodes the harness resolves 15–20-pt differences. The most
probable outcome is therefore a tie-by-construction, and the pre-stated default ("cheapest
serializer wins") makes the experiment's conclusion nearly independent of its data. That default
is a legitimate engineering rule — but then the A/B cannot claim to have *tested* label style,
only to have bounded it ("no effect larger than X detectable"), and the field-gap contribution
shrinks accordingly. The fix is the same as issue 1's: per-decision probes multiply n by the
decisions-per-episode factor at zero robot cost (replay fetches are free; LLM calls are the only
cost), and 500–1000 frozen decisions gets the minimum detectable effect down toward the range the
prior actually predicts. State the MDE up front either way.

**Issue 4 (confound): "short semantic descriptors" can smuggle task information.** A descriptor
tier built from geometry ("top face," "far side") is representation-only. A descriptor that leaks
content ("the labelled side," anything derived from prior VLM reports) answers part of the QA item
inside the menu itself; if that tier wins, the finding is "more task-relevant information helps,"
not a label-style result. The variants must be information-matched — same facts, different
encodings — with content-derived naming either excluded or run as a separately-labelled arm.

**Issue 5 (self-inconsistency): the experiment cannot touch the engineer's own prior.** The prior
on record is "label style matters less than frame explicitness and menu stability; cheap to
disprove." But frame explicitness is non-negotiable in every variant, and menu stability is held
fixed (same precomputed menus across arms). The A/B therefore varies only the factor the prior
calls least important and freezes both factors it calls decisive — it can rank styles but can
neither confirm nor disprove the prior, which remains an import from RoboSpatial/ObjView rather
than our measurement. Two cheap arms would fix this: (a) one frame-implicit control (replicates
the field's largest interface effect on our task; also the single most likely large-delta result
in the whole A/B); (b) one churn arm (stable content-derived names vs per-round renumbering) —
which additionally prices, in advance, the degraded interface that Part C's disproof condition
("identity flips are common") would force the system into.

**Issue 6 (underspecification): the two-stage vs flat ablation's interaction with serializer
choice is unaddressed.** Two-stage changes what the leaf labels must express (views *of an
already-chosen surface* vs global views); a serializer that wins flat may lose two-stage. Neither
interpretation says which serializer the two-stage ablation runs under, nor whether a winner is
picked before or after it. One line fixes it (e.g., run two-stage vs flat under the top-2
serializers, or pre-commit to crossing a 2×2 subset).

**Issue 7 (minor): the open-tier arm lacks the mandated instruction-following pre-check.**
Branch 6 §1.1's consequence — "a cheap instruction-following pre-check must gate the whole
design" — is nowhere in Part D/F. Without it, the deployable-open results may measure
instruction-following failure, not label comprehension, and the "winner changes across tiers"
expectation becomes unfalsifiable noise. Also minor: "seeds" on closed frontier APIs means
sampled repeats, not true seeds — say so; and the adversarial top-2-at-extremes order condition
(academia only) requires knowing the model's top-2, i.e. a pre-run — either specify it or drop
to the industry version (canonical/reversed/two shuffles).

**Issue 8 (minor, decision hygiene): no tie-break rule between accuracy and order-stability.**
Instability across permutations is declared "a failure signal, not noise" — good — but if
serializer A wins mean grounded accuracy while B is more order-stable, the part has no pre-stated
decision rule, and with 4–5 serializers × 5 orders × 2 tiers the many-comparisons winner's-curse
risk at small n is real. Pre-state: winner = best grounded per-decision accuracy, order-stability
as lexicographic tie-break, deltas below the MDE → cheapest.

## 4. Does it prove/disprove rather than confirm?

Partly. It genuinely can disprove the architect's angle-label preference (a large win for words
would do it, and the architect has pre-committed to shipping the winner). It cannot, as designed,
disprove the engineer's prior (issue 5), and its most probable outcome is an uninformative tie
(issue 3). The falsification structure is honest — the tie branch is pre-declared and the
decision default is stated — but the experiment as specified is closer to "confirm that label
style doesn't matter much, then pick cheap" than to a test that could surprise anyone. Adding the
frame-implicit and churn arms converts it into an experiment with at least two cells where a
large, decision-changing effect is actually *expected* by the literature — which is what a
falsification attempt should look like.

## 5. Architect's standard (falsification + rendered evidence + usage story)

- **Falsification attempt:** present but weakened by issues 3 and 5; fixable as above.
- **Evidence:** numbers tables with variance, per-tier, per-order — meets the standard once the
  per-decision metric is made explicit (score the *choice* against the geometry-known oracle view,
  not just episode QA outcome; the current text never names E7's metric).
- **Usage story:** clean and direct — the winner ships as the menu serializer; the two-stage
  result sets the decision structure. This is the strongest aspect of the part.

## 6. Summary of required changes

1. Primary instrument = teacher-forced frozen decision points (state + menu + oracle-best view),
   500–1000 items; closed-loop episodes demoted to a secondary check. (Issues 1, 3)
2. Deconfound frame from format (factorize, or narrow the novelty claim to "presentation
   packages"). (Issue 2)
3. Information-match all variants; quarantine content-derived descriptors. (Issue 4)
4. Add one frame-implicit control arm and one menu-churn arm. (Issue 5)
5. Specify the serializer under which two-stage vs flat runs. (Issue 6)
6. Add the instruction-following pre-check gate for the open tier; note API-sampling ≠ seeds;
   specify or drop the adversarial order condition. (Issue 7)
7. Pre-state MDE and the winner/tie-break decision rule. (Issues 3, 8)

None of these changes the part's identity, cost class, or schedule slot; all of them change
whether its numbers can be believed for the purpose the feedback assigns them.
