# Round 6 — Adversarial verification: `benchmark-harness` ("The three tiers as instruments")

**Verifier stance:** break it. **Part under test:** Tier 0 single-image decisions; Tier 1 dense
pre-capture replay with union-capture (uniform grid + geometry-derived views) and snap-error as a
first-class column; Tier 2 full quantitative runs; mechanical-first QA authoring with free-form
quarantined; orthogonality as one pluggable scoring mode; paired stats over matched cases, ≥3 seeds,
WSR intervals, per-stratum reporting, random floor + oracle ceiling in every table, day-one holdout.

**Inputs read in full:** round-4 feedback (binding), all three round-5 interpretations, Branch 2
synthesis (`benchmarking-view-planning.md`, all 1,686 lines), the framing doc; targeted checks of the
other four syntheses.

**Verdict: needs-work.** The tier skeleton, the mechanical-scoring move, and the statistics
discipline are the best-grounded pieces of the whole round-5 methodology — almost every element
traces to a read source or to the architect's own feedback, not to round-5 invention. But the two
genuinely novel elements (union-capture, snap-error column) do not fully survive contact with the
incremental-loop constraint, the QA-authoring policy has an unnamed selection effect on the headline
claim, and two load-bearing guards that exist elsewhere in round 5 are missing from this part's own
spec. None of this is fatal; all of it is fixable in the harness spec before code is written.

---

## 1. Grounding audit — literature, feedback, or invented in round 5?

| Element | Provenance | Verdict |
|---|---|---|
| Three-tier structure (0/1/2) | Architect's round-4 feedback, verbatim ("Benchmark vision — 3 tiers") | **Feedback-grounded.** Not a round-5 invention. |
| Tier 0 = oracle-view cell | `view-level-evidence-benchmark` Acc_oracle; `e3vs-bench` VQA-at-goal (3.58 ≈ human 3.53) | **Grounded.** The tier split is justified by e3vs's own numbers: the bottleneck is finding, not reading. |
| Tier 1 replay, move = fetch | `ma-scvp-longtail` §V-F1 real-world protocol (exhaustive 32-view capture as GT); travel from poses = feedback's own ruling ("kinematic sandbox... already implemented") | **Grounded.** |
| Union-capture | **Round-5 invention** (academia §1.5). No precedent — MA-SCVP's arms all drew from one shared candidate set, so the problem never arose there | Invented, but as a *response to a real bias* the framing doc itself names (the orthogonal view falls between grid bins — applied to our own dataset). Honest instrument, not a confirmation device. See §2 for where it breaks. |
| Snap-error column | **Round-5 invention**, modeled on `pbr-structured-light-sim`'s gap-as-a-column pattern (Branch 2 Q10) | Grounded in pattern, novel in application. Under-specified — see §2.3. |
| Mechanical-first QA scoring | `ycb-protocols` (every score a physical measurement) + `planning-with-the-views` (judge-free geometric success) — both read, both explicitly cited as the transplant | **Grounded.** Also the correct answer to Branch 2 Q3/Q17 (judge circularity; e3vs judge–human ρ=0.54; undisclosed judge identity in the one clean benchmark). Quarantining free-form + disclosing judge identity is exactly Q17's recommendation. |
| Orthogonality as pluggable mode | Architect's feedback, verbatim ("orthogonality is just one pluggable mode of the framework, not the benchmark target") | **Feedback-grounded.** |
| Paired matched cases | `ma-scvp-longtail` (3,000 matched cases; reader: "scales down fine, it is about pairing, not sample size") | **Grounded.** |
| WSR intervals | `binomial-eval-statistics`, including the round-4 *correction* (WSR, not Clopper–Pearson, for bounded non-binary scores) | **Grounded**, and correctly carries the corrected claim, not the falsified "15×" number. |
| ≥3 seeds on VLM arms | Branch 2 F9 (16% of 445 benchmarks report any uncertainty; `view-level-evidence-benchmark`'s 5-run CI as existence proof) + architect's keep-list | **Grounded.** |
| Random floor / oracle ceiling | `secord`'s naive-floor/oracle-ceiling scale; `planning-with-the-views` random row in the main table; `vin-nbv` Oracle-NBV | **Grounded.** With a circularity caveat — §3. |
| Per-stratum reporting | `objview-bench` bucket-covering sampling; `ma-scvp-longtail` (simple stratum saturates at ~99.9%, pooled average hides the effect) | Grounded in mechanism; the *strata themselves* (pure/compound/near-primitive) have no precedent — Branch 2 Q11 says so explicitly, and the syntheses acknowledge it. Fine: original-by-necessity, flagged as such. |
| Day-one holdout | `benchmark-lottery` §5 (intra-team adaptive overfitting; "we are the N=1 worst case") | **Grounded.** Unpriced — §4. |

**Conclusion on grounding:** this part is the *least* invented part of round 5. Nothing here
contradicts the feedback; the two novel elements are anti-bias instruments pointed at the design's
own arm, which is the right direction for invention to point.

## 2. Attack: Tier 1 vs. the incremental loop — the part's strongest claim is its weakest joint

### 2.1 Union-capture cannot serve the treatment arm exactly, by construction

The union-capture promise ("the replay executor serves exact frames for both arms") holds only if
the treatment arm's requested views coincide with the geometry-derived views computed **from a
reference fit at capture time** — i.e. from a complete, dense cloud. But the system under test is an
incremental loop (binding feedback: "we do NOT have the scanned object up front"): at round *k* it
derives views from a **partial-cloud fit**, whose normals differ from the reference fit's by exactly
the angular error E2/E1 is being built to measure. So:

- The **uniform arm's snap error is ~0 by construction** (its requests are on-grid).
- The **treatment arm's snap error is nonzero in every round before the fit converges**, equal to
  partial-fit angular error + capture discretization.

The part's own honesty clause — "if the arms' snap-error distributions differ, Tier-1 comparisons
are confounded and only Tier 2 can referee" — will therefore trigger **by construction**, not as a
rare failure. Taken seriously, it disqualifies Tier 1 as arbiter for the one comparison the month
cares about (space vs. space), which contradicts Tier 1's billing as "the workhorse."

There is a second, unnamed compounding effect: in replay, the served (snapped) frame's **depth**
enters the accumulated cloud, which changes the next refit, which changes the next request. Snap
error does not just perturb the evidence view — it perturbs the whole trajectory. Replay of the
incremental loop is not a passive fetch sequence; it is a slightly different episode than the real
robot would run. No interpretation names this.

**What survives, and it is a lot:** for the A/Bs where both arms share the *identical* menu —
menu representation (E3/E7), selector floor (E5), order permutations, model tiers, two-stage vs
flat — snap error is common-mode and cancels. Tier 1 is a clean, cheap, unconfounded arbiter for
exactly those. That is also where seeds and permutations make Tier 1 indispensable.

**Required fix (spec-level, cheap):** scope Tier 1's arbiter role *in the harness spec*: Tier 1
referees same-menu comparisons; the space-vs-space headline runs on Tier 1 only as a development
signal, with the citable number coming from Tier 2 (or from Tier 1 episodes whose per-episode snap
error falls inside a pre-stated tolerance — see §2.3). Academia §1.5 gestures at this ("the arbiter
role... must be scoped to the choices it can actually referee") but the part-as-stated still
presents union+snap as *making* the space comparison fair on Tier 1. It does not; it makes the
unfairness measurable. Those are different claims, and only the second is true.

### 2.2 Union-capture couples the dataset to the current menu-crafting implementation

The geometry-derived half of the union is computed by Part-D's recipes *as they exist on capture
day*. Every change to standoff math, orbit spacing, or the incidence gate moves the derived views —
and silently converts an exact-serve dataset back into a snapping dataset (or forces recapture of
every object × placement, which is precisely the maintenance bill the academia interpretation warns
about via YCB's 32-of-33-dead object sets). The day-one holdout makes this worse: holdout captures
are, by definition, captured early, under the *earliest* recipe version, and can never be recaptured
without breaking the holdout discipline.

**Fix:** capture derived views with a local bracket (e.g. the derived pose ± one tolerance step in
the two view-sphere directions), so recipe drift within the tolerance band still serves exact-ish
frames; version-stamp the recipe hash into every capture set; state the recapture policy in advance.

### 2.3 The snap-error column has no decision rule attached

As specified, snap error is descriptive. Nothing states what magnitude is disqualifying, so the
column can absorb any inconvenient value ("we reported it" is not a control). The project already
plans to measure the one number that gives the column teeth: the legibility-vs-incidence flat band
(anton E6 / industry E12 — code-reading accuracy vs angle at matched distance). A snap that stays
inside the flat band is provably harmless to the QA outcome; a snap outside it is a confound.

**Fix:** pre-state the per-episode snap tolerance as "within the measured legibility flat band,"
and report the fraction of treatment-arm episodes exceeding it. This turns the column from honesty
theatre into a filter with a falsifiable threshold — and it costs nothing beyond E6/E12, which are
already planned.

## 3. Attack: mechanical-first QA authoring selects on the dependent variable

The treatment's proven mechanism is resolution/foreshortening (1/cos θ; the 8× point-density loss).
Mechanically-verifiable items — decode a barcode, exact-match an OCR string — are precisely the
question class **most sensitive to that mechanism**. Angle-insensitive questions (is the lid on?
how many objects? which color is the top face?) are harder to score mechanically and get quarantined
into the judge-scored slice. So the authoring policy correlates question type with the treatment's
strength: the primary metric is computed over the question distribution where geometry-conditioning
wins by physics. This is `benchmark-lottery`'s rigging-the-lottery pattern operating *through item
authoring* rather than metric choice — subtler than the coverage-metric trap the interpretations do
pre-empt, and named nowhere in round 5.

Two honest resolutions, both cheap:

1. **Stratify by question type** (angle-sensitive reading vs angle-insensitive), not only by object
   geometry. Report both strata; expect the treatment's margin to concentrate in the first. That is
   fine — but then the headline claim ships scoped as "geometry-conditioned view spaces beat the
   uniform viewsphere *on surface-reading QA*," which is also exactly Demo v2's construct, so the
   scope costs nothing this month.
2. **Author angle-insensitive mechanical items deliberately** (counts, presence/absence, color —
   all exact-match scorable) so the mechanical slice is not 100% reading tasks. This keeps the
   judge-free property without the selection effect.

Also, one mechanical nit with teeth: exact-string-match on OCR is brittle (case, whitespace,
hyphenation, partial reads). Define the normalization in the item schema, or "mechanical" quietly
becomes "someone eyeballs the near-misses" — a judge by the back door.

## 4. Attack: guards that exist elsewhere in round 5 but are missing from this part's spec

1. **The circularity guard on the angular mode and the oracle ceiling.** The per-region angular
   score and the oracle selector both need a normal. If that normal comes from the same fitted
   primitive that proposed the view, the treatment arm is partially self-scored (Branch 2 C4: the
   shared-assumption channel survives geometric GT). The fix exists — industry §5: score achieved
   orthogonality **against the raw cloud's locally-refit normals, never the fitted primitive that
   generated the view** — but it lives in one interpretation's benchmark section and is absent from
   the harness part as summarized. It must be in the harness spec, because it changes what the
   scoring code reads as input.
2. **Symmetry equivalence in view-correct scoring.** "View-correct" for the grounded 2×2 is
   undefined on symmetric primitives without a per-(object, region) equivalence set (Branch 2 Q8:
   only Dutagaci's Σ construction is usable; exclusion of ambiguous items — the
   `view-level-evidence-benchmark` escape — is structurally unavailable because symmetric primitives
   are our common case, not an edge case). Academia's pre-registration list has it; the harness part
   spec does not. Without it, the grid penalizes correct-but-symmetric evidence views and the
   grounded metric is noise on cylinders and near-cubes.
3. **Holdout is unpriced.** At the project's honestly-stated resolution (n≲100 episodes resolves
   15–20 points), reserving objects is expensive. Nothing states how many objects exist in total,
   how many are held out, or whether "untouched" means captured-day-one (recorder-version confound,
   §2.2) or designated-day-one-captured-later. One sentence in the spec fixes it; its absence is the
   kind of gap that gets resolved silently and favourably mid-month.
4. **Tier1↔Tier2 rank-correlation validation is underpowered as stated.** "Validate once on a small
   overlapping slice" computes a rank correlation over what — 4–6 arms? A Spearman ρ over n=5
   methods has essentially no resolution; it cannot distinguish 0.9 from 0.4. Rename it what it is —
   a directional sanity check (does the Tier-1 winner also win on Tier 2, yes/no, per comparison) —
   or it becomes a validity claim the harness cannot support.

## 5. Constraint compliance — checked item by item

- **Incremental partial-cloud fitting:** respected in intent; violated in the fine print of Tier 1
  (§2.1 — the reference-fit-at-capture-time is a complete-survey artifact serving a partial-fit
  loop). Acceptable for the *instrument* (dense capture as offline GT is precedented and is not the
  system), but the mismatch must be named and instrumented, not assumed away.
- **Real sensor defects:** in by construction — Tier 1 serves *real* captured RGB-D, so the fits and
  the images carry the real dropout/registration profile. Better than the field's own practice
  (Branch 2: zero geometric benchmarks inject defects). One inconsistency: academia's §4 losing
  condition references "the defect conditions" (injected sweeps) which the feedback scoped out;
  either drop that clause or scope it to the naturally-occurring defect strata.
- **Kinematic reachability:** respected, and elegantly — anything in the capture set was reached by
  the real arm to be captured at all, so Tier-1 action spaces are reachability-realistic for free.
- **Discrete LLM menu selection:** respected; the harness never exposes geometry, serves menus and
  frames only.
- **Benchmark-as-auxiliary:** under strain. Union capture + bracketing + snap columns + holdout +
  Tier1/Tier2 cross-validation + three tiers + strata is not an auxiliary workload. Academia's own
  sprawl warning applies to this part's own spec. The must-have core is: recorder (day one, feedback
  says so), Tier-1 replay executor for same-menu A/Bs, mechanical Tier-0 items, the grounded grid,
  floors/ceilings, paired design. Everything else (union bracketing, Tier-2 validation slice) should
  be explicitly gated on the headline experiment actually being reached in weeks 3–4.
- **ADAM regression:** none. Per-object *data* is collected; per-object *recipes* are not.
- **Scope (no VLA/RL/defect detection; geometry never to the LLM):** clean. The feedback's own
  "maybe sim and RL someday" stays a plug-in slot, not a build item.

## 6. Does it meet the architect's standard?

- **Falsification attempt:** mostly yes, and unusually well — the losing condition is real, the snap
  column and the Tier1/Tier2 check are instruments pointed at the harness's *own* validity, and E11's
  self-checks (replay determinism, travel-cost sanity, auditing every scored-wrong Tier-0 item) are
  genuine. Weak points: the snap column lacks a threshold (§2.3) and the rank-correlation check
  cannot fail informatively (§4.4).
- **Rendered evidence:** the harness produces it by construction (evidence frames are real captures;
  worst-case-per-object reported beside means — the exact anti-mean discipline Albert's complaint
  requires).
- **Concrete usage story:** present and specific — every representation A/B is a dataset query with
  seeds and permutations at zero robot cost; every application run records benchmark-grade data. The
  one usage story that is *oversold* is Tier 1 refereeing the space-vs-space headline (§2.1).

## 7. Fix list (ordered by what it protects)

1. Scope Tier 1's arbiter role in the spec: same-menu A/Bs only; the space-vs-space headline number
   comes from Tier 2 or from snap-tolerance-passing Tier-1 episodes. (Protects the month's one claim.)
2. Attach the decision rule to the snap column: tolerance = the measured legibility flat band from
   E6/E12; report the exceed-fraction per arm. (Turns honesty into a control.)
3. Add question-type stratification and a handful of angle-insensitive mechanical items; ship the
   headline scoped to surface-reading QA if the margin concentrates there. (Protects claim validity.)
4. Move the raw-cloud locally-refit-normal circularity guard and the per-region symmetry-equivalence
   set into the harness spec as scoring-code requirements. (Protects the angular mode and the oracle.)
5. Bracket-capture derived views + recipe-hash versioning + a stated recapture policy; state holdout
   size and whether it is captured or designated on day one. (Protects the dataset's shelf life.)
6. Name the trajectory-divergence effect of snapped depth in replay; decide whether replay refits on
   served depth (realistic, divergent) or on requested-pose synthetic fetch (impossible) — the first,
   obviously, but say it and let the snap column carry it. (Protects against a silent assumption.)
7. Downgrade "Tier1↔Tier2 rank correlation" to a per-comparison winner-agreement check. (Protects
   against an unsupportable validity claim.)

**Bottom line:** grounded, honest, and the right shape — the part fails not by inventing evidence
but by promising slightly more than its own instruments can deliver on the one comparison that
matters, and by leaving two known guards out of its own spec. Apply the fix list and this is the
soundest part of the round-5 methodology.
