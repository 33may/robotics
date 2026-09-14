# Round 6 — Verification: `partial-fit-identity`

**Part under review:** Partial fitting and identity — the load-bearing wall. Two-hour
library shakedown (Open3D `detect_planar_patches` + CGAL/PCL cylinder + EMS) on real
fused PLYs, then three measurements: (1) fit error vs coverage fraction incl. the
half-cylinder case; (2) refit churn / identity-flip rate across loop iterations with a
geometric-content matching layer and hysteresis; (3) fit-score → derived-view
angular-error calibration yielding a fit-maturity gate with measured
false-accept/false-reject rates. Plus the curvature guard against the documented
cylinder-tiling false positive.

**Role:** adversarial verifier. **Verdict: needs-work** — the part is genuinely
grounded and genuinely falsifiable, but two of its three headline measurements are
underspecified in ways that would let them silently pass while wrong, and the trap is
documented in the part's own inputs.

---

## 1. Grounding audit — literature or round-5 invention?

Traced element by element against Branch 3 (`primitive-structure-from-survey-clouds.md`,
round 4) and the architect's feedback:

| Element | Provenance | Verdict |
|---|---|---|
| Two-hour shakedown on real fused PLYs | Branch 3 §9.5, near-verbatim ("Run the two-hour empirical check… Open3D `detect_planar_patches` + CGAL/PCL cylinder on one of our actual fused PLYs, and EMS") | **Grounded** |
| Fit error vs coverage, angular error of derived viewpoint | Branch 3 §3.2.2 (no benchmark reports angular error — field-wide reporting gap), §6.1, §8.2.1, §10 ("make 'angular error of the derived orthogonal viewpoint' the first number our benchmark produces") | **Grounded** |
| Half-cylinder centrality | Architect's own feedback (loop correction: "Branch 3's partial-shape fitting (the least-supported case: half-scanned cylinder) become central rather than edge cases") + Branch 3 §2.2 ("Nobody tests a half-scanned cylinder with one-sided dropout") | **Grounded, feedback-driven** |
| Identity by geometric content, RIoU + Hungarian matching, deterministic-detector preference | Branch 3 §7-RQ3 design consequences 1–4, derived from documented library properties (CGAL Efficient RANSAC seedless and randomized — verified accessor-level) | **Grounded** |
| Re-scan/refit repeatability protocol | Branch 3 §3.2.4 + §8.2.3 (zero published protocols; explicitly proposed as our experiment) | **Grounded** |
| Score→error calibration with FA/FR rates | Branch 3 §8.2.4, verbatim ("a threshold with a measured false-accept/false-reject rate. Without this the two-tier fallback gate is a guess") | **Grounded** |
| Curvature guard | Branch 3 §4.1/§8.4 (RSPD §5's acknowledged, unfixed tiling failure; measured cost: Plant precision 0.63, Boiler recall 0.60) | **Grounded** |
| **Hysteresis** | Appears only in round-5 industry interpretation (§1.4.2, Part B). Nowhere in round 4. | **Round-5 invention** — acceptable engineering, no literature cover, and it ships with its own on/off measurement. But see Issue 2: it creates a failure mode the experiment as designed cannot see. |

So the part is *not* invented in round 5 — it is the most literature-derived part of
the whole methodology, in the negative sense: every one of its three measurements is
a confirmed absence in the read corpus, and two (angular-error reporting, uncertainty
output) are survey-backed absences, not just per-paper ones. One caveat on the
novelty rhetoric: the absence census comes from 20 of 39 mapped documents, with
sections E/F entirely unread and the two closest unread candidates named by Branch 3
itself — `find-superquadric` (#20, partial cloud + table prior, "the closest
deployment match in the whole map") and robust extrinsic symmetry (#35, the only
partial-cylinder axis-completion candidate). "No paper has run this" is safe for the
three measurements; "nothing close exists for partial fitting on a table" is not yet
safe to say.

## 2. Falsification audit — prove/disprove, or only confirm?

The disproof conditions are real, and correctly placed at the cheapest point:

- **E2 (coverage sweep)** carries the strongest kill condition in the month: if
  fitted-normal error on loop-realistic partial clouds exceeds the menu's angular
  resolution, the geometry-conditioned menu cannot beat uniform *by construction* and
  the core claim dies before any LLM experiment runs. That is a genuine falsification
  attempt, not confirmation theatre. (One nit: "the menu's angular resolution" needs a
  number — presumably the nearest-neighbour spacing of the budget-matched uniform
  control, ≈15° for the current 12×30° grid — written down before the sweep runs.)
- **E3 (churn)** disproves stable addressing; the fallback (per-round anonymous menus)
  is named and priced as materially worse.
- **E4 (calibration)** can disprove residual-only gating (flat mapping), which
  Branch 3 §1.3/§5-C9 predicts is a live possibility (SHREC's 1e11 parameter error at
  low residual). The design does not assume the gate exists; it measures whether any
  score separates good from bad. Correct posture.

But three specification gaps would let two of the experiments pass while wrong:

**Issue 1 — the reference is unnamed, and the circularity trap is documented in the
part's own inputs.** All three measurements score angular error "against reference"
(industry Part B) without saying where reference normals/axes come from. Branch 3
§8.3 flags exactly this: Araújo & Oliveira build GT by fitting hand-selected inliers
*with the same recipe their detector uses*, and "a benchmark built that way cannot
detect its own fitter's systematic bias — which is precisely the failure mode we most
need to catch." Our defect triad's most dangerous member (systematic bias — the ~4 cm
fused-offset class) is invisible to residual scores *and* to a reference built by
fitting the same cloud with the same stack. If the E2/E4 reference is "fit the full
cloud, call it truth," the load-bearing wall is certified by a circular instrument.
The academic interpretation says "CAD or metrology-grade reference scan of the
calibration subset" but the part as scoped ("real fused PLYs") never commits. For the
demo box and a cylinder this is cheaply fixable (measured CAD + pose from
table-plane + fiducials/manual correspondence), but it must be *stated*, because
this is the number everything downstream trusts.

**Issue 2 — the identity metric is gameable under hysteresis, and the dangerous
direction is unmeasured.** Flip rate after matching, with hysteresis on, is minimized
trivially by freezing labels — an aggressive hysteresis band produces a beautiful
churn number and a menu whose "code face" can quietly stay attached to the wrong
physical face after a large refit correction (e.g. the curvature guard dissolving a
tiled pseudo-plane into a cylinder — a *legitimate* identity death that hysteresis
would suppress). Stable-and-wrong is worse than unstable for demo v2: the failure is
an orthogonal evidence image of the wrong face, scored confident. E3 as designed
measures stability, not correctness. It needs ground-truth correspondence (known
re-placement poses, fiducials, or manual annotation of which physical face each
detection is) and a **wrong-attachment rate** reported beside the flip rate; the
hysteresis on/off comparison then trades those two numbers against each other
explicitly instead of optimizing the visible one.

**Issue 3 — the calibration measures its FA/FR on the distribution it was tuned on.**
E4 derives the threshold and its false-accept/false-reject rates from the same
corrupted-cloud corpus, with no calibration/validation split named anywhere in the
part (the day-one capture holdout exists in the benchmark part, but E4 predates the
harness in the build order). Worse, the *deployed* FA/FR depends on the mix of
coverage fractions and defect levels the loop actually encounters, which is unknown
before the loop runs — a chicken-and-egg the part doesn't acknowledge. Cheap fix:
split by object (calibrate on some, report FA/FR on others), and recalibrate from
recorder logs once real episodes exist.

## 3. Constraint audit

- **Incremental partial-cloud fitting:** central to the part; the coverage sweep and
  churn protocol exist *because* of the loop correction. Compliant.
- **Real sensor defects:** the shakedown and sweep run on real fused PLYs. One
  tension: a dose-response sweep (10–100% coverage; controlled dropout) needs
  controllable inputs, and the academic interpretation's answer is a synthetic
  corruption triad (registration jitter, incidence-keyed dropout, systematic offset).
  That drifts toward the corruption-suite research program adjacent to what the
  architect scoped out ("slim check", no metrology campaign). The lean version —
  sweep coverage by including/excluding *real* captured views, in loop-realistic
  accumulation order — stays on real defects and needs no simulator. Flagged as
  scope-creep risk, not violation.
- **Coverage as an independent variable is confounded** (view count × view geometry ×
  where the dropout sits): 30% coverage from one frontal view is not 30% from two
  oblique views. Anton's "distributions not means" and "accumulated multi-round
  clouds" partially cover this; the sweep should be stratified by view configuration,
  and the clouds should come from reachability-constrained captures (a turntable-style
  sweep would include coverage states the real arm can never produce — a quiet
  kinematic-reachability leak into the calibration).
- **Discrete LLM menu / geometry never exposed:** compliant — the identity layer is
  exactly what keeps internal addresses internal while the menu shows stable names;
  nothing in the part leaks primitives to the LLM.
- **ADAM regression:** none. Everything is per-primitive and object-agnostic;
  identity-by-geometric-content is the opposite of a per-object recipe.
- **Benchmark-as-auxiliary:** the three measurements are build-gating instruments,
  not benchmark sprawl. Compliant, with the §3 scope-creep caveat.
- **No VLA/RL/defect detection:** untouched. Compliant.

## 4. Smaller findings

- **The curvature guard has no test of its own.** The FA/FR discipline applied to
  the maturity gate is not applied to the guard: does it reject true-but-noisy
  planes (curvature estimation on partial defective clouds is itself noisy)? And the
  documented failure is for *large-radius* cylinders — if the test objects are
  small-radius cans, the guard sits unvalidated while being claimed as day-one
  protection. Add one large-radius/chamfered case to the shakedown set.
- **Hysteresis concentrates churn at the worst moment.** A fit improving past the
  hysteresis band relabels exactly when it crosses the maturity gate — i.e. churn
  peaks when the menu starts to matter. Report churn conditioned on gate state, not
  just per round.
- **Gate role conflation.** The design uses the calibrated gate for two different
  decisions with different costs: fallback-to-uniform (per fit) and
  exploration-vs-evidence-capture (fit maturity). One score→error curve can serve
  both at different thresholds, but each threshold's FA/FR should be reported against
  its own decision cost.
- **Camera flexibility invalidates the curves.** The calibration and gate threshold
  are camera-specific; the feedback keeps the camera swappable and demo v2 may
  trigger a swap. State that a swap re-runs E2/E4 (they're cheap), so gate values
  don't silently survive a hardware change.
- **Rendered evidence is implied, never required.** The architect's standard asks
  for rendered evidence; the part's natural artefacts (fitted primitives overlaid on
  the partial cloud per round; identity matches coloured across rounds; the gate's
  accept/reject cases as images) are exactly what would make E2/E3 auditable and are
  one line to mandate.
- **Determinism is oversold in one respect and the part knows it elsewhere:** CGAL
  region growing is deterministic *given identical input ordering*, but the loop's
  accumulating cloud changes ordering every round by construction — so determinism
  contributes nothing across rounds and identity rests entirely on the matching
  layer. Branch 3 says this ("necessary but not sufficient"); the part's summary
  phrasing ("deterministic detectors preferred") should not be read as mitigation
  for cross-round identity.

## 5. Would it survive the architect's standard?

- **Falsification attempt:** yes — E2's kill condition is the genuine article and is
  sequenced first for exactly that reason.
- **Rendered evidence:** implied, not required (see above). Fixable in one line.
- **Concrete usage story:** yes and specific — the gate decides when the intelligent
  viewsphere abstains and the uniform tier serves; identity keeps "the code face"
  addressable across refits, which demo v2's move-and-rotate protocol directly
  exercises.

## 6. Verdict

**needs-work.** The part is the best-grounded piece of the round-5 methodology — its
three measurements are confirmed absences in the literature, its falsification
conditions are real, and it neither regresses to ADAM nor violates any scope rule.
But it is the load-bearing wall, and as specified it can be certified by a circular
reference (Issue 1) and can report a stability number that hides the one identity
failure that would wreck demo v2 (Issue 2), with a calibration whose error rates are
tuned-and-tested on the same corpus (Issue 3). All three fixes are cheap,
specification-level, and must land before the wall bears load.
