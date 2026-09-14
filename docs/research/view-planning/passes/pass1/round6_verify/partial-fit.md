# Round 6 — Verification: `partial-fit` (primitive fitting on partial, growing clouds)

**Part under test:** anton.md §2 Part B (with tendrils into §1.2–1.3); cross-read against
academia.md (§1.2, §2.2, E1/E2/E8), industry.md (§1.4.1, Part B, E2–E4, E11), Branch 3
synthesis (`primitive-structure-from-survey-clouds.md`), Branch 1 synthesis (Nguyen local-normal
pattern), the architect's round-4 feedback (binding), and the framing doc.
**Role:** adversarial verifier. Default scepticism applied.
**Verdict: needs-work** — the best-grounded part of the whole design, with a genuinely
falsifiable kill condition in the right place, but the kill condition itself is mis-calibrated,
and the load-bearing experiment is underspecified in exactly the ways that would let it
confirm instead of prove.

---

## 1. Grounding audit — round-4 evidence or round-5 invention?

**Grounded, traceably.** Every component of the proposed stack maps to a specific round-4
finding, not to round-5 improvisation:

| Component | Source | Status |
|---|---|---|
| Open3D `detect_planar_patches` (bounded faces) | Branch 3 §3.3, §4.1 — verified accessor-level; "literally the scan-face-N tuple" | grounded |
| CGAL region-growing / PCL cylinder | Branch 3 §3.3 (CGAL Eff. RANSAC seedless/non-reproducible — verified; region growing deterministic) | grounded |
| Own cylinder-extent extraction | Branch 3 §7 RQ2 — "CGAL returns infinite primitives for all five types, verified accessor by accessor"; Kaiser's citable recipe | grounded |
| EMS cross-check | Branch 3 §3.1/§3.3 (91 ms, per-point posterior, reads .ply) | grounded, but see Issue 7 |
| Curvature guard vs RSPD cylinder-tiling | Branch 3 §4.1, §8.4 — acknowledged, unfixed failure inherited by Open3D | grounded |
| Wider-band (3ε) refit | Branch 3 §4.1 (Schnabel §4.6) | grounded **with a caveat Part B drops** — see Issue 6 |
| Locally re-fit normals near inspected region | Branch 1 §4.4 (Nguyen live-frame pattern), Branch 3 fused-offset thread | grounded |
| E2's target number (angular error of derived view vs coverage) | Branch 3 §3.2 ("no benchmark reports an angular error"), §6 item 1, §8.2 item 1 — confirmed absences | grounded as an absence claim, see Issue 8 |
| "Half-scanned cylinder is the common case" | The architect's own feedback (`view-set-selection` correction) + Branch 3 §2.2 ("Nobody tests a half-scanned cylinder with one-sided dropout") | grounded |
| Kill condition (fit error vs menu resolution) | New in round 5 — a legitimate synthesis move, but its threshold is wrong as stated (Issue 1) | invented, fixable |

The part is the round-4 measurement plan ("make angular error of the derived orthogonal
viewpoint the first number our benchmark produces — because it does not exist anywhere
else", Branch 3 §10) promoted to load-bearing status by the feedback's loop correction.
That promotion is exactly what the feedback demands. No scope violation: no VLA/RL, no
defect detection, everything internal to the geometry tier (LLM never sees any of it), no
per-object recipes (per-primitive deterministic rules only — blessed by the framing doc's
Non-Goals). The depth-adequacy check as scoped respects the demotion of the sensor
campaign. Constraint compliance is clean.

## 2. Falsification audit — does E2 prove, or only confirm?

The kill condition is real, cheap, and placed first — this is the architect's standard done
right in outline. But three defects in its current statement would let a losing fit pass:

**2.1 The threshold is ~2× too lenient.** "Fitted-normal error exceeds the menu's angular
resolution" reads as the 30° azimuth spacing of the 12×30°×3 grid. The correct break-even
is the **snap error** of the competing uniform set — the angular distance from the true
orthogonal to the *nearest* uniform view — which is bounded by the covering radius, not
the spacing. For the in-house 36-view grid that is ≲15–20°; for the budget-matched
Tammes/HEALPix control that both other interpretations (academia §1.6, industry §1.3.4)
and anton.md's own Part E declare mandatory, a matched budget of ~10–20 views on the
reachable hemisphere gives a covering radius around 15–25°, and the *expected* snap error
over random placements is well below that (~8–15°). Meanwhile the literature's ceilings
say the pass band is narrow: Efficient RANSAC is at 7.0–7.5° axis error on *clean complete*
CAD (SPFN Table 1), and partials can only be worse. A fit at 20° error "passes" the
30°-as-stated kill condition and still loses to the strong uniform control. The
parenthetical in anton.md E2 ("no closer to orthogonal than a uniform-sphere neighbour")
is the right criterion; the headline phrase is not, and the two are not equivalent.
As written, the experiment can confirm the design while the headline claim (Part G) dies
later, expensively — defeating the "cheapest possible place to find out" purpose.

**2.2 Two regimes are conflated.** Industry §1.4.1 makes the distinction Part B's kill
condition omits: early-iteration fits only need to *aim the next scan* (loose bar, the
loop self-corrects); only the evidence-time fit must beat the uniform snap error (tight
bar). Applied strictly at 10–20% coverage the kill condition falsely kills a workable
loop; applied as a mean over the sweep it falsely passes one whose evidence round is still
bad. "Distributions not means" helps but does not resolve which coverage fraction the
verdict is read at. The verdict must be conditioned on the coverage the loop actually
reaches by menu-derivation time — a number Part A's E5 produces and E2 currently doesn't
consume.

**2.3 How the partials are generated is unstated — and this is the trap the branch itself
condemned.** Branch 3's central finding is that every published robustness number is
measured under the wrong corruption model. A 10–100% coverage sweep needs controlled
partials; if these are made by geometrically masking a dense cloud, E2 reproduces the
field's hole-punching error and the resulting curve is not "loop-realistic" — the very
word the kill condition uses. Academia's E1 specifies the triad (per-view rigid
registration jitter injected *before* fusion; incidence-keyed dropout; systematic per-view
depth offset); anton.md's E2 does not. Real per-round accumulations from actual loop runs
are the only partials that fully carry the defect profile, and they don't give a smooth
10–100% sweep — both sources are needed and must be labelled.

**2.4 Ground truth is unaddressed.** Measuring angular error of the derived view requires
true normals/axes. Branch 3 §8.3 documents the circularity trap (GT built with the same
fitter cannot see the fitter's systematic bias) and records that real-scan GT is *named
unsolved future work by the field itself*. Industry's guard (score against the raw cloud's
locally re-fit normals) is a circularity guard for the *benchmark's* view scoring, but it
is not usable as E2's GT — those normals come from the same defective sensor and would
absorb exactly the systematic tilt E2 exists to catch (Branch 3 §6 item 5, "the single
most dangerous open question in the branch"). Also a measurement-floor problem: the demo
object is cardboard; a bulged cardboard face's "true normal" is ill-defined at the few-
degree level, which is the same order as the effect being measured. E2 needs
CAD-known/machined calibration objects and a stated GT uncertainty ≪ effect size. None of
this appears in Part B.

## 3. Component-level findings

**3.1 Wider-band refit — precedent misapplied.** Branch 3 §4.1 carries an explicit
caveat: the 5–25× reduction is measured in *position/scale*, "we cannot claim refitting
fixes angular error from this paper." Part B cites the precedent bare, inside a pipeline
whose kill condition is entirely angular. The refit is cheap and worth keeping — but its
angular effect must be measured in E2 (with/without arms), not implied by the citation.

**3.2 EMS cross-check — independence overstated, semantics undefined.** The cross-check's
value is redundancy, but both fitters are unvalidated on precisely the input class where a
cross-check is needed (one-sided directional dropout — EMS's 86.4%@20% is a *contiguous
synthetic cap*, a different partial), and a one-sided partial plausibly biases both toward
the observed side, so agreement can be shared bias, not correctness. Undefined: who wins
on disagreement; what happens on compound objects EMS's own family excludes (27–33% of
KIT/BigBIRD). Safe role: sanity vote whose disagreement triggers abstention to the
fallback tier; never a bias detector, never an adjudicator.

**3.3 Curvature guard — required, but currently untestable and possibly miscalibrated for
our scale.** The RSPD tiling failure was measured on architectural/industrial clouds
(Plant, Boiler Room); whether it fires at D405 point spacing on tabletop-radius cylinders
is unmeasured in either direction — the guard could be guarding a failure that doesn't
manifest, or missing one that manifests differently. No experiment in Part B exercises it.
Branch 3 §8.4 hands over the failure case for free; it should be an explicit E1/E2 case
(large-radius cylinder at our standoff) with the guard's false-positive rate on true
planes also measured.

**3.4 Slim depth-adequacy check — one dimension short.** As scoped ("fill rate on our
materials at scanning standoffs") it answers "can we fit at all" but not "which scan views
are admissible." The loop's *scan* captures are not orthogonal by construction — early
rounds capture at whatever incidence they get — and Part D's cylinder orbit spacing is
defined "by the incidence tolerance," a number that currently has no source except the
literature's 60–70° band, which Branch 1 established belongs to the wrong sensor classes
entirely. Both other interpretations independently added the same half-day fix (academia
E8, industry E11/G): one plate, fill/quality vs incidence, matte vs glossy. That is
within the "slim" scope — it serves the fitting loop, not evidence, and doesn't resurrect
the metrology campaign. Also: the check as stated has no go/no-go threshold, so it cannot
fail; industry's framing (go/no-go on the camera, exercised in week 1 while the swap
option is still live) is the falsifiable version.

**3.5 "The number that exists nowhere in the literature" — overclaims its own basis.**
The absence is confirmed only across the 20 of 39 map documents read; map sections E and F
(sensor + failure detection) were read not at all, find-superquadric (the closest
deployment match), Point2CAD, and the symmetry-completion paper are unread, and Wang
T-Mech 2024 — the closest full precedent — has never been read from its own tables. The
honest phrase is "nowhere in the corpus we read." Low risk that the number exists, but the
part's flagship novelty claim should carry its own provenance, per the project's own
evidential hygiene (academia §0).

**3.6 What survives scrutiny untouched.** The identity/matching layer being split into
Part C is correct scoping. Local normal re-fit near the region is the right response to
the 4 cm fused-offset hazard and has direct precedent (Nguyen). Preferring deterministic
detectors over seedless Efficient RANSAC is derived from a verified library property, not
taste. E1-first sequencing (two hours on real PLYs before more reading) is the single
best de-risking move in the month. The half-cylinder case being claim-critical rather than
demo-critical (demo v2 is a box) is fine *if* E2 stratifies box-face vs cylinder results —
per-stratum reporting is already Part E discipline; it just needs to reach into E2.

## 4. Architect's standard

- **Falsification attempt:** present and correctly placed, but mis-calibrated (Issues 1–3).
  Fixed as below, it is a genuine falsifier, not a confirmation ritual.
- **Rendered evidence:** not specified. Per-round overlays (partial cloud + fitted
  primitives + derived-view arrows + GT normal) are nearly free and are the artifact that
  makes an E2 verdict inspectable.
- **Concrete usage story:** intact — round-k partial → fit → derived orthogonal views →
  menu; demo box moved/rotated; fallback tier when the gate says no.
- **ADAM regression / scope:** clean. No geometry exposed to the LLM anywhere in the part.

## 5. Issues (ranked)

1. **Kill-condition threshold too lenient (~2×).** "Menu's angular resolution" (30°
   bin spacing) is not the break-even; the break-even is the snap-error distribution of
   the budget-matched *strong* uniform control (covering radius ~15–25°, expected snap
   ~8–15°). With Efficient RANSAC at 7°+ on clean complete CAD, the pass band is narrow;
   as stated the experiment can pass a fit that loses the headline comparison.
2. **Two fit regimes conflated.** Exploration fits (aim the next scan) and evidence-time
   fits (beat uniform) need separate bars; the verdict must be read at the coverage the
   loop actually reaches by menu-derivation time, not at an arbitrary sweep point or mean.
3. **Partial-generation method unstated.** Synthetic masking without the defect triad
   (pre-fusion registration jitter, incidence-keyed dropout, per-view offset) reproduces
   the exact noise-model flaw Branch 3 condemned; "loop-realistic" is then unearned.
4. **Ground truth unaddressed.** Circularity trap (Branch 3 §8.3) unhandled; locally
   re-fit sensor normals are not valid GT for E2 (they absorb the systematic tilt E2 must
   catch); cardboard's GT-normal floor is at the effect-size scale.
5. **Slim depth-adequacy check missing the incidence dimension and a go/no-go
   threshold** — the loop's scan-admissibility gate and the cylinder orbit spacing
   otherwise rest on a 60° number from wrong sensor classes, and the check cannot fail.
6. **Wider-band refit precedent misapplied:** measured for position/scale only; Branch 3's
   own caveat ("we cannot claim refitting fixes angular error") is dropped in Part B.
7. **EMS cross-check independence overstated, semantics undefined:** shared-bias risk on
   one-sided partials; no disagreement-resolution rule; EMS family excludes compounds.
8. **"Exists nowhere in the literature" overstates provenance:** 20/39 documents read,
   sections E/F unread, closest precedent (Wang T-Mech) never read primary.
9. **Curvature guard has no test:** its firing regime at D405 scale is unverified in both
   directions; no false-positive measurement on true planes; Branch 3 §8.4's free failure
   case unused.

## 6. Suggestions

1. Operationalize the kill condition as a **paired comparison of distributions**: per
   placement, derived-view angular error vs snap error of (a) the in-house 12×30°×3 grid
   and (b) budget-matched Tammes/HEALPix; report median and 90th percentile; the losing
   condition is "not stochastically better than (b)."
2. Split the bar: exploration bar (error small enough that the next scan lands on the
   intended surface — derive from FOV footprint) vs evidence bar (beats (b)'s snap error).
   Have E2 consume E5's per-round coverage trajectory and report *the round at which the
   evidence bar is first met*, per object class.
3. Generate partials three ways, all reported: real per-round loop accumulations
   (primary), real single views, and synthetic sweeps corrupted with the triad — never
   plain masking.
4. E2 GT from CAD-known/machined calibration objects (box + cylinder), with stated GT
   uncertainty; keep the demo cardboard box for the depth-adequacy check only.
5. Fold the half-day incidence sweep (one plate, matte vs glossy) into the slim check;
   outputs: provisional gate τ, orbit-spacing tolerance, and a written go/no-go threshold
   (fill ≥ X% and plane fit within Y° on demo materials) so the check can fail while the
   camera swap is still cheap.
6. Keep the wider-band refit but give it with/without arms in E2 — measure its angular
   effect instead of citing the position/scale number.
7. Pin the EMS role: sanity vote; disagreement → abstain to fallback tier; agreement never
   claimed as absence of bias (only E2's GT comparison can rule on bias).
8. Add the RSPD tiling case explicitly to E1/E2 (large-radius cylinder at our point
   spacing) and measure the curvature guard's false-accept and false-reject.
9. Render per-round fit overlays as standing artifacts — the rendered-evidence half of
   the architect's standard, currently unmentioned in Part B.
10. Rephrase the novelty claim as "absent from the 20-document corpus read; Wang T-Mech
    primary text and map sections E/F remain the audit debt" — consistent with the
    reading-debts list both interpretations already carry.

## 7. Bottom line

The part is not invented — it is the round-4 branch's own measurement plan, correctly
promoted by the feedback's loop correction, built from verified library facts, and
centred on a number whose absence the corpus confirms. It respects every architectural
constraint and contains the month's single best falsification lever. But the lever's
calibration is wrong in the lenient direction, the experiment's inputs (partials, GT) are
unspecified at exactly the points where the branch itself documented the field's failure,
and two cheap checks (incidence sweep, guard test) are missing. Fix the threshold, split
the regimes, specify partial generation and ground truth — then this part survives the
architect's standard and deserves its "load-bearing wall" title.
