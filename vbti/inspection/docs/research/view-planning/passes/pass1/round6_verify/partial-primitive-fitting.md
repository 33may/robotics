# Round 6 — Adversarial verification: `partial-primitive-fitting`

**Part under test:** geometry from defective partial clouds — Open3D `detect_planar_patches`
(bounded faces = the action tuple) + CGAL/PCL cylinder + own extent extraction + wider-band
refit + curvature guard against RSPD's plane-tiling-on-cylinders; the identity layer
(geometric-content addressing in the table/gravity frame, Hungarian matching across refits,
deterministic detectors preferred); fallback tier to the uniform sphere with explicit
abstention; acceptance bar and fitter choice settled by the score→error calibration
experiment (academia E1/E4, anton E2/E4, industry E2–E4).

**Inputs verified against:** round-4 feedback (binding), all three round-5 interpretations,
Branch 3 synthesis (`primitive-structure-from-survey-clouds.md`), framing doc. Plus two
checks run on this machine, reported in §1.

**Verdict: needs-work.** The part is genuinely grounded, respects every scope constraint,
and its disproof conditions are real. But as written, several of its experiments would
produce curves without pass/fail meaning — the ground-truth protocol, the gate's feature
set, and two acceptance thresholds are missing in ways that would let the experiments
*confirm* the design rather than falsify it. Each gap is closable; none is fatal.

---

## 1. Provenance audit — grounded or invented?

### 1.1 Grounded, and independently re-verified where cheap

- **The stack is Branch 3's stack, not a round-5 invention.** Every component traces to the
  synthesis's verified tooling table (§3.3): bounded faces only from Open3D; CGAL infinite
  primitives verified accessor-by-accessor; extent extraction ours by necessity; Schnabel
  §4.6 wider-band refit; RSPD §5 cylinder-tiling failure acknowledged by its own authors.
- **Re-verified on this machine:** Open3D 0.19 in the `robo` env ships
  `detect_planar_patches → list[OrientedBoundingBox]` with the z-column of `.R` as the
  patch normal and `.center`/`.extent` populated — the "bounded face = action tuple" claim
  holds against the installed library, not just the docs.
- **The loop premise is materially satisfied.** 27 real runs exist under
  `inspection/data/runs/*/` with `fused_cloud.npy` **and per-view capture directories
  preserved**. The two-hour shakedown has its input today, and — important for the
  corruption suite — registration jitter *can* be injected before fusion, because the
  per-view data the suite needs was actually recorded. This premise is often asserted and
  false in such plans; here it checks out.
- **The honest-absence framing is accurate.** The synthesis's six confirmed absences (no
  angular-error benchmark, no re-scan repeatability protocol, no score→failure calibration,
  no detector uncertainty, no partial-cylinder-with-dropout test, no registration-covariance
  propagation) are carried into round 5 without inflation. Nobody claims literature support
  that doesn't exist.

### 1.2 Invented in round 5 — flagged as such, mostly legitimately

- **Identity layer as an inference-time device.** SPFN's RIoU + Hungarian is a
  *training-time* matcher; using it across refits is an extrapolation the synthesis itself
  labels "a layer we write." Legitimate, carries its own experiment (E3).
- **Fit-maturity gate** ("coarse fits aim the next scan; evidence capture only from mature
  fits" — industry §1.4.1). Pure invention, no literature, and a *good* one: it lowers the
  accuracy bar for exactly the low-coverage rounds where the literature is silent. It is
  also the part's quiet scope-saver and should be promoted to first-class design, not left
  as one persona's aside.
- **Hysteresis on labels, menu freezing after iteration k, per-region fallback.** All
  round-5 inventions with zero literature cover. Acceptable as design responses — but the
  last one is an unreconciled fork (§3.3).
- **The fallback tier's gate design has *zero read literature* behind it.** Map section F
  (failure detection / introspection, 2 documents) was never read; PEARL's ∅-label is a
  per-point mechanism the synthesis explicitly says does not extend to a per-object
  decision. The gate is the least-grounded element of the entire part — which is precisely
  why E4's calibration must be able to fail (§3.2 says it currently can't fail informatively).

**Net:** nothing here is a round-5 fabrication dressed as literature. The inventions are
confessed inventions with experiments attached — with the specific exceptions in §3.

---

## 2. Constraint compliance — passes

- **Incremental partial-cloud fitting:** the part *is* the feedback's correction made
  concrete; the half-scanned-cylinder case is treated as the common case, per the feedback.
- **Real sensor defects:** the corruption triad (per-view rigid jitter before fusion,
  incidence-keyed dropout, systematic per-view offset) is structurally different from every
  published suite and matches the framing doc's defect list. The ~4 cm offset is handled by
  the local-refit-near-region rule rather than wished away.
- **Geometry never exposed to the LLM:** internal addresses stay internal in all three
  interpretations; the abstention/fallback surfaces as a *menu item*, which is a label, not
  geometry. Compliant.
- **No ADAM regression:** all rules are per-primitive (face, orbit), none per-object; the
  demo's robustness to re-placement comes from refit + content-based finding, not a recipe.
- **No VLA/RL/defect detection:** absent. The deliberately-absent list (academia §3) is
  correct.
- **Benchmark-as-auxiliary:** E1–E4 are part-level instrumentation, not benchmark sprawl —
  though see §3.7 on effort sizing.
- **Reachability:** correctly out of this part (menu-crafting owns it); the part only
  promises fit outputs that reachability filtering consumes.

---

## 3. Where it breaks — issues, most severe first

### 3.1 The load-bearing number has no ground-truth protocol, and the known circularity trap is unaddressed

Everything in this part terminates in one measurement: **angular error of the derived
orthogonal viewpoint against truth** (E1/E2/E4). All three interpretations say "CAD or
metrology-grade reference scan" and stop. That is a hand-wave over the part's own warning:
Branch 3 §8.3 documents the field's GT circularity (Araújo & Oliveira fit their GT planes
with their own detector's recipe — a benchmark built that way *cannot see its fitter's
systematic bias*), and both SHREC2022 and SuperFlex name real-scan GT as unsolved on their
own roadmaps. The specific hard part is not the object's shape (a box is caliper-measurable)
— it is the **true normal direction in the robot frame** for the placed object. The top
face is free (gravity + table plane). Vertical side faces need an azimuth reference; a
tilted or cylindrical surface needs more. If GT normals are obtained by "a really careful
scan with the same camera and the same fusion pipeline," E1 measures the fitter against
itself and the systematic-bias axis — the one axis the whole part exists to catch — is
invisible by construction. **Fix:** specify the GT protocol before E1 is scored: gravity +
table frame for horizontal faces; printed fiducials on faces, or robot touch-off (the
freedrive-tap machinery already in this repo is exactly this instrument) for verticals and
the cylinder axis. Without this, E1 confirms; it cannot falsify.

### 3.2 The gate calibration feeds almost entirely on features the part itself declares pre-doomed

The part's sharpest literature finding: **every off-the-shelf confidence signal is
residual-family and structurally blind to systematic bias** (Branch 3 §1.3, C9: parameter
error 1e11 at low residual). Yet the feature list E4 plans to calibrate — coverage,
residual RMS, MND, EMS posterior, radius plausibility — is *that same residual family*. If
the score→error mapping comes out flat, that is the predicted outcome, not a finding; and
"disproves residual-only gating" was already disproved by the literature review. Meanwhile
the two genuinely non-residual signals the design already computes elsewhere are never
named as gate candidates: **(a) cross-fitter disagreement** (Open3D patch vs CGAL/PCL
cylinder vs EMS cross-check — independent estimators disagreeing is exactly what systematic
corruption produces), and **(b) cross-round parameter dispersion** (E3's own metric —
a fit that moves every refit is not mature). **Fix:** put both in E4's candidate feature
set explicitly. Only then does E4 have a live hypothesis: "some computable signal separates
good fits from bad" — and a genuine disproof condition if *nothing* separates them
(industry's "the fallback tier triggers blind and the design needs rethink" — the one
persona that states the real kill condition).

### 3.3 The fallback granularity fork is unreconciled, and it contaminates the headline experiment

Three documents, three fallbacks: academia degrades **per-region** ("uniform sphere for
that region"); anton reads as **whole-menu** ("the uniform tier serves instead"); industry's
S0 is the fallback **action space**. This is not bookkeeping. Per-region fallback produces
a *hybrid* menu that partially contains the uniform baseline — so in E8/E6's headline 2×2,
the "intelligent viewsphere" arm converges toward the control exactly when the gate
triggers often, diluting the one comparative claim; and the abstention item means different
things ("this face is unscannable" vs "geometry has failed, reverting"). It also changes
what E10 (abstention correctness) scores. **Fix:** the architect decides the granularity
before Part D is built, and the benchmark reports gate-trigger rate per arm so hybrid
contamination is visible in every table.

### 3.4 The disproof threshold depends on an experiment scheduled after the experiment it gates

E1/E2's kill condition — "fitted-normal error exceeds the menu's angular resolution / the
evidence tolerance band" — is well-posed only once the tolerance band exists, and the band
comes from the Tier-0 legibility-vs-incidence curve (anton's E6, scheduled with the demo;
industry's E12, correctly flagged as "cheapest experiment with the most direct effect").
As sequenced by the architect, E1 runs immediately but cannot be *judged* until weeks
later. **Fix:** run the code-reading-vs-angle curve in week 1 alongside E1 — it is a
tripod, a printed code, and an afternoon — so E1's distributions land against a defined
bar. Otherwise the "cheapest possible place to find out the claim dies" produces data that
cannot kill anything yet.

### 3.5 Identity churn is measured but never linked to failure — and is ill-posed for cylinders

E3 reports identity flip rate, but no experiment maps flip rate → task damage, so the
acceptance bar ("if flips are common") is undefined. Worse, the feared consequence is
argued from NBV-Net's 0.988→0.472 collapse — a *trained* selector facing a *silently*
shrunken action space — while this design already announces menu diffs to the LLM, which
plausibly makes moderate churn harmless. Either the diff-announcement mitigation works (then
strict identity stability is over-engineered) or it doesn't (then E3's bar must be near
zero) — nothing planned distinguishes these. Two protocol defects besides: (a) Hungarian
matching per consecutive round permits *drift* — "the code face" can migrate label-free
across many small refits; E3 must score identity against a round-1 anchor, not just
round-to-round. (b) For cylinder orbit stations identity is not even well-defined: the
azimuth origin is an artifact of the fit, rotates freely between refits, and there is no GT
identity to flip against. **Fix:** add a cheap Tier-1 probe — induced churn with vs without
diff announcements — to set E3's bar from downstream effect; anchor E3 to round 1; define
an azimuth-anchoring convention (scene-fixed direction, not fit-derived) before orbit
identity is scored.

### 3.6 The curvature guard is named in every document and designed in none

"Curvature-consistency guard" appears as a solved item, but no interpretation specifies
what it computes, its threshold, or its falsification case. It must pass a *two-sided*
test: a large-radius cylinder yields zero spurious planar faces in the menu, **and** a
slightly bowed real cardboard face (the demo object; cardboard bulges) is *not* suppressed
as "curved." A guard tuned only on the first case silently deletes true faces — a failure
mode worse than the tiling it prevents, because tiling produces visibly absurd menus while
over-suppression produces quietly incomplete ones. It also inherits the part's own problem:
curvature estimates on partial defective clouds are exactly as untrustworthy as the normals
the part refuses to trust. **Fix:** the guard gets its own named accept/reject pair inside
E1 (cylinder true-positive suppression rate; bowed-box false-suppression rate).

### 3.7 Smaller findings

- **Claim hygiene on the wider-band refit:** the 5–25× precedent is *position/scale*
  (sphere radius/centre); Branch 3 explicitly warns it cannot be claimed for orientation.
  Anton's Part B cites it uncaveated inside a part whose load-bearing quantity is angular.
  One clause fixes it; the architect's standard is exactness.
- **Runtime-computable features only:** E1's sweep axis (surface-coverage fraction)
  is not knowable at runtime for planes — the patch extent is defined *by* the observed
  points, so "coverage of the face" is circular; cylinder arc-coverage *is* computable
  (fitted radius gives the full circumference). The gate must be calibrated on features
  the deployed system can evaluate, or E4's mapping never deploys.
- **Effort sizing:** the full E1 spec (10–100% coverage × three defect axes × per fitter ×
  distributions + FA/FR curves) plus E2 + E3 + E4 is a mini research program inside a month
  whose benchmark is "auxiliary." The fit-maturity gate justifies the honest cut: calibrate
  only in the coverage regime the loop actually visits (first-round partials ~15–40%, per
  academia) on the two demo-class shapes, and stop.
- **Rendered evidence not mandated:** the architect's standard is falsification + rendered
  evidence + usage story. E1–E4 as specced emit curves and rates but no images. Mandate
  per-case renders (fitted primitive + derived view arrow overlaid on the cloud, worst
  cases first) as required artifacts — nearly free and they are what catches the failure a
  distribution hides.
- **The cylinder half of the part has no usage story this month.** Demo v2 is a box; the
  cylinder axis fitter, arc extent, orbit stations, and the guard serve only the benchmark
  objects. Legitimate (the framing doc's target behaviour includes the cylinder orbit), but
  the plan should name the demo-critical path explicitly: if the month compresses, the
  box-face path alone carries demo v2, and the cylinder stack is the cut line.

---

## 4. Do the experiments prove/disprove, or only confirm?

As designed **in intent**, they disprove — the kill conditions are real sentences with real
consequences (E1: "the core claim dies by construction"; E3: interface degrades to
anonymous menus; E4: fallback triggers blind). As currently **specified**, three of four
cannot fail informatively: E1 lacks a bias-proof GT protocol (§3.1) and a defined tolerance
bar (§3.4); E4's feature set predetermines its outcome (§3.2); E3 lacks an acceptance
threshold connected to anything downstream (§3.5). E2's shakedown-on-real-clouds-first
ordering is the one unambiguously well-designed piece and should run exactly as written.

## 5. What survives scrutiny

The part passes the checks most such plans fail: it is built from a verified tooling read
(re-verified here against the installed library), its literature absences are stated
rather than papered over, its inventions are confessed and instrumented, the per-view data
its corruption suite requires actually exists on disk, the determinism caveat ("necessary
but not sufficient") is carried correctly, and it violates none of the architect's scope
lines — no geometry to the LLM, no per-object recipes, no defect detection, benchmark kept
in the instrument role. The verdict is needs-work, not broken: every issue above is a
protocol specification, a decision, or a one-line correction — nothing requires new
architecture.
