# Round 7 — Completeness verification (whole pass)

**Role:** completeness critic over the entire pass 1 (rounds 1–6).
**Inputs read:** framing doc · round-4 feedback (binding, overrides framing) · round-1 plan ·
all six round-4 accumulations (coverage sections + gap inventories) · all three round-5
interpretations · all 16 round-6 verifications · round-2/3 file inventory.
**Question asked:** what did the pass NOT do — unanswered questions, unverified load-bearing
claims, unread high-priority documents, methodology parts with no experiment, benchmark blind
spots, and gaps between the methodology and the demo-v2 goal. This is the work-list for pass 2.

**Verdict: needs-work.** The pass is unusually honest about its own coverage (every round-4
synthesis audits what it did not read; every round-6 verifier attacks its own part), and the
round-6 fix lists are high quality. What is missing is not rigor inside the parts — it is
(a) a consolidation layer that does not exist, (b) a reading debt that is larger than the one
round 5 recorded, (c) three design elements with no experiment anywhere, and (d) two questions
the QA reframe moved rather than closed. All items below are enumerable and none invalidates
the month plan.

---

## 1. Structural gaps — the pass has no spine yet

### 1.1 Three methodologies, three colliding experiment numberings, no reconciliation
Round 5 produced three parallel build plans with different part names for the same components
(loop-skeleton / loop-scaffold; partial-fit / partial-primitive-fitting / partial-fit-identity;
menu-crafting / view-crafting / intelligent-viewsphere; …) and **three independent E-numberings
that collide**: "E1" is the two-hour fit shakedown (anton), the partial-fit fidelity curve
(academia), and the loop smoke test (industry) — three different experiments under one label.
Round 6 then verified 16 part-documents (all "needs-work") with fix lists that partially
overlap and partially conflict. **Nobody merged any of this.** Until a single canonical
part list + experiment list exists, "did every methodology part get an experiment" is not even
a well-posed question, and the round-6 fixes cannot be tracked to adoption. This is pass 2's
first deliverable.

### 1.2 Unresolved forks that need an architect decision (found, not decided)
- **Pre-registration.** Anton §1.6 explicitly drops it ("pre-registration theatre… What I
  drop"); academia §4 specifies a full pre-registration; the round-6 verifiers lean on
  pre-registered validity gates ~35 times, and `headline-comparison` Issue 7 makes the losing
  conditions *depend* on them ("a losing condition only counts when its validity gates
  passed"). The architect's stated preference and the verified methodology are in direct
  contradiction. A minimal resolution exists (one versioned file freezing: thresholds, arms,
  budgets in all currencies, the oracle definition, the losing conditions — no α rituals),
  but someone must rule.
- **Stop-rule design.** Three stop machines across the three interpretations
  (`loop-skeleton` Issue 1); the two-phase variant quietly re-imports survey-first. Flagged,
  not decided.
- **Fallback tier space.** Anton/academia: Tammes/HEALPix equal-area sphere; industry S0
  wording keeps the naive {h,v} grid as fallback. Plus the fallback *granularity* fork
  (`partial-primitive-fitting` §3.3, "unreconciled… contaminates the headline experiment").
- **E2 kill-condition threshold.** `partial-fit` Issue 1 showed "menu angular resolution"
  is ~2× too lenient vs the snap-error break-even of the strong uniform control. The fix is
  written; nothing marks it adopted.

### 1.3 The contract is stale and the demo target has no paper trail
- The framing doc still carries the falsified sensor story (active-stereo/projector — refuted
  by the datasheet read), the survey-first given ("the geometry is a survey cloud"), and the
  defect-flavoured inspection framing. The feedback overrides all three but the contract was
  never revised; pass 2 readers citing "the contract" will cite dead premises.
- **Demo v2 ("box with codes, robust to move/rotate") appears in no round-4 document on
  disk** (`demo-v2-acceptance` §1 provenance note — grep confirms zero hits in feedback.md
  and all six syntheses). It is binding only via the orchestrator scope statement. Write it
  into the architect's feedback file or the contract, or a later reader cannot distinguish
  architect intent from round-5 drift.

---

## 2. Unread high-priority documents — the real reading debt

Round 5 (academia §5) recorded a debt of five items. The round-4 coverage audits show the
actual debt is roughly three times larger, and several unlisted items sit under load-bearing
design decisions:

**On the surviving debt list (correctly carried):**
1. **Wang et al., T-Mech 2024** — the closest full precedent (decompose → per-primitive
   orthogonal viewpoints, real 6-DoF arm), never read from its own tables; anton's §4
   refuses to build on its numbers. Blocks every novelty claim in academia §1.7.
2. **SPAZER + TagaVLM** — the two most direct published threats to the text-only blind
   decider (+3.6% over random; +0.94% for text topology). Unverified either way.
3. find-superquadric (partial cloud + table prior — closest deployment match for
   first-iteration fits), M3C2-EP, solids-of-revolution aspect graph.

**NOT on any surviving debt list — the completeness finding:**
4. **Branch 3 map sections E and F: read not at all** (10 documents). Section E is our
   specific defect modes (D405/D415 characterization, dropout physics, hand-eye error,
   depth-noise simulation); **section F is failure detection / the fallback trigger** — i.e.
   the literature behind Part B/E4's fit-maturity gate, which currently has *zero* sources
   behind it. The gate is one of the month's three genuine unknowns; its designated
   literature was never opened.
5. **GS-NBV** — the one candidate cylinder-orbit precedent in the corpus
   (`intelligent-viewsphere` grounding table: the orbit recipe is "invented… must ship as
   ours, derived"), unread while the cylinder orbit is half the view-crafting design.
6. **Robust extrinsic symmetry estimation (Branch 3 #35)** — the *only* candidate for
   partial-cylinder axis completion, the exact load-bearing half-cylinder case; unread.
7. **Branch 1 Tier 7 (image-space quality scoring): entirely unread** — and the QA reframe
   moved the evidence construct into RGB image space. Mechanical decode success substitutes
   for most of it, but the menu's framing/standoff math and any future evidence-image quality
   check (sharpness, exposure, blur at close range) have no literature behind them. At
   minimum, revisit the tier's top 2–3 entries rather than leaving the demotion implicit.
8. **Staderini et al., 3DV 2024 (photometric visibility matrix)** — named *in the feedback
   itself* as "the top unread item" for the glare-at-perpendicular worry. Still unread, and
   the worry is still live (see §3.1).
9. **Branch 2 Group H (applied inspection-planning evaluation, 0/5 read)** — including
   `cpp-measurement-uncertainty`, rated high, "the closest published evaluation design to
   ours". Also: `quoniam` (the concrete anti-circularity instrument for Q3), Karaszewski
   2016 (the one comparative NBV benchmark across scanners *and manipulators* — flagged
   "single highest-priority lead" for baselines), Tanks&Temples/DTU (metric conventions,
   second-hand only), `how-many-views-prv`, `scoutvla`.
10. Katz HPR primary + Open3D `hidden_point_removal` parameter semantics — the mesh-free
    visibility hazard (radius sensitivity on non-uniform density; do dropout holes read as
    "visible through") flagged in Branch 1 §4.2 and never validated; SEE-style
    point-proximity occlusion is in the Part C spec.
11. Vasquez-Gomez expected-utility-under-pose-uncertainty — the closest published machinery
    for the branch's "strongest contribution claim" (uncertainty propagation into view
    scores); unread, so the novelty claim's nearest neighbor is unexamined.

Pass 2 should triage this into: read-before-claiming (1, 2), read-before-building (4F, 5, 6,
10), read-if-time (rest), and explicitly-retired-with-reason (Group G defect-visibility,
arXiv:2201.11924, image-space tier if the decision is to rely on decode-only scoring).

---

## 3. Questions the pass left open (not merely under-specified)

### 3.1 The glare-at-orthogonal worry was moved, not closed
The feedback keeps the worry alive verbatim ("a shiny surface reflecting the room light
straight into the lens at perpendicular — no paper we read answers that") and then scopes
sensors out ("once we are in orthogonal view we care only about the rgb image"). But the
worry *is* an RGB worry: specular highlight blowing out the evidence image at exactly normal
incidence — on glossy tape, laminated labels, printed codes, i.e. the demo material. Anton
§1.6 declares "specular-at-normal worry: out of scope", which is only true for the depth
leg. **No experiment covers the RGB leg.** Cheap closure: one sentence requiring E6
(reading-accuracy vs incidence) to include glossy code material under the actual demo-cell
lighting; if a dip appears at 0°, the menu needs a small off-normal offset option — better
discovered in week 2 than at the demo.

### 3.2 The discovery capability of the eyes tier has no experiment
The loop's search phase depends entirely on the VLM image agent detecting "codes present on
this face" from **oblique, partial, distant first views** — that is what steers the loop to
the right face when the box is rotated. Tier 0 tests reading at the served view; E6 tests
reading vs angle; `demo-v2-acceptance` Issue 6 stratifies discovery vs capture — but **no
experiment measures detection-from-bad-views recall**, the number that determines how many
moves discovery costs and whether the hard cap fires before discovery. One Tier-0 item
family ("is there a code/label on this face? y/n" over oblique captures) closes it.

### 3.3 The scoring instrument's own error bar has no experiment
Branch 2 Q2's recommendation — report the analytical GT's *own noise floor on the same scale
as the score* (the Dutagaci Inconsistency discipline) — was accepted in principle and
instantiated nowhere. E1 measures fit error on calibration objects; `demo-v2` Issue 1 flags
the episode-time reference normal as unspecified/circular; but no protocol estimates the
error of the **episode-time view-correct label** (locally refit normals from the same
defective depth, on cardboard whose "true normal" is ill-defined at the few-degree level —
`partial-fit` Issue 4). Without it, the grounded 2×2's view axis has an unknown noise floor,
and near-threshold episodes are unscoreable. Needs one calibration experiment linking E1's
error curve to the episode-time scorer.

### 3.4 Smaller unexperimented design elements
- **Menu-diff announcements** ("two new views appeared after the last scan") — anton §1.2's
  design response (b) to menu churn. E3/E7 test labels and order; nothing tests whether diff
  announcements help, confuse, or bloat context. Cheap rider on the representation A/B.
- **Agent stop quality as its own number** — Branch 2 Q6's "measure agent stopping
  separately" is adopted in prose by all three interpretations and owned by no experiment.
- **Per-round wall-clock budget** — fit + derive + exact IK/collision on survivors + menu
  build + LLM call, per round, in a live session. Industry mandates analytic visibility
  because ray-tracing takes hours, ILP is ~1 s — but no experiment verifies the *whole
  round* fits an interactive budget. E11's self-checks cover replay, not the live loop.
- **The curvature guard** (RSPD cylinder-tiling) — required on day one per anton §1.3,
  untested by any experiment (`partial-fit` Issue 9 flags it; unadopted).

---

## 4. Load-bearing claims still resting on unverified ground

- **"Exists nowhere in the literature"** (E2's angular-error-vs-coverage number, the
  re-scan repeatability protocol, the (object, surface-region) unit): all absence claims are
  confirmed only over the ~40% of mapped documents actually read (20/39 in Branch 3; 20/58
  in Branch 2). `partial-fit` Issue 8 already demands the honest phrasing ("absent from the
  corpus we read"); apply it to all three novelty claims in academia §1.7.
- **Branch 5's measure-zero argument** (orthogonal views are boundary points of any sphere
  partition) is the keystone of anton §1.1's "no resolution of the grid fixes that" — it is
  a mathematical argument from one synthesis, quoted forward without any round-6 check. Low
  risk, but it is the single most-cited structural justification and deserves a two-line
  formal statement in pass 2 (for a cube face it is true for generic orientations; state the
  measure-zero claim precisely so it cannot be quoted past its validity).
- **"~1 s exact ILP"**, MOKA's hierarchy effect, K≈2 vs K=5, SCAFFOLD deltas: verified
  against primary text by round-3 readers per the branch hygiene notes — acceptable. The
  Wang T-Mech numbers and SPAZER/TagaVLM remain the two unverified items that could actually
  change a design decision (see §2).

---

## 5. Benchmark blind spots (beyond what round 6 already caught)

Round 6 caught the big ones — coverage saturation, QA-item saturation, arm-asymmetric
snapping, oracle circularity, QA-authoring selection effect, symmetry equivalence, holdout
pricing. Two additions:

- **Tier-0 sampling bias.** "Images sampled from existing data" means sampled from captures
  made under the *old uniform viewsphere* — a distribution that rarely contains
  near-orthogonal views. Tier-0 item pools built from it will under-represent exactly the
  view class the treatment produces. E6's matched-distance sweep requires new captures
  anyway; say explicitly that Tier-0 pools must be capture-source-balanced or
  angle-stratified.
- **Tier-2 "sim" is a named mode with zero scoping.** The feedback says "sim will definitely
  be used at some point"; Branch 2 Q4 established that simulation from our input requires a
  fitted mesh + synthetic materials (a raw cloud has no material channel) and that glossy
  dropout is the case even PBR simulation has not closed. Nothing in pass 1 decides what
  sim is *for* or when. Fine to defer — but record it as deferred-with-reason, or it will
  be improvised mid-month.

---

## 6. Methodology ↔ demo-v2 gaps

- **The cylinder never meets the real robot.** Cylinder-orbit is half the view-crafting
  design; its experiments are sandbox loop convergence (anton E5) and fitting sweeps
  (E2's half-cylinder). The demo object is a box; the only cylinder-on-hardware appearance
  is `demo-v2-acceptance`'s *suggested* generalization probe. If the month's claim is about
  geometry-conditioned spaces (not box-conditioned spaces), one real cylinder episode is
  cheap and should be a named deliverable, not a suggestion.
- **Adopt the round-6 demo fixes as protocol, not commentary:** generalization probe
  (second box ≠ demo box, ideally the cylinder) as the mechanical ADAM guard; adversarial
  placements (face-down → scored abstention) so the acceptance test can fail; the 2×2 grid
  + abstention cell instead of the single fraction; frozen thresholds + pass bar + K×M
  sizing; discovery/capture stratification. All five exist as verifier text only.
- **Demo-cell conditions are load-bearing and unrecorded** for both legs: depth (smoke-test
  Issue 6: lighting/stream profile) and RGB evidence (§3.1 glare). The acceptance protocol
  should pin lighting and stream profile as first-class recorded conditions.

---

## 7. Pass-2 work-list (priority order)

1. **Consolidate**: one canonical part list + one experiment list (unique IDs), merging the
   three interpretations and folding in all 16 round-6 fix lists with an adopted/rejected
   mark per item. (§1.1)
2. **Architect rulings** on the four forks: pre-registration-minimal vs none; stop-rule;
   fallback space + granularity; E2 kill threshold. (§1.2)
3. **Repair the paper trail**: revise the contract (QA reframe, passive-stereo sensor
   story, loop premise); write demo v2 into the binding record. (§1.3)
4. **Reading queue, triaged** per §2 — Wang T-Mech and Branch 3 section F before their
   dependent claims/designs ship; explicit retirement notes for what is dropped.
5. **Close the moved-not-closed questions**: E6 glossy-material/lighting rider (§3.1);
   discovery-recall Tier-0 item family (§3.2); episode-time scorer noise floor (§3.3).
6. **Add the missing small experiments**: menu-diff rider, stop-quality number, per-round
   wall-clock check, curvature-guard test. (§3.4)
7. **Benchmark hygiene additions**: Tier-0 sampling balance; sim deferred-with-reason.
   (§5)
8. **Demo hardening**: real cylinder episode; adopt the five round-6 demo fixes; pin cell
   conditions. (§6)
9. **Novelty-claim provenance**: rephrase all three absence claims with corpus scope; add
   the two-line measure-zero statement. (§4)
