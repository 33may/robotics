# Round 6 — Verification: `sensor-smoke-test` (timeboxed)

**Part under test:** industry interpretation Part G / E11 (`round5_interpretation/industry.md` §1.3.1, §2 Part G, §4 E11), with siblings in `anton.md` (§1.1 "slim check", Part B/E2, open question 5) and `academia.md` (§1.4 dissent, E8).
**Claim of the part:** half a day, week 1. Depth fill rate and plane-fit quality on the actual demo objects across the operating standoff band and incidence angles; verify Min-Z (100 mm at full resolution) and the invalid-band numbers. Output: a go/no-go on the camera and a default incidence gate τ. Explicitly not a metrology campaign.
**Verdict: needs-work** — the concept is grounded, cheap, scope-compliant and genuinely falsifiable, but as written the part cannot deliver its own two declared outputs: the go/no-go has no pre-stated pass bar, and τ may be unidentifiable from the stated measurement design. Both fixes are protocol wording, not new work.

---

## 1. Grounding audit — literature-derived or round-5 invention?

**Derived, with a clean paper trail.** Every load-bearing element traces to round-3/4 material:

| Element of the part | Source in the corpus | Status |
|---|---|---|
| D405 is passive stereo, no projector | Branch 1 §2.1 — datasheet table-level read (Tbl 2-2, 3-15, 3-49), flagged as *the* load-bearing correction of the pass | primary, verified |
| Textureless-face dropout, angle-independent | Branch 1 §2.1 — "a second candidate root cause… passive stereo can lose fill rate *independently of incidence angle*, purely from lack of texture"; explicitly "separable causes and should be separated experimentally" | **hypothesis**, not a measured D405 result (see Issue 2) |
| Min-Z = 100 mm at 1280×720 (not the marketed 7 cm) | Branch 1, D405 Tbl 4-11 | primary, verified |
| Invalid-depth band ≈10 % of image width at 10 cm | Branch 1 §2.1, `B·F/Z` derivation | primary, verified |
| τ must be measured, not inherited | Branch 1 Q1 verdict ("a genuine gap that must be closed by our own measurement"); Branch 4 §3.3 ("published numbers span 8.6°–75°… no universal threshold and we must measure ours for the D405") | unanimous across two branches |
| Plane-fit residual as the quality instrument | Branch 1 term table — Soudarissanane's per-patch precision `σ = RMSE of plane-fit residuals` (Eq 4.21) | precedented instrument, correctly chosen |
| Two-tier gate-plus-smooth-model shape the τ output feeds | Branch 1 §2.2 — six independent groups | precedented |

The part is the *demoted descendant* of Branch 1's own round-5 priority #1 (full σ(range, θ) campaign, matte/glossy, with/without texture), scoped down after the architect's feedback ("lets discuss… if we need to go that deep"; "we dont care about the sensors once we are in a good view"). It was not invented in round 5; round 5 shrank it. The shrink is legitimate: the architect's own interpretation retains the identical content as "a slim check inside the fitting work" (anton.md §1.1) and open question 5 answers "how much depth-quality work" with exactly this. All three personas independently converged on a half-day version — that convergence is itself evidence the carve-out is load-bearing rather than a persona's hobby.

## 2. Scope-compliance audit

- **"Deep sensor-noise modeling" out of scope:** respected. No σ(ρ,θ) curve is fitted, no noise model shipped; fill rate + plane-fit residual + two datasheet verifications is the floor, not a campaign. The feedback's "we measure it ourselves" line licenses self-measurement; the feedback's scepticism ("if we need to go that deep") is answered by the timebox, not violated by it.
- **"Once in a good view we care only about RGB":** not contradicted. The smoke test serves the loop's *perception before* the good view — the incremental loop the feedback itself made central runs scan→fit→derive on **depth** every round. Depth serves the loop; RGB serves the evidence; the test is entirely on the loop side.
- **Incremental partial-cloud fitting:** served, not violated — but see Issue 3: the test must measure *single-view* captures to be loop-realistic, and the part doesn't say so.
- **LLM exposure / ADAM regression / VLA / RL / defect detection:** none involved. A bench characterization of the sensor is not a per-object recipe.
- **Benchmark-as-auxiliary:** untouched; this is not benchmark work.
- **Kinematic reachability:** marginally relevant — the "operating standoff band" swept should be the *reachable* band (shell minimum vs Min-Z); the part implies this but doesn't state it.

## 3. Falsification-power audit

Does the experiment prove/disprove, or only confirm? **It is one of the few parts in round 5 designed to find a failure where it is most likely**: passive stereo's hypothesized worst case (textureless matte face) is literally the demo object. It can fail, the failure is cheap to observe, and the response is concrete and time-critical (exercise the flexible-camera option in week 1, while swapping is still free). That is the right shape.

But the architect's standard is falsification attempt + rendered evidence + concrete usage story, and the part as written erodes on the first leg:

- **Usage story: real.** τ feeds Part C's admissibility gate stack; the verified standoff band feeds the framing/standoff math (Min-Z at chosen resolution bounds the shell from below; invalid band argues against the minimum); go/no-go feeds procurement. Three named consumers, all week-1–2.
- **Rendered evidence: implicit but trivial.** Fill-rate maps and residual-vs-angle plots fall out of the analysis for free; the part just doesn't mandate keeping them.
- **Falsification: compromised by a missing bar** — Issue 1 below. A go/no-go without a pre-stated criterion is an impression, not a decision, and week-1 procurement pressure is exactly the condition under which an impression gets rationalized to "go."

## 4. Issues

**Issue 1 — the go/no-go has no pass bar, so the test cannot produce its own primary output.** "If fill collapses on textureless faces, request a different camera" — collapse is undefined, and raw fill percentage is the wrong axis anyway: a plane fitter is robust to heavy but unbiased dropout; sparse-and-unbiased passes where dense-and-biased fails. The criterion that matters already exists one part over (anton.md Part B disproof condition): **fitted-normal angular error vs the menu's angular tolerance**. The bar must be written before the half day runs: *no-go if, on single-view captures of the worst demo face at operating standoff, the fitted plane normal errs by more than the menu's angular resolution (or the fit finds no plane at all) in the sub-τ incidence regime.* Without this, the test fails the architect's own falsification standard while claiming to embody it.

**Issue 2 — τ may be unidentifiable from the stated design, and no fallback is stated.** Branch 1 says texture-dropout (angle-independent) and incidence-dropout (angle-dependent) are separable causes that must be separated experimentally. Measured only on demo objects, a textureless face can produce a flat, low fill curve with **no knee** — then "a default incidence gate τ" simply does not come out of the data, and the joint curve conflates the two mechanisms the branch said to separate. The materials list (cardboard, tape, printed codes) spans texture, but the analysis plan doesn't say τ is derived from the textured/valid-correspondence regime while go/no-go is judged on the textureless one. Academia's E8 has the missing piece — matte vs glossy separated, **60° adopted as provisional** (majority report: Soudarissanane, Lichti, Johnson, Mehdi-Souzani) until the sweep says otherwise; the industry part dropped the fallback. State it: *if no knee appears within the band, τ = 60° provisional and the gate is flagged as literature-default, not measured.*

**Issue 3 — single-view vs fused is unspecified, and only single-view answers the loop's question.** The loop's first iteration fits primitives to one capture. Fused clouds structurally hide single-view fill collapse (another view fills the hole), so any check run on existing fused PLYs — or on multi-capture accumulations — answers a different question than the one the incremental-loop correction made load-bearing. The two-hour E1 fit shakedown on existing PLYs does **not** make the smoke test redundant for exactly this reason, but only if the smoke test explicitly measures per-single-view fill and per-single-view plane fits.

**Issue 4 — the go/no-go's authority overlaps the architect's reserved decision, and the axis split is unstated.** Anton.md open question 6: "Camera upgrade — decided by demo-v2's RGB evidence quality, not in advance." The smoke test claims week-1 camera go/no-go. These are compatible — two different failure axes (depth adequacy for fitting vs RGB adequacy for evidence) — but the part must say so: a smoke-test **"go" does not clear the camera** (RGB remains demo-v2's call), and a smoke-test "no-go" is a depth-axis veto the architect's flexible-camera provision explicitly invites. Left implicit, the week-1 result will be read as settling camera choice fully.

**Issue 5 — a scalar τ silently assumes isotropy the corpus specifically warns about.** Branch 4 §3.3: Scott 2009's 60°/70° asymmetry — tighter in the triangulation-baseline plane than orthogonal to it — "is a direct warning for a stereo sensor with a baseline. Our D405 has one. This should be tested, not assumed away." A half-day sweep tilting in one plane bakes an orientation dependence into the gate that nobody would guess. Cost of covering it: tilt about both axes at 2–3 angles — minutes, not hours.

**Issue 6 (minor) — uncontrolled conditions that move passive-stereo results.** Fill and correspondence quality for a projector-less stereo pair are ambient-light-dependent (Branch 1: "texture-dependent, ambient-light dependent"; the datasheet KPIs are qualified at ~250 lux on a *textured* target). One lighting condition ≠ the demo cell's conditions unless it is the demo cell. And the resolution choice couples three things at once — Min-Z (100/70/40 mm), depth fill, and (via the shared left imager) the RGB evidence stream. The test should run in the actual cell lighting and record the exact stream profile it validates, since "go" is only meaningful for that profile.

**Issue 7 (bookkeeping) — "documented worst case" slightly overstates the provenance.** Branch 1 presents textureless-face fill loss as a *candidate root cause* inferred from sensor class plus the datasheet's textured-target qualification — not as a measured D405 result. This is a point in the test's favour (it tests a live hypothesis rather than confirming a known), but the part's write-up should not cite it as established, or a "no fill problem" result will look like a contradiction of the corpus instead of a hypothesis resolved.

## 5. Suggestions

1. **Pre-state the no-go bar in one sentence before running**: single-view plane fit on the worst demo face, at operating standoff, sub-τ incidence — normal error must be within the menu's angular tolerance; else no-go on the depth axis. (Issue 1)
2. **Split the protocol into its two questions**: (a) go/no-go on textureless demo faces (fill + fit, single-view); (b) τ from a textured control face — the printed-codes face is the natural one, plus a flat plate if available — where correspondence is valid and the knee is observable. (Issues 2, 3)
3. **Adopt academia E8's fallback**: no knee in the data → τ = 60° provisional, recorded as literature-default. (Issue 2)
4. **Add the two-axis tilt** (about the baseline and orthogonal to it) to check anisotropy in passing; if asymmetric, τ becomes a pair, and Part C's gate takes the tighter value until told otherwise. (Issue 5)
5. **Scope the output line**: "go/no-go on the camera *for depth-driven fitting*; RGB evidence quality remains demo-v2's decision (architect's open question 6)." (Issue 4)
6. **Record conditions as first-class**: cell lighting, exact stream profile (resolution/fps), exposure mode; verify Min-Z and invalid band at that profile only, and keep the fill maps and residual plots as the rendered evidence. (Issue 6)

## 6. Bottom line

The carve-out is real, not invented: it descends from Branch 1's strongest empirical gap (no passive-stereo incidence data at 10–40 cm exists anywhere), was independently retained by all three personas after the architect's scope cut, and is the cheapest possible test of the one hypothesis that could stall every downstream part in week 3 instead of week 1. It respects every architect constraint — it is the loop's-eyes complement to the RGB-evidence scope cut, not a violation of it. What it needs before execution is a written pass bar tied to the fitting requirement, a texture/angle separation in the analysis so τ is actually identifiable, the single-view stipulation, and the 60°-provisional fallback. All four fit inside the same half day; none change the part's identity.
