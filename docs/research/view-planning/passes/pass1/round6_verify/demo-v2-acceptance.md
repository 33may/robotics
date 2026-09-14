# Round 6 — Verification: `demo-v2-acceptance`

**Part under test:** the demo v2 acceptance protocol — K placements × M rotations; score = fraction of episodes ending with a readable orthogonal evidence image of the code side; plus moves-to-answer, wall time, failure narrative per miss; robustness by construction (per-episode refit, content-based target finding); "doubles as the ADAM-regression guard — per-primitive rules only, no per-object scripts."
**Primary source of the wording:** `round5_interpretation/industry.md` Part F (§2, lines 296–303) and E10. Siblings: `academia.md` §2.7 + E9; `anton.md` Part H + E6/E12.
**Verifier stance:** adversarial. Verdict at the end.

---

## 1. Grounded or invented?

Mostly grounded — the protocol is a *composition* of elements that all trace to the round-4 corpus, not a round-5 invention:

- **Failure narrative per miss** ← `ycb-protocols`' mandatory "reasons for failed attempts" submission field (`benchmarking-view-planning.md` lines 1016–1019, adopted as recommendation at line 1039 and in the reporting template at line 1612). Real precedent, correctly transplanted.
- **Placement/rotation perturbation protocol** ← `ycb-protocols`' peg-insertion perturbation grid (5 mm/10 mm, line 252, 618) and the 5-objects × 5-placements reachability audit (line 84). The specific K×M shape is new, but it is protocol engineering over precedented structure, not an invented empirical claim.
- **Moves-to-answer, wall time** ← the architect's own round-4 feedback, verbatim (Tier 1 "moves to answer, travel distance"; Tier 2 "% correct, moves, duration"). Not invented by any persona.
- **Per-episode refit / content-based target finding** ← the architect's loop correction (feedback §view-set-selection) and the canonicalization open problem (`structuring-the-viewpoint-space.md` lines 491, 753 — azimuthal canonicalization unsolved, box-like shapes the failure class), which genuinely forces content-based finding over persistent object-canonical names. Industry §1.4.3's reasoning here is sound and literature-backed.
- **The orthogonality-for-QA premise** ← the one proven angle effect (resolution/foreshortening, 8× point loss), with Anton's E6 correctly designed to *soften* the requirement if reading is flat to 40–50° off-normal. This is a real disproof condition, not decoration.

One sentence is invented and over-claimed — see Issue 1.

Note on provenance: "demo v2 = box with codes, robust to move/rotate" appears in no round-4 document on disk (grep over `feedback.md` and all six syntheses: zero hits for "demo", "box with codes", "code face"). It enters the record via the round-5 headers and the orchestrator's scope statement. Since the scope statement is binding it is not a fabrication, but the pass's paper trail has a hole: the demo target should be written into the architect's own feedback file, or a later reader cannot distinguish architect intent from round-5 consensus drift.

## 2. Does it prove/disprove, or only confirm?

### Issue 1 (strongest) — "doubles as the ADAM-regression guard" is a claim the protocol cannot support as designed

K placements × M rotations of **one box** does not discriminate per-primitive rules from a per-object script. A hand-coded box recipe with a pose-estimation front end — "register the known box, visit each reachable face orthogonally, run the decoder on each" — passes every placement and every rotation, because pose robustness is exactly what pose estimation buys. The protocol detects memorized *trajectories* (original ADAM) but not memorized *object recipes* — and the framing doc's non-goal explicitly forbids the latter ("fixed recipes per whole object or class are a regression"). As written, "the protocol is a check on the whole loop, not on a memorized recipe" is enforceable only by code review, which is not a protocol.

Fix that makes the guard real: a generalization probe inside the acceptance run — the identical, unmodified pipeline on at least one object the code never names: a box of different dimensions at minimum, ideally Albert's own second sketch (cylinder orbited with a label). Any required code change = ADAM regression caught, mechanically. Without this, delete the guard sentence; the protocol is still valuable, but it is a pose-robustness check, nothing more.

### Issue 2 — the single-fraction score collapses the grounded grid the project's own corpus mandates

"Fraction of episodes ending with a readable orthogonal evidence image" is the grounded cell (view✓ ∧ answer✓) reported alone. The project's strongest benchmark finding says exactly why that is not enough: 26.2% of trials pooled across six models are wrong-view/right-answer; raw accuracy overstates grounded accuracy by up to 70 points; worst case 78.7% raw vs 8.2% grounded (`benchmarking-view-planning.md` lines 185–187, 1089–1093). A grazing image that still decodes — the *precise complaint that started this project* — is indistinguishable in the single fraction from a fitting failure or a search failure. Academia's E9 and Anton's Part H both already specify the 2×2 grid + abstention cell; the industry Part F wording (the version quoted as the part under test) regressed from its own §5 scoring section, which specifies the two-layer grid. Reconcile to the grid. Failure narratives are qualitative compensation, not a substitute.

### Issue 3 — no unwinnable placements → the protocol can only confirm

If every one of the K×M placements leaves the code face reachable, every episode is winnable and the abstention/fallback/disclosure machinery — which the design elsewhere calls mandatory, on the strength of 16.7–33.3% frontier abstention accuracy and "correctly abstained is a scored outcome, not a footnote" (lines 945–946, 1127, 1544) — is never exercised. A system that always returns *some* face with *some* image is never caught. Academia's E9 already has the fix: adversarial placements (code face down, code face toward the base / near-table) where **correct abstention is the scored success**. The part under test omits them; that omission converts an acceptance test into a confirmation instrument, and it is also the only place in the whole demo where the kinematic-reachability constraint gets end-to-end tested.

### Issue 4 — "readable" and "orthogonal" are not operationalized, and the sequencing creates a self-rigging channel

- "Readable" scored by the system's own VLM agent is circular (the same model produces and grades the answer). Academia §1.1 has the correct, corpus-backed fix: mechanical verification — a barcode/QR decoder decodes or it does not; OCR exact-match — the `ycb-protocols` every-score-a-physical-measurement pattern. The part under test never names its scorer.
- "Orthogonal" needs a tolerance band and a reference. The reference exists in industry's own §5 (angular deviation against the *raw cloud's locally-refit normals*, not the fitted primitive that proposed the view — the circularity guard); the band value must come from E6/E12 (reading-vs-incidence, GSD floor). But the protocol is to be "written before the demo is built," while its two thresholds come from experiments that need captures. Unless the protocol pre-registers the threshold *sources* and freezes values before the first scored run, the N=1 team tunes "readable/orthogonal" post-hoc on the same episodes — the exact `benchmark-lottery` rigging case the project flagged against itself (academia §0.4).

### Issue 5 — no acceptance bar, no sizing

The protocol yields a fraction but states no pass threshold and no K×M. An acceptance test without a bar is a measurement. And the project's own sizing honesty applies to itself: at n≲100 episodes you resolve 15–20-point differences. If K×M = 20, a 90% bar and an observed 80% are statistically indistinguishable. State the bar, size K×M against it, or call the artifact a robustness report rather than an acceptance test.

### Issue 6 — discovery and capture are conflated in one score

Codes on one side + content-based finding means episodes split into two populations: code face visible from the first scan (tests evidence capture only) vs hidden (tests search — the loop must explore until the VLM spots codes, under a hard cap that can fire first). Nothing in the loop *guarantees* discovery; a hard-cap-before-discovery miss is a search failure, not a view-planning failure, and pooling both into one fraction makes the score's variance an uncontrolled function of the placement draw. Stratify placements by initial code-face visibility (or report discovery-moves separately) and classify misses in the narrative taxonomy.

## 3. Constraint compliance

| Constraint | Status |
|---|---|
| Incremental partial-cloud fitting | Respected — per-episode refit is the mechanism of robustness. |
| Real sensor defects | Respected in principle (real hardware), but the demo object — cardboard, textureless faces — is the documented worst case for passive stereo. The sensor smoke test (industry Part G / E11) is an unstated *prerequisite* of this protocol; gate acceptance runs on it. |
| Kinematic reachability | Only tested if Issue 3's adversarial placements are added. |
| Discrete LLM menu / geometry never exposed | Untouched by this part; content-based finding runs through the VLM eyes tier, consistent with the blind-orchestrator design. |
| Benchmark-as-auxiliary | Respected — the demo is the system check; the comparative claim stays with Part G/E8. The part correctly does not claim the demo evidences "geometry beats uniform" (it couldn't: a uniform-grid system could also pass). |
| No VLA / RL / defect detection | Clean. |

## 4. Architect's standard (falsification + rendered evidence + usage story)

- **Rendered evidence:** yes, natively — the evidence image *is* the artifact, plus narratives.
- **Usage story:** yes, directly — this is Albert's sketched behaviour made into a scored run.
- **Falsification:** partial. The score can fail, but as designed the protocol cannot falsify the ADAM-guard claim (Issue 1), never tests abstention (Issue 3), and lets its own thresholds float (Issue 4). Three fixes, all cheap, all already present in fragments across the three round-5 docs.

## 5. Verdict

**needs-work.** The protocol skeleton is genuinely grounded (ycb-protocols failure narratives + perturbation grids, the grounded-grid evidence, the architect's own metrics) and it is the right artifact to write before building. But the version under test over-claims one thing it cannot show (ADAM guard), under-specifies the three things that make it falsifiable (grid scoring, adversarial placements, frozen operational thresholds + pass bar), and pools two failure populations. Every fix already exists in the round-5 corpus; this is reconciliation work, not new design.
