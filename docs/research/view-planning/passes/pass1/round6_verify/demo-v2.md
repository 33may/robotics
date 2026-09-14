# Round 6 — Verification: "Box-with-codes as a grounded-success artifact" (demo-v2)

**Part under test:** the demo is also an experiment — box moved and rotated per trial,
adversarial placements included (code face away, near-table, face-down → abstention
expected); success = the grounded cell of the 2×2 grid (evidence image within the
angular tolerance band of the code face's normal AND codes mechanically decoded) plus
scored abstention — never a single end-to-end accuracy.
**Primary sources of the part:** academia §2.7 + E9; anton Part H + E6 + E12; industry
Part F + E10 + E12. Checked against the round-4 feedback (binding), the framing doc,
and the round-4 branch syntheses.

**Verdict: needs-work.** The part is genuinely grounded in the reviewed literature —
not a round-5 invention — and it is one of the few parts whose experiments can actually
fail. But the success criterion as currently written is under-specified in three places
where the naive resolution is circular or self-confirming, and the three interpretations
contradict each other on two of them. Fix the specification and the part is sound.

---

## 1. Provenance — grounded or invented?

Grounded. Every load-bearing element traces to round-4 text, verified line-by-line:

- **The 2×2 grid and the 26.2% number.** `benchmarking-view-planning.md` F7 (lines
  182–194) and Q15 (lines 1086–1115): `view-level-evidence-benchmark`'s four cells —
  grounded 55.9% pooled, wrong-view/right-answer **26.2%**, worst case Qwen2.5-VL-7B
  raw 78.7% vs grounded 8.2% (the "up to 70 points" overstatement). Q15's
  recommendation is verbatim the part's rule: "Any end-to-end inspection number we
  report ships with the grid, or it is not trustworthy on its own. Add a fifth cell
  for abstention." Corroborated from two independent directions (`e3vs-bench`
  self-report; `planning-with-the-views` wander-and-match ≥90%).
- **The abstention cell.** Branch 2 Q16 (line 1127): abstention accuracy 16.7–33.3%
  even for frontier models on unanswerable items; recommendation "add abstention as a
  scored outcome," mapped onto exactly our fall-back case. Branch 4's fallback ladder
  (rungs i+iv, Yu & Gupta) supplies the mechanism the demo's abstention depends on.
- **Mechanical decoding as the score.** `ycb-protocols` pattern, Branch 2 (line 423):
  "every score is a physical measurement." A decoder fires or it doesn't — the
  judge-free scoring the QA reframe buys, taken deliberately (academia §1.1).
- **Tolerance band rather than a point.** Branch 2 (line 679): `bonaventura`'s
  plateaus — "'good' is a region, not a peak. That supports building our GT as a
  tolerance band around the normal."
- **Moved-and-rotated robustness.** The architect's own framing doc and feedback
  ("robust to moving and rotating the box"); perturbation-graded protocols precedented
  by `ycb-protocols`' peg-insertion perturbation grid. Failure narratives mandatory
  per the same source.
- **Adversarial placements** are a round-5 operationalization (no round-4 text lists
  face-down/near-table), but they are the honest operationalization of "robust to
  placements" plus the reachability findings (ObjView 45.4/128 reachable; NBV-Net
  0.988→0.472 collapse), and they are what gives the demo disproof capacity. New but
  legitimate.

One transplant to keep honest: `view-level-evidence-benchmark` is a multi-camera
driving-VQA benchmark (CAM_FRONT et al.), free-form answers, no robot. Academia §1.7
declares the adjacency ("the 2×2 exists once, in an adjacent domain") — good. The
26.2% is motivation for measuring grounding, not a predicted rate for us: with
mechanical decode as the answer criterion, wrong-view/right-answer can only occur in
the band-violated-but-still-decodable angular region, so our rate is bounded by decode
physics, not by VLM guessing habits. The part's "since the one benchmark measuring it
found up to 26%" is acceptable as motivation; it must not be cited as an expected rate.

## 2. Do the experiments prove/disprove, or only confirm?

Mostly genuine falsification — this part survives that test better than most:

- **E9/E10 can fail**: face-down placements hit exactly the capability the literature
  says fails 67–83% of the time absent interface support; near-table stresses IK and
  table collision; per-stratum reporting + mandatory failure narratives prevent
  averaging the failures away.
- **E6 (anton) / E12 (industry)** — decode accuracy vs incidence, GSD vs legibility —
  directly test the premise that orthogonal evidence is better *for this task*, with a
  stated disproof condition (flat band to 40–50° would soften the requirement).
- The demo is correctly positioned as acceptance test, not the headline claim (Part
  G / E8 carries that), so it does not smuggle in the comparative conclusion.

But three specification holes let the criterion confirm itself if resolved lazily:

### Issue 1 — the reference normal for "view-correct" is unspecified, and the default is circular (blocking)

"Evidence image within the angular tolerance band of the code face's normal" — whose
normal? If it is the system's own fitted primitive, view-correct is true by
construction whenever the fit is trusted (modulo execution error), and the grid
degenerates to decode-success — the exact circularity Branch 2 Q15 flags ("the grid
requires view-level labels, which requires Q2's GT to exist"). Industry's guard
(score against the raw cloud's locally-refit normals) is better but still shares the
sensor and registration pipeline with the system under test. For a rigid printed box
there is a fully independent GT available and nobody named it: **the decoded code's own
corner geometry.** QR/DataMatrix/AprilTag corners + known code size + camera
intrinsics → PnP → the code face's plane normal in the camera frame, computed purely
from the RGB evidence image, independent of depth, fitting, and registration. It
covers every decode-successful trial mechanically; the residual cells (no decode)
need per-trial manual pose annotation, which at demo scale (tens of trials) is cheap.
The protocol must name its GT source per cell before the first scored run.

### Issue 2 — the tolerance band has three conflicting owners across the interpretations, and two of them drift the construct (blocking)

- academia §4 pre-registers the band "from E1/E8" — but E1 is *fit* angular error and
  E8 is the *depth* incidence gate. Setting the evidence band from E1 tunes the
  success criterion to what our own fitting happens to achieve (pass by
  construction); setting it from E8 imports a depth-dropout number into an RGB
  legibility claim (category error — the feedback says evidence is judged on RGB).
- anton E6's softening clause ("if reading accuracy is flat out to 40–50°, the
  requirement softens to 'any view within the flat band'") sets the band from decode
  limits. Then view-correct ≈ answer-correct for decodable codes, the
  wrong-view/right-answer cell empties *by definition*, and the demo can ship a 45°
  oblique evidence image that decodes — satisfying the grid while violating Albert's
  actual, twice-flagged requirement ("evidence images should be the *most orthogonal*
  view"). Decode is the mechanical proxy; the construct is evidence legibility for a
  human consumer. E6's flat band should inform *menu density*, not the success band.
- industry E12 feeds GSD into the standoff math — fine, but a third input.

One owner, pre-registered, derived from the evidence-quality construct (Albert's
requirement + the menu's angular resolution), independent of both the decode curve and
the fit-error curve. The two curves then remain what they should be: diagnostics that
can *fail* the band, not definitions of it.

### Issue 3 — the near-table and code-face-away strata have no success cell (moderate)

Face-down is clean: no reachable view exists, abstention is correct, scored. But
near-table (orthogonal view unreachable, oblique reachable) is exactly the case the
design's own fallback ladder (Branch 4 rung iv: defer to next-best with disclosure)
handles *correctly* — and the grid as written can only score that correct behavior as
view-wrong or as abstention. The E9 phrasing ("code face away, near-table, face-down →
abstention expected") is ambiguous about whether abstention is expected for all three;
if read that way, it is wrong for code-face-away (solvable by orbiting when reachable)
and rewards non-attempts. Fix: pre-register the expected outcome per placement stratum
(away → grounded success; near-table → best-reachable-with-disclosure, scored as its
own named outcome or as view-correct conditioned on the feasible set; face-down →
abstention), before the first run.

## 3. Constraint compliance

- **Incremental partial-cloud fitting:** respected — the demo inherits the loop;
  per-trial re-placement forces a fresh fit every episode. No survey-first regression.
- **Real sensor defects:** respected in design, with one named dependency — cardboard
  is the D405's textureless-matte worst case (industry §1.3.1); the demo's viability
  hangs on the week-1 sensor smoke test (E11/Part G). That is a dependency of the
  part, not a defect in it, and it is correctly sequenced early while the camera is
  still swappable.
- **Kinematic reachability:** respected and actively exercised — the adversarial
  placements are the reachability findings turned into test cases.
- **Discrete LLM menu / geometry never exposed:** respected. The abstention pathway
  requires only the flagged-not-silently-removed disclosure ("underside cannot be
  inspected: table contact") — a text fact, not geometry exposure, per Branch 6.
- **Benchmark-as-auxiliary:** respected — the demo is the acceptance artifact; its
  protocol reuses the harness's scoring, it does not grow the harness.
- **No VLA/RL/defect detection:** clean.

## 4. ADAM regression check

No regression. The code face is found by content (the VLM sees codes), not by a
per-object script; per-trial re-placement makes a memorized recipe fail by
construction; industry Part F names the guard explicitly ("no per-object scripts
anywhere; only per-primitive rules"). The residual risk is subtler — iterating the
loop against the single demo box until it passes (N=1 overfitting). Mitigations exist
in the design (holdout captures; the benchmark carries the general claim) but the demo
protocol itself should state that protocol parameters freeze before the scored runs.

## 5. Remaining issues (non-blocking)

- **Issue 4 — abstention scored one-way.** The 16.7–33.3% literature number is about
  *failing to abstain*; the demo must also score *false* abstention on inspectable
  placements, or always-abstain games the adversarial strata. Per-stratum reporting
  makes this detectable, but the protocol should state both directions explicitly.
- **Issue 5 — "codes mechanically decoded" conflates artifact and system.** Two
  different answer-correct criteria hide in the phrase: (a) the *evidence image* is
  decodable by the offline scorer (artifact quality — third-party-verifiable evidence,
  the right construct for Albert's requirement); (b) the *system's* VLM-reported
  string exactly matches the decoded GT (QA outcome — the thing the month's framing
  says inspection is). They can disagree (decodable image, VLM misreads). Score both,
  report both; picking silently one collapses either the evidence claim or the QA
  claim.
- **Issue 6 — cross-episode identity is requirement inflation.** Academia §2.7 has the
  demo inherit "stable face identity across re-placements"; industry §1.4.3 correctly
  argues the demo needs only within-episode stability, because the code face is
  re-found by content each episode and cross-episode canonicalization is unsolved
  (box-like shapes its dominant failure class). The weaker requirement suffices;
  adopting the stronger one burns build time on a zero-literature-cover problem the
  demo doesn't need solved.
- **Issue 7 — stratum rates at tiny N.** K placements × M rotations split per stratum
  gives single-digit counts per cell; Branch 2's own sizing note (n≲100 resolves
  15–20 points) forbids quoting stratum percentages. The part's "failures narrated,
  not averaged away" is the right instinct — make it explicit that the demo grid
  ships raw counts + narratives, never rates.

## 6. The architect's standard

- **Falsification attempt:** yes, once Issues 1–3 are fixed — adversarial placements,
  a scorable abstention expectation, and disproof conditions on the premise itself
  (E6) give the demo real ways to fail. As written, Issue 2's E6-softening path is the
  one door through which the demo could "pass" while failing Albert's requirement.
- **Rendered evidence:** the evidence image *is* the artifact, and PnP-from-code makes
  its correctness mechanically checkable — stronger than anything else in the plan.
- **Concrete usage story:** Albert's box demo, verbatim from the framing doc.

## Bottom line

Keep the part. It is the best-grounded acceptance criterion in the round-5 set and the
only one whose primary score needs no judge. Before the first scored run, pin three
things in one pre-registered protocol paragraph: (1) the GT normal source per grid
cell (PnP-from-code where decode succeeds; manual pose annotation otherwise — never
the system's own fit), (2) a single owner for the tolerance band, derived from the
evidence construct and independent of both the decode curve and the fit-error curve,
(3) expected outcome per placement stratum, with next-best-under-disclosure as a named
outcome and abstention scored in both directions. Without those, the grounded cell can
be satisfied circularly; with them, the demo is a real experiment.
