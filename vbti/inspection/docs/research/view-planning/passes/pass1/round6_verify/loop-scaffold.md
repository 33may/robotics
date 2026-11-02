# Round 6 — Verification: `loop-scaffold` (incremental perception–planning loop)

**Part under test:** the skeleton — capture → integrate partial cloud → fit/refit
primitives → derive views → filter → craft menu → LLM picks → execute → repeat;
imports Pan 2025 treat-prior-as-fallible, exact set-cover re-solve per round (~1 s,
greedy tax 12.5–40%), incidence against locally-refit normals in the live frame,
reachability-exhaustion as a valid reported stop. Claimed status: "pure engineering
with strong NBV precedent; no fundamental open choice."
**Sources checked:** `round4_accumulate/feedback.md` (binding), all three round-5
interpretations, `round4_accumulate/view-set-selection-under-constraints.md`,
`view-quality-scoring.md`, `structuring-the-viewpoint-space.md`,
`benchmarking-view-planning.md`, `vlm-spatial-interfaces.md`, and the framing doc.
**Verdict: needs-work** — grounded, in-scope, the best-precedented piece of the whole
design; but as stated it exempts itself from falsification and four of its imported
citations are quietly stretched past what the syntheses support.

---

## 1. Provenance — grounded or invented in round 5?

**Grounded.** The loop flip is not a round-5 invention: it is the architect's own
binding round-4 correction (`feedback.md` §view-set-selection: "we do NOT have the
scanned object up front… It is a loop"), and the round-4 agent consequence note
already drew the conclusion the part restates ("the iterative NBV structure IS our
structure, just with primitive fitting in the middle and no learned selector").
Every imported design rule traces to a read synthesis, verified line-by-line:

| Claim in the part | Where it lives | Check |
|---|---|---|
| Exact ILP set cover ~1 s | view-set-selection §2.3: Pan 2025 Gurobi ~1 s @800 candidates; MA-SCVP <10 s; Glorieux threshold <1000 candidates; our instances tens of candidates | ✔ faithful |
| Greedy tax 12.5–40% | same §2.3: Scott 2009 40% (mask), 12.5–25% (bunny); Glorieux 5.5% / 23.8% | ✔ faithful (elsewhere quoted 5.5–40%; pick one) |
| Pan 2025 conservative inflation | view-set-selection §6.2: Eq. 4, `𝒫̃⁺ = 𝒫̃ ∪ 𝒬 ∪ ℐ`, measured 96.99→98.06% (maize) | ✔ exists — but see issue 3 |
| Live-frame locally-refit normals (Nguyen) | view-quality-scoring §4.4 design response 1 | ✔ exists — but see issue 4 |
| Reachability exhaustion = valid stop (Vasquez-Gomez) | view-set-selection §4.1, non-rung 2 — *via Peuzin-Jubert §4.3* | ✔ exists — proxy read, see issue 8 |
| "Strong NBV precedent" | Kriegel next-best-scan, Wu autoscanning, SEE — all primary reads in view-quality-scoring | ✔ |

The "rejoins the field's mainstream, cite it that way" framing is honest: the
anti-NBV argument in Branch 4 explicitly loses its premise once the survey-first
assumption is withdrawn, and the mainstream lineage (Scott 2003 → Kriegel → Wu →
SEE → SCVP/MA-SCVP → Pan) is read literature, not vibes. Also correctly claimed as
novel on top of that mainstream: no paper in 20 conditions viewpoint generation on a
fitted primitive decomposition (view-set-selection §8, gap 1) — the loop is
precedented, the thing inside it is not, which is exactly how the part frames it.

## 2. Scope compliance

- **Incremental partial-cloud fitting:** it is the definition of the part. ✔
- **Kinematic reachability:** filter stage + exhaustion-stop + flagged-not-silent
  removal (NBV-Net 0.988→0.472 collapse cited as the anti-pattern). ✔
- **Discrete LLM menu selection, geometry never exposed:** the scaffold hands the
  LLM only the crafted menu; primitives stay internal. ✔
- **No VLA/RL, no defect detection:** none present; no learned selector, no learned
  stopping. ✔
- **ADAM regression check:** no per-object recipes anywhere in the loop;
  per-primitive deterministic rules only, and target-finding is by content
  (image agent) not by memorized pose — the demo's move-and-rotate robustness comes
  from per-episode refit, which is the anti-ADAM property itself. ✔
- **Benchmark-as-auxiliary:** the scaffold consumes the benchmark (replay mode),
  doesn't inflate it. ✔

## 3. Issues (ranked)

### 3.1 The part declares itself unfalsifiable — that fails the architect's standard
Academia §2.1 closes with "Open choice: none fundamental — this part is engineering
with strong precedent," and **no experiment in academia's E1–E9 targets the loop
itself**. The falsification apparatus exists but lives in the *other two*
interpretations: Anton's E5 (loop convergence vs a one-shot dense-survey control at
matched budget, with the real disproof condition — "if the incremental loop needs as
many moves as a dense survey to produce a usable menu, the architecture reverts to
survey-then-inspect") and industry's E1 (end-to-end smoke test in the sandbox,
scripted picker, week 1). The part must adopt both, or it is the one piece of the
design that ships on precedent alone. Precedent says loops like this work; it does
not say *our* loop with primitive fitting in the middle converges in fewer moves
than a dumb survey — that is precisely the un-precedented insertion.

### 3.2 "~1 s per round" is the solver, not the round
The measured ~1 s is the ILP solve. The corpus's own numbers say the per-round cost
sink is building the coverage/visibility matrix, not solving it: Mosbach ray-traces
41 s–62 s for a 36–48-candidate cylinder; Gospodnetić 3.81 s *per candidate*; Pan's
visibility computation is 250–2000 s against its 1 s solve. None of that fits inside
a live session round. Industry §1.2.3 names the fix — visibility/occlusion analytic
at primitive level, SEE's mesh-free point-proximity test pre-fit — but the part
under test doesn't carry the commitment. Without it the "re-solve every round is
free" sentence is true and misleading. State it: *the solve is free; the matrix must
be analytic-by-construction at primitive level, and that is a design obligation of
this part, with the mesh-free fallback named.*

### 3.3 The Pan 2025 import has no operational definition in our setting
In Pan, the fallible prior is a previous-session reconstruction and inflation unions
it with the current coarse scan plus divergence points. In our loop there is no
previous session — primitives are refit from the accumulating cloud every round. So
what, concretely, is the fallible prior? The only coherent reading: **the fitted
primitive's extrapolated surface** (the four unobserved faces of a box fit from two).
No interpretation states the rule that follows, and it matters for the demo: a menu
item on extrapolated geometry is legitimate as *exploration* ("go see the far side")
and illegitimate as *evidence* ("this is the orthogonal view of the code face" for a
face no pixel has ever observed). The part imports the pattern's name without the
rule. Also the evidential weight is thin and should be said plainly: Pan's measured
contribution is ~1 pp of coverage, simulation-only, synthetic Gaussian pose noise —
the synthesis itself instructs "treat their robustness numbers as a lower bound on
difficulty." Importing it as cheap, directionally-right hygiene: fine. Citing it as
if it derisks the loop's prior-trust problem: not supported.

### 3.4 "Sidesteps most of the fused-offset hazard by construction" — overstated
Live-frame locally-refit normals fix the *scoring/validation* side: incidence is
measured against geometry from the same frame, no cross-source mismatch. They do not
fix the *commanded pose*: the menu's view poses are still derived from primitives
fit to the fused accumulating cloud, so a 4 cm fused-frame error still puts the
camera 4 cm wrong relative to reality — the live-frame check can only detect it
after arrival. The synthesis's verdict was explicitly hybrid: compute locally AND
carry Tarbox-style margin in the admissible set AND chase the registration defect
(it cannot be sensor extrinsics — D405's depth imager is the colour sensor). The
part imports rung 1 of 2 and upgrades "avoids cross-source mismatch in the score" to
"sidesteps most of the hazard." Also a wording nit that matters for citation
hygiene: Nguyen's machinery is σ-weighted ICP/fusion, not view scoring — the
synthesis says so (§4.3: "into pose estimation and fusion, not into a view score");
this is a pattern-level analogy and should be written as one.

### 3.5 Set cover's role in a QA loop is a category slip as cited
The greedy tax (12.5–40% extra *executed views* in full-coverage plans) is the
justification given for exact re-solving — but in this loop, set cover is menu
*pruning* (dedup to 6–12 items), and what gets executed is whatever the LLM picks,
question-driven, one view at a time. The tax number does not measure our cost of
greediness; the honest justification is simply "exact costs ~1 s, so why not."
Deeper: coverage-minimal dedup optimizes the wrong objective for QA — it can prune
away the redundant alternative views of exactly the face the question targets, and
the synthesis's own contested point 4 records that Gospodnetić's experts *chose*
1.5–2× the optimum because minimal sets are worse to execute; the small-menu
pressure here comes from agent context (K≈2 beats K=5), a different argument that
must be made on its own terms. The part should state set cover as
coverage-guaranteed pruning subordinate to the question-driven selector, and keep
question-relevant redundancy exempt from dedup.

### 3.6 "No fundamental open choice" is false: the stop policy is one
Three interpretations, three stop stacks: academia §2.1 (plan exhaustion / LLM stop
action / hard cap), Anton Part A (menu stability + question answered + hard cap),
industry (MS@δ map stabilization for the scan phase + explicit stop action for the
agent phase). These are not the same loop. And stopping is a *documented* failure
axis for this architecture: the blind decider's measured deficit is decisiveness
(MapGPT −3.2 SR, +5.8 OSR — it keeps exploring), academia's own §1.3 says the stop
"cannot be left to the LLM's own judgement alone," and MS@δ's honest caveat is that
it stops at 9.0 views where 15.4 was optimal — it measures map stability, not
answer quality. A QA loop needs an *answer-driven* stop arbitrated against a
geometry-driven one, and nobody has written the arbitration. That is an open choice,
it is fundamental to the demo (the failure mode is the robot orbiting a box it has
already answered), and it deserves a scripted-policy experiment in the skeleton week.

### 3.7 "Integrate into the partial cloud" names no registration method
The scaffold owns the accumulating cloud, and the accumulating cloud's registration
is the single thread the ~4 cm offset investigation points at (not extrinsics — the
pipeline). Kriegel is cited in the corpus for both failure directions: no ICP → ~4 mm
per-scan misalignment; too much ICP → model over-contraction. Kinematics + hand-eye
only? ICP-refined? σ-weighted fusion? The part is silent, yet every downstream
number (E1's fit-vs-coverage curve is run on *synthetically* jittered clouds) rests
on what the real registration does. One sentence and one measurement (registration
residual per round, logged by the recorder from day one) closes this.

### 3.8 Minor — provenance flag on the Vasquez-Gomez stop
The exhaustion-as-valid-stop pattern is read via Peuzin-Jubert §4.3, not from
Vasquez-Gomez's own text. The pass's own hygiene rule (Wang T-Mech: "structure
trusted, tables not"; refuse third-hand numbers) should apply symmetrically: fine to
adopt the pattern, cite it as "via Peuzin-Jubert." Zero numbers rest on it, so this
is a citation-honesty nit, not a design risk. Note also Scott 2001's alternative
(objective-switch to max-coverage when no feasible cover exists) was dropped without
comment — acceptable, but record it as a decision.

## 4. Do the proposed experiments prove/disprove, or only confirm?

Taken across the three interpretations (the part's own doc has none — issue 3.1):

- **Anton E5** (loop vs one-shot dense survey at matched budget) is a genuine
  falsifier — it can kill the loop's premise, and its failure consequence is named
  (revert to survey-then-inspect). **Adopt as the part's primary experiment.**
- **Industry E1** (smoke test) disproves only "it integrates" — necessary, weak.
- **E2/E1 partial-fit sweeps** falsify the loop's input assumption (fits usable at
  low coverage) — they live in `partial-fit` but their disproof condition is what
  actually decides whether this loop can exist. The dependency should be explicit.
- **Missing falsifier #1 — stop quality:** no experiment anywhere scores the stop
  decision itself (stopped-too-early / too-late vs oracle stop on replay episodes).
  Cheap on Tier 1; directly targets 3.6.
- **Missing falsifier #2 — fallible-prior value:** the Pan-style rule is imported
  but never tested. A/B the conservative-inflation discipline on/off over replay
  episodes (does it ever change a chosen view? does it ever prevent an
  extrapolated-face evidence claim?). If it changes nothing at our scale, it's a
  free hygiene line; if it does, it earned its citation. Either result is worth one
  table row.
- **Rendered evidence + usage story:** the demo-v2 protocol (moved/rotated box,
  grounded 2×2, failure narratives) covers the architect's standard for the loop's
  output; the loop's *internal* evidence (per-round cloud, fit, menu snapshots from
  the recorder) is implied by the recorder design and should be named as the
  rendered artifact for this part.

## 5. What would make this part sound

1. Attach Anton E5 (with its revert-to-survey disproof) and industry E1 to the part;
   delete "open choice: none fundamental."
2. Add the analytic-visibility commitment: matrix by construction at primitive
   level, SEE mesh-free fallback; keep "~1 s" scoped to the solve.
3. Operationalize fallible-prior: per-primitive observed-vs-extrapolated support
   tags; extrapolated ⇒ exploration-only, never evidence; evidence requires
   live-frame validation at arrival. Cite Pan as pattern, ~1 pp, sim-only.
4. Reword 3.4: live-frame normals *detect* fused-frame error at the evidence view;
   margin (Tarbox erosion) carries the *prevention*; both imported.
5. Reframe set cover as coverage-guaranteed menu pruning subordinate to the
   question; exempt question-relevant redundancy; drop the greedy-tax number as
   justification (keep it as background).
6. Write the stop arbitration (answer-driven ∧ geometry-driven ∧ hard cap) as an
   explicit open choice with a Tier-1 stop-quality metric.
7. Name the registration method and log its residual per round.

None of these threaten the part's existence — the skeleton is the field's mainstream
plus one well-scoped novelty, exactly as claimed. They are the difference between a
part that inherits the pass's evidence discipline and one that waves at it.
