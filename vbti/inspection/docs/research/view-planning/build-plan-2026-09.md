# Build plan — September 2026

**Goal:** working incremental inspection system that arrives at geometry-specific views.
**Claim:** geometry-conditioned menu beats the deployed 12×30°×3 uniform viewsphere, proved by the component benchmark.
**Milestone:** demo v2 — box with codes; evidence image = orthogonal view of the coded side, robust to moving + rotating the box.
**Out:** VLA, RL, defect detection, deep sensor-noise modeling. All other machinery unchanged.

Derived from [`passes/pass1/methodology_v1.md`](passes/pass1/methodology_v1.md) (parts P0–P8, experiments X0–X18) and [`passes/pass1/REPORT.md`](passes/pass1/REPORT.md). This doc is the executable distillation; drill into methodology_v1 for every contested decision and verifier issue.

---

## The system being built (one line per part)

| Part | What it is |
|---|---|
| P0 `bench` | Two week-1 numbers we otherwise guess: depth admissibility angle (τ_scan) + RGB legibility-vs-angle curve |
| P1 `loop` | The skeleton: capture → register → refit primitives → derive views → menu → pick → execute → repeat. Owns the recorder |
| P2 `fit` | Primitives from partial growing clouds, with a calibrated "trust this fit?" gate. **The load-bearing wall** |
| P3 `identity` | "The code face" stays the same menu entry across refits (content-addressing + hysteresis) |
| P4 `factory` | The intelligent viewsphere: closed-form per-primitive views, gate + rank scoring, reachability, abstain/fallback |
| P5 `interface` | What the LLM sees: two-stage menu, stable names, stop/abstain/anti-repeat, serializer swappable (angles vs words = X10) |
| P6 `harness` | The 3-tier benchmark: recorder, replay, scoring, arms, stats. Auxiliary in effort, arbiter in role |
| P7 `comparison` | The one headline experiment: spaces × selectors at matched budget |
| P8 `demo` | Demo v2 acceptance protocol + one-object generalization probe |

## Week by week

**Week 1 — cheap kills.**
- X1 fit shakedown on real fused PLYs (2 h: Open3D + CGAL/PCL + EMS).
- X2 depth bench + X3 RGB legibility bench (half a day each) — X2 can veto the camera while swapping is free; X3 can *soften the orthogonality premise itself* (if codes read fine at 40° we relax menu density).
- X0 error-budget memo (one page) — decides whether the comparative claim is even resolvable, and against what.
- P1 loop skeleton in the kinematic sandbox on replayed captures; first dense capture recorded.
- **Gate G1:** claim target fixed; camera go/no-go on depth axis.

**Week 2 — the wall.**
- X4 fit fidelity vs coverage (does the derived view beat the grid's snap error on *partial* clouds?), X5 maturity-gate calibration, X6 identity churn; P4 factory + X8 construction audit (achieved vs planned incidence on real images).
- **Gate G2:** evidence bar reachable, usable trust signal, flips below tolerance. Failure ⇒ re-scope to the null-world deliverable *before* harness effort is spent.

**Week 3 — interface + instrument.**
- P5 + P6 core; X10 menu-representation A/B (angles vs words — your hypothesis, teacher-forced frozen decision points); X11 two-stage vs flat; X7 flip-tolerance; X14/X15/X16 harness self-checks.
- **Gate G3:** frozen-decisions file versioned (thresholds, arms, budgets, losing conditions) — no comparative run before it.

**Week 4 — the claim + the demo.**
- X12 headline comparison (space × selector, oracle row included), X13 selector floor (random/scripted/LLM — if LLM ties, ship geometry-selects + LLM-answers), X17 demo v2 acceptance (K×M placements, grounded 2×2 + abstention), X18 generalization probe (second box + one cylinder episode, zero code changes allowed).

**Null-world floor (pre-committed):** even if every comparison ties, the month ships the working system, demo v2 with narrated outcomes, five never-published curves (X3/X4/X5/X6/X10), and the reusable 3-tier harness + capture set.

## Decisions needed from you at kickoff

The 10 open questions, with recommendations, are in [`methodology_v1.md §9`](passes/pass1/methodology_v1.md). The three that block week 1:
- **OQ-1** claim target: deployed grid (recommended) vs Tammes-matched — error budget says Tammes margin is unresolvable this month.
- **OQ-3** who sets τ_evidence ("orthogonal enough" for the demo): you as a requirement (≤15° proposed), not derived from the decode curve.
- **OQ-10** camera: X2 vetoes only the depth axis; RGB evidence quality stays your demo-v2 call.
The rest (OQ-2, 4–9) can be answered before their gates.
