# Round 6 — Verification: `loop-skeleton`

**Part under test:** wire the full incremental loop (scan → fit → derive → menu → pick →
execute → refit) end-to-end in week 1, in the kinematic sandbox, on the demo box,
planes-only, scripted auto-picker instead of the LLM; includes the scan-phase
map-stabilization stop rule, the agent's explicit stop action, and the benchmark-grade
recorder (RGB/depth/pose/intrinsics per view) wired in from day one so every run feeds
the Tier-1 dataset.

**Sources checked:** round-4 feedback (binding), all three round-5 interpretations
(anton §Part A/E5; academia §2.1; industry §Part A/§1.4.4/§1.5), Branch 4 synthesis
(`view-set-selection-under-constraints.md`, esp. Q14 stopping and §6.2), Branch 2
synthesis (`benchmarking-view-planning.md`, Q6 and the MS@δ diagnostic), Branch 6
synthesis (stop action, prompted-VLM-vs-random), framing doc, and the actual repo
(`inspection/motion/{ik,rrt,plan,execute}.py`, `inspection/perception/capture.py`,
`inspection/view/viewsphere.py`, `inspection/cell/`).

**Verdict: needs-work.** The trunk is grounded and the build order is right; the
defects are in the stop-rule design (which can quietly re-import survey-first), an
overstated recorder claim, an unspecified scripted picker (the one ADAM-regression
vector in this part), and an over-readable sandbox experiment.

---

## 1. Provenance — grounded or invented in round 5?

Grounded, item by item. Nothing load-bearing was invented in round 5:

- **The incremental loop itself** is the architect's binding round-4 correction
  ("we do NOT have the scanned object up front... it is a loop"), and once flipped,
  it lands on the field's best-precedented skeleton: the canonical NBV loop
  (Kriegel, Wu, SEE via academia §1.2), Pan 2025's treat-prior-as-fallible union as
  the only measured fallible-prior technique in the corpus (Branch 4 §6.2), and
  per-round exact set-cover at ~1 s (Pan 2025/Gurobi; MA-SCVP <10 s; Glorieux 15.6 s —
  all at instance sizes above ours). Re-solving every round is legitimately free.
- **Scripted auto-picker first** is doubly grounded: (a) no published result shows a
  *prompted* VLM beating random view selection with variance (Branch 6 Q15), so a
  scripted/oracle selector arm is mandatory in the headline 2×2 anyway — building it
  first means the control arm exists before the treatment arm; (b) it removes LLM
  variance from the integration test. This is the field's hold-one-stage-fixed
  discipline, not a round-5 invention.
- **MS@δ map stabilization** is ObjView-Bench's rule, carried with its own honest
  caveat in Branch 2 Q6 (residual gap 0.0026 main pool vs 0.0081 slow-saturation —
  not difficulty-invariant). Real, citable — but see Issue 1 for how it is being used.
- **Explicit agent stop action**: MapGPT's always-listed stop item; e3vs-bench's
  stop + hard cap; plus the abstention finding (frontier models abstain correctly
  16.7–33.3% unless the interface makes stop/abstain first-class). Grounded.
- **Recorder schema** (RGB, depth, pose, intrinsics, nothing else) is the architect's
  own binding ruling in the feedback, and the dense-capture-replay pattern it serves
  is precedented (MA-SCVP's exhaustive real-world capture protocol).
- **Week-1 feasibility** is engineering, not literature — but it is supported by the
  repo: IK/RRT/planning/collision already exist (`inspection/motion/`), capture
  exists (`inspection/perception/capture.py`), the uniform viewsphere exists
  (`inspection/view/viewsphere.py`), and the feedback states travel-costing in the
  sandbox is "already implemented." Planes-only also dodges the field's actual
  bottleneck — the visibility computation (Branch 4 finding 3: ray-tracing hours per
  object) — because plane-face visibility is analytic. The week-1 target is
  plausible, not bravado.

---

## 2. Issues

### Issue 1 — the "scan-phase MS@δ stop rule" can smuggle survey-first back in, and the three interpretations disagree on the stop design (main defect)

The part description inherits industry's **two-phase** stop structure (§1.4.4: scan
phase stops on MS@δ; agent phase stops on explicit stop). Anton's own Part A says
something different: stop on **menu stability + question answered + hard cap**, with
MS@δ only as a "mechanical fallback." Academia §2.1 says a third thing: "plan
exhaustion, LLM stop action, or hard cap." Three stop machines, one part.

The two-phase reading is the dangerous one. A dedicated scan phase that runs until
map stabilization *before* the menu loop begins is a mini complete survey up front —
the exact premise the architect's feedback overrode ("There is no complete
reconstruction from the beginning"). And MS@δ is a *reconstruction-saturation*
criterion: at its published operating point it stops at ~9.0 views (of 128) on the
main pool. On a tabletop QA task where 1–2 extra views capture essentially all
achievable gain (GraspView/TwNV, anton §1.2), burning ~9 views on a stabilization
scan before the first menu would also hand every matched-budget comparison (E5, the
headline E8/E6) to the uniform baseline by construction.

**Fix (cheap, before the skeleton locks interfaces):** the loop's primary stop is
fit/menu stability + answered + hard cap; MS@δ is demoted to a fallback termination
guard; any bootstrap scan is a fixed tiny count (1–3 views to seed the first fit),
never run-to-stabilization. Carry the ObjView-Bench difficulty-variance caveat
wherever MS@δ appears.

### Issue 2 — "every run feeds the Tier-1 dataset" is overstated on two counts

(a) Week-1 runs are in the **kinematic sandbox** — there is no real RGB/depth to
record. The recorder can be schema-exercised, not fed. (b) Even on the real robot, an
episodic run records only the views the policy actually chose: a sparse,
policy-biased set. Tier-1 replay requires a **dense capture superset** so that a
*different* policy's requests can be served (MA-SCVP protocol; academia §1.5's
union-capture/snap-error correction). Sparse episode logs feed Tier-0 image sampling
and evaluation archives — they cannot serve Tier-1 replay for other arms. The
feedback itself files "every application run records benchmark-grade data" under
**"later."**

**Fix:** restate the recorder's day-one purpose as "the loop and the dense-capture
tool share one writer and one schema," and if the Tier-1 claim is to be true in week
1 at all, make the first real dense capture of the demo box an explicit week-1
deliverable alongside the sandbox skeleton.

### Issue 3 — E5 in the sandbox is asymmetric evidence, and the scan-step source is unspecified

Anton's E5 (loop vs one-shot dense survey at matched budget, in the sandbox) has a
real disproof condition — good. But run on clean synthetic clouds it can only
**kill** ("the loop fails even on clean data") and cannot **confirm** the premise:
rounds-to-stable-menu on real clouds is driven by dropout, registration jitter, and
partial-fit error, none of which the sandbox exhibits. A sandbox pass must not be
reported as "the incremental premise holds." Related gap: nothing in the part says
what "scan" returns in the sandbox — synthetic raycast clouds or replayed real
captures. The repo already has real captures and the documented ~4 cm offset;
replaying real single-view clouds through the sandbox loop costs nothing at robot
time and makes even the week-1 skeleton exercise the real-defect constraint.

**Fix:** state the scan source; prefer replayed real captures where they exist; mark
E5's sandbox verdict as falsification-only, re-run on real captures once Part B lands.

### Issue 4 — the scripted auto-picker is unspecified, and it is this part's one ADAM-regression vector

If the week-1 picker is a canned sequence ("visit faces 1–5 in order"), it is a
per-object recipe — and industry explicitly nominates the scripted arm (S1) as its
candidate for **what actually ships demo v2**, so the script has a real chance of
becoming the product. One sentence closes the hole: the picker must be a generic
rule over the *menu* (e.g., pick the highest-value unvisited menu item; stop when
the target is read), recomputed from the fitted primitives each round, with zero
object-specific constants. Then it is a primitive-level deterministic rule —
explicitly permitted by the architect — and it doubles as the oracle/scripted
selector arm of the headline 2×2 with no extra work.

### Issue 5 (minor) — stop policy should be a swappable component, like the selector

Branch 2 Q6's recommendation binds the future harness: hold stopping fixed
(benchmark-side) when comparing action spaces; measure agent stopping separately as
its own number. If the week-1 skeleton hard-wires stop logic into the loop body, the
harness fights it in week 3. Make the stop policy injectable now, same as the picker.

---

## 3. Constraint and scope compliance

- **Incremental partial-cloud fitting:** yes — the loop is literally the corrected
  premise (conditional on Issue 1's fix; the two-phase reading would violate it).
- **Real sensor defects:** deferred to Part B by design; acceptable for a skeleton
  *if* Issue 3's scan-source fix is taken; otherwise week 1 proves integration only.
- **Kinematic reachability:** in the pipeline (verify-reachability before menu), in
  the sandbox with exact IK+collision — grounded (Scott 2001 filter-at-generation;
  ObjView-Bench 45.4/128 base rate; flagged-not-silently-removed discipline).
- **Discrete LLM menu selection:** respected — no LLM in week 1; the menu machinery
  is built LLM-shaped from the start; geometry is never exposed to any model.
- **Benchmark-as-auxiliary:** respected — the recorder is the only benchmark surface
  in this part, and it is cheap plumbing (modulo Issue 2's honesty fix).
- **No VLA / RL / defect detection:** clean.
- **ADAM regression:** clean *conditional on Issue 4* — the scripted picker is the
  single place a per-object recipe could leak in and then ship.

## 4. Would it survive the architect's standard?

- **Falsification attempt:** present (E5's disproof condition; industry E1's
  integration smoke test can genuinely fail), with Issue 3's asymmetry stated.
- **Rendered evidence:** demonstrable — an end-to-end sandbox episode with per-round
  menus, moves, travel, and recorder output is exactly the artifact to show.
- **Concrete usage story:** demo v2 runs on this skeleton; every later part (B–G)
  swaps into a running loop instead of assembling at the end. The strongest property
  of the whole month's plan, and it survives scrutiny.

**Bottom line:** keep the part; fix the stop-rule design (Issue 1) and the recorder
claim (Issue 2) before the week-1 build locks interfaces, specify the picker rule
(Issue 4) and the sandbox scan source (Issue 3) in one sentence each. With those
fixes this part is the soundest in the methodology; without Issue 1's fix it can
quietly contradict the premise the whole design change was made for.
