# View-planning POC — experiment roadmap

**Hypothesis under test:** for a growing object cloud from a recorded inspection run, a
mesh→sample→normal-offset→quality-weighted-greedy pipeline produces a small, stable set of
near-orthogonal camera views — and it does so from an *early* step of the run. This is the
alternative to the parked primitive-decomposition track (`inspection/fit/`): instead of fitting
primitives and deriving views closed-form, we plan directly on the reconstructed surface.

**Core question:** after which step of the run is the accumulated geometry good enough for a
good plan?

**Inputs (fixed):** accumulated step clouds `derived/<run>/steps/step_NNN.ply` for
`2408-cup3` (000–029), `2408-cup4` (000–032), `2408-cup5` (000–029; raw run has no 029 dir —
documented quirk, the derived PLY exists). Base frame, Z-up, metres, no normals. 1k–7k points,
~2.5 mm spacing, 3–6 mm registration noise. D405 standoff window 0.07–0.20 m, sweet spot
0.08–0.12 m — treated as physical constants, not knobs.

**Outputs (fixed by settled requirements):** plans as `derived/<run>/viewplan/<variant>/step_NNN.json`
(T_base_cam, OpenCV convention, 4×4, base frame, metres) + meshes + renders + metrics; rerun
per run at `derived/<run>/rrd/<run>-viewplan.rrd`. No IK, no reachability; single guard =
cull below-table / looking-up-from-under-table views.

---

## 1. Variant matrix

The expensive artifact per (run, step) is the **measurability matrix** (candidates × surface
patches, with the *continuous* incidence angle and gate flags stored per pair, not a boolean).
It is computed once per (run, step, mesher) and every selector variant re-reads it. Selector
variants are therefore nearly free; mesher variants pay full price. This dictates the shape of
the matrix: several selectors on one mesh, one alternative mesh under one selector.

| ID | Mesher | Selector objective | Incidence gate | Runs full grid? | Renders |
|---|---|---|---|---|---|
| `binary` | Poisson (+fallback) | greedy, binary marginal area | 60° | yes — every step, all 3 runs | anchor steps only |
| `cos` **(primary)** | Poisson (+fallback) | greedy, cos-weighted marginal quality | 60° | yes | all milestone steps |
| `cos2` | Poisson (+fallback) | greedy, cos²-weighted | 60° | yes | anchor steps only |
| `cos-altmesh` | alternative mesher (BPA or alpha-shape, one pick — build team's call) | greedy, cos-weighted | 60° | yes | anchor steps only |
| `cos75` | (reuses `cos` matrices) | greedy, cos-weighted | 75° | re-selection only | none |

Greedy objective, `cos` family: at each iteration pick the candidate maximizing
Σ_patches area(p) · max(0, gain in q(p)), where q(p) = cosᵏ(θ_best-selected-view(p)) and pairs
failing standoff / occlusion / incidence-gate contribute zero. `binary`: q(p) ∈ {0,1}.
Termination: marginal gain < ε or the ≤12-view cap, whichever first (record which fired).

**Why each variant earns its slot**

- **`cos` (primary).** The hypothesis itself. Pass-1 finding 6 (REPORT.md §2.6) recorded the
  critique that boolean-gate + set cover makes orthogonality a floor, not an objective —
  "covered" at ~τ off-normal recreates the original complaint inside our own machinery. Our
  answer is structural: **the greedy IS quality-ranked** — the selection objective is the
  continuous orthogonality quality, the gate only zeroes inadmissible pairs. Set cover here is
  not the ranking; it is the pruning applied to a ranked field, exactly the subordination the
  pass-1 review demanded.
- **`binary` (ablation control).** Proves the critique has teeth in *our* data, with numbers:
  same mesh, same candidates, same gate, only the objective degraded to plain coverage. Expected
  outcome: similar n_views and coverage@60, visibly worse mean/median incidence of covered area
  and worse worst-patch. If `binary` ≈ `cos` on incidence metrics, the weighting was decoration
  and we say so.
- **`cos2` (sharpness probe, pre-registered kill).** How hard should the objective push toward
  orthogonality? cos matches the physical 1/cos GSD loss; cos² penalizes obliquity harder and
  may buy better worst-patch at the price of more views. Cheap (an exponent). **Kill rule:**
  if pose-set distance D(`cos`, `cos2`) < δ_stable at ≥80 % of steps across all runs, the
  objective is insensitive to sharpness in this regime — reported as one line + one table row,
  dropped from all headline figures.
- **`cos-altmesh` (meshing confound probe).** The core question is "when are the *points*
  enough", but Poisson on 1–2k-point partials can hallucinate (blown-up bubbles, phantom area) —
  the coverage denominator and the normals both come from the mesh. A second, mechanistically
  different mesher separates the two failure sources: if `cos` and `cos-altmesh` plans agree at
  step k while each still differs from its own final plan, remaining instability is geometric
  (points); if they disagree with each other, meshing is the confound and the "good enough"
  step for Poisson is not a statement about the data.
- **`cos75` (free sensitivity row, not a full variant).** The 60-vs-75 gate question from the
  τ-debate (pass-1 finding 5: published knees 45–75°, none transferable). Because the matrix
  stores continuous incidence, gate 75 is a re-selection pass over existing matrices — zero
  raycasts, no renders, no rrd channel. Prediction worth checking: under cos weighting the gate
  is not load-bearing (a 75° pair carries weight cos75° ≈ 0.26 and rarely wins), so
  D(`cos`, `cos75`) should be small; under `binary` it would be large. If confirmed in the
  table, the τ knob stops being an argument.

**Killed as vanity** (do not build, do not run):

- *Exact ILP vs greedy.* Pass-1 already established ILP solves in ~1 s at 10–100× our scale;
  but the exact solver can't express the continuous-quality objective that IS the hypothesis,
  and a coverage-optimality gap is not a POC question. Greedy only.
- *Poisson depth / parameter sweeps.* Tuning, not an experiment. Build team picks one depth +
  one density-trim rule, records it in the plan JSON, done.
- *Standoff sweeps.* 0.07–0.20 m is the D405's physics. Candidates are generated inside the
  sweet spot with the window as hard gate; sweeping it tests the datasheet, not the planner.
- *Candidate-count / sampler sweeps (Poisson-disk vs uniform surface sampling).* Pick one,
  record the seed and count, report wall-time. If coverage-achievable (§2, M8) shows candidate
  starvation, that is a bug to fix, not a variant to publish.
- *A 5th+ selector (submodular variants, lazy vs plain, orderings).* Ordering of the output
  list is presentation; lazy evaluation is an optimization. Neither changes the plan.

Full grid: 5 variant rows × 3 runs × every step (30/33/30) — matrices computed once per
(run, step) for Poisson + once for altmesh; expected wall-cost dominated by ~2 × 93 matrix
builds at seconds each (600 k-ray order of magnitude per matrix — well inside
`RaycastingScene` throughput). Total budget target: under an hour on the workstation; the
per-step wall-time metric (§2, M9) verifies rather than assumes this.

---

## 2. Per-step metrics

One record per (run, step, variant) appended to `derived/<run>/viewplan/<variant>/metrics.jsonl`
(plus echoed inside `step_NNN.json`). Definitions are binding — the build team implements
these, not approximations of these.

Per surface patch p (sampled point + area weight + normal), the **achieved incidence**
θ*(p) = min over *selected* views v of θ(v,p), over pairs passing all gates (standoff window,
incidence gate, occlusion raycast, table guard). Patch covered ⟺ θ*(p) ≤ 60°.

- **M1 `n_views`** — plan size; also which terminator fired (gain-ε vs 12-cap).
- **M2 `coverage@60`** — area-weighted fraction of mesh patches covered. Same pass computes
  `coverage@45` and `coverage@75` from the stored angles (free; context columns).
- **M3 incidence of covered area** — area-weighted mean and median of θ*(p) over covered
  patches. The headline quality number; this is where `binary` vs `cos` separates.
- **M4 worst-patch incidence** — p95 and max of θ*(p) over covered patches (max alone is one
  noisy patch; p95 is the honest tail).
- **M5 plan stability — pose-set distance (concrete definition).** A view is
  v = (p ∈ ℝ³, b ∈ S²): camera position + boresight (+Z column of T_base_cam). Roll excluded —
  candidate generation fixes roll deterministically, so roll churn is not plan churn.
  Pairwise: d(v,w) = ‖p_v − p_w‖ + λ·arccos(b_v·b_w), **λ = 0.1 m/rad** (at the 0.1 m sweet-spot
  standoff, 1 rad of boresight change displaces the viewed surface by ~0.1 m — the two terms
  are commensurate by construction). Set distance: symmetric Chamfer
  D(A,B) = ½·( mean_{a∈A} min_{b∈B} d(a,b) + mean_{b∈B} min_{a∈A} d(a,b) ), plus |n_A − n_B|
  reported alongside. Two curves per (run, variant):
  **churn** D(plan_k, plan_{k−1}) and **distance-to-final** D(plan_k, plan_K).
  **δ_stable = 0.02** (20 mm-equivalent ≈ 20 mm position or ~11.5° boresight) — provisional,
  frozen after eyeballing one run, then applied unchanged to all three.
- **M6 cross-variant distance** — D(`cos`_k, `binary`_k), D(`cos`_k, `cos2`_k),
  D(`cos`_k, `cos-altmesh`_k), D(`cos`_k, `cos75`_k) per step. Feeds Q2/Q3/Q5 and the `cos2`
  kill rule.
- **M7 mesh health** — vertex/triangle count, is_watertight, connected components, total area,
  and **area ratio vs the same variant's final-step mesh**. Poisson blow-up on sparse steps
  shows up here first; without M7, M2's denominator silently moves and coverage curves lie.
- **M8 achievability decomposition** — area fraction of patches with *zero admissible
  candidates in the full candidate set*, broken down by first-failing cause (standoff / table
  guard / occlusion / incidence). Defines `coverage_achievable` = covered / (1 − unachievable).
  The cup's table-contact ring is physically unviewable from any legal pose; M8 measures that
  ring instead of letting it be argued about. Note for the bar (§6): the settled ≥95 % target
  is reported **both** raw and achievable-normalized — both numbers, always, no choosing.
- **M9 planning wall-time** — split: load, mesh, sample, candidate-gen, matrix (raycast),
  greedy. Per step, stored. The matrix column is the one expected to dominate.
- **M10 render truthfulness** (milestone steps only, §3) — per rendered view, the fraction of
  its claimed newly-covered patches whose centers re-project inside the image and pass a depth
  test against the render's depth buffer. Target 100 %; any miss is listed by view id. This is
  the "every render actually frames the surface it claims to cover" bar made mechanical
  instead of eyeballed.

---

## 3. Render & rerun cadence

Rendering every selected view at every step is ~1000 images per run per variant — noise, not
evidence. Cadence:

- **Milestone steps** M(run) = {0, 2, 5, 8, 12, 16, 20, 25, last} ∩ available:
  cup3 → {0,2,5,8,12,16,20,25,29}; cup4 → {0,2,5,8,12,16,20,25,32}; cup5 → {0,2,5,8,12,16,20,25,29}.
- **Anchor steps** A(run) = {2, 12, last} — early / mid / converged.
- **`cos` (primary):** at every milestone step — one shaded offscreen render per selected view
  (RaycastingScene + numpy shading, no OpenGL) **plus** one coverage heatmap (mesh colored by
  θ*(p), uncovered area in a screaming color). ≈ 9 steps × ≤12 views × 3 runs ≤ ~330 images.
- **`binary`, `cos2`, `cos-altmesh`:** anchor steps only — heatmap + per-view renders.
  ≈ 3 variants × 3 steps × 3 runs × ≤12 ≤ ~330 images. Enough for visual variant comparison
  without a flood. `cos75`: no renders.
- **Renders live under** `derived/<run>/viewplan/<variant>/renders/step_NNN/view_MM.png` +
  `heatmap.png`, mirroring the `fits/renders` habit.
- **Rerun — every step, no exceptions:** `inspection/viewplan/workspace.py` builds
  `derived/<run>/rrd/<run>-viewplan.rrd` reusing `record/rr` builders (only strictly additive
  functions may be added to `inspection/record/rr/log.py`). Per step on the int "step"
  timeline: accumulated cloud, that step's mesh, and **every planned view as a logged pinhole
  camera + frustum for every full variant** on per-variant channels (swept like `fits/`
  channels), so Anton can scrub the timeline and snap into any planned camera at any step
  interactively. Cameras are transforms — logging them every step is cheap; that is why images
  get a cadence and cameras don't. Main `workspace.py` and its rrd untouched.

---

## 4. Per-run analysis questions

**Q1 — after which step is the geometry enough? (the core question)**
Operationalized: **k\*** = the smallest k such that for all j ≥ k,
D(plan_j, plan_final) ≤ δ_stable **and** |coverage@60_j − coverage@60_final| ≤ 2 pp
(both on the `cos` variant). Answered by: the per-run **convergence dashboard** — four aligned
curves vs step (n_views, coverage@60 raw + achievable, mean incidence M3, distance-to-final +
churn M5) with k\* marked — plus the milestone render strip visibly freezing after k\*.
Headline: k\* for each of cup3/4/5 and the median. "Stable plan by mid-run" (bar B1) means
k\* ≤ 15.

**Q2 — does quality weighting matter (the ranking-vs-pruning critique, answered with data)?**
`binary` vs `cos` at final step, per run: paired M1–M4 table + overlaid θ*(p) histograms +
side-by-side heatmaps. Win condition for the critique's relevance: `cos` beats `binary` on
M3/M4 at comparable M1/M2. If it doesn't, that is reported as-is.

**Q3 — is early instability geometry or meshing?**
D(`cos`_k, `cos-altmesh`_k) curve vs each variant's own distance-to-final, plus M7 area-ratio
curves. Cross-mesher agreement before self-convergence ⇒ points are the limiter (k\* is about
the data); persistent cross-mesher disagreement ⇒ Poisson is a confound and k\* is
mesher-conditional — said explicitly either way.

**Q4 — is ≤12 views @ ≥95 % physically feasible here?**
M8 decomposition per step: how much area is unviewable by *any* legal candidate and why. If the
table-ring makes raw 95 % unreachable, the dashboard shows exactly how large the ring is and
what achievable-normalized coverage the plan reaches.

**Q5 — is the 60° gate load-bearing under cos weighting?**
D(`cos`_k, `cos75`_k) + paired M1–M4 at final steps. Small distance ⇒ the gate value is not
worth arguing about once the objective is continuous; large ⇒ τ_menu is real and gets its own
follow-up outside this POC.

**Q6 — do the three runs agree?**
k\*(cup3) vs k\*(cup4) vs k\*(cup5), and final-step M1–M4 across runs. Same object, three
recordings, 3–6 mm registration noise ⇒ spread in k\* bounds how much the answer depends on
the recording rather than the object. Three agreeing runs is the difference between a result
and an anecdote.

---

## 5. Evidence list for Anton (the judging pack)

Everything below must exist before showing; items 1–4 are the 10-minute read, 5–8 the drill-down.

1. **RESULTS.md** (`docs/research/view-planning/poc/RESULTS.md`) — one page: k\* per run,
   bar verdicts (§6) as pass/fail with numbers, the Q2 verdict sentence, the Q3 verdict
   sentence, known failures listed bluntly. No optimism.
2. **Convergence dashboards** — one figure per run (Q1), k\* marked.
3. **Variant table + Q2 histogram/heatmap pair** — final-step M1–M4/M9 for all five variant
   rows × three runs; the single plot that answers the pass-1 critique.
4. **Milestone render contact sheets** — per run, steps × selected-views grid for `cos`, with
   the coverage heatmap column; anchor-step strips for the other variants.
5. **Rerun files** — `derived/<run>/rrd/<run>-viewplan.rrd` × 3: scrub the step timeline,
   watch the plan converge, snap into any planned camera. This is the ultimate gate; the rrd
   must open with a working blueprint, zero manual setup.
6. **Render-truthfulness report** — M10 per rendered view, target 100 %, misses itemized.
7. **Machine-readable trail** — `viewplan/<variant>/step_NNN.json` (poses in the recorded
   OpenCV T_base_cam convention + per-view claimed patches + metrics echo), `metrics.jsonl`,
   saved meshes. Every number in items 1–3 recomputable from these.
8. **Wall-time table** (M9 aggregate) — proves per-step planning cost is compatible with
   eventually running this inside the live loop, which is the point of asking k\* at all.

## 6. Bars & honesty notes

Internal bar (checked before showing, from the settled requirements):
**B1** stable plan (k\* per Q1) with ≤12 views by mid-run (k\* ≤ 15);
**B2** ≥95 % mesh area covered at ≤60° at final step — reported raw **and**
achievable-normalized (M8), both numbers shown, discrepancy explained by the measured
unviewable-area decomposition, not by prose;
**B3** M10 = 100 % (every render frames what it claims).

Known threats, tracked not hidden: Poisson phantom area on sparse steps (M7 catches it,
`cos-altmesh` cross-checks it); registration noise 3–6 mm vs δ_stable 20 mm (if churn never
drops below δ_stable, the honest conclusion is that stability is noise-bound — reported, not
tuned away); the unseeded-RANSAC warning in `cell/geometry.py:fit_table` (table plane for the
guard must be seeded/pinned per run so plans are reproducible run-to-run); cup5's missing raw
step-029 dir (derived PLY exists; planner consumes derived only — noted so nobody "fixes" it).
