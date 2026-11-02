# Round 5 — Interpretation (Anton)

**Date:** 2026-09-01
**Inputs:** contract [`geometry-aware-view-planning.md`](../../../../geometry-aware-view-planning.md) · my round-4 feedback [`../round4_accumulate/feedback.md`](../round4_accumulate/feedback.md) (overrides the contract where they conflict) · all six round-4 branch syntheses.
**Role:** architect. This replaces my earlier draft, which was written before the round-4 feedback existed. Design as it stands now: inspection = answering questions about objects (QA), not defect detection. Planning is an incremental loop — scan, fit primitives to the partial cloud, derive views, LLM picks from a menu, execute, refit, replan. The LLM sees the **intelligent viewsphere**: a small finite menu of precomputed good views (orthogonal, reachable, well-framed). It never sees primitives, coordinates, or parameter knobs. The month goal is a working system plus one comparative claim — geometry-conditioned view spaces beat the uniform viewsphere — proved by the 3-tier component benchmark. Demo v2: a box with codes on one side; after inspection the evidence image is the orthogonal view of that side, robust to moving and rotating the box.

---

## 1. What the reading does to the updated design

### 1.1 The founding premise survives, and the QA reframe makes it *stronger*

The geometry-agnostic sphere is the field's own rejected design, not just my complaint: Gospodnetić measured sphere sampling discarding 33–77% of candidates as inadmissible vs 0.08–32% for surface-anchored generation ("Space Sampling methods show no benefits and their use is not recommended"); Scott 2003 said the same in the canonical survey. And Branch 5's structural finding is the one I keep: **in view-space terms, the orthogonal view of a face is a measure-zero boundary element** — for a cube the corner regions have area, the six face-on views are points. The views we want are the seams of any sphere partition. No resolution of the {h,v} grid fixes that. Construction runs surface → view, never view-space → partition.

The QA reframe (my feedback) makes the orthogonality claim easier to defend, not harder. The old framing dragged in the defect-contrast question, which the literature genuinely cannot answer (the inspection-planning field never says "defect"; raking light points the other way for scratches). That whole tension is now out of scope. For **reading things off a surface** — codes, labels, text, logos — orthogonality stands on the one angle effect that IS proven: resolution. Pixel density on the surface degrades as 1/cos θ; measured 38,500 → 4,900 points on a fixed target from 0° to 80°. Foreshortening kills legibility. That is a measurement claim I can also verify myself in an afternoon on RGB (E6 below), instead of a defect-visibility claim nobody can source.

Consequence of the reframe for the sensor question: **depth exists to serve the loop (fitting), RGB serves the evidence.** Once we are at the good view, we care only about the RGB image. So the deep σ(range, θ) depth-noise characterisation campaign from my old draft is demoted — the literature only lends a formula template anyway, and I don't need the curve to build the system. What survives is a slim check inside the fitting work: is depth fill/quality at our scanning standoffs good enough to fit primitives on. Two D405 facts still bind and cost nothing to respect: Min-Z is 100 mm at full resolution (not the marketed 70), and the stereo invalid-depth band eats ~10% of image width at 10 cm. Camera choice is flexible per the feedback — if the D405's RGB limits evidence-image quality, we request a better camera; that's procurement, not research.

### 1.2 The incremental loop: the literature was read against the wrong premise, and flipping it mostly helps

Branch 4 argued "we have the object up front, so the iterative NBV apparatus is irrelevant." My correction: we don't. The loop is scan → fit → derive → pick → execute → refit. That flip does three things to the findings:

- **The iterative NBV structure IS our structure** — with primitive fitting in the middle and no learned selector. The field's machinery for it transfers directly: per-round re-solving is cheap (exact ILP set cover ~1 s at our instance sizes, Pan 2025 with Gurobi; <10 s at MA-SCVP's scale), so refit-and-replan every iteration costs nothing. The one measured technique for treating prior geometry as fallible evidence — Pan 2025's conservative inflation (union the prior with the current scan plus divergence points) — moves from edge case to core pattern.
- **Partial-shape fitting becomes the load-bearing wall.** The least-supported case in all of Branch 3 — a half-scanned cylinder with one-sided dropout — is now the *common* case, every round before the last. What exists: EMS holds 86.4% global-optimality at 20% surface coverage (superquadric, contiguous synthetic cap); Raffo's ~1/8-coverage works for a coarse initial estimate; region growing "does not perform well when data is missing" (Kaiser); nobody anywhere tests our exact case, and symmetry-based axis completion appears in zero of 20 documents. This must be measured on our clouds, round-by-round (E2).
- **Stopping is solved-enough.** Convergent evidence across four systems: 1–2 extra views capture essentially all achievable gain (GraspView: +35 / +15 / +0 for NBVs 1/2/3; TwNV: N=1 best, N=2 regresses). Plus MS@δ map stabilization as a mechanical fallback stop. The loop terminates on menu stability + question answered, with a hard cap. No learned stopping.

One new first-class risk the loop creates: **the menu changes every round by construction.** The old framing worried about a static menu shrinking (NBV-Net collapses 0.988 → 0.472 when its action space is restricted, because it emits one index and dies if infeasible). Our version is menu *churn* between rounds — and Branch 5/6's warning stands that a shape-shifting menu is exactly what a blind LLM handles badly. Two design responses, both ours to build because the literature has neither: (a) view identity by geometric content, stable across refits (§1.3); (b) menu diffs stated explicitly to the LLM ("two new views of the far side appeared after the last scan") rather than silently swapping the list.

### 1.3 Primitive fitting: available, fast, unvalidated for us — and identity is the new hard part

Verdict from Branch 3 stands: decomposition is off-the-shelf and runs in seconds at our cloud sizes (Efficient RANSAC 0.57 s @12k pts; EMS 91 ms; Open3D `detect_planar_patches` gives bounded face + centroid + normal today), but every robustness number in the literature is under isotropic Gaussian noise or hole-punching — none under registration jitter, directional dropout, or systematic per-view bias. Residual-family fit scores are structurally blind to systematic bias. No detector outputs uncertainty. So: build it, instrument it, trust nothing until measured on our clouds.

Loop-specific additions:

- **Identity persistence across refits is now load-bearing, not a benchmark curiosity.** Within one session the fit is recomputed every round; "the code face" must remain the same address across rounds or the agent's context is garbage. CGAL Efficient RANSAC is randomized with no seed parameter; every learned method uses unordered slots; **no re-scan repeatability experiment exists anywhere in the field.** Design: identity by geometric content (normal direction in the table+gravity frame, area, position — never extraction index), deterministic detectors preferred (CGAL region growing, RSPD), a matching layer we write (RIoU + Hungarian, SPFN's device used at inference), and our own repeatability protocol (E3) because none is published.
- **The known trap to guard on day one:** RSPD (inside `detect_planar_patches`) tiles large-radius cylinders with spurious planar patches — acknowledged, unfixed. A curvature-consistency guard is required, since half the design is cylinder-orbit.
- **The fallback gate cannot be a residual threshold alone.** SHREC's cone-degeneracy case: parameter error 1e11 while the residual looks fine. The gate is calibrated by the fit-score → derived-viewpoint-angular-error mapping (E4) — the number that exists nowhere else and that decides when the intelligent viewsphere abstains and the uniform tier serves instead.
- The ~4 cm fused-vs-single-view offset partially deflates under the loop — there is no monolithic survey cloud to be globally wrong; we refit on the accumulating in-session cloud. It remains an engineering thread (it cannot be sensor extrinsics — RGB and depth share the left imager — so it's our registration pipeline), and the cheap insurance stays: derive angles from locally re-fit normals near the region, not from the global cloud.

### 1.4 The intelligent viewsphere: the literature almost unanimously endorses the pattern, and identifies exactly the gap my A/B fills

The interface correction from my feedback — geometry computes the good views internally; the LLM gets a finished finite menu, never primitives or knobs — turns out to be the published consensus pattern, stated six times independently (Branch 6 §1.7): **geometry proposes, the model disposes, and the disposal is always a label.** Nobody asks a prompted model to construct or parameterize the candidate set. Supporting numbers:

- Selection over emission: +60.7 points (SoM, same model, same benchmark). The contract's "VLMs cannot emit metric poses" is now cited, not asserted.
- Small menus: K≈2 beats K=5 by 6 pp at matched descriptions (30 pp on hard cases). Deployed systems sit at 3–10 options. Per-primitive derivation lands at 8–20 views per object; two-stage decomposition keeps each *decision* small.
- **Two-stage selection (which surface → which view of it) — which I already endorsed — is the largest measured effect in the visual-prompting literature:** MOKA's hierarchy ablation, success 0.6–1.0 → 0.0–0.2 without the decomposition step. One spatial operation per turn (AP-VLM: adding a second rotation axis alone dropped SR 0.8→0.5 despite being a strict superset). Orientation baked into every candidate, never offered as an axis.
- Every relation precomputed and handed over as flat facts, never inferred: open models are at/below chance on camera–camera relations (MMSI); the best of 33 models scores 8.34% on building topology from text (SiT). The menu can *say* "opposite side of the current view"; it must never ask the model to derive it.
- Reachability resolved before the menu exists (GraspView's pattern; ObjView's 45.4/128 average reachable is the base rate that makes this mandatory) — with the flagged-not-silently-removed discipline for anything that drops between rounds.
- Free hygiene items with precedent: an always-listed stop action (MapGPT), an anti-repeat instruction (Think3D), an explicit fallback/abstain item — frontier models abstain correctly only 16.7–33.3% of the time unless the interface makes it a first-class option, and our near-primitive fallback depends on it.

And the gap: **nobody has ever ablated the label format for a discrete camera-pose menu.** Think3D deploys integer angles and never varies them; ViewSpatial uses compass words throughout and never varies them; TwNV's "discrete" condition is the least-documented thing in its own paper. My angle-label preference is exactly the untested cell. Two adjacent findings shape the A/B: any explicit scheme beats none (+5.7…+10.3, SCAFFOLD) while the choice *among* schemes was worth less (≤4.6) — so I should expect modest deltas and design for sensitivity (paired items, permuted order, multiple seeds); and RoboSpatial's frame-implicit collapse (~0% for 6 of 8 models) says that whatever the labels, **the reference frame is stated explicitly in every prompt** — "left" with respect to what is not a nicety, it's the difference between working and chance.

### 1.5 The benchmark: my 3-tier framework is buildable from precedented parts, and the metric warnings all survive

The component framework I sketched maps onto the field's converged practice almost 1:1:

- **Tier 0 (single image, one decision)** = the oracle-view cell of `view-level-evidence-benchmark` and e3vs's VQA-at-goal. E3vs's own numbers justify the tier split: hand the model the right view and it matches humans (3.58 vs 3.53) — the bottleneck is finding, not reading. Tier 0 isolates reading; Tiers 1–2 isolate finding.
- **Tier 1 (replay from dense capture)** is precedented by MA-SCVP's real-world protocol: exhaustively capture all candidate views once, then everything replays against the dataset — a "move" is a fetch. Travel costs computed in the kinematic sandbox from poses, as I said. The minimal per-view schema (RGB, depth, camera pose/intrinsics, nothing else) is consistent with what every replay precedent actually consumed; and every real application run can record benchmark-grade data unattended later.
- **Tier 2 (full quantitative runs)** — real robot, replay, or sim — carries the headline statistics.
- **Component swapping is the field's cleanest evaluation move** (vin-nbv's fitness-swap; gennbv's 2×2 cross-implementation): hold the loop fixed, swap one stage. That's exactly what "menu representation as a swappable serializer" and "orthogonality as one pluggable mode" require.

The warnings that survive untouched and constrain the harness: **the metric decides the answer** — the uniform sphere saturates coverage metrics at 98%+, so a coverage-primary benchmark loses to the baseline we're obsoleting for metric reasons; grounded scoring is mandatory — raw answer accuracy overstates grounded (view-correct × answer-correct) accuracy by up to 70 points, 26.2% of trials in the precedent were wrong-view/right-answer; **no published result shows a prompted VLM beating random view selection with variance** — so random-from-the-same-menu is the floor every table carries, and "geometry selects, LLM only reads and answers" is the null architecture the LLM-selects design must beat; menu order is a variable (oracle worst-to-best ordering swings up to 73 points), so we permute; a matched-budget no-new-view control (same calls, more deliberation) is the only way to attribute a gain to the view rather than the extra turn.

On statistics: the benchmark is auxiliary and I'm not running an academic ceremony. What I keep because it's cheap and buys real power at small N: paired comparisons over matched cases (same object, same placement, same QA item across conditions — pairing scales down fine), ~3 seeds on anything with a VLM in it, per-stratum reporting (pure primitive / compound / near-primitive), medians beside means, and the honest sizing note that at n≲100 episodes we resolve 15–20-point differences, not 5. What I drop: pre-registration theatre, α rituals, significance stars. The one discipline I keep from that world, stated in one line: **the losing condition is real** — if the geometry-conditioned menu does not beat the uniform sphere at matched budget on grounded QA, that ships as the result.

### 1.6 Things from the old draft that are now dead or demoted

- **Deep sensor-noise characterisation (σ(range,θ) sweeps, dropout physics):** demoted to a slim fitting-adequacy check. Out of scope this month per the feedback; the evidence view is judged on RGB.
- **Defect-visibility framing, raking light, specular-at-normal worry:** out of scope. Inspection is QA.
- **Survey-first framing everywhere:** replaced by the loop. Anything in the syntheses premised on "the object is fully scanned before planning" gets re-read as "per-round, on the partial cloud."
- **Primitive tuples / parameter knobs in the prompt:** dead. `(plane, face_3, {standoff, roll})` is the *internal* address; the LLM sees "View 4 — top face, straight down" or whatever the A/B says it should see.
- **Pre-registration and significance machinery:** dropped to the one-line losing condition + paired design.
- **The optimization track:** stays demoted (field constructs + selects, doesn't climb; MVTN's direct view-parameter optimization lost to a noise baseline). Constructive derivation ships; local refinement only if it ever provably buys more than the fit uncertainty.

---

## 2. Methodology for the build month

Build order: walking skeleton first, then harden the load-bearing wall (partial fitting + identity), then the menu, with the benchmark harness growing alongside as the arbiter. Every part names its solutions and the experiment that would prove or **disprove** each.

### Part A — `loop-skeleton` (the incremental scan–fit–replan loop, end to end, dumb first)

**Solution.** One pipeline in the kinematic sandbox: capture (or fetch, in replay) → accumulate cloud → fit primitives → derive candidate views → verify reachability → build menu → selector picks (scripted policy first, LLM later) → execute → refit → replan. Per-round exact set-cover solve over quality-gated candidates (~1 s, no greedy tax). Stop on menu stability + answered, hard cap; prior treated as fallible (Pan-style conservative union). No component is clever yet — the point is that every later part slots into a running loop.

**Experiments.** E5: loop convergence on box / cylinder / compound in the sandbox — rounds to menu stabilization, moves, travel distance; against a one-shot dense-survey control at matched budget. **Disproof condition:** if the incremental loop needs as many moves as a dense survey to produce a usable menu, the "no complete reconstruction up front" premise buys nothing and the architecture reverts to survey-then-inspect.

### Part B — `partial-fit` (primitive fitting on partial, growing clouds — the load-bearing wall)

**Solution.** Open3D `detect_planar_patches` (bounded faces: centroid, extent, normal) + CGAL region-growing / PCL cylinder + EMS as a cross-check; our own extent extraction for cylinders; curvature-consistency guard against RSPD's cylinder-tiling failure; refit-with-wider-band after detection (cheap, 5–25× parameter-error reduction precedent); normals re-fit locally near the inspected region rather than trusted from the global cloud.

**Experiments.** E1 (first, two hours): the whole stack on our *actual* clouds — single-view partials and accumulated multi-round clouds — before any more reading. E2: the number that exists nowhere in the literature — **angular error of the derived orthogonal viewpoint vs surface-coverage fraction** (sweep 10–100%, box faces and the half-cylinder-with-one-sided-dropout case), with distributions not means; plus the slim depth-adequacy check (fill rate on our materials at scanning standoffs). **Disproof condition:** if fitted-normal error on loop-realistic partial clouds exceeds the menu's angular resolution (i.e. the derived "orthogonal" view is no closer to orthogonal than a uniform-sphere neighbour), the geometry-conditioned menu cannot beat uniform *by construction* and the core claim dies there — cheapest possible place to find that out.

### Part C — `identity-persistence` (stable addresses across refits)

**Solution.** Address by geometric content (normal in table+gravity frame + area + position), never detector output order; deterministic detectors preferred; RIoU + Hungarian matching layer across rounds; internal addresses stay internal — the menu shows stable human-readable identities.

**Experiments.** E3: the re-scan/refit repeatability protocol no one has published — run the loop N times per object and across rounds within a session; report primitive-count stability, parameter dispersion, **identity flip rate** after matching, and menu churn per round. **Disproof condition:** if identity flips are common even with matching, the menu cannot promise "the same face keeps its name," and the interface degrades to per-round anonymous menus — a materially worse design the LLM experiments would then have to absorb.

### Part D — `menu-crafting` (the intelligent viewsphere)

**Solution.** Per-primitive constructive derivation, closed-form: plane face → centroid + normal at derived standoff; cylinder → orbit stations spaced by the incidence tolerance + cap views; standoff from the density↔standoff↔FOV relation bounded below by Min-Z@resolution. Every menu item pre-verified (IK + collision, exact check on survivors of a cheap filter) before the LLM ever sees it. Two-stage menu: which surface (plain words) → which view of it. Always-listed stop, explicit fallback/abstain item, anti-repeat, reference frame stated in every prompt, menu diffs announced between rounds ("flagged, never silently removed/changed"). Fallback tier: equal-area uniform sphere (Tammes/HEALPix — *not* the naive {h,v} grid) served when Part B's calibrated gate says the fit can't be trusted.

**Experiments.** E4: fit-score → derived-viewpoint-error calibration on corrupted clouds → gate threshold with measured false-accept/false-reject (**disproves** residual-only gating if the mapping is flat). E9: two-stage vs flat single-menu ablation on our task (does MOKA's hierarchy effect transfer). E10: abstention correctness on near-primitive objects — does the agent take the fallback item when no good view exists, scored, not assumed. **Disproof condition for two-stage:** if flat menus of ≤10 items match two-stage on grounded accuracy, the extra turn is dropped.

### Part E — `benchmark-harness` (the 3-tier component framework, auxiliary but the arbiter)

**Solution.** Dense-capture recorder writing the minimal schema (RGB, depth, pose, intrinsics); Tier 1 replay engine (move = fetch; travel costed in the sandbox); Tier 0 sampler drawing single images + QA items from the same captures (QA written by the LLM under my guidance); Tier 2 runner for real/replay/sim episodes. Everything a swappable component: view-space generator, selector, menu serializer, scoring mode. Scoring: grounded 2×2 grid (view-correct × answer-correct) + abstention cell as primary; moves, travel, duration always; per-region angular score as **one pluggable mode** (kept because the orthogonality claim needs a metric coverage can't express — a voxel seen at 85° counts as seen under every standard metric); worst-case-per-object reported beside means (one grazing evidence image is the exact failure Albert flagged; a mean hides it). Permanent baselines in every table: the current 12×30°×3 grid (the thing being obsoleted never leaves the table), budget-matched uniform, Tammes/HEALPix at matched budget (beating naive az/el is a weak claim), random-from-menu floor, oracle ceiling. Paired matched cases, ~3 seeds on VLM arms, per-stratum reporting, a design-untouched holdout of capture sets from day one.

**Experiments.** E11: harness self-checks — replay determinism, travel-cost sanity against real executions, Tier 0 item quality audit (manually verify every scored-wrong answer; the precedent found benchmark GT itself wrong in places).

### Part F — `menu-representation-ab` (the open choice my intuition doesn't get to settle)

**Solution candidates**, as swappable serializers over the identical underlying views: (1) angle-style labels (my tentative preference — az/el-flavoured, e.g. "45° right, 30° up from current"); (2) plain directional words ("top face, straight down"; "far side, camera-left"); (3) hybrid word + angle; (4) numbered-only with a one-line description. All variants state the reference frame explicitly (non-negotiable per RoboSpatial).

**Experiments.** E7 on Tier 1 replay: same episodes, same menus, serializer swapped; paired items, order permuted (canonical / reversed / shuffled), ≥3 seeds, one frontier + one deployable-open model (results change sign across that boundary). Also fills a genuine field gap: az/el textual comprehension — literally our current interface — has never been measured by anyone. Secondary probe: option similarity at fixed count (near-identical orbit stations vs well-separated views — the untested founding worry). **Disproof condition:** my angle-label preference loses to directional words → the benchmark decides and I ship the winner; expected deltas are small (SCAFFOLD ceiling ~5 pts), so if all variants tie within noise, the cheapest-to-generate serializer wins by default and the A/B closes.

### Part G — `headline-comparison` (the one claim, and the selector null)

**Solution.** The full 2×2 on Tier 1/2: {uniform viewsphere, intelligent viewsphere} × {scripted/oracle selector, LLM selector}, matched budgets, all four cells reported. The oracle row is what makes an LLM-row win interpretable (a small legible menu could win on legibility, not view quality). Mandatory arms: random-from-same-menu floor; blind no-geometry baseline; matched-budget deliberation control.

**Experiments.** E8 is the experiment. **Disproof conditions, both real:** (a) if the intelligent viewsphere doesn't beat uniform at matched budget on grounded QA under the oracle selector, the geometry conditioning itself failed — the month's claim is dead regardless of interface; (b) if the LLM selector doesn't beat random-from-menu, the LLM-selects design is dead and we ship geometry-selects / LLM-reads-and-answers (which no published result currently refutes as the better architecture).

### Part H — `demo-v2` (acceptance test)

**Solution.** The box with codes on one side. Between runs the box is moved and rotated. The loop must scan, find the code face, and deliver the orthogonal evidence image of it. Scored on the grounded grid (right face found × code read), plus moves and duration; failures narrated, not averaged away.

**Experiments.** E6 (feeds this and grounds the premise): Tier 0 code-reading accuracy vs incidence angle at matched distance on real RGB — the direct measurement that orthogonal evidence is better *for QA*. **Disproof condition:** if reading accuracy is flat out to 40–50° off-normal, the orthogonality requirement softens from "the orthogonal view" to "any view within the flat band" — which would relax menu density and is worth knowing early. E12: demo robustness runs across placements/rotations as the standing integration test.

**Sequencing.** E1 immediately (two hours, before anything else). A + B in week 1 — the skeleton and the wall. C + D in week 2, E growing in parallel from day one (the recorder especially — every run should produce benchmark-grade data). F + G in weeks 3–4, gated on B not killing the premise. H continuous from the first end-to-end run.

---

## 3. Open questions still mine (or Albert's), flagged not assumed

1. **Menu representation** — Part F decides; my angle-label preference is a hypothesis, on the record.
2. **Seams (edges/corners/junctions) in the menu** — no literature cover either way; default out of scope for the first slice.
3. **Coverage metric** — in as a secondary mode for NBV-literature comparability, never primary (the sphere saturates it).
4. **Blind orchestrator** — stays; the eyes tier reads images, the orchestrator navigates symbols. Revisit only if Part G's numbers demand it; a marked render of a defective cloud has never been shown to work and the fidelity-floor evidence says it can be net-negative.
5. **How much depth-quality work the fitting actually needs** — Part B's slim check answers it; I will not run a metrology campaign for a QA system.
6. **Camera upgrade** — decided by demo-v2's RGB evidence quality, not in advance.

## 4. What I refuse to build on

- Wang T-Mech 2024's numbers (read third-hand; structure trusted, tables not).
- Any incidence-knee value from the 45–75° band (all wrong sensor class — and mostly irrelevant now that evidence is RGB).
- SPAZER's +3.6%-over-random and TagaVLM's +0.94% topology claims (unverified; the two most direct threats to the blind design — verify before leaning either way).
- Human best-view labels as ground truth for our construct (mature machinery, wrong construct — recognition/aesthetics, not evidence; human-vs-robot preference correlation 0.13–0.21).
- Residual-based fit confidence as the fallback gate on its own (structurally blind to systematic bias).
- "Just run CGAL Efficient RANSAC" as the fitting plan (52.9% type accuracy, 7.5° axis error on *clean* CAD; non-reproducible run-to-run, no seed parameter).
- Any assumption that the LLM can infer view relations, frames, or topology — precompute everything, hand it over flat.
