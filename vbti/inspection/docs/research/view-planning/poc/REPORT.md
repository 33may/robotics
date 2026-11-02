# View-Planning POC — Decision Report

2026-09-07 · branch `data-engine/record-layer` · all commands from repo root, `p` = the robo-env python (`/home/anton/miniconda3/envs/robo/bin/python`)

## 1. Objective and what was built

**Hypothesis under test:** for a growing object point cloud from a recorded inspection run, compute the best set of inspection camera poses — "best" = orthogonal views, boresight opposing the local surface normal. This is the alternative to the parked primitive-decomposition approach in `inspection/fit/`.

**Pipeline** (offline replay, every step of 2408-cup3/cup4/cup5): accumulated step PLY → Poisson mesh (alpha-shape→hull fallback for sparse steps) → 3000 surface samples with cloud-derived normals → 300 normal-offset candidate poses at 0.10 m standoff → visibility matrix (0.07–0.20 m standoff window, 60° incidence gate, FOV check, self-occlusion raycast via Open3D RaycastingScene) → orthogonality-weighted greedy set cover, capped at 12 views → OpenCV-convention `T_base_cam` per view, base frame, metres. Guards: cull cameras below the table plane, cull cameras <30 mm from the cloud (D405 half-body). No IK, no reachability — poses only, by design.

Code (all new, peer of `inspection/fit/`):

- `inspection/viewplan/contract.py` — StepPlan dataclasses, lossless JSON round-trip
- `inspection/viewplan/mesh.py` — reconstruction ladder + deterministic table plane
- `inspection/viewplan/planner.py` — sampling, candidates, visibility, greedy, cross-eval scoring
- `inspection/viewplan/render.py` — offscreen raycast renders (no OpenGL)
- `inspection/viewplan/cli.py` — `run / show / variants / crosseval / budget / curves`
- `inspection/viewplan/workspace.py` — builds `derived/<run>/rrd/<run>-viewplan.rrd`
- `inspection/record/rr/log.py` — strictly additive viewplan block (+151 lines, 0 touched)
- `inspection/viewplan/tests/` — 98 tests, all green

Artifacts (deterministic, seed 0; full grid = 6 variants × 93 steps = 558 plans, ~70 s at `--jobs 8`, zero abstains): per `(run, variant)` under `inspection/data/derived/<run>/viewplan/<variant>/` — `step_NNN.json` (plan + poses + metrics), `meshes/`, `samples/` (npz), `renders/`, `scores.jsonl`; per run — `viewplan/curves.png` and `rrd/<run>-viewplan.rrd` (+ `rrd/continuity/` for the stability variant).

Variants: `base` (cosine incidence weight, 60° gate — the hypothesis), `binary` (unweighted set cover — the control), `cos2` (steeper weight), `cos75` (75° gate), `altmesh` (alpha-shape mesher — mesh-confound probe), `continuity` (base + warm-start from previous step's poses, ranking-only tie-break bonus — the stability fix).

## 2. How to review

Every command below was run and verified on 2026-09-07. Do **not** run `cli.py --help` — a `%` in one help string crashes argparse (known wart, listed in §5); the subcommands themselves are fine.

**Step 1 — the interactive 3D review (the main event):**

```
p inspection/viewplan/workspace.py --run 2408-cup3 --open
```

Builds and opens `inspection/data/derived/2408-cup3/rrd/2408-cup3-viewplan.rrd` in the native rerun viewer (the binary lives in the robo env; if `--open` reports no viewer, run `~/miniconda3/envs/robo/bin/rerun <rrd> <rbl>` yourself). In the viewer:

- **Scrub the `step` timeline** at the bottom (starts paused): drag the playhead or use arrow keys. Watch the cloud grow, the mesh follow, and the magenta planned frusta move step to step.
- **Entity tree (left, under `/world`):** `cloud` = accumulated points, `plan/mesh` = that step's reconstruction, `plan/coverage` = samples coloured red (uncovered) → green → yellow-white (covered, brighter = closer to orthogonal), `plan/cam/NN` = the planned views (magenta pinhole frusta), `table` = the plan's own deterministic table plane. Eye-icons toggle each.
- **Snap into a planned camera:** double-click a magenta frustum in the 3D view — the view takes that camera's PoV and tracks it. Double-click empty space to reset. To keep an external view too, clone the 3D view in the blueprint panel first.
- **Right panel:** milestone renders (steps 0,2,5,8,12,16,20,25,last for `base`; most steps intentionally log none).

Then the stability variant (goes to a separate directory so base's rrd survives):

```
p inspection/viewplan/workspace.py --run 2408-cup3 --variant continuity --out inspection/data/derived/2408-cup3/rrd/continuity --open
```

Scrub the last ~10 steps in both: `base` frusta jump around, `continuity` frusta hold still. Repeat for `2408-cup4` / `2408-cup5` as desired (same commands, run name swapped).

**Step 2 — the convergence curves (the core-question answer):**

```
xdg-open inspection/data/derived/2408-cup3/viewplan/curves.png
```

(also `2408-cup4`, `2408-cup5`.) Read the **top row**: coverage of the FINAL surface (rises), fraction of final-plan coverage with k* guide lines, incidence on the final surface, plan churn vs the 0.02 m target. The bottom-left self-scored curve **falls** — that is the growing-denominator artifact (mesh area panel next to it explains it), not plan regression. Regenerate with `p inspection/viewplan/cli.py curves --run 2408-cup3,2408-cup4,2408-cup5 --variant all`.

**Step 3 — printed evidence:**

```
p inspection/viewplan/cli.py show --run 2408-cup3 --variant base --steps final
p inspection/viewplan/cli.py crosseval --run 2408-cup3,2408-cup4,2408-cup5 --variant base,continuity
p inspection/viewplan/cli.py budget --run 2408-cup3,2408-cup4,2408-cup5 --variant base
```

`show` prints the 12 poses with per-view coverage/incidence and render paths. `crosseval` prints coverage-on-final-geometry, k*(90/95%) and churn per variant. `budget` prints how many views 90/95/99% of achievable coverage actually costs.

**Step 4 — optional, reproduce everything from scratch (~70 s + tests):**

```
p inspection/viewplan/cli.py run --run 2408-cup3,2408-cup4,2408-cup5 --steps all --variant all --seed 0 --jobs 8
p -m pytest inspection/viewplan/tests -q
```

Re-ran today: byte-stable plans (seed 0), same headline numbers, 98 tests pass in ~34 s.

## 3. Results

**Final-step plans, per run × variant** (n_views / coverage@60 raw / of-achievable / mean / p95 best-incidence). Every cell terminates on the 12-view cap, never on gain convergence:

| run | base | binary | cos2 | cos75 | altmesh † | continuity |
|---|---|---|---|---|---|---|
| cup3 (s29) | 12 / .761 / .871 / 27.9° / 53.2° | 12 / .774 / .886 / 29.7° / 54.4° | 12 / .760 / .869 / 27.7° / 53.5° | 12 / .766 / .893 / 30.3° / 61.1° | 12 / .634 / .967 / 25.7° / 51.9° | 12 / .748 / .855 / 27.6° / 52.9° |
| cup4 (s32) | 12 / .757 / .891 / 26.5° / 53.6° | 12 / .762 / .896 / 30.3° / 56.0° | 12 / .755 / .888 / 25.4° / 51.2° | 12 / .755 / .911 / 29.1° / 62.4° | 12 / .669 / .961 / 24.5° / 49.4° | 12 / .756 / .887 / 27.5° / 52.3° |
| cup5 (s30) | 12 / .631 / .827 / 28.2° / 52.7° | 12 / .629 / .824 / 29.0° / 52.6° | 12 / .633 / .829 / 27.5° / 53.0° | 12 / .629 / .824 / 30.5° / 62.8° | 12 / .872 / .985 / 25.8° / 47.7° | 12 / .617 / .807 / 27.4° / 52.6° |

† altmesh numbers are self-scored on its **own** alpha mesh — always read them with the mesh-area/cloud-area ratio: 1.25 / 1.21 / **0.70**. The cup5 "0.872" is a denominator artifact (the alpha mesh caps the cavity, deleting ~30 % of the surface); re-scored on the common final surface it is 0.627 vs base's 0.631 — indistinguishable. See §5.

**Convergence and stability** (base vs continuity; k* = first step sustaining ≥95 % / ≥90 % of the final plan's coverage on the FINAL geometry; churn = mean consecutive pose-set delta over the last 10 steps, target ≤0.02 m):

| run | k*95 base | k*90 base | churn base | churn continuity | k*(churn≤0.02) cont. | continuity coverage cost |
|---|---|---|---|---|---|---|
| cup3 (29 steps) | 22 | 20 | 0.071 m (never ≤0.02) | **0.013 m** | 24 | −1.3 pp |
| cup4 (32 steps) | 25 | 12 | 0.066 m (never) | **0.007 m** | 13 | −0.1 pp |
| cup5 (30 steps) | 21 | 13 | 0.074 m (never) | **0.015 m** | None (9/10 steps under; one at 0.023) | −1.4 pp |

**View budget on the final geometry** (uncapped greedy, `budget` subcommand): 90 % of achievable = 17 / 14 / 25 views; 95 % = **28 / 23 / 46**; at the 12-cap you get 0.871 / 0.891 / 0.827 of achievable. 13–24 % of each Poisson mesh is unachievable from **any** legal D405 pose (occlusion-dominated: cavity interior).

**Success bar as pre-registered:** B3 (every render frames what it claims) **passes** — min framed_frac 0.995 over all 54 rendered step-variant plans. B2 (≥95 % area at ≤60° with ≤12 views) **fails on all runs** — jointly infeasible, see budget. B1 (stable plan by mid-run) **fails for base** (churn never ≤0.02), and with `continuity` passes on cup4 (step 13), late on cup3 (24), near-miss on cup5.

**Variant recommendation, with evidence:** **`base` for pose quality, `continuity` for the loop.** Paired per-step differences over the last 10 steps of each run (n=30): binary vs base = +0.004±0.008 coverage but **+2.27°±0.97 worse mean incidence** (t=12.8) — orthogonality weighting delivers measurably more orthogonal coverage for equal area, which is the POC's hypothesis, confirmed. `cos2` tracks base within noise on every metric (pre-registered kill rule fires — drop it). `cos75` buys +2–4 pp of-achievable at +8–10° worse p95 incidence — a gate trade, not a win. `continuity` costs ≤1.4 pp coverage and cuts churn ~5× — take it wherever plans feed a running system. `altmesh` is a confound probe, not a candidate (§5).

**Positive results verified independently:** 12 final poses well-distributed (azimuth gaps ≤63°, elevation −15..+80°), all at 0.100 m standoff, all above the table, min camera-to-cloud clearance 30.5 mm across all 558 plans; the cup **handle** is covered 89/96/95 % (better than the body — the body deficit is the cavity, not clustering); planned `T_base_cam` verified as genuine OpenCV +Z-boresight base-frame poses (rebuilding all 30 recorded cup3 poses matches at median 0.00°).

## 4. After which step is the geometry enough?

The self-scored coverage curve cannot answer this — it *falls* (0.99 → 0.63–0.76) because the mesh area grows 2.8–3.4× and the denominator grows with it. The honest metric is cross-evaluation: score every step-k plan against the run's **final** surface (`xe_*` columns in `scores.jsonl`, top row of `curves.png`). That curve rises monotonically, and:

- **≥95 % of final-plan performance:** step **22 / 25 / 21** of 29 / 32 / 30 — roughly **two-thirds to three-quarters through each run**.
- **≥90 %:** step 20 / 12 / 13 — cup4 and cup5 reach "good enough for a first pass" near **mid-run**; cup3 (handle seen late) does not.
- Mid-run plans sit at ~70–85 % of final-plan coverage — usable, not final.

Sketch answer: *if 90 % of the eventual plan quality is acceptable, geometry is enough around step 12–20; for 95 % you wait until ~step 21–25.* The spread is run-dependent (when the handle/cavity enters the cloud), so any live gate should be a threshold on the cross-eval-style metric, not a fixed step index.

## 5. Limitations and open questions for Anton

Honest limits of what was built:

1. **No reachability/IK** (by design, settled requirement): poses may be unreachable for the arm; the only guards are the table cull and 30 mm object clearance. Camera roll is pinned deterministically by `look_at_cv` — if the real robot constrains roll differently, poses need a roll pass.
2. **The bar was missed, and is partly unwinnable:** 95 % @ 60° at 12 views needs 28/23/46 views on the real geometry, and 13–24 % of the mesh is unreachable by any legal D405 pose (cavity). Decision needed: relax to ~87–89 % of achievable at 12 views (what we get), raise the budget to ~25, or accept the cavity as uninspectable with this sensor.
3. **Mesher is the biggest unresolved lever (Q3):** achievable-area fraction swings 0.66–0.88 across meshers — an order of magnitude more than any weighting variant moves anything. Neither Poisson (balloons, needs 6 mm trim) nor alpha (caps the cavity on cup5, area/cloud 0.70) is ground truth; cup5's Poisson mesh is the worst of the three (non-monotonic area, 24 % unachievable). Every altmesh coverage number in §3 must be read with its area ratio.
4. **No external ground truth:** all evaluation is self-referential to reconstructed meshes; cross-eval fixes the *direction* of the curve but the yardstick is still the final Poisson mesh, itself carrying 3–6 mm registration noise.
5. **Stability is fixed by construction, not convergence:** `continuity`'s calm comes from warm-start hysteresis; underlying near-ties among candidates remain (cup5 still grazes the threshold). If plan identity matters downstream, this is a design constraint to carry, not a solved problem.
6. Smaller warts: `cli.py --help` / `budget --help` crash on a `%` in a help string (subcommands unaffected); the viewplan rrd filename carries the run only, so a second `--variant` overwrites unless `--out` is used (documented in the module); `scores.jsonl` omits which mesher actually answered (altmesh fell back to Poisson on 5/95 steps — visible only in the step JSONs); coverage panels have no in-image legend (key: **red = uncovered, green→yellow-white = covered, brighter = more orthogonal**); step-0 renders show a ~1k-point blob, correctly framed but not a cup; the main `inspection/tests` suite has 14 pre-existing websocket/brain-bus failures (documented before viewplan existed, zero viewplan involvement — viewplan's own 98 tests are green).

Open questions (your call): (a) which bar replaces B2 — %-of-achievable at 12, or a bigger budget? (b) adopt `continuity` as the default for the loop? (c) is the mesher worth investment before wiring this into the live system, or is Poisson-with-trim good enough given cross-eval scoring? (d) does the cavity need a different sensor/standoff, or is "uninspectable" acceptable there?

## 6. Gallery — what to eyeball, in order

All paths relative to repo root; each verified today.

1. `inspection/data/derived/2408-cup3/viewplan/base/renders/step_029_view_00.png` — the plan's best-ranked final view: shaded mug in profile, handle right, orange dots = the 479 samples this view claims at ≤60°; all on visible surface.
2. `inspection/data/derived/2408-cup3/viewplan/base/renders/step_029_coverage.png` — final coverage panel (2×2 orbit): mostly green/yellow; the red flecks on the rim and interior are the honest holes.
3. `inspection/data/derived/2408-cup4/viewplan/base/renders/step_032_coverage.png` — the clearest picture of the *unachievable* region: the solid red patch in the top-right quadrant is the cavity interior no legal D405 pose can cover.
4. `inspection/data/derived/2408-cup5/viewplan/base/renders/step_030_coverage.png` — the worst run (coverage .631): red concentrated inside and low on the body; this is what 0.827-of-achievable looks like.
5. `inspection/data/derived/2408-cup5/viewplan/altmesh/renders/step_030_coverage.png` — the mesh confound in one image: almost everything yellow ("0.872 coverage") because the alpha mesh capped the cavity — compare with #4 and distrust any coverage number without its area ratio.
6. `inspection/data/derived/2408-cup3/viewplan/base/renders/step_012_view_00.png` — mid-run: a recognizable partial cup, correctly framed — plans are already sane here even though only ~80 % of final quality.
7. `inspection/data/derived/2408-cup3/viewplan/base/renders/step_000_view_00.png` — step 0: a lumpy blob. Correctly framed, but it is why every early "99 % coverage" self-score is meaningless.
8. `inspection/data/derived/2408-cup3/viewplan/curves.png` — the decision figure: top row rises to k*=22 (base) / 20 (continuity); bottom-left falls (denominator artifact, explained by the area panel beside it); churn panel shows continuity (red) alone under the 0.02 line.
9. `inspection/data/derived/2408-cup4/viewplan/curves.png` and `inspection/data/derived/2408-cup5/viewplan/curves.png` — same read; cup4 is the friendliest run (k*90 at step 12), cup5 the hardest.
