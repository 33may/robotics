# Completion Loop — Operating Guide

2026-09-10 · branch `data-engine/record-layer` · repo root `/home/anton/projects/robotics` · `p` = `/home/anton/miniconda3/envs/robo/bin/python`, always run from repo root. **Do not commit anything.** Anton is away ~2 h; every decision the loop needs is pre-made below. When this guide and improvisation disagree, the guide wins.

## 1. Objective and success bar

Build `complete(P: np.ndarray) -> o3d.geometry.TriangleMesh` in `inspection/viewplan/complete.py`: from one partial per-step cloud, reconstruct the box **including unseen walls**, for all steps of all 9 runs (`0809-box2, box3, box4-3, box5, box6, box7, box8, box9, box10`; 19–28 steps each, ~227 (run, step) pairs; clouds at `inspection/data/derived/<run>/steps/step_NNN.ply`, accumulated). Anton's requirement, literal: **star-visibility** — from the cloud centroid every outward ray hits surface. The bench proxy for that is `face_coverage`; the headline metric is **worst-face coverage** (min over the 5 GT faces), which punishes exactly today's failure (holey box sides that break planning, see `regions.py` docstring 2026-09-10 review).

**Success bar (PROVISIONAL until the iteration-1 baseline row exists, then FROZEN — freeze rule below):**

- min-over-runs worst-face coverage ≥ **0.85** at each run's **final** step, and ≥ **0.60** at each run's **mid** step (step `floor(0.5*(n_steps-1))`);
- mean `hallucination_frac` over final steps ≤ **0.15**;
- mean `surface_err_mm` over final steps ≤ **12 mm**.

Rationale: tau = 10 mm sits just above the 3–6 mm registration noise plus a few mm of hand-annotation error, so seen surface should score near 1.0 and the bar really measures fill quality of unseen walls. 0.85 tolerates one rounded corner strip per face; 0.60 mid-run concedes that a wall nobody has looked at yet is a guess. 15 % hallucination allows the fill to overshoot corners without allowing a balloon. **Freeze rule:** after the iteration-1 (baseline) row is logged, adjust once and only once: if baseline already beats a number, raise that number to baseline + 10 pp (coverage) / baseline − 5 pp (hallucination); otherwise keep as written. Then freeze — no further movement, ever, regardless of how the ladder goes. If no candidate meets the frozen bar, the deliverable is the best-by-headline candidate plus an honest gap statement, not a moved bar.

**Stopping:** bar met + leave-one-run-out check (§4) passed → stop candidate work, spend remaining time on the report (§6); attempt the hybrid (f) only if > 30 min remain after that. Bar never met → work the ladder until T−15 min, then report.

## 2. Candidate ladder (ordered, cheapest first — this order is the backlog)

All candidates: open3d + numpy only, deterministic (`n_threads=1`, seeded RNG where sampling exists), signature `def c_<name>(P, cams=None, seed=0) -> (mesh, info)`. `cams = planner.recorded_cams(run)` is legal input (recorded data, not GT). `mesh.table_plane(P)` gives the table (cloud-derived, legal). **Nothing may touch `truth/`** (§4).

**(1) `poisson20` — Poisson trim-sweep baseline.** `meshmod.reconstruct(P, cams=cams, trim_m=0.02)` — exactly the `membrane` lane of `regions.refine_step` (20 mm measured as the sweet spot on 0809-box6 steps 12/26, `/tmp/membrane_check` sweep, 2026-09-10; 6 mm default kills the membrane, ~50 mm admits table skirts). One line; it exists to calibrate the bar and the leaderboard. Expected failure: truly-unseen walls stay open by construction — worst-face coverage will be low mid-run and possibly at final steps; that number is the problem statement.

**(2) `hull` — convex hull floor/control.** `meshmod.reconstruct(P, ...)` ladder ends at `_hull`; call `pc.compute_convex_hull()` directly on the cloud (no normals needed). Closed by construction → worst-face coverage should be high; hallucination and surface_err expose the cost (bulges over concavities, bridges cloud outliers, top-edge overhangs become slanted planes). Any candidate that cannot beat hull on surface_err while matching its coverage is not pulling its weight. Trivial to write; its whole iteration should take < 10 min.

**(3) `star` — radial completion on an icosphere (HIGH priority — Anton's requirement made literal).** Center `c = P.mean(0)`. Directions: `o3d.geometry.TriangleMesh.create_icosahedron(1.0)` + `subdivide_midpoint(number_of_iterations=4)` (2562 verts), renormalize vertices to unit — do NOT use `create_sphere` (UV pole clustering) or `subdivide_loop` (shrinks). For each cloud point compute `(dir, r)` about `c`; bin points to nearest sphere vertex via `scipy.spatial.cKDTree` on the unit dirs. Vertex radius = **median** of member radii; vertex is `measured` if ≥ 3 members, else `interpolated`. Fill interpolated radii by Laplacian relaxation over the sphere mesh's vertex adjacency (build adjacency from triangles; ~300 Jacobi iterations averaging neighbors, measured vertices clamped). Final mesh = sphere triangles with vertices at `c + r_i * dir_i`; store the measured/interpolated flags in `info` (and as vertex colors white/red for renders). Clamp: vertices below the table plane get z = table z. Expected failures: (i) centroid is pulled up by the dense top face → grazing side directions bin badly; permitted fix (same iteration, §3): center = `min_yaw_aabb` box center at mid-height instead of `P.mean(0)`; (ii) icosphere resolution rounds the 4 box edges → hallucination strip per edge; (iii) a holey-but-seen side with < 3 pts/bin reads as unseen → interpolation smooths a wall we actually measured; (iv) the box IS star-shaped from any interior point, so the representation itself cannot fail — only the center and binning can.

**(4) `boxsnap` — min-yaw-cuboid prior snap.** `from inspection.cell.geometry import min_yaw_aabb`; `center, dims, yaw = min_yaw_aabb(P, pct=1.0)` (CLOUD only — never the GT json; the 1 % trim is the same stray-point guard the recorder uses). Bottom face: z from `mesh.table_plane(P)` (the cloud's low tail sits 8 mm above the real table — `TABLE_MARGIN_M` un-does that; height = z_max(P) − z_table). Build `create_box(l, w, h)`, translate so it's centered at (center_xy, table + h/2), rotate by yaw about +z. Expected failures: (i) mid-run a wholly-unseen far wall → cloud extent under-reads that dim by up to half the box (the box ends where the points end); (ii) pct trim under-reads by a few mm systematically; (iii) any non-cuboid detail (tape bulge, open flap) is flattened — hallucination and surface_err will say by how much. Cheap (~20 lines) and probably strong at final steps on literal boxes.

**(5) `planes` — RANSAC plane extension.** Iterative `pcd.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=1000)` (seed via `o3d.utility.random.seed` if available, else accept and note nondeterminism in `info`), remove inliers, repeat ≤ 5 planes, keep planes with ≥ 300 inliers. Classify: top (|n·z| > 0.9), side (|n·z| < 0.3, snap normal to horizontal); table from `table_plane(P)`. If < 4 sides found, synthesize the missing ones from `min_yaw_aabb` yaw + the opposing found side (mirror through the cloud extent). Faces = mutual intersections: each side clipped by table, top, and its two neighbor sides; triangulate each polygon (fan). Expected failures: (i) holey sides fail the inlier count → most of the run degenerates to the boxsnap prior anyway; (ii) plane intersection bookkeeping is fiddly — wrong winding/ordering makes non-faces; (iii) slightly non-parallel opposing planes make skewed corners. **Timebox: if implementation exceeds 25 min, kill it, log "killed: timebox", move on** — boxsnap likely dominates it on this data.

**(6) `hybrid` — poisson where seen + star/prior fill where not.** Take the `poisson20` mesh; from the star center cast the 2562 icosphere rays into it (`render.scene_of(m)`, `scene.cast_rays`); directions with no hit = blind set; take the `star` (or `boxsnap`, whichever leads the board) surface restricted to blind-set triangles (a triangle is blind if all 3 vertices are blind or interpolated); output = concatenation of both triangle sets (one `TriangleMesh`, non-manifold seams are fine — the bench and raycasts don't care about manifoldness, cf. `mesh.py` "not watertight and that is correct"). Expected failures: double-wall seams where the two surfaces overlap (counts as hallucination), gaps at the seam (counts against coverage). Design it from the leaderboard, not in advance — that's why it's last.

## 3. The loop protocol (one Ralph iteration = one candidate)

State lives in files, not in memory: every iteration starts by reading the tail of `LOG.md` and the `LEADERBOARD.md` to find where it is. Then:

**Iteration 0 — preflight (once, ~15 min budget):**
1. Check `inspection/data/derived/<run>/truth/box.json` for all 9 runs (fields `{run, center_xy, z_table, yaw_deg, dims_lwh}`, metres, l along yaw, bottom on table, yaw about +z). Anton is annotating now — **if some are missing, bench the runs that have truth, record `runs=N/9` in every leaderboard row, and re-check for new truth files at the start of every iteration. Never block, never fabricate a truth box.** If ZERO truth files exist, iterations still run: implement candidates, render sheets, log — leaderboard metric columns say `no-truth`; backfill by re-running bench for every logged algo the moment truth appears.
2. Check `inspection/viewplan/truthbox.py` provides `bench` (it did **not exist** as of 2026-09-10 14:30 — Anton is building it; take whatever signature exists and adapt). If absent or incomplete, build it to this spec, per (run, step, algo): `face_coverage` — sample each GT face (4 sides + top, bottom excluded) on a ~5 mm grid, fraction within tau = 10 mm of recon surface (`o3d.t.geometry.RaycastingScene.compute_distance`); `surface_err_mm` — symmetric mean of (GT face samples → recon distance) and (recon surface samples → analytic point-to-cuboid: transform to box frame by −yaw/−center, `q = |p| − half_dims`, unsigned surface distance = `‖max(q,0)‖ + min(max(q), 0)` in abs); `hallucination_frac` — fraction of recon surface samples (area-uniform, `sample_points_uniformly(5000)`, seeded) with point-to-cuboid distance > tau; aggregation — per-run mean over steps, headline = min over faces then min over runs. `truthbox.py` is EVAL-ONLY code and the only file allowed to read `truth/`.
3. Smoke-test bench + timing on `poisson20`, 0809-box6 only. Extrapolate: if full 9-run serial bench > 8 min, **freeze the step set to {every 2nd step + final} for ALL iterations** (comparability beats coverage) and say so in the leaderboard header. Use `--jobs 8`-style multiprocessing only if trivially available; don't build it.

**Iterations 1..N — per candidate, in ladder order (~15–20 min each):**
1. **Implement** the one candidate in `complete.py` (register in `ALGOS`). No other candidate touched.
2. **Bench** it: one invocation, all truth-available runs × the frozen step set. **One candidate per iteration, no combinatorial sweeps, no parameter grids.** Within the iteration, at most **2 fix-and-rerun cycles** and only when renders show a clear defect (e.g. the star-center fix in §2(3)); a third problem parks the candidate with verdict `parked`.
3. **Leaderboard row** (append-only, never edit old rows — §5 columns).
4. **Evidence sheet** — mandatory, the `/tmp/membrane_check` 2×4 pattern: fixed side viewpoints **h0/h3/h6/h9 at ring v=0** via `regions.shell_of(run)` + `regions.orig_pose(shell, h, 0)`, rendered with `render.scene_of(mesh)` + `render.render_view(scene, T, planner.session_intrinsics(run), scale=0.5, window=(0.07, shell["r"]+0.05))`, labels burned in with `cv2.putText` (algo, step, h, that step's worst-face number), rows/cols concatenated with `np.concatenate` exactly like `regions._ring_sheet`. Two sheets: **0809-box6 steps 12+26** (rows) × h0/h3/h6/h9 (cols), and the same layout for the **current worst run** (lowest per-run worst-face on this iteration's bench) at its mid + final steps. Save under `/tmp/completion_bench/<algo>/`.
5. **Verdict** by pre-made rules: `keep` (new leaderboard leader on headline), `kill` (headline > 2 pp below the leader AND no other metric better by ≥ 20 %), `iterate` (renders show a specific fixable defect — allowed twice, see above), `parked`, `killed: timebox`. Log to `LOG.md`, move to the next ladder entry.

**Budget:** ~2 h total. Iteration 0 ≈ 15 min (more if bench must be built — cap 35 min), then candidates in order; **reserve the final 15 min for §6 regardless of ladder position** — at T−15 min stop implementing, even mid-candidate (log `parked: time`). Realistic yield: baseline + hull + star + boxsnap, planes/hybrid only if time allows. That is fine — a clean board of 4 beats a messy board of 6.

## 4. Anti-self-deception rules (non-negotiable)

- **GT is EVAL-ONLY.** No algorithm code reads `truth/box.json` — `boxsnap` uses `min_yaw_aabb` on the CLOUD, never GT dims. Enforced check each iteration: `grep -n "truth" inspection/viewplan/complete.py` must return nothing, and `complete.py` must not import `truthbox`.
- **Leave-one-run-out spot check before declaring a winner:** every constant that was chosen or changed because of bench feedback (star center rule, min-members = 3, inlier threshold, …) gets re-derived with one run held out (use 0809-box9, the shortest); if the held-out run's worst-face drops > 5 pp vs the all-runs figure, the winner is flagged `overfit-suspect` in the report — still reported, honestly labeled.
- **Every leaderboard row records the algo version hash:** `git hash-object inspection/viewplan/complete.py` (first 8 chars) at bench time — uncommitted edits are captured, which is the point.
- **Renders are mandatory every iteration.** Numbers can lie about geometry (REPORT.md §5: altmesh "0.872" was a denominator artifact caught only by looking). No row without its two sheets.
- **If bench and renders disagree, the renders are right.** Fix the bench, then re-run bench for EVERY previously logged algo and append corrected rows (marked `rebench:<reason>`); never edit or delete old rows.
- Determinism: re-running the winner's bench in a fresh process must reproduce its row exactly; if not, find out why before the report.

## 5. Artifact layout

- **Code:** `inspection/viewplan/complete.py` — module docstring stating purpose + honesty caveats (house style, see `mesh.py`); `ALGOS: dict[str, callable]` registry; one function per candidate `c_<name>(P, cams=None, seed=0) -> (mesh, info)`; top-level `complete(P, algo="<current leader>", cams=None, seed=0)`; CLI per repo convention — bottom of file exactly:
  `if __name__ == "__main__": import fire; fire.core.Display = lambda lines, out: print(*lines, file=out); fire.Fire({...})`
  (see `regions.py:428–431`). Subcommands: `run` (one run/step/algo → mesh + printed info), `sheet` (evidence sheets for an algo). Bench CLI lives in `truthbox.py` (or wherever Anton put it) and imports `complete.ALGOS`. Do not modify `mesh.py`, `regions.py`, `planner.py`, `render.py`.
- **Leaderboard:** `docs/research/view-planning/completion/LEADERBOARD.md` — append-only table, one row per iteration: `iter | time | algo | hash | runs | steps | headline (min-run worst-face, final) | mean worst-face final | mean worst-face mid | surface_err_mm | halluc_frac | s/step | verdict`.
- **Renders:** `/tmp/completion_bench/<algo>/box6_s12_s26.png`, `/tmp/completion_bench/<algo>/<worstrun>_mid_final.png`.
- **Log:** `docs/research/view-planning/completion/LOG.md` — per iteration, append: what was tried (2–3 sentences), the numbers, what the renders showed, verdict, next action. This file is the loop's memory; write it like you'll be rebooted before reading it — because you will be.

## 6. Report for Anton's return

Write `docs/research/view-planning/completion/REPORT.md` in the REPORT.md §5 register — matter-of-fact, every claim with its number, artifacts listed with paths, failures stated plainly (the POC report called its own bar "partly unwinnable"; that is the standard). Contents:

1. Final leaderboard verbatim + which rows are `rebench` rows and why.
2. Winner's evidence sheets: all 9 runs at final step (render the 7 not already produced, same 2×4 pattern, into `/tmp/completion_bench/<winner>/final_<run>.png`) + the worst 3 mid-run (run, step) cases by worst-face.
3. Honest limitations list: runs missing truth, steps skipped by the frozen step set, constants tuned on all runs (LOO result), known geometric sins of the winner visible in renders but under tau, bench quirks discovered, anything `parked`/`killed: timebox`.
4. One paragraph, ≤ 6 sentences: recommended algorithm, its frozen-bar verdict (met / not met, by how much), the single biggest remaining risk, and the one next step you'd take.

No marketing. If the bar was not met, the first sentence of the paragraph says so.

## 7. Amendment (Anton, 2026-09-10 ~15:10) — freedom clause

The candidate ladder (§2) is a starting backlog, NOT a cage. The loop MAY:
- research approaches beyond the ladder and implement novel candidates;
- change any code under inspection/viewplan/ and build new tools/files
  (viewers, probes, analysis scripts) when the work needs them;
- restructure complete.py if the registry shape fights a better design.

What stays NON-NEGOTIABLE (unchanged): GT is eval-only (§4); the frozen
step set + append-only leaderboard (comparability); determinism of any
candidate that enters the board; renders every iteration; the success
bar freeze rule; do not commit.
