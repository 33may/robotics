# View-planning POC — technical research (verified by experiment)

Every claim below was measured on 2026-09-07 by running real code in the
`robo` env (`/home/anton/miniconda3/envs/robo/bin/python`, open3d 0.19.0,
numpy 2.3.5, py3.11) against the actual step PLYs of 2408-cup3/cup4/cup5.
Nothing here is from memory. Scratch scripts lived in `/tmp/vp_poc/e1..e9`.

## 0. Data facts that override the briefing

- **cup5 steps are `000–028` + `030`.** Step `029` is absent in BOTH
  `data/runs/2408-cup5/` and `derived/2408-cup5/steps/` (`unfused_dirs` in
  its summary.json is `[]` — it was simply never written). cup3 ends at 029,
  cup4 at 032. **Enumerate steps by glob, never by `range()`.**
- `summary.json` top-level keys: `run, crop, pinned, seed, steps,
  verification, fallbacks, unfused_dirs, renders`. cup5 verification:
  `n_stored == n_replayed == 6904, max_abs_coord_diff_m 0.0, "exact"`.
  No table-plane field anywhere — see §D.
- Measured cloud stats: cup3 step 005 = 2942 pts, step 029 = 6777 pts,
  median NN spacing 2.40 / 2.49 mm. First steps: cup3 1152, cup4 930,
  cup5 1032 pts.
- Recorded capture standoffs (cam → final-cloud centroid, cup3): min 0.235,
  median 0.246, max 0.440 m — the runs were recorded OUTSIDE the D405 sweet
  spot. Planned views at 0.10 m go closer than any recorded view ever did.
  Also note `cell/geometry.py MIN_RANGE_M = 0.13` (gripper-finger cut):
  if these plans are ever executed, depth inside 0.13 m would be discarded
  by the current fusion pipeline. Out of scope for the POC (poses only),
  but it must be said.

## A. Meshing recipe (verified)

### A1. Normals: estimate, then orient toward the recorded cameras

Step PLYs have no normals. Estimation params match `fit/common.py`
(radius 12 mm, knn 30). Three orientation strategies measured on cup3,
scored as agreement with "normal points toward the nearest recorded camera
position" (camera positions = `T_base_cam[:3,3]` from
`runs/<run>/NNN/meta.json`, steps 0..k):

| strategy | step 005 | step 029 |
|---|---|---|
| A: outward from centroid (`fit/common.py` rule) | 0.738 | 0.891 |
| B: flip toward nearest recorded camera | 1.000 (by construction) | 1.000 |
| C: `orient_normals_consistent_tangent_plane(30)` + global vote | 0.769 | 0.686 |
| B2: flip toward best-aligned camera | disagrees with B 8.0% | **54.8%** — degenerates |

- A fails in the cup cavity/handle (26% wrong at step 5).
- C is the worst and graph-dependent — rejected (also rejected by
  `fit/common.py` for drift between steps).
- B2 (flip so `max_c dot(n, unit(cam_c - p))` wins) degenerates once
  cameras surround the object: with 30 cameras there is a camera on both
  sides of nearly every tangent plane, so the flip decision is noise.
- **Recipe: strategy B — orient each normal toward the nearest recorded
  camera position (cameras of steps 0..k only).** Deterministic, uses only
  data available at that step.

```python
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=30))
N = np.asarray(pc.normals).copy()
near = np.argmin(np.linalg.norm(P[:, None, :] - cams[None, :, :], axis=2), axis=1)
N[np.einsum("ij,ij->i", N, cams[near] - P) < 0] *= -1
pc.normals = o3d.utility.Vector3dVector(N)
```

### A2. Poisson depth + trim

Depth sweep on cup3 (times on this machine):

- depth 8 and depth 9 give byte-identical meshes (octree saturated at
  2.5 mm sample spacing) — **depth=8**. Runtime 2.2–5.5 s per step.
- Raw Poisson output at step 005: v=9298 t=18404, extent
  [95, 132, 132] mm vs true cloud extent [93, 120, 90] — it balloons a
  closed skin over the unseen back side.
- **Density-quantile trim does NOT fix the balloon**: q0.05 still leaves
  extent [93, 132, 114].
- **Distance-to-input-cloud trim does**: drop mesh vertices whose NN
  distance to the input cloud > 6 mm (≈ 2.4x spacing, ≈ registration
  noise). Step 005 → v=7868 t=15205, extent [90, 129, 98]; step 029 →
  v=17405 t=34503, extent [92, 112, 92] (true [98, 115, 91]). 4 mm was
  too aggressive (halved area at step 5), 10 mm re-admits balloon.

```python
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8)
d, _ = cKDTree(P).query(np.asarray(mesh.vertices))
mesh.remove_vertices_by_mask(d > 0.006)
mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
```

- Watertight: `is_watertight()` is False even for the raw Poisson output
  here (and trimming opens the unseen side on purpose). Irrelevant for
  `RaycastingScene`. "Watertight-ish" is not achieved and not needed.
- Outward orientation: signed volume of the step-29 mesh is +459 cm³ raw /
  +391 cm³ trimmed (positive = outward winding, consistent with the input
  normal orientation). Sign is meaningless on the open early-step sheets
  (cup4 step 0 gives −115 cm³) — so **never trust mesh winding for
  normals; re-orient every mesh-sampled normal from the nearest oriented
  cloud point** (§C1). That one rule makes Poisson and any fallback mesh
  interchangeable.

### A3. Sparse-step fallback

Measured on the first steps (930–1152 pts): Poisson d8 + 6 mm trim
**still works** (cup3: v=3657 t=6995 ext [95, 127, 81]; cup4: v=2182;
cup5: v=2528) — no fallback was actually needed on any step of these runs.
Fallback if Poisson ever throws or yields < ~500 triangles:
`create_from_point_cloud_alpha_shape(pc, alpha=0.012)` (cup3 step 0:
v=899 t=1920, extent matches cloud; winding arbitrary — normals come from
the cloud anyway). Ball pivoting (radii 3/6/12 mm) also ran but leaves
more holes (t=1666 vs 1826 at cup4 step 0); alpha shape is the fallback.

## B. Open3D 0.19 RaycastingScene + render path (verified)

API as actually run:

```python
scene = o3d.t.geometry.RaycastingScene()
gid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))  # -> 0 (int)
rays = o3d.core.Tensor(arr_f32)          # (N, 6) float32: ox oy oz dx dy dz
ans = scene.cast_rays(rays)
# keys: primitive_uvs, primitive_ids, geometry_ids, primitive_normals, t_hit
t_hit = ans["t_hit"].numpy()             # metres along dir (unit dirs); inf on miss
prim  = ans["primitive_ids"].numpy()     # uint32; INVALID_ID (0xFFFFFFFF) on miss
pnorm = ans["primitive_normals"].numpy() # (N, 3) float32, winding-sided
```

Measured: **407,040 rays (full 848×480 frame) in 0.015 s.**

Render path, verified end-to-end against reality: rays from color
intrinsics (fx 431.6435, fy 431.2938, cx 417.2223, cy 237.8014, 848×480,
pinhole — distortion ignored) + the RAW recorded `T_base_cam` of cup3
step 005 (`rgb_rotation_deg` is 0 for that step, so raw == upright there);
mesh from step 029.

```python
R, t = T[:3, :3], T[:3, 3]                       # T_base_cam, OpenCV axes, cam->base
u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
dirs_cam  = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
dirs_base = dirs_cam @ R.T
dirs_base /= np.linalg.norm(dirs_base, axis=-1, keepdims=True)
rays = np.concatenate([np.broadcast_to(t, dirs_base.shape), dirs_base], -1).astype(np.float32)
# shading (double-sided abs() so mesh winding cannot black out a face):
# shade = 0.15 + 0.55*|n·view| + 0.30*|n·(0.3,0.3,0.9)/norm|, background 0
```

**Result: hit-pixel bbox [309, 125, 497, 320] vs the recorded SAM mask box
[303, 134, 507, 328] — agreement within ~10 px** (inside the hand-eye +
registration budget; pinhole-vs-distortion error is below that). The
side-by-side PNG shows the mesh render at the same image position/scale as
the real mug, handle on the same side, cavity visible. The render path is
correct; t_hit range 0.205–0.298 m matched the true standoff.

## C. Candidate generation (verified)

### C1. Sampling — and a determinism trap

`mesh.sample_points_poisson_disk(n)` is **nondeterministic across calls in
the same process** (measured: two consecutive calls differ).
`o3d.utility.random.seed(k)` before each call pins it exactly (measured:
same seed → allclose True, different seed → differs). Seed it, always.

- Coverage samples: `sample_points_poisson_disk(3000,
  use_triangle_normal=True)` on the trimmed mesh (~4 mm spacing at the
  ~480 cm² step-29 mesh), 0.3 s.
- Sample normals: take the mesh normal but **flip to agree with the
  nearest oriented cloud normal** (kills the winding problem for Poisson
  and alpha meshes alike):

```python
S = np.asarray(sp.points); SN = np.asarray(sp.normals).copy()
_, idx = cKDTree(P).query(S)
SN[np.einsum("ij,ij->i", SN, N[idx]) < 0] *= -1
```

### C2. Offset distance

Candidates = 300 poisson-disk seed points (seeded) offset along their
oriented normal. Offset sweep at step 29 (coverable fraction = candidates
surviving the table cull; coverage numbers from §E):

| offset | candidates kept /300 | 12-view coverage |
|---|---|---|
| 0.10 m | 244 | 0.756 |
| 0.12 m | 229 | 0.757 |
| 0.16 m | 226 | ~same, coverable denom drops 0.79→0.71 |

Larger offsets buy nothing (the 89° HFOV already sees a 0.20 m footprint
at 0.10 m — twice the cup) and push low candidates under the table.
**Recommend 0.10 m**: middle of the D405 sweet spot (0.08–0.12), inside
the usable window with margin on both sides.

### C3. Look-at rotation (OpenCV convention) — verified against 30 recorded poses

Columns of R are the camera axes in base frame; +Z boresight, +Y image-down:

```python
def lookat_cv(pos, target, down=np.array([0., 0., -1.])):
    z = target - pos; z /= np.linalg.norm(z)          # boresight
    y = down - np.dot(down, z) * z                    # image-down ~ world-down
    if np.linalg.norm(y) < 1e-6:                      # top-down degenerate
        y = np.array([0., 1., 0.]) - np.dot([0., 1., 0.], z) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)   # x = y × z (right-handed)
```

Verification: for each of the 30 recorded cup3 `T_base_cam`, rebuild R from
position + recorded boresight with this rule and measure the rotation-angle
difference (pure roll by construction). **Median 0.0°.** Exceptions:
exactly poses {1, 2, 3, 17, 21, 22, 23} are 180.0° off — and those are
**precisely the poses where `cell/geometry.py is_half_turn()` is True**
(the known raw-orientation trap: the planner half-turns the camera for
reachability; physically equivalent view). Pose 000 is 16° off (operator
roll). So the recorded system itself uses the world-down roll rule modulo
half-turns. **Roll disambiguation: world-down rule = upright image; store
that; half-turns are an execution-time concern, not a planning one.**
Recorded boresights pass 2.7° median (max 12.8°) from the final centroid —
sanity that "camera looks at the object" holds in this data.

Planned views are stored as 4×4 `T_base_cam`, OpenCV convention, base
frame, metres — bitwise the same contract as the recorded steps.

## D. Table plane — deterministic, no RANSAC

The step clouds are object-only (the recorder already dropped the table and
kept points ≥ 8 mm above its fitted plane, `ABOVE_TABLE_M`). So the cloud's
low tail IS the table height plus 8 mm. Measured z-percentiles (metres):

| run | final step | z min | z p1 | step 002 p1 |
|---|---|---|---|---|
| cup3 | 029 | 0.0133 | 0.0160 | 0.0195 |
| cup4 | 032 | 0.0130 | 0.0156 | 0.0189 |
| cup5 | 030 | 0.0116 | 0.0147 | 0.0160 |

Stable to ~4 mm across runs AND across steps of the same run.
**Rule: `table_z = np.percentile(P[:,2], 1) - 0.008`** (p1 is robust to a
few low outliers; −8 mm undoes the recorder's above-table margin).
Cull: drop any candidate with `cam_z <= table_z + 0.01`; a camera above the
table whose normal-offset origin survives cannot "look up from under the
table" because its boresight passes through its above-table target.
Culls ~19% of candidates (244/300 kept at 0.10 m offset).
`fit_table` RANSAC is not used (unseeded, nondeterministic — known trap).

## E. Greedy orthogonality-weighted set cover (verified + timed)

### E1. Quality matrix

`Q[v, s] = cos(incidence)` if ALL gates pass, else 0. Gates, all measured
working: standoff `0.07 ≤ d ≤ 0.20`; incidence `cos > cos 60°` (normal vs
direction sample→camera); in-FOV with 20 px margin (project with fx, fy,
cx, cy; z > 0); self-occlusion: one ray camera→sample,
`t_hit > d − 0.004` (sample lies ON the mesh, so first hit ≈ the sample;
4 mm tolerance ≈ mesh roughness).

Cost at s=3000 × v=244: geometry terms vectorized + 125k–181k occlusion
rays = **0.06–0.13 s total**. The briefing's worry about runtime is dead:
the whole per-step pipeline is **≈ 6 s, dominated by Poisson meshing**
(mesh 4.8–5.5 s, sampling 0.3 s, candidates 0.03 s, visibility 0.1 s,
greedy < 0.01 s). Three full runs ≈ 10 min; renders 0.015 s/frame.

### E2. Scoring formula and variants (measured)

Weighted greedy: `gain(v) = Σ_s max(Q[v,s] − best[s], 0)`, pick argmax,
`best = max(best, Q[v])`. Compared against pure-binary greedy
(count newly covered, cos only as ε-tiebreak) on cup3 step 29:

| variant | cov@12 | cov@16 |
|---|---|---|
| weighted (cos credit) | 0.756 | 0.784 |
| binary + ε·cos tiebreak | 0.754 | 0.787 |

Identical coverage to within noise — **the orthogonality-weighted greedy
costs nothing in coverage and additionally maximizes view quality. Use it**
(as required). Doubling candidates to 600 bought +0.7% upper bound and
+0.6% cov@12 for 2× matrix cost — keep 300.

### E3. Stopping rule (recommendation)

Measured binary marginal gains (new samples/view, step 29): 457, 439, 375,
280, 181, 135, 96, 68, 42, 39, 36, 33, 25, 24, 15, 15, 15, 14, 12, 12.
Smooth decay, no clean knee; a coverage target alone would run forever
against the ceiling (§E4). **Recommend: stop when the newly-covered sample
count of the picked view < 1% of coverable samples (≈30), with a hard cap
of 12 views.** On this data the cap binds first (the 1% rule alone stops
at ~14); both are reported per plan so the curve is visible.

### E4. The measured ceiling — read this before holding the 95% bar

**≥95% of TOTAL mesh area at ≤60° incidence is not achievable on these
objects, by physics, not by algorithm.** Upper bound = union of ALL ~244
candidates (cup3 step 29): **82.8% of all samples**. Breakdown: 11.6% of
samples face downward (`n_z < −0.5`: the underside ring the recorder never
saw but Poisson closes near the cloud), 13.8% sit within 10 mm of the
table band, and the deep cavity wall fails 60° from every above-table
standoff-legal position (0.7% fail incidence everywhere; 16.4% pass
incidence somewhere but are occlusion/standoff/FOV-blocked everywhere).
At ≤70° incidence the upper bound rises to 86.9%.

With the denominator restricted to **coverable** samples (those whose own
normal-offset viewpoint survives the table cull — the same rule the
candidates live by; 71–79% of samples):

| run (final step) | UB over coverable | cov@6 | cov@12 | cov@16 |
|---|---|---|---|---|
| cup3 (029) | 0.874 | 0.647 | 0.763 | 0.793 |
| cup4 (032) | 0.844 | — | 0.749 | 0.775 |
| cup5 (030) | 0.735 | — | 0.622 | 0.644 |

12 views reach 87% of the achievable union on cup3; pushing to 95% of the
union needs ~28 views of one-sample-each tail. cup5 is worst (deeper
cavity fraction). **Recommendation: report three numbers per plan —
coverage of total surface, coverage of coverable surface, and fraction of
achievable union — and put the success bar on the third.** The bar "95% of
mesh area at ≤60°" as literally written fails on every run at every step;
that is a fact about cups on tables seen by a standoff-limited camera,
and renegotiating the metric with Anton is part of showing the evidence.

Early-step behaviour (cup3 step 5, total-surface denominator): 1 view
32.1%, 3 views 64.8%, 6 views 82.6%, 12 views 90.5% — early sparse meshes
are EASIER to cover (less resolved concavity), so plan-vs-step curves must
be read together with mesh area, not alone.

### E5. Proof renders

Rendered greedy view #1 (pos [0.204, −0.299, 0.052], 462 samples claimed)
via the §B path with its claimed-covered samples projected as green dots:
every dot lands on the visible, well-oriented surface of the cup body +
handle in frame — the render frames exactly what it claims to cover
(`/tmp/vp_poc/planned_view1.png`, to be regenerated under
`derived/<run>/viewplan/` by the pipeline).

## F. Answers in one line each

- **(a)** estimate normals (r=12 mm, knn=30) → orient toward nearest
  recorded camera → Poisson depth=8 → trim vertices >6 mm from cloud;
  alpha-shape α=12 mm fallback (never triggered on these runs);
  ~7.9k/17.4k verts at steps 5/29; not watertight (fine); outwardness by
  cloud-normal transfer, never mesh winding.
- **(b)** API verified verbatim (§B); 407k rays 15 ms; recorded-pose render
  matches the recorded mask box within 10 px.
- **(c)** 3000 seeded poisson-disk samples / 300 candidate seeds; offset
  0.10 m; look-at with world-down roll rule — 0.0° median vs recorded
  poses, 180° outliers are exactly `is_half_turn` poses; seed
  `o3d.utility.random` or nothing reproduces.
- **(d)** `table_z = p1(z) − 8 mm`, verified stable across all 3 runs and
  across steps; cull `cam_z ≤ table_z + 1 cm`; no RANSAC anywhere.
- **(e)** weighted greedy `Σ max(Q−best, 0)`; stop at gain <1% of
  coverable or 12 views; visibility 0.1 s, greedy instant, ~6 s/step
  end-to-end; hard coverage ceiling 73–87% of coverable surface — the
  95%-of-total bar is physically unreachable and must be re-based on the
  achievable union.
