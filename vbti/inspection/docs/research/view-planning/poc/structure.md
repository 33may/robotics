# viewplan POC — definitive build spec

`inspection/viewplan/` is a peer of `inspection/fit/`: same repo conventions
(module docstring with `p path/mod.py cmd --flags` usage lines, argparse
subparsers with `set_defaults(fn=...)`, `main() -> int`, `sys.exit(main())`),
same contract discipline (dataclasses + lossless JSON round-trip, abstain as a
first-class answer), same output discipline (everything derived, under
`inspection/data/derived/<run>/viewplan/`). No imports from `vbti.*`. Python is
always `/home/anton/miniconda3/envs/robo/bin/python` (py3.11, open3d 0.19.0,
rerun-sdk 0.37.0, scipy 1.17.1 — all verified installed).

Pipeline per (run, step): accumulated object PLY -> Poisson mesh (artifact)
-> deterministic surface sampling -> normal-offset candidate cameras ->
visibility/measurability matrix (standoff window, incidence cutoff, FOV,
self-occlusion raycast) -> orthogonality-weighted greedy set cover -> ordered
`PlannedView` list + metrics + renders + rerun workspace.

## Pinned environment facts (verified on disk, 2026-09-07)

- Step clouds: `inspection/data/derived/<run>/steps/step_NNN.ply`, Open3D
  binary PLY, double xyz, base frame, Z-up, metres, **no normals, no colors**.
  2408-cup3: steps 000–029; 2408-cup4: 000–032; 2408-cup5: 000–029 with **no
  `runs/2408-cup5/029/` raw dir** (documented quirk — never index raw
  `runs/<run>/NNN/` by step count; index by the NNN in the step filename, and
  tolerate a missing raw dir). Accumulated: last step == final fused cloud.
- Final clouds: z in [0.012, 0.105] m; the table is horizontal in base frame
  and the clouds are object-only (everything ≥ 8 mm above the fitted plane
  was kept — `cell/geometry.py ABOVE_TABLE_M = 0.008`).
- Intrinsics: `runs/<run>/session.json -> intrinsics["color"]`
  (fx≈431.64, fy≈431.29, ppx≈417.2, ppy≈237.8, 848x480).
  `depth_scale_m_per_unit` ≈ 1e-4.
- `runs/<run>/NNN/meta.json` has `T_base_cam` (RAW orientation — the
  rotation trap of `investigation/normal_probe.py`: `rgb.png` may be stored
  upright while the pose stays raw; we only READ poses, never pair them with
  images, so the trap does not bite the planner, but `workspace.py` must not
  invent image pairing either).
- Camera convention for ALL planned views: **OpenCV** (+Z boresight, +X
  right, +Y down), 4x4, base frame, metres — identical to `T_base_cam` on
  disk and to `cell/geometry.py:project_to_pixels`. The viewsphere +X-forward
  convention (`view/viewsphere.py`, `_CV_TO_XFWD` at `cell/geometry.py:36`)
  is NOT used anywhere in viewplan.
- D405 usable standoff 0.07–0.20 m, sweet spot 0.08–0.12 m.
- Open3D 0.19 API (verified signatures):
  - `TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0,
    scale=1.1, linear_fit=False, n_threads=-1)` — **pass `n_threads=1`** for
    determinism.
  - `sample_points_poisson_disk` / `sample_points_uniformly` have **NO seed
    parameter** in this build → surface sampling must be our own
    numpy `default_rng(seed)` sampler (spec below), never the o3d samplers.
  - `o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic_matrix,
    extrinsic_matrix, width_px, height_px)` — extrinsic is **world→camera**,
    i.e. `np.linalg.inv(T_base_cam)`.
- `cell/geometry.py:fit_table` is unseeded RANSAC — **do not call it** in the
  planner. The table plane is derived deterministically (below).
- Rerun: `inspection/record/rr/log.py` is the ONLY module that talks to
  rerun. Timeline is the int sequence `"step"`; world is base frame Z-up.
- READ-ONLY: `inspection/data/runs/**`. Another session owns
  `inspection/record/` — the only permitted touch is **strictly additive
  functions appended to `inspection/record/rr/log.py`** (one delimited block
  at end of file, no edits to existing lines).

## Output directory layout

```
inspection/data/derived/<run>/viewplan/<variant>/
  step_NNN.json                 # StepPlan (schema below) — NNN = pose id, 3 digits
  meshes/step_NNN.ply           # reconstructed mesh artifact (o3d binary PLY)
  samples/step_NNN.npz          # surface samples + coverage arrays (schema below)
  renders/step_NNN_view_KK.png  # shaded raycast render from planned view KK
  renders/step_NNN_coverage.png # coverage-coloured render (top-3/4 orbit panels)
  scores.jsonl                  # one metrics row per step, merge-never-truncate
                                #   (exactly the fit/cli.py merge pattern)
inspection/data/derived/<run>/rrd/<run>-viewplan.rrd   (+ sidecar .rbl)
```

`step_NNN.json` NNN is the **pose id** from the step filename (matches
`steps/step_NNN.ply`), same as fits/.

## Package layout & ownership

```
inspection/viewplan/
  __init__.py         AGENT-CORE   (empty module docstring, created first)
  contract.py         AGENT-CORE
  mesh.py             AGENT-CORE
  planner.py          AGENT-CORE
  render.py           AGENT-CORE
  cli.py              AGENT-INTEGRATE
  workspace.py        AGENT-RR
  tests/
    __init__.py       AGENT-CORE
    synth.py          AGENT-CORE   (synthetic sphere/cylinder/plane clouds)
    test_contract.py  AGENT-CORE
    test_mesh.py      AGENT-CORE
    test_planner.py   AGENT-CORE
    test_render.py    AGENT-CORE
    test_workspace.py AGENT-RR     (JSON-fixture driven, no core imports)
    test_cli.py       AGENT-INTEGRATE
inspection/record/rr/log.py   AGENT-RR — append-only block (functions below)
```

The file sets are disjoint; no two agents touch the same file. The interface
between them is **this document's JSON/npz schema**, not Python imports:
AGENT-RR reads `step_NNN.json` as a plain dict + the npz + the mesh PLY, and
never imports `inspection.viewplan.contract` (decoupling is deliberate — RR
can build and test against hand-written fixture dicts before CORE lands).
AGENT-INTEGRATE imports CORE's public API (named below) and shells out to
nothing.

---

## contract.py (AGENT-CORE)

Mirrors `fit/contract.py` exactly in style: dataclasses with `__post_init__`
coercion to plain python types (reuse the `_jsonify` pattern), total and
lossless serialisation — `plan_from_json(plan_to_json(p)) == p` — no custom
JSON encoder needed.

```python
@dataclass
class PlannedView:
    view_id: int                      # rank in greedy order, 0-based
    T_base_cam: List[List[float]]     # 4x4 row-major, OpenCV conv, base frame, m
    standoff_m: float                 # commanded surface distance of this view
    gain: float                       # weighted-area marginal gain at selection [m^2]
    covered: List[int]                # ALL sample ids this view measures (matrix row)
    n_covered: int                    # len(covered)  (derived, checked in __post_init__)
    area_m2: float                    # Σ area of covered samples
    mean_incidence_deg: float         # over covered samples
    p95_incidence_deg: float
    cum_area_frac: float              # coverage fraction after this view (greedy order)
    render: str = ""                  # relative path, filled by the render pass
    frame_fill: float = 0.0           # fraction of render pixels hitting mesh (render pass)

    # properties: .T -> np.ndarray 4x4; .position -> T[:3,3]; .boresight -> T[:3,2]
    # __post_init__: validate 4x4, R orthonormal within 1e-6, det=+1; coerce ints/floats
    # to_json() / from_json(d)  — same shape as Primitive
```

```python
@dataclass
class StepPlan:
    run: str
    step: int                         # pose id (the NNN)
    variant: str
    version: str                      # "<variant>/0.1+p:<sha1[:8]>" (fingerprint below)
    seed: int
    abstain: bool = False             # sparse/degenerate early steps refuse honestly
    reason: str = ""
    table_plane: List[float] = ...    # [a,b,c,d], unit normal +z, base frame
    mesh: Dict[str, Any] = ...        # artifact reference block (below)
    samples: Dict[str, Any] = ...     # sidecar reference block (below)
    params: Dict[str, Any] = ...      # the FULL variant parameter dict (self-describing)
    views: List[PlannedView] = ...
    metrics: Dict[str, Any] = ...
    # __post_init__: abstain=True ⇒ views empty (raise otherwise), coerce all floats
    # to_json() / from_json(d); module-level plan_to_json / plan_from_json wrappers
```

```python
def fingerprint(variant: str, params: Dict) -> str:
    # sha1 of json.dumps(params, sort_keys=True), first 8 hex chars →
    # f"{variant}/0.1+p:{h}"  — same idea as fit/common.py version fingerprints
def abstained(run, step, variant, params, seed, reason) -> StepPlan
```

### Exact JSON schema of `step_NNN.json`

```jsonc
{
  "run": "2408-cup3",
  "step": 10,
  "variant": "base",
  "version": "base/0.1+p:ab12cd34",
  "seed": 0,
  "abstain": false,
  "reason": "",
  "table_plane": [0.0, 0.0, 1.0, -0.0053],       // ax+by+cz+d=0, n̂ points up
  "mesh": {
    "path": "meshes/step_010.ply",               // relative to the variant dir
    "method": "poisson",                          // "poisson" | "bpa" | "hull"
    "depth": 7,                                   // poisson octree depth (0 if not poisson)
    "n_vertices": 4210,
    "n_triangles": 8390,
    "area_m2": 0.0312,
    "watertight": false
  },
  "samples": {
    "path": "samples/step_010.npz",
    "count": 1500,
    "area_per_sample_m2": 2.08e-05                // mesh.area_m2 / count
  },
  "params": {                                     // full variant config, verbatim
    "standoff_m": 0.10,
    "standoff_window_m": [0.07, 0.20],
    "incidence_cutoff_deg": 60.0,
    "n_samples": 1500,
    "n_candidates": 256,
    "occlusion_tol_m": 0.006,
    "weight": "cosine",                           // "cosine" | "binary"
    "max_views": 12,
    "min_gain_frac": 0.005,
    "fov_margin_px": 24
  },
  "views": [ /* PlannedView objects, greedy order */
    {
      "view_id": 0,
      "T_base_cam": [[...],[...],[...],[0.0,0.0,0.0,1.0]],
      "standoff_m": 0.10,
      "gain": 0.0086,
      "covered": [3, 7, 12, ...],
      "n_covered": 412,
      "area_m2": 0.0086,
      "mean_incidence_deg": 22.4,
      "p95_incidence_deg": 51.0,
      "cum_area_frac": 0.276,
      "render": "renders/step_010_view_00.png",
      "frame_fill": 0.34
    }
  ],
  "metrics": {
    "n_candidates": 256,                          // after generation, before culls
    "n_culled_table": 41,
    "n_culled_unreached": 0,                      // candidates covering zero samples
    "n_views": 9,
    "coverage_area_frac": 0.968,                  // area covered at ≤ cutoff incidence
    "weighted_coverage": 0.81,                    // Σ area·best_cos / Σ area
    "mean_best_incidence_deg": 24.0,
    "uncovered_area_m2": 0.001,
    "seconds": 3.2
  }
}
```

Round-trip rule: every field above survives `from_json∘to_json` bit-exactly;
unknown keys in `from_json` are ignored, missing optional keys default (same
tolerance as `Primitive.from_json`).

### `samples/step_NNN.npz` schema (numpy-only sidecar; the RR contract)

| key          | shape  | dtype | meaning                                            |
|--------------|--------|-------|----------------------------------------------------|
| `points`     | (S,3)  | f64   | sample positions, base frame, m                    |
| `normals`    | (S,3)  | f64   | outward unit normals                               |
| `area`       | (S,)   | f64   | area share per sample (constant = area_m2/S)       |
| `best_cos`   | (S,)   | f64   | best incidence cosine over SELECTED views, 0=uncovered |
| `covered_by` | (S,)   | i32   | view_id achieving best_cos, −1 = uncovered         |

Written with `np.savez_compressed`. This is everything `workspace.py` and the
coverage render need — no recomputation, no core import.

---

## mesh.py (AGENT-CORE)

PLY-cloud → triangle-mesh reconstruction with a robust fallback ladder for
sparse early steps. Pure functions; no CLI.

```python
def maturity_reason(P: np.ndarray) -> str
    # reuse thresholds from fit/common.py (MIN_POINTS=250, MIN_EXTENT_M=0.02,
    # MIN_BBOX_RANK=2). Import fit.common — allowed (fit is a sibling package,
    # dependency direction fit ← viewplan is fine and avoids copy-paste).

def reconstruct(P: np.ndarray, *, seed: int = 0) -> tuple["o3d TriangleMesh", dict]
    # 1) normals: fit.common.estimate_normals (radius 12 mm, knn 30, oriented
    #    outward from centroid — deterministic, sign-pinned).
    # 2) Poisson: depth chosen from cloud extent (aim ~2× point spacing per
    #    octree cell; depth 6 for <1k pts, 7 otherwise), n_threads=1,
    #    linear_fit=False.
    # 3) density trim: drop vertices below the 5th percentile of the returned
    #    density vector (kills the Poisson bubble).
    # 4) crop to cloud AABB inflated by 6 mm (2 voxels); keep the largest
    #    connected component; remove degenerate/duplicated triangles.
    # 5) sanity gate: n_triangles ≥ 100 AND mesh area ∈ [0.3, 3.0] × a crude
    #    cloud-area estimate (S · spacing², spacing from fit.common.cloud_spacing).
    #    On failure → fallback: ball-pivoting at radii (1.5, 2.5, 4)× spacing
    #    ("bpa"); if THAT fails → convex hull ("hull") — always returns a mesh
    #    or raises ValueError with a one-sentence reason (caller turns it into
    #    an abstain).
    # info dict = the "mesh" JSON block minus "path".

def save(mesh, path: Path) -> None      # o3d binary PLY, parents created
def load(path: Path) -> "o3d TriangleMesh"

def table_plane(P: np.ndarray) -> list[float]
    # DETERMINISTIC, no RANSAC: the step clouds are object-only points that
    # were kept ≥ 8 mm above the fitted table, base frame is Z-up, table is
    # horizontal ⇒ plane = [0,0,1,-(min_z − 0.008)]. Document the assumption
    # in the docstring; it is exact for these runs (final min_z ≈ 0.012–0.013).
```

---

## planner.py (AGENT-CORE)

Pure, seeded, deterministic functions. `np.random.default_rng(seed)` is the
only randomness; every argsort/argmax tie-break is by lowest index.

```python
VARIANTS: dict[str, dict] = {
  "base":     {standoff 0.10, weight "cosine", n_candidates 256, ...},   # the defaults above
  "binary":   {... weight "binary" ...},              # plain set cover — the comparison Anton asked for
  "close":    {... standoff 0.08 ...},                # sweet-spot near edge
  "far":      {... standoff 0.14 ...},                # wider frustum, fewer views?
  "dense":    {... n_candidates 512, n_samples 3000 ...},   # does candidate density matter
}
def variants() -> list[str]           # sorted(VARIANTS) — cli mirrors bank.methods()

def sample_surface(mesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]
    # OUR OWN sampler (o3d's has no seed): area-weighted triangle choice via
    # rng.choice(cum-area), uniform barycentric coords; normal = triangle
    # normal (outward — flip any normal whose dot with (sample − cloud
    # centroid) < 0, matching fit.common's sign rule). Returns (P (n,3), N (n,3)).

def look_at_cv(position: np.ndarray, boresight: np.ndarray) -> np.ndarray
    # 4x4 OpenCV-convention pose: column Z = unit boresight; column X = the
    # DETERMINISTIC tangent (cross of the world axis least parallel to Z with
    # Z, normalised — the frame_from_axis rule from fit/contract.py); column
    # Y = Z × X. Right-handed, det +1. THE only pose constructor in the package.

def candidate_views(P, N, params, table_plane) -> tuple[np.ndarray, dict]
    # 1) subsample the S samples to n_candidates anchor points by seeded
    #    farthest-point (start at index of point closest to centroid — no rng
    #    needed, fully deterministic).
    # 2) each anchor → camera at p + standoff·n̂, boresight −n̂ (opposing the
    #    normal = orthogonal view), pose via look_at_cv.
    # 3) TABLE CULL (the single guard): drop a candidate iff camera_z is
    #    below the plane (n̂·pos + d < 0.01) OR boresight points upward more
    #    than 15° below horizontal from under the object (boresight_z > 0.26
    #    ⇒ looking up ⇒ physically under the table for these table-top
    #    objects — both tests, cheap and sufficient; NO IK, NO reachability).
    # returns (T (M,4,4), {"n_generated": ..., "n_culled_table": ...})

def visibility(mesh, T: np.ndarray, P, N, intr, params) -> tuple[np.ndarray, np.ndarray]
    # cover (M,S) bool, cosv (M,S) f64. A view measures a sample iff ALL of:
    #   range:      standoff_window[0] ≤ dist(cam, p) ≤ standoff_window[1]
    #   incidence:  cos = n̂·(cam−p)/dist ≥ cos(incidence_cutoff_deg)
    #   fov:        project_to_pixels-style pinhole (reimplement the 4 lines
    #               locally to keep cell/ out of the dependency set, or import
    #               cell.geometry.project_to_pixels — importing is fine per
    #               AGENTS.md direction, prefer the import) puts (u,v) inside
    #               [fov_margin_px, W−fov_margin_px] × [same for H]
    #   occlusion:  one RaycastingScene built ONCE from the mesh
    #               (scene.add_triangles(TriangleMesh.from_legacy(mesh)));
    #               rays origin=cam, dir=(p−cam)/dist for the M·S candidate
    #               pairs that survived the cheap tests, batched;
    #               visible iff t_hit ≥ dist − occlusion_tol_m (0.006 = 2
    #               voxels; the sample IS on the mesh, so the first hit is
    #               itself unless something interposes).
    # cosv is cos where cover else 0.

def greedy(cover, cosv, area, params) -> list[dict]
    # Orthogonality-weighted greedy set cover:
    #   quality q_vs = cosv (weight="cosine")  or  cover.astype(float) ("binary")
    #   gain(v) = Σ_s area_s · max(0, q_vs − best_s)      (best_s starts at 0)
    #   pick argmax gain (ties → lowest v), update best_s, repeat until
    #   n_views == max_views OR gain < min_gain_frac · total_area.
    # returns per-pick dicts: {view_idx, gain, covered_ids, best_after (S,)}
    # — pure numpy, no rng at all.

def plan_step(P_cloud, run, step, variant, seed) -> tuple[StepPlan, mesh, dict[str, np.ndarray]]
    # THE composition the cli and the tests both call (the bank.fit analogue):
    # maturity gate → abstain | reconstruct → table_plane → sample_surface →
    # candidate_views → visibility → greedy → StepPlan + npz arrays
    # (points/normals/area/best_cos/covered_by). Any raise inside becomes an
    # abstain naming the exception, exactly like bank.fit.
```

Cost sanity: M=256 candidates × S=1500 samples = 384k pairs; the cheap masks
kill ~90 % before raycasting; the remaining ≤40k rays are one batched
`cast_rays` call — well under a second per step on CPU.

---

## render.py (AGENT-CORE)

Offscreen evidence via RaycastingScene + numpy shading. NO OpenGL, NO
pyrender, matplotlib only for contact sheets.

```python
def render_view(mesh_or_scene, T_base_cam, intr, *, scale=0.5,
                cover_ids=None, samples=None) -> np.ndarray  # (H,W,3) uint8
    # rays = RaycastingScene.create_rays_pinhole(K', inv(T_base_cam), W', H')
    #   with K' = K·scale, W'=424, H'=240 at scale 0.5 (renders are evidence,
    #   not measurements).
    # shading: lambert n·v headlight (light at the camera), primitive_normals
    #   from the cast, background dark slate; depth-window tint: pixels whose
    #   t_hit is outside the standoff window get a red wash (shows at a glance
    #   when a view is out of the D405 window).
    # if cover_ids/samples given: splat the covered sample points into the
    #   image (project, 2-px dots, accent orange) so "the render frames the
    #   surface it claims to cover" is visually checkable per view.
    # returns image; also computes frame_fill = hits/pixels for the caller.

def coverage_panel(mesh, samples_npz: dict, out_png: Path) -> Path
    # 4 orbit viewpoints (fixed elev/azim), samples coloured by best_cos:
    #   uncovered = red, cutoff..1 = viridis; mesh in neutral grey underneath.
    #   Rendered by the same raycaster (orbit cameras via look_at_cv) — one
    #   PNG per step: renders/step_NNN_coverage.png.

def render_plan(variant_dir: Path, plan_json: dict, intr) -> list[Path]
    # loads mesh + npz by the paths IN the json, renders every PlannedView,
    # writes renders/step_NNN_view_KK.png, RETURNS the per-view frame_fill so
    # the caller (cli) can write them back into step_NNN.json. Works from the
    # dict, not the dataclass, so INTEGRATE can call it late in the pipeline.
```

---

## cli.py (AGENT-INTEGRATE)

Mirrors `fit/cli.py` flag-for-flag (same `steps_of`, `select`, `load`
helpers, same merge-never-truncate `scores.jsonl` logic, same multiprocessing
`spawn` `--jobs` shape with a top-level `_job`).

```
p inspection/viewplan/cli.py run  --run 2408-cup3 --steps all --variant base,binary --seed 0 --jobs 4
p inspection/viewplan/cli.py run  --run 2408-cup3,2408-cup4,2408-cup5 --steps all --variant all
p inspection/viewplan/cli.py show --run 2408-cup3 --variant base --steps final
p inspection/viewplan/cli.py variants
```

- `run`: per (run, step, variant): load PLY → `planner.plan_step` → write
  mesh artifact (`mesh.save`), npz, `step_NNN.json` → `render.render_plan` +
  `render.coverage_panel` → patch `render`/`frame_fill` fields into the JSON
  → append metrics row to `scores.jsonl` (row = metrics block + run/step/
  variant/seconds/abstain/reason). `--no-render` flag skips the render pass
  (fast iteration). Prints the fit/cli.py-style table: run, step, N, variant,
  n_views, coverage, mean incidence, seconds, reason.
- `show`: print a stored plan — per view: id, position, standoff, n_covered,
  area, mean/p95 incidence, cum coverage.
- `variants`: names + versions (fingerprints) of `planner.VARIANTS`.
- After `run`, per (run, variant) summary line: first non-abstain step, first
  step reaching coverage ≥ 0.95, final n_views/coverage — **the "after which
  step is the 3D good enough" answer printed where Anton will see it.**

## workspace.py (AGENT-RR)

Builds `derived/<run>/rrd/<run>-viewplan.rrd` (+ `.rbl`) for ONE variant
(default `base`, `--variant` to switch; app id `viewplan-<run>-<variant>` —
dash, never slash, per the 0.37 app-id trap in record/rr/workspace.py:224).
Reads ONLY: `derived/<run>/steps/` (clouds + summary.json),
`derived/<run>/viewplan/<variant>/` (JSON dicts via `json.loads`, npz via
`np.load`, mesh PLY via o3d), `runs/<run>/session.json` (intrinsics). No
`inspection.viewplan` imports. Never touches `derived/<run>/rrd/<run>.rrd`
(the record session's file) and never imports anything from
`inspection/record/` except `record.rr.log`.

Per step k (timeline `at(k)`, walking `steps/summary.json` order exactly like
record/rr/workspace.py):

1. `log_cloud(step points)` — existing.
2. `log_viewplan_mesh(V, F)` — NEW: this step's reconstructed mesh.
3. `log_viewplan_coverage(points, best_cos, cutoff_cos)` — NEW: samples
   coloured by measurability.
4. per PlannedView: `log_viewplan_camera(view_id, T, intr, label=...,
   image_path=renders/step_NNN_view_KK.png)` — NEW: pinhole + frustum +
   render-on-image-plane in a `/world/plan` subtree, re-logged each step
   (plans CHANGE between steps — stale frusta must be retired with
   `rr.Clear`, the `log_primitives`/`clear_primitives` diff pattern: the RR
   function returns paths, workspace diffs them step to step).
5. `log_viewplan_meta(variant, **metrics)` — NEW: abstain/coverage/n_views on
   the plan entity for the selection panel.
6. static once: `log_world()`, `log_table(plan.table_plane, cloud xy mean)`.

Blueprint: `viewplan_blueprint(...)` — NEW: 3D scene left (cloud + mesh +
coverage + planned frusta), right a 2D grid of the current step's view
renders, time panel pinned to `step`, paused, `auto_views=False`. Saved via
existing `save(path, blueprint, app_id)`.

### The additive functions in `inspection/record/rr/log.py` (AGENT-RR)

One appended block, delimited `# ---- viewplan additions (additive only) ----`,
no existing line touched. Exact signatures:

```python
VIEWPLAN = "/world/plan"            # new subtree: one eye-click kills the plan
VP_MESH = VIEWPLAN + "/mesh"
VP_COVER = VIEWPLAN + "/coverage"
VP_CAMS = VIEWPLAN + "/cam"        # /world/plan/cam/<view_id>
C_PLANNED = (255, 96, 210)          # magenta — far from every existing colour

def log_viewplan_mesh(vertices: np.ndarray, faces: np.ndarray,
                      vertex_normals: np.ndarray | None = None) -> None:
    # rr.Mesh3D at VP_MESH, neutral translucent albedo, per current step.

def log_viewplan_coverage(points: np.ndarray, best_cos: np.ndarray,
                          cutoff_cos: float) -> None:
    # rr.Points3D at VP_COVER: uncovered (best_cos < cutoff_cos) red,
    # covered mapped cutoff→1 on a green ramp; radii R_CLOUD*1.3.

def log_viewplan_camera(view_id: int, T_base_cam: np.ndarray, intr: Mapping,
                        label: str = "", image_path: str | Path | None = None) -> str:
    # exactly log_camera's three-level pattern (Transform3D + child Pinhole
    # with camera_xyz=RDF + optional textured-mesh preview via the existing
    # _log_preview_quad), but under VP_CAMS/<view_id:02d> and C_PLANNED.
    # RETURNS the entity path so workspace can diff & Clear retired views.

def log_viewplan_meta(variant: str, **fields) -> None:
    # rr.AnyValues on VIEWPLAN (the log_decomposition_meta pattern).

def viewplan_blueprint(run: str, variant: str,
                       render_slots: Sequence[str]) -> rrb.Blueprint:
    # Spatial3D (origin /world, contents $origin/**) left; rrb.Grid of
    # Spatial2DView over /renders/<slot> right; TimePanel pinned to "step",
    # paused, auto_views=False — the blueprint() pattern minus the chain.
```

(Plus `log_viewplan_render(slot: str, image_path)` → `rr.EncodedImage` at
`/renders/<slot>`, the `log_chain` pattern, for the right-hand grid.) Six
small functions, all new symbols, zero collision surface with the record
session.

## tests

Run with `p -m pytest inspection/viewplan/tests -q`; the existing
`p -m pytest inspection/tests -q` must stay green throughout (nothing in
viewplan may import from `inspection/record/` except `record.rr.log`, and
log.py changes are additive, so the only risk is a syntax error — RR runs
both suites after touching log.py).

`tests/synth.py` (CORE): seeded generators returning (points, analytic truth):
`sphere_cloud(r=0.04, n=4000, seed)`, `cylinder_cloud(r=0.03, h=0.10, ...)`,
`half_sphere_on_table(...)` (z ≥ table), all in a base-frame-like pose
(centre ≈ [0.1, −0.4, 0.05] — the real workspace, so table-cull tests mean
something).

Key cases:

- **test_contract.py**: round-trip equality on a 3-view plan built with
  numpy inputs (proves `_jsonify`); abstain-with-views raises; `from_json`
  tolerates missing optional keys; `look_at_cv`-produced pose passes the
  orthonormal/det gate; `fingerprint` changes when any param changes and is
  stable under key reordering.
- **test_mesh.py**: sphere cloud → mesh area within 15 % of 4πr²; 150-point
  stub → `maturity_reason` non-empty; pathological 60-point line → fallback
  ladder ends in "hull" or a clean ValueError (never a crash); `table_plane`
  on half_sphere returns d within 2 mm of truth.
- **test_planner.py**:
  - *sphere, analytic*: every selected view sits at ‖pos−centre‖ =
    r+standoff ± 1 mm and boresight·(centre−pos) direction error < 5°;
    coverage_area_frac ≥ 0.95 with ≤ 12 views (the success bar, proven where
    the answer is known).
  - *cylinder, analytic*: side-covering views have |boresight·axis| < 0.2;
    the top cap earns a near-vertical view.
  - *table cull*: half_sphere_on_table → no view position below the plane,
    no boresight_z > 0.26; the culled count lands in metrics.
  - *self-occlusion*: two parallel plane patches 3 cm apart, camera on one
    side → far patch samples not covered from that side (raycast proof).
  - *greedy, hand-built*: 3×4 cover/cos matrix with a known optimal order —
    weighted greedy picks it; binary variant picks the known different order.
  - *determinism*: `plan_step` twice, same seed → `json.dumps(to_json())`
    byte-identical (this pins Poisson `n_threads=1` and the no-o3d-sampler
    rule forever).
- **test_render.py**: sphere plan → `render_view` frame_fill > 0.05 for every
  selected view; ≥ 90 % of that view's covered samples project inside the
  image bounds and have |t_hit − dist| < 5 mm ("every render frames the
  surface it claims to cover", as a test, not a promise); output PNG exists
  and is non-black.
- **test_workspace.py** (RR): with a hand-written fixture dir (tiny JSON +
  npz + 8-triangle mesh in tmp_path) `build()` writes an .rrd + .rbl > 0
  bytes; retired-view diffing: step 0 has views {0,1}, step 1 has {0} →
  Clear was issued for view 1 (assert via the returned path bookkeeping).
- **test_cli.py** (INTEGRATE): `select`/`steps_of` on a tmp derived tree;
  `run --steps 0` on a synthetic PLY dropped into tmp (monkeypatched DERIVED)
  produces step_000.json + mesh + npz + scores.jsonl; second run with
  `--steps 0` merges rather than truncates scores.jsonl.

## Build order & integration protocol

1. AGENT-CORE lands `__init__.py` + contract.py + tests first (everything
   else codes against it), then mesh.py, planner.py, render.py.
2. AGENT-RR starts immediately from THIS schema with fixture dicts — no wait
   on CORE. Appends to log.py once, early, so the record session sees the
   block and avoids the tail of the file.
3. AGENT-INTEGRATE writes cli.py against the named CORE APIs
   (`planner.plan_step`, `planner.variants`, `mesh.save`,
   `render.render_plan`, `render.coverage_panel`), then runs:
   `run --run 2408-cup3,2408-cup4,2408-cup5 --steps all --variant all`,
   then `workspace.py` per run, then both pytest suites, and reports the
   success-bar numbers (≤12 views, ≥95 % area at ≤60°, stability step) per
   run/variant — facts, not optimism.

No git commits; everything stays in the working tree.
