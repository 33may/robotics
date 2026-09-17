# Ray Views — Stage A Implementation Plan (sampler offline, the falsifier)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ray sampler in the kinematic sandbox and prove on recorded runs that closest-safe-sample endpoints beat the shipped formula endpoints — renders + measured timings. No robot, no machine.py changes.

**Architecture:** A view is a ray (anchor + direction); distance = closest sample that passes sensor-range, IK and collision checks, computed offline. Cells: rays from the centroid along the {h,v} grid. Faces: a ray family — parallel +z shifts ascending (fronto-parallel, no keystone), then the 5°..60° cone. One sampler serves both.

**Tech Stack:** numpy, Pinocchio+coal via `inspection.cell.world.RobotCell`, EAIK via `inspection.motion.ik.UR5eIK`, script-style tests (`main()` pattern, no pytest).

**Spec:** `inspection/docs/2026-09-17-ray-views-design.md` — read it first.

## Global Constraints

- Stage A is sandbox-only: nothing imports `rigs`, `machine`, or touches the UR5e.
- `RAY_STEP_M = 0.02` and `MIN_RANGE_M = 0.105` are module constants in `rays.py` — configurable, never inlined.
- Tests are script-style: `test_*` functions + `main()` running all of them, `print("OK <name>")` at the end (copy the shape of `inspection/tests/test_faceview.py`).
- Run commands with the repo convention `p <file>` (python alias, conda env `robo` active; agents: `~/miniconda3/envs/robo/bin/python`).
- Recorded runs under `inspection/data/runs/` are READ-ONLY; bench artifacts go to `/tmp/ray_views/`.
- Rolls are `inspection.view.viewsphere.ROLLS` (0, π) — always both, per sample.
- Do not delete or modify `grid.py`/`viewsphere.py`/`faceview.py` in Stage A — the old path must stay runnable for the A/B comparison.

---

### Task 1: `rays.py` — Ray, pose builder, single-ray sampler

**Files:**
- Create: `inspection/view/rays.py`
- Test: `inspection/tests/test_rays.py`

**Interfaces:**
- Consumes: `RobotCell()` (`.set_object(name, dims, pose, parent="base")`, `.is_colliding(q)`), `UR5eIK().branches(T_flange, q_near=None) -> list[q]`, `T_FLANGE_CAM` + `ROLLS` from `inspection.view.viewsphere`, `face_cam_pose(center, normal, r, roll)` from `inspection.view.faceview` (the pose builder is ray geometry already — reuse, do not copy), `R_MAX` from `inspection.view.grid`.
- Produces: `Ray(anchor, direction)` (frozen dataclass, unit direction), `Sample(depth, roll, q, T_cam, T_flange)` (namedtuple), `sample_ray(world, ik, t_flange_cam, ray, clear_m=0.0, q_near=None) -> list[Sample]` — ascending depth, only provably-safe samples; `clear_m` is the camera-to-surface distance already consumed at depth 0 (0 for face rays — anchor on the wall; the anchor→surface distance along the ray for cell rays, Task 2 computes it).

- [ ] **Step 1: Write the failing tests** (new file; `_box_world()` helper builds a `RobotCell` with a 15×15×7.5 cm object box at `(0.05, -0.38, 0.0375)` like the 1709 demo, via `world.set_object("object", [0.15, 0.15, 0.075], [0.05, -0.38, 0.0375, 0, 0, 0.17])`)

```python
import numpy as np
from inspection.cell.world import RobotCell
from inspection.motion.ik import UR5eIK
from inspection.view.viewsphere import T_FLANGE_CAM


def _box_world():
    w = RobotCell()
    w.set_object("object", [0.15, 0.15, 0.075],
                 [0.05, -0.38, 0.0375, 0, 0, 0.17])
    return w, UR5eIK()


def test_samples_ascend_and_respect_sensor_floor():
    from inspection.view.rays import Ray, sample_ray, MIN_RANGE_M
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, -0.9701, 0.2425))
    out = sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=0.075)
    assert out, "an open oblique ray must have samples"
    depths = [s.depth for s in out]
    assert depths == sorted(depths)
    assert depths[0] - 0.075 >= MIN_RANGE_M - 1e-9


def test_no_sample_collides():
    from inspection.view.rays import Ray, sample_ray
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, -0.9701, 0.2425))
    for s in sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=0.075):
        assert not w.is_colliding(s.q)


def test_buried_ray_is_empty():
    from inspection.view.rays import Ray, sample_ray
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, 0.0, -1.0))   # into the table
    assert sample_ray(w, ik, T_FLANGE_CAM, ray) == []


def test_sample_pose_is_on_the_ray_looking_at_anchor():
    from inspection.view.rays import Ray, sample_ray
    w, ik = _box_world()
    ray = Ray((0.05, -0.38, 0.0375), (0.0, -0.9701, 0.2425))
    s = sample_ray(w, ik, T_FLANGE_CAM, ray, clear_m=0.075)[0]
    pos = s.T_cam[:3, 3]
    d = pos - np.asarray(ray.anchor)
    assert np.allclose(d / np.linalg.norm(d), ray.direction, atol=1e-6)
    assert np.allclose(s.T_cam[:3, 0], -np.asarray(ray.direction), atol=1e-6)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK test_rays")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify failure** — `p inspection/tests/test_rays.py` → ImportError (`inspection.view.rays` missing).

- [ ] **Step 3: Implement `inspection/view/rays.py`**

```python
"""Ray views (design: docs/2026-09-17-ray-views-design.md).

A view is a ray, not a point: distance is never computed by a formula —
it is the closest sample on the ray that is provably safe (sensor range,
IK, collision), found offline in the kinematic sandbox.
"""
from dataclasses import dataclass
from collections import namedtuple

import numpy as np

from inspection.view.faceview import face_cam_pose
from inspection.view.grid import R_MAX
from inspection.view.viewsphere import ROLLS

#: One knob for every ray discretisation: depth grid AND face shift grid.
RAY_STEP_M = 0.02
#: D405 minimum depth range at 720p (datasheet + camera.py note).
MIN_RANGE_M = 0.105


@dataclass(frozen=True)
class Ray:
    """Anchor point and unit direction, base frame."""
    anchor: tuple
    direction: tuple

    def __post_init__(self):
        d = np.asarray(self.direction, dtype=float)
        n = float(np.linalg.norm(d))
        if not np.isfinite(n) or n < 1e-9:
            raise ValueError("ray direction must be a nonzero vector")
        object.__setattr__(self, "direction", tuple(d / n))
        object.__setattr__(self, "anchor",
                           tuple(float(v) for v in self.anchor))


Sample = namedtuple("Sample", ["depth", "roll", "q", "T_cam", "T_flange"])


def sample_ray(world, ik, t_flange_cam, ray, clear_m=0.0, q_near=None):
    """All provably-safe samples on `ray`, ascending depth.

    `clear_m` = camera-to-object-surface distance already consumed at depth
    0 (0 for a face ray — its anchor sits ON the wall; the anchor→surface
    run for a centroid ray). The sensor floor is surface distance, so a
    sample is admissible from depth clear_m + MIN_RANGE_M.
    """
    t_fc_inv = np.linalg.inv(np.asarray(t_flange_cam, dtype=float))
    lo = clear_m + MIN_RANGE_M
    d0 = float(np.ceil(lo / RAY_STEP_M)) * RAY_STEP_M
    out = []
    for d in np.arange(d0, R_MAX + 1e-9, RAY_STEP_M):
        for roll in ROLLS:
            T_cam = face_cam_pose(ray.anchor, ray.direction, float(d), roll)
            T_fl = T_cam @ t_fc_inv
            free = [q for q in ik.branches(T_fl, q_near=q_near)
                    if not world.is_colliding(q)]
            if free:
                out.append(Sample(float(d), float(roll), free[0],
                                  T_cam, T_fl))
                break                     # first roll wins at this depth
    return out
```

- [ ] **Step 4: Run tests to verify pass** — `p inspection/tests/test_rays.py` → `OK test_rays`.

- [ ] **Step 5: Commit**

```bash
git add inspection/view/rays.py inspection/tests/test_rays.py
git commit -m "rays: Ray + sample_ray — distance is the closest provably-safe sample"
```

---

### Task 2: cell rays and the face ray family

**Files:**
- Modify: `inspection/view/rays.py` (append)
- Test: `inspection/tests/test_rays.py` (append; new tests auto-run via `main()`)

**Interfaces:**
- Consumes: Task 1 `Ray`; `H_BINS`, `V_ELEVATIONS` from `inspection.view.grid`; `face_frame(prim, face)` + `FACES` from `inspection.view.faceview`; boxguard `Primitive` (has `.center`, `.axes`, `.params["extents"]` — see `_box_prim()` in `inspection/tests/test_faceview.py`, copy that helper into the test file).
- Produces:
  - `cell_ray(center, h, v) -> Ray` and `cell_clear_m(prim_or_aabb, ray) -> float` (anchor→padded-surface run along the ray, slab method on the object AABB).
  - `face_family(prim, face, fy, height_px) -> list[tuple[Ray, float, float, float]]` — ordered `(ray, shift, theta_deg, d_min)` candidates: shift rung ascending (θ=0), then cone rung (θ = 5..60 step 5, 12 azimuths, shift=0). **AMENDED by ruling 2026-09-17 (execution proved the original cap formula rejects all shifts for a 15 cm face):** FOV containment is per-depth, so each member carries its own minimum depth `d_min = max(MIN_RANGE_M, (2*fy/height_px)*(shift + ext_v/2))` for side faces (`_d_contain`), `max(MIN_RANGE_M, _d_contain(max(ext_a, ext_b)/2))` for the top face, and plain `MIN_RANGE_M` for cone members (tilted views don't promise containment). Shift enumeration is bounded by `d_min <= R_MAX`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_cell_ray_matches_grid_direction():
    from inspection.view.rays import cell_ray
    r = cell_ray((0.05, -0.38, 0.0375), h=0, v=0)
    # h=0 faces the base: direction's azimuth == azimuth of center->origin
    az = np.arctan2(0.38, -0.05)
    assert np.isclose(np.arctan2(r.direction[1], r.direction[0]), az,
                      atol=1e-6)


def test_cell_clear_is_box_surface_run():
    from inspection.view.rays import Ray, cell_clear_m
    aabb = ((-0.025, -0.455, 0.0), (0.125, -0.305, 0.075))
    ray = Ray((0.05, -0.38, 0.0375), (1.0, 0.0, 0.0))
    assert np.isclose(cell_clear_m(aabb, ray), 0.075, atol=1e-6)


def test_face_family_shifts_before_tilts():
    from inspection.view.rays import face_family
    p = _box_prim()                      # copied from test_faceview.py
    fam = face_family(p, "y-", fy=647.0, height_px=720)
    shifts = [(s, t) for _, s, t in fam if t == 0.0]
    tilts = [(s, t) for _, s, t in fam if t > 0.0]
    assert shifts and tilts
    # the family is ordered: ALL shift entries strictly before ANY tilt
    assert [t for _, _, t in fam[:len(shifts)]] == [0.0] * len(shifts)
    assert [s for s, _ in shifts] == sorted(s for s, _ in shifts)
    assert all(s == 0.0 for s, _ in tilts)


def test_face_family_shift_cap_is_fov_containment():
    from inspection.view.rays import face_family, MIN_RANGE_M
    p = _box_prim()
    fam = face_family(p, "y-", fy=647.0, height_px=720)
    ext_v = p.params["extents"][2]
    cap = MIN_RANGE_M * 720 / (2 * 647.0) - ext_v / 2
    assert all(s <= cap + 1e-9 for _, s, t in fam if t == 0.0)
```

- [ ] **Step 2: Run to verify failure** — `p inspection/tests/test_rays.py` → AttributeError on `cell_ray`.

- [ ] **Step 3: Implement (append to `rays.py`)**

```python
from inspection.view.grid import H_BINS, V_ELEVATIONS
from inspection.view.faceview import face_frame

CONE_STEP_DEG, CONE_MAX_DEG, CONE_AZIMUTHS = 5.0, 60.0, 12


def cell_ray(center, h, v):
    """Ray of viewsphere cell {h, v}: centroid anchor, grid direction —
    the same az0 convention as ViewSphere (h=0 faces the robot base)."""
    center = np.asarray(center, dtype=float)
    az0 = np.arctan2(-center[1], -center[0])
    az = az0 + h * (2 * np.pi / H_BINS)
    el = np.deg2rad(V_ELEVATIONS[v])
    return Ray(tuple(center),
               (np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                np.sin(el)))


def cell_clear_m(aabb, ray):
    """Anchor→surface run along the ray (slab method) — feeds sample_ray's
    clear_m so the sensor floor measures SURFACE distance, not anchor
    distance."""
    mn, mx = (np.asarray(v, dtype=float) for v in aabb)
    o, d = np.asarray(ray.anchor), np.asarray(ray.direction)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (mn - o) / d, (mx - o) / d
    t_exit = float(np.nanmin(np.maximum(t1, t2)))
    return max(t_exit, 0.0)


def face_family(prim, face, fy, height_px):
    """Ordered (ray, shift, theta_deg) candidates for one boxguard face:
    parallel +z shifts ascending (fronto-parallel — a plane parallel to the
    sensor projects as a pure similarity, zero keystone), then the classic
    cone. First family member with samples wins; no member -> face
    inaccessible, honestly."""
    center, normal, (ext_a, ext_b) = face_frame(prim, face)
    out = []
    if abs(normal[2]) < 0.5:                      # side faces shift up
        cap = MIN_RANGE_M * height_px / (2.0 * fy) - ext_b / 2.0
        s = 0.0
        while s <= cap + 1e-9:
            out.append((Ray(tuple(center + np.array([0, 0, s])),
                            tuple(normal)), float(s), 0.0))
            s += RAY_STEP_M
    else:                                         # top face: exact only
        out.append((Ray(tuple(center), tuple(normal)), 0.0, 0.0))
    a = np.array([0.0, 0.0, 1.0])
    if abs(normal @ a) > 0.95:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    th = CONE_STEP_DEG
    while th <= CONE_MAX_DEG:
        t = np.deg2rad(th)
        for k in range(CONE_AZIMUTHS):
            phi = 2 * np.pi * k / CONE_AZIMUTHS
            d = (np.cos(t) * normal
                 + np.sin(t) * (np.cos(phi) * u + np.sin(phi) * v))
            if d[2] < 0.05:               # never approach from under table
                continue
            out.append((Ray(tuple(center), tuple(d)), 0.0, float(th)))
        th += CONE_STEP_DEG
    return out
```

- [ ] **Step 4: Run tests to verify pass** — `p inspection/tests/test_rays.py` → `OK test_rays`.

- [ ] **Step 5: Commit**

```bash
git add inspection/view/rays.py inspection/tests/test_rays.py
git commit -m "rays: cell rays + face family — shifts before tilts, FOV-computed cap"
```

---

### Task 3: `resolve_ray_view` — one resolution call for any address

**Files:**
- Modify: `inspection/view/rays.py` (append)
- Test: `inspection/tests/test_rays.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2 (`sample_ray`, `cell_ray`, `cell_clear_m`, `face_family` — the AMENDED 4-tuple form `(ray, shift, theta_deg, d_min)`).
- Produces: `RayResolution = namedtuple("RayResolution", ["T_cam", "T_flange", "q", "depth", "shift", "theta_deg", "roll", "samples"])`; `resolve_cell(world, ik, t_flange_cam, center, aabb, h, v, q_near=None) -> RayResolution | None`; `resolve_face(world, ik, t_flange_cam, prim, face, fy, height_px, q_near=None) -> RayResolution | None`. `samples` carries the full valid list of the winning ray (Stage B's slide-out search needs it). Closest sample = element 0.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_resolve_cell_picks_closest_sample():
    from inspection.view.rays import (Ray, cell_ray, cell_clear_m,
                                      resolve_cell, sample_ray)
    w, ik = _box_world()
    aabb = ((-0.025, -0.455, 0.0), (0.125, -0.305, 0.075))
    res = resolve_cell(w, ik, T_FLANGE_CAM, (0.05, -0.38, 0.0375), aabb,
                       h=3, v=1)
    assert res is not None
    raw = sample_ray(w, ik, T_FLANGE_CAM,
                     cell_ray((0.05, -0.38, 0.0375), 3, 1),
                     clear_m=cell_clear_m(
                         aabb, cell_ray((0.05, -0.38, 0.0375), 3, 1)))
    assert np.isclose(res.depth, raw[0].depth)


def test_resolve_face_side_uses_shift_family():
    # AMENDED by ruling 2026-09-17: execution proved no fixture lands in
    # the "shift>0 wins" gap (tall box: exact clears; real-height box: the
    # whole shift family is blocked and the cone wins). The unit contract
    # is the FAMILY: theta==0 proves a fronto-parallel member won, and
    # Task 2's ordering test already pins shifts-before-tilts.
    from inspection.view.rays import MIN_RANGE_M, resolve_face
    w, ik = _box_world()
    p = _box_prim()
    res = resolve_face(w, ik, T_FLANGE_CAM, p, "y-", fy=647.0,
                       height_px=720)
    assert res is not None
    assert res.theta_deg == 0.0, "shift family must win before any tilt"
    assert res.shift >= 0.0
    assert res.depth >= MIN_RANGE_M - 1e-9


def test_resolve_face_buried_is_none():
    from inspection.view.rays import resolve_face
    w, ik = _box_world()
    p = _box_prim(center=(0.05, -0.38, -0.5))     # face underground
    assert resolve_face(w, ik, T_FLANGE_CAM, p, "y-", fy=647.0,
                        height_px=720) is None
```

- [ ] **Step 2: Run to verify failure** — AttributeError on `resolve_cell`.

- [ ] **Step 3: Implement (append)**

```python
RayResolution = namedtuple("RayResolution",
                           ["T_cam", "T_flange", "q", "depth", "shift",
                            "theta_deg", "roll", "samples"])


def _resolution(samples, shift, theta):
    s = samples[0]
    return RayResolution(s.T_cam, s.T_flange, s.q, s.depth, shift, theta,
                         s.roll, samples)


def resolve_cell(world, ik, t_flange_cam, center, aabb, h, v, q_near=None):
    """Closest safe sample on cell {h,v}'s ray, or None (inaccessible)."""
    ray = cell_ray(center, h, v)
    samples = sample_ray(world, ik, t_flange_cam, ray,
                         clear_m=cell_clear_m(aabb, ray), q_near=q_near)
    return _resolution(samples, 0.0, 0.0) if samples else None


def resolve_face(world, ik, t_flange_cam, prim, face, fy, height_px,
                 q_near=None):
    """Closest safe sample in the face's family, fronto-parallel first.
    AMENDED by bench ruling 2026-09-17 (Task 4 measured 12-20 cm of
    standoff lost on `x+` in all four runs): the SHIFT rung is scanned
    WHOLE and the member whose closest valid sample is shallowest wins
    (tie-break smaller shift) — it is no longer the first shift with any
    valid sample. Tilt stays strictly later: cone members are consulted
    only if no theta=0 member has a sample, first hit then wins.
    Each member's d_min (containment floor, per the earlier 2026-09-17
    ruling) reaches sample_ray through clear_m: floor = clear_m +
    MIN_RANGE_M."""
    family = face_family(prim, face, fy, height_px)
    best = None                                   # rung 1: whole shift rung
    for ray, shift, theta, d_min in family:
        if theta != 0.0:
            continue
        samples = _sample(ray, d_min)
        if samples and (best is None
                        or (samples[0].depth, shift) < best[0]):
            best = ((samples[0].depth, shift), samples, shift, theta)
    if best is not None:
        return _resolution(best[1], best[2], best[3])
    for ray, shift, theta, d_min in family:       # rung 2: cone, first hit
        if theta == 0.0:
            continue
        samples = _sample(ray, d_min)
        if samples:
            return _resolution(samples, shift, theta)
    return None
```

**Ordering note (bench amendment 2026-09-17, ruling recorded):** within the
shift rung the winner is the shallowest valid sample, tie-break smaller
shift; tilt is strictly last. This replaces "shift ascending before depth
(predictable ladder)" — the plan's original pick — because Task 4's bench
measured that order costing 12–20 cm on `x+` in every run while a 2–6 cm
shift held a θ=0, in-frame sample at 0.12–0.16 m. A shift trades away no
keystone, so closeness wins. Measured cost of scanning the rung whole:
52–67 ms per face family (vs 12–40 ms stopping early).

- [ ] **Step 4: Run tests to verify pass** — `p inspection/tests/test_rays.py` → `OK test_rays`.

- [ ] **Step 5: Commit**

```bash
git add inspection/view/rays.py inspection/tests/test_rays.py
git commit -m "rays: resolve_cell/resolve_face — one resolution call, honest None"
```

---

### Task 4: replay bench — the Stage A gate

**Files:**
- Create: `inspection/investigation/ray_views/__init__.py` (empty), `inspection/investigation/ray_views/ray_replay.py`
- No test file: this is an investigation CLI; its output IS the deliverable.

**Interfaces:**
- Consumes: Task 3 resolvers; `Run` from `inspection.record.run` (`Run(path)`, `.steps`, per-step `decomposition.json` via `step.decomposition`, `T_base_cam`, `view`); recorded runs `1709-cam-ai-demo1` plus the two most recent `*bench*` runs under `inspection/data/runs/` (list them, take what exists).
- Produces: `/tmp/ray_views/<run>/comparison.png` per run (3D matplotlib: object box, old executed camera positions, ray endpoints, cell ring), `/tmp/ray_views/<run>/timings.txt` (measured per-ray, per-refresh, per-family wall times vs the design table), and a printed per-view table: `address | old dist-to-surface | ray dist-to-surface | old theta | ray theta/shift`.

- [ ] **Step 1: Implement the CLI** (fire entrypoint per repo convention — guard-local `import fire`, Display override; copy the pattern from `inspection/investigation/run_replay/step_replay.py`). Reconstruct the world per run: `RobotCell()` + `set_object` from the LAST decomposition box (+ keepouts come free from cell.yaml). For every executed step: old endpoint = recorded `T_base_cam` position; new endpoint = `resolve_cell`/`resolve_face` output. Time `time.perf_counter()` around each resolve, and around a full 36-cell refresh; write both PNG and timings.

- [ ] **Step 2: Run over the runs and eyeball** — `p inspection/investigation/ray_views/ray_replay.py --run 1709-cam-ai-demo1` etc. Verify: face endpoints land ON or inside the ring, θ=0 with small shifts on side faces, no endpoint inside the padded box, timings within ~2× of the design table.

- [ ] **Step 3: Commit**

```bash
git add inspection/investigation/ray_views/
git commit -m "ray_views: replay bench — old vs ray endpoints, measured timings"
```

- [ ] **Step 4: GATE — present renders + timing table to Anton.** Stage B is planned only after his yes; if the renders don't visibly win, stop and say so.

---

## Self-review notes

- Spec coverage: sampler ✓ (T1), vocabulary ✓ (T2), resolution ✓ (T3), falsifier bench + timing measurement ✓ (T4). Transaction/legs, machine, menu, schema = Stages B/C by design, not gaps.
- Types consistent: `Ray`, `Sample`, `RayResolution` defined once, consumed by name everywhere.
- `_box_prim()` in tests is copied from `inspection/tests/test_faceview.py` — executors: copy the helper, don't import across test files.
