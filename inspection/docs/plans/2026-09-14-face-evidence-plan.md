# Face-Evidence Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Boxguard face normals become prioritized, capturable evidence views in the live loop, with a request-time resolution ladder (exact → z-lift → cone search → honest refusal).

**Architecture:** A new pure-geometry module `view/faceview.py` derives face targets and resolves accessibility; the Supervisor grows a stability gate fed by `_fit_step`, a `("face", name)` target kind through the existing request→plan→execute path, and face entries in the views payload; the orchestrator menu/prompt marks faces as soft-priority evidence.

**Tech Stack:** numpy, Pinocchio/coal (`RobotCell`), existing `UR5eIK` + `plan_viewpoint` ladder, pydantic record schema (no schema changes — `address: Any`, `ViewCandidate.scores` carry everything).

**Spec:** `inspection/docs/2026-09-14-face-evidence-design.md`

## Global Constraints

- Python: `~/miniconda3/envs/robo/bin/python` (the `robo` env), repo root `/home/anton/projects/robotics` on `sys.path` (run from repo root).
- Tests follow the repo pattern: plain functions + `main()` registry, runnable as a module; also add each new test to its file's `main()`.
- Face handles are exactly `"x+", "x-", "y+", "y-", "top"` (box-frame, yaw-stable). Bottom face excluded.
- Gate: opens after **2 consecutive** refit-to-refit comparisons with max face-normal delta ≤ `FACE_STABLE_TOL_DEG = 5.0`.
- Ladder defaults: z-lift step 0.02 m capped at 45° implied off-angle; cone rings θ = 5°..60° step 5°, 12 azimuths/ring; standoff `face_radius` clamped to `R_MIN` (done inside `radius_for_extent`).
- Menu priority language is SOFT: "evidence mainly comes from face views" — never "only".
- No UI panel work; the views payload just carries the faces block.

---

### Task 1: `view/faceview.py` — face frames, poses, framing radius, stability

**Files:**
- Create: `inspection/view/faceview.py`
- Test: `inspection/tests/test_faceview.py`

**Interfaces:**
- Consumes: `inspection.fit.contract.Primitive` (`.axes` 3×3 columns X/Y/Z, `.center`, `.params["extents"] = [l, w, h]`), `inspection.view.grid.radius_for_extent(extent, fy, height_px)`.
- Produces:
  - `FACES = ("x+", "x-", "y+", "y-", "top")`
  - `face_frame(prim, face) -> (center: np.ndarray(3), normal: np.ndarray(3), extent2d: tuple[float, float])`
  - `face_cam_pose(center, normal, r) -> np.ndarray(4,4)` (camera +X boresight = −normal, +Z image-up world-preferred — same convention as `ViewSphere.cam_pose`)
  - `face_radius(extent2d, fy, height_px) -> float`
  - `normals_delta_deg(prim_a, prim_b) -> float` (max angular change across FACES)

- [ ] **Step 1: Write the failing tests**

```python
#!/usr/bin/env python3
"""Face-evidence geometry: frames, poses, framing, stability delta.
Run: p inspection/tests/test_faceview.py"""
import numpy as np

from inspection.fit.contract import Primitive
from inspection.view.faceview import (FACES, face_frame, face_cam_pose,
                                      face_radius, normals_delta_deg)


def _box_prim(l=0.20, w=0.10, h=0.15, yaw=0.0, center=(0.1, -0.35, 0.075)):
    c, s = np.cos(yaw), np.sin(yaw)
    pose = [[c, -s, 0.0, center[0]], [s, c, 0.0, center[1]],
            [0.0, 0.0, 1.0, center[2]], [0.0, 0.0, 0.0, 1.0]]
    return Primitive(kind="box", pose=pose,
                     params={"extents": [l, w, h]}, label="box")


def test_face_frame_axis_aligned():
    p = _box_prim()
    c, n, ext = face_frame(p, "x+")
    assert np.allclose(n, [1, 0, 0])
    assert np.allclose(c, [0.1 + 0.10, -0.35, 0.075])   # center + l/2 along +x
    assert ext == (0.10, 0.15)                          # (w, h) of that wall
    c, n, ext = face_frame(p, "top")
    assert np.allclose(n, [0, 0, 1])
    assert np.allclose(c, [0.1, -0.35, 0.075 + 0.075])
    assert ext == (0.20, 0.10)                          # (l, w) of the lid


def test_face_frame_follows_yaw():
    p = _box_prim(yaw=np.pi / 2)
    _, n, _ = face_frame(p, "x+")
    assert np.allclose(n, [0, 1, 0], atol=1e-9)         # +x face rotated to +y


def test_face_cam_pose_is_orthogonal_lookat():
    c, n, _ = face_frame(_box_prim(), "x+")
    T = face_cam_pose(c, n, r=0.30)
    assert np.allclose(T[:3, 3], c + 0.30 * n)          # on the normal ray
    assert np.allclose(T[:3, 0], -n)                    # boresight back at face
    assert T[2, 2] > 0.5                                # image-up is up-ish


def test_face_radius_clamps_to_rmin():
    from inspection.view.grid import R_MIN
    assert face_radius((0.01, 0.01), fy=430.0, height_px=480) == R_MIN


def test_normals_delta_yaw():
    a, b = _box_prim(yaw=0.0), _box_prim(yaw=np.deg2rad(10))
    assert abs(normals_delta_deg(a, b) - 10.0) < 0.1
    assert normals_delta_deg(a, a) < 1e-9


def main():
    test_face_frame_axis_aligned()
    test_face_frame_follows_yaw()
    test_face_cam_pose_is_orthogonal_lookat()
    test_face_radius_clamps_to_rmin()
    test_normals_delta_yaw()
    print("OK test_faceview")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_faceview.py`
Expected: `ModuleNotFoundError: inspection.view.faceview`

- [ ] **Step 3: Implement `inspection/view/faceview.py` (geometry half)**

```python
"""Boxguard face-evidence views (design: docs/2026-09-14-face-evidence-design.md).

The boxguard cuboid already carries orthogonality: a face normal IS the
orthogonal view direction. This module turns a box Primitive into camera
targets — and (Task 2) resolves the closest ACCESSIBLE view when the exact
orthogonal pose cannot be reached.
"""
import numpy as np

from inspection.view.grid import radius_for_extent

#: Box-frame face handles, stable under yaw. Bottom is excluded — it faces
#: the table.
FACES = ("x+", "x-", "y+", "y-", "top")

#: Stability gate tolerance: max face-normal delta between consecutive
#: boxguard refits that still counts as "unchanged" (design §gate).
FACE_STABLE_TOL_DEG_DEFAULT = 5.0

#: (axis column, sign, indices of the two extents that span the face)
_FACE_DEF = {"x+": (0, +1.0, (1, 2)), "x-": (0, -1.0, (1, 2)),
             "y+": (1, +1.0, (0, 2)), "y-": (1, -1.0, (0, 2)),
             "top": (2, +1.0, (0, 1))}


def face_frame(prim, face):
    """Face center, outward unit normal, and (a, b) extents of that wall."""
    axis, sign, (i, j) = _FACE_DEF[face]
    ext = [float(v) for v in prim.params["extents"]]
    n = sign * prim.axes[:, axis]
    c = prim.center + n * (ext[axis] / 2.0)
    return c, n, (ext[i], ext[j])


def face_cam_pose(center, normal, r):
    """Camera 4x4 on the normal ray: +X boresight at the face, +Z image-up
    world-preferred — the same up-vector rule as ViewSphere.cam_pose."""
    normal = np.asarray(normal, dtype=float)
    pos = np.asarray(center, dtype=float) + r * normal
    x = -normal
    ref = np.array([0.0, 0.0, 1.0])
    if abs(x @ ref) > 0.95:                      # top face: up degenerate
        ref = np.array([1.0, 0.0, 0.0])
    y = np.cross(ref, x); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
    return T


def face_radius(extent2d, fy, height_px):
    """Per-face framing standoff; R_MIN clamp lives in radius_for_extent."""
    return radius_for_extent(max(extent2d), fy, height_px)


def normals_delta_deg(prim_a, prim_b):
    """Max angular change of any face normal between two box fits [deg]."""
    worst = 0.0
    for face in FACES:
        _, na, _ = face_frame(prim_a, face)
        _, nb, _ = face_frame(prim_b, face)
        dot = float(np.clip(na @ nb, -1.0, 1.0))
        worst = max(worst, float(np.degrees(np.arccos(dot))))
    return worst
```

- [ ] **Step 4: Run tests to verify pass**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_faceview.py`
Expected: `OK test_faceview`

- [ ] **Step 5: Commit**

```bash
git add inspection/view/faceview.py inspection/tests/test_faceview.py
git commit -m "faceview: face frames, orthogonal cam poses, framing radius, stability delta"
```

---

### Task 2: resolution ladder — exact → z-lift → cone → None

**Files:**
- Modify: `inspection/view/faceview.py` (append)
- Test: `inspection/tests/test_faceview.py` (append)

**Interfaces:**
- Consumes: `RobotCell.is_colliding(q)`, `UR5eIK.branches(T_flange, q_near=None)`, Task 1 functions. `t_flange_cam`: the 4×4 the sphere uses (`ViewSphere` takes `t_flange_cam=T_FLANGE_CAM` from `inspection.view.grid`); flange pose = `T_cam @ inv(t_flange_cam)`.
- Produces:
  - `FaceResolution` namedtuple `(T_cam, T_flange, theta_deg, rung)` with rung in `{"exact", "zlift", "cone"}`
  - `resolve_face_view(world, ik, t_flange_cam, prim, face, r, q_near=None) -> FaceResolution | None`

- [ ] **Step 1: Write the failing tests (append to `test_faceview.py`; add to `main()`)**

```python
def test_resolve_exact_when_clear():
    from inspection.cell.world import RobotCell
    from inspection.motion.ik import UR5eIK
    from inspection.view.grid import T_FLANGE_CAM
    from inspection.view.faceview import resolve_face_view
    w, ik = RobotCell(), UR5eIK()
    p = _box_prim(center=(0.10, -0.35, 0.075))
    w.set_object("object", [0.20, 0.10, 0.15], [0.10, -0.35, 0.075, 0, 0, 0])
    res = resolve_face_view(w, ik, T_FLANGE_CAM, p, "x+", r=0.30)
    assert res is not None and res.rung == "exact" and res.theta_deg == 0.0


def test_resolve_zlift_for_slim_object():
    from inspection.cell.world import RobotCell
    from inspection.motion.ik import UR5eIK
    from inspection.view.grid import T_FLANGE_CAM
    from inspection.view.faceview import resolve_face_view
    w, ik = RobotCell(), UR5eIK()
    # 2 cm tall: the side face's orthogonal camera sits AT the table.
    p = _box_prim(l=0.15, w=0.10, h=0.02, center=(0.10, -0.35, 0.01))
    w.set_object("object", [0.15, 0.10, 0.02], [0.10, -0.35, 0.01, 0, 0, 0])
    res = resolve_face_view(w, ik, T_FLANGE_CAM, p, "x+", r=0.30)
    assert res is not None and res.rung == "zlift"
    assert 0.0 < res.theta_deg <= 45.0
    assert res.T_cam[2, 3] > 0.03            # the camera actually rose


def test_resolve_none_for_buried_face():
    from inspection.cell.world import RobotCell
    from inspection.motion.ik import UR5eIK
    from inspection.view.grid import T_FLANGE_CAM
    from inspection.view.faceview import resolve_face_view
    w, ik = RobotCell(), UR5eIK()
    p = _box_prim()
    # A wall of obstacle straight through every approach to the x+ face.
    w.set_object("object", [1.6, 1.6, 1.2], [0.30, -0.35, 0.60, 0, 0, 0])
    assert resolve_face_view(w, ik, T_FLANGE_CAM, p, "x+", r=0.30) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_faceview.py`
Expected: `ImportError: cannot import name 'resolve_face_view'`

- [ ] **Step 3: Implement the ladder (append to `faceview.py`)**

```python
from collections import namedtuple

#: Resolution ladder defaults (design §Request-time resolution ladder).
ZLIFT_STEP_M = 0.02          # rung 2 raise increment
ZLIFT_MAX_DEG = 45.0         # rung 2 hands over to the cone past this
CONE_STEP_DEG = 5.0          # rung 3 ring spacing
CONE_MAX_DEG = 60.0          # classic measurability cutoff
CONE_AZIMUTHS = 12

FaceResolution = namedtuple("FaceResolution",
                            ["T_cam", "T_flange", "theta_deg", "rung"])


def _endpoint_ok(world, ik, T_cam, t_fc_inv, q_near):
    T_fl = T_cam @ t_fc_inv
    free = [q for q in ik.branches(T_fl, q_near=q_near)
            if not world.is_colliding(q)]
    return (T_fl, free[0]) if free else None


def resolve_face_view(world, ik, t_flange_cam, prim, face, r, q_near=None):
    """Closest accessible view of `face`, most-orthogonal-first.

    Rungs (each the cheapest fix for its own failure mode):
      exact  — the orthogonal pose itself.
      zlift  — slim/short objects put a side camera at table height; rise
               along +Z looking at the face center until valid or 45° off.
      cone   — tilt rings around the normal, 5°..60°, 12 azimuths; the
               first valid endpoint is the most orthogonal accessible.
    Returns None when nothing within 60° has a collision-free IK branch —
    the face is honestly unreachable.
    """
    t_fc_inv = np.linalg.inv(np.asarray(t_flange_cam, dtype=float))
    center, normal, _ = face_frame(prim, face)

    T = face_cam_pose(center, normal, r)                       # rung 1
    hit = _endpoint_ok(world, ik, T, t_fc_inv, q_near)
    if hit:
        return FaceResolution(T, hit[0], 0.0, "exact")

    if abs(normal[2]) < 0.5:                                   # rung 2
        base = center + r * normal
        z_cap = r * np.sin(np.deg2rad(ZLIFT_MAX_DEG))
        dz = ZLIFT_STEP_M
        while dz <= z_cap:
            pos = base + np.array([0.0, 0.0, dz])
            d = pos - center
            theta = float(np.degrees(np.arccos(
                np.clip((d / np.linalg.norm(d)) @ normal, -1.0, 1.0))))
            T = face_cam_pose(center, d / np.linalg.norm(d),
                              float(np.linalg.norm(d)))
            hit = _endpoint_ok(world, ik, T, t_fc_inv, q_near)
            if hit:
                return FaceResolution(T, hit[0], theta, "zlift")
            dz += ZLIFT_STEP_M

    a = np.array([0.0, 0.0, 1.0])                              # rung 3
    if abs(normal @ a) > 0.95:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    theta = CONE_STEP_DEG
    while theta <= CONE_MAX_DEG:
        th = np.deg2rad(theta)
        for k in range(CONE_AZIMUTHS):
            phi = 2 * np.pi * k / CONE_AZIMUTHS
            d = (np.cos(th) * normal
                 + np.sin(th) * (np.cos(phi) * u + np.sin(phi) * v))
            if d[2] < 0.05:               # never approach from under the table
                continue
            T = face_cam_pose(center, d, r)
            hit = _endpoint_ok(world, ik, T, t_fc_inv, q_near)
            if hit:
                return FaceResolution(T, hit[0], float(theta), "cone")
        theta += CONE_STEP_DEG
    return None
```

Note the `d[2] < 0.05` guard applies to cone directions on SIDE faces too — a
tilted side approach may dip below the table plane; skipping those keeps every
candidate physical. `theta_deg` for zlift is the implied off-angle, computed
from the actual (lifted) view direction vs the normal.

- [ ] **Step 4: Run tests to verify pass**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_faceview.py`
Expected: `OK test_faceview` (3 new tests may take ~seconds each — IK + coal).

- [ ] **Step 5: Commit**

```bash
git add inspection/view/faceview.py inspection/tests/test_faceview.py
git commit -m "faceview: resolution ladder — exact, z-lift for slim objects, cone search"
```

---

### Task 3: offline replay — gate trace + ladder outcomes over recorded runs

**Files:**
- Create: `inspection/investigation/face_replay.py`

**Interfaces:**
- Consumes: recorded runs under `inspection/data/runs/<id>/steps/NNN/fused/cloud.npy`, `inspection.fit.box.boxguard_fit`, Task 1–2 functions, `RobotCell`, `UR5eIK`, `T_FLANGE_CAM`, `radius_for_extent` args from run intrinsics (fall back to fy=430.0, height_px=480 with a LOUD stamp — see `fallbacks-scream-on-real-inputs`).
- Produces: a CLI report (stdout table) — per step: normals delta vs previous fit, gate state, and per face the ladder rung + achieved θ. No record writes.

- [ ] **Step 1: Implement the replay CLI**

```python
#!/usr/bin/env python3
"""Replay boxguard face-evidence derivation over a recorded run.

Answers, offline and before any live wiring: when does the stability gate
open, and what does the resolution ladder achieve per face?

Usage (repo root, robo env):
    p inspection/investigation/face_replay.py run 1409-lan-ai_bench1
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inspection.cell.world import RobotCell
from inspection.fit.box import boxguard_fit
from inspection.motion.ik import UR5eIK
from inspection.view.faceview import (FACES, FACE_STABLE_TOL_DEG_DEFAULT,
                                      face_frame, face_radius,
                                      normals_delta_deg, resolve_face_view)
from inspection.view.grid import T_FLANGE_CAM

RUNS = Path(__file__).resolve().parents[1] / "data" / "runs"
FY_FALLBACK, HPX_FALLBACK = 430.0, 480    # D405-ish; stamped when used


def run(run_id: str):
    rdir = RUNS / run_id
    steps = sorted((rdir / "steps").iterdir())
    world, ik = RobotCell(), UR5eIK()
    print(f"== {run_id}: {len(steps)} steps  "
          f"(intrinsics fallback fy={FY_FALLBACK} h={HPX_FALLBACK})")
    prev, stable = None, 0
    for sdir in steps:
        cloud = np.load(sdir / "fused" / "cloud.npy")
        d = boxguard_fit(cloud)
        prim = d.primitives[0]
        delta = None if prev is None else normals_delta_deg(prev, prim)
        stable = (stable + 1 if delta is not None
                  and delta <= FACE_STABLE_TOL_DEG_DEFAULT else 0)
        gate = "OPEN" if stable >= 2 else "closed"
        print(f"step {sdir.name}: delta="
              f"{'-' if delta is None else f'{delta:5.1f}deg'} gate={gate}")
        if gate != "OPEN":
            prev = prim
            continue
        yaw = float(np.arctan2(prim.axes[1, 0], prim.axes[0, 0]))
        world.set_object("object", [float(v) for v in prim.params["extents"]],
                         list(map(float, prim.center)) + [0.0, 0.0, yaw])
        for face in FACES:
            _, _, ext = face_frame(prim, face)
            r = face_radius(ext, FY_FALLBACK, HPX_FALLBACK)
            res = resolve_face_view(world, ik, T_FLANGE_CAM, prim, face, r)
            if res is None:
                print(f"    {face:3s}: UNREACHABLE (r={r:.2f})")
            else:
                print(f"    {face:3s}: {res.rung:5s} theta={res.theta_deg:4.1f} "
                      f"r={r:.2f}")
        prev = prim


def main():
    import fire
    from fire import core
    core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire({"run": run})


if __name__ == "__main__":
    main()
```

`FACE_STABLE_TOL_DEG_DEFAULT` comes from Task 1's module — the one shared
default the machine (Task 4) and this replay both read.

- [ ] **Step 2: Run over the two bench runs and eyeball**

Run: `~/miniconda3/envs/robo/bin/python inspection/investigation/face_replay.py run 1409-lan-ai_bench1`
Run: `~/miniconda3/envs/robo/bin/python inspection/investigation/face_replay.py run 1409-box-ai_bench1`
Expected: gate trace prints; on the box run the gate opens by step 2–3; faces
report `exact`/`zlift`/`cone` with θ — the top face should resolve (`exact`
or low-θ cone), side faces mostly `exact`. Paste the output into the task
report for Anton's eyeball.

- [ ] **Step 3: Commit**

```bash
git add inspection/investigation/face_replay.py inspection/view/faceview.py
git commit -m "face_replay: offline gate + ladder trace over recorded runs"
```

---

### Task 4: Supervisor — stability gate state fed by `_fit_step`

**Files:**
- Modify: `inspection/run/machine.py` (init block ~line 128; `_fit_step` ~line 896)
- Test: `inspection/tests/test_machine.py` (append; register in `main()`)

**Interfaces:**
- Consumes: `normals_delta_deg`, `FACE_STABLE_TOL_DEG_DEFAULT` from `inspection.view.faceview`; the fitted `Decomposition d` already in `_fit_step`.
- Produces (later tasks rely on): `self._face_prim` (freshest stable-or-not box `Primitive | None`), `self._face_stable: int`, `self._face_theta: dict[str, float]`, `self.face_visited: set[str]`, and `Supervisor._face_states() -> dict[str, dict]` mapping face → `{"status": "settling"|"ready"|"visited"|"unreachable", "off_angle_deg": float|None}`.

- [ ] **Step 1: Write the failing test (append to `test_machine.py`)**

```python
def test_face_gate_opens_after_two_stable_fits():
    import numpy as np
    from inspection.fit.contract import Decomposition, Primitive

    def prim(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        pose = [[c, -s, 0, 0.1], [s, c, 0, -0.35], [0, 0, 1, 0.07],
                [0, 0, 0, 1]]
        return Primitive(kind="box", pose=pose,
                         params={"extents": [0.2, 0.1, 0.15]}, label="box")

    sup, rig, bus, th = make_sup()
    try:
        # Feed the gate directly — the unit under test is the counter.
        sup._update_face_gate(prim(0.0))
        assert sup._face_states()["x+"]["status"] == "settling"
        sup._update_face_gate(prim(np.deg2rad(1.0)))     # small delta #1
        assert sup._face_states()["x+"]["status"] == "settling"
        sup._update_face_gate(prim(np.deg2rad(2.0)))     # small delta #2
        assert sup._face_states()["x+"]["status"] == "ready"
        sup._update_face_gate(prim(np.deg2rad(40.0)))    # box jumped
        assert sup._face_states()["x+"]["status"] == "settling"
        sup.face_visited.add("top")
        sup._update_face_gate(prim(np.deg2rad(40.5)))
        sup._update_face_gate(prim(np.deg2rad(41.0)))
        st = sup._face_states()
        assert st["top"]["status"] == "visited"
        assert st["x+"]["status"] == "ready"
    finally:
        sup.request_shutdown(); th.join(10)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/robo/bin/python -c "from inspection.tests.test_machine import test_face_gate_opens_after_two_stable_fits as t; t()"`
Expected: `AttributeError: 'Supervisor' object has no attribute '_update_face_gate'`

- [ ] **Step 3: Implement gate state in `machine.py`**

In `__init__` (next to `self._pending_request = None` block):

```python
        self._face_prim = None      # freshest boxguard Primitive (or None)
        self._face_stable = 0       # consecutive small normal deltas (gate: >=2)
        self._face_theta = {}       # face -> achieved off-angle from resolution
        self._face_unreachable = set()  # faces whose ladder returned None
        self.face_visited = set()   # face handles captured this run
```

New methods (place next to `_fit_step`):

```python
    def _update_face_gate(self, prim):
        """Feed one boxguard fit to the stability gate (design §gate).

        Called from `_fit_step` on the dispatcher for every non-abstain
        boxguard fit. Two consecutive refit deltas within tolerance open
        the gate; any jump closes it and voids cached resolutions — the
        box moved, so every achieved theta is stale.
        """
        from inspection.view.faceview import (FACE_STABLE_TOL_DEG_DEFAULT,
                                              normals_delta_deg)
        if (self._face_prim is not None
                and normals_delta_deg(self._face_prim, prim)
                <= FACE_STABLE_TOL_DEG_DEFAULT):
            self._face_stable = min(self._face_stable + 1, 2)
        else:
            self._face_stable = 0
            self._face_theta.clear()
            self._face_unreachable.clear()
        self._face_prim = prim

    def _face_states(self):
        """Face menu block: {face: {status, off_angle_deg}} (design §menu)."""
        from inspection.view.faceview import FACES
        out = {}
        gate_open = self._face_stable >= 2
        for f in FACES:
            if f in self.face_visited:
                status = "visited"
            elif not gate_open:
                status = "settling"
            elif f in self._face_unreachable:
                status = "unreachable"
            else:
                status = "ready"
            out[f] = {"status": status,
                      "off_angle_deg": self._face_theta.get(f)}
        return out
```

In `_fit_step`, after the `rec = record_of(...)` line and before the write,
feed the gate (non-abstain boxguard only):

```python
        if not d.abstain and d.method == "boxguard" and d.primitives:
            self._update_face_gate(d.primitives[0])
```

- [ ] **Step 4: Run the new test, then the full machine suite**

Run: `~/miniconda3/envs/robo/bin/python -c "from inspection.tests.test_machine import test_face_gate_opens_after_two_stable_fits as t; t(); print('PASS')"`
Expected: `PASS`
Run: `timeout 590 ~/miniconda3/envs/robo/bin/python inspection/tests/test_machine.py`
Expected: `OK test_machine (task 5)`

- [ ] **Step 5: Commit**

```bash
git add inspection/run/machine.py inspection/tests/test_machine.py
git commit -m "machine: face stability gate — 2 consecutive stable boxguard fits"
```

---

### Task 5: Supervisor — `("face", name)` target through request→plan→capture

**Files:**
- Modify: `inspection/run/machine.py` (`_normalize_target` ~line 1113; `_on_view_request` ~line 270; `_plan_worker` ~line 519; `_exec_worker`'s `begin_step` call ~line 655; `_on_settle_done` visited marking ~line 465; `_view_state` ~line 976)
- Modify: `inspection/run/app.py` (add `face_method()` next to `viewsphere_method()` ~line 158; include it in `view_methods=[...]` ~line 278)
- Modify: `inspection/tests/test_machine.py` `make_writer` (add the face method to `view_methods`)
- Test: `inspection/tests/test_machine.py` (append; register in `main()`)

**Interfaces:**
- Consumes: Task 2 `resolve_face_view(...) -> FaceResolution | None`, Task 4 state, `plan_viewpoint(world, ik, q_from, T_flange, seed)` (already imported), `self.rig.intr["fy"]`, `self.rig.intr["height"]`, `T_FLANGE_CAM` via `self.sphere._t_fc_inv` is PRIVATE — import `T_FLANGE_CAM` from `inspection.view.grid` instead.
- Produces: face targets flow end to end; steps record `{"method": "face", "address": [name]}`; achieved θ lands in `self._face_theta[name]` and in `ViewCandidate.scores` (Task 6's `_view_state` face entries).

- [ ] **Step 1: Write the failing test (append to `test_machine.py`)**

```python
def test_face_request_full_cycle():
    import numpy as np
    from inspection.fit.contract import Primitive

    sup, rig, bus, th = make_sup(q_start=DEMO_PARK.copy() + np.radians(
        [0, 0, 0, 0, 0, 8]))
    try:
        full_boot(sup)
        pose = [[1, 0, 0, 0.10], [0, 1, 0, -0.35], [0, 0, 1, 0.05],
                [0, 0, 0, 1]]
        prim = Primitive(kind="box", pose=pose,
                         params={"extents": [0.12, 0.08, 0.10]}, label="box")
        for _ in range(3):                      # open the gate
            sup._update_face_gate(prim)
        sup.events.put({"cmd": "view/request", "target": ["face", "top"]})
        wait_for(lambda: sup.phase == "previewing", timeout=60,
                 msg="face preview")
        assert sup.target == ("face", "top")
        sup.events.put({"cmd": "view/confirm", "target": ["face", "top"]})
        wait_for(lambda: "top" in sup.face_visited, timeout=60,
                 msg="face visited")
        assert sup._face_theta.get("top") is not None
        # The record carries the face identity.
        import json as _json
        from pathlib import Path as _P
        run_dir = _P(sup.writer.run_dir)
        steps = sorted((run_dir / "steps").iterdir())
        step = _json.load(open(steps[-1] / "step.json"))
        assert step["view"]["method"] == "face"
        assert step["view"]["address"] == ["top"]
    finally:
        sup.request_shutdown(); th.join(10)
```

Note: if `RunWriter` exposes the run directory under a different attribute
than `writer.run_dir`, read `record/writer.py` and use its actual name;
`make_writer` builds the writer, so the attribute exists either way.

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/robo/bin/python -c "from inspection.tests.test_machine import test_face_request_full_cycle as t; t(); print('PASS')"`
Expected: request dropped (`_normalize_target` returns None for lists of
strings) → timeout on "face preview".

- [ ] **Step 3: Implement the plumbing**

`run/app.py` — next to `viewsphere_method()`:

```python
def face_method() -> dict:
    """Boxguard face-evidence views (docs/2026-09-14-face-evidence-design.md)."""
    from inspection.view.faceview import (CONE_MAX_DEG,
                                          FACE_STABLE_TOL_DEG_DEFAULT)
    return {"id": "face", "kind": "boxguard-face", "version": "1",
            "params": {"stable_tol_deg": FACE_STABLE_TOL_DEG_DEFAULT,
                       "cone_max_deg": CONE_MAX_DEG}}
```

and register it wherever `view_methods=[viewsphere_method(r)]` is passed:
`view_methods=[viewsphere_method(r), face_method()]`. Mirror the same
addition in `tests/test_machine.py:make_writer`.

`machine.py::_normalize_target` — accept the face forms before the cell
regex (list `["face", "x+"]` from the bus, string `"face:x+"` from markers):

```python
        from inspection.view.faceview import FACES
        if (isinstance(raw, (list, tuple)) and len(raw) == 2
                and raw[0] == "face" and raw[1] in FACES):
            return ("face", raw[1])
        if isinstance(raw, str) and raw.startswith("face:") \
                and raw[5:] in FACES:
            return ("face", raw[5:])
```

`machine.py::_on_view_request` — face gate branch, INSTEAD of the
`self._reach` check when the target is a face (insert before the existing
`else:` cell checks):

```python
        elif isinstance(target, tuple) and target[0] == "face":
            st = self._face_states()[target[1]]["status"]
            if st != "ready":
                log.info("view/request face %s dropped: %s", target[1], st)
                self.pub.log("warn", f"face {target[1]} dropped: {st}")
                return
```

`machine.py::_plan_worker` — face branch next to the survey/cell branches:

```python
            elif isinstance(target, tuple) and target[0] == "face":
                from inspection.view.faceview import (face_frame, face_radius,
                                                      resolve_face_view)
                from inspection.view.grid import T_FLANGE_CAM
                face, prim = target[1], self._face_prim
                _, _, ext = face_frame(prim, face)
                r_f = face_radius(ext, self.rig.intr["fy"],
                                  self.rig.intr["height"])
                res = resolve_face_view(self.world, self.ik, T_FLANGE_CAM,
                                        prim, face, r_f, q_near=q_now)
                if res is None:
                    self._face_unreachable.add(face)
                    path, detail = None, "face unreachable within 60deg"
                else:
                    self._face_theta[face] = res.theta_deg
                    path, rep = plan_viewpoint(self.world, self.ik, q_now,
                                               res.T_flange, seed=self.seed)
                    detail = (f"tier {rep['tier']} "
                              f"{res.rung} {res.theta_deg:.0f}deg"
                              if path else "plan refused")
```

`_exec_worker`'s `begin_step` — the address dict is built from `target`;
extend the expression:

```python
                {"method": "face" if isinstance(target, tuple)
                                     and target[0] == "face" else VIEW_METHOD,
                 "address": ([target[1]] if isinstance(target, tuple)
                             and target[0] == "face"
                             else None if target == "survey" else list(target))}
```

`_on_settle_done` — route face targets to `face_visited` (replace the
`self.visited.add(target)` line):

```python
                if isinstance(target, tuple) and target[0] == "face":
                    self.face_visited.add(target[1])
                else:
                    self.visited.add(target)
```

(the neighboring `self.captures[target] = ...` and `self.current = target`
lines stay as they are — a face tuple is a fine dict key, and `_view_state`'s
`cell == self.current` comparisons simply never match it.)

- [ ] **Step 4: Run the new test, then the full machine suite**

Run: `~/miniconda3/envs/robo/bin/python -c "from inspection.tests.test_machine import test_face_request_full_cycle as t; t(); print('PASS')"`
Expected: `PASS`
Run: `timeout 590 ~/miniconda3/envs/robo/bin/python inspection/tests/test_machine.py`
Expected: `OK test_machine (task 5)`
Also run the record suite (schema/writer touched via new method id):
`~/miniconda3/envs/robo/bin/python inspection/tests/test_record_validate.py`
Expected: its OK line.

- [ ] **Step 5: Commit**

```bash
git add inspection/run/machine.py inspection/run/app.py inspection/tests/test_machine.py
git commit -m "machine: ('face', name) target kind — gate-checked, ladder-resolved, recorded as method=face"
```

---

### Task 6: faces in the views payload and the step's ViewState

**Files:**
- Modify: `inspection/run/machine.py` (`_view_state` ~line 976; `_publish_views` call site — find `def _publish_views` and the `publish_views(` call in it)
- Modify: `inspection/ui/publisher.py` (`publish_views` ~line 431 — add optional `faces=None` parameter, emit `payload["faces"] = faces or {}`)
- Test: `inspection/tests/test_machine.py` (append; register in `main()`)

**Interfaces:**
- Consumes: Task 4 `_face_states()`, Task 1 `face_frame`/`face_cam_pose`/`face_radius`.
- Produces: `views/state` bus payload gains `"faces": {name: {"status": ..., "off_angle_deg": ...}}`; `ViewState.candidates` gains one `ViewCandidate` per READY/VISITED face with `address=["face", name]` and `scores={"off_angle_deg": θ}` when known.

- [ ] **Step 1: Write the failing test (append to `test_machine.py`)**

```python
def test_views_payload_carries_faces():
    import numpy as np
    from inspection.fit.contract import Primitive

    sup, rig, bus, th = make_sup()
    try:
        full_boot(sup)
        views = bus.last("views/state")
        assert "faces" in views
        assert views["faces"]["x+"]["status"] == "settling"
        pose = [[1, 0, 0, 0.1], [0, 1, 0, -0.35], [0, 0, 1, 0.05],
                [0, 0, 0, 1]]
        prim = Primitive(kind="box", pose=pose,
                         params={"extents": [0.1, 0.1, 0.1]}, label="box")
        for _ in range(3):
            sup._update_face_gate(prim)
        sup._publish_views()
        assert bus.last("views/state")["faces"]["x+"]["status"] == "ready"
    finally:
        sup.request_shutdown(); th.join(10)
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/robo/bin/python -c "from inspection.tests.test_machine import test_views_payload_carries_faces as t; t(); print('PASS')"`
Expected: `KeyError/AssertionError: 'faces' not in views payload`

- [ ] **Step 3: Implement**

- `ui/publisher.py::publish_views`: add keyword `faces=None`, include
  `"faces": faces or {}` in the published dict (read the function first and
  put the key beside its `cells` sibling).
- `machine.py::_publish_views`: pass `faces=self._face_states()`.
- `machine.py::_view_state`: after the sphere-cells loop, append face
  candidates so the RECORD sees the same menu the agent did:

```python
        if self._face_prim is not None:
            from inspection.view.faceview import (face_cam_pose, face_frame,
                                                  face_radius)
            for f, st in self._face_states().items():
                if st["status"] == "settling":
                    continue
                c, n, ext = face_frame(self._face_prim, f)
                r_f = face_radius(ext, self.rig.intr["fy"],
                                  self.rig.intr["height"])
                theta = st["off_angle_deg"]
                cands.append(ViewCandidate(
                    address=["face", f], status=st["status"],
                    pose=_mat(face_cam_pose(c, n, r_f)),
                    scores={} if theta is None
                    else {"off_angle_deg": float(theta)}))
```

(if `ViewCandidate` rejects a `scores` kwarg, read its definition at
`record/schema.py:269` and use the field it actually has for per-candidate
numbers; the earlier recorded runs show `"scores": {}` in
`view_state.json`, so the field exists.)

- [ ] **Step 4: Run the new test, then the machine suite**

Run: `~/miniconda3/envs/robo/bin/python -c "from inspection.tests.test_machine import test_views_payload_carries_faces as t; t(); print('PASS')"`
Expected: `PASS`
Run: `timeout 590 ~/miniconda3/envs/robo/bin/python inspection/tests/test_machine.py`
Expected: `OK test_machine (task 5)`

- [ ] **Step 5: Commit**

```bash
git add inspection/run/machine.py inspection/ui/publisher.py inspection/tests/test_machine.py
git commit -m "views: faces block in payload and step ViewState (address + off-angle scores)"
```

---

### Task 7: orchestrator — faces in nav() and the soft-priority prompt

**Files:**
- Modify: `inspection/brain/live.py` (`nav()` ~line 108)
- Modify: `inspection/brain/loop.py` (the system-prompt block that contains "Addresses are [azimuth, elevation]"; the MOVES/STATE menu rendering that reads `_nav()`)
- Test: `inspection/tests/test_live_faces.py` (create)

**Interfaces:**
- Consumes: `sup._face_states()` (Task 4), `mover.request(("face", "x+"))` — `SupervisorMover.request` already tuples list input (`cell = ... tuple(cell)`), and `_normalize_target` (Task 5) accepts it on the machine side.
- Produces: `nav()["faces"] = {name: {"status": ..., "off_angle_deg": ...}}`; prompt text telling the agent faces are the preferred evidence source.

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""Faces reach the orchestrator: nav block + prompt language.
Run: p inspection/tests/test_live_faces.py"""


class _SupStub:
    _reach = {(0, 0): 0.0}
    visited = {(0, 0)}
    blocked = set()
    phase = "idle"
    target = None

    @staticmethod
    def _face_states():
        return {"x+": {"status": "ready", "off_angle_deg": None},
                "top": {"status": "visited", "off_angle_deg": 12.0}}


def test_nav_carries_faces():
    from inspection.brain.live import SupervisorMover
    mover = SupervisorMover.__new__(SupervisorMover)   # no bus, no run dir
    mover.sup = _SupStub()
    nav = mover.nav()
    assert nav["faces"]["x+"]["status"] == "ready"
    assert nav["faces"]["top"]["off_angle_deg"] == 12.0


def test_prompt_mentions_face_priority():
    import inspection.brain.loop as loop
    import inspect
    src = inspect.getsource(loop)
    assert "face views" in src and "mainly" in src, \
        "system prompt must carry the SOFT face-evidence priority"


def main():
    test_nav_carries_faces()
    test_prompt_mentions_face_priority()
    print("OK test_live_faces")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify failure**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_live_faces.py`
Expected: `KeyError: 'faces'`

- [ ] **Step 3: Implement**

`brain/live.py::nav()` — add to the returned dict (keep the docstring's
list of "the SAME three sets" honest by updating it to four):

```python
            "faces": self.sup._face_states(),
```

`brain/loop.py` — two insertions:

1. In the system-prompt block that explains addresses (it contains the
   sentence "Addresses are [azimuth, elevation]."), append this paragraph,
   verbatim:

```
FACE VIEWS. When the object's box has settled, the menu offers face views —
addresses like ["face", "x+"] (sides) and ["face", "top"] — each one the
head-on, orthogonal look at that wall of the object. Evidence mainly comes
from face views: prefer them when they serve the question, and fall back to
numbered viewpoints for coverage, exploration, or anything the faces do not
show. A face marked `settling` is not ready yet; `degraded (N deg)` means
the closest reachable look is N degrees off head-on — still usually the
best evidence available. Move to a face by requesting its address, exactly
like a cell.
```

2. Where the MOVES/STATE menu text is rendered from `_nav()` (the code that
   builds the block listing azimuths — find it by searching `loop.py` for
   `"MOVES"`), append a FACES line per face from `nav["faces"]`, e.g.:

```python
        faces = (nav or {}).get("faces") or {}
        if faces:
            rows = []
            for name, st in faces.items():
                extra = ("" if st["off_angle_deg"] is None
                         else f" ({st['off_angle_deg']:.0f} deg off)")
                rows.append(f'  ["face", "{name}"] — {st["status"]}{extra}')
            menu += "\nFACES (orthogonal evidence):\n" + "\n".join(rows)
```

Adapt the variable the menu accumulates into (`menu` here) to the actual
local name at that site — read the surrounding function first; the row
format is the contract, the variable name is not.

- [ ] **Step 4: Run the tests**

Run: `~/miniconda3/envs/robo/bin/python inspection/tests/test_live_faces.py`
Expected: `OK test_live_faces`
Also: `~/miniconda3/envs/robo/bin/python inspection/tests/test_vlm_agent.py`
and `~/miniconda3/envs/robo/bin/python inspection/tests/test_decider.py`
Expected: their OK lines (prompt text changed; these guard the loop).

- [ ] **Step 5: Commit**

```bash
git add inspection/brain/live.py inspection/brain/loop.py inspection/tests/test_live_faces.py
git commit -m "brain: faces in nav + soft-priority face-evidence prompt"
```

---

### Task 8: bench validation — the demo-v2 criterion

**Files:** none created; this is the live acceptance pass.

- [ ] **Step 1: Bench run one** — coded box on the table, real rig:
`~/miniconda3/envs/robo/bin/python -m inspection.run.app` with the usual bench
flags (mirror whatever `1409-box-ai_bench1` was launched with — check
`inspection/data/runs/1409-box-ai_bench1/config.json` and `session.json`
for the exact invocation), question: "what is this and what does it say?".
Expected: gate opens after 2–3 captures; agent requests a face view; the
cited evidence step has `view.method == "face"`.

- [ ] **Step 2: Bench run two** — move AND rotate the box ~30–45°, rerun.
Expected: face views re-derive from the fresh boxguard; evidence is again
the orthogonal shot of the coded side. This is the demo-v2 success
criterion from the spec.

- [ ] **Step 3: Report** — paste both runs' `events.jsonl` tails, the face
menu lines from the trace, and the evidence images for Anton's review. No
commit; findings drive the next iteration.

---

## Self-review notes

- **Spec coverage:** foundation → T1/T2; gate → T4; merged menu + soft
  priority → T6/T7; ladder → T2 (endpoint) + T5 (plan-level, records θ);
  honest refusal → T5 (`_face_unreachable` → menu `unreachable`); offline
  first → T3; machine tests → T4/T5/T6; bench → T8. R_MIN clamp → T1
  (`face_radius` via `radius_for_extent`). Achieved-θ recording → T5
  (state) + T6 (ViewCandidate.scores).
- **Deviation, deliberate:** face status is resolved lazily — `ready`
  until the first request discovers θ or unreachability; the spec's
  `orthogonal/degraded` distinction appears after resolution. Rationale:
  running the ladder for 5 faces on every publish costs seconds of IK in
  the dispatcher's hot path. Flagged for Anton in the T8 report.
- **Known seam:** `_plan_worker` face branch reads `self._face_prim`
  from the worker thread — same benign-race pattern as `_reach` (write on
  dispatcher, read on worker, tuple/dataclass ref swap is atomic in
  CPython).
