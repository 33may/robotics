# Primitive deconstruction — contract, bank, storage

**Date:** 2026-09-03 · **Status:** approved by Anton in chat · **Task:** MAY-189 (X1→X4)
**Companion spec:** `2026-09-03-rerun-review-workspace-design.md` (rendering consumes this contract).

## Purpose

Fit primitive decompositions (bounded box faces, cylinders, superquadrics) to the
object clouds our loop accumulates — on any partial step-k cloud, uniformly across
algorithms, reviewable in the Rerun workspace, callable both from CLI (research)
and as a plain function (the real loop later).

## The contract

A fitter is a **pure function**: points in → `Decomposition` out. No disk, no
viewer, no globals. Deterministic under a fixed seed wherever the algorithm allows.

**Input:** `(N,3) float64` points, base frame. Optional context: table normal,
previous step's `Decomposition` (warm start). Nothing else.

**Output — `Decomposition` (JSON-serializable dict):**

```
method: str                  # bank name, e.g. "planar", "cyl", "ems"
version: str                 # algo version + parameter fingerprint
abstain: bool, reason: str   # fitter may refuse: "cloud not ripe" (maturity-gate hook)
primitives: [
  { kind:   "box" | "cylinder" | "plane_patch" | "superquadric" | "mesh",
    pose:   4x4 row-major,   # EVERY kind poses the same way, base frame, Z-up
    params: per-kind,        # box: extents[3] · cylinder: radius, length ·
                             # plane_patch: extents[2] · superquadric: scale[3], eps[2] ·
                             # mesh: vertices, faces (inline)
    label:  str,             # stable, content-derived ("face_0", "body")
    inliers: [int],          # indices into the input cloud
    residual_mm: float, support: int },
  ...
]
coverage: float              # fraction of cloud points claimed by any primitive
residual_mm: float           # decomposition-level
n_primitives: int
```

**The load-bearing rule: kinds are CLOSED, methods are OPEN.** Every algorithm —
today's three and anything adopted later — must express its answer in the five
kinds. Storage, Rerun rendering, and the future best-view derivation understand
kinds only; a new method therefore needs zero downstream work. Non-analytic
output degrades to `mesh` and still renders.

## The bank

- Registry: name → fitter callable. Adding a method = registering one function.
- **In code** (the real-loop entry point later): `bank.fit(points, methods=[...]) -> {name: Decomposition}`
- **CLI** (thin wrapper over the same call, repo fire style):
  `p <fit-cli> run --run 2408-cup3 --steps all --methods planar,cyl,ems`
  plus `--steps final` / `--steps 5,10,20`.

## Algorithms in scope now (MAY-189; the shakedown judges them)

1. **planar** — Open3D `detect_planar_patches` → bounded oriented boxes per face.
   Guard: curvature check so large-radius cylinders don't get tiled with fake planes.
2. **cyl** — cylinder RANSAC → axis, radius; extent computed from inliers.
   Library route (CGAL bindings / pyransac3d / own normals+LSQ) is vetted at build
   time, not decreed here.
3. **ems** — EMS probabilistic superquadric recovery (~90 ms/object).
Out of scope: learned/neural fitters, mesh reconstruction, multi-object splitting.

Known traps (from pass-1 research, carry into implementation): address primitives
by geometric content never extraction index (RANSAC nondeterminism); scale any
absolute-threshold defaults to our ~0.1 m objects; residual-family scores are
blind to coherent bias (the ~4 cm fused-vs-live offset) — never sold as accuracy.

## Storage (decided by Anton delegation, 2026-09-03)

Durable derived tree, gitignored, regeneratable, out of /tmp:

```
inspection/data/derived/<run>/
  steps/step_NNN.ply           # from step_replay (migrated from /tmp)
  fits/<method>/step_NNN.json  # the Decomposition, verbatim
  rrd/<run>.rrd                # the review workspace
```

The workspace producer sweeps `fits/*/` and emits one toggleable channel per
method — Anton never touches Rerun code; the closed-kind vocabulary is the
guarantee that anything a fitter emits can be drawn.

## Acceptance (X1 slice)

- Three fitters registered and run on 2408-cup3/4/5: final cloud + all steps.
- Fit JSONs on disk per (run, method, step); evidence renders of each fitter's
  decomposition over the cloud (final + a few representative steps).
- A written plausible-vs-garbage verdict per fitter per run, with the failure
  modes named (e.g. planar tiling the cup body) — that judgment IS the X1 output.
- Scores logged at every step (X5 by-product).
