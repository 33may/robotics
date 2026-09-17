# Ray views — distance is emergent, not computed (2026-09-17)

Anton's ruling, designed in session 2026-09-17 after the 1709-cam-ai-demo1 run.
Supersedes the distance-selection half of `2026-09-14-face-evidence-design.md`
(the face *gate* and the face *addresses* survive; the resolution ladder and
every distance formula die).

## Problem

Distance selection had grown into seven interacting mechanisms across two
frames: `FILL_TARGET` + three extent rules, `CLOSENESS`, static `R_MIN`,
per-run `derive_r_floor`, `max(want, floor)`, and a face variant that borrowed
the shell floor across frames. The demo showed the failure class this breeds:
face evidence views ended 0.28–0.34 m from center while ring cells sat at
0.24 — 40% farther from the surface they inspect — and the z-lift ladder then
tilted three of five faces 35–42° into keystone. The *model* (distance as a
computed formula with stacked guards) makes frame-mismatch bugs natural; no
patch removes the class.

## Requirement (Anton, verbatim intent)

Get as much object as possible in the frame, never collide with the object,
and always be able to escape the position.

## Core idea

**A view is a ray, not a point. Distance is the closest sample on the ray
that is provably safe.** No fill formula, no floors, no closeness slider —
the sampler's stop conditions replace them all:

| requirement            | mechanism                                            |
|------------------------|------------------------------------------------------|
| max object in frame    | closest safe sample wins                             |
| never collide          | every sample validated in the sandbox before motion  |
| always escapable       | the exit leg is part of every move's plan            |

Field anchor: PB-NBV (arXiv 2501.10663) and ObjSplat (arXiv 2601.06997) both
anchor viewpoint distance to the object *surface* (sensor constant + bounding
half-diagonal); the classic constraint school (Cowan & Kovesi 1988) never
computes distance at all — a view must pass a set of lower bounds and the
closest passing distance wins. This design is the constraint school executed
by sampling.

## View vocabulary

- **Cell rays** — origin: object centroid (current belief at request time);
  directions: the existing {h,v} grid (12×30° azimuth, elevation bins, h=0
  facing base). Addresses, coverage bitmap, menu text, egocentric glosses:
  unchanged. Nothing above the motion layer notices the rebuild.
- **Face rays** — a *family* per boxguard face; gate stays 2 stable refits.
  - **Rung 1 — parallel shift:** origin lifted +z from the face center in
    `RAY_STEP_M` steps, direction locked to the outward normal. A vertical
    plane parallel to the sensor projects as a pure similarity — zero
    keystone at any shift; the face rides lower in frame. Shift cap = FOV
    containment, computed per depth, not a knob. Shifts walk ascending; the
    first shift whose ray has any valid sample wins.
  - **Rung 2 — tilt:** only when no shift works. The 5°..60° × 12-azimuth
    cone re-expressed as rays through the face center, sampled identically.
  - Views record `{shift, theta, depth}` — evidence states what kind of view
    it cites ("fronto-parallel, 4 cm lifted" vs "tilted 25°").
- Rolls stay {0°, 180°}, evaluated per sample (per-roll sample lists).

## The ray sampler (one component serves cells and faces)

Sample depths every `RAY_STEP_M` (0.02, configurable constant) from the
padded-box surface + tool clearance out to `R_MAX` (arm physics, the outer
bound). Per sample, cheapest check first:

1. sensor min range (D405 720p min-Z ≈ 0.105 m, datasheet constant)
2. IK branches at both rolls (~3 µs, EAIK nominal-DH + refine)
3. collision of every free branch (~21 µs/config, Pinocchio+coal)

Output per ray: ordered valid-sample lists, one per roll, or empty. A full
36-cell refresh is estimated in the low hundreds of ms → rerun after every
refit; menu accessibility marks are always current. There is no frozen shell
— coverage keys are directions.

## Move transaction — three planned legs

1. **Slide-out** — radially outward on the *current* ray to the shallowest
   depth from which leg 2 plans. Search is linear and current-depth-first:
   entry on the target ray is pinned to its farthest valid sample, so the
   space is ≤ ~18 out-depths × 1 plan (~30 ms each), not out×entry
   quadratic (18×18 ≈ 10 s — infeasible; widen entry only if the pinned
   search exhausts).
2. **Hop** — free-space move between rays via the existing two-tier ladder
   (direct → OMPL best-of-N). **The hop absorbs roll flips** — the wrist may
   rotate 180° here, in open space, never during a slide.
3. **Slide-in** — short radial leg to the target ray's closest safe sample.
   Roll-locked, fixed IK branch, no ±π re-expression mid-leg, waypoints
   recorded.

All legs validate against the world they were planned under (the rig-gate
rule). First move after survey: legs 2–3 only. Fault mid-transaction: the
retreat retraces the recorded waypoints in reverse — the retreat machinery
becomes the executor of a normal system state, not a rescue.

Why hop-then-slide: the long move stays in open space where planning is
easy; the delicate approach is short, straight, and radial — it cannot
laterally sweep the object.

## Failure semantics

No valid sample in the ray (or ray family), or no out-depth from which the
transition plans → the address is **inaccessible**; the menu says so. No
fallback substitution, no silent degradation (fallbacks scream). Flat-object
side faces stay servable via the shift rung (~7 cm lift on the 1709 box →
fronto-parallel instead of 35–42° keystone).

## Record & schema

`view_methods` params: ray grid + `RAY_STEP_M` + shift/tilt families replace
`r`/`r_floor` (additive migration; old runs read fine). Steps record the
address, chosen depth, shift/θ for faces, and all legs' waypoints.
Decomposition, coverage, transcripts untouched.

## Dies / stays / later

- **Dies:** `CLOSENESS`, `FILL_TARGET` + extent rules, static `R_MIN`,
  `derive_r_floor`, `max(want, floor)`, frozen shell radius, the face floor,
  the z-lift rung.
- **Stays:** object padding (the one safety number), sensor min range
  (datasheet), `R_MAX` (physics), two-tier motion ladder, approval gate,
  roll pair, face stability gate, retreat-retrace.
- **Later (explicitly out of scope):** a closeness *cap* as one more stop
  condition — camera-closeness control plugs in with zero new structure.

## Feasibility estimates (paper numbers — Stage A measures them)

| computation                                | combinations                 | est. time          |
|--------------------------------------------|------------------------------|--------------------|
| one ray validity                           | ~18 depths × 2 rolls × 8 br  | ~6 ms              |
| full 36-cell refresh (per refit)           | 36 rays                      | ~230 ms            |
| face family, typical (shift rung hits)     | 1–3 rays                     | ~20 ms             |
| face family, worst (full cone)             | ~8 + 144 rays                | ~1 s               |
| slide-in resolve                           | segment validation only      | <1 ms              |
| slide-out search                           | ≤18 plans, linear            | 30–60 ms typ.      |
| hop                                        | 1 plan                       | ~30 ms             |
| **total per move**                         |                              | **<0.5 s typ., ~2 s worst** |

## Build stages (each gated; A is the falsifier)

- **A — sampler offline.** Ray sampler + accessibility map, kinematic
  sandbox only. Replay bench over recorded runs (cup, flat box, lan, 1709
  demo): render old endpoint vs ray endpoint per view, measure the timing
  table for real. **Gate: renders visibly win and timings hold, else stop.**
- **B — machine.** Three-leg transaction, `adjusting` phase, ray resolution
  replaces shell + faceview resolution. TDD on the fake rig, including
  transition-inaccessible and mid-slide fault → reverse retrace.
- **C — surfaces.** Menu accessibility marks, brain prompt reality (face
  entries carry shift/θ), record schema migration, UI markers at sampled
  endpoints.
- **D — live.** Bench run under the updated keepout world; then the demo-v2
  rerun on the L515 box — side labels fronto-parallel is the acceptance
  picture.

Branch: `ray-views`; test offline + live bench, then merge.

## Settled parameters

`RAY_STEP_M = 0.02` (configurable constant, shared by depth and shift
sampling) · hop entry = farthest valid sample · exit search
current-depth-first · shift ascending before depth (predictable ladder) ·
tilt keeps 12 azimuths · slides roll-locked, hop flips.
