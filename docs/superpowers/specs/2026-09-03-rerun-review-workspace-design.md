# Rerun review workspace for view-planning v2 — design

**Date:** 2026-09-03 · **Owner:** Anton · **Status:** approved in chat, ready to build
**Context:** MAY-189 (primitive deconstruction from the PLY) and the September build
month (`docs/research/view-planning/passes/pass1/methodology_v1.md`). The offline
research loop produces step-k clouds (replay of recorded runs) and primitive
decompositions (multiple fitter backends). Anton reviews everything in one
interactive workspace: a Rerun `.rrd` file per run.

## Scope

**Offline, passive, complete-file.** Research scripts write a complete `.rrd`; the
Rerun viewer (0.37) renders it as an interactive workspace with a step slider. No
live-loop instrumentation (the robot UI stays porthole). Streaming arrives later as
an increment — do not preclude it, do not build it.

**One `.rrd` per run** (e.g. `2408-cup3.rrd`), holding everything computed about
that run. Producers re-emit the whole file when they rerun; last writer wins.
Output location: `/tmp/rrd/<run>.rrd`. Recorded runs under `inspection/data/runs/`
are read-only artifacts — never write into them.

## User stories (v1 — all must work)

1. **Open a run.** `rerun /tmp/rrd/<run>.rrd` → workspace arrives pre-arranged
   (blueprint saved in the file): 3D viewer dominant, chain panel beside it, one
   step slider. Zero setup clicks, identical layout every run.
2. **Scrub the scan.** The slider is scan step k (views fused so far). Dragging it:
   the cloud grows view by view, each camera frustum appears at the pose it actually
   shot from and stays, the points view k contributed glow in an accent color at
   step k. Answers: when does the cloud stop being garbage? What did view 17 add?
3. **Review a deconstruction at each step.** Primitives from any fitter backend
   drape over the cloud at the step they were fitted on. Orbiting the 3D is the
   review act: do faces hug the surface, is the cylinder axis upright, did a
   method hallucinate a plane where the cloud is thin? Multiple methods can be
   present; plain entity show/hide is the only toggle machinery — no custom
   filters.
4. **See what the loop saw.** The chain panel shows the current step's identity
   chain, as in the porthole chain tab: prompt image, mask, kept points
   (`chain_prompt.png` / `chain_mask.png` / `chain_kept.png` from the view dir).
   Runs that predate the chain (e.g. 2408-cup3) fall back to `rgb.png` + `mask.png`.
5. **Trust the replay.** The stored `fused_cloud.npy` and the replayed final cloud
   overlay in two colors; divergence shows as a visible halo. (Bit-exact match is
   impossible — live table-plane RANSAC is unseeded; the acceptance bar is the
   verified sub-voxel agreement, see step_replay verification.)
6. **Receive evidence.** Every claim about geometry made by an agent this month
   ships as an `.rrd` Anton interrogates himself — same workspace, not curated
   screenshots.

## Anticipated, explicitly NOT in v1 (leave room, build nothing)

- **Score window**: fit-score curves vs step as an additional panel. Not designed yet.
- **Best-view rendering**: a backend will later compute best views from a primitive
  reconstruction; candidate views will render in 3D and as a camera view. Not now.
- **Streaming**: new objects/steps/primitives appearing live as they are computed.

## The primitive-drawing contract (the load-bearing interface)

One logging module owns the schema: `inspection/investigation/rr_log.py`. Producer
scripts call it; nothing else logs to Rerun directly. Its core contract: a fitter
backend hands over, per step, a plain list of primitives —

```python
{"kind": "box" | "cylinder" | "mesh",
 "pose": 4x4 or (center, rotation),      # base frame, Z-up
 "params": ...,                          # box: extents; cylinder: radius, length; mesh: vertices+faces
 "label": str,                           # e.g. "face_0", "body"
 "method": str,                          # e.g. "o3d_planes", "ransac_cyl", "ems_sq"
 "inliers": optional point indices}      # points this primitive claims
```

— and `rr_log` renders it. Adding a new fitter = producing this structure; zero
viewer work. Colors are assigned per method deterministically. Superquadrics and
anything non-analytic enter as `mesh` (triangulate on the producer side, e.g.
trimesh). Exact Rerun archetype names must be verified against the installed
rerun-sdk 0.37 docs at build time — the spec names capabilities, not symbols.

## Workspace content (implementation guide, not UX)

- Timeline: single int sequence `step`. Step k = state after fusing view k.
- 3D scene (base frame, Z-up): fused cloud at step k (neutral color); fresh points
  of step k (accent, present only at their step); camera frusta with the view's rgb
  (appear at their step, persist); table plane (thin, translucent); fitted
  primitives per method per step; stored-final-cloud overlay channel (static, one
  color) next to replayed-final.
- Chain panel: 2D images per step (chain trio, or rgb+mask fallback).
- Blueprint saved into the `.rrd` so the layout opens as designed; default
  visibility: cloud + cameras + chain on, primitive methods on (few exist yet),
  stored/replayed overlay off by default.
- First producer: `step_replay.py` (exists; emits step-k clouds — wire it to call
  `rr_log`). Fitter scripts follow and add their primitive channels.

## Acceptance (prove it with the real artifact, not by assertion)

1. Build `/tmp/rrd/2408-cup3.rrd` (and cup4/cup5) from the step_replay outputs.
2. Include one placeholder decomposition to prove the primitive contract renders
   before real fitters exist: per-step AABB of the cloud logged as `method="aabb"`
   box — clearly labeled placeholder, deleted when real fitters land.
3. Open in the viewer: layout arrives arranged; slider scrubs 0→N with cloud
   growth + camera appearance + fresh-point glow; chain panel tracks the slider;
   placeholder box drapes the cloud each step; stored-vs-replayed overlay toggles.
4. Screenshot evidence of each story for the report, but the `.rrd` files are the
   deliverable Anton opens himself.

## Environment

- `rerun-sdk` 0.37 into the `robo` conda env (`~/miniconda3/envs/robo`).
- Run scripts with that env's python. No hardware access. `inspection` is
  pip-installed editable; absolute imports; no sys.path hacks.
