# Geometry-Aware View Planning for AI-Driven Inspection

## Context

The project demonstrates AI-driven inspection: a VLM-orchestrated robot (UR5e,
eye-in-hand camera) derives *how to inspect* from a question instead of replaying
a programmed path — the baseline being obsoleted is ADAM, which runs pre-defined
trajectories. Today the agent picks views from a viewsphere addressed as {h,v}
cells: 12×30° horizontal bins, 3 vertical bins, with all views vertical. This
sphere is constructed with no relation to the inspected object's geometry — cells
are uniform around the object regardless of its shape.

Albert (VBTI, 2026-08-27) flagged — twice already in communication — that evidence
images should be the most orthogonal view of the inspected surface, which the
geometry-agnostic sphere cannot guarantee. He sketched the target behaviour: a box
should be scanned face-by-face orthogonally; a cylinder orbited with the camera
normal to the surface — the robot recognizing the shape and planning coverage
from it.

## Problem

The current viewsphere lives in complete isolation from the geometry of the
inspected object — clearly not what we expect from an inspection process, where
the object's geometry should actually influence the views.

Concretely: inspection starts from a uniform viewsphere around the object, so
the view set may contain no view that is clearly orthogonal to a given surface —
for a box, the true orthogonal direction can fall between two neighbouring
views. The {h,v} sphere cannot express this: view directions are fixed by the
sphere's bins, not by the object's faces, so for most geometries the orthogonal
view simply does not exist in the action space. For complex geometries a uniform
sphere can also limit visual coverage, but the current focus is objects of
primitive shape.

The second half of the problem is navigability. The uniform space is easy for
the LLM agent to navigate — that simplicity is why it works. Making the space
geometry-adaptive makes it complex, and that complexity must somehow be
presented to an agent that, in the current design, is blind and sees the world
through vision subagents. The design could change, but the limitation stands:
there is no reliable method for presenting spatial structure to an LLM. So the
core problem is how to model the action space such that the model can both
understand it and use it in the inspection process.

## Goals

- **Action space:** define an action space conditioned on the inspected object's
  geometry — views oriented by the object's surfaces (orthogonal where a surface
  exists) and covering everything worth seeing on that shape — rather than by
  uniform bins around it. The space must stay simple enough for the VLM to use;
  and/or we design a method to represent it to the VLM in a tractable way that
  allows navigation.
- **(Auxiliary) view-quality algorithm:** integrate an algorithmic solution that
  can judge whether a given view is optimal within its neighbouring region —
  i.e., an algorithm that produces the best possible view positions given the
  object's PLY geometry.

## Non-Goals

- **Full coverage of complex geometries** — the uniform-sphere coverage
  limitation is noted, but this research targets objects of primitive shape.
- **Hand-programmed inspection recipes per object — partially.** Fixed recipes
  per whole object or class are a regression to ADAM. But if the shape can be
  decomposed into primitives, deterministic rules at the primitive level (box
  face, cylinder orbit) are fine.

## Research Questions

The questions are roughly sequential, though not strictly dependent.

- **RQ1 — Judging a view:** given the surveyed geometry (PLY) — or the captured
  image itself — how do we algorithmically score the quality of a single view
  for a surface region: how orthogonal it is, how well it frames the region, and
  whether it is the best possible view within its neighbouring region?
- **RQ2 — Selecting best views:** how do we derive the set of best view
  positions from the object's geometry? This may build on RQ1's judgement, but
  might also work without it if views follow directly from the shape.
- **RQ3 — Composing the action space:** how do we compose the selected views
  into an action space — how are views addressed and related to each other, and
  how does the space stay consistent across different object shapes?
- **RQ4 — Presenting to the VLM:** how do we present this action space to the
  VLM so it can understand and navigate it during inspection, keeping a
  geometry-conditioned space as tractable as the current uniform grid? The
  orchestrator being blind is the current design, not a constraint — the
  research may show it should see.

## Constraints & Givens

- **The geometry is a survey cloud, not a model** — a fused multi-view point
  cloud (PLY) with real defects: pose error of a few mm, per-view depth dropout
  on glossy/dark surfaces, and a measured ~4 cm offset between the fused cloud
  and what a single view's own depth reports.
- **Not every desired view is reachable** — views must survive IK and collision
  checking (object inflation, gripper+camera hull, table); the reachable shell
  radius has a hard minimum. An orthogonal view that exists geometrically may
  not exist kinematically.
- **VLMs cannot emit metric poses** — working systems have the model pick from
  pre-enumerated views; whatever the new space looks like, selection stays
  discrete.
- **Platform is fixed** — UR5e, RG2 + D405 eye-in-hand, kinematic sandbox for
  development; the arm only for calibration and demo.

## Candidate Directions

Two branches, with a shared hunch that this task is most probably optimization.

- **Analytical track:** solve the geometry — derive the best pose directly from
  the PLY (and/or the image). If the view judgement is possible in PLY space,
  this track may close the problem on its own.
- **Optimization track:** build an objective judge of how good and optimal a
  view is, then optimize the camera position within a region until it arrives
  at the optimal view. The judgement should be tried in PLY space first; only
  if it turns out image space is required does the question of modeling unseen
  views arise. (Speculative aside for that case only: generative world models
  such as NVIDIA Cosmos might predict nearby views under small camera moves for
  the judge to score — interesting research, not a first-line solution.)

**Methodology — benchmark-driven from the start.** Before building solutions we
design a quantitative framework to test and refine them: a benchmark for view
quality / view selection, so every candidate is measured, not eyeballed. All
research on this project is benchmark-driven.

## Sources
