# Face-Evidence Views — boxguard normals in the live loop

*2026-09-14, designed with Anton. Status: awaiting review.*

## Goal

The evidence image for an inspected object is the **orthogonal view of the
relevant face**, derived from geometry — never from a programmed pose. The
boxguard cuboid already carries the orthogonality: each face's normal IS the
orthogonal view direction. This design wires those face views into the live
loop: the agent is steered to prefer them, and a resolution ladder finds the
closest accessible view when the exact orthogonal pose cannot be reached.

Success criterion (demo v2): repositioning the box moves its face views with
it, and the cited evidence is the orthogonal shot of the coded side.

## Face targets — the foundation

- Source: the **freshest step's boxguard** (`decomposition.json`), re-resolved
  at *request time*. A repositioned box re-derives its views automatically.
- Five faces, bottom excluded. Handles are box-frame stable under yaw:
  `+x, -x, +y, -y, top`.
- Target pose per face: camera at `face_center + n̂ · r_f`, boresight `−n̂`.
  `r_f` comes from `radius_for_extent(face extent)` — per-face framing, same
  convention as the survey — **clamped to R_MIN**: the padding-is-standoff
  rule keeps owning the floor (see viewsphere-radius-reachability-cliff).
- Roll about the boresight is a **searched DOF, not rule-fixed** (Anton
  2026-09-14): the up-vector rule gives roll 0, and every ladder candidate
  walks `ROLLS` (0°, 180° — the same set `cell_reachable`/`plan_to_cell`
  use) before being declared invalid. The camera slab sits off-axis on the
  tool cross, so the flipped roll clears collisions the upright roll cannot.
- Face targets are a **second target kind** in the Supervisor:
  `("face", name)` next to cell tuples. The motion ladder already plans to
  arbitrary flange poses; the cell address was only ever a lookup in front
  of it.

## Stability gate — when face views become capturable

Face views are offered for capture only once the boxguard is **stable across
two consecutive refits**: max angular change of any face normal (yaw delta,
in practice) below tolerance between refit *k−1* and *k*, two times in a row.

- Default tolerance: 5° (tunable; benchmarked later, not sacred).
- Before the gate opens, faces appear in the menu as `settling` — visible so
  the agent knows they are coming, not requestable.
- Rationale (Anton): a boxguard mid-convergence yields normals that swing
  between captures; capturing "orthogonal" evidence against a moving box
  wastes a motion and cites a view that is not actually orthogonal.

## Menu integration — one menu, faces marked as priority

Face views are **merged into the existing menu**, not a parallel block:

- Each face is a menu entry alongside cells, marked as a face view with its
  status: `orthogonal` / `degraded (θ° off)` / `settling` / `unreachable` /
  `visited`.
- The menu prompt states the priority explicitly: **evidence mainly comes
  from face views — prefer them when they serve the question; cells remain
  for coverage, exploration, and evidence the faces don't show** (a soft
  priority, not a rule: the LLM stays free to cite any view that answers
  better).
- The tier split holds: eyes report framing, the LLM decides. `answer`
  cites a face step like any other step.

## Request-time resolution ladder

On `request(("face", name))`, resolve in order — each rung is the cheapest
fix for its own failure mode:

1. **Exact orthogonal pose.** Endpoint check (IK branches + collision, the
   same primitive `cell_reachable` uses). Valid → ladder-plan it.
2. **Z-lift — the slim-object rung.** A short or slim object puts a side
   face's orthogonal camera at table height. If the exact pose is invalid,
   raise the camera along world +Z in small steps (default 2 cm), keeping
   the boresight on the face center, until the endpoint is valid or the
   implied off-angle reaches 45° (then rung 3 takes over with its full
   freedom). Azimuth stays face-aligned; only elevation degrades.
3. **Cone search — the general rung.** Tilt rings around `n̂`:
   θ = 5°,10°,…,60°, ~12 azimuths per ring, standoff `r_f`, always looking
   at the face center. First ring with a valid endpoint that also plans →
   done; first plannable is the most orthogonal accessible by construction.
4. **Honest refusal.** Nothing plannable by θ = 60° (the classic
   measurability cutoff) → the face is `unreachable` in the menu; the agent
   falls back to cells.

Whatever rung wins, the **achieved off-angle θ is recorded** in the step and
shown in the menu (`degraded 15°`) — the agent knows the quality of the
evidence it cites.

## Supervisor and record plumbing

- `("face", name)` flows through request → plan → preview → confirm →
  execute → capture exactly like a cell; the request gate consults the face
  status (gate open, not visited) instead of `_reach`.
- The step's `ViewState`/address records the face handle, the resolved pose,
  and achieved θ. Faces get their own `visited` marks; **cell coverage
  semantics are untouched** — face views are evidence on top of the orbit,
  not replacements for it.
- The capture-refit self-trap applies doubly to close face stances; the
  3-fails retreat (2026-09-14) is the escape hatch. Endpoint clearance
  headroom was considered and deferred.

## Out of scope

- No VLM-side changes; eyes tier untouched.
- No viewsphere changes; cells, rings, and R_MIN derivation stay as they are.
- No benchmark work in this build — the gate tolerance and search defaults
  are engineering defaults until the benchmark arbitrates them.

## Testing

- **Offline, first:** replay recorded runs (`1409-*`, `1009-*`): boxguard per
  step → face targets, gate open/close trace, resolution ladder outcomes and
  achieved θ per face. Eyeball in rerun via the extension-mode pattern.
- **Machine tests:** face request path on `FakeRig` — gate closed → request
  dropped; gate open → planned and captured; unreachable face → honest menu
  status; z-lift rung exercised with a low face target.
- **Bench:** live run on the coded box, then reposition and rerun — the
  demo-v2 criterion.
