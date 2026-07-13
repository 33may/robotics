# nav/ — the navigation module (plan + follow + orchestrate)

Answers **"get me to this goal"**. Read `../AGENTS.md` first (module map, hard rules, the
Protocol pattern).

## Structure (post-split — keep it this way)

| piece | role | state it owns |
|---|---|---|
| `nav.py: Nav` | THIN orchestrator: per-tick `localize → mapping.latest() → plan → follow`, hold semantics, mode stamping | the current `GoalCoordinate` + module refs — nothing else |
| `planner.py: plan_path` | pure A* (8-connected, soft clearance cost, weighted heuristic) | none |
| `planner.py: Planner` | stateful planning module | derived ROBOT layer (inflate + clearance) cached on `(grid id, map.version)`; path cache; goal-change detection; full/local re-plan |
| `controller.py: PurePursuit` | follower: (pose, path) → body twist | none |
| `arm.py: ArmedNav` | gate: disarmed = joystick, armed = Nav | armed flag |
| `types.py: GoalCoordinate` | nav's own input contract | — |

## The rules that make this module what it is

- **`Planner.plan(pose, goal, world_map)` consumes the emitted `Map` VALUE — it must NEVER hold
  a `MappingModule` (or any module) ref** (design.md D8; enforced by a signature check in
  `tests/oli/reason/test_architecture.py`). Nav does the `latest()` pull — that is the
  orchestrator's job, and the only place module refs are allowed.
- A map `(grid identity, version)` change rebuilds the robot layer AND drops the cached path
  (the spliced tail was planned against the old world — design.md D9).
- `Planner.clear()` forgets goal+path (next plan is FULL) but keeps the robot layer (it depends
  on the map, not the goal). Nav calls it on `set_goal`/`clear_goal`/mapless ticks.
- Direct-Planner semantics note: an equal-value goal on consecutive `plan()` calls = SAME goal
  (local re-plan). Forcing a full re-plan is `set_goal`'s job (it calls `clear()`).
- Robot/policy knobs (footprint radius, clearance, ε, horizon) are `Planner` constructor args —
  the map stays footprint-free (one baked map serves any robot).
- Hold semantics: no goal / no pose / no map / no path → zero-velocity `Intent`, NEVER a joint
  command.
- No typed Path contract (Anton, 2026-07-13): paths stay `List[Tuple[float, float]]` world
  waypoints. The plan→motion upgrade (follower) is later, separate work.

## Extending

- A second planner (e.g. learned) → extract a `Planner` protocol THEN (shape is already
  `plan/clear/path`), not before. See "protocol = we intend to swap this" in `../AGENTS.md`.
- Anything touching the localize seam: the pose source is `..localization` — do not add pose
  logic here.
