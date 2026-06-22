---
name: reference-oli-corpus-structured-tools
description: How to use the 9 new structured oli-corpus-mcp tools (robots/joints/links/sdk_joint_order/pkg_info/nodes/topics/find_symbol/raw_file) and what changed in the 2026-06-22 expansion
metadata:
  type: reference
---

# Oli corpus — structured tools (added 2026-06-22)

The corpus expanded from 3 LimX-webpage doc_ids (`quick-start`, `sdk-guide`, `user-manual`) to **6 doc_ids covering 1,066 chunks** plus a typed graph layer (`robots`, `joints`, `links`, `packages`, `pkg_deps`, `launch_nodes`, `node_topics`, `node_params`, `api_symbols`).

For the existing free-text tools (`list_docs`, `search`, `get_section`, `cite`), see [[reference-oli-corpus-mcp]].

## Decision tree — which tool first?

```
Question is about ...        →  Use ...
─────────────────────────────────────────────────────────
robot variants / what exists →  robots()
joint order / axes / limits  →  joints(robot_id)
link inertials / meshes      →  links(robot_id)
sim-to-real joint mapping    →  sdk_joint_order(robot_id)   ← critical
ROS pkg deps / dependents    →  pkg_info(name)
what launches start          →  nodes(launch_uri? pkg?)
topic pub/sub                →  topics(...) — empty for MROS YAML; fallback to find_symbol
C++ struct/class/enum lookup →  find_symbol(query)
need the raw file content    →  raw_file(source_uri)
free-text / fuzzy / configs  →  search(query)               ← still the existing tool
```

If unsure, start with `search()` — it covers the whole corpus including structured chunks' markdown summaries.

## The 9 new tools, with realistic usage

### `robots()` — list robot variants
- Returns 26 rows; each has `robot_id`, `description`, `source_uri` of the URDF
- RL variants are suffixed `_rl` (e.g. `HU_D04_01` and `HU_D04_01_rl` are both indexed)
- Use this first when you don't know the exact robot id

### `joints(robot_id, include_fixed=False)`
- Default skips fixed (welded) joints — you almost always want DoFs only
- Each row: `urdf_idx` (declaration order, all joints), `dof_idx` (DoF sequential), `sdk_idx` (SDK array position, NULL if unresolved), `axis` (x,y,z) tuple, `lower`/`upper`/`effort`/`velocity` limits, parent/child links
- Critical: `dof_idx` matches IsaacLab's articulation DoF order; `sdk_idx` matches the robot's `q`/`dq`/`tau` array. They're usually different.

### `links(robot_id)`
- Mass, COM (`com_x/y/z`), inertia tensor (`ixx ixy ixz iyy iyz izz`), and mesh URIs
- Mesh URIs can be passed straight to `raw_file()` to fetch STL bytes for URDF→USD conversion

### `sdk_joint_order(robot_id)` ★ sim-to-real correctness anchor
- Returns `{"robot_id", "order": [name, ...], "source_uri"}` — the canonical joint sequence the SDK uses for `q`, `dq`, `tau`, `Kp`, `Kd`
- Sourced from `humanoid-rl-deploy-python/controllers/<robot>/walk_controller/walk_param.yaml` comment annotations
- **Currently resolved for `HU_D04_01`, `HU_D04_01_rl`, `HU_D03_03`, `HU_D03_03_rl`.** Other robots raise `ValueError` with a clear "deploy config missing" message
- **Never silently guesses** — if a sim adapter author needs the order for an unresolved robot, they must add the canonical sequence to the deploy repo first

### `pkg_info(name)`
- Returns metadata + `deps` (what this package needs) + `dependents` (what needs this package)
- 148 packages indexed from the tarball's `install/share/*/package.xml`
- Good for impact analysis: "if I touch mroscontrol, what breaks?"

### `nodes(launch_uri=None, pkg=None)`
- 13 nodes across 7 indexed launch files (1 file skipped due to non-printable chars)
- LimX's MROS YAML launch format declares: node name, executable path, disabled/oneshot/autostart flags, argv

### `topics(node=None, kind=None, topic=None)`
- **Currently returns empty.** MROS YAML launch files don't declare topic remaps — those are baked into binaries
- Reserved for future source roots (ROS XML launches, runtime topic-graph snapshots)
- To find topics, fall back to `find_symbol()` on `RobotCmd`/`RobotState`/etc. or `raw_file()` on the relevant binary's source

### `find_symbol(query, kind=None, limit=50)`
- 397 C/C++ symbols indexed (41 from `limxsdk`, 356 from tarball `install/mbl/include/`)
- Exact-match first, falls back to substring
- `kind` filter: `class`, `struct`, `enum`, `typedef`, `using`
- Each row: `lib` (path-derived), `source_uri`, `symbol`, `kind`, `signature` (full body for struct/class, prototype for typedef), `docstring` (extracted Doxygen)
- For wire-level structs (`RobotCmd`, `RobotState`, `ImuData`, `SensorJoy`) the limxsdk hit is canonical
- Same name can appear in multiple libs (e.g. `RobotState` is both a limxsdk struct AND an mbl enum) — check `lib`/`source_uri` to disambiguate

### `raw_file(source_uri, max_bytes=1_048_576)` ★ universal escape hatch
- Returns `{"encoding": "utf-8" | "base64", "content": "...", "size_bytes", "truncated"}`
- Text files: utf-8 string. Binaries (`.STL`, `.png`, `.onnx`, `.rknn`, `.so`, etc.): base64
- Refuses URIs whose doc_id isn't a structured source root — use `get_section()` for webpage docs
- Refuses symlink escapes and `..` traversal
- Default 1 MB cap; pass `max_bytes=10_000_000` for big meshes; check the `truncated` flag

## Common compound patterns

### Port robot X to Isaac Sim
```
1. robots()                                          → confirm robot_id
2. joints(robot_id)                                  → DoF list with axes + limits
3. links(robot_id)                                   → inertials + mesh URIs
4. for each mesh URI: raw_file(uri) (base64 STL)     → write to disk for converter
5. raw_file(<urdf_source_uri>)                       → utf-8 URDF for URDF→USD
6. sdk_joint_order(robot_id)                         → mapping for the sim adapter
7. find_symbol("RobotCmd")                           → command struct fields for adapter
```

### Train RL policy and deploy on real Oli
```
1. joints("HU_D04_01")                               → action space dims + clip ranges
2. find_symbol("RobotState")                         → observation struct (limxsdk one)
3. find_symbol("RobotCmd")                           → action mode + Kp/Kd semantics
4. sdk_joint_order("HU_D04_01")                      → IsaacLab→SDK index map
5. raw_file("oli-corpus://rl-deploy-python#main.py") → deployment entry point
6. raw_file("oli-corpus://rl-deploy-python#controllers/HU_D04_01/walk_controller/walk_param.yaml")
                                                     → loop rate, normalization, defaults
7. find_symbol("FallDetector")                       → safety envelope
```

### Investigate package coupling
```
1. pkg_info("mroscontrol")          → who depends on it (dependents)
2. for each dependent: pkg_info(d)  → walk the graph
3. raw_file(pkg.source_uri)         → see the package.xml directly if needed
```

## Source roots

Three vendored, LimX-authored source roots feed the structured layer:

| doc_id | path | role |
|---|---|---|
| `oli-main-2.2.12` | `humanoid/vendor/oli-main-software-2.2.12/install/` | Main Software tarball (gitignored payload, MANIFEST.tsv committed) |
| `limxsdk` | `humanoid/vendor/humanoid-mujoco-sim/limxsdk-lowlevel/include/limxsdk/` | C++ wire-level struct headers |
| `rl-deploy-python` | `humanoid/vendor/humanoid-rl-deploy-python/` | Production deploy reference + controller configs |

URI scheme: `oli-corpus://<doc_id>#<section>` where `section` is the relpath within that root.

## Gotchas

- **MCP server reload required after rebuild.** Tools are registered when the MCP server starts. After running `python -m _research.build_index`, restart the Claude Code session (or `/mcp` reload) or new tool calls hit stale code.
- **FTS5 syntax is unforgiving.** Don't include `(`, `.`, `"` literally in `search()` queries — they're operators. Quote-escape or rephrase. `search("damping_kd")` works; `search("tau_(")` errors.
- **`topics()` is empty for now.** Don't infer "no topics exist" — only "no ROS XML launch declared them in our indexed sources". Wire-level topic discovery still needs the probe.
- **`sdk_joint_order` raises for most robots.** Only `HU_D04_01`/`_rl` and `HU_D03_03`/`_rl` are covered today. If you need another robot, add the canonical sequence to `humanoid-rl-deploy-python/controllers/<robot>/walk_controller/walk_param.yaml` first.
- **Same C++ symbol name, different libs.** `RobotState` is both a limxsdk struct (data) and an mbl enum (state machine). Always check `lib`/`source_uri` to disambiguate.
- **Binary files: watch `truncated`.** STL meshes routinely exceed 1 MB. Pass `max_bytes=10_000_000` or chunk.

## Rebuild procedure

```bash
cd humanoid/docs/oli-corpus
conda run -n hum python -m _research.build_index
```

(Must be `-m _research.build_index`, not direct script invocation — the orchestrator uses relative imports.)

To verify: `tools.robots()` should return 26; `tools.sdk_joint_order("HU_D04_01")` should return 31 names.

## Design spec

Full design rationale at `humanoid/docs/superpowers/specs/2026-06-22-oli-corpus-structured-ingest-design.md`. Two use-case stress tests (Isaac Sim port; RL sim-to-real) drove the tool surface.

Related: [[reference-oli-corpus-mcp]] (overview), [[oli-main-software-tarball]] (vendor source), [[oli-sdk-3-layer-architecture]]
