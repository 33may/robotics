# Oli Corpus — Structured Ingest Design

**Date**: 2026-06-22
**Author**: humanoid-dev-lead (Claude) + Anton
**Status**: Draft for review

## Problem

The `oli-corpus-mcp` server currently indexes only three LimX webpages (`quick-start`, `user-manual`, `sdk-guide`). It cannot answer the questions agents actually ask when doing real work on Oli:

- Joint orderings, axes, and limits for a specific robot variant
- The wire-level command/state struct fields and their units
- The mapping from URDF joint index to the SDK's `q`/`dq`/`tau` array index
- Mesh paths for sim conversion
- Launch-file → topic → node graphs
- Controller config values (loop rates, gains, safety envelopes)

The Main Software v2.2.12 tarball that LimX ships contains the answers, but as a heterogeneous colcon ROS2 install workspace (148 packages, 7000+ entries, mixed text and binaries) — not as queryable docs.

This spec defines how to ingest the tarball and two adjacent vendored repos into the corpus so that an agent porting Oli to Isaac Sim or deploying an RL policy can get answers without reading raw files by hand.

## Scope

**In scope**: ingesting LimX-authored artifacts from three vendored source roots into the existing `oli-corpus-mcp` server. New typed SQLite tables and MCP tools. Extractors for URDF, package.xml, launch XML, C++ headers, and a `raw_file` retrieval tool.

**Out of scope**: ingesting our own project code (we keep the corpus a LimX-authored mirror, per Anton's scope decision on 2026-06-22). Recovering ROS `.msg`/`.srv` IDL — the tarball strips them, so wire-protocol topic shapes remain a future task for a runtime probe. Indexing the `oli_mcp_server` package contents.

## Use cases the design must satisfy

Two use-case walkthroughs validated the tool surface before writing this spec:

1. **Port Oli to Isaac Sim.** Agent needs robot variants, joint tables, link inertials, mesh paths, raw URDFs, SDK command struct, and the SDK joint array order. Without `raw_file()` and `sdk_joint_order()` the agent cannot complete the port; with them all 14 sub-questions are answerable.
2. **Train a locomotion policy in IsaacLab and deploy via `humanoid-rl-deploy-python`.** Agent needs joints, action/observation space (`RobotCmd`/`RobotState`), control rate, normalization constants, policy load path, safety envelope. Requires the extractor to also walk the vendored deploy-python repo. 11 of 14 sub-questions answered directly; 2 are correctly out of corpus scope (domain randomization, undocumented runtime behavior).

Both walkthroughs converged on the same critical patches: `raw_file(uri)`, `sdk_joint_order(robot)`, and `links(robot)` with mesh references.

## Approach

**One database, layered schema, hybrid tool surface.** Extend the existing `index/corpus.sqlite` with typed tables that coexist with the FTS chunks. Every typed row carries an `oli-corpus://` URI back to its source file, so structured answers cite the same way text answers do. Existing tools (`search`, `get_section`, `cite`, `list_docs`) keep working unchanged.

Two alternatives were rejected: flat ingest (Approach A — fails the structured-query use case) and parallel database (Approach B — two DBs, harder citation, no real isolation benefit).

## Source roots

The extractor walks three vendored directories, all LimX-authored:

1. `humanoid/vendor/oli-main-software-2.2.12/install/` — the unpacked Main Software tarball. **Gitignored** (1.5 GB); committed in the repo only as a versioned README + `manifest.yaml` (relpath → sha256). Each developer downloads and unpacks the tarball themselves; the indexer hashes against the manifest at extract time to detect drift.
2. `humanoid/vendor/humanoid-mujoco-sim/limxsdk-lowlevel/include/limxsdk/` — the C++ headers carrying the wire-level structs (`RobotCmd`, `RobotState`, `ImuData`, `SensorJoy`, `DiagnosticValue`). Already vendored via git submodule.
3. `humanoid/vendor/humanoid-rl-deploy-python/` — the production Python deploy reference. README, `main.py`, controller YAMLs. Already vendored.

The version in the folder name (`oli-main-software-2.2.12`) ensures upgrades land in a new folder rather than silently overwriting.

## Citation URI scheme

Existing webpage docs keep their current `oli-corpus://{doc_id}#{section}?part={n}` form.

New tarball-derived sources use a versioned namespace:

```
oli-corpus://oli-main/2.2.12/{relpath}#{anchor}
oli-corpus://limxsdk/{relpath}#{symbol}
oli-corpus://rl-deploy-python/{relpath}#{anchor}
```

Versioned URIs let multiple tarball versions coexist in the index if needed for diffing.

## Schema additions

New SQLite tables, in addition to the existing `chunks` and `chunks_fts`:

```sql
CREATE TABLE robots (
  robot_id TEXT PRIMARY KEY,           -- e.g. "HU_D04", "HU_D04_with_gripper"
  source_uri TEXT NOT NULL,            -- URI of the .urdf
  description TEXT
);

CREATE TABLE joints (
  robot_id TEXT NOT NULL,
  urdf_idx INTEGER NOT NULL,           -- order in the URDF
  sdk_idx INTEGER,                     -- order in SDK's q/dq/tau array, NULL if unresolved
  name TEXT NOT NULL,
  type TEXT NOT NULL,                  -- revolute, continuous, prismatic, fixed
  parent_link TEXT NOT NULL,
  child_link TEXT NOT NULL,
  axis_x REAL, axis_y REAL, axis_z REAL,
  lower REAL, upper REAL,              -- position limits (rad or m)
  effort REAL,                          -- torque/force limit
  velocity REAL,                        -- velocity limit
  mimic_of TEXT,                        -- joint name if mimicking
  source_uri TEXT NOT NULL,
  PRIMARY KEY (robot_id, urdf_idx)
);

CREATE TABLE links (
  robot_id TEXT NOT NULL,
  name TEXT NOT NULL,
  mass REAL,
  com_x REAL, com_y REAL, com_z REAL,
  ixx REAL, ixy REAL, ixz REAL, iyy REAL, iyz REAL, izz REAL,
  visual_mesh_uri TEXT,                 -- oli-corpus:// URI to mesh file
  collision_mesh_uri TEXT,
  source_uri TEXT NOT NULL,
  PRIMARY KEY (robot_id, name)
);

CREATE TABLE packages (
  name TEXT PRIMARY KEY,
  version TEXT,
  description TEXT,
  maintainer TEXT,
  source_uri TEXT NOT NULL
);

CREATE TABLE pkg_deps (
  pkg TEXT NOT NULL,
  dep TEXT NOT NULL,
  kind TEXT NOT NULL,                   -- build, exec, test
  PRIMARY KEY (pkg, dep, kind)
);

CREATE TABLE launch_nodes (
  launch_uri TEXT NOT NULL,
  pkg TEXT NOT NULL,
  exec TEXT NOT NULL,
  name TEXT NOT NULL,
  namespace TEXT,
  PRIMARY KEY (launch_uri, name)
);

CREATE TABLE node_topics (
  launch_uri TEXT NOT NULL,
  node TEXT NOT NULL,
  kind TEXT NOT NULL,                   -- pub, sub, remap, srv-client, srv-server
  topic TEXT NOT NULL,
  remap_from TEXT
);

CREATE TABLE node_params (
  launch_uri TEXT NOT NULL,
  node TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT,
  value_kind TEXT                       -- string, int, float, bool, list, file
);

CREATE TABLE api_symbols (
  symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
  lib TEXT NOT NULL,                    -- e.g. "limxsdk", "mbl/robot_data"
  source_uri TEXT NOT NULL,
  symbol TEXT NOT NULL,                 -- class/struct/method/function name
  kind TEXT NOT NULL,                   -- class, struct, enum, method, function, typedef
  signature TEXT,                       -- formatted prototype
  docstring TEXT                        -- leading comment block, if any
);
CREATE INDEX api_symbols_name ON api_symbols(symbol);
```

Every typed row exposes `source_uri`, so any structured answer can also produce the same `oli-corpus://` citation the existing tools emit.

## Extractors

Each extractor is a standalone Python script, runs idempotently, writes its own table rows, and emits markdown chunks into the existing FTS pipeline.

| Script | Input glob | Tables populated | FTS chunks emitted |
|---|---|---|---|
| `extract_urdf.py` | `**/*.urdf`, `**/*.srdf` | `robots`, `joints`, `links` | Per-robot markdown table summary |
| `extract_packages.py` | `**/package.xml` | `packages`, `pkg_deps` | One `package: <name>` section per pkg |
| `extract_launch.py` | `**/launch/*.xml`, `**/*.launch` | `launch_nodes`, `node_topics`, `node_params` | Markdown-rendered launch tree per file |
| `extract_headers.py` | `mbl/include/**/*.{h,hpp}`, `share/*/include/**/*.{h,hpp}`, `limxsdk/include/**/*.h` | `api_symbols` | Per-header markdown with code blocks |
| `extract_configs.py` | `share/**/*.yaml`, `share/**/*.json` | *(none — flat FTS only)* | Per-file markdown of namespaced key→value listings |
| `extract_sdk_joint_order.py` | `mbl/include/robot_data/*.h` (and adjacent) | Backfills `joints.sdk_idx` | *(none)* |

The SDK joint-order extractor is intentionally separate because it's the riskiest: it needs to locate a `static const std::vector<std::string>` or equivalent enum in the headers. If the constant is not found for a given robot, `joints.sdk_idx` stays NULL and the `sdk_joint_order()` tool returns an error with a clear message. We never silently guess the mapping.

## MCP tool surface

Existing tools (unchanged): `search`, `get_section`, `cite`, `list_docs`.

New tools:

| Tool | Signature | Returns |
|---|---|---|
| `robots()` | — | List of `{robot_id, source_uri, description}` |
| `joints(robot_id)` | `robot_id: str` | Ordered list of joint rows for that robot, with `urdf_idx` and (when resolved) `sdk_idx` |
| `links(robot_id)` | `robot_id: str` | List of link rows including mass, COM, inertia, and mesh URIs |
| `sdk_joint_order(robot_id)` | `robot_id: str` | Ordered list of joint names matching the SDK's `q`/`dq`/`tau` array, or an error if the order constant was not recoverable |
| `pkg_info(name)` | `name: str` | Package metadata + dependency edges (both directions) |
| `nodes(scope, scope_kind)` | `scope: str`, `scope_kind: "launch" \| "package"` | Nodes from a launch file or all launch files in a package |
| `topics(node?, kind?)` | optional filters | Topic-graph rows, citing the launch file that declares them |
| `find_symbol(query, kind?)` | `query: str`, optional `kind` filter | API symbol rows matching the query, ordered by lib then name |
| `raw_file(source_uri)` | `source_uri: str` | Raw text (or base64 if binary) of the file at that URI. Required for URDF→USD conversion, STL retrieval, YAML inspection. |

`raw_file` is the universal escape hatch. Every typed answer's `source_uri` becomes actionable through it.

## Reindex pipeline

One CLI script orchestrates the full rebuild:

```
python _research/extract.py                  # existing 3 LimX webpages
python _research/extract_tarball.py          # NEW: walks all three vendored roots,
                                             #      calls extract_urdf/packages/launch/
                                             #      headers/configs/sdk_joint_order
python _research/build_index.py              # extended: builds FTS + typed tables
```

`extract_tarball.py` checks the vendored tarball's `manifest.yaml` against on-disk hashes before extracting. If the manifest is missing or stale, it errors with a one-line instruction to re-download from LimX.

## Error handling

- **Missing source root**: extractor logs a warning and skips that root. Other roots still ingest. The corpus reports which roots were available at index time via `list_docs` output.
- **Malformed URDF**: skipped with a structured error logged. The robot does not appear in `robots()`. Other robots still ingest.
- **Unresolved `sdk_idx`**: `joints.sdk_idx` stays NULL. `sdk_joint_order()` returns a clear error rather than guessing.
- **Header parse failure**: that symbol is skipped, the rest of the file is still processed.
- **`raw_file` on missing URI**: returns a structured error with the URI it tried, so the caller knows whether to re-ingest or fix the citation.

## Testing strategy

Each extractor has unit tests on small fixtures:

- `tests/extractors/test_urdf.py` — three fixture URDFs (simple, with-mimic, malformed). Verifies joint ordering, axis parsing, limit parsing, mesh URI resolution.
- `tests/extractors/test_packages.py` — package.xml with multiple dep kinds.
- `tests/extractors/test_launch.py` — launch file with includes, remaps, params.
- `tests/extractors/test_headers.py` — one struct, one class with methods, one typedef.
- `tests/extractors/test_sdk_joint_order.py` — header with a known constant; header without one (should NULL gracefully).

Integration test: run the full pipeline on a fixture that mirrors the tarball structure at small scale (one robot, two packages, two launch files, three headers), assert the database tables match expected snapshots.

## What this design intentionally does not solve

- **Wire-level topic schemas** — no `.msg`/`.srv` shipped, so the runtime probe (`probe_contract.py`) remains the only path. Tracked as a separate future workstream.
- **Cross-version diffing** — the URI scheme supports multiple versions but the tools don't yet query across them.
- **Indexing the vendored mujoco-sim simulator code itself** — only the `limxsdk-lowlevel/include/` slice is ingested; `simulator.py` and the rest of the harness stay out. Can be added later by extending the source-roots list.
- **Ingesting `oli_mcp_server` (LimX's own MCP)** — out of scope. Worth a separate investigation later because it may collide or compose with this corpus.

## Open invariants the implementer must verify

- The `urdf_idx` exposed by `joints()` MUST equal the position of `<joint>` in the URDF excluding `type="fixed"` joints, matching the convention IsaacLab uses when building articulations from URDF. The URDF extractor enforces this and documents it in the extractor docstring.
- `joints.sdk_idx`, when resolved, MUST equal the index of that joint name in the SDK's canonical array. If the SDK header changes this order, the extractor must re-run and downstream consumers must be notified.
- `raw_file` MUST NOT follow symlinks outside the vendored source roots. Extractor validates each indexed path is canonical and resides under one of the three known roots before storing the URI.
