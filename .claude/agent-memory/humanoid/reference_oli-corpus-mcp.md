---
name: reference-oli-corpus-mcp
description: oli-corpus-mcp server — queryable corpus of LimX/Oli docs with citation URIs; always use when working on the Oli robot
metadata:
  node_type: memory
  type: reference
  originSessionId: 46565123-02ae-4973-8c67-3b95734aade3
---

`oli-corpus-mcp` is the MCP server exposing official LimX/Oli documentation as a queryable, citation-ready corpus. Project-local; appears as `mcp__oli-corpus-mcp__*` once the project MCP config is loaded.

**Always use it when working on the Oli robot.** Any question about Oli SDK, hardware, operating modes, safety, control surfaces, or assets — go to this corpus first, before websearch and before re-reading PDFs. Cite back with `oli-corpus://...` URIs so answers are traceable.

**Tool surface (expanded 2026-06-22):**

Free-text: `list_docs`, `search` (FTS), `get_section`, `cite`.

Structured (typed-table backed):
- `robots()` — 26 robot variants from URDFs (HU_D04, HU_L01, HU_N01, DA, UB, arm-only, _rl variants)
- `joints(robot_id)` — ordered table: `urdf_idx`, `dof_idx`, `sdk_idx`, axes, limits, parent/child links
- `links(robot_id)` — mass, COM, inertia tensor, mesh URIs (visual + collision)
- `sdk_joint_order(robot_id)` — canonical joint-name order for SDK `q`/`dq`/`tau` arrays; raises if not resolved (HU_D04_01 + HU_D03_03 only, sourced from walk_param.yaml comments). NEVER silently guesses.
- `pkg_info(name)` — 148 ROS pkgs from tarball; deps + dependents both directions
- `nodes(launch_uri?, pkg?)` — launch-declared nodes (7 launch files indexed)
- `topics(node?, kind?, topic?)` — empty for now; MROS YAML launches don't declare topics
- `find_symbol(query, kind?)` — 397 C/C++ symbols (struct/class/enum/typedef/using) from limxsdk + mbl/include headers
- `raw_file(source_uri, max_bytes=1MB)` — universal escape hatch: utf-8 text or base64 binary for any indexed file

**Available docs (as of 2026-06-22):**
- `quick-start` — 1 section
- `sdk-guide` — 358 sections
- `user-manual` — 72 sections
- `oli-main-2.2.12` — 611 sections (URDFs + package.xml + launches + headers + configs from vendored Main Software tarball at `humanoid/vendor/oli-main-software-2.2.12/`)
- `limxsdk` — 14 sections (C++ headers from `humanoid/vendor/humanoid-mujoco-sim/limxsdk-lowlevel/include/limxsdk/`)
- `rl-deploy-python` — 9 sections (controller configs from `humanoid/vendor/humanoid-rl-deploy-python/`)

**Citation URI format:** `oli-corpus://<doc_id>#<section>[?part=N]` — e.g. `oli-corpus://sdk-guide#5.1?part=2`.

**Source markdown** under `docs/oli-corpus/sources/` in the humanoid repo (e.g. `Oli_EDU_SDK_Development_Guide.md`).

**Quirks (verified in session, fix-later items):**
- FTS is literal. Concrete domain nouns work (`mujoco`, `realsense`, `publishRobotCmd`, `websocket`); abstract phrasing returns `[]` (e.g. `"testing commands safely"` → empty).
- Don't include `.` in search queries — `"4.9.1"` triggers `fts5: syntax error near "."`. Quote-escape or rephrase.
- Top-level `get_section("sdk-guide", "4")` can fail with `section not found`; the chapter still resolves via `?part=N`. Possible parent-section indexing gap.

**Default workflow for any Oli question:**
1. `list_docs` to confirm what's available.
2. `search` with a concrete noun → ranked snippets with citation URIs.
3. `get_section` to pull the body for citation.
4. Quote with the `oli-corpus://...` URI so the answer is traceable.

Validated end-to-end on 2026-06-17 by answering all 9 [[tasks/may-137-explore-oli-sdk-and-control-interfaces|MAY-137]] research questions purely from corpus citations.

Related: [[oli-sdk-3-layer-architecture]], [[humanoid-oli-docs-before-sim]]
