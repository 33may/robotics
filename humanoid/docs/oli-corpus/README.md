# Oli Documentation Corpus

This corpus is the repo-local source of truth for Oli / LimX documentation used by AI agents.

## Contract

Agents MUST cite any Oli factual claim with an `oli-corpus://...` URI obtained from the MCP server, or explicitly state that no source was found. Prefer this corpus over web search for Oli-specific facts.

Files under `sources/` are regenerated from official LimX documentation and must not be hand-edited. Hand-written analysis belongs under `notes/` and is excluded from default search unless `include_notes=true` is requested.

## Layout

- `sources/` — generated upstream markdown, localized images, manifest, extraction log.
- `_research/extract.py` — webpage source extractor and drift checker.
- `_research/build_index.py` — SQLite FTS5 index builder; orchestrates all structured extractors.
- `_research/structured_schema.py` — DDL for typed tables (robots, joints, links, packages, pkg_deps, launch_nodes, node_topics, node_params, api_symbols, file_index).
- `_research/extract_urdf.py` — URDF/SRDF → `robots`, `joints`, `links`.
- `_research/extract_packages.py` — `package.xml` → `packages`, `pkg_deps`.
- `_research/extract_launch.py` — LimX MROS YAML launch files → `launch_nodes`.
- `_research/extract_headers.py` — C/C++ `.h`/`.hpp` → `api_symbols`.
- `_research/extract_sdk_joint_order.py` — comment-annotated `walk_param.yaml` → backfills `joints.sdk_idx`.
- `_research/extract_configs.py` — YAML/JSON configs → flat FTS chunks.
- `index/corpus.sqlite` — generated search index (FTS + typed tables).
- `server/` — local stdio MCP server (see "Tool surface" below).
- `source_map.md` — MAY-137 questions mapped to source citations.
- `gaps.md` — missing, unclear, blocked, or Chinese-only source gaps.
- `notes/` — curated analysis, separate from upstream source truth.

## Tool surface

Free-text (existing): `list_docs`, `search`, `get_section`, `cite`.

Structured (added 2026-06-22):

- `robots()` — list all robots indexed from URDFs
- `joints(robot_id, include_fixed=False)` — joint table; `urdf_idx` = declaration order, `dof_idx` = DoF sequential, `sdk_idx` = the SDK's `q`/`dq`/`tau` array position
- `links(robot_id)` — mass, COM, inertia tensor, visual/collision mesh URIs
- `sdk_joint_order(robot_id)` — canonical joint name list for the SDK array; raises if not resolved (never silently guesses)
- `pkg_info(name)` — package metadata + deps + dependents
- `nodes(launch_uri?, pkg?)` — launch-declared nodes
- `topics(node?, kind?, topic?)` — topic graph (empty for MROS YAML launches; reserved for ROS XML launches in future source roots)
- `find_symbol(query, kind?, limit=50)` — C/C++ symbol lookup with exact-match-then-substring fallback
- `raw_file(source_uri, max_bytes=1MB)` — return raw text (utf-8) or binary (base64) bytes for any indexed file

## Source roots

The structured extractors walk three vendored, LimX-authored source roots:

| doc_id | path |
|---|---|
| `oli-main-2.2.12` | `humanoid/vendor/oli-main-software-2.2.12/install/` (the unpacked Main Software tarball, gitignored) |
| `limxsdk` | `humanoid/vendor/humanoid-mujoco-sim/limxsdk-lowlevel/include/limxsdk/` |
| `rl-deploy-python` | `humanoid/vendor/humanoid-rl-deploy-python/` |

URI scheme: `oli-corpus://<doc_id>#<section>`. Section is the relative path of the source file within its root.

## Build

```bash
conda create -n hum python=3.11
conda run -n hum python -m pip install -r requirements.txt
conda run -n hum python -m pip install -e docs/oli-corpus/server
conda run -n hum python docs/oli-corpus/_research/extract.py --no-fetch
cd docs/oli-corpus
conda run -n hum python -m _research.build_index
```

(The build_index command MUST be run with `-m _research.build_index` because the orchestrator uses relative imports across the extractor package.)

Use drift detection without modifying files:

```bash
conda run -n hum python docs/oli-corpus/_research/extract.py --check
```

## Claude Code MCP Config

```json
{
  "mcpServers": {
    "oli-corpus-mcp": {
      "command": "/home/may33/miniconda3/envs/hum/bin/oli-corpus-mcp",
      "args": []
    }
  }
}
```

## OpenCode MCP Config

```json
{
  "mcp": {
    "oli-corpus-mcp": {
      "type": "local",
      "command": [
        "/home/may33/miniconda3/envs/hum/bin/oli-corpus-mcp"
      ],
      "enabled": true
    }
  }
}
```
