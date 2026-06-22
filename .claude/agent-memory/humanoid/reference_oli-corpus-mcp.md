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

**Tool surface:** `list_docs`, `search` (FTS), `get_section`, `cite`.

**Available docs (as of 2026-06-17):**
- `quick-start` — LimX EDU Quick Start Guide (1 section)
- `sdk-guide` — Oli EDU SDK Development Guide (358 sections)
- `user-manual` — Oli EDU User Manual (72 sections)

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
