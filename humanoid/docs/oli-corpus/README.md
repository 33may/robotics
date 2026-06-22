# Oli Documentation Corpus

This corpus is the repo-local source of truth for Oli / LimX documentation used by AI agents.

## Contract

Agents MUST cite any Oli factual claim with an `oli-corpus://...` URI obtained from the MCP server, or explicitly state that no source was found. Prefer this corpus over web search for Oli-specific facts.

Files under `sources/` are regenerated from official LimX documentation and must not be hand-edited. Hand-written analysis belongs under `notes/` and is excluded from default search unless `include_notes=true` is requested.

## Layout

- `sources/` — generated upstream markdown, localized images, manifest, extraction log.
- `_research/extract.py` — source extractor and drift checker.
- `_research/build_index.py` — SQLite FTS5 index builder.
- `index/corpus.sqlite` — generated search index.
- `server/` — local stdio MCP server exposing `list_docs`, `search`, `get_section`, and `cite`.
- `source_map.md` — MAY-137 questions mapped to source citations.
- `gaps.md` — missing, unclear, blocked, or Chinese-only source gaps.
- `notes/` — curated analysis, separate from upstream source truth.

## Build

```bash
conda create -n hum python=3.11
conda run -n hum python -m pip install -r requirements.txt
conda run -n hum python -m pip install -e docs/oli-corpus/server
conda run -n hum python docs/oli-corpus/_research/extract.py --no-fetch
conda run -n hum python docs/oli-corpus/_research/build_index.py
```

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
