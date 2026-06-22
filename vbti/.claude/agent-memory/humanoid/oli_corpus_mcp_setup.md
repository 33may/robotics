---
name: oli-corpus-mcp-setup
description: MAY-139 Oli docs corpus uses hum conda env, local MCP, FTS/vector/hybrid search, and citation-strict workflow
metadata:
  type: project
  origin: may-139-oli-docs-corpus implementation chat
---

Oli documentation corpus lives in `/home/may33/projects/ml_portfolio/robotics/humanoid/docs/oli-corpus/`.

Key implementation details:
- Python environment is conda env `hum` at `/home/may33/miniconda3/envs/hum`, not the older temporary `docs/oli-corpus/.venv`.
- Repo-level dependencies are tracked in `humanoid/requirements.txt`; MCP package is installed editable from `humanoid/docs/oli-corpus/server`.
- MCP executable path is `/home/may33/miniconda3/envs/hum/bin/oli-corpus-mcp`.
- Index builder is `docs/oli-corpus/_research/build_index.py`; it writes `index/corpus.sqlite` for FTS5 chunks and `index/vectors.npz` for normalized local embeddings.
- Embedding model is `sentence-transformers/all-MiniLM-L6-v2`; vector search is local and should not use hosted APIs.
- MCP `search` supports `mode="fts"` by default, plus `mode="vector"` and `mode="hybrid"`; all results keep `oli-corpus://...` citations and `layer` tags.
- Agents must cite Oli factual claims with `oli-corpus://...` or explicitly state no source was found.

Registration notes:
- Claude Code MCP registry is user-scoped with `claude mcp add --scope user oli-corpus-mcp -- /home/may33/miniconda3/envs/hum/bin/oli-corpus-mcp`; `claude mcp list` showed it connected. It was initially local/project-scoped, which made it invisible from Claude sessions launched outside the robotics git root.
- OpenCode/August config is `~/.config/opencode/opencode.jsonc` and uses OpenCode's `mcp.<name>.type="local"` shape with `command` as an array.
- During this chat, no `aug` or `opencode` CLI binary was on shell `PATH`, so OpenCode live tool listing could not be verified from the terminal even though the config entry was written.

How to apply:
- For future humanoid Oli questions, prefer MCP/corpus search over web search.
- If vector search fails, check `index/vectors.npz` and rebuild with `conda run -n hum python docs/oli-corpus/_research/build_index.py`.
- If OpenCode/August does not show the server, restart OpenCode/August first because config is loaded on startup.
