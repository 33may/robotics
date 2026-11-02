# MCP serving research for oli-docs-corpus

## TL;DR

- **Recommendation: Option B — small custom MCP server over the in-repo corpus.** Hybrid (also PR Oli docs to context7) is optional and low-priority.
- Context7's parser, crawler, and index are **closed source**; only the thin MCP shim is open (MIT). You cannot self-host it, only point its hosted backend at a public GitHub repo. That kills control over chunking, refresh cadence, and citation format — the three properties this corpus is built around.
- A custom MCP server is small (~1–2 days): stdio MCP wrapping `ripgrep` + a SQLite FTS5 index over the markdown we already produce. Reusable for vbti, SO-ARM, and future hardware docs.

## How context7 actually works

### Architecture (from the public repo)

Repo: `upstash/context7` (MIT). Cloned to `/tmp/ctx7/context7`.

- `packages/mcp/src/index.ts` — only ~1k LoC stdio/HTTP MCP shim. It just calls `searchLibraries()` and `fetchLibraryContext()` from `lib/api.ts`, which hit `https://mcp.context7.com` / `https://context7.com` REST endpoints.
- `README.md:136` says explicitly: *"This repository hosts the MCP server's source code. The supporting components — API backend, parsing engine, and crawling engine — are private and not part of this repository."*
- So: **not self-hostable in any meaningful sense**. Running the open-source binary still requires hitting Upstash's hosted index.

### Ingestion model

From `docs/adding-libraries.mdx` and `docs/howto/private-sources.mdx`:

- Public flow: paste a **GitHub repo URL** into `context7.com/add-library`. Their crawler parses the repo, optionally guided by a `context7.json` at repo root (`folders`, `excludeFolders`, `excludeFiles`, `projectTitle`, `rules`). No `llms.txt` upload directly for public — they index from the repo.
- Refresh: automatic ("based on popularity"), or trigger via a GitHub Action on push.
- Private sources: GitHub/GitLab/Bitbucket/Confluence/OpenAPI — gated behind **Pro/Enterprise plan**. So a private Oli corpus is paid.

### Tool surface

Two MCP tools, that's it:

- `resolve-library-id(libraryName, query)` → list of `{id: "/org/project", description, snippets, score}`
- `query-docs(libraryId, query)` → returns concatenated snippets, each with a `Source:` URL header

### Live sample (real output)

`query-docs(/huggingface/lerobot, "how to load a dataset")` returned 5 snippets, each formatted as:

```
### Load RoboMME Dataset
Source: https://github.com/huggingface/lerobot/blob/main/docs/source/robomme.mdx
<prose>
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("lerobot/robomme")
```
```

So the chunking is **code-snippet-centric** with title + source URL + short prose + code. No section anchors, no structured navigation, no `list_docs` / table-of-contents tool. Good for "give me the API call" — bad for "walk me through the Oli startup procedure as a coherent doc".

### Contribution path

- Open: anyone with a public GitHub repo URL can submit. Lead time ~minutes to hours.
- Closed: cannot influence chunking strategy beyond `context7.json` include/exclude paths. Cannot enforce citation format. Refresh cadence is on their schedule unless you wire a GitHub Action.

## Option A: contribute to context7

- **Feasibility:** trivial if `oli-docs-corpus` lives in a public GitHub repo. Submit URL → indexed.
- **Cost we eat:** lose control of chunking (snippet-style only); lose control of citation format (`Source: <github URL>` only, no `(doc_id, section_anchor)`); refresh cadence is theirs; cannot add custom tools like `list_docs` or `cite`.
- **Privacy:** LimX docs are publicly published — fine. But making Oli notes/gaps/internal commentary public is a separate decision.
- **Net:** zero engineering cost, near-zero control. Useful as a public mirror for outside agents, not as the primary access path for our Claude Code / Cursor workflow.

## Option B: custom MCP server

### Reference architecture

Stdio MCP, ~1 file, Python (matches your stack) or TS. Reads the in-repo corpus directly.

Tool surface (critiqued):

- `list_docs()` → `[{doc_id, title, source_url, version, anchor_count}]`. Keep.
- `get_section(doc_id, anchor)` → markdown of that section verbatim. Keep — this is the citation primitive.
- `search(query, top_k=8, doc_id=None)` → SQLite FTS5 over section text; returns `[{doc_id, anchor, title_path, snippet, score}]`. Keep. Optional rerank later.
- `cite(doc_id, anchor)` → canonical citation string. **Drop as a tool, make it a convention** baked into every `get_section` / `search` response (already returns `doc_id`+`anchor`). Saves a tool call.
- Add: `get_outline(doc_id)` → flat heading tree. Cheap, useful for the "walk me through" case context7 fails at.

Storage:

- Source of truth: the markdown files in `humanoid/docs/oli-corpus/sources/`.
- Index: SQLite FTS5 built on demand from those files, cache-keyed by file mtime. No embeddings needed at this corpus size (~3 docs, tens of sections). Add embeddings later only if FTS misses.

Install path:

- Local stdio MCP, registered in `~/.claude/mcp.json` and `~/.cursor/mcp.json` per the standard config block. `uvx oli-docs-mcp` once published, or just `python -m oli_docs_mcp` from the repo during dev.
- No HTTP, no auth, no hosted dependency.

### Effort estimate

**Small** — 1 to 2 focused days:

- Day 1: parse markdown → section index (heading anchors already exist from your pandoc pass), SQLite FTS5 build script, three MCP tools wired up over the official Python MCP SDK.
- Day 0.5: config blocks for Claude Code + Cursor, a `make index` rule, smoke tests.

### What we reuse

- `modelcontextprotocol/python-sdk` — stdio server boilerplate; the `filesystem` and `git` reference servers (`modelcontextprotocol/servers`) are direct templates for "expose local content as MCP tools".
- The Anthropic skill examples for Claude Code (already on this machine) show the config wiring.
- Optional later: borrow context7's response format (title + source URL + body) for `search` results to keep agents comfortable.

## Option C: hybrid

Cheap and worth doing **after** Option B lands:

- Primary path: custom MCP server, local, authoritative, fast, citation-correct.
- Secondary path: push the cleaned markdown corpus to a public GitHub repo and submit to context7 for outside agents (and as a free mirror for yourself when working off a fresh machine). Add a `context7.json` to scope what they index. Maintenance cost ≈ zero once set up.

This costs nothing extra if the corpus is already going to live in a public repo, and gives you a fallback if the local MCP isn't running.

## Recommendation

**Build the custom MCP server (Option B).** Optionally mirror the corpus into context7 later (Option C) once Option B is stable.

Trade-off accepted: ~1–2 days of engineering and ongoing ownership of a tiny index/server, in exchange for: full control of chunking + citation format, sub-100ms latency, no hosted dependency, reusability for vbti/SO-ARM/future docs, privacy by default. Context7's hosted snippet-search UX does not match the "cite section anchors, prefer corpus over web, track gaps explicitly" contract in the proposal.

## Risks / open questions

- **Embeddings later?** FTS5 will probably be enough for a 3-doc corpus. If recall sags once GAPS.md and SDK examples grow, add sentence-transformers + sqlite-vec. Defer.
- **Multi-repo serving.** If we want one MCP server fronting Oli + vbti + SO-ARM corpora, decide early whether `doc_id` is globally unique or namespaced as `corpus:doc_id`. Recommend namespacing from day one.
- **MCP transport drift.** Claude Code's local stdio MCP wiring has shifted (plugin marketplace vs `~/.claude/mcp.json`). Pin a config example in the repo README.
- **Public mirror scope.** If we go hybrid, decide whether `GAPS.md` and internal commentary belong in the public mirror or stay private.
- **context7 closed backend.** If Upstash changes pricing/policy, public mirror could break — Option B is unaffected.
