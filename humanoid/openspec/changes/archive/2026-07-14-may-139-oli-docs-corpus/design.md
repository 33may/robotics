## Context

`oli-docs-corpus` is the AI-accessible knowledge layer for the Oli humanoid stack. It exists because I work continuously with AI coding agents and need them grounded in real Oli documentation in every session — not paraphrased from memory, not pasted into context manually, not lost when sessions reset.

Three official LimX docs (Quick Start, User Manual, SDK Guide) are the initial content. Two research arms (see `humanoid/docs/oli-corpus/_research/`) settled the previously-open questions:

- **Extraction** (`extraction_research.md`): the LimX docs site is Next.js SSR; plain HTTP + BeautifulSoup-clean of `.md-editor-preview` + pandoc gfm + a language-fence post-pass gives high-fidelity markdown. Verified end-to-end on all three docs (7/7 fidelity checks pass).
- **Serving** (`mcp_serving_research.md`): context7's parser / index / crawler are closed source and its tool surface is snippet-search-only — wrong fit for a citation-strict corpus. A small custom stdio MCP server over markdown + SQLite FTS5 plus a local vector sidecar is still compact, gives us full control of chunking and citation, and is reusable for future docs (vbti, SO-ARM, additional humanoid hardware).

Three consumer types must be served by the same corpus:

1. **Interactive Claude / Cursor sessions** — targeted, citable lookups.
2. **Long-running research / planning tasks** (e.g. MAY-137 analysis) — broad cross-doc sweeps.
3. **Human (me) skimming** — readable markdown in editor / Obsidian.

## Goals / Non-Goals

**Goals**

- Repo-local, regeneratable corpus that any AI agent can query without setup.
- High-fidelity markdown source layer produced deterministically from upstream HTML.
- Local stdio MCP server exposing `list_docs`, `search`, `get_section`, `cite`.
- Strict layering: upstream sources are regenerated, never hand-edited; our notes live in a separate folder.
- Drift detection when LimX updates docs.

**Non-Goals**

- No public docs site.
- No LLM translation of Chinese-only content in this change — gap-logged for follow-up.
- No SDK wrappers (e.g. Python bindings around limxsdk) — separate future capability.
- No hosted / remote MCP — local stdio only.
- No hosted embedding service — semantic search must be local, reproducible, and optional at query time.

## Decisions

### D1. Folder layout

```
humanoid/docs/oli-corpus/
  sources/                  # extract.py output — regeneratable, never hand-edited
    LimX_EDU_Quick_Start_Guide.md
    Oli_EDU_User_Manual.md
    Oli_EDU_SDK_Development_Guide.md
    images/
    _meta/
      manifest.yaml         # per-doc: source_url, fetched_at, raw+clean sha256
      extraction_log.md
  index/                    # built by build_index.py
    corpus.sqlite           # FTS5 over chunks + vector metadata
    vectors.npz             # local embedding matrix aligned to chunk ids
  notes/                    # our analysis — never mixed into sources/
  source_map.md             # sections ↔ MAY-137-style questions
  gaps.md                   # missing / unclear / Chinese-only / blocked
  README.md                 # how AI agents are expected to consume the corpus
  server/                   # custom MCP server
    oli_corpus_mcp/
      __main__.py
      tools.py
      index.py
    pyproject.toml
  _research/                # research artifacts kept for traceability
    extract.py
    build_index.py
    extraction_research.md
    mcp_serving_research.md
```

Migration: rename existing `humanoid/docs/limx_source_docs/` → `humanoid/docs/oli-corpus/`, `clean/` → `sources/`.

### D2. Extraction toolchain

`extract.py` (already built, verified): fetch SSR HTML with gzip handling, BeautifulSoup-clean of `.md-editor-preview`, pandoc gfm, language-fence post-pass driven by `<code class="language-X">` order, image localization (dedup by sha1). Custom HTML-table → GFM-table fallback for tables containing `<br>` (pandoc drops to raw HTML otherwise). Re-runnable; caches raw HTML at `_research/raw/<doc_id>.html` for `--no-fetch` reuse.

### D3. Chunking unit

Section as primary chunk (heading-bounded by `## N` or `### N.M` etc., everything until the next same-or-higher heading). Sub-chunk any section exceeding ~500 tokens on paragraph boundaries; sub-chunks carry `?part=N` in citations.

Rationale: keeps citations human-meaningful (`#3.3` maps to "3.3 MCP Tool Interface Description") while bounding chunk size so FTS5 recall stays sharp on long sections.

### D4. Serving — local stdio MCP server `oli-corpus-mcp`

Python stdio server, MCP protocol, started per Claude / Cursor session via MCP config. Tool surface:

- `list_docs()` → `[{doc_id, title, section_count, fetched_at}]`
- `search(query, top_k=10, doc_id=None, include_notes=False, mode="fts")` → ranked chunks with snippets and citations. `mode="fts"` uses SQLite FTS5 BM25. `mode="vector"` uses local embeddings. `mode="hybrid"` combines both result sets with deterministic rank fusion.
- `get_section(doc_id, section, part=None)` → full chunk markdown (with images resolved to local paths)
- `cite(doc_id, section, part=None)` → canonical citation string and round-trippable file path

No daemon, no remote endpoint, no auth. Implementation: stdlib + `mcp` + `sqlite3` + a lightweight markdown reader (`mistune`) + local embedding dependencies pinned in the server package.

### D4a. Vector search

The indexer builds a local embedding sidecar for every chunk using a pinned sentence-transformers model. The default model is `sentence-transformers/all-MiniLM-L6-v2` unless implementation constraints require a smaller pinned local model. Embeddings are normalized and stored in `index/vectors.npz` with chunk ids aligned to `chunks.id` in `corpus.sqlite`.

`search(..., mode="vector")` embeds the query locally, computes cosine similarity against the stored matrix, and returns the same result shape and citation contract as FTS search. `search(..., mode="hybrid")` runs both FTS and vector search and merges scores by reciprocal-rank fusion so exact technical terms and paraphrased questions both work.

The vector layer SHALL NOT call external APIs at query time. If the embedding model is missing, the server must fail clearly for `mode="vector"` / `mode="hybrid"` and continue serving `mode="fts"`.

### D5. Citation contract

Canonical form: `oli-corpus://<doc_id>#<section_number>[?part=N]`
Examples: `oli-corpus://sdk-guide#3.3` , `oli-corpus://sdk-guide#3.3?part=2`

`cite()` returns both the URI and the resolved repo-relative path (`humanoid/docs/oli-corpus/sources/Oli_EDU_SDK_Development_Guide.md` + section anchor). The URI is the agent-facing form; the path is for human / IDE follow-through.

Doc IDs are short stable slugs: `quick-start`, `user-manual`, `sdk-guide`.

### D6. Refresh / drift detection

`manifest.yaml` records raw HTML sha256 and cleaned-md sha256 per doc. `extract.py --check` fetches current upstream HTML, compares hashes, exits non-zero on drift without modifying files. Cron / manual trigger. If raw HTML differs but cleaned md doesn't, treat as cosmetic upstream change — no action needed.

### D7. Strict source / notes separation

`sources/` is *only* written by `extract.py`. Any analysis or interpretation goes in `notes/` with frontmatter pointing at the source section(s) it interprets. The MCP server tags search results with their origin layer (`source` vs `note`) so agents know what is upstream truth vs our interpretation. Default `search()` queries `source` only; `search(... include_notes=True)` opts in.

### D8. MCP server lives in this change (not a separate one)

Scope decision: build extraction + sources + index + server together. The corpus has no operational value as files-only; the MCP server is the user-facing interface. Splitting would leave the "AI-native" goal half-done.

## Open Questions

### OQ1. Chinese-only sections

Some LimX content may be Chinese-only (the joint-index page already shows `复制代码` markers, and Section 2 Remote Controller refers out). Resolution path: complete extraction → survey `sources/` for residual Chinese → decide between (a) keep verbatim + `<!-- TODO: translate -->`, (b) LLM-translate with explicit `<!-- translation:llm -->` block, (c) gap-log only. Decision needed before `source_map.md` is finalized.

### OQ2. `notes/` schema

Frontmatter fields (`source_refs`, `confidence`, `author`, `date`), file naming convention, link-back format. Defer until first note is written — premature schema decisions tend to be wrong.

### OQ3. Index rebuild trigger

Manual `python build_index.py`, or auto-rebuild when MCP server detects `sources/` mtime newer than `corpus.sqlite`? Lean toward manual — predictable, no surprise rebuilds mid-session.

### OQ4. Multi-corpus future

If we later add `vbti-docs-corpus`, `so-arm-docs-corpus`, does each get its own MCP server, or one server with multi-corpus support? Out of scope here, but D5 citation URI scheme is designed so a single server can host many corpora later (`oli-corpus://...` vs `so-arm-corpus://...`).

## Risks / Trade-offs

- **[Risk] MCP server maintenance burden.** Custom server = ours to fix. **Mitigation:** keep it small (<300 LoC), pin to MCP Python SDK, no embeddings / vectors in v1.
- **[Risk] Vector dependency size / model availability.** Local embeddings add download and storage overhead. **Mitigation:** keep FTS5 as default and make vector/hybrid modes explicit; fail clearly when the model is unavailable.
- **[Risk] Semantic false positives.** Vector search can retrieve plausible but wrong neighbors. **Mitigation:** every result still cites exact chunks; agents must inspect cited sections before making factual claims.
- **[Risk] Sub-chunk citation ambiguity.** `?part=2` is brittle if section structure changes between extractions. **Mitigation:** index records section sha; if sha changes between runs, `cite()` flags it.
- **[Risk] Drift-detection noise.** Hash on raw HTML catches cosmetic upstream changes (whitespace, tracking pixels). **Mitigation:** also compare cleaned-md sha; surface only meaningful drift.
- **[Risk] Naming collision with existing `humanoid/docs/limx_source_docs/`.** Migration is a rename — straightforward, but any external references break. **Mitigation:** grep repo for `limx_source_docs` before migration; update references in same commit.

## Migration Plan

1. `git mv humanoid/docs/limx_source_docs humanoid/docs/oli-corpus` and `git mv humanoid/docs/oli-corpus/clean humanoid/docs/oli-corpus/sources`.
2. Update `extract.py` output paths and any references in `_meta/manifest.yaml`.
3. Author `README.md` with the AI-agent consumption protocol (cite via MCP, never paraphrase, prefer corpus over web).
4. Implement `build_index.py` — read `sources/`, chunk per D3, write `index/corpus.sqlite` (FTS5) and `index/vectors.npz` (local embeddings).
5. Implement `oli-corpus-mcp` server (tools per D4) and add a `pyproject.toml` for installation.
6. Document Claude Code MCP config snippet in `README.md`.
7. Author `source_map.md` and `gaps.md` from actual extracted content.

No rollback needed — content addition + new server, no code is depending on this yet.
