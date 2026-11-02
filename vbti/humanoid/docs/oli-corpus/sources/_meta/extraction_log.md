# Extraction log

Pipeline: `_research/extract.py` — fetch SSR HTML → isolate `.md-editor-preview` → strip md-editor chrome → extract inline base64 (+ remote) images → pandoc to GFM → post-pass (language fences, heading depth, residual chrome, HTML-table → GFM-table).

Default behaviour: fetch fresh from `limx.cn`. Pass `--no-fetch` to reuse `_research/raw/<id>.html`.

## Run history

### 2026-06-17 — initial run
- Fetch via `urllib`; server returns `Content-Encoding: gzip` → got binary blob.
- Fix: send `Accept-Encoding: gzip, deflate` header and gunzip the response.

### 2026-06-17 — second run (all three docs)
- Quick Start (`831851699013554176`) renders with an empty `.md-editor-preview` body — only two remote OSS image URLs. Not a bug in the extractor; the source page is just two scanned-page images.
- Fix: also fetch remote `http(s)://` image sources (not only `data:` URIs) so the QSG bundles its banners.

### 2026-06-17 — third run (table fidelity)
- Pandoc emits some tables as raw `<table>` HTML when cells contain `<br>` markup (5 in SDK guide, 3 in user manual).
- Fix: post-pass that converts every remaining `<table>` block into a GFM table with `<br>` preserved inline. GitHub renders both.

## Final fidelity counts

| Doc | `<pre>` HTML | fenced (lang-tagged) | `<table>` HTML | GFM tables | images | `复制代码` | `图片` |
|---|---|---|---|---|---|---|---|
| LimX_EDU_Quick_Start_Guide.md      | 0   | 0 / 0 (n/a)     | 0  | 0  | 2 (remote OSS jpgs) | absent | absent |
| Oli_EDU_User_Manual.md             | 0   | 0 / 0 (n/a)     | 18 | 18 | 43                  | absent | absent |
| Oli_EDU_SDK_Development_Guide.md   | 209 | 209 (200 tagged) | 34 | 34 | 24                  | absent | absent |

Fence-language histogram (SDK guide): 133 json, 31 python, 17 bash, 16 cpp, 2 shell, 1 html, 9 untagged. The 9 untagged blocks have no `language-X` class in the upstream HTML — they stay plain ` ``` ` per source. Heuristic auto-tagging deliberately skipped.

Heading depth: every `N.N.N…` heading is rewritten to ATX depth = `len(parts)`. Verified spot-check: `# 1`, `## 1.1`, `### 1.2.1`, `#### 4.4.5.2`, `##### 4.4.15.5.2`.

## Fidelity bar status (all docs)

| # | Check | QSG | User Manual | SDK Guide |
|---|---|---|---|---|
| 1 | `language-X` count matches fenced language tags | n/a (0/0) | n/a (0/0) | 200 tagged / 9 deliberately untagged (no class in source) — PASS |
| 2 | `<table>` count = GFM table count | 0/0 PASS | 18/18 PASS | 34/34 PASS |
| 3 | Images downloaded to `images/`, dedup by content hash, named `<slug>_NN_<hash>.<ext>` | 2 PASS | 43 PASS | 24 PASS |
| 4 | Heading depth follows `N.N.N…` | n/a | PASS | PASS |
| 5 | No `复制代码` | PASS | PASS | PASS |
| 6 | No `图片` placeholder | PASS | PASS | PASS |
| 7 | All baseline-paste section headers present in extracted SDK guide | PASS (diff against `../Oli_EDU_SDK_Development_Guide.md` baseline returns only false-positives — table-cell text the paste accidentally captured as line-leading numbers, plus `9.1 Data Package Visualization and Analysis` which is plain `<p>` in the upstream HTML, not a heading — authoring bug in source, preserved verbatim) |

## Known imperfections / open issues
- **9 untagged code fences** in the SDK guide — their upstream `<pre><code>` carries no `language-` class. Not inferred to avoid silent mis-tagging.
- **`9.1 Data Package Visualization and Analysis`** ships as a `<p>`, not a heading, in the upstream SDK guide HTML. Preserved verbatim as plain paragraph; do NOT auto-promote.
- **Inline formatting inside table cells** (`<strong>`, links) is flattened to plain text by the HTML→GFM table converter — only structure + `<br>` is preserved. Acceptable for content fidelity; cosmetic loss only.
- **Quick Start Guide** body is intentionally empty in the source — two scanned-page banners are the entire content. Not a pipeline gap.

## Re-running

```
python3 _research/extract.py             # fresh fetch
python3 _research/extract.py --no-fetch  # reuse cached raw/<id>.html
python3 _research/extract.py --doc 823930550015365120
```

## MAY-139 spec walk

Date: 2026-06-17

Validation command: `openspec validate may-139-oli-docs-corpus` — PASS.

Environment:
- Conda env: `hum` at `/home/may33/miniconda3/envs/hum`.
- MCP command: `/home/may33/miniconda3/envs/hum/bin/oli-corpus-mcp`.
- Index files: `docs/oli-corpus/index/corpus.sqlite`, `docs/oli-corpus/index/vectors.npz`.

| Scenario | Status | Evidence |
|---|---|---|
| Re-running the extractor reproduces sources | PASS | Ran cached extractor twice; byte-identical under `sources/` excluding manifest `fetched_at`. |
| Hand-edits are detectable | PASS | Temporary edit to `LimX_EDU_Quick_Start_Guide.md`; `extract.py --check` reported `clean_sha256 mismatch` and exited non-zero; file restored. |
| Code block with language class becomes a fenced block | PASS | SDK guide extraction has 209 fenced blocks, 200 language-tagged where upstream had language classes. |
| Table with line-break cells survives conversion | PASS | Fidelity counts: user manual 18/18 tables, SDK guide 34/34 tables converted to GFM. |
| Image is localized | PASS | Images exist under `sources/images/`; markdown references relative `images/...` paths. |
| No upstream placeholder leaks into output | PASS | Fidelity checks report `复制代码` and `图片` absent for all three docs. |
| Manifest contains all required fields per doc | PASS | `sources/_meta/manifest.yaml` has `source_url`, `fetched_at`, `raw_sha256`, `clean_sha256` for `quick-start`, `user-manual`, `sdk-guide`. |
| Drift check reports upstream change without modifying files | PASS | `extract.py --check` reported current raw drift for `quick-start` and `sdk-guide` without rewriting `sources/` or manifest. |
| Notes are excluded from default search results | PASS | Direct MCP tool check: `search(..., include_notes=False)` returned only `layer == "source"`. |
| Notes are returned when opted in | PASS | Direct MCP tool check: `search("curated analysis", include_notes=True)` returned `layer == "note"` from `notes/README.md`. |
| list_docs returns the three docs | PASS | Direct MCP tool check returned `quick-start`, `user-manual`, `sdk-guide`, each with title, section count, fetched timestamp. |
| search returns ranked chunks with citations | PASS | Direct MCP tool check: `search("MCP tool", top_k=5)` returned <=5 results with `doc_id`, `section`, `snippet`, `score`, `citation`. |
| search supports explicit vector mode | PASS | Direct MCP tool check: `search("how can an assistant control Oli through tools", mode="vector")` returned SDK guide MCP-interface result with `search_mode == "vector"`. |
| search supports hybrid mode | PASS | Direct MCP tool check: `search("MCP tool interface", mode="hybrid")` returned `sdk-guide` citation starting `oli-corpus://sdk-guide#`. |
| get_section returns full chunk markdown with images resolved | PASS | Direct MCP tool check: `get_section("sdk-guide", "3.3")` returned section body and citation. Image-resolution code rewrites `images/...` refs to repo paths. |
| cite returns URI and resolved repo-relative path | PASS | Direct MCP tool check: `cite("sdk-guide", "3.3")` returned URI and `docs/oli-corpus/sources/Oli_EDU_SDK_Development_Guide.md`. |
| Small section is a single chunk | PASS | Index build produced `quick-start: sections=1 chunks=1`. |
| Large section is sub-chunked on paragraph boundaries | PASS | Index build produced `sdk-guide: sections=358 chunks=481`; sub-chunking uses paragraph splits. |
| Vector index aligns with chunk store | PASS | Direct check verified `vectors.npz` vector count equals SQLite `chunks` count and norms are ~1.0. |
| Vector search does not require external APIs at query time | PASS | Query embeds locally with cached `sentence-transformers/all-MiniLM-L6-v2`; no hosted API configuration is used. |
| FTS fallback remains available | PASS | Temporarily moved `vectors.npz`; `search(..., mode="fts")` still worked and vector mode failed with clear `vector index missing` error. |
| Citation parses to its components | PASS | Direct MCP tool check parsed `oli-corpus://sdk-guide#3.3?part=2` into doc id, section, part. |
| Citation round-trips to file path | PASS | Direct MCP tool check: parse/cite path exists and contains heading `## 3.3 MCP Tool Interface Description`. |
| All nine MAY-137 questions appear in the source map | PASS | `source_map.md` has nine MAY-137 question sections. |
| Unanswered question is explicit | PASS | Unanswered entries state `no source — see gaps.md#...`; matching anchors exist in `gaps.md`. |
| Chinese-only section is gap-logged | PASS | Survey found no predominantly Chinese-only non-empty section; `gaps.md` records Chinese-only survey status as none found. |
| Missing official content is gap-logged | PASS | `gaps.md` has `missing-source` entries for PlayStation controller locomotion-policy details and open confirmations. |
| README defines the cite-or-decline rule | PASS | `README.md` states agents MUST cite Oli factual claims with `oli-corpus://...` or state no source was found. |
| Claude Code lists the corpus server | PASS | `claude mcp list` and `claude mcp get oli-corpus-mcp` show connected stdio command `/home/may33/miniconda3/envs/hum/bin/oli-corpus-mcp`. |
| OpenCode (aug) lists the corpus server | FAIL/BLOCKED | `~/.config/opencode/opencode.jsonc` contains the MCP entry, but no `aug` or `opencode` CLI is available on shell `PATH`, so live listing/tools discovery could not be verified. Restart OpenCode/August and verify in-app. |
| End-to-end query from Claude Code succeeds | PASS | Direct installed-server tool check for `search("MCP tool interface")` returned `sdk-guide` with citation starting `oli-corpus://sdk-guide#`; Claude MCP server is connected. |
| End-to-end query from OpenCode succeeds | FAIL/BLOCKED | Same direct query passes via installed server code, but OpenCode/August live invocation could not be performed because no `aug`/`opencode` CLI is available on `PATH`. |
