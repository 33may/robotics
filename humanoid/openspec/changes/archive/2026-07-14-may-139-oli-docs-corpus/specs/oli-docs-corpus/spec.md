## ADDED Requirements

### Requirement: Sources are regeneratable from upstream

The corpus SHALL produce its source-layer markdown deterministically from the three official LimX documentation URLs (Quick Start, User Manual, SDK Guide). Hand-edits to files under `sources/` are forbidden; the only writer of `sources/` is the extraction script.

#### Scenario: Re-running the extractor reproduces sources
- **WHEN** the extractor is run twice in succession against the cached upstream HTML
- **THEN** the byte content of every file under `sources/` (excluding `_meta/manifest.yaml` `fetched_at` field) is identical between runs

#### Scenario: Hand-edits are detectable
- **WHEN** a file under `sources/` is modified by hand and the extractor is re-run with `--check`
- **THEN** the extractor reports the cleaned-md sha256 mismatch and exits non-zero without overwriting the file

### Requirement: Source markdown preserves structural fidelity

The corpus SHALL preserve code blocks (with language fences when the upstream HTML provides a `language-<X>` class), tables (as GitHub-flavoured Markdown tables), images (downloaded locally and referenced by relative path), and heading hierarchy (where each numeric prefix `N.M.K` maps to a markdown heading level).

#### Scenario: Code block with language class becomes a fenced block
- **WHEN** the upstream HTML contains `<pre><code class="language-python">...</code></pre>`
- **THEN** the extracted markdown contains a fenced block ` ```python ` with the same body

#### Scenario: Table with line-break cells survives conversion
- **WHEN** the upstream HTML contains a `<table>` with `<br>` inside cells
- **THEN** the extracted markdown contains a GFM table row whose cells preserve the `<br>` as inline markup

#### Scenario: Image is localized
- **WHEN** the upstream HTML contains an `<img src="<remote-or-base64>">` inside the doc body
- **THEN** the image file exists under `sources/images/` (deduplicated by sha1) and the markdown references it by relative path

#### Scenario: No upstream placeholder leaks into output
- **WHEN** the extractor runs against any of the three doc URLs
- **THEN** the resulting markdown contains neither the literal token `复制代码` nor the literal token `图片`

### Requirement: Manifest tracks provenance and enables drift detection

Each extracted document SHALL be accompanied by a manifest entry recording its upstream URL, fetch timestamp (ISO-8601 UTC), the sha256 of the raw upstream HTML, and the sha256 of the cleaned markdown.

#### Scenario: Manifest contains all required fields per doc
- **WHEN** the extractor completes successfully on all three docs
- **THEN** `sources/_meta/manifest.yaml` contains an entry per `doc_id` with keys `source_url`, `fetched_at`, `raw_sha256`, `clean_sha256`

#### Scenario: Drift check reports upstream change without modifying files
- **WHEN** `extract.py --check` is run and the upstream HTML has changed for at least one doc
- **THEN** the script prints which docs drifted with their previous and current `raw_sha256`, exits with non-zero status, and leaves all files under `sources/` and `_meta/manifest.yaml` unmodified

### Requirement: Upstream sources and curated notes are strictly separated

The corpus SHALL maintain a hard boundary between regenerated upstream content (`sources/`) and hand-written analysis (`notes/`). The MCP server SHALL tag every result with its origin layer.

#### Scenario: Notes are excluded from default search results
- **WHEN** an agent calls `search(query="...")` without an `include_notes` argument
- **THEN** every returned result has `layer == "source"` and no result references a file under `notes/`

#### Scenario: Notes are returned when opted in
- **WHEN** an agent calls `search(query="...", include_notes=true)`
- **THEN** results may include entries with `layer == "note"` and each note result references a file under `notes/`

### Requirement: Corpus is queryable via a local stdio MCP server

The corpus SHALL be served by a Python stdio MCP server `oli-corpus-mcp` exposing the tools `list_docs`, `search`, `get_section`, and `cite`.

#### Scenario: list_docs returns the three docs
- **WHEN** an agent calls `list_docs()`
- **THEN** the response contains exactly three entries with `doc_id` in `{"quick-start", "user-manual", "sdk-guide"}`, each carrying `title`, `section_count`, and `fetched_at`

#### Scenario: search returns ranked chunks with citations
- **WHEN** an agent calls `search(query="MCP tool", top_k=5)`
- **THEN** the response is a list of at most 5 entries, each containing `doc_id`, `section`, `snippet`, `score`, and a `citation` matching the URI scheme of the citation contract

#### Scenario: search supports explicit vector mode
- **WHEN** an agent calls `search(query="how can an assistant control Oli through tools", top_k=5, mode="vector")`
- **THEN** the response is a list of at most 5 entries, each containing `doc_id`, `section`, `snippet`, `score`, `citation`, and `search_mode == "vector"`
- **AND** at least one returned result references the SDK guide MCP interface section

#### Scenario: search supports hybrid mode
- **WHEN** an agent calls `search(query="MCP tool interface", top_k=5, mode="hybrid")`
- **THEN** the response is a list of at most 5 entries, each containing `doc_id`, `section`, `snippet`, `score`, `citation`, and `search_mode == "hybrid"`
- **AND** at least one returned result has `doc_id == "sdk-guide"` and a citation starting `oli-corpus://sdk-guide#`

#### Scenario: get_section returns full chunk markdown with images resolved
- **WHEN** an agent calls `get_section(doc_id="sdk-guide", section="3.3")`
- **THEN** the response contains the full markdown of section 3.3 with image references pointing at local paths under `sources/images/`

#### Scenario: cite returns URI and resolved repo-relative path
- **WHEN** an agent calls `cite(doc_id="sdk-guide", section="3.3", part=2)`
- **THEN** the response contains both the canonical URI `oli-corpus://sdk-guide#3.3?part=2` and the repo-relative file path of the source markdown

### Requirement: Chunks are heading-bounded with size-based sub-chunking

The corpus SHALL chunk source documents by markdown heading (each heading and its descendant content form one chunk) and SHALL sub-chunk any chunk exceeding approximately 500 tokens on paragraph boundaries. Sub-chunks SHALL share the parent chunk's section identifier and add a `part` index starting at 1.

#### Scenario: Small section is a single chunk
- **WHEN** the indexer processes a section whose body is below the size threshold
- **THEN** the chunk store contains exactly one row for that section with no `part` index

#### Scenario: Large section is sub-chunked on paragraph boundaries
- **WHEN** the indexer processes a section whose body exceeds the size threshold
- **THEN** the chunk store contains multiple rows sharing the same `doc_id` and `section`, each carrying a `part` index `1..N`, and no sub-chunk splits mid-paragraph

### Requirement: Local vector search is built from the same chunks

The corpus SHALL build local embeddings for the same chunk rows used by FTS search and SHALL use those embeddings only from local files at query time.

#### Scenario: Vector index aligns with chunk store
- **WHEN** the indexer completes successfully
- **THEN** `index/vectors.npz` exists and contains one normalized vector per indexed chunk id in `index/corpus.sqlite`

#### Scenario: Vector search does not require external APIs at query time
- **WHEN** an agent calls `search(query="...", mode="vector")`
- **THEN** the MCP server embeds the query locally and returns results without calling a hosted embedding API

#### Scenario: FTS fallback remains available
- **WHEN** the vector model or vector sidecar is unavailable
- **THEN** `search(query="...", mode="fts")` still works
- **AND** `search(query="...", mode="vector")` fails with a clear vector-index/model error rather than silently falling back

### Requirement: Citation contract is canonical and round-trippable

Every citation SHALL follow the form `oli-corpus://<doc_id>#<section>[?part=N]` and SHALL be reversible to a repo-relative source-file path via the `cite` MCP tool.

#### Scenario: Citation parses to its components
- **WHEN** any returned `citation` string is parsed
- **THEN** it yields a `doc_id`, a `section`, and an optional `part` integer, with no other fields

#### Scenario: Citation round-trips to file path
- **WHEN** `cite(doc_id, section, part)` is called with values from any prior `search` or `get_section` result
- **THEN** the returned repo-relative path exists on disk under `sources/` and the section heading is present in that file

### Requirement: Source map links sections to consumer questions

The corpus SHALL maintain a `source_map.md` that links upstream sections to specific consumer questions, seeded with the nine open questions from MAY-137.

#### Scenario: All nine MAY-137 questions appear in the source map
- **WHEN** `source_map.md` is opened
- **THEN** the file contains at least nine question entries matching the MAY-137 questions and each entry lists zero or more `oli-corpus://...` citations pointing at supporting sections

#### Scenario: Unanswered question is explicit
- **WHEN** a MAY-137 question has no supporting section in the corpus
- **THEN** its `source_map.md` entry states "no source — see gaps.md#<anchor>" and a matching anchor exists in `gaps.md`

### Requirement: Gaps are tracked, not silently dropped

The corpus SHALL maintain a `gaps.md` listing missing, unclear, blocked, or Chinese-only content that prevents answering known consumer questions.

#### Scenario: Chinese-only section is gap-logged
- **WHEN** a non-empty source section consists predominantly of non-ASCII script after extraction
- **THEN** an entry exists in `gaps.md` referencing the section's `oli-corpus://...` citation with category `chinese-only`

#### Scenario: Missing official content is gap-logged
- **WHEN** a MAY-137 question cannot be answered from any extracted section
- **THEN** an entry exists in `gaps.md` referencing the question with category `missing-source`

### Requirement: README defines the AI-agent consumption protocol

The corpus SHALL ship a `README.md` instructing AI agents to access Oli documentation through the MCP server, to cite via the canonical URI, and to prefer the corpus over web search for Oli-specific facts.

#### Scenario: README defines the cite-or-decline rule
- **WHEN** `README.md` is read
- **THEN** it states that agents MUST cite any Oli factual claim with an `oli-corpus://...` URI obtained from the MCP server, or explicitly state that no source was found

### Requirement: MCP server is installed and reachable from Claude Code and OpenCode (aug)

The MCP server SHALL be installed on the development machine and registered with both Claude Code and OpenCode (aug) such that an interactive session in either tool can call the corpus tools without per-session setup.

#### Scenario: Claude Code lists the corpus server
- **WHEN** Claude Code MCP servers are listed
- **THEN** `oli-corpus-mcp` appears in the list as a stdio server with a working command line and the four tools `list_docs`, `search`, `get_section`, `cite` are discoverable

#### Scenario: OpenCode (aug) lists the corpus server
- **WHEN** OpenCode (aug) MCP servers are listed
- **THEN** `oli-corpus-mcp` appears in the list as a stdio server and the four tools are discoverable

#### Scenario: End-to-end query from Claude Code succeeds
- **WHEN** a Claude Code session calls `search(query="MCP tool interface")` via the corpus MCP server
- **THEN** at least one result is returned whose `doc_id == "sdk-guide"` and whose `citation` starts with `oli-corpus://sdk-guide#`

#### Scenario: End-to-end query from OpenCode succeeds
- **WHEN** an OpenCode session calls `search(query="MCP tool interface")` via the corpus MCP server
- **THEN** at least one result is returned whose `doc_id == "sdk-guide"` and whose `citation` starts with `oli-corpus://sdk-guide#`
