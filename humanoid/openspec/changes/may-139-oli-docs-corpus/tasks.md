## 1. Migration to oli-corpus layout

- [x] 1.1 `git mv humanoid/docs/limx_source_docs humanoid/docs/oli-corpus`
- [x] 1.2 `git mv humanoid/docs/oli-corpus/clean humanoid/docs/oli-corpus/sources`
- [x] 1.3 Update path references in `extract.py`, `_meta/manifest.yaml`, and any docs referencing the old path
- [x] 1.4 grep repo for `limx_source_docs` to confirm no stragglers

## 2. Extraction hardening (Spec R1, R2, R3)

- [x] 2.1 Re-run `extract.py` against new paths; confirm all 7 fidelity checks still pass on all three docs
- [x] 2.2 Implement `extract.py --check`: re-fetch upstream HTML, compare `raw_sha256`, exit non-zero on drift without modifying files
- [x] 2.3 Run extractor twice in succession; diff output (excluding `fetched_at`); confirm byte-identical (R1 idempotency)
- [x] 2.4 Address pyright noise on `extract.py` (NavigableString import path, Tag/None narrowing, AttributeValueList handling)

## 3. Chunking + index build (Spec R6)

- [x] 3.1 Implement `humanoid/docs/oli-corpus/_research/build_index.py`: parse `sources/*.md` by markdown headings, sub-chunk sections >~500 tokens on paragraph boundaries
- [x] 3.2 Write FTS5 schema (`chunks(doc_id, section, part, heading, body, layer)`) into `humanoid/docs/oli-corpus/index/corpus.sqlite`
- [x] 3.3 Store per-chunk sha for stability detection (D5/R7 sub-chunk citation note)
- [x] 3.4 Verify chunk row count is plausible per doc (section count ≤ chunk count ≤ section count × 4)
- [x] 3.5 Build local embedding sidecar `humanoid/docs/oli-corpus/index/vectors.npz`, aligned one vector per `chunks.id`
- [x] 3.6 Verify vector row count matches indexed chunk row count and vectors are normalized

## 4. MCP server scaffold (Spec R5)

- [x] 4.1 Create `humanoid/docs/oli-corpus/server/` with `pyproject.toml`, entry point `oli-corpus-mcp`
- [x] 4.2 Pin to `mcp` Python SDK + stdlib `sqlite3` + `mistune` + local embedding dependencies; no hosted embedding API (D4/D4a)
- [x] 4.3 Implement `list_docs()` per R5 scenario
- [x] 4.4 Implement `search(query, top_k=10, doc_id=None, include_notes=False, mode="fts")` per R5 scenario (FTS5 BM25 default)
- [x] 4.4a Implement `search(..., mode="vector")` over local embeddings per vector-search scenarios
- [x] 4.4b Implement `search(..., mode="hybrid")` with deterministic rank fusion
- [x] 4.5 Implement `get_section(doc_id, section, part=None)` per R5 scenario; resolve image paths
- [x] 4.6 Implement `cite(doc_id, section, part=None)` per R5 scenario; return URI + repo-relative path
- [x] 4.7 Tag every result with `layer in {"source","note"}` (R4)

## 5. Citation contract (Spec R7)

- [x] 5.1 Implement URI parser/serializer for `oli-corpus://<doc_id>#<section>[?part=N]`
- [x] 5.2 Test round-trip: parse → cite → file path exists → section heading present in file
- [x] 5.3 Flag sub-chunk citation when section sha differs from indexed sha (D5 stability)

## 6. Sources / notes separation (Spec R4)

- [x] 6.1 Create `humanoid/docs/oli-corpus/notes/` with a placeholder `README.md` stating the contract
- [x] 6.2 MCP `search` defaults `include_notes=False`; opt-in returns both layers tagged correctly

## 7. Source map (Spec R8)

- [x] 7.1 Read MAY-137's nine questions from `/home/may33/Documents/vbti/vbti/humanoid/tasks/may-137-explore-oli-sdk-and-control-interfaces.md`
- [x] 7.2 Author `humanoid/docs/oli-corpus/source_map.md`: each question → 0..N `oli-corpus://` citations
- [x] 7.3 For unanswered questions, link to a `gaps.md#<anchor>` entry

## 8. Gaps (Spec R9)

- [x] 8.1 Author `humanoid/docs/oli-corpus/gaps.md`
- [x] 8.2 Survey `sources/` for Chinese-only sections; gap-log each (category `chinese-only`) with citation
- [x] 8.3 Cross-link `source_map.md` "no-source" entries (category `missing-source`)

## 9. README and consumption protocol (Spec R10)

- [x] 9.1 Author `humanoid/docs/oli-corpus/README.md`
- [x] 9.2 Document the cite-or-decline rule (MUST cite Oli facts via `oli-corpus://...` or state no source found)
- [x] 9.3 Include the Claude Code MCP config snippet
- [x] 9.4 Include the OpenCode (aug) MCP config snippet

## 10. MCP registration on this machine (Spec R11)

- [x] 10.1 Install `oli-corpus-mcp` (editable: `pip install -e humanoid/docs/oli-corpus/server`)
- [x] 10.2 Register with Claude Code; confirm `claude mcp list` shows `oli-corpus-mcp` and its four tools
- [ ] 10.3 Register with OpenCode (aug); confirm OpenCode lists the server and tools
- [x] 10.4 End-to-end from a Claude Code session: `search(query="MCP tool interface")` returns at least one `sdk-guide` result with `citation` starting `oli-corpus://sdk-guide#`
- [ ] 10.5 End-to-end from OpenCode: same query, same expectation
- [x] 10.6 End-to-end vector query: `search(query="how can an assistant control Oli through tools", mode="vector")` returns at least one SDK-guide MCP-interface result
- [x] 10.7 End-to-end hybrid query: `search(query="MCP tool interface", mode="hybrid")` returns at least one `sdk-guide` result with `citation` starting `oli-corpus://sdk-guide#`

## 11. Final validation

- [x] 11.1 Run `openspec validate may-139-oli-docs-corpus` — must report valid
- [x] 11.2 Walk every `#### Scenario:` in `specs/oli-docs-corpus/spec.md`; record pass/fail in `_meta/extraction_log.md` (append section "MAY-139 spec walk")
- [ ] 11.3 If all scenarios pass, mark change ready to archive (`openspec archive may-139-oli-docs-corpus` — DO NOT run, just note readiness)
