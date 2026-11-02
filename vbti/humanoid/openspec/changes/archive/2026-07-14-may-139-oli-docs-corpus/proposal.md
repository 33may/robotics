## Why

I work continuously with AI coding agents on the humanoid stack, and they need full, accurate context about the Oli robot — its SDK, control modes, sensors, startup flow, simulators — available at any point in any session, without me re-explaining or pasting PDFs. LimX ships three official documents (LimX EDU Quick Start, Oli EDU User Manual, Oli EDU SDK Development Guide), but they currently live as web pages / PDFs that agents can't reliably grep, cite, or reason over.

This change creates a curated, AI-ready Oli documentation corpus inside the repo. It is a prerequisite for MAY-137 (SDK & control interface exploration) and for every later humanoid task that needs to ground claims about Oli in real documentation rather than assumption.

Linear: [MAY-139](https://linear.app/may33/issue/MAY-139/prepare-oli-documentation-as-ai-ready-data-source)
Consumer: [MAY-137](https://linear.app/may33/issue/MAY-137/explore-oli-sdk-and-control-interfaces)

## What Changes

- Add a new capability `oli-docs-corpus` — a versioned, in-repo corpus of LimX / Oli documentation in clean markdown, with a source map and per-document metadata.
- Acquire and parse the three official LimX docs into markdown under a stable folder layout.
- Author a `SOURCE_MAP.md` listing each document, its scope, key sections, and which MAY-137-style questions it answers.
- Author a `GAPS.md` capturing missing / unclear / blocked sources (e.g. remote controller details, MCP server protocol specifics, low-level SDK schemas).
- Author a corpus `README.md` explaining how AI agents should consume it (cite section anchors, never paraphrase as fact without citation, prefer corpus over web).
- **Research track** (resolved in `design.md`, not here): pick the corpus *format* and *access pattern* — flat markdown vs chunked + indexed (context7-style MCP / RAG), and how agents query it.

## Capabilities

### New Capabilities
- `oli-docs-corpus`: A repo-local, parsed, indexed documentation corpus for the Oli humanoid robot and LimX SDK. Defines the corpus contract — which sources are included, how they are structured, how they are cited, how completeness / gaps are tracked, and how AI agents access them.

## Impact

- **New content** under a corpus root (path decided in `design.md`): parsed markdown of the three LimX docs, `SOURCE_MAP.md`, `GAPS.md`, `README.md`.
- **No production code changes** in this proposal — corpus is content. Tooling for parsing / indexing / serving is scoped in `design.md`.
- **Possible new dev dependencies**: a markdown extraction toolchain (e.g. `pandoc`, `marker`, `mineru`, manual cleanup) and optionally an indexing / MCP layer — to be decided in `design.md`.
- **Direct consumer**: MAY-137 — its nine open questions become acceptance evidence that the corpus is actually usable.
- **Downstream consumers**: every future humanoid task touching Oli (sim, SDK bindings, control policies, MCP integration).
