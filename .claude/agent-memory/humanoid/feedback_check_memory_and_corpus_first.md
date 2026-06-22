---
name: feedback-check-memory-and-corpus-first
description: Before posing architectural questions on the humanoid project, query agent memory AND oli-corpus MCP — the answer is often already there.
metadata:
  type: feedback
---

Before asking Anton architectural / "how should we structure this" questions on the humanoid project, **first** check:
1. Local agent memory under `.claude/agent-memory/humanoid/` (USD locations, vendor map, prior decisions like Isaac Sim over Lab).
2. `oli-corpus-mcp` (`mcp__oli-corpus-mcp__*`) for any SDK / control / interface question. Cite back with `oli-corpus://...`.
3. The actual filesystem — `humanoid/assets/`, `humanoid/logic/simulation/isaacsim/`, `humanoid/vendor/` — for artifacts that may already exist.

**Why:** Asking Anton to choose "USD source: MJCF auto-import vs ask LimX vs handcraft" when memory already records that the vendor ships layered USDs under `humanoid/vendor/humanoid-mujoco-sim/humanoid-description/HU_D04_description/usd/` AND there is already an imported asset at `humanoid/assets/oli/usd/HU_D04_01.usd` plus a working `load_oli.py` smoke loader — wastes his time and signals that I did not orient before proposing.

**How to apply:**
- Treat the question "is this in memory or in corpus already?" as a mandatory step before any architectural option list.
- When the corpus has a relevant section, cite with `oli-corpus://<doc_id>#<section>?part=N` so the answer is traceable.
- When something is already implemented in the repo, build on it; don't invent a parallel path.
- Related: [[reference-oli-corpus-mcp]], [[humanoid-oli-docs-before-sim]], [[vendor-humanoid-mujoco-sim]].
