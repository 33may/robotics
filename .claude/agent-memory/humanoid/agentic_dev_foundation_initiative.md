---
name: agentic-dev-foundation-initiative
description: 2026-07-02 initiative — stand up a stable agent-legible foundation (AGENTS.md + architecture.md + ADRs + path-scoped rules + CI invariance gate + OpenSpec archive→specs sync); Anton drafts decisions/content, agent assembles the machinery; heading to a Linear task.
metadata:
  type: project
---

**Initiative (2026-07-02):** build a scalable/stable agentic-development foundation for the humanoid repo, driven by July-2026 best-practice research. Split of labor: **Anton = architect** (drafts decisions, content, diagrams); **agent = writer** (assembles the standardized files). This is deliberately weighty ceremony reserved for foundational work — skip it for small fixes.

**Deliverables:**
- `AGENTS.md` (root, ~100-150 lines) = source of truth; `CLAUDE.md` = 2-line `@AGENTS.md` shim. (Claude Code reads CLAUDE.md; AGENTS.md is the Linux-Foundation cross-tool standard both Claude + Codex read — one file, both agents.)
- `docs/architecture/architecture.md` = the ports-and-adapters map (see [[arch-dataflow-bus-and-policyrunner]]); MUST be linked from AGENTS.md (orphaned docs get <10% agent discovery).
- `docs/architecture/adr/NNNN-*.md` = one ADR per hard decision (invariance, dual-sim + glide, Path 2 dataflow bus, PolicyRunner process).
- `.claude/rules/{brain,world}.md` with `paths:` frontmatter = path-scoped rules that load only when editing that area.
- **CI invariance gate** — a test enforcing "Brain/PolicyRunner import neither `isaacsim` nor `limxsdk`" (repo has ZERO CI today; enforcement is discipline-only). Rule of thumb: merge-blocking → CI; eyebrow-raising → CLAUDE.md.
- **OpenSpec hygiene** — wire `archive` → `openspec/specs/` sync so the spec corpus finally accumulates (`specs/` is currently empty = spec-first, not spec-anchored).

**Research-backed constraint:** small + HUMAN-written rules files beat comprehensive/LLM-generated ones (measured −2% success / +23% cost) → Anton drafts content, agent only structures it. Don't auto-generate a god-doc.

**Status:** `docs/architecture/architecture.md` **drafted** (2026-07-06) — 11 sections, section-by-section with Anton, grounded in `logic/oli/comm/` real contracts; a **living doc**, extended as the project evolves. WORKPACKET (the source Q&A) at `docs/architecture/WORKPACKET.md`. Next: `AGENTS.md` (thin root linking to architecture.md) + ADRs, then `.claude/rules/`, CI invariance gate, OpenSpec archive→specs sync; then shape a Linear task via `/manager`.

Related: [[arch-dataflow-bus-and-policyrunner]], [[middleware-inner-comm-research]], [[feedback_ai_native_documentation]], [[humanoid-linear-task-flow]], [[feedback_general_not_task_scoped_docs]].
