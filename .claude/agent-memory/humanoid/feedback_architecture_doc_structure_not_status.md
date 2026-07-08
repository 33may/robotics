---
name: feedback-architecture-doc-structure-not-status
description: The architecture document captures stable structure only — not transient status, blockers, or roadmap; those live in daily notes / Linear / memory.
metadata:
  type: feedback
---

**Rule:** The architecture document (`docs/architecture/architecture.md`) describes **stable structure** — components, contracts, boundaries, invariants — NOT transient status, blockers, or roadmap. State the fact ("Isaac has walk and glide modes"), not the status ("walk is LimX-gated on a first-step contact gap, blocked on their reply").

**Why:** 2026-07-06, drafting §5, I put the walk-mode blocker (LimX-gated, contact-fidelity gap, awaiting reply) into the doc. Anton cut it: *"this should not be part of the architecture document, we just say we have walk and glide mode."* Status rots and belongs where it's tracked; the architecture doc must stay true across time so agents can trust it.

**How to apply:** In architecture/design docs keep structural facts (which contract a mode uses, spine-vs-aux, process boundaries) and strip status/blockers/dates/roadmap. Transient state → daily notes, Linear (MAY-XXX), or memory. Litmus test: "will this still be true in 3 months?" — if not, it doesn't belong in the architecture doc.

Relates to [[agentic-dev-foundation-initiative]], [[feedback_general_not_task_scoped_docs]], [[feedback_ai_native_documentation]].
