---
name: feedback-general-not-task-scoped-docs
description: Documentation for modules, vendors, configs, and subsystems should be general reference docs, not scoped to the task that created them
metadata:
  type: feedback
---

**Rule:** Whatever we build or vendor — modules, scripts, subsystems, third-party repos, configs — gets documented as a **general reference**: what it is, why it exists, and how to use it. Do not scope the doc to the task that spawned it.

The project-specific "how to apply in our context" layer lives in **agent memory** and points back at the general doc.

**Why:** Docs scoped to "current task reuse" bias future decisions before they're made and rot the moment the task changes. Memory decays cheaply; general docs stay clean.

**How to apply:**
- New module / vendor / config → general doc in `docs/` or `docs/vendor/`. No "we will use…", no "skip…", no current-sprint framing.
- Project-context notes — what we reuse, when to come back, how this fits our flow — go in memory files that link to the general doc.
- When extending an existing doc, preserve general framing and push task-specific notes to memory.
