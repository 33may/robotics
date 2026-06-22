---
name: feedback-ai-native-documentation
description: Documentation for every module, subsystem, and functionality should be structured so AI agents can consume and act on it directly
metadata:
  type: feedback
---

**Rule:** For every module, subsystem, or functionality, build documentation that an AI agent can natively consume and act on.

Required shape:
- explicit **entry points**: file path, function/class, CLI invocation
- explicit **inputs/outputs**: types, shapes, file formats
- explicit **side effects** and dependencies
- explicit **failure modes**
- **linkable structure**: stable headings, code blocks, tables; not prose walls

**Why:** An agent landing in the repo cold should be able to use a module correctly from the doc alone and extend the doc as the module grows. Prose-style indexes fight that.

**How to apply:**
- Default writing mode for any new doc, even quick notes.
- When reading a doc that's prose-heavy, treat upgrading it to AI-native structure as in-scope work.
- Cross-link docs by stable headings and stable file paths.

Related: [[feedback-general-not-task-scoped-docs]]
