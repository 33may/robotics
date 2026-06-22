---
name: feedback_md_tables_not_ascii
description: In any .md file always use normal Markdown tables, never ASCII/box-drawing tables
metadata:
  type: feedback
---

In any `.md` file (daily notes, task notes, docs, anywhere in the Obsidian vault), always use standard Markdown pipe tables. Never use ASCII / box-drawing tables (`╭ ╮ ╰ ╯ ─ │ ├ ┤`) inside Markdown files.

**Why:** Obsidian renders Markdown tables natively — they sort, wrap, and look right. Box-drawing tables only render as a monospace blob and break visual flow.

**How to apply:** The box-drawing/checklist symbol convention from `/work` applies *only* to terminal/chat output (fenced `text` blocks shown in the Claude TUI). The moment the destination is a `.md` file, switch to normal Markdown tables — same content, pipe syntax.
