---
name: humanoid-linear-task-flow
description: Humanoid Linear tasks should be drafted and clarified before being pushed to Linear.
metadata: 
  node_type: memory
  type: project
  originSessionId: 7039e4db-0875-47de-a24d-4989de40814b
---

Humanoid project task management should not blindly add a flood of rough tasks into Linear. `/manager` should first help plan and clarify tasks under the user's guidance, using the documentation-writing workflow where useful, then push only clearly defined tasks to Linear.

**Why:** The user wants Linear to stay clean and reliable as the source of truth, not become cluttered with half-formed planning thoughts or speculative task dumps.

**How to apply:** Design `/manager` with a task-dispatch flow: draft/refine task candidates in conversation first, require user approval for each task or batch, then create Linear issues with explicit team, project, state, assignee, and clear descriptions. In Claude terminal/chat responses, use fenced `text` blocks with compact terminal tables/boxes, not Markdown headings/bold. In Obsidian or any `.md` file, use normal Markdown tables instead. Default table columns: `ID | Task | Status | Done | Due | Next`; shorten task names so each row stays one line, keep column width only as wide as the largest value, and make `Next` concrete. Do not write the daily note during weekly planning; wait until the whole plan is finished and approved, then draft the note once. Link this with [[humanoid-session-capture-repo]] and [[humanoid-summer-2026-plan]].
