---
name: feedback-combine-adjacent-tasks
description: When two open tasks naturally share substrate (same docs, same code, same investigation), bundle them in a single working pass instead of scheduling serially
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0cbe35b7-b6f8-46f7-98d7-42a6e7c32aa2
---

When proposing a day plan, if two tasks naturally share substrate (same docs, same code path, same investigation), bundle them into one working pass rather than scheduling them serially.

**Why:** Anton's call on 2026-06-17 — picked MAY-137 (SDK exploration) as the focus, then immediately asked to combine MAY-139 (AI-ready docs) into it: *"lets also combine this exploration with AI ready Oli notes, since we still will be looking at documentation"*. Validated end of day — the combined pass produced the corpus *as the substrate* of the research, turning MAY-137 into trivial corpus queries. Two tasks closed instead of one. Serial scheduling would have meant re-reading the same docs twice in different sessions.

**How to apply:** When drafting daily plans or task ordering, look for overlap in inputs (docs, repos, hardware) and outputs (notes, artifacts). If two tasks would read the same source material or touch the same system, propose bundling them — keep the two outputs distinct (so each task has a clear deliverable), but run the work as a single session. Surface the bundling option explicitly so Anton can confirm or override.

Counter-cases: don't bundle when the tasks have different stakeholders, different deadlines that would muddy "done", or when one is a thinking task and the other is a building task.

Related: [[humanoid-linear-task-flow]]
