---
name: feedback-autonomous-notebook-verdict
description: When Anton asks for autonomy, first ask "this terminal or separate?" — if separate, write the autonomous agent prompt to a file (NOTEBOOK-driven, stop-with-verdict) and point him to it.
metadata:
  type: feedback
---

When Anton asks for **autonomy** on a task, FIRST ask whether he wants it run in **this
terminal** or a **separate terminal**. If **separate**: write the autonomous agent prompt
to a **file** and refer him to that file so he can paste/run it in the second terminal.

The autonomous prompt follows the NOTEBOOK-driven pattern: an orient list (files to read
in order), known dead-ends ("don't re-derive"), a ranked list of the strongest leads, a
**measurement-based success bar**, live deliverables (update NOTEBOOK.md + memory + the
Obsidian daily as it goes), and an explicit **stopping condition** — pursue the leads,
judge by measurement, then write a clear **verdict and STOP rather than grind**.

**Why:** Used twice on the Isaac walk. 2026-07-01 directive: *"make the prompt follow the
same idea as we had in the NOTEBOOK md make the clauyde autonomouslt try to fix the task,
while I will be working on otehr tasks. give me the promtp that I will run in second
terminal"* and *"pursue the two strongest leads, judge by measurement, and if they don't
crack it, write a clear verdict and STOP (don't grind)."* Running it separately lets Anton
work other tasks in parallel while a bounded, honest investigation runs.

**How to apply:** On any "do this autonomously" request, ask this-terminal-vs-separate
before starting. If separate, author the prompt into a file (e.g. under the task's dir or
`/tmp`), then tell Anton the path. Don't grind past the ranked leads — deliver a verdict.
Related: [[feedback-reliable-walking-not-falltime]].
