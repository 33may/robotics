---
name: feedback-branch-naming-33may
description: Always use `33may/` as the branch prefix, not Linear's suggested `antonnedf/` (which mirrors Anton's Linear display name).
metadata:
  type: feedback
---

When creating branches for Linear issues on this project, use the prefix **`33may/`**, not `antonnedf/`.

Example: `33may/may-145-capture-robotcmd-wire-shape-via-sim-role-probe` — not `antonnedf/may-145-...`.

**Why:** Linear's `gitBranchName` field auto-builds from the assignee's Linear display name ("Anton Novokhatskiy" → `antonnedf/`). Anton's actual GitHub user is `33may`, and he wants branches to match that for consistency across the rest of the org.

**How to apply:**
- When opening a branch for a Linear-tracked task, take the slug suffix from Linear (`may-XXX-<kebab-title>`) but replace the prefix with `33may/`.
- Do NOT copy the `gitBranchName` field verbatim from `mcp__linear__get_issue` output.
- If a branch has already been pushed with the wrong prefix (like `antonnedf/may-145-...` from this session), leave it — don't rename mid-PR. Apply the rule prospectively.
