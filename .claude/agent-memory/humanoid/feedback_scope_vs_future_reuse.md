---
name: feedback-scope-vs-future-reuse
description: Separate current-task scope from future architectural possibility; avoid absolute “not reusable” claims unless proven
metadata:
  type: feedback
---

**Rule:** When evaluating whether something is useful, separate **current-task scope** from **future architectural possibility**. Do not say "not reusable" or "not needed" when the accurate claim is "not needed for this task / with the current asset."

**Why:** Anton caught the `kinematic_projection` explanation being too absolute. The current serial USD did not need it, but future high-fidelity locomotion sim2real might.

**How to apply:** Use phrasing like: "not needed for tonight because X; future reuse becomes plausible if Y changes." Especially for sim2real, SDK, physics fidelity, or vendor tooling.
