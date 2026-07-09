---
name: feedback-build-on-branches
description: Always do development work on a 33may/ feature branch, never commit dev work directly to main
metadata:
  type: feedback
---

Always build on a branch, never do development work directly on `main`.

**Why:** Anton course-corrected mid-build (2026-07-09) when the nav stack was progressing on `main`. Keeps `main` clean/releasable and work reviewable — matches the repo's existing habit (e.g. the `33may/may-149-oli-cameras` block-2 branch).

**How to apply:** At the START of any implementation, if on `main`, create+checkout a `33may/<slug>` branch FIRST — task-numbered when it maps to one Linear ID (`33may/may-173-...`), descriptive when it spans several (`33may/nav-slam-poc`). Commit slices there at green checkpoints. Only merge to `main` when Anton asks. Pairs with the global "commit/push only when asked" rule — branch eagerly, commit/push on request. Be surgical with `git add`: the working tree here also carries unrelated churn (ruflo `.claude-flow/`, vendor submodule edits, `vbti/` experiments) — never sweep those into a nav commit.

**Waiver:** This is the default, not absolute — Anton can waive per-task. He did on 2026-07-09, OK'ing the nav-PoC work continuing on `main`. If he says "main is fine for this," respect it and don't re-prompt.
