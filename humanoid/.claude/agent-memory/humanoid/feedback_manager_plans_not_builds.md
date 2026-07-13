---
name: feedback-manager-plans-not-builds
description: In /manager (planning) sessions, stay at planning altitude — shape tasks/board/dailies; do NOT dive into code or run verification. Building is a separate /code session.
metadata:
  type: feedback
---

In a `/manager` / planning session, plan — don't build or verify code. Reading a couple of
files to *reshape a task* (e.g. "this is already plumbed → it's a verify-run, not a build") is
fine and useful. But do NOT start diving into implementation details: no checking payloads
resolve, no running sims, no editing code, no verification runs.

**Why:** Anton separates the operating loop deliberately — `/manager` shapes Linear/dailies at
task altitude; `/code` does the TDD build. Mixing them turns a planning session into a
half-started build and burns the session's purpose. Correction given 2026-07-10.

**How to apply:** In planning mode, stop at "here's what the task actually is and what it needs."
Hand the actual booting/verifying/editing to a later `/code` session or to Anton. Related:
[[feedback-learn-before-recommending]], the operating-loop split in AGENTS.md.
