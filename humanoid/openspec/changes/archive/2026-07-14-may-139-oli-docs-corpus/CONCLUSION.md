# Conclusion — AI-ready Oli docs corpus + oli-corpus MCP: shipped

**Status:** closed 2026-07-14. Corpus + MCP server done and in daily use; only the secondary
OpenCode-editor integration is deferred (below).

## What it delivered

A curated, AI-ready Oli documentation corpus in-repo (`docs/oli-corpus/`) plus the `oli-corpus`
MCP server, so coding agents can grep, cite, and reason over the LimX/Oli docs (EDU Quick Start,
Oli User Manual, SDK Development Guide) — SDK, control modes, sensors, startup flow, joint order —
at any point in a session without re-pasting PDFs. Grounds agent claims about Oli in real
documentation instead of assumption.

## Why it matters downstream

It is the standing reference layer for every humanoid task (a prerequisite named in AGENTS.md's
golden rules: "check memory + oli-corpus first"). Used throughout the MAY-173 work.

## Deferred (secondary client only)

- §10.3 / §10.5 register + end-to-end verify the server under **OpenCode** (a second editor).
  The Claude Code / Codex path works and is what's in use; OpenCode is a bonus client, not a
  blocker.
- §11.3 was the "mark ready to archive" gate — this file is that archive.
