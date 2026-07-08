---
name: feedback-tests-in-repo-tdd
description: Write tests as a real TDD suite committed in the repo (red→green→refactor), never throwaway /tmp smoke scripts; tests are durable artifacts.
metadata:
  type: feedback
---

**Rule:** Tests live in the repo as a proper TDD suite, not as throwaway `/tmp` scripts. Follow red→green→refactor: write the failing test first, watch it fail for the right reason, then write minimal code to pass.

**Why:** Anton caught me writing one-off verification scripts in `/tmp` (e.g. `/tmp/test_contracts.py`, `/tmp/probe_onnx.py`) during MAY-147 and said: "write the tests not in tmp, do the complete TDD that lives in repo." Throwaway smokes vanish, can't re-run when code changes, and prove nothing was actually test-driven.

**How to apply:** For every new component (comm, reason, action/policy_runner, runtime, …) add a committed test under the repo's test tree BEFORE the implementation. Run it red, implement to green, refactor. No `/tmp` smokes for anything that should be reproducible. Turn hard-won findings (e.g. the implicit-PD instability [[isaac-pd-implicit-drive]]) into regression tests. Related: [[feedback-scope-vs-future-reuse]].
