---
name: feedback_sequential_sweeps
description: Never batch-launch multi-run research sweeps; run sequentially so bugs/insights can steer the next run
type: feedback
originSessionId: 6b8a2b91-4695-46a5-967e-f7b16a9949f2
---
Run research sweeps sequentially, one at a time. Don't launch v1→v6 (or m1→m4 etc.) in a single background loop.

**Why:** mid-sweep you find data bugs (example: 2026-04-20 — filter was rescuing zero-detection rows via interpolation, would have poisoned all 6 runs), or results from run N change what run N+1 should test. A batch launch wastes GPU-hours on bad premises and hides the learning loop.

**How to apply:** for any multi-experiment sweep: (1) design the FIRST run fully, (2) kick it off, (3) analyze results, (4) only then design/launch the next. Each run is a checkpoint where you may redesign the remaining plan. If the user says "overnight sweep, I'll check tomorrow", that's still sequential internally — you as the researcher decide between runs, not upfront-batched.
