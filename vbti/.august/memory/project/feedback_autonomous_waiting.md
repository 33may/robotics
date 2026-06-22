---
name: Autonomous agent — efficient waits for long-running processes
description: Rules for minimizing context-cache misses when an autonomous agent has to wait on training runs, background jobs, or unknown-duration processes.
type: feedback
originSessionId: 30fca98b-27f8-4a81-b12a-6cb0d2ef57fc
---
When running as an autonomous agent that waits on long processes (training runs, sweeps, bg jobs), the dominant token cost is context re-hydration on each wakeup — not work during wakeups. Minimize wakeups.

**Why:** The prompt cache has a ~5-minute TTL. Every wake-up after that pays a full cache miss (entire conversation re-reads). Autonomous runs that do 4-10 polls per hour burn context repeatedly for no new information.

**How to apply:**

1. **Never poll in foreground bash for unknown-duration waits.** Replace `until ! pgrep ...; do sleep 60; done` patterns with `run_in_background: true` on a wait command. The bash returns immediately with a handle; the agent gets ONE notification on completion. Zero polls, zero re-reads during the wait.

2. **Sleep durations must respect the 5m cache TTL:**
   - `<270s` (4 min): cache warm, cheap to wake
   - `300s–1200s`: WORST ZONE — cache miss without amortization. Avoid.
   - `≥1200s` (20+ min): one cache miss buys a long wait. OK.
   - Never sleep 5–20 minutes. Either 4 min or 20+ min.

3. **Short summaries on wake, not full log dumps.** Write compact heartbeat lines to a file. When checking progress, read `tail -3` / `tail -1`, not `tail -15`. Full log reads = big context spend per tick.

4. **Use ScheduleWakeup (when available) for dynamic loops.** Call `ScheduleWakeup(delaySeconds=1800)` and exit — the conversation unloads entirely between wakeups. Cheaper than keeping bash blocked. (Not all tool sets expose this; run_in_background is the fallback.)

5. **Apply even when bash would "just block for free."** Yes, a foreground bash costs nothing DURING the block — but any timeout fires and reloads the full context. Background + notify is idiomatically the right shape for any wait whose duration you can't prove is under 10 min.

**Source:** User correction 2026-04-21 during autonomous distillation sweep. I was using foreground `until ... sleep 60` with 10-min timeout for waits of unknown multi-hour duration, causing repeated cache misses on each timeout-retry cycle.
