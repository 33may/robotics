---
name: daily-note-timestamps
description: Every block written to the Obsidian humanoid daily note must carry the current time (HH:MM) so Anton can track time spent per task
metadata:
  type: feedback
---

Every block/section appended to the humanoid daily note gets the current time, e.g. `## 14:32 — Linear cleanup` or a `*14:32*` line at the block start.

**Why:** Anton started time-tracking his tasks (2026-07-08); the daily note is the tracking surface, so untimed blocks lose the data.

**How to apply:** Before drafting any daily-note block, run `date +%H:%M` and stamp the block header. Applies to all new blocks in `vbti/humanoid/daily/DD-MM-YYYY.md` — not retroactively to old notes. Also: don't read old session transcripts to reconstruct times (token budget); just stamp from now.
