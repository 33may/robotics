# Loop stability review — 2026-08-24

Written by the main session while four investigation agents ran in parallel
(`ur_rtde_library.md`, `app_client_audit.md`, `camera_usb.md`,
`network_link.md`). This file holds only what was established *here*, from
run artifacts and code tracing. No robot was touched to produce it.

## Reference run

`inspection/data/runs/2408-seeded` — first run with seeded object growth.
26 turns, 25 captures, full orbit (12 azimuths x 3 elevations), ~18 min.
Completed normally.

## 1. The freshness guard held on every capture — verified

The stale-pose failure has a fingerprint: consecutive captures stamped with
bit-identical joint values while the arm demonstrably moved. Run `2108-ui`
had **six**.

`2408-seeded`: **25 of 25 captures carry distinct joint vectors; zero
duplicates.** The RTDE stream did die once in this run (detected by the
`plan-survey` thread, healed by reconnect), and no capture was stamped from
the frozen window. The guard did the job it was built for.

This is the strongest available evidence that captures cannot be poisoned by
a stream death: `RealRig.capture()` reads the pose through the guarded
`q()`, so a death during capture aborts the turn rather than lying.

## 2. Seeded growth held across a full orbit — verified

| | run `2408-geomtest` (pre-fix) | run `2408-seeded` (post-fix) |
|---|---|---|
| fused extent | 287 x 352 x 160 mm | **97 x 63 x 102 mm** (main body) |
| centroid | (148, -331, 62), drifting | **(95, -419, 58)** |
| shape | two blobs 250 mm apart | hollow rim, ~100 mm body, handle stub |

Survey view put the object at (94, -419, 63). After 24 further views the
centroid sat at (95, -419, 58) — **~1 mm of drift** and 5 mm in z (the cloud
grew downward as the sides came into view, which is expected).

Residual: **14 points of 5452 (0.3%)** in a 6x8x8 mm cluster at z=126-134,
about 16 mm above the rim. This is the known and accepted leak mode of
single-linkage growth. Cost: it inflates the object's collision box by
~16 mm in z. Not worth chasing yet — revisit if the box ever blocks a
viewpoint.

## 3. Worst-case failure path is safe — traced, not merely assumed

The open risk is that `reconnect()` eventually fails (it has succeeded 2/2,
which is not a sample size). Traced what happens if it fails **mid-move**,
the most dangerous moment:

1. `_wait_async` polls `_safety_ok()` -> `_assert_fresh()` raises
   (`motion/execute.py`)
2. `execute()`'s `except` calls `self.stop()` -> `stopJ(2.0)`, the arm
   decelerates, then re-raises
3. `_exec_worker` catches and queues `exec_done` with `outcome="fault"`
   (`run/machine.py:485-491`)
4. `_on_exec_done` records the turn, sets phase `fault`, tells the operator
   "check pendant; Exit only" (`run/machine.py:404-412`)
5. `app.py`'s `finally` still writes `fused_cloud.npy` if the accumulator
   holds anything

**So a failed heal costs the remainder of the orbit and nothing else.** The
arm is commanded to stop, the state machine reaches a defined phase, and
every capture taken up to that point survives on disk. That is an acceptable
worst case and it downgrades the RTDE bug from "unsafe" to "expensive".

## 4. Camera droughts did not cost a capture — verified

Three `Frame didn't arrive within 2000` events. All 25 capture directories
are complete (rgb + both IR + raw and aligned depth + meta), depth validity
80-90% on every one, no degraded capture. `capture()` calls
`fresh_bundle(min_new=3)`, which blocks for three *new* frames, so a drought
at capture time would surface as `capture failed`, not as a stale frame. It
never did.

The missing directory `003` is not a gap: turn 4 was an operator software
stop, aborted before any capture.

## 5. Open items this review did NOT settle

- **Why the RTDE session ends.** Unreproduced in 23 isolated trials; the
  autopsy showed one 30004 socket ESTABLISHED and one in TIME-WAIT, with
  `isConnected()` reporting False. Which socket belonged to the recv is
  still unresolved — see the agent reports.
- **Heal reliability.** n=2. No basis yet for a failure rate.
- **Why the camera drops frames.** n=3, no kernel USB errors seen, but the
  D405 sits behind three chained hubs.

## 6. Instrumentation gap worth closing

`stream_autopsy()` cannot say which of the process's 30004 sockets is the
receive interface's own, so the two readings of the same evidence point at
different culprits. Recording the recv's local port at construction (diff
the process's 30004 sockets immediately before and after building it) would
make the next death unambiguous. Deferred here only to avoid editing
`execute.py` while the client-audit agent was reading it.

## Cosmetic

A single 2 s camera hiccup prints a full stack trace (`run/rigs.py:110`,
first-failure-gets-the-traceback). In an operator log that reads as a crash.
Worth demoting to a one-line warning with the traceback at debug level.
