---
name: servo-leader-follower-voltage
description: Identify leader vs follower SO-101 arm by supply voltage in scan_all output
metadata: 
  node_type: memory
  type: reference
  originSessionId: 934fc31d-8d22-4972-8251-ba505270e8e5
---

The two SO-101 arms share `/dev/ttyACM*` ports that can swap on replug/reboot. Tell them apart with `python -m vbti.logic.servos.scan_all` — read the **Voltage** column:

- **~5 V** → the **leader** arm (hand-driven, holds no load) → `--teleop.port`
- **~12 V** → the **follower** arm (executes actions under load) → `--robot.port`, or inference `--port`

`lerobot-find-port` (interactive cable-unplug) is the generic tool, but the voltage tell from `scan_all` is faster and non-interactive. Confirmed 2026-05-18 from live scan_all output. Related: [[vbti-user-account]]
