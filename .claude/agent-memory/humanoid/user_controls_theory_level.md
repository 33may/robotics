---
name: user-controls-theory-level
description: Anton is a strong ML/systems engineer and the architect, but explicitly not deeply versed in low-level controls/robotics theory — explain those concepts plainly and short.
metadata:
  type: user
---

Anton is the architect and a strong ML/systems engineer, but explicitly *not* deeply versed in low-level controls/robotics theory (PD control, parallel mechanisms, AB↔PR joint spaces, drive modes). When a control or robotics-theory concept comes up, lead with a one- or two-sentence plain-language explanation before the formal version, and prefer small tables for comparisons.

**Why:** He said so directly and repeatedly during MAY-147 (2026-06-22): asked "whart is AB PR I dont really uidnesratnd what that means," asked for "one sentence what was problem and how exactly B solves it, I am not too good with controls and robotics theory," and asked for short table answers ("answer questions short… tell me as small table").

**How to apply:** On any controls/robotics-theory topic (PD gains, kinematic projection, control-law modes, parallel mechanisms), open with intuition + a concrete analogy, keep it short, use MD tables for sim-vs-real / option comparisons. He decides architecture; my job is to make the theory legible — don't assume controls fluency. Related: [[feedback-md-tables-not-ascii]].
