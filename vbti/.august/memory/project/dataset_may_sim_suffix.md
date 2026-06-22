---
name: may-sim suffix meaning
description: "_may-sim" suffix on datasets indicates they were recalibrated from old to current ("may") calibration via a conversion script — NOT a sim-rendered variant
type: project
originSessionId: 92f50957-8ae3-4bb7-a2e5-c27a13fabe74
---
The `_may-sim` (and related `_may-sim_depth`, `_may-sim_detection`, etc.) suffix on eternalmay33 datasets does NOT mean simulation-rendered.

It means: dataset was recorded on the OLD calibration, then converted via a script to the CURRENT ("may") calibration profile. So `_may-sim` = "now on the actual / current calibration."

**Why:** there was a calibration change at some point; older datasets needed to be transformed to match the new state/action space before being usable for current training.

**How to apply:**
- When picking training data, treat `_may-sim` variants as the "valid for current robot" copy of older datasets
- The original (non-suffixed) `01_black_gripper_front`, `02_black_full_center` etc. are on the OLD calibration → don't mix with newer datasets directly without recalibrating
- New datasets (04+ depth series, 10_black_cup_red_bg_depth, etc.) were collected on current calibration → no `_may-sim` needed
- Recalibration script in repo: see `git log --grep=recalibrate` (commit `3fa1be6 feat: add recalibrate command to transform datasets between calibration profiles`)
- Earlier I incorrectly assumed `_may-sim` meant Cosmos-Transfer or sim-rendered output — it does not
