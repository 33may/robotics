---
name: Phase 1 data-efficiency plan (post-v020)
description: 4-slice sweep (6.25/12.5/25/50%) of v020's 765-ep corpus → heatmap SR(epoch, dataset_size) on dual_cup_60. Plan written 2026-05-11, target close 2026-05-18.
type: project
originSessionId: e616531f-dca0-4ae3-b364-de744e41eb54
---
**Goal:** characterize how many demos v020's task actually needed. Task = solved at 100% (93% on dual_cup_60). Now build the data-efficiency curve as a heatmap.

**Plan file (Obsidian):** `vbti/sessions/sprint4/Evaluation plan.md`

**Slice → version mapping (small first):**
- v021 = 6.25% (48 ep, ~3.5h train)
- v022 = 12.5% (96 ep, ~7h train)
- v023 = 25% (192 ep, ~13h train)
- v024 = 50% (383 ep, ~27h train)
- v020 = 100% (765 ep, anchor — already done)

**Subsamples** built locally via `python -m vbti.logic.dataset.subsample --src=eternalmay33/duck_cup_v020_all --stride=N --dst=eternalmay33/duck_cup_v020_everyN`. Pre-built datasets (not runtime `episodes:` filter) because engine doesn't forward the filter to lerobot-train and `info.json[total_frames]` is read globally.

**Per-run knobs:** identical to v020 except `dataset.sources[0].repo_id` = `_everyN` and `logging.save_freq` = `{3500,7000,14000,22000}` (every 2 epochs). 12 epochs, BS=16, vision unfrozen, lr_scale=0.1, 5 cams, brightness/contrast/sharpness/affine aug, depth bypassed.

**Plot:** heatmap `epochs × dataset_size`, color = SR on `dual_cup_60` real-robot eval. 6 ckpts/row × 4 rows = 24 cells (or 12 if culled to back-half — decide after looking at val_loss).

**Why:** When the user asks "what's next for duck-cup" or "Phase 1 status", anchor on this plan, not the old `duck_cup_sota_plan.md`. The SOTA-chasing era is over; Phase 1 is an evaluation/data-efficiency study. Phases 2 (+ sim data) and 3 (UVA aux-task arch) follow if Phase 1 closes by May 18 as targeted.

**How to apply:** If user asks for a v021/v022/v023/v024 config, the only diffs from v020 are repo_id + save_freq. Don't reintroduce the runtime `episodes:` filter. Out of scope for Phase 1: sim, UVA, GR00T.
