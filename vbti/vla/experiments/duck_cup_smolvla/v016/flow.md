# v016 — Research Flow Journal

Insights, decisions, surprises, dead ends, pivots. Chronological.

---

## 2026-04-29 — orientation
Workspace created. `notes.md` is the design brief (frozen entry state). Two big forks pending: Q1 (old-dataset handling: backfill / estimate-everywhere / dual-path) and Q2 (depth ingestion: extra camera / side-branch / channel concat).

Hard constraint confirmed with the user: v013/v014/v015 datasets must stay co-trainable. This kills any approach that requires identical schemas without a backfill or branching strategy.

## 2026-04-29 — decision: experiment ordering
Picked four investigations as the first batch, ordered by which decisions they unblock:
1. **exp01 — code investigation (cheap, fast)**: tells us whether the dual-path option (C) is feasible without deep engineering. If LeRobot/SmolVLA require uniform image keys, option C dies and we're in {A, B} territory.
2. **exp02 — hardware capture pilot (real-world risk)**: confirms 4× D405 depth+color is bandwidth-feasible. If FPS collapses, real depth is off the table and we're forced to option B.
3. **exp03 — Depth Anything backfill cost**: timing + visual quality tells us how painful option A is.
4. **exp04 — depth representation study**: doesn't unblock a decision but is needed before any training run; cheap to do early.

Rationale: 1 and 2 can each kill an option outright. 3 changes the cost calculus. 4 is plumbing for whatever wins.

Skipped for now: open question 5 (does v013 already learn implicit depth from parallax) — interesting but expensive to answer cleanly, deferred to synthesis stage if still relevant.

## 2026-04-29 — exp01 accepted
Code investigation came back clean. Three actionable verdicts:
1. **Extra-camera path (Q2-B1) is trivial** — zero SmolVLA edits, just config + dataset + capture. This is the recommended v016 starting point.
2. **Channel-concat (Q2-B3) is dead** — would defrost the SigLIP patch conv and break the `freeze_vision_encoder=True` paradigm. Not worth the risk.
3. **Co-training requires backfilled depth on old datasets** because `MultiLeRobotDataset` is disabled and `aggregate_datasets` strict-checks feature equality. The `<key>_padding_mask` hook then attention-masks the backfilled depth so it doesn't pollute the gradient.

Surprise that genuinely changed our plan: `empty_cameras=0` default means missing keys are silently dropped, NOT padding-masked. So even for the "missing modality" path, we MUST emit a zero/dummy depth tensor — we cannot just leave the key out. This is exactly why backfill is the only viable Q1 path.

## 2026-04-29 — pivot: user-led exp02 redesign
User redirected exp02 from a generic "hardware capture pilot" to a head-to-head **real D405 depth vs Depth-Anything-estimated depth** comparison. This is sharper: it directly answers "how close is estimated to real, and what augmentation closes the gap?" — which is the core empirical question that decides whether Q1-A backfill is even viable in practice.

Plan: record one episode with depth-enabled D405s into a new dataset, sample one episode from `01_02_03_merged_may-sim_detection*`, run DA3 on it, compare. Then research augmentation strategies (noise injection, distortion, smoothing, etc.) for stable mixed training.

User's instruction: walk through the recording pipeline **together** before delegating. No subagent for the recording-setup investigation phase. Lookup-and-explain mode.

## 2026-04-29 — DECISION: gripper-depth-only, Q2-B1, B2 deferred
User chose to narrow v016 scope:
- **Q2 path: B1 (extra camera).** Confirmed.
- **Cameras with depth: gripper only (1, not 4).** Rationale: minimize token budget (1 extra ViT pass instead of 4), test the hypothesis at the most critical contact point first.
- **Q3 (depth representation / colorization): defer.** Decide after exp02 visual comparison shows what survives both estimated and real depth.
- **Q1 path (old data handling): still pending** — exp02 informs it.
- **Q2-B2 (side-branch with gripper-tailored attention) deferred to v017** if v016-pilot shows the depth signal carries any value.

Recording-pipeline gaps to close (lerobot upstream changes, not vbti):
1. `RealSenseCamera.read_latest_depth()` doesn't exist (only `read_depth()` synchronous path).
2. `so_follower.get_observation()` calls only `cam.read_latest()` — no depth path.
3. `so_follower._cameras_ft` declares `(H,W,3)` shape only — needs separate `<cam>_depth` feature when `use_depth=True`.
4. Depth must be 3-channel 8-bit on disk (LeRobot image-feature constraint) — colorize or 3-ch broadcast before write.

These four gaps are the engineering cost of B1 even before any model training. exp02 (head-to-head real vs estimated depth) does NOT yet require them — we can capture a single episode with a one-off script, no need to plumb `lerobot-record` until the depth signal is shown to be worth the trouble.
