# v016 Research TODO

## Current Phase: Decision-informing investigations

**Decided 2026-04-29:** gripper-depth-only, Q2-B1 (extra camera), Q2-B2 deferred to v017.

### In Progress
*(none — user reviewing before exp02 starts)*

### Up Next

- [ ] **exp02a_estimate_depth_preview** — Run Depth Anything (v2 metric or DA3) on a sample of gripper-camera frames from `01_02_03_merged_may-sim_detection*`. Visualize with various colormaps (turbo, viridis, gray) and clip ranges (auto, 0.05–0.5m, 0.1–1m). Goal: pick the canonical colorization scheme that we'll bake into BOTH real D405 capture AND backfilled estimated depth, so the policy sees a uniform-looking depth feature across episodes.

- [ ] **exp02b_real_vs_estimated** — After v016 has captured at least one real-depth episode, sample one matching frame from each source, render under the chosen colormap, compare side-by-side. Validates that the colormap choice from exp02a actually produces visually similar inputs across the two depth sources.

- [ ] **exp03_depth_representation_study** — Once real D405 frames in hand: render grayscale-uint16, grayscale-uint8, turbo-colorized, viridis-colorized, with fixed-clip vs per-frame-norm. Pick representation deliberately. Informs the Q3 deferred decision.

- [ ] **exp04_record_pipeline_plumbing** — Close the four lerobot upstream gaps (read_latest_depth, so_follower.get_observation, so_follower._cameras_ft, on-the-fly colorization). Only after exp02 confirms depth signal is worth the engineering.

- [ ] **exp05_gripper_depth_pilot_dataset** — small (~10–20 episode) RGB-D dataset using the closed pipeline.

- [ ] **exp06_v016_pilot_train** — small fine-tune. Compare to v014/v015 baseline. If signal is weak → consider escalating to v017 with B2 side-branch.

### Completed
- [x] **exp01_smolvla_multicam_mechanics** — accepted 2026-04-29. Q2-B1 trivial / B3 dead / B2 moderate. Co-training requires backfill of old data + `<key>_padding_mask` hook.

### Open questions deferred
- Whether v013 already implicitly learned depth from 4-camera stereo parallax — revisit during synthesis.
- Co-training schedule (uniform mix vs warm-up) — postpone until pilot training.
- Q1 path (estimated-everywhere vs backfill-then-real-on-new) — exp02 directly informs this.
