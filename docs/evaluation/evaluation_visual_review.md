# Evaluation visualization review board

Use this document to review all generated figures in one place. Each figure has an Obsidian-style embedded image and a feedback block.

## Canonical data table

| Version | Dataset | E2 | E4 | E6 | E8 | E10 | E12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v021 | 48 ep | `3619` 0/20 (0%) | `7238` 0/20 (0%) | `10857` 1/20 (5%) | `14476` 0/20 (0%) | `18095` 0/20 (0%) | `21716` 2/20 (10%) |
| v022 | 96 ep | `7032` 0/20 (0%) | `14064` 2/20 (10%) | `21096` 1/20 (5%) | `28128` 4/20 (20%) | `35160` 4/20 (20%) | `42197` 2/20 (10%) |
| v023 | 192 ep | `14043` 4/20 (20%) | `28086` 4/20 (20%) | `42129` 6/20 (30%) | `56172` 6/20 (30%) | `70215` 6/20 (30%) | `84258` 7/20 (35%) |
| v024 | 383 ep | `27990` 6/20 (30%) | `55980` 12/20 (60%) | `83970` 11/20 (55%) | `111960` 13/20 (65%) | `139950` 13/20 (65%) | `167940` 14/20 (70%) |
| v020 | 765 ep | `80000` 12/20 (60%) | `110000` 13/20 (65%) | `170000` 18/20 (90%) | `220000` 18/20 (90%) | `280000` 19/20 (95%) | `336940` 19/20 (95%) |

## Figure review

### 01. Main overview heatmap

![[figures/01_success_rate_heatmap.png]]

**Purpose / reasoning:** Shows success rate by dataset scale and normalized checkpoint. This is the clearest thesis figure: increasing dataset coverage produces a strong performance jump, and v020 full data dominates.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [ ] Remove from final report

Notes:

v versioning is not clear, we need to preciseliy iclude what it is the dataset size diiferences


### 02. Heatmap with Wilson intervals

![[figures/02_success_rate_heatmap_with_ci.png]]

**Purpose / reasoning:** Adds the n=20 uncertainty directly in each cell. It matters because adjacent differences can be visually tempting but statistically weak; the broad v020 vs subsample gap is the robust message.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 03. Normalized learning curves

![[figures/03_learning_curves_normalized_epoch.png]]

**Purpose / reasoning:** Compares learning dynamics on the shared six-column schedule. It emphasizes that v020 is already strong early and that v024 continues improving, while v021–v023 remain lower.

**Feedback:**

- [ ] Keep as-is
- [x] Needs label/title changes
- [ ] Needs data/mapping check
- [ ] Remove from final report

Notes:

move the legend to the corner


### 04. Raw-step learning curves

![[figures/04_learning_curves_raw_steps.png]]

**Purpose / reasoning:** Shows the same data without epoch normalization. It matters as a transparency figure: v020 used much larger raw step counts, so normalized and raw views answer different questions.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 05. Best SR versus dataset size

![[figures/05_best_success_vs_dataset_size.png]]

**Purpose / reasoning:** Tests the dataset-scaling hypothesis directly. The log-scaled x-axis highlights that performance improves with more demonstrations, with the full dataset reaching 95%.

**Feedback:**

- [x] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [ ] Remove from final report

Notes:


### 06. Final versus best checkpoint

![[figures/06_final_vs_best_success_bar.png]]

**Purpose / reasoning:** Separates final checkpoint performance from peak observed performance. This prevents over-claiming when a model peaks before the final saved checkpoint.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 07. Improvement after first checkpoint

![[figures/07_improvement_from_first_checkpoint.png]]

**Purpose / reasoning:** Shows whether performance comes from more training or from dataset coverage. Large gains in v020/v024 suggest continued training helps once enough data exists.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 08. Scene-type robustness

![[figures/08_scene_success_heatmap.png]]

**Purpose / reasoning:** Aggregates failures by single-cup versus dual-cup scene. This probes whether errors are language/color selection failures or lower-level manipulation failures.

**Feedback:**

- [x] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [ ] Remove from final report

Notes:


### 09. Episode duration by outcome

![[figures/09_steps_distribution_success_failure.png]]

**Purpose / reasoning:** Compares steps taken for successes and failures. Longer failures indicate retry/placement difficulty rather than immediate perception collapse.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 10. Fixed-trial outcome matrix

![[figures/10_final_trial_outcome_matrix.png]]

**Purpose / reasoning:** Shows which protocol trials fail at final checkpoints. Since the protocol is fixed, recurring columns identify hard cases rather than random noise.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 11. Spatial outcome scatter

![[figures/11_spatial_outcome_scatter_final.png]]

**Purpose / reasoning:** Maps failures onto initial duck positions. This tests for workspace-region bias and helps diagnose spatial generalization limits.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 12. Orientation sensitivity

![[figures/12_pointing_bucket_success.png]]

**Purpose / reasoning:** Groups trials by whether the duck initially points toward, sideways, or away from the target cup. This probes an orientation-dependent manipulation hypothesis.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 13. Target side robustness

![[figures/13_target_side_success.png]]

**Purpose / reasoning:** Tests whether left/right target position affects performance in dual-cup scenes.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


### 14. Closer/farther target robustness

![[figures/13_target_closer_success.png]]

**Purpose / reasoning:** Tests whether the closer cup creates ambiguity or easier/harder reaches.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [x] Remove from final report

Notes:


## Global feedback

- [ ] Version order is correct: v021, v022, v023, v024, v020.
- [ ] PNG-only output is enough for review.
- [ ] Main report story is clear.
- [ ] Statistical caveat is clear enough.

Notes:
