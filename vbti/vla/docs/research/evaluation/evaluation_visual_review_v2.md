# Evaluation visualization review board — v2

This v2 applies the feedback from `evaluation_visual_review.md`: unclear versioning was fixed with explicit dataset-size labels, the curve legend was moved to a corner, and figures marked “Remove from final report” were removed from this review set.

## Canonical data table

| Version | Dataset meaning | E2 | E4 | E6 | E8 | E10 | E12 |
|---|---|---:|---:|---:|---:|---:|---:|
| v021 | 1/16 subsample (48 ep) | `3619` 0/20 (0%) | `7238` 0/20 (0%) | `10857` 1/20 (5%) | `14476` 0/20 (0%) | `18095` 0/20 (0%) | `21716` 2/20 (10%) |
| v022 | 1/8 subsample (96 ep) | `7032` 0/20 (0%) | `14064` 2/20 (10%) | `21096` 1/20 (5%) | `28128` 4/20 (20%) | `35160` 4/20 (20%) | `42197` 2/20 (10%) |
| v023 | 1/4 subsample (192 ep) | `14043` 4/20 (20%) | `28086` 4/20 (20%) | `42129` 6/20 (30%) | `56172` 6/20 (30%) | `70215` 6/20 (30%) | `84258` 7/20 (35%) |
| v024 | 1/2 subsample (383 ep) | `27990` 6/20 (30%) | `55980` 12/20 (60%) | `83970` 11/20 (55%) | `111960` 13/20 (65%) | `139950` 13/20 (65%) | `167940` 14/20 (70%) |
| v020 | Full dataset (765 ep) | `80000` 12/20 (60%) | `110000` 13/20 (65%) | `170000` 18/20 (90%) | `220000` 18/20 (90%) | `280000` 19/20 (95%) | `336940` 19/20 (95%) |

## Figures kept for v2 review

### 01. Main dataset-scale heatmap

![[figures_v2/v2_01_success_rate_heatmap_dataset_labels.png]]

**V2 change / reasoning:** Kept from v1, but y-axis now spells out exact dataset sizes and subsampling ratios so the version names are no longer ambiguous.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [ ] Remove from final report

Notes:


### 02. Normalized learning curves

![[figures_v2/v2_03_learning_curves_corner_legend.png]]

**V2 change / reasoning:** Kept with the legend moved to the upper-left corner and labels expanded to include dataset size.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [ ] Remove from final report

Notes:


### 03. Best success versus dataset size

![[figures_v2/v2_05_best_success_vs_dataset_size.png]]

**V2 change / reasoning:** Kept as-is conceptually because it directly supports the dataset-scaling hypothesis.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [ ] Remove from final report

Notes:


### 04. Scene-type robustness

![[figures_v2/v2_08_scene_success_heatmap_dataset_labels.png]]

**V2 change / reasoning:** Kept as failure-mode support; y-axis now also includes dataset sizes.

**Feedback:**

- [ ] Keep as-is
- [ ] Needs label/title changes
- [ ] Needs data/mapping check
- [ ] Remove from final report

Notes:


## Removed from v2 report set

- Heatmap with Wilson intervals — removed per feedback.
- Raw-step learning curves — removed per feedback.
- Final vs best bar chart — removed per feedback.
- Improvement from first checkpoint — removed per feedback.
- Episode duration, trial matrix, spatial scatter, orientation, target side, and target closer plots — removed per feedback.

## Global feedback

- [ ] Version/dataset-size labeling is now clear.
- [ ] Only the intended report figures remain.
- [ ] Legend placement is acceptable.
- [ ] v020 full-dataset baseline belongs at the bottom/end.

Notes:
