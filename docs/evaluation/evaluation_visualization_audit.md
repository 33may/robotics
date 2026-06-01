# Evaluation visualization audit — v020–v024 checkpoint sweep

This document explains exactly what was generated, which data was used, how checkpoints were mapped, and how to verify the outputs in this folder.

## Output folder

All generated artifacts are in:

```text
/home/may33/Documents/vbti/vbti/oficial/docs/evaluation
```

Generated files:

```text
evaluation_visualization_report.md          report-ready figure section with captions
evaluation_visualization_audit.md           this verification/audit document
generate_evaluation_visualizations.py       reproducible plotting utility
tables/checkpoint_sweep_summary.csv         one row per evaluated checkpoint
tables/checkpoint_sweep_trials.csv          one row per evaluated trial
figures/*.png                               raster figures for quick use in docs/slides
figures/*.pdf                               vector-ish publication exports
figures/*.svg                               editable vector exports
```

The figure directory contains 14 PNG figure files for review. PDF/SVG export is disabled for now to keep the review folder simple.

## Source data

The script parses real-robot checkpoint-sweep `session.json` files from:

```text
/home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/<version>/eval_sessions/*/session.json
```

Included model versions:

| Version | Intended comparison role | Dataset episodes used in plots |
|---|---|---:|
| v021 | smallest subsample | 48 |
| v022 | 1/8 subsample | 96 |
| v023 | 1/4 subsample | 192 |
| v024 | 1/2 subsample | 383 |
| v020 | full dataset baseline | 765 |

Only `protocol = checkpoint_sweep` sessions are used for the main v020–v024 comparison. Older `dual_cup_30` and `dual_cup_60` sessions in v020 are not used in the heatmaps or learning curves because they are different protocols.

## Canonical checkpoint mapping

The comparison uses six shared normalized checkpoint columns: `E2`, `E4`, `E6`, `E8`, `E10`, `E12`.

For v021–v024, the columns correspond to the six checkpoint-sweep saves in each version. For v020, the columns use the convenience epoch-boundary mapping discussed during evaluation:

| Normalized column | v020 checkpoint | v020 result |
|---:|---:|---:|
| E2 | 80,000 | 12/20 = 60% |
| E4 | 110,000 | 13/20 = 65% |
| E6 | 170,000 | 18/20 = 90% |
| E8 | 220,000 | 18/20 = 90% |
| E10 | 280,000 | 19/20 = 95% |
| E12 | 336,940 | 19/20 = 95% |

There is also a v020 `180,000` checkpoint-sweep session in the raw data:

```text
v020, step_180000, 13/20 = 65%
```

This is preserved in `tables/checkpoint_sweep_summary.csv` with `include_heatmap=False`, but it is not plotted in the canonical six-column heatmap because it does not belong to the agreed v020 mapping.

## Canonical result table

This is the exact table used for the main heatmap and normalized learning curves:

| Version | Dataset | E2 | E4 | E6 | E8 | E10 | E12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v021 | 48 ep | 0/20 (0%) | 0/20 (0%) | 1/20 (5%) | 0/20 (0%) | 0/20 (0%) | 2/20 (10%) |
| v022 | 96 ep | 0/20 (0%) | 2/20 (10%) | 1/20 (5%) | 4/20 (20%) | 4/20 (20%) | 2/20 (10%) |
| v023 | 192 ep | 4/20 (20%) | 4/20 (20%) | 6/20 (30%) | 6/20 (30%) | 6/20 (30%) | 7/20 (35%) |
| v024 | 383 ep | 6/20 (30%) | 12/20 (60%) | 11/20 (55%) | 13/20 (65%) | 13/20 (65%) | 14/20 (70%) |
| v020 | 765 ep | 12/20 (60%) | 13/20 (65%) | 18/20 (90%) | 18/20 (90%) | 19/20 (95%) | 19/20 (95%) |

## Confidence intervals

Each checkpoint-sweep cell has `n=20` physical-robot episodes. The plotting utility computes Wilson 95% confidence intervals using the same formula as `eval_helpers.py`:

```python
def wilson(s, n, z=1.96):
    p = s / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return centre - half, centre + half
```

Important interpretation caveat:

- 20 episodes is enough for broad comparison signals.
- 20 episodes is not enough to treat adjacent 5–10 percentage-point differences as strong evidence.
- The report should emphasize large patterns, such as v020 greatly outperforming small subsamples and v024 outperforming the smaller subsampled runs.

## Figures generated

Each figure is currently exported only as `.png`.

### 01 — `01_success_rate_heatmap`

Main overview heatmap: model version by normalized checkpoint column. This is the clearest thesis/report figure for the dataset-scale story.

Shows:

- v020 full dataset starts high and reaches 95%.
- v021–v023 remain low.
- v024 is the strongest subsampled model and reaches 70% at the final checkpoint.

Use when explaining the main evaluation outcome.

### 02 — `02_success_rate_heatmap_with_ci`

Same heatmap, but each cell shows success count and Wilson interval. This is the more statistically careful version.

Use when the assessor may ask about uncertainty or small sample size.

### 03 — `03_learning_curves_normalized_epoch`

Line chart over normalized epoch/checkpoint labels.

Shows:

- Learning trajectory shape for each version.
- v020 improves strongly after the first two columns.
- v024 jumps early and then improves more slowly.
- v021–v023 remain much lower.

Use when discussing training progress rather than only final performance.

### 04 — `04_learning_curves_raw_steps`

Line chart over raw training steps.

Shows:

- v020 was evaluated at much larger step counts than v021–v024.
- Raw-step and normalized-column views answer different questions.

Use as a transparency figure, not as the main comparison figure.

### 05 — `05_best_success_vs_dataset_size`

Scatter plot of dataset episodes versus best observed success rate, with Wilson intervals.

Shows:

- Best observed performance increases with dataset size.
- v020 reaches 19/20.
- v024 is the strongest subsampled model at 14/20.

Use for the dataset-scaling hypothesis.

### 06 — `06_final_vs_best_success_bar`

Grouped bar chart comparing each version's final checkpoint to its best checkpoint.

Shows:

- Whether a model peaked before the final checkpoint.
- v020 final and best are both 19/20.
- v024 final and best are both 14/20 in this sweep.

Use to avoid overclaiming if final checkpoint is not the peak.

### 07 — `07_improvement_from_first_checkpoint`

Line chart showing percentage-point improvement from the first evaluated checkpoint.

Shows:

- v020 gains +35 percentage points from E2 to E12.
- v024 gains +40 percentage points from E2 to E12.
- Small subsamples have weaker or noisy improvement.

Use to discuss whether additional training helped once the dataset had enough coverage.

### 08 — `08_scene_success_heatmap`

Aggregated success by scene type across all recorded checkpoint-sweep trials.

Shows whether failures concentrate in `single_red`, `both_red`, or other scene tags present in the protocol.

Use to separate scene/language/color sensitivity from general manipulation ability.

### 09 — `09_steps_distribution_success_failure`

Boxplot of episode step counts for successful versus failed trials.

Shows:

- Failures often take longer than successes.
- This supports the interpretation that many failures are retry/placement/convergence failures, not immediate perception failures.

Use when discussing failure modes.

### 10 — `10_final_trial_outcome_matrix`

Binary matrix of final-checkpoint trial outcomes by fixed protocol trial id.

Shows:

- Which exact protocol trials are hard across versions.
- v020 final fails only one trial.
- Low-data versions fail many of the same fixed trial positions.

Use to argue that the protocol is fixed and comparable across models.

### 11 — `11_spatial_outcome_scatter_final`

Scatter plots of final-checkpoint outcomes by initial duck pixel position.

Shows:

- Whether failures cluster in parts of the workspace.
- v020 succeeds across most initial positions.
- v021/v022 fail broadly, indicating capability/data limitation rather than one small spatial region.

Use for spatial failure-mode analysis.

### 12 — `12_pointing_bucket_success`

Bar chart grouping trials by initial duck orientation relative to the target cup: `toward`, `sideways`, `away`.

Shows whether initial orientation predicts success.

Use to test the hypothesis that some failures come from orientation-dependent manipulation difficulty.

### 13 — `13_target_side_success`

Bar chart of success rate by target side.

Shows whether left/right target position creates asymmetry.

Use for side-bias analysis in dual-cup scenes.

### 14 — `13_target_closer_success`

Bar chart of success rate by whether the target cup is closer/farther/single.

Shows whether target distance/ambiguity changes success.

Use for target-distance/ambiguity analysis.

## Visual QA performed

The generated plots were visually inspected through the image reader. Specific fixes made after inspection:

1. Removed an overlapping heatmap footnote that collided with the x-axis label.
2. Adjusted the dataset-size scatter plot so the v020 label and error bar fit cleanly.
3. Changed the dataset-size x-axis from powers-of-two tick labels to explicit episode counts: `48`, `96`, `192`, `383`, `765`.
4. Re-rendered all PNG exports after the fixes.
5. Re-ran the saved plotting utility from the official evaluation folder to confirm reproducibility.

## Reproducibility command

From any working directory with the same Python environment:

```bash
python /home/may33/Documents/vbti/vbti/oficial/docs/evaluation/generate_evaluation_visualizations.py
```

Expected behavior:

- Rewrites `tables/checkpoint_sweep_summary.csv`.
- Rewrites `tables/checkpoint_sweep_trials.csv`.
- Rewrites all figure exports in `figures/`.
- Prints the canonical checkpoint table to stdout.

## Verification checklist

Use this checklist to verify the assets before using them in the report:

- [ ] Open `tables/checkpoint_sweep_summary.csv` and confirm 31 rows total: header + 30 canonical/extra data rows.
- [ ] Confirm v020 has six heatmap rows plus one extra non-heatmap row at `180000`.
- [ ] Confirm v020 `80000` is `12/20 = 60%`.
- [ ] Confirm v020 `336940` is `19/20 = 95%`.
- [ ] Confirm v024 final `167940` is `14/20 = 70%`.
- [ ] Confirm every main heatmap cell has `n = 20`.
- [ ] Open `figures/01_success_rate_heatmap.png` and confirm the x-axis label does not overlap text.
- [ ] Open `figures/02_success_rate_heatmap_with_ci.png` and confirm intervals are visible.
- [ ] Open `figures/05_best_success_vs_dataset_size.png` and confirm x-axis ticks show actual episode counts.
- [ ] Open `evaluation_visualization_report.md` and confirm figure links render in your Markdown viewer.

## Main report takeaway

The central finding supported by these visualizations is that dataset coverage strongly affects real-robot checkpoint-sweep performance. The full-data v020 model reaches 19/20 successes at the best/final checkpoints, while the strongest subsampled model, v024, reaches 14/20. Smaller subsamples remain substantially lower. Because each cell uses 20 trials, small adjacent differences should be treated cautiously, but the large gap between full-data and low-data models is visually and practically strong.

## Known limitations

1. Each checkpoint has only 20 real-robot trials, so confidence intervals are wide.
2. The normalized checkpoint columns make v020 comparable to v021–v024 for visualization, but raw training step counts differ substantially.
3. The `180000` v020 session exists but is intentionally excluded from the canonical heatmap mapping.
4. These plots summarize the fixed `checkpoint_sweep` protocol only; they do not include older v020 `dual_cup_30` or `dual_cup_60` results.
