# SmolVLA Training Analysis

This directory contains comprehensive analysis of SmolVLA policy training for the pick-place task.

## Problem Statement

After training SmolVLA for 10,000 steps on the pick_place dataset:
- **Some joints train successfully** (Joint 3 shows good predictions)
- **Other joints fail completely** (Joints 0, 2, 4, 5 predict near-constant values)

This is a **partial mode collapse** where certain action dimensions don't learn properly.

## Directory Structure

```
research/
├── analyze_predictions.py    # Main analysis script
├── plots/                     # Generated visualizations
│   ├── validation_timeseries.png
│   ├── train_timeseries.png
│   ├── validation_scatter.png
│   ├── train_scatter.png
│   └── error_histograms.png
├── data/                      # Saved data
│   ├── predictions.npz        # All predictions and ground truth
│   └── statistics.json        # Summary statistics
└── README.md                  # This file
```

## Running the Analysis

```bash
cd /home/may33/projects/ml_portfolio/robotics
python smolvla_in_isaac/research/analyze_predictions.py
```

This generates:
1. **Time-series plots**: GT vs Predicted actions over time for all 6 joints
2. **Scatter plots**: Predicted vs GT for each joint (with R² correlation)
3. **Error histograms**: Distribution of prediction errors
4. **Statistics**: Mean/max errors per joint

## Key Findings

### Validation Set Results
(To be filled after running analysis)

- Overall mean error: TBD
- Overall max error: TBD
- Per-joint errors: TBD

### Problematic Joints
(To be filled after running analysis)

Joints with high error (>10°):
- Joint X: TBD

Joints with low variation in predictions (mode collapse):
- Joint X: TBD

## Similar Previous Issues

This is the SAME problem we encountered before with other models:
- **Some action dimensions don't train**
- Predictions collapse to mean/constant values
- Issue persists across different model architectures

### Previous Root Causes Found
1. **Imbalanced action ranges** - some joints move little in training data
2. **Normalization issues** - certain joints have very different scales
3. **Loss weighting** - MSE loss doesn't weight all dimensions equally
4. **Training data distribution** - some joints don't vary much

## Hypotheses for Current Issue

### H1: Action Space Imbalance
- Dataset has limited variation in certain joints
- Model learns to predict mean value as it minimizes MSE loss

**Test**: Check action range per joint in training data

### H2: Normalization Problem
- MEAN_STD normalization may not work well for all joints
- Some joints have very small std → large normalized values

**Test**: Inspect normalization statistics from `stats.json`

### H3: Loss Function Issue
- MSE loss doesn't weight joints equally
- Joints with larger GT ranges dominate the loss

**Test**: Train with per-joint loss weighting

### H4: Model Capacity
- Vision encoder frozen → limited capacity
- Model can't learn fine-grained per-joint control

**Test**: Unfreeze vision encoder, increase model size

## Next Steps

### Investigation
1. [ ] Analyze training data distribution per joint
2. [ ] Check normalization statistics
3. [ ] Visualize actual robot movements to verify data quality
4. [ ] Compare working vs non-working joint characteristics

### Potential Fixes
1. [ ] **Per-joint loss weighting**: Weight loss by inverse of joint variance
2. [ ] **Unfreeze vision encoder**: Allow full model to train
3. [ ] **Reduce chunk size**: 50 → 25 actions (easier to learn)
4. [ ] **Collect more varied data**: Ensure all joints have good coverage
5. [ ] **Try different architecture**: Test with ACT or other policies
6. [ ] **Add regularization**: L2 on action outputs to prevent collapse

### Training Experiments
1. [ ] Continue training for 20-30k more steps
2. [ ] Train with smaller learning rate (1e-5 → 5e-6)
3. [ ] Train with unfrozen vision encoder
4. [ ] Train with per-joint weighted loss

## Data Files

### predictions.npz
Contains:
- `val_pred`: Validation predictions (N x 6)
- `val_gt`: Validation ground truth (N x 6)
- `val_obs`: Validation observations (N x 6)
- `train_pred`: Train predictions (N x 6)
- `train_gt`: Train ground truth (N x 6)
- `train_obs`: Train observations (N x 6)

### statistics.json
Contains:
- Mean error per joint (train and val)
- Max error per joint (train and val)
- Overall statistics

## References

- Dataset: `eternalmay33/pick_place_test` (39 episodes, 8159 frames)
- Model: SmolVLA base fine-tuned
- Training: 10,000 steps, batch_size=8, lr=1e-5, frozen vision encoder
- Checkpoint: `outputs/finetune/smolvla_pick_place/checkpoints/best/`
