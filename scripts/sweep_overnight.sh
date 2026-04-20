#!/usr/bin/env bash
# Overnight distillation sweep. Run 4 model configs sequentially, producing
# training/<run>/ folders + comparison artifacts.
#
# Usage:
#   scripts/sweep_overnight.sh               # all 4 runs
#   scripts/sweep_overnight.sh m1 m2         # only m1 and m2
#
# Each run: ~60-90 min (4 cams sequential, max-epochs 40, 5090).
# Total overnight budget: ~5 hours.
#
# Artifacts per run:
#   training/<run>/{cam}/{best.pt, last.pt, metrics.csv, training.png, config.json}
#   training/<run>/comparison_student_vs_dino.png
#   training/<run>/tracking_ep210.mp4
#   training/<run>/training_compare_<prev>_vs_<this>.png  (if --compare-with prev)
#   training/_sweep_summary.md   (final summary table)

set -euo pipefail

export CONDA_RUN="conda run -n lerobot --no-capture-output python -u"
REPO_ROOT="/home/may33/projects/ml_portfolio/robotics"
cd "$REPO_ROOT"

TRAINING_ROOT="/home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/detection/distillation/training"

# Run registry: name | cli args | comparison target
declare -A RUN_ARGS
RUN_ARGS[m1_baseline]="--model mobilenet_v3_small"
RUN_ARGS[m2_focal]="--model mobilenet_v3_small --focal-gamma 2.0"
RUN_ARGS[m3_aug]="--model mobilenet_v3_small --augment"
RUN_ARGS[m4_large]="--model mobilenet_v3_large"

declare -A RUN_COMPARE
RUN_COMPARE[m1_baseline]=""            # no comparison for first run
RUN_COMPARE[m2_focal]="m1_baseline"
RUN_COMPARE[m3_aug]="m1_baseline"
RUN_COMPARE[m4_large]="m1_baseline"

RUN_ORDER=("m1_baseline" "m2_focal" "m3_aug" "m4_large")

# Override run list if passed as args
if [[ $# -gt 0 ]]; then
    RUN_ORDER=("$@")
fi

log() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "  $(date +'%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
}

t0=$(date +%s)

for run in "${RUN_ORDER[@]}"; do
    log "RUN: $run"
    args="${RUN_ARGS[$run]}"
    log_file="/tmp/sweep_${run}.log"

    # Train 4 cams
    $CONDA_RUN -m vbti.logic.detection.distill train --all \
        --run "$run" \
        --max-epochs 40 --min-epochs 15 --patience 10 \
        --batch-size 128 --num-workers 4 --lr 1e-3 \
        $args 2>&1 | tee "$log_file"

    # Render artifacts
    compare_with="${RUN_COMPARE[$run]}"
    compare_flag=""
    if [[ -n "$compare_with" ]]; then
        compare_flag="--compare-with $compare_with"
    fi
    log "RENDER: $run  $compare_flag"
    $CONDA_RUN scripts/distill_compare.py \
        --run "$run" \
        $compare_flag \
        2>&1 | tee -a "$log_file" || true
done

t1=$(date +%s)
total_min=$(( (t1 - t0) / 60 ))
log "SWEEP COMPLETE in ${total_min} minutes"

# Summary markdown
summary="$TRAINING_ROOT/_sweep_summary.md"
{
    echo "# Distillation sweep — $(date +'%Y-%m-%d')"
    echo ""
    echo "Total time: ${total_min} min"
    echo ""
    echo "| Run | Config |"
    echo "|---|---|"
    for run in "${RUN_ORDER[@]}"; do
        echo "| \`$run\` | ${RUN_ARGS[$run]} |"
    done
    echo ""
    echo "## Artifacts"
    echo ""
    for run in "${RUN_ORDER[@]}"; do
        echo "### $run"
        echo "- Comparison grid: \`training/$run/comparison_student_vs_dino.png\`"
        echo "- Tracking video: \`training/$run/tracking_ep210.mp4\`"
        if [[ -n "${RUN_COMPARE[$run]}" ]]; then
            echo "- Curves vs ${RUN_COMPARE[$run]}: \`training/$run/training_compare_${RUN_COMPARE[$run]}_vs_$run.png\`"
        fi
        echo ""
    done
    echo "## Quick metric dumps"
    echo ""
    for run in "${RUN_ORDER[@]}"; do
        echo "### $run"
        echo '```'
        for cam in left right top gripper; do
            csv_path="$TRAINING_ROOT/$run/$cam/metrics.csv"
            if [[ -f "$csv_path" ]]; then
                # last line = final epoch metrics
                best_line=$(tail -1 "$csv_path")
                echo "$cam: $best_line"
            fi
        done
        echo '```'
        echo ""
    done
} > "$summary"

echo ""
echo "Summary: $summary"
