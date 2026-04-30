#!/bin/bash
# ==============================================================================
# SDA v2 Ablation Study Runner
#
# Paired comparison of SDA v2 insertion positions vs. baseline:
#   baseline → sda_input → sda_s2 → sda_decoder → sda_multiscale
#
# Same seed / split / k-fold for all configs → paired comparison avoids
# mistaking split variance for module improvement.
#
# Usage:
#   bash run_sda_ablation.sh                          # 7:3 split
#   bash run_sda_ablation.sh --kfold 5                # 5-fold CV
#   bash run_sda_ablation.sh --kfold 5 --seeds 42,123 # custom seeds
# ==============================================================================

set -e

CONFIGS=(
    "baseline"
    "sda_input"
    "sda_s2"
    "sda_decoder"
    "sda_multiscale"
)

SEEDS=(42 123 456)

# Parse arguments
KFOLD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kfold)
            KFOLD="$2"
            shift 2
            ;;
        --seeds)
            IFS=',' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

KFOLD_FLAG=""
MODE_DESC="default 7:3 split"
if [ "$KFOLD" -gt 0 ]; then
    KFOLD_FLAG="--kfold ${KFOLD}"
    MODE_DESC="${KFOLD}-fold cross-validation"
fi

echo "=============================================="
echo " SDA v2 Ablation Study"
echo " Configs:  ${CONFIGS[*]}"
echo " Seeds:    ${SEEDS[*]}"
echo " Mode:     ${MODE_DESC}"
echo " Metrics:  class-1 IoU, F1, Precision, Recall"
echo "=============================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# SEED = ${seed}"
    echo "############################################################"

    for config in "${CONFIGS[@]}"; do
        echo ""
        echo "----------------------------------------------"
        echo " ${config} | seed=${seed} | ${MODE_DESC}"
        echo "----------------------------------------------"

        if [ "$KFOLD" -gt 0 ]; then
            OUTPUT_DIR="outputs/${config}_seed${seed}_kfold${KFOLD}"
        else
            OUTPUT_DIR="outputs/${config}_seed${seed}"
        fi

        python train_eval.py \
            --config "configs/${config}.yaml" \
            --seed "${seed}" \
            --output_dir "${OUTPUT_DIR}" \
            ${KFOLD_FLAG}
    done
done

# Aggregate results (single-split mode)
if [ "$KFOLD" -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo " Aggregating results..."
    echo "=============================================="
    python aggregate_results.py || echo "(aggregate_results.py skipped/failed)"
fi

echo ""
echo "=============================================="
echo " SDA v2 ablation complete!"
echo " Compare: baseline vs sda_input vs sda_s2 vs sda_decoder vs sda_multiscale"
echo "=============================================="
