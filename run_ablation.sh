#!/bin/bash
# ==============================================================================
# Ablation Study Runner
# Runs all 5 configurations x 3 seeds for the MobileNetV2-UNet MSI segmentation.
# ==============================================================================

set -e

CONFIGS=("baseline" "se" "spconv" "convglu" "spconv_se")
SEEDS=(42 123 456)

echo "=============================================="
echo "MSI Bruise Baseline Ablation Study"
echo "Configs: ${CONFIGS[*]}"
echo "Seeds:   ${SEEDS[*]}"
echo "=============================================="

# Step 1: Run spectral pre-analysis (if data exists)
if [ -d "data/images" ]; then
    echo ""
    echo "[Pre-analysis] Running spectral analysis..."
    python utils/spectral_analysis.py \
        --data_dir data \
        --output_dir outputs/spectral_analysis
    echo "[Pre-analysis] Done."
fi

# Step 2: Train all configurations
for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "=============================================="
        echo "Training: ${config} | Seed: ${seed}"
        echo "=============================================="
        python train.py \
            --config "configs/${config}.yaml" \
            --seed "${seed}" \
            --output_dir "outputs/${config}_seed${seed}"
    done
done

# Step 3: Evaluate all configurations on test set
echo ""
echo "=============================================="
echo "Running evaluation on test sets..."
echo "=============================================="

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CKPT="outputs/${config}_seed${seed}/checkpoints/best_model.pth"
        if [ -f "${CKPT}" ]; then
            echo "Evaluating: ${config} | Seed: ${seed}"
            python eval.py \
                --checkpoint "${CKPT}" \
                --seed "${seed}" \
                --split test \
                --output_dir "outputs/${config}_seed${seed}/eval_results" \
                --num_vis 10
        else
            echo "WARNING: Checkpoint not found: ${CKPT}"
        fi
    done
done

# Step 4: Aggregate results
echo ""
echo "=============================================="
echo "Aggregating results..."
echo "=============================================="
python aggregate_results.py

echo ""
echo "All ablation experiments completed!"
