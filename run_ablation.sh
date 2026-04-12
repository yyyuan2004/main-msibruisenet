#!/bin/bash
# ==============================================================================
# Ablation Study Runner (自动化消融实验)
#
# 运行所有消融配置 x 多个随机种子的完整实验流程：
#   每个 seed 下跑完所有 config，再切换下一个 seed。
#   每个 (config, seed) 组合自动完成: 训练 → 评估(val set) → 指标曲线图
#
# 用法:
#   bash run_ablation.sh                    # 运行全部
#   bash run_ablation.sh --vis_augment      # 额外生成增强可视化（仅首次）
# ==============================================================================

set -e

# 消融实验配置列表（不含 cross-method 对比实验）
CONFIGS=(
    "baseline"
    "se"
    "spconv"
    "spconv_se"
    "cbam"
    "aspp"
    "band_attn"
    "global_branch"
    "input_band_se"
    "input_band_se_with_se"
    "lcn_baseline"
    "pca_baseline"
)

SEEDS=(42 123 456)

# 是否生成增强可视化（从命令行参数获取）
VIS_AUGMENT=""
if [[ "$1" == "--vis_augment" ]]; then
    VIS_AUGMENT="--vis_augment"
fi

echo "=============================================="
echo " MSI Bruise Ablation Study (Automated)"
echo " Configs: ${CONFIGS[*]}"
echo " Seeds:   ${SEEDS[*]}"
echo " Strategy: all configs per seed, then next seed"
echo "=============================================="

# Step 1: 光谱预分析 (仅运行一次)
DATA_DIR=$(python -c "
import yaml
with open('configs/baseline.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg['data']['data_dir'])
" 2>/dev/null || echo "")

if [ -n "$DATA_DIR" ] && [ -d "${DATA_DIR}/images" ]; then
    echo ""
    echo "[Pre-analysis] Running spectral analysis..."
    python utils/spectral_analysis.py \
        --data_dir "${DATA_DIR}" \
        --output_dir outputs/spectral_analysis || true
    echo "[Pre-analysis] Done."
fi

# Step 2: 按 seed 分组，每个 seed 下依次跑完所有 config
FIRST_RUN=true
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# SEED = ${seed}"
    echo "############################################################"

    for config in "${CONFIGS[@]}"; do
        echo ""
        echo "=============================================="
        echo " Train-Eval: ${config} | Seed: ${seed}"
        echo "=============================================="

        EXTRA_FLAGS=""
        # 增强可视化只在第一次运行时生成
        if [ "$FIRST_RUN" = true ] && [ -n "$VIS_AUGMENT" ]; then
            EXTRA_FLAGS="--vis_augment"
            FIRST_RUN=false
        fi

        python train_eval.py \
            --config "configs/${config}.yaml" \
            --seed "${seed}" \
            --output_dir "outputs/${config}_seed${seed}" \
            ${EXTRA_FLAGS}
    done
done

# Step 3: Cross-method comparison configs (if they exist)
CROSS_CONFIGS=(
    "deeplabv3plus_fang"
    "smp_deeplabv3plus_mobilenetv2"
    "smp_fpn_efficientnetb0"
    "smp_unet_resnet34"
)

HAS_CROSS=false
for config in "${CROSS_CONFIGS[@]}"; do
    if [ -f "configs/${config}.yaml" ]; then
        HAS_CROSS=true
        break
    fi
done

if [ "$HAS_CROSS" = true ]; then
    echo ""
    echo "############################################################"
    echo "# Cross-Method Comparison Experiments"
    echo "############################################################"

    for seed in "${SEEDS[@]}"; do
        for config in "${CROSS_CONFIGS[@]}"; do
            if [ -f "configs/${config}.yaml" ]; then
                echo ""
                echo "=============================================="
                echo " Train-Eval: ${config} | Seed: ${seed}"
                echo "=============================================="
                python train_eval.py \
                    --config "configs/${config}.yaml" \
                    --seed "${seed}" \
                    --output_dir "outputs/${config}_seed${seed}"
            fi
        done
    done
fi

# Step 4: 汇总结果
echo ""
echo "=============================================="
echo " Aggregating results..."
echo "=============================================="
python aggregate_results.py

echo ""
echo "=============================================="
echo " All ablation experiments completed!"
echo "=============================================="
