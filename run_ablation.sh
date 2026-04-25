#!/bin/bash
# ==============================================================================
# Ablation Study Runner (自动化消融实验)
#
# 运行所有保留的 8 个 config x 多个随机种子的完整实验流程：
#   每个 seed 下跑完所有 config，再切换下一个 seed。
#   每个 (config, seed) 组合自动完成: 训练 → 评估(val set) → 指标曲线图
#
# 用法:
#   bash run_ablation.sh                          # 7:3 划分（默认）
#   bash run_ablation.sh --kfold 5                # 5-fold 交叉验证
#   bash run_ablation.sh --vis_augment            # 7:3 + 生成增强可视化
#   bash run_ablation.sh --kfold 5 --vis_augment  # 5-fold + 增强可视化
# ==============================================================================

set -e

# 保留的 8 个 config：4 个自研架构 + 4 个 cross-method 对照
CONFIGS=(
    "baseline"
    "spconv_se"
    "input_band_se"
    "global_branch"
    "smp_deeplabv3plus_mobilenetv2"
    "smp_fpn_efficientnetb0"
    "smp_unet_resnet34"
    "deeplabv3plus_fang"
)

SEEDS=(42 123 456)

# 解析命令行参数
KFOLD=0
VIS_AUGMENT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kfold)
            KFOLD="$2"
            shift 2
            ;;
        --vis_augment)
            VIS_AUGMENT="--vis_augment"
            shift
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
echo " MSI Bruise Ablation Study (Automated)"
echo " Configs: ${CONFIGS[*]}"
echo " Seeds:   ${SEEDS[*]}"
echo " Mode:    ${MODE_DESC}"
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
        echo " Train-Eval: ${config} | Seed: ${seed} | Mode: ${MODE_DESC}"
        echo "=============================================="

        EXTRA_FLAGS=""
        # 增强可视化只在第一次运行时生成
        if [ "$FIRST_RUN" = true ] && [ -n "$VIS_AUGMENT" ]; then
            EXTRA_FLAGS="--vis_augment"
            FIRST_RUN=false
        fi

        if [ "$KFOLD" -gt 0 ]; then
            OUTPUT_DIR="outputs/${config}_seed${seed}_kfold${KFOLD}"
        else
            OUTPUT_DIR="outputs/${config}_seed${seed}"
        fi

        python train_eval.py \
            --config "configs/${config}.yaml" \
            --seed "${seed}" \
            --output_dir "${OUTPUT_DIR}" \
            ${KFOLD_FLAG} \
            ${EXTRA_FLAGS}
    done
done

# Step 3: 汇总结果（仅 7:3 模式下，k-fold 模式各 run 已自带聚合）
if [ "$KFOLD" -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo " Aggregating results..."
    echo "=============================================="
    python aggregate_results.py || echo "(aggregate_results.py skipped/failed)"
fi

echo ""
echo "=============================================="
echo " All ablation experiments completed!"
echo "=============================================="
