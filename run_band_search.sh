#!/bin/bash
# ==============================================================================
# HSI Band Range Exhaustive Search Pipeline (自动化波段范围穷举)
#
# 在 204-channel 高光谱数据上，依次执行多组 (range, k) 配置：
#   1. 在指定范围 [start, end] 内均匀采样 9 个波段
#   2. 对 C(9, k) 所有组合做穷举训练搜索
#   3. 输出 JSON 结果 + 3-panel 频率热力图
#
# 用法:
#   bash run_band_search.sh                    # 使用下方默认配置
#   bash run_band_search.sh --epochs 50        # 自定义 epoch 数
#   bash run_band_search.sh --seed 123         # 自定义随机种子
#
# 自定义搜索范围: 修改下方 SEARCH_CONFIGS 数组
#   格式: "start-end:k"
#   示例: "60-110:4" 表示在 [60, 110] 范围均分 9 bands，穷举 C(9,4)
# ==============================================================================

set -e

# ===== 用户配置区 =====

# 搜索配置列表 (格式: "start-end:k")
# 每一项依次执行，跑完一个再跑下一个
SEARCH_CONFIGS=(
    "60-110:4"
    "70-100:4"
    "75-95:4"
    "80-100:4"
    "95-115:4"
)

# 基础配置
# Note: configs now use _base inheritance. The Python script uses
# utils.config.load_config() to resolve the full config.
BASE_CONFIG="configs/baseline.yaml"
SEED=42
EPOCHS=30
OUTPUT_DIR="outputs/band_range_search"

# HSI 数据目录 (留空则使用 config 中的 data_dir)
DATA_DIR=""

# ===== 解析命令行参数 =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

# 构建 --searches 参数 (逗号分隔)
SEARCHES=""
for cfg in "${SEARCH_CONFIGS[@]}"; do
    if [ -z "$SEARCHES" ]; then
        SEARCHES="${cfg}"
    else
        SEARCHES="${SEARCHES},${cfg}"
    fi
done

echo "=============================================="
echo " HSI Band Range Exhaustive Search Pipeline"
echo "=============================================="
echo " Config:     ${BASE_CONFIG}"
echo " Seed:       ${SEED}"
echo " Epochs:     ${EPOCHS}"
echo " Output:     ${OUTPUT_DIR}"
echo " Searches:   ${#SEARCH_CONFIGS[@]} configs"
for cfg in "${SEARCH_CONFIGS[@]}"; do
    echo "   - ${cfg}"
done
echo "=============================================="

# 构建命令 (-u: 不缓冲 stdout, 让每个 epoch 的进度立即可见)
CMD="python -u scripts/band_range_search.py \
    --config ${BASE_CONFIG} \
    --searches ${SEARCHES} \
    --seed ${SEED} \
    --epochs ${EPOCHS} \
    --output_dir ${OUTPUT_DIR}"

if [ -n "$DATA_DIR" ]; then
    CMD="${CMD} --data_dir ${DATA_DIR}"
fi

echo ""
echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=============================================="
echo " Band range search pipeline completed!"
echo " Results: ${OUTPUT_DIR}/"
echo "=============================================="
