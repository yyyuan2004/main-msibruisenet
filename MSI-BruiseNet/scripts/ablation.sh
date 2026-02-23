#!/bin/bash
# 消融实验：逐一替换/移除组件
for attn in lsaa se cbam eca none; do
    python scripts/train.py --config configs/config.yaml \
        --override model.attention=$attn \
        --tag ablation_attention_$attn
done

# LSAA 内部消融
# 1. 去除 ConvGLU → 简单 concat
python scripts/train.py --config configs/config.yaml \
    --override model.use_convglu=false \
    --tag ablation_no_convglu

# 2. 去除 Bypass 残差连接
python scripts/train.py --config configs/config.yaml \
    --override model.lsaa_bypass=false \
    --tag ablation_no_bypass

# 3. 不同窗口大小 k
for k in 3 5 7 9; do
    python scripts/train.py --config configs/config.yaml \
        --override model.lsaa_kernel_size=$k \
        --tag ablation_kernel_$k
done
