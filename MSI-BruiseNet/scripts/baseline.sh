#!/bin/bash
# 基线对比方法
# UNet (vanilla), UNet+SE, UNet+CBAM, UNet+ECA, 本文方法
for attn in none se cbam eca lsaa; do
    python scripts/train.py --config configs/config.yaml \
        --override model.attention=$attn \
        --tag baseline_$attn
done
