#!/bin/bash
# =============================================================================
# ablation.sh — Ablation experiment batch script for MSI-BruiseNet
# =============================================================================
# Runs systematic ablation studies to evaluate the contribution of each
# component in the proposed architecture.
#
# Usage:
#   bash scripts/ablation.sh
#   bash scripts/ablation.sh configs/config.yaml
# =============================================================================

set -e

CONFIG="${1:-configs/config.yaml}"
echo "============================================"
echo "MSI-BruiseNet Ablation Experiments"
echo "Config: ${CONFIG}"
echo "============================================"

# ---------- 1. Attention module ablation ----------
echo ""
echo ">>> Ablation: Attention Modules"
echo "    (LSAA vs SE vs CBAM vs ECA vs None)"
for attn in lsaa se cbam eca none; do
    echo ""
    echo "--- Running attention=${attn} ---"
    python scripts/train.py --config "${CONFIG}" \
        --override model.attention="${attn}" \
        --tag "ablation_attention_${attn}"
done

# ---------- 2. Fusion module ablation ----------
echo ""
echo ">>> Ablation: Fusion Modules"
echo "    (ConvGLU vs Concat)"
for fusion in convglu concat; do
    echo ""
    echo "--- Running fusion=${fusion} ---"
    python scripts/train.py --config "${CONFIG}" \
        --override model.fusion="${fusion}" \
        --tag "ablation_fusion_${fusion}"
done

# ---------- 3. LSAA bypass (residual connection) ablation ----------
echo ""
echo ">>> Ablation: LSAA Bypass Residual Connection"
for bypass in true false; do
    echo ""
    echo "--- Running bypass=${bypass} ---"
    python scripts/train.py --config "${CONFIG}" \
        --override model.bypass="${bypass}" \
        --tag "ablation_bypass_${bypass}"
done

# ---------- 4. LSAA kernel size ablation ----------
echo ""
echo ">>> Ablation: LSAA Local Window Size k"
for k in 3 5 7 9; do
    echo ""
    echo "--- Running lsaa_kernel_size=${k} ---"
    python scripts/train.py --config "${CONFIG}" \
        --override model.lsaa_kernel_size="${k}" \
        --tag "ablation_kernel_${k}"
done

# ---------- 5. Auxiliary loss weight ablation ----------
echo ""
echo ">>> Ablation: Auxiliary Loss Weight"
for aux_w in 0.0 0.1 0.5 1.0; do
    echo ""
    echo "--- Running aux_weight=${aux_w} ---"
    python scripts/train.py --config "${CONFIG}" \
        --override loss.aux_weight="${aux_w}" \
        --tag "ablation_auxloss_${aux_w}"
done

echo ""
echo "============================================"
echo "All ablation experiments completed!"
echo "Results saved to outputs/results/"
echo "============================================"
