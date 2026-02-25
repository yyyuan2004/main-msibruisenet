#!/bin/bash
# =============================================================================
# ablation.sh - One-click ablation experiment runner for MSI-BruiseNet
#
# Runs multiple training configurations to evaluate the contribution of
# each component: attention module type, LSAA kernel size, fusion method,
# and bypass (residual) connection.
#
# Usage:
#   chmod +x scripts/ablation.sh
#   bash scripts/ablation.sh
# =============================================================================

set -e
CONFIG="configs/config.yaml"

echo "=========================================="
echo " MSI-BruiseNet Ablation Experiments"
echo "=========================================="

# --- 1. Attention Module Ablation ---
echo ""
echo "[Ablation 1/4] Attention module type"
for attn in lsaa se cbam eca none; do
    echo "  Running: attention=$attn"
    python scripts/train.py --config "$CONFIG" \
        --override model.attention="$attn" \
        --tag "ablation_attention_${attn}"
done

# --- 2. LSAA Kernel Size Ablation ---
echo ""
echo "[Ablation 2/4] LSAA kernel size"
for k in 3 5 7 9; do
    echo "  Running: lsaa_kernel_size=$k"
    python scripts/train.py --config "$CONFIG" \
        --override model.attention=lsaa model.lsaa_kernel_size="$k" \
        --tag "ablation_kernel_${k}"
done

# --- 3. Fusion Method Ablation ---
echo ""
echo "[Ablation 3/4] Fusion method (ConvGLU vs Concat)"
for fusion in convglu concat; do
    echo "  Running: fusion=$fusion"
    python scripts/train.py --config "$CONFIG" \
        --override model.attention=lsaa model.fusion="$fusion" \
        --tag "ablation_fusion_${fusion}"
done

# --- 4. Bypass (Residual Connection) Ablation ---
echo ""
echo "[Ablation 4/4] LSAA bypass residual connection"
for bypass in true false; do
    echo "  Running: bypass=$bypass"
    python scripts/train.py --config "$CONFIG" \
        --override model.attention=lsaa model.bypass="$bypass" \
        --tag "ablation_bypass_${bypass}"
done

echo ""
echo "=========================================="
echo " All ablation experiments completed."
echo " Results saved to outputs/results/"
echo "=========================================="
