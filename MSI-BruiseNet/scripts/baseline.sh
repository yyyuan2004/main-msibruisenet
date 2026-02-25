#!/bin/bash
# =============================================================================
# baseline.sh - One-click baseline comparison runner for MSI-BruiseNet
#
# Trains UNet with different attention modules as baselines:
#   - UNet (vanilla, no attention)
#   - UNet + SE
#   - UNet + CBAM
#   - UNet + ECA
#   - MSI-BruiseNet (UNet + LSAA + ConvGLU, proposed method)
#
# Usage:
#   chmod +x scripts/baseline.sh
#   bash scripts/baseline.sh
# =============================================================================

set -e
CONFIG="configs/config.yaml"

echo "=========================================="
echo " MSI-BruiseNet Baseline Comparison"
echo "=========================================="

for attn in none se cbam eca lsaa; do
    echo ""
    echo "  Running baseline: attention=$attn"
    python scripts/train.py --config "$CONFIG" \
        --override model.attention="$attn" \
        --tag "baseline_${attn}"
done

echo ""
echo "=========================================="
echo " All baseline experiments completed."
echo " Results saved to outputs/results/"
echo "=========================================="
