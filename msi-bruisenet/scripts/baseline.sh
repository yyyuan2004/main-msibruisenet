#!/bin/bash
# =============================================================================
# baseline.sh — Baseline comparison batch script for MSI-BruiseNet
# =============================================================================
# Runs the proposed method and all baseline variants (UNet backbone with
# different attention modules) for fair comparison.
#
# Usage:
#   bash scripts/baseline.sh
#   bash scripts/baseline.sh configs/config.yaml
# =============================================================================

set -e

CONFIG="${1:-configs/config.yaml}"
echo "============================================"
echo "MSI-BruiseNet Baseline Comparisons"
echo "Config: ${CONFIG}"
echo "============================================"

# ---------- Baseline methods ----------
# UNet (vanilla)         — no attention, concat fusion
# UNet + SE              — SE attention, concat fusion
# UNet + CBAM            — CBAM attention, concat fusion
# UNet + ECA             — ECA attention, concat fusion
# MSI-BruiseNet (ours)   — LSAA attention, ConvGLU fusion

echo ""
echo ">>> Baseline: UNet (vanilla, no attention, concat fusion)"
python scripts/train.py --config "${CONFIG}" \
    --override model.attention=none model.fusion=concat \
    --tag "baseline_unet_vanilla"

echo ""
echo ">>> Baseline: UNet + SE"
python scripts/train.py --config "${CONFIG}" \
    --override model.attention=se model.fusion=concat \
    --tag "baseline_unet_se"

echo ""
echo ">>> Baseline: UNet + CBAM"
python scripts/train.py --config "${CONFIG}" \
    --override model.attention=cbam model.fusion=concat \
    --tag "baseline_unet_cbam"

echo ""
echo ">>> Baseline: UNet + ECA"
python scripts/train.py --config "${CONFIG}" \
    --override model.attention=eca model.fusion=concat \
    --tag "baseline_unet_eca"

echo ""
echo ">>> Our Method: MSI-BruiseNet (LSAA + ConvGLU)"
python scripts/train.py --config "${CONFIG}" \
    --override model.attention=lsaa model.fusion=convglu model.bypass=true \
    --tag "baseline_msi_bruisenet"

echo ""
echo "============================================"
echo "All baseline experiments completed!"
echo "Results saved to outputs/results/"
echo "============================================"
