"""CLI wrapper for spectral pre-analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config
from utils.spectral_analysis import run_spectral_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_spectral_analysis(
        image_dir=Path(cfg["data"]["image_dir"]),
        mask_dir=Path(cfg["data"]["mask_dir"]),
        out_dir=Path(cfg["output"]["root"]) / "spectral_analysis",
    )


if __name__ == "__main__":
    main()
