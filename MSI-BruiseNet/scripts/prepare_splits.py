"""Generate 5-fold split JSON for MSI dataset.

Key I/O:
- Input dir from config: data.image_dir
- Output JSON to config data.split_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import KFold

from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_dir = Path(cfg["data"]["image_dir"])
    split_dir = Path(cfg["data"]["split_dir"])
    split_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted([p.stem for p in image_dir.glob("*.npy")])
    kf = KFold(n_splits=int(cfg["train"]["num_folds"]), shuffle=True, random_state=args.seed)
    folds = {}
    for i, (tr, va) in enumerate(kf.split(ids)):
        folds[f"fold_{i}"] = {"train": [ids[j] for j in tr], "val": [ids[j] for j in va]}

    with (split_dir / "folds.json").open("w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    with (split_dir / f"folds_seed{args.seed}.json").open("w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)


if __name__ == "__main__":
    main()
