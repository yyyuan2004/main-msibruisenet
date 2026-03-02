"""Aggregate ablation results across seeds and produce summary table."""

import json
import os
import sys
import numpy as np
from collections import defaultdict


CONFIGS = ["baseline", "se", "spconv", "convglu", "spconv_se"]
CONFIG_DISPLAY = {
    "baseline": "baseline",
    "se": "+SE",
    "spconv": "+1D-SpConv",
    "convglu": "+ConvGLU",
    "spconv_se": "+1D-SpConv+SE",
}
SEEDS = [42, 123, 456]
METRICS = ["mIoU", "F1_macro", "Precision_macro", "Recall_macro"]
METRIC_DISPLAY = {
    "mIoU": "mIoU(%)",
    "F1_macro": "F1(%)",
    "Precision_macro": "Precision(%)",
    "Recall_macro": "Recall(%)",
}


def load_results():
    """Load results from all experiment runs."""
    results = defaultdict(lambda: defaultdict(list))

    for config in CONFIGS:
        for seed in SEEDS:
            result_path = os.path.join(
                "outputs", f"{config}_seed{seed}", "eval_results", "results.json"
            )
            if not os.path.exists(result_path):
                print(f"WARNING: Missing results: {result_path}")
                continue

            with open(result_path, "r") as f:
                data = json.load(f)

            for metric in METRICS:
                if metric in data:
                    results[config][metric].append(data[metric])

    return results


def count_params(config):
    """Count model parameters for a config."""
    try:
        import yaml
        import torch
        from model.model import build_model

        config_path = os.path.join("configs", f"{config}.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        model = build_model(cfg)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        return params
    except Exception:
        return None


def main():
    results = load_results()

    if not results:
        print("No results found. Run training and evaluation first.")
        sys.exit(1)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Build summary table
    lines = []
    header = f"{'Config':<18} | {'Params(M)':>10}"
    for metric in METRICS:
        header += f" | {METRIC_DISPLAY[metric]:>16}"
    lines.append(header)
    lines.append("-" * len(header))

    for config in CONFIGS:
        display_name = CONFIG_DISPLAY.get(config, config)
        params = count_params(config)
        params_str = f"{params:.2f}" if params is not None else "-"

        row = f"{display_name:<18} | {params_str:>10}"

        for metric in METRICS:
            values = results[config].get(metric, [])
            if values:
                mean = np.mean(values) * 100
                std = np.std(values) * 100
                row += f" | {mean:>6.1f} ± {std:>4.1f}  "
            else:
                row += f" | {'N/A':>16}"

        lines.append(row)

    table = "\n".join(lines)
    print("\nAblation Study Results:")
    print("=" * len(lines[0]))
    print(table)
    print("=" * len(lines[0]))

    # Save table
    with open(os.path.join(output_dir, "ablation_table.txt"), "w") as f:
        f.write("Ablation Study Results\n")
        f.write("=" * len(lines[0]) + "\n")
        f.write(table + "\n")
        f.write("=" * len(lines[0]) + "\n")
        f.write(f"\nSeeds: {SEEDS}\n")
        f.write("Values: mean ± std over 3 seeds\n")

    print(f"\nTable saved to {os.path.join(output_dir, 'ablation_table.txt')}")


if __name__ == "__main__":
    main()
