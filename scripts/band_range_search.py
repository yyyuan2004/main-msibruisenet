"""高光谱波段范围穷举搜索脚本。

在 204-channel HSI .npy 数据上，从指定波段范围均匀采样 9 个波段，
然后对 C(9, k) 所有组合做穷举训练搜索，找出最优波段子集。

支持自动化流水线：按顺序执行多组 (range, k) 配置，每组结束后自动输出
JSON 结果 + 3-panel 频率热力图。

用法:
    # 单次搜索
    python scripts/band_range_search.py \
        --config configs/baseline.yaml \
        --data_dir /path/to/204ch_hsi \
        --range 60-110 --k 4

    # 流水线搜索（多组 range:k）
    python scripts/band_range_search.py \
        --config configs/baseline.yaml \
        --data_dir /path/to/204ch_hsi \
        --searches "60-110:4,70-100:4,75-95:3,95-115:4"
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MSIDataset, get_dataset_kwargs
from data.augment import get_train_transforms, get_val_transforms
from data.split import get_data_splits
from model.model import build_model
from model.loss import SegmentationLoss
from utils.metrics import SegmentationMetrics
from train import set_seed


def uniform_sample_bands(start, end, n=9):
    """在 [start, end] 范围内均匀采样 n 个波段 index。"""
    return np.linspace(start, end, n, dtype=int).tolist()


def train_and_eval(cfg, seed, band_indices, num_epochs, device,
                   verbose=True, combo_tag=""):
    """Train a model with given band_indices and return val class-1 IoU.

    Args:
        verbose: If True, print per-epoch train loss and val IoU.
        combo_tag: Optional prefix string for per-epoch print lines.
    """
    set_seed(seed)

    data_dir = cfg["data"]["data_dir"]
    splits = get_data_splits(
        data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        seed=seed,
    )

    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    ds_kwargs = get_dataset_kwargs(cfg)
    ds_kwargs["band_indices"] = list(band_indices)
    # Lazy-load: only read the selected bands from each .npy file (mmap).
    # Avoids loading all 200+ channels for each sample.
    ds_kwargs["lazy_load"] = True

    train_dataset = MSIDataset(
        splits["train"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=train_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )
    val_dataset = MSIDataset(
        splits["val"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=val_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )

    train_cfg = cfg["train"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
    )

    cfg_copy = json.loads(json.dumps(cfg))
    cfg_copy["data"]["num_channels"] = len(band_indices)
    model = build_model(cfg_copy).to(device)

    criterion = SegmentationLoss(
        loss_type=train_cfg.get("loss", "ce_dice"),
        ce_weight=train_cfg.get("ce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    best_iou = 0.0
    for epoch in range(1, num_epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, masks, _raw, _amask, _stems in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        avg_loss = running_loss / max(n_batches, 1)

        model.eval()
        metrics.reset()
        with torch.no_grad():
            for images, masks, _raw, _amask, _stems in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images).argmax(dim=1)
                metrics.update(preds, masks)
        results = metrics.compute()
        defect_iou = float(results["IoU_per_class"][1])
        if defect_iou > best_iou:
            best_iou = defect_iou

        if verbose:
            elapsed = time.time() - epoch_t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"    {combo_tag} epoch {epoch:>3}/{num_epochs} "
                  f"loss={avg_loss:.4f} val_IoU={defect_iou:.4f} "
                  f"best={best_iou:.4f} lr={lr:.2e} {elapsed:.1f}s",
                  flush=True)

    return best_iou


def run_single_search(cfg, seed, num_epochs, device,
                      range_start, range_end, k, output_dir):
    """执行单次 (range, k) 穷举搜索，返回 results dict。"""
    sampled_hsi_bands = uniform_sample_bands(range_start, range_end, n=9)
    combos = list(itertools.combinations(range(9), k))
    n_combos = len(combos)

    print(f"\n{'='*60}")
    print(f"  Range: [{range_start}, {range_end}] → 9 bands: {sampled_hsi_bands}")
    print(f"  C(9, {k}) = {n_combos} combinations")
    print(f"  Epochs/combo: {num_epochs}, Seed: {seed}")
    print(f"{'='*60}")

    results = []
    best_iou = 0.0
    best_combo = None
    total_start = time.time()

    for i, combo in enumerate(combos):
        hsi_indices = [sampled_hsi_bands[j] for j in combo]
        t0 = time.time()
        combo_tag = f"[combo {i+1}/{n_combos} hsi={hsi_indices}]"
        print(f"\n  {combo_tag} starting...", flush=True)
        iou = train_and_eval(
            cfg, seed, hsi_indices, num_epochs, device,
            verbose=True, combo_tag=combo_tag,
        )
        elapsed = time.time() - t0

        results.append({
            "local_idx": list(combo),
            "hsi_idx": hsi_indices,
            "iou": iou,
        })

        if iou > best_iou:
            best_iou = iou
            best_combo = combo

        best_hsi = [sampled_hsi_bands[j] for j in best_combo]
        print(f"  >> [{i+1}/{n_combos}] done | local={list(combo)} "
              f"hsi={hsi_indices} IoU={iou:.4f} "
              f"(best={best_iou:.4f} @ hsi={best_hsi}) {elapsed:.1f}s",
              flush=True)

    total_elapsed = time.time() - total_start
    results.sort(key=lambda x: x["iou"], reverse=True)

    range_tag = f"{range_start}-{range_end}"
    search_result = {
        "range": range_tag,
        "range_start": range_start,
        "range_end": range_end,
        "k": k,
        "seed": seed,
        "epochs_per_combo": num_epochs,
        "sampled_9bands_hsi": sampled_hsi_bands,
        "n_combinations": n_combos,
        "best_local_idx": list(best_combo),
        "best_hsi_idx": [sampled_hsi_bands[j] for j in best_combo],
        "best_iou": best_iou,
        "total_time_sec": round(total_elapsed, 1),
        "all_results": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"search_{range_tag}_k{k}.json")
    with open(json_path, "w") as f:
        json.dump(search_result, f, indent=2)
    print(f"\n  Results saved to {json_path}")
    print(f"  Best k={k} subset: hsi={search_result['best_hsi_idx']} "
          f"IoU={best_iou:.4f} ({total_elapsed:.1f}s total)")

    return search_result


def plot_search_results(search_result, output_dir):
    """生成 3-panel 可视化图：
    (a) Top combinations bar chart
    (b) Band enrichment heatmap
    (c) Band frequency bar chart
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots.")
        return

    range_tag = search_result["range"]
    k = search_result["k"]
    sampled_bands = search_result["sampled_9bands_hsi"]
    all_results = search_result["all_results"]
    n_total = len(all_results)

    # Top 10% threshold
    top_n = max(1, n_total // 10)
    top_results = all_results[:top_n]

    # --- Compute band frequency in top results ---
    band_freq = np.zeros(9)
    for r in top_results:
        for idx in r["local_idx"]:
            band_freq[idx] += 1
    band_freq_pct = band_freq / top_n * 100

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Top combinations IoU bar chart
    ax = axes[0]
    show_n = min(20, top_n)
    top_show = top_results[:show_n]
    labels = [str(r["hsi_idx"]) for r in top_show]
    ious = [r["iou"] for r in top_show]
    colors_a = plt.cm.RdYlGn(np.linspace(0.9, 0.5, show_n))
    bars = ax.barh(range(show_n), ious, color=colors_a)
    ax.set_yticks(range(show_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Class-1 IoU")
    ax.set_title(f"(a) Top-{show_n} Combinations\n[{range_tag}] k={k}")
    for bar, v in zip(bars, ious):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=7)

    # (b) Band enrichment heatmap (top 10% frequency matrix)
    ax = axes[1]
    freq_matrix = np.zeros((top_n, 9))
    for i, r in enumerate(top_results):
        for idx in r["local_idx"]:
            freq_matrix[i, idx] = 1.0

    cmap = LinearSegmentedColormap.from_list("enrich", ["#f0f0f0", "#d62728"])
    im = ax.imshow(freq_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"B{i}\n({sampled_bands[i]})" for i in range(9)],
                       fontsize=8)
    ax.set_ylabel(f"Top {top_n} combinations (sorted by IoU)")
    ax.set_title(f"(b) Band Enrichment Heatmap\n(top 10%, n={top_n})")
    if top_n > 40:
        ax.set_yticks([0, top_n // 4, top_n // 2, 3 * top_n // 4, top_n - 1])
    fig.colorbar(im, ax=ax, shrink=0.6, label="Selected")

    # (c) Band frequency bar chart (percentage in top 10%)
    ax = axes[2]
    band_labels = [f"B{i}\n({sampled_bands[i]})" for i in range(9)]
    colors_c = plt.cm.YlOrRd(band_freq_pct / max(band_freq_pct.max(), 1) * 0.8 + 0.1)
    bars_c = ax.bar(range(9), band_freq_pct, color=colors_c, edgecolor="gray")
    ax.set_xticks(range(9))
    ax.set_xticklabels(band_labels, fontsize=8)
    ax.set_ylabel("Frequency in Top 10% (%)")
    ax.set_title(f"(c) Band Selection Frequency\n[{range_tag}] k={k}")
    ax.set_ylim(0, 105)
    for bar, v in zip(bars_c, band_freq_pct):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", fontsize=9)

    fig.suptitle(f"Band Range Search: [{range_tag}], k={k}, "
                 f"9 uniform bands from HSI",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, f"search_{range_tag}_k{k}_heatmap.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved to {plot_path}")


def plot_pipeline_summary(all_search_results, output_dir):
    """流水线汇总图：各 range 的最优 IoU 对比 + 全局波段频率。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if len(all_search_results) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Best IoU per range
    ax = axes[0]
    tags = [r["range"] for r in all_search_results]
    best_ious = [r["best_iou"] for r in all_search_results]
    ks = [r["k"] for r in all_search_results]
    labels = [f"[{t}]\nk={kk}" for t, kk in zip(tags, ks)]
    colors = plt.cm.Set2(np.linspace(0, 1, len(tags)))
    bars = ax.bar(range(len(tags)), best_ious, color=colors, edgecolor="gray")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best Class-1 IoU")
    ax.set_title("(a) Best IoU per Search Range")
    for bar, v in zip(bars, best_ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", fontsize=9)

    # (b) Global band frequency across all searches
    ax = axes[1]
    all_hsi_set = set()
    for r in all_search_results:
        all_hsi_set.update(r["sampled_9bands_hsi"])
    all_hsi_sorted = sorted(all_hsi_set)

    global_freq = {b: 0 for b in all_hsi_sorted}
    total_top_combos = 0
    for sr in all_search_results:
        n_total = len(sr["all_results"])
        top_n = max(1, n_total // 10)
        top_results = sr["all_results"][:top_n]
        total_top_combos += top_n
        sampled = sr["sampled_9bands_hsi"]
        for r in top_results:
            for idx in r["local_idx"]:
                hsi_band = sampled[idx]
                global_freq[hsi_band] += 1

    bands = list(global_freq.keys())
    freqs = [global_freq[b] / total_top_combos * 100 for b in bands]
    max_f = max(freqs) if freqs else 1
    colors_b = plt.cm.YlOrRd(np.array(freqs) / max_f * 0.8 + 0.1)
    bars_b = ax.bar(range(len(bands)), freqs, color=colors_b, edgecolor="gray")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([str(b) for b in bands], fontsize=7, rotation=45)
    ax.set_xlabel("HSI Band Index")
    ax.set_ylabel("Freq in Top 10% (%)")
    ax.set_title("(b) Global Band Frequency (all ranges)")
    for bar, v in zip(bars_b, freqs):
        if v > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", fontsize=7)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "pipeline_summary.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPipeline summary saved to {plot_path}")


def parse_searches(searches_str):
    """Parse search config string like '60-110:4,70-100:4,75-95:3'."""
    configs = []
    for item in searches_str.split(","):
        item = item.strip()
        range_part, k_part = item.split(":")
        start_str, end_str = range_part.split("-")
        configs.append({
            "range_start": int(start_str),
            "range_end": int(end_str),
            "k": int(k_part),
        })
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="HSI band range exhaustive search with frequency heatmap")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Base config YAML")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="HSI data directory (overrides config)")
    parser.add_argument("--range", type=str, default=None,
                        help="Single range, e.g. '60-110'")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of bands to select (for single --range)")
    parser.add_argument("--searches", type=str, default=None,
                        help="Pipeline: 'start-end:k,...' e.g. '60-110:4,70-100:4,75-95:3'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per combination (quick search)")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/band_range_search")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    # Build search configs list
    if args.searches:
        search_configs = parse_searches(args.searches)
    elif args.range:
        start_str, end_str = args.range.split("-")
        search_configs = [{
            "range_start": int(start_str),
            "range_end": int(end_str),
            "k": args.k,
        }]
    else:
        parser.error("Must specify --range or --searches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" HSI Band Range Exhaustive Search")
    print(f" Config: {args.config}")
    print(f" Data:   {cfg['data']['data_dir']}")
    print(f" Seed:   {args.seed}, Epochs/combo: {args.epochs}")
    print(f" Device: {device}")
    print(f" Searches: {len(search_configs)} configs")
    for i, sc in enumerate(search_configs):
        bands = uniform_sample_bands(sc["range_start"], sc["range_end"])
        n_combos = len(list(itertools.combinations(range(9), sc["k"])))
        print(f"   [{i+1}] range=[{sc['range_start']}, {sc['range_end']}] "
              f"k={sc['k']} → C(9,{sc['k']})={n_combos} "
              f"bands={bands}")
    print("=" * 60)

    all_search_results = []

    for i, sc in enumerate(search_configs):
        print(f"\n{'#'*60}")
        print(f"# Search {i+1}/{len(search_configs)}: "
              f"range=[{sc['range_start']}, {sc['range_end']}] k={sc['k']}")
        print(f"{'#'*60}")

        result = run_single_search(
            cfg=cfg,
            seed=args.seed,
            num_epochs=args.epochs,
            device=device,
            range_start=sc["range_start"],
            range_end=sc["range_end"],
            k=sc["k"],
            output_dir=args.output_dir,
        )
        all_search_results.append(result)

        plot_search_results(result, args.output_dir)

    # Pipeline summary
    if len(all_search_results) > 1:
        summary = {
            "pipeline": [
                {
                    "range": r["range"],
                    "k": r["k"],
                    "sampled_9bands_hsi": r["sampled_9bands_hsi"],
                    "best_hsi_idx": r["best_hsi_idx"],
                    "best_iou": r["best_iou"],
                    "n_combinations": r["n_combinations"],
                    "total_time_sec": r["total_time_sec"],
                }
                for r in all_search_results
            ],
            "seed": args.seed,
            "epochs_per_combo": args.epochs,
        }
        summary_path = os.path.join(args.output_dir, "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nPipeline summary saved to {summary_path}")

        plot_pipeline_summary(all_search_results, args.output_dir)

    # Final summary
    print(f"\n{'='*60}")
    print(" All searches completed!")
    print(f"{'='*60}")
    for r in all_search_results:
        print(f"  [{r['range']}] k={r['k']}: "
              f"best={r['best_hsi_idx']} IoU={r['best_iou']:.4f} "
              f"({r['total_time_sec']:.0f}s)")
    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
