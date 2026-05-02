"""波段选择方法对比：SPA / CARS / MI vs. Exhaustive (CNN 下游评估).

三种经典波段选择算法（SPA、CARS、MI）在 pixel-level 特征上选出 k 个波段，
再用 baseline CNN（MobileNetV2-UNet, 70 epoch）在 val set 上评估 class-1 IoU。
与穷举搜索（Exhaustive）的已知最优结果做 paired comparison。

CNN 训练不在本脚本内实现，而是通过 subprocess 调用项目现有的 train.py，
保证评估条件（网络结构、优化器、数据划分）与穷举搜索完全一致。

候选 9 波段（HSI 原始 index）: [64, 68, 73, 77, 82, 87, 90, 94, 98]
数据集 local idx: 0..8

用法:
    python scripts/band_selection_comparison.py \\
        --data_dir /root/autodl-tmp/datasets/185_9bands \\
        --output_dir outputs/band_selection_comparison \\
        --seeds 42 \\
        --ks 2,3,4

    python scripts/band_selection_comparison.py \\
        --data_dir /root/autodl-tmp/datasets/185_9bands \\
        --seeds 42,123,456 \\
        --ks 2,3,4
"""

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# 项目根目录（scripts/ 的上一级）
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# 候选波段定义
# ---------------------------------------------------------------------------

# HSI 原始 100 波段里的实际 index
HSI_BAND_INDICES = [64, 68, 73, 77, 82, 87, 90, 94, 98]

# 9-band 数据集中的 local idx: 0..8，一一对应上面的 HSI index
N_TOTAL_BANDS = len(HSI_BAND_INDICES)

# ---------------------------------------------------------------------------
# 穷举搜索已知结果（baseline CNN 70 ep, 同网络/同 split）
# 直接硬编码，不重新训练——这些值来自之前完成的 band_search.py 实验
# ---------------------------------------------------------------------------

EXHAUSTIVE_RESULTS = {
    1: {"local_idx": [3],          "iou": 0.7291},
    2: {"local_idx": [1, 3],       "iou": 0.7513},
    3: {"local_idx": [0, 4, 6],    "iou": 0.7170},
    4: {"local_idx": [1, 3, 5, 6], "iou": 0.7204},
}


# ============================================================================
# SPA (Successive Projections Algorithm)
# ============================================================================
# 经典前向变量选择：每步选择与已选波段正交投影残差最大的波段。
# 原始论文: Araújo et al., Chemometrics and Intelligent Laboratory Systems, 2001.

def spa_select(X, k, start_band=None):
    """Successive Projections Algorithm.

    Args:
        X: (N_pixels, n_bands) 光谱矩阵.
        k: 要选择的波段数.
        start_band: 起始波段 index（None 则遍历所有起始点取最优）.

    Returns:
        selected: list of k band indices (sorted).
    """
    n_samples, n_bands = X.shape

    if start_band is not None:
        return _spa_single(X, k, start_band)

    # 遍历所有起始点，按投影残差的方差选最优
    best_selected = None
    best_score = -np.inf
    for s in range(n_bands):
        selected = _spa_single(X, k, s)
        # 评价标准：选出波段子空间能解释的方差
        X_sub = X[:, selected]
        score = np.var(X_sub @ np.linalg.pinv(X_sub) @ X.T)
        if score > best_score:
            best_score = score
            best_selected = selected
    return sorted(best_selected)


def _spa_single(X, k, start):
    """SPA with fixed starting band."""
    n_samples, n_bands = X.shape
    selected = [start]
    # 工作副本：逐步投影到已选波段的正交补空间
    residual = X.copy()

    for _ in range(k - 1):
        # 投影到已选最后一个波段的方向，取正交残差
        proj_vec = residual[:, selected[-1]]
        proj_vec = proj_vec / (np.linalg.norm(proj_vec) + 1e-10)

        for j in range(n_bands):
            coeff = residual[:, j].dot(proj_vec)
            residual[:, j] = residual[:, j] - coeff * proj_vec

        # 选残差范数最大的波段（排除已选）
        norms = np.array([np.linalg.norm(residual[:, j])
                          if j not in selected else -1
                          for j in range(n_bands)])
        next_band = int(np.argmax(norms))
        selected.append(next_band)

    return sorted(selected)


# ============================================================================
# CARS (Competitive Adaptive Reweighted Sampling)
# ============================================================================
# 基于 PLS 回归系数的自适应波段筛选。每轮迭代中按指数衰减采样概率，
# 并用交叉验证 RMSECV 选择最优子集。
# 原始论文: Li et al., Analytica Chimica Acta, 2009.

def cars_select(X, y, k, n_iterations=50, seed=42):
    """CARS band selection (simplified).

    Iteratively shrinks the candidate set by exponential decay of PLS
    regression coefficients.  Returns the k bands with highest cumulative
    selection frequency across all iterations.

    Args:
        X: (N_pixels, n_bands) 光谱矩阵.
        y: (N_pixels,) 标签 (0/1).
        k: 要选择的波段数.
        n_iterations: 迭代次数.
        seed: 随机种子.

    Returns:
        selected: list of k band indices (sorted).
    """
    rng = np.random.RandomState(seed)
    n_samples, n_bands = X.shape

    # 累计每个波段被"留下"的次数
    selection_count = np.zeros(n_bands, dtype=np.float64)
    active_mask = np.ones(n_bands, dtype=bool)

    for it in range(n_iterations):
        active_idx = np.where(active_mask)[0]
        if len(active_idx) <= k:
            selection_count[active_idx] += 1
            break

        X_active = X[:, active_idx]

        # 用最小二乘近似 PLS 系数（避免依赖外部 PLS 库）
        # beta = (X^T X + λI)^{-1} X^T y
        lam = 1e-4
        XtX = X_active.T @ X_active + lam * np.eye(len(active_idx))
        Xty = X_active.T @ y
        beta = np.linalg.solve(XtX, Xty)

        # 指数衰减采样：系数大的波段被保留的概率更高
        importance = np.abs(beta)
        importance = importance / (importance.sum() + 1e-10)

        # 按指数衰减保留比例：从 n_bands 衰减到 k
        ratio = k / len(active_idx)
        keep_ratio = max(ratio, (1.0 - it / n_iterations) ** 2)
        n_keep = max(k, int(np.ceil(len(active_idx) * keep_ratio)))

        # 概率采样
        probs = importance / (importance.sum() + 1e-10)
        chosen = rng.choice(
            len(active_idx), size=min(n_keep, len(active_idx)),
            replace=False, p=probs,
        )

        new_mask = np.zeros(n_bands, dtype=bool)
        new_mask[active_idx[chosen]] = True
        active_mask = new_mask
        selection_count[active_idx[chosen]] += 1

    # 返回累计频次最高的 k 个波段
    top_k = np.argsort(selection_count)[::-1][:k]
    return sorted(top_k.tolist())


# ============================================================================
# MI (Mutual Information)
# ============================================================================
# 计算每个波段与标签之间的互信息，贪心地选 MI 最高的 k 个波段。
# 使用直方图法估计互信息。

def mi_select(X, y, k, n_bins=32):
    """Mutual Information based band selection.

    Greedy selection: each round picks the band with highest MI to label,
    conditional on already-selected bands (approximated by subtracting
    redundancy via min MI between candidate and selected bands).

    Args:
        X: (N_pixels, n_bands) 光谱矩阵.
        y: (N_pixels,) 标签 (0/1).
        k: 要选择的波段数.
        n_bins: 离散化 bin 数.

    Returns:
        selected: list of k band indices (sorted).
    """
    n_bands = X.shape[1]

    # 预计算每个波段与标签的 MI
    mi_with_label = np.array([
        _mutual_info_hist(X[:, b], y, n_bins) for b in range(n_bands)
    ])

    # 预计算波段间的 MI 矩阵（用于去冗余）
    mi_matrix = np.zeros((n_bands, n_bands))
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            m = _mutual_info_hist(X[:, i], X[:, j], n_bins)
            mi_matrix[i, j] = m
            mi_matrix[j, i] = m

    # 贪心选择：mRMR (max Relevance - min Redundancy)
    selected = []
    remaining = list(range(n_bands))

    for _ in range(k):
        if not selected:
            # 第一步：选 MI 最高的
            best = remaining[int(np.argmax(mi_with_label[remaining]))]
        else:
            # 后续步骤：relevance - mean redundancy
            scores = []
            for c in remaining:
                relevance = mi_with_label[c]
                redundancy = np.mean([mi_matrix[c, s] for s in selected])
                scores.append(relevance - redundancy)
            best = remaining[int(np.argmax(scores))]

        selected.append(best)
        remaining.remove(best)

    return sorted(selected)


def _mutual_info_hist(x, y, n_bins=32):
    """Estimate mutual information via histogram."""
    # Discretize continuous x
    x_d = np.digitize(x, bins=np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1))
    y_d = y.astype(int) if y.dtype != int else y

    # Joint and marginal counts
    joint = {}
    for xi, yi in zip(x_d, y_d):
        joint[(xi, yi)] = joint.get((xi, yi), 0) + 1

    n = len(x)
    px = np.bincount(x_d, minlength=n_bins + 2).astype(float) / n
    py = np.bincount(y_d, minlength=max(y_d) + 1).astype(float) / n

    mi = 0.0
    for (xi, yi), count in joint.items():
        pxy = count / n
        if pxy > 0 and px[xi] > 0 and py[yi] > 0:
            mi += pxy * np.log(pxy / (px[xi] * py[yi]) + 1e-10)
    return max(mi, 0.0)


# ============================================================================
# Pixel-level 特征池提取
# ============================================================================

def load_pixel_pool(data_dir, image_dir="images", mask_dir="masks",
                    max_pixels=100000, seed=42):
    """从数据集中采样像素，构建 (X, y) pixel pool 供 SPA/CARS/MI 使用.

    Args:
        data_dir: 数据集根目录.
        max_pixels: 最多采样像素数（避免内存过大）.
        seed: 随机种子.

    Returns:
        X: (N, 9) float32 光谱矩阵.
        y: (N,) int 标签 (0=bg, 1=defect).
    """
    rng = np.random.RandomState(seed)
    image_root = os.path.join(data_dir, image_dir)
    mask_root = os.path.join(data_dir, mask_dir)

    stems = sorted([f[:-4] for f in os.listdir(image_root) if f.endswith(".npy")])

    all_spectra = []
    all_labels = []

    for stem in stems:
        img = np.load(os.path.join(image_root, stem + ".npy")).astype(np.float32)
        # img: (H, W, 9)

        mask_npy = os.path.join(mask_root, stem + ".npy")
        mask_png = os.path.join(mask_root, stem + ".png")
        if os.path.exists(mask_npy):
            mask = np.load(mask_npy).astype(np.int64)
        elif os.path.exists(mask_png):
            from PIL import Image
            mask = np.array(Image.open(mask_png)).astype(np.int64)
        else:
            continue

        H, W, C = img.shape
        pixels = img.reshape(-1, C)       # (H*W, 9)
        labels = mask.reshape(-1)         # (H*W,)
        all_spectra.append(pixels)
        all_labels.append(labels)

    X = np.concatenate(all_spectra, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # 下采样到 max_pixels
    if len(X) > max_pixels:
        idx = rng.choice(len(X), max_pixels, replace=False)
        X = X[idx]
        y = y[idx]

    print(f"Pixel pool: {X.shape[0]} pixels, {X.shape[1]} bands, "
          f"class-1 ratio={y.mean():.4f}")
    return X, y


# ============================================================================
# CNN 下游评估（通过 subprocess 调用 train.py）
# ============================================================================

def run_cnn_eval(band_indices, method, k, seed, output_root, base_config_path,
                 num_epochs=70):
    """用 baseline CNN 训练+评估指定波段子集，返回 best class-1 IoU.

    流程：
        1. 复制 baseline.yaml，修改 band_indices / num_channels / num_epochs / 关闭早停
        2. subprocess 调用 python train.py --config <yaml> --seed <seed>
        3. 从 training_log.json 读 best IoU_class1

    不在本脚本里实现 CNN 训练循环，完全复用项目现有的 train.py。

    Args:
        band_indices: list of int, 选出的 local band indices.
        method: str, 方法名 (用于 experiment_name).
        k: int, 波段数.
        seed: int, 随机种子.
        output_root: str, 输出根目录.
        base_config_path: str, baseline.yaml 路径.
        num_epochs: int, 训练 epoch 数.

    Returns:
        best_iou: float, best class-1 IoU from training log.
    """
    experiment_name = f"{method}_k{k}_seed{seed}"
    exp_output_dir = os.path.join(output_root, experiment_name)

    # 1. 生成修改后的 config YAML
    config_dir = os.path.join(output_root, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{experiment_name}.yaml")

    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["experiment_name"] = experiment_name
    cfg["data"]["band_indices"] = sorted(band_indices)
    cfg["data"]["num_channels"] = len(band_indices)
    cfg["train"]["num_epochs"] = num_epochs
    cfg["train"]["early_stopping_patience"] = 0   # 关闭早停，跑满所有 epoch

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # 2. subprocess 调用 train.py
    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "train.py"),
        "--config", config_path,
        "--seed", str(seed),
        "--output_dir", exp_output_dir,
    ]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] train.py failed for {experiment_name}")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    # 3. 从 training_log.json 读出 best IoU_class1
    log_path = os.path.join(exp_output_dir, "training_log.json")
    if not os.path.exists(log_path):
        print(f"  [ERROR] training_log.json not found at {log_path}")
        return None

    with open(log_path, "r") as f:
        logs = json.load(f)

    best_iou = max(entry["IoU_class1"] for entry in logs)
    print(f"  {experiment_name}: bands={sorted(band_indices)} "
          f"best_IoU={best_iou:.4f} ({elapsed:.1f}s)")
    return best_iou


# ============================================================================
# 可视化：波段重合热图
# ============================================================================

def plot_band_overlap(results_by_method, k, hsi_indices, output_path):
    """绘制指定 k 下各方法选出波段的重合热图.

    行=方法名, 列=9个候选波段, 选中=1 / 未选=0。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping overlap plot.")
        return

    methods = list(results_by_method.keys())
    n_bands = len(hsi_indices)

    matrix = np.zeros((len(methods), n_bands), dtype=int)
    for i, method in enumerate(methods):
        info = results_by_method[method]
        for idx in info["local_idx"]:
            if idx < n_bands:
                matrix[i, idx] = 1

    fig, ax = plt.subplots(figsize=(10, max(3, len(methods) * 0.6 + 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_bands))
    ax.set_xticklabels([f"B{i}\n({hsi_indices[i]})" for i in range(n_bands)],
                        fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel("Band (local idx / HSI idx)")
    ax.set_title(f"Selected Bands Overlap (k={k})")

    # 在每个格子里标注 0/1
    for i in range(len(methods)):
        for j in range(n_bands):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] else "gray", fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Selected")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Band overlap heatmap saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="波段选择方法对比 (SPA/CARS/MI vs. Exhaustive, CNN 评估)")
    parser.add_argument("--data_dir", type=str,
                        default="/root/autodl-tmp/datasets/185_9bands",
                        help="9-band 数据集根目录")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/band_selection_comparison",
                        help="输出目录")
    parser.add_argument("--base_config", type=str,
                        default="configs/baseline.yaml",
                        help="Baseline config YAML 路径")
    parser.add_argument("--ks", type=str, default="2,3,4",
                        help="要测试的波段数 k，逗号分隔 (e.g. 2,3,4)")
    parser.add_argument("--seeds", type=str, default="42",
                        help="随机种子列表，逗号分隔 (e.g. 42,123,456)")
    parser.add_argument("--epochs", type=int, default=70,
                        help="CNN 训练 epoch 数")
    parser.add_argument("--max_pixels", type=int, default=100000,
                        help="pixel pool 最大像素数")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    base_config_path = os.path.join(PROJECT_ROOT, args.base_config)

    print("=" * 70)
    print(" Band Selection Comparison (CNN downstream evaluation)")
    print(f" Data:    {args.data_dir}")
    print(f" k:       {ks}")
    print(f" Seeds:   {seeds}")
    print(f" Epochs:  {args.epochs}")
    print(f" Methods: SPA, CARS, MI, Exhaustive(hardcoded)")
    print("=" * 70)

    # ---- Step 1: Load pixel pool for band selection algorithms ----
    print("\n[Step 1] Loading pixel pool for SPA / CARS / MI...")
    X, y = load_pixel_pool(args.data_dir, max_pixels=args.max_pixels, seed=42)

    # ---- Step 2: Run band selection (pixel-level) for each (method, k) ----
    # 波段选择只依赖 pixel pool，与 seed 无关（除 CARS 内部有随机性）
    print("\n[Step 2] Band selection (pixel-level)...")
    selection_results = {}   # {(method, k): list_of_local_idx}

    for k in ks:
        print(f"\n--- k = {k} ---")

        # SPA
        spa_bands = spa_select(X, k)
        selection_results[("SPA", k)] = spa_bands
        print(f"  SPA:  {spa_bands}")

        # CARS
        cars_bands = cars_select(X, y, k, n_iterations=50, seed=42)
        selection_results[("CARS", k)] = cars_bands
        print(f"  CARS: {cars_bands}")

        # MI (mRMR)
        mi_bands = mi_select(X, y, k)
        selection_results[("MI", k)] = mi_bands
        print(f"  MI:   {mi_bands}")

    # ---- Step 3: CNN downstream evaluation for each (method, k, seed) ----
    print("\n[Step 3] CNN downstream evaluation (train.py subprocess)...")
    all_results = []   # list of dicts

    for k in ks:
        for method in ["SPA", "CARS", "MI"]:
            band_idx = selection_results[(method, k)]

            for seed in seeds:
                iou = run_cnn_eval(
                    band_indices=band_idx,
                    method=method,
                    k=k,
                    seed=seed,
                    output_root=output_dir,
                    base_config_path=base_config_path,
                    num_epochs=args.epochs,
                )
                all_results.append({
                    "method": method,
                    "k": k,
                    "seed": seed,
                    "local_idx": band_idx,
                    "hsi_idx": [HSI_BAND_INDICES[i] for i in band_idx],
                    "iou": iou,
                })

    # ---- Step 4: 加入 Exhaustive 结果（硬编码，不重新训练） ----
    for k in ks:
        if k in EXHAUSTIVE_RESULTS:
            ex = EXHAUSTIVE_RESULTS[k]
            # Exhaustive 只有一个 "seed" 的结果，对所有 seed 都用同一值
            all_results.append({
                "method": "Exhaustive",
                "k": k,
                "seed": "all",
                "local_idx": ex["local_idx"],
                "hsi_idx": [HSI_BAND_INDICES[i] for i in ex["local_idx"]],
                "iou": ex["iou"],
            })

    # ---- Step 5: 聚合 (method, k) 跨 seed 的 mean ± std ----
    print("\n[Step 4] Aggregating results across seeds...")
    aggregated = {}   # {(method, k): {"mean": ..., "std": ..., "ious": [...], ...}}

    for method in ["SPA", "CARS", "MI", "Exhaustive"]:
        for k in ks:
            entries = [r for r in all_results
                       if r["method"] == method and r["k"] == k and r["iou"] is not None]
            if not entries:
                continue
            ious = [e["iou"] for e in entries]
            aggregated[(method, k)] = {
                "method": method,
                "k": k,
                "local_idx": entries[0]["local_idx"],
                "hsi_idx": entries[0]["hsi_idx"],
                "ious": ious,
                "mean": float(np.mean(ious)),
                "std": float(np.std(ious)) if len(ious) > 1 else 0.0,
                "n_seeds": len(ious),
            }

    # ---- Step 6: 输出 ----
    # 6a. results.json
    json_output = {
        "config": {
            "data_dir": args.data_dir,
            "ks": ks,
            "seeds": seeds,
            "epochs": args.epochs,
            "hsi_band_indices": HSI_BAND_INDICES,
            "exhaustive_results": {str(kk): v for kk, v in EXHAUSTIVE_RESULTS.items()},
        },
        "per_run": all_results,
        "aggregated": {f"{v['method']}_k{v['k']}": v
                       for v in aggregated.values()},
    }
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"\nDetailed results saved to {json_path}")

    # 6b. results.txt (human-readable table)
    txt_path = os.path.join(output_dir, "results.txt")
    with open(txt_path, "w") as f:
        header = (f"{'Method':<12} {'k':>2}  {'Bands (local)':<20} "
                  f"{'Bands (HSI)':<28} {'IoU mean':>10} {'IoU std':>10} {'N':>3}")
        sep = "-" * len(header)
        f.write("Band Selection Comparison — CNN Downstream Evaluation\n")
        f.write(f"Epochs: {args.epochs}, Seeds: {seeds}\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")

        for k in ks:
            for method in ["SPA", "CARS", "MI", "Exhaustive"]:
                key = (method, k)
                if key not in aggregated:
                    continue
                a = aggregated[key]
                local_str = str(a["local_idx"])
                hsi_str = str(a["hsi_idx"])
                line = (f"{method:<12} {k:>2}  {local_str:<20} "
                        f"{hsi_str:<28} {a['mean']:>10.4f} {a['std']:>10.4f} "
                        f"{a['n_seeds']:>3}")
                f.write(line + "\n")
            f.write(sep + "\n")

        # Best per k
        f.write("\nBest method per k:\n")
        for k in ks:
            entries_k = [(key, val) for key, val in aggregated.items() if key[1] == k]
            if entries_k:
                best_key, best_val = max(entries_k, key=lambda x: x[1]["mean"])
                f.write(f"  k={k}: {best_val['method']} "
                        f"bands={best_val['local_idx']} "
                        f"IoU={best_val['mean']:.4f}\n")

    # Also print to stdout
    with open(txt_path, "r") as f:
        print(f.read())
    print(f"Table saved to {txt_path}")

    # 6c. Band overlap heatmap for each k
    for k in ks:
        methods_for_k = {}
        for method in ["SPA", "CARS", "MI", "Exhaustive"]:
            key = (method, k)
            if key in aggregated:
                methods_for_k[method] = aggregated[key]
        if methods_for_k:
            plot_path = os.path.join(output_dir, f"selected_bands_overlap_k{k}.png")
            plot_band_overlap(methods_for_k, k, HSI_BAND_INDICES, plot_path)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
