"""Post-processing utilities for segmentation prediction refinement.

[NEW] 推理时后处理: 零训练改动，在生成预测后做锐化/细化。

方向2实现:
    - unsharp_mask_probs: Unsharp Masking锐化预测概率图
    - guided_filter_probs: 引导滤波锐化（用原始光谱图做引导）
    - refine_predictions: 整合后处理pipeline

原理:
    - Unsharp Masking: 增强概率图的高频细节，使缺陷边界更锐利
    - 引导滤波: 用原始光谱图引导概率图平滑，在保留光谱边缘的同时
      抑制分割噪声。比全局锐化更智能——它能区分"该清晰的地方"和
      "该平滑的地方"，因为引导图（光谱）提供了边缘位置信息。
"""

import numpy as np
import cv2


def unsharp_mask_probs(prob_map, sigma=1.0, strength=1.5):
    """Unsharp Masking锐化预测概率图.

    原理: 用高斯模糊提取低频，原图减去低频得到高频细节，
    再将高频加回原图。结果是边缘更锐利。

    Args:
        prob_map: (H, W) or (C, H, W) 预测概率图, 值域[0,1].
        sigma: 高斯模糊的标准差，控制锐化的空间尺度.
            - 小sigma(0.5~1.0): 只增强细微边缘
            - 大sigma(2.0~3.0): 增强更粗的边缘结构
        strength: 锐化强度.
            - 1.0: 中等锐化
            - 2.0: 较强锐化
            - 太大(>3.0)会产生高频伪影

    Returns:
        锐化后的概率图, 同shape, 值域clip到[0,1].
    """
    if prob_map.ndim == 3:
        # Per-channel sharpening
        return np.stack([
            unsharp_mask_probs(prob_map[c], sigma, strength)
            for c in range(prob_map.shape[0])
        ], axis=0)

    blurred = cv2.GaussianBlur(prob_map.astype(np.float32), (0, 0), sigma)
    sharpened = cv2.addWeighted(
        prob_map.astype(np.float32), 1.0 + strength,
        blurred, -strength, 0
    )
    return np.clip(sharpened, 0, 1)


def guided_filter_probs(prob_map, guide_image, radius=4, eps=0.01,
                        detail_strength=1.5):
    """引导滤波锐化: 用原始光谱图做引导.

    原理: 引导滤波(Guided Filter)是一种边缘保持滤波器。
    它用guide_image的边缘结构来决定在哪里平滑、在哪里保留细节。
    当guide = 光谱图时，光谱边缘（缺陷边界）处保留细节，
    平坦区域（正常苹果表面）处抑制噪声。

    步骤:
        1. 引导滤波平滑: smoothed = GuidedFilter(prob, guide)
        2. 提取高频细节: detail = prob - smoothed
        3. 增强细节: result = prob + strength * detail

    Args:
        prob_map: (H, W) 单通道概率图.
        guide_image: (H, W) 引导图（建议用光谱图的某个波段或PCA第一主成分）.
        radius: 引导滤波窗口半径. 越大平滑越强.
        eps: 正则化参数. 越小越保留细节，越大越接近全局平滑.
        detail_strength: 高频增强强度.

    Returns:
        锐化后的概率图, 值域clip到[0,1].
    """
    prob_f32 = prob_map.astype(np.float32)
    guide_f32 = guide_image.astype(np.float32)

    # Normalize guide to [0,1]
    g_min, g_max = guide_f32.min(), guide_f32.max()
    if g_max - g_min > 1e-6:
        guide_f32 = (guide_f32 - g_min) / (g_max - g_min)

    try:
        # OpenCV ximgproc module (需要opencv-contrib-python)
        smoothed = cv2.ximgproc.guidedFilter(
            guide=guide_f32, src=prob_f32, radius=radius, eps=eps
        )
    except AttributeError:
        # Fallback: 用bilateralFilter近似（如果没有ximgproc）
        # bilateralFilter也是边缘保持滤波器，效果类似但不完全等价
        smoothed = cv2.bilateralFilter(
            prob_f32, d=radius * 2 + 1,
            sigmaColor=eps * 10, sigmaSpace=radius
        )

    detail = prob_f32 - smoothed
    result = prob_f32 + detail_strength * detail
    return np.clip(result, 0, 1)


def refine_predictions(logits_np, images_np=None, method="none",
                       sharpen_sigma=1.0, sharpen_strength=1.5,
                       guide_radius=4, guide_eps=0.01,
                       detail_strength=1.5, guide_band=4):
    """整合后处理pipeline.

    Args:
        logits_np: (B, C, H, W) 模型输出的概率图（softmax之后）.
        images_np: (B, 9, H, W) 原始光谱输入，引导滤波需要.
        method: 后处理方法:
            - "none": 不做后处理
            - "unsharp": Unsharp Masking锐化
            - "guided": 引导滤波锐化（需要images_np）
            - "unsharp+guided": 先引导滤波再Unsharp Masking
        sharpen_sigma: Unsharp Masking的高斯sigma.
        sharpen_strength: Unsharp Masking的锐化强度.
        guide_radius: 引导滤波窗口半径.
        guide_eps: 引导滤波正则化参数.
        detail_strength: 引导滤波细节增强强度.
        guide_band: 用哪个波段做引导图（default: 波段5, 中间波段）.

    Returns:
        refined_preds: (B, H, W) 精化后的预测标签.
    """
    if method == "none":
        return logits_np.argmax(axis=1)

    refined = logits_np.copy()

    for b in range(refined.shape[0]):
        for c in range(refined.shape[1]):
            prob = refined[b, c]

            if "guided" in method and images_np is not None:
                guide = images_np[b, guide_band]
                prob = guided_filter_probs(
                    prob, guide, radius=guide_radius,
                    eps=guide_eps, detail_strength=detail_strength
                )

            if "unsharp" in method:
                prob = unsharp_mask_probs(
                    prob, sigma=sharpen_sigma, strength=sharpen_strength
                )

            refined[b, c] = prob

    return refined.argmax(axis=1)
