---
title: >-
  [论文解读] Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration
description: >-
  [CVPR 2025][3D视觉][开放词汇3D理解] 提出 Dr. Splat，绕过渲染过程直接将语言对齐的 CLIP 嵌入注册到 3D 高斯上，结合在大规模图像数据上预训练的乘积量化（PQ）实现 6.25% 的嵌入压缩，在完全不需要逐场景优化的前提下（~10 分钟 vs 现有方法 1-24 小时），在开放词汇 3D 语义分割、3D 物体定位和 3D 物体选择任务上显著超越现有方法。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "开放词汇3D理解"
  - "3D高斯"
  - "CLIP嵌入"
  - "乘积量化"
  - "特征注册"
  - "语义分割"
  - "3D定位"
---

# Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration

**会议**: CVPR 2025  
**arXiv**: [2502.16652](https://arxiv.org/abs/2502.16652)  
**代码**: [https://drsplat.github.io/](https://drsplat.github.io/)  
**领域**: 3D视觉 / 场景理解  
**关键词**: 开放词汇3D理解, 3D高斯, CLIP嵌入, 乘积量化, 特征注册, 语义分割, 3D定位

## 一句话总结

提出 Dr. Splat，绕过渲染过程直接将语言对齐的 CLIP 嵌入注册到 3D 高斯上，结合在大规模图像数据上预训练的乘积量化（PQ）实现 6.25% 的嵌入压缩，在完全不需要逐场景优化的前提下（~10 分钟 vs 现有方法 1-24 小时），在开放词汇 3D 语义分割、3D 物体定位和 3D 物体选择任务上显著超越现有方法。

## 研究背景与动机

**领域现状**：开放词汇 3D 场景理解是将自然语言与 3D 空间关联的核心挑战。现有基于 3DGS 的语言嵌入方法（LangSplat、LEGaussians 等）遵循"渲染蒸馏"范式——在 3D 高斯上附加语言特征，通过体积渲染生成 2D 特征图并与 CLIP 嵌入对齐来优化。

**现有痛点**：
- **渲染导致嵌入失真**：渲染过程对多个高斯的嵌入做加权混合（alpha blending），产生的混合特征不再是原始 CLIP 空间中的有效向量，导致 3D 高斯上的嵌入与 CLIP 文本特征之间存在系统性偏差。
- **推理效率低**：要在 3D 空间中定位物体，必须预渲染大量视角的 2D 特征图再做检索，复杂度高且需要额外的 2D→3D 回溯机制。
- **逐场景优化开销大**：现有方法需要对每个新场景训练 1-24 小时来优化嵌入或训练场景级编解码器/codebook。

**核心矛盾**：体积渲染方程对语言嵌入的加权求和本质上破坏了嵌入的语义结构——这是渲染蒸馏范式的固有缺陷，无法通过优化改善。

**本文目标** 如何在不经过渲染的情况下将高保真 CLIP 嵌入直接关联到 3D 高斯上，同时实现高效存储和检索？

## 方法详解

### 整体框架

Dr. Splat 采用"注册而非蒸馏"的新范式，分三步：(1) 预处理阶段：训练标准 3DGS 并构建通用 PQ codebook（一次性）；(2) 特征注册阶段：对每张训练图像提取 SAM mask 和 CLIP 嵌入，将每个像素的嵌入直接分配给对该像素贡献最大的 Top-k 个高斯，聚合多视角嵌入并用 PQ 压缩存储；(3) 推理阶段：直接在 3D 高斯上计算文本查询的余弦相似度 + LeRF 相关性评分。整个过程约 10 分钟，无需梯度优化。

### 关键设计

1. **直接特征注册（无渲染蒸馏）**:
    - 功能：保留 CLIP 嵌入的原始语义结构，避免渲染混合引起的失真
    - 核心思路：遍历训练图像，对每个像素 $\mathbf{r}$，找到沿光线贡献权重最高的 Top-k 个高斯（即 $w_i(\mathbf{I}, \mathbf{r}) = T_i \cdot \tilde{\alpha_i}$），将该像素对应的 CLIP 嵌入 $\mathbf{f}_j^{map}$ 以权重 $w_{ij}$ 累积到这些高斯上。最终每个高斯 $i$ 的聚合特征为：$\dot{\mathbf{f}}_i = \mathbf{f}_i / \|\mathbf{f}_i\|_2$，其中 $\mathbf{f}_i = \sum_j \frac{w_{ij}}{\sum_k w_{ik}} \mathbf{f}_j^{map}$。
    - 设计动机：可视为体积渲染的逆过程（inverse volume rendering），但无需梯度优化。直接聚合 + 归一化保持了嵌入在 CLIP 空间中的语义一致性。

2. **通用乘积量化（Product Quantization）**:
    - 功能：高效压缩存储高维 CLIP 嵌入，且无需逐场景训练
    - 核心思路：将 512 维 CLIP 向量分为 $L$ 个子向量，每个子向量在预训练 codebook 中找到最近质心并用 8-bit 索引替代。PQ 在 LVIS 数据集（1.2M 实例）的 CLIP 嵌入上一次训练，适用于任意场景。每个高斯仅存储 PQ 索引而非完整嵌入，压缩比达 6.25%（512×32bit → L×8bit）。检索时利用预计算的查询-质心距离查找表，单样本检索复杂度从 $O(D)$ 降至 $O(1)$。
    - 设计动机：OpenGaussian 也用 codebook 但需逐场景构建；PQ 的通用性是其核心优势——一次训练、处处可用。

3. **体积感知的 3D 评估协议**:
    - 功能：解决 3D 高斯定位评估中传统点云 IoU 不适用的问题
    - 核心思路：定义每个高斯的重要性分数 $d_i = s_{ix} \cdot s_{iy} \cdot s_{iz} \cdot \alpha_i$（椭球体积 × 不透明度），用加权 IoU 替代传统的等权 IoU。这避免了将大体积高斯与小体积高斯同等对待的不合理性。
    - 设计动机：3D 高斯的体积差异巨大（可达数个数量级），传统点云 IoU 严重低估/高估特定高斯的贡献。

### 损失函数 / 训练策略

Dr. Splat 的特征注册过程是**无优化的（optimization-free）**——不涉及梯度计算和反向传播。3DGS 本身用标准渲染损失训练；PQ codebook 用 k-means 在 LVIS CLIP 嵌入上预训练。

## 实验关键数据

### 3D 物体选择（LeRF-OVS 数据集）

| 方法 | mIoU↑ | mAcc@0.25↑ | 逐场景优化 | 时间 |
|------|-------|------------|-----------|------|
| LangSplat-m | 9.83 | 15.94 | 需要 (~4h) | 慢 |
| OpenGaussian | 43.06 | 59.61 | 需要 (~1h) | 中 |
| **Dr. Splat (Top-40)** | **43.58** | 63.87 | **不需要** | **~10min** |

Dr. Splat 在不需逐场景优化的前提下超越 OpenGaussian。

### 3D 物体定位（ScanNet）

| 方法 | mIoU↑ | IoU>0.15 | IoU>0.3 | IoU>0.45 |
|------|-------|----------|---------|----------|
| LangSplat-m | 8.0 | 17.1 | 7.8 | 2.9 |
| LEGaussians-m | 9.5 | 19.1 | 8.9 | 7.3 |
| OpenGaussian | 25.2 | 59.5 | 38.0 | 18.3 |
| **Dr. Splat (Top-40)** | **25.4** | **60.7** | **40.3** | **25.6** |

在高阈值 IoU>0.45 上 Dr. Splat 超越 OpenGaussian 7.3 个百分点（25.6 vs 18.3），体现更精确的定位。

### 关键发现

- 渲染蒸馏方法（LangSplat-m）在 3D 任务上几乎完全失败（mIoU 仅 8-10），验证了渲染混合对嵌入的致命损害（Fig. 2 可视化清晰展示了差异）。
- PQ 在 64 个子向量（1/32 压缩）下仍保持接近无损的性能，128 子向量时几乎无损。
- Top-k 增大时 mIoU 持续提升但内存也增大，Top-40 在精度和效率间取得较好平衡。
- Dr. Splat 在语义分割上也达到同级别性能（19 类 mIoU 29.6/mAcc 47.7 vs OpenGaussian 30.1/46.5），尽管并非专门设计用于分割。

## 亮点与洞察

- **"注册而非蒸馏"的范式转换**：从根本上解决了渲染混合导致的嵌入失真问题。通过逆渲染的视角理解特征分配，将加权聚合操作从可微渲染管道中解放出来。
- **通用 PQ 消除逐场景训练**：一次在 LVIS 上训练 codebook，即可压缩任意场景的 CLIP 嵌入。这是真正实现"开箱即用"3D 场景理解的关键。
- **体积感知 IoU 评估指标**：指出了 3D 高斯定位评估的基础性问题——不同高斯的体积可能差异数个数量级，传统 IoU 完全不适用。提出的加权 IoU 有望成为后续工作的标准评估协议。
- 整个方法约 10 分钟完成，比需要逐场景优化的方法快 6-150 倍。

## 局限与展望

- Top-k 机制本质上是启发式选择，对于高度遮挡的小物体可能分配不到足够的嵌入。
- PQ 压缩不可避免引入量化误差，对语义边界模糊的类别（如"fabric" vs "cloth"）可能影响细粒度区分。
- 特征注册质量依赖 SAM 分割质量——SAM 过分割或欠分割会直接传播到 3D 嵌入。
- 在极大规模场景中（数百万高斯），即使 PQ 压缩，存储和检索仍有挑战。
- 当前仅用余弦相似度 + LeRF 相关性评分，未探索更复杂的检索机制。

## 相关工作与启发

- **vs LangSplat**: LangSplat 通过渲染蒸馏分配嵌入，每场景需 4 小时训练且嵌入失真；Dr. Splat 直接注册，10 分钟完成且嵌入保真。
- **vs OpenGaussian**: 最直接对比——OpenGaussian 也在 3D 空间直接操作，但需逐场景训练 codebook；Dr. Splat 用通用 PQ 免除逐场景训练且性能相当或更优。
- **vs LEGaussians**: LEGaussians 也用 codebook 但仍是渲染蒸馏范式，3D 性能弱。
- **启发**：当"优化到 2D 特征图再回推到 3D"这条路走不通时，直接建立 2D→3D 的显式映射（即使是简单的加权聚合）可能是更好的选择。这一思路可推广到其他需要将 2D 基础模型特征迁移到 3D 的任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ 提出直接注册范式替代渲染蒸馏，PQ消除逐场景训练
- 实验充分度: ⭐⭐⭐⭐ 三个任务×两个数据集，消融覆盖PQ和Top-k关键参数
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，数学符号规范，图示直观展示嵌入失真
- 价值: ⭐⭐⭐⭐ 显著降低了开放词汇3D理解的使用门槛（无需逐场景训练，10分钟完成）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)
- [\[CVPR 2026\] ST4R-Splat: Spatio-Temporal Referring Segmentation in 4D Gaussian Splatting](../../CVPR2026/3d_vision/st4r-splat_spatio-temporal_referring_segmentation_in_4d_gaussian_splatting.md)
- [\[ICML 2025\] ReferSplat: Referring Segmentation in 3D Gaussian Splatting](../../ICML2025/3d_vision/refersplat_referring_segmentation_in_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization](morpheus_text-driven_3d_gaussian_splat_shape_and_color_stylization.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](../../NeurIPS2025/3d_vision/segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)

</div>

<!-- RELATED:END -->
