---
title: >-
  [论文解读] DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
description: >-
  [CVPR 2025][语义分割][RGBD分割] 提出将深度图作为几何先验而非通过神经网络编码，设计几何自注意力（GSA）将深度距离和空间距离融合为衰减因子调制注意力权重，以约一半 FLOPs 匹配或超越双编码器 RGBD 分割方法。 领域现状：RGBD 语义分割通常用双编码器分别处理 RGB 和 Depth 然后融合…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "RGBD分割"
  - "几何先验"
  - "深度图"
  - "自注意力改进"
---

# DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2504.04701](https://arxiv.org/abs/2504.04701)  
**代码**: [https://github.com/VCIP-RGBD/DFormer](https://github.com/VCIP-RGBD/DFormer)  
**领域**: 分割  
**关键词**: RGBD分割、几何先验、深度图、自注意力改进、语义分割

## 一句话总结
提出将深度图作为几何先验而非通过神经网络编码，设计几何自注意力（GSA）将深度距离和空间距离融合为衰减因子调制注意力权重，以约一半 FLOPs 匹配或超越双编码器 RGBD 分割方法。

## 研究背景与动机

**领域现状**：RGBD 语义分割通常用双编码器分别处理 RGB 和 Depth 然后融合。深度图经过完整的 backbone 编码，参数和计算量几乎翻倍。

**现有痛点**：双编码器方案计算量大（GeminiFusion 256G FLOPs），且深度编码器学到的表征可能与 RGB 编码器不对齐。深度图本质上是几何信息，用神经网络"编码"是一种间接且浪费的利用方式。

**核心矛盾**：深度图提供了明确的3D几何关系（物体间距离、共面性），但经过神经网络编码后这种显式几何信息被隐式化了，模型需要重新"学习"本已显式的信息。

**本文目标** 直接利用深度图的几何信息作为注意力先验，而不是将其编码为特征。

**切入角度**：深度图可以直接告诉我们哪些 patch 在3D空间中接近（可能属于同一物体），哪些远离。将这种几何关系转化为自注意力的衰减因子——3D距离近的 patch 对注意力更强，远的更弱。

**核心 idea**：用深度距离和空间距离的融合先验作为注意力的几何衰减因子，无需深度编码器即可有效利用深度信息。

## 方法详解

### 整体框架
RGB 图像 → 单编码器 ViT + GSA（几何自注意力）→ 轻量解码头。深度图不经过编码器，直接在每层注意力中作为先验使用。

### 关键设计

1. **几何自注意力（GSA）**:

    - 功能：用深度几何信息调制标准自注意力
    - 核心思路：$\text{GeoAttn}(Q,K,V,G) = (\text{Softmax}(QK^T) \odot \beta^G)V$，其中 $G$ 是几何先验矩阵，$\beta \in (0,1)$ 是可学习衰减基。几何先验大 → $\beta^G$ 接近 0（抑制注意力），几何先验小 → $\beta^G$ 接近 1（保持注意力）
    - 设计动机：将深度信息从"特征"转变为"注意力权重调制"，不需要额外编码器参数

2. **几何先验融合**:

    - 功能：结合深度距离和空间距离两种几何信号
    - 核心思路：$G$ 融合深度距离 $D_{ij}$（两个 patch 的深度值差异）和空间曼哈顿距离 $S_{ij}$（patch 在图像中的位置差异），通过可学习的 memory 权重进行融合。Memory 基融合比卷积融合、加法融合、逐元素乘法融合都更好（56.2 vs 55.8/54.6/54.9 mIoU）
    - 设计动机：深度距离反映3D关系，空间距离反映2D邻近性，两者互补

3. **轴分解（Axes Decomposition）**:

    - 功能：降低 GSA 的计算复杂度
    - 核心思路：将 2D 几何先验分解为水平和垂直两个方向分别计算，计算量减半。效果轻微下降（56.0 vs 56.2 mIoU）但计算量显著减少
    - 设计动机：标准全局 GSA 在高分辨率下不可扩展

### 损失函数 / 训练策略
标准交叉熵分割损失。RGB-D 预训练在 ImageNet-1K 上（深度由估计模型生成）。

## 实验关键数据

### 主实验

| 模型 | 参数 | FLOPs | NYU mIoU | SUN mIoU |
|------|------|-------|---------|---------|
| GeminiFusion (MiT-B5) | 137.2M | 256.1G | 57.7 | 53.3 |
| **DFormerv2-B** | **53.9M** | **67.2G** | **57.7** | 52.1 |
| **DFormerv2-L** | **95.5M** | **124.1G** | **58.4** | **53.3** |

### 消融实验

| 配置 | NYU mIoU | 说明 |
|------|---------|------|
| 标准注意力 | 51.7 | 无深度 |
| +深度先验 | 54.3 | +2.6 |
| +深度+空间先验 | 56.2 | +4.5 |
| +轴分解 | 56.0 | 计算减半，仅-0.2 |

### 关键发现
- **半计算量匹配双编码器**：DFormerv2-B（67G FLOPs）= GeminiFusion（256G FLOPs）的精度，计算量仅 26%
- **几何先验贡献 +4.5 mIoU**：从 51.7 提升到 56.2，比任何其他深度利用方式都有效
- **首次证明深度图不需要编码器**——直接作为注意力先验就足够

## 亮点与洞察
- **"几何先验替代深度编码"的范式**非常优雅——深度图的信息本来就是显式的几何量，不需要神经网络"重新理解"
- **衰减因子 $\beta^G$ 的设计**简洁有效——通过指数衰减自然地将距离转化为注意力权重

## 局限与展望
- 依赖深度图质量，估计深度的误差会传播到几何先验
- 仅在室内场景（NYU/SUN）验证，室外场景深度范围差异很大
- Memory 基融合的可解释性不强

## 相关工作与启发
- **vs CMX / CMNext**：双编码器融合方案，计算量大。DFormerv2 用单编码器+几何先验达到同等效果
- **vs DFormer v1**：v1 也仅用单编码器但没有显式几何先验。v2 的 GSA 提供了更原理性的深度利用方式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 深度图作为注意力先验而非编码特征是范式创新
- 实验充分度: ⭐⭐⭐⭐ NYU/SUN/Deliver 三数据集，详细的融合策略对比
- 写作质量: ⭐⭐⭐⭐ 方法动机清楚
- 价值: ⭐⭐⭐⭐⭐ 对 RGBD 理解领域有重要意义，效率优势显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Soft Self-Labeling and Potts Relaxations for Weakly-Supervised Segmentation](soft_self-labeling_and_potts_relaxations_for_weakly-supervised_segmentation.md)
- [\[ECCV 2024\] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](../../ECCV2024/segmentation/sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [\[CVPR 2025\] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [\[CVPR 2026\] VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation](../../CVPR2026/segmentation/vggt-segmentor_geometry-enhanced_cross-view_segmentation.md)
- [\[ECCV 2024\] Attention Decomposition for Cross-Domain Semantic Segmentation](../../ECCV2024/segmentation/attention_decomposition_for_cross-domain_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
