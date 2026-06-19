---
title: >-
  [论文解读] Enhancing Transformers Through Conditioned Embedded Tokens
description: >-
  [ICCV 2025][语义分割][Transformer] 揭示 Transformer 自注意力矩阵存在固有的 ill-conditioning 问题，通过理论分析建立自注意力条件数与嵌入令牌条件数的直接关系，提出 Conditioned Embedded Tokens 方法（对嵌入矩阵添加基于 SVD 的修正项），在图像分类、目标检测、实例分割和 NLP 等多种任务上一致提升性能。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "Transformer"
  - "条件数"
  - "自注意力"
  - "嵌入令牌"
  - "优化稳定性"
---

# Enhancing Transformers Through Conditioned Embedded Tokens

**会议**: ICCV 2025  
**arXiv**: [2505.12789](https://arxiv.org/abs/2505.12789)  
**代码**: 无  
**领域**: 图像分割 / Transformer 通用改进  
**关键词**: Transformer, 条件数, 自注意力, 嵌入令牌, 优化稳定性

## 一句话总结

揭示 Transformer 自注意力矩阵存在固有的 ill-conditioning 问题，通过理论分析建立自注意力条件数与嵌入令牌条件数的直接关系，提出 Conditioned Embedded Tokens 方法（对嵌入矩阵添加基于 SVD 的修正项），在图像分类、目标检测、实例分割和 NLP 等多种任务上一致提升性能。

## 研究背景与动机

### 问题定义

Transformer 的核心是自注意力机制，它通过 $\mathbf{A}(X) = \text{softmax}(XW_QW_K^TX^T)XW_V$ 建模全局依赖关系。矩阵的**条件数**（最大奇异值与最小奇异值之比）是衡量优化难度的关键指标：条件数越大，梯度优化越不稳定。

### 已有方法的不足

1. 前馈网络中的条件数研究已有较多探索（权重矩阵预条件、NTK 条件改善），但**自注意力的条件数问题**几乎未被研究
2. 现有对 Transformer 的优化改进（如 skip connection、多头注意力）是间接改善条件数，缺乏系统的理论框架
3. 实际中嵌入令牌矩阵的条件数通常非常大，导致梯度不稳定

### 核心 idea

自注意力矩阵的条件数上界与嵌入令牌矩阵 $X$ 的条件数直接相关（线性注意力中 $\kappa(X)^3$）。通过 SVD 分解 $X$ 并添加修正项 $C$ 使 $\kappa(X+C) \leq 2$，可以显著降低自注意力的 ill-conditioning 程度。

## 方法详解

### 整体框架

对 Transformer 第一层输入的嵌入令牌矩阵 $X = [Ex_1 \cdots Ex_N]^T \in \mathbb{R}^{N \times d}$ 添加一个修正矩阵 $C \in \mathbb{R}^{N \times d}$，使得修正后的 $X+C$ 条件数大幅降低。修正项 $C$ 通过 $X$ 的 SVD 分解计算得到。修正后的矩阵送入 Transformer 的第一层。

### 关键设计

#### 1. 自注意力条件数分析

- **功能**：建立自注意力矩阵条件数的理论上界
- **核心结果**（Proposition 4.2）：
    - 线性注意力：$\kappa(\mathbf{LA}(X)) \leq \kappa(W_Q) \cdot \kappa(W_K) \cdot \kappa(W_V) \cdot \kappa(X)^3$
    - Softmax 注意力：$\kappa(\mathbf{A}(X)) \leq \kappa(\text{softmax}(XW_QW_K^TX^T)) \cdot \kappa(X) \cdot \kappa(W_V)$
- **设计动机**：$\kappa(X)$ 在实际训练中通常极大，是主要瓶颈；降低 $\kappa(X)$ 可同时降低整个自注意力的条件数

#### 2. Conditioned Embedded Tokens

- **功能**：构造修正矩阵 $C$ 使 $\kappa(X+C) \leq 2$
- **核心定理**（Theorem 4.4）：对任意 $\kappa(X) > 2$ 的嵌入矩阵，存在 $C$ 使得 $\kappa(X+C) \leq 2$
- **具体方法**：对 $X$ 做 SVD 分解，通过调整奇异值来构造最优修正项。修正项的计算是确定性的，不引入额外可学习参数
- **设计动机**：即使上界只是近似估计，实验表明条件数的降低与性能提升高度相关

#### 3. 跨层传播效应

- **功能**：验证第一层的条件数改善能传播到后续层
- **核心发现**：虽然理论只证明了第一层，但实验显示所有层（平均）的自注意力条件数都显著降低
- **设计动机**：第一层的输出作为第二层的输入，条件数的改善具有级联效应

### 损失函数 / 训练策略

不引入额外损失函数，仅作为 drop-in 替换现有嵌入层。所有原有训练配置保持不变。

## 实验关键数据

### 主实验

**ImageNet-1k 图像分类（Top-1 %）**

| 模型 | 原始 | +Conditioned | 提升 |
|------|------|-------------|------|
| ViT-Base | 80.3 | 81.3 | +1.0 |
| DeiT-Base | 81.6 | 82.5 | +0.9 |
| Swin-Base | 83.1 | 83.9 | +0.8 |
| XCiT-Medium | 82.2 | 82.9 | +0.7 |
| DaViT-Base | 83.6 | 84.6 | +1.0 |

**COCO 目标检测与实例分割（Mask R-CNN, AP）**

| 模型 | AP_box | AP50_box | AP_mask | AP50_mask |
|------|--------|---------|---------|----------|
| XCiT-S 原始 | 44.9 | 66.1 | 40.1 | 63.1 |
| XCiT-S +Cond. | **45.7** | **66.6** | **40.4** | **63.5** |
| XCiT-M 原始 | 45.7 | 66.8 | 40.8 | 63.6 |
| XCiT-M +Cond. | **46.2** | **67.4** | **41.4** | **63.8** |

**GLUE 基准（Crammed BERT, Accuracy）**

| 任务 | MNLI | SST-2 | RTE | QNLI | QQP | MRPC | CoLA | GLUE平均 |
|------|------|-------|-----|------|-----|------|------|---------|
| 原始 | 83.8 | 92.3 | 55.1 | 90.1 | 87.3 | 85.0 | 48.9 | 78.6 |
| +Cond. | **84.2** | **92.5** | **55.6** | **91.1** | **87.4** | **86.3** | **53.7** | **79.7** |

### 消融实验

**不同 Transformer 架构下条件数对比（ViT-B, 训练全程平均）**

| 指标 | 原始 | +Conditioned |
|------|------|-------------|
| 嵌入令牌 $\kappa(X)$ | ~10³ 量级 | ~10¹ 量级 |
| 第一层注意力 $\kappa$ | 显著更高 | 显著降低 |
| 全层注意力 $\kappa$ 平均 | 较高 | 显著降低 |

**GPT-2 验证损失（TinyStories）**

| 模型 | 验证损失↓ |
|------|----------|
| GPT-2 原始 | 2.41 |
| GPT-2 +Conditioned | **2.36** |

**Nyströmformer 长序列基准（LRA, Accuracy %）**

| 任务 | ListOps | Text | Retrieval | Image | Pathfinder |
|------|---------|------|-----------|-------|------------|
| 原始 | 37.1 | 63.8 | 79.8 | 39.9 | 72.9 |
| +Cond. | **37.9** | **64.9** | **80.9** | **40.1** | **73.3** |

### 关键发现

- 条件数的改善在所有测试架构（ViT、DeiT、Swin、XCiT、DaViT、BERT、GPT-2、Nyströmformer）上都一致有效
- 改进不仅对标准自注意力有效，对 shifted window、cross-covariance、Nyström 近似等高级注意力机制同样有效
- 第一层的条件数改善确实会级联传播到所有后续层
- 方法零额外参数、零额外损失，纯粹作为 drop-in 替换

## 亮点与洞察

1. **理论与实践的良好结合**：条件数分析提供了清晰的理论动机，虽然从条件数到优化收敛的完整证明尚缺，但实验一致支持结论
2. **通用性极强**：同一方法在 CV、NLP、长序列建模中都有效，且可直接嵌入各种现代 Transformer 架构
3. **实现简洁**：仅需对嵌入矩阵做 SVD 并添加修正项，不需要修改训练配置或引入超参数
4. **揭示了被忽视的优化瓶颈**：嵌入令牌的条件数经常达到 10³ 量级，这个问题之前几乎未被关注

## 局限与展望

1. **理论缺失最后一步**：未证明条件数改善→NTK 改善→优化收敛加速的完整链条
2. **softmax 注意力的条件数上界需要额外假设**（Eq.14 的条件假设）
3. **SVD 计算开销**：每次前向传播都需要对嵌入矩阵做 SVD，大规模模型中可能增加开销
4. **仅改善第一层的理论保证**：虽然实验显示多层都受益，但缺乏后续层的理论分析
5. **对非常深的 Transformer（如 LLM）的影响未充分验证**

## 相关工作与启发

- Weight conditioning [Saratchandran et al., 2025] 对前馈网络权重矩阵做预条件，本文将思路扩展到注意力机制
- NTK 条件数分析 [Liu et al., 2022] 证明了条件数与梯度下降收敛的关系，但仅限前馈网络
- Skip connection 被证明可改善注意力块的条件数 [Ji et al., 2025]，本文方法与之互补

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次系统分析自注意力的条件数并提出理论驱动的修正方法
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 5 种 CV 模型、2 种语言模型、1 种长序列模型、4 个任务，极为全面
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，实验组织有序
- **价值**: ⭐⭐⭐⭐ — 作为 drop-in 改进有很高的实用价值，但 SVD 开销可能限制大模型应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] What If: Understanding Motion Through Sparse Interactions](what_if_understanding_motion_through_sparse_interactions.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)

</div>

<!-- RELATED:END -->
