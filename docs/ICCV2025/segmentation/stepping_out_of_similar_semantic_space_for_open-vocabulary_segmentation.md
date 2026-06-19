---
title: >-
  [论文解读] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation
description: >-
  [ICCV 2025][语义分割][开放词汇分割] 揭示现有开放词汇分割（OVS）测试集与训练语义空间高度相似的评估偏差，提出新基准 OpenBench 和方法 OVSNet，通过梯度无关聚合（GFA）融合异构特征和代理校准（PC）零成本扩展训练空间，在已有基准和 OpenBench 上均取得 SOTA。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "开放词汇分割"
  - "基准评估"
  - "语义空间"
  - "CLIP微调"
  - "梯度无关聚合"
  - "代理校准"
---

# Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation

**会议**: ICCV 2025  
**arXiv**: [2506.16058](https://arxiv.org/abs/2506.16058)  
**代码**: 未公开  
**领域**: 图像分割  
**关键词**: 开放词汇分割, 基准评估, 语义空间, CLIP微调, 梯度无关聚合, 代理校准

## 一句话总结

揭示现有开放词汇分割（OVS）测试集与训练语义空间高度相似的评估偏差，提出新基准 OpenBench 和方法 OVSNet，通过梯度无关聚合（GFA）融合异构特征和代理校准（PC）零成本扩展训练空间，在已有基准和 OpenBench 上均取得 SOTA。

## 研究背景与动机

开放词汇分割（OVS）旨在根据任意文本输入分割图像中的对象。现有方法主要借助 CLIP 等大规模预训练视觉-语言模型，发展出几条技术路线：两阶段先分割后分类、冻结 CLIP 特征集成分割器、微调 CLIP 编码器等。

本文提出一个关键发现：**微调 CLIP 在现有测试集上持续提升性能，但这与开放词汇任务本身的目标矛盾**。微调 CLIP 会使其适配特定训练语义空间，降低泛化能力——而泛化正是 OVS 的核心需求。

通过统计分析，作者揭示了问题根源：

- **现有测试集与训练集语义高度重叠**：VOC 与训练集的平均相似度高达 0.97，PC-59 为 0.95，即使被认为"困难"的 ADE-847 也达到 0.79
- **评估偏差**：在这种高重叠下，微调 CLIP 的性能提升实际是过拟合训练语义的表现，而非真正的开放词汇理解能力提升

这一发现驱动了两个方面的工作：

**评估层面**：构建与训练语义显著不同的 OpenBench（平均相似度仅 0.61，最大 0.79），包含 286 个细粒度类别和 6056 张图像，无语义重复问题

**方法层面**：设计 OVSNet，在不牺牲 CLIP 泛化能力的前提下提升分割性能

## 方法详解

### 整体框架

OVSNet 基于 CLIP 视觉和文本编码器，结合训练的分割解码器（Mask2Former）：
1. CLIP 图像编码器提取视觉特征，分割解码器生成 mask proposal 和 query 嵌入
2. 通过 Mask Pooling 从预测 mask 区域提取 CLIP 特征
3. 梯度无关聚合（GFA）融合 query 嵌入和 CLIP 特征
4. 聚合后的视觉嵌入与文本嵌入进行视觉-语言对齐
5. 训练时使用代理校准（PC）扩展训练空间

### 关键设计一：梯度无关聚合（Gradient-Free Aggregation）

核心矛盾：分割解码器的 query 嵌入对训练类别有强区域对齐先验，但对新类别泛化差；CLIP mask-pooled 特征泛化性强，但缺乏区域级对齐。

学习型融合（如 self-attention、cross-attention）在训练中会过度依赖 query 嵌入而忽视 CLIP 特征，损害泛化能力。因此采用受 Random Walk 算法启发的梯度无关方式：

初始化亲和矩阵：$\mathcal{Z} = \lambda F_C^0 (F_Q^0)^\top$

迭代更新公式：

$$F_Q^t = \omega \text{Norm}(\mathcal{Z})^\top F_C^{t-1} + (1-\omega) F_Q^0$$

$$F_C^t = \omega \mathcal{Z} F_Q^t + (1-\omega) F_C^0$$

利用 Neumann 级数近似 $t \to \infty$ 时的闭式解：

$$F_C^\infty = (1-\omega)(I - \omega^2 A)^{-1}(\omega \mathcal{Z} F_Q^0 + F_C^0)$$

其中 $A = \mathcal{Z} \text{Norm}(\mathcal{Z})^\top$，$\omega \in (0,1)$ 控制融合程度。整个过程不引入梯度，避免了训练语义的过拟合。

### 关键设计二：代理校准（Proxy Calibration）

更广的语义训练空间有助于更强的泛化表示，但分割任务扩展训练空间成本高昂。PC 通过对训练嵌入的凸组合生成代理嵌入，零成本模拟未见语义：

$$F'_{Qmn} = \alpha \cdot F_{Qm} + (1-\alpha) \cdot F_{Qn}$$

其中 $\alpha \sim \text{Beta}(\gamma, \gamma)$。对 query 嵌入 $F_Q$、CLIP 特征 $F_C$ 和文本嵌入 $F_T$ 同步执行凸组合，然后施加余弦距离监督：

$$\mathcal{L}_{PQ} = 1 - \frac{\mathbf{F'_Q} \cdot \mathbf{F'_T}}{\|\mathbf{F'_Q}\|_2 \|\mathbf{F'_T}\|_2}, \quad \mathcal{L}_{PC} = 1 - \frac{\mathbf{F'_C} \cdot \mathbf{F'_T}}{\|\mathbf{F'_C}\|_2 \|\mathbf{F'_T}\|_2}$$

### 损失函数

总损失 = 分割损失（dice loss + CE loss，权重5）+ 分类损失（CE loss，权重2）+ 代理损失（$\mathcal{L}_{PQ} + \mathcal{L}_{PC}$）

## 实验

### 主实验

在现有基准和 OpenBench 上的全面对比（Base-level CLIP Backbone）：

| 方法 | ADE-150 | ADE-847 | PC-59 | PC-459 | VOC | OpenBench | Avg. |
|------|---------|---------|-------|--------|-----|-----------|------|
| SAN | 27.5 | 10.1 | 53.8 | 12.6 | 94.0 | 39.6 | 39.6 |
| CATSeg | 31.8 | 12.0 | 57.5 | 19.0 | 94.6 | 36.1 | 41.8 |
| MAFT+ | 34.6 | 13.8 | 57.5 | 16.2 | 95.4 | 43.7 | 43.5 |
| **OVSNet** | **35.8** | **14.5** | **58.6** | **19.1** | **95.7** | **44.9** | **44.8** |

Large-level CLIP Backbone 下 OVSNet 平均 47.4，超越 MAFT+（46.0）和 SED（45.6）。

关键发现：微调 CLIP 的 CATSeg 在现有基准上表现良好（ADE-150: 31.8），但在 OpenBench 上仅 36.1，显著低于冻结 CLIP 的方法。这验证了微调 CLIP 是对训练语义的过拟合。

### 消融实验

| 方法 | ADE-150 | PC-459 | OpenBench |
|------|---------|--------|-----------|
| Baseline | 33.1 | 14.3 | 42.3 |
| + GFA | 34.7 (+1.6) | 16.0 (+1.7) | 43.7 (+1.4) |
| + PC | 33.9 (+0.8) | 17.2 (+2.9) | 44.3 (+2.0) |
| + Both | **35.8** (+2.7) | **19.1** (+4.8) | **44.9** (+2.6) |

GFA vs 学习型融合（OpenBench）：

| 方法 | ADE-150 | PC-459 | OpenBench |
|------|---------|--------|-----------|
| Self Attention | 34.0 | 14.7 | 40.6 |
| Cross Attention | 34.2 | 14.8 | 41.2 |
| GFA | **34.7** | **16.0** | **43.7** |

关键发现：
1. GFA 在 OpenBench 上相比 Cross Attention 提升 2.5 mIoU，证明学习型融合确实过拟合训练语义
2. PC 的最大收益在 PC-459（+2.9）和 OpenBench（+2.0）上，即类别多且与训练集差异大的场景
3. $\gamma=2$（Beta 分布中间概率密度高）效果最佳

### 推理类别数的影响

有趣发现：相同模型和图像，推理时给定的类别越多，性能越差。SAN 和 MAFT+ 在 ADE-150 上的性能随无关类别数增加呈持续下降趋势。

## 亮点与洞察

1. **评估层面的深刻反思**：首次系统量化现有 OVS 测试集与训练集的语义重叠度，揭示了评估偏差的本质
2. **OpenBench 设计**：解决了 PC-459 和 ADE-847 的语义重复问题（fine-grained 但无 duplication），包含"others"类别更贴近真实场景
3. **GFA 的理论优雅性**：利用 Neumann 级数闭式解避免迭代计算，同时消除过拟合风险
4. **PC 的零成本**：仅在嵌入空间做凸组合，不需要额外数据或标注

## 局限性

1. OpenBench 主要从现有分割数据集筛选而来，覆盖范围受限于源数据集
2. GFA 的 Neumann 级数近似需要矩阵求逆，输入规模较大时计算成本不可忽略
3. 在与训练集高度相似的 VOC 上，性能略低于微调 CLIP 的方法（因其本质上保留了 CLIP 泛化空间而非最大化训练集性能）
4. $\omega$ 和 $\gamma$ 等超参数的选择需要一定调参

## 相关工作

- **两阶段 OVS**：SimSeg、OVSeg 先生成 mask proposal 再用 CLIP 分类
- **统一空间方法**：SAN、FCCLIP 将 CLIP 特征集成到分割器中
- **CLIP 微调方法**：CATSeg、SED 基于 cost map 的早期融合范式
- **视觉-语言预训练**：CLIP、ALIGN 等大规模对比学习模型

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 评估偏差的洞察和 OpenBench 具有领域贡献价值
- **技术质量**: ⭐⭐⭐⭐ — GFA 和 PC 设计简洁有效，消融充分
- **实用性**: ⭐⭐⭐⭐ — OpenBench 可供社区评估使用，方法有实际提升
- **写作质量**: ⭐⭐⭐⭐ — 动机论证有力，数据分析充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Exploring Simple Open-Vocabulary Semantic Segmentation](../../CVPR2025/segmentation/exploring_simple_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Auto-Vocabulary Semantic Segmentation](auto-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
