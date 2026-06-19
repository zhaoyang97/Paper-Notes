---
title: >-
  [论文解读] Handling Imbalanced Pseudolabels for Vision-Language Models with Concept Alignment and Confusion-Aware Calibrated Margin
description: >-
  [ICML2025][多模态VLM][VLM伪标签] 提出 CAP 框架，通过**概念对齐**（检测并修复 concept mismatch）和**混淆感知校准边距**（缓解 concept confusion），解决 VLM 生成伪标签时的类别不平衡问题，在六个数据集三种范式下相对 SOTA 提升 6.29%。
tags:
  - "ICML2025"
  - "多模态VLM"
  - "VLM伪标签"
  - "CLIP微调"
  - "伪标签不平衡"
  - "概念对齐"
  - "校准边距"
  - "无监督/半监督学习"
---

# Handling Imbalanced Pseudolabels for Vision-Language Models with Concept Alignment and Confusion-Aware Calibrated Margin

**会议**: ICML2025  
**arXiv**: [2505.02056](https://arxiv.org/abs/2505.02056)  
**代码**: [GitHub](https://anonymous.4open.science/r/CAP-C642/)  
**领域**: 多模态VLM  
**关键词**: VLM伪标签, CLIP微调, 伪标签不平衡, 概念对齐, 校准边距, 无监督/半监督学习

## 一句话总结

提出 CAP 框架，通过**概念对齐**（检测并修复 concept mismatch）和**混淆感知校准边距**（缓解 concept confusion），解决 VLM 生成伪标签时的类别不平衡问题，在六个数据集三种范式下相对 SOTA 提升 6.29%。

## 研究背景与动机

利用 VLM（如 CLIP）的零样本能力为下游无标注数据生成伪标签（pseudolabel）进行微调是近年热点。核心挑战在于：**VLM 对不同类别存在偏好偏差**，导致伪标签分布严重不平衡，进而引发确认偏差（confirmation bias）。

现有方法如 UPL、FPL 采取每类取 top-k 置信度样本强制平衡，GRIP 逐轮递增 k 值，CPL 为每个样本分配候选伪标签集合——但都是**事后补救**，未深入分析不平衡的根本原因。

本文首次深入剖析不平衡的两大成因：

**概念失配（Concept Mismatch）**：类名文本特征与图像特征严重不对齐，导致该类别几乎无法被正确预测（如 RESISC45 约 5% 类存在此问题）

**概念混淆（Concept Confusion）**：相似类别的文本特征无法捕捉最具区分性的视觉概念，导致预测偏向某一类（约 30% 类受影响）

作者在 RESISC45 上可视化了准确率最低 5 类的聚类分布——虽然图像特征聚类良好，但 CLIP 的零样本预测准确率极低，证实了语义鸿沟（semantic gap）的存在。

## 方法详解

### 整体框架 CAP

CAP（Concept-Adaptive Pseudolabeling）分三步：

1. **概念对齐**（§3.1）：检测 concept mismatch 类别，用 LLM 增强文本描述
2. **混淆感知校准边距**（§3.2）：基于类间相似度和预测倾向构建边距矩阵
3. **双适配器微调**（§3.3）：分别从高质量伪标签和动态伪标签学习

### 3.1 概念对齐（Concept Alignment）

**Mismatch 检测算法**：迭代式聚类策略逐步移除匹配良好的类，剩余即为 mismatch 类。

- 对图像特征 $\mathcal{I}$ 做 K-Means 聚类（簇数=类数）
- 计算文本特征与聚类中心的相似度矩阵 $\mathbf{S}^{\mathcal{TC}}$，softmax 得到概率矩阵
- 找到置信度最高的 (文本特征, 聚类中心) 对 $(i^*, j^*)$，将其移除
- 迭代直到剩余类数低于阈值 $t$，剩余类即候选 mismatch 类
- 取与预测样本最少类的交集 $\mathcal{Y}_{\text{MM}} = \mathcal{Y}_{\text{final}} \cap \mathcal{Y}_{\text{low-}t}$

**LLM 文本增强**：对 mismatch 类，调用 LLM 生成 $n$ 个增强描述，选出与聚类中心相似度最高的描述，替代原始类名模板，按 top-k 余弦相似度分配伪标签。

### 3.2 混淆感知校准边距（Confusion-Aware Calibrated Margin）

核心思想：在交叉熵损失中加入自适应边距，鼓励模型在容易混淆的类之间做出更有区分度的预测。

**校准边距损失**：

$$\mathcal{L}_m(y, \mathbf{z}) = -\log \frac{e^{z_y}}{e^{z_y} + \sum_{c \neq y} e^{z_c + \mathbf{M}_{yc}}}$$

**边距矩阵 $\mathbf{M}$ 的构建**：

1. **类间相似度矩阵** $\mathbf{S}$：取视觉原型相似度和文本原型相似度的最大值
   $$\mathbf{S}_{ij} = \max(\text{sim}(\bar{\mathbf{v}}_i, \bar{\mathbf{v}}_j), \text{sim}(\mathbf{w}_i, \mathbf{w}_j))$$

2. **类别预测倾向** $\delta_c$：统计置信度超过阈值 $\tau$ 且被预测为类 $c$ 的样本数 $\sigma(c)$
   $$\delta_c = 1 - \frac{\sigma(c)}{\max_j \sigma(j)}$$

3. **类别边距缩放** $m_c = m \times \Delta \times \delta_c$，其中 $\Delta = \max_c(\delta_c)$

4. **最终边距矩阵**：$\mathbf{M} = \mathbf{S} \odot \mathbf{m}$（Hadamard 积）

关键设计：$\mathbf{M}$ 每个 epoch 更新一次，逐步缓解混淆。对预测倾向低（$\delta_c$ 大）且与其他类相似的类，施加更大边距惩罚。

### 3.3 双适配器微调框架

基于 MaPLe 提示调优，部署两个独立视觉适配器：

- **Main Adapter**（$\phi^m$）：仅从概念对齐阶段的高精度伪标签 $\mathcal{D}_{\text{PL}}$ 学习，同时为未标注数据生成伪标签
- **Pseudo Adapter**（$\phi^p$）：仅从动态伪标签的未标注数据 $\mathcal{D}_{\text{UL}}$ 学习（FixMatch 风格，阈值 $\tau$ 过滤）
- 文本分支也部署适配器 $\psi^a$
- **推理时禁用所有适配器**

总损失：$\mathcal{L} = \mathcal{L}_{\text{PL}} + \mathcal{L}_{\text{UL}}$（无监督），SSL/TRZSL 额外加 $\mathcal{L}_{\text{L}}$

## 实验关键数据

在 6 个数据集（Flowers102、RESISC45、DTD、EuroSAT、CUB、FGVCAircraft）上对比 3 种范式（SSL / UL / TRZSL）：

| 方法 | Flowers102 (UL) | RESISC45 (UL) | DTD (UL) | EuroSAT (UL) | CUB (UL) |
|------|---------|---------|------|---------|------|
| Zero-shot CLIP | 63.40 | 54.46 | 43.45 | 30.54 | 51.57 |
| FPL | 65.67 | 68.13 | 44.96 | 48.96 | 53.04 |
| GRIP | 69.84 | 74.11 | 46.09 | 57.21 | 51.42 |
| CPL | 72.90 | 80.98 | 51.91 | 67.26 | — |
| **CAP (Ours)** | **76.80** | **83.32** | **55.29** | — | — |

- 在 Flowers102 UL 上比 CPL 提升 3.9 个点，RESISC45 UL 提升 2.3 个点
- 三种范式（SSL/UL/TRZSL）均取得 SOTA；整体相对 CPL 提升 **6.29%**
- 特别在 concept mismatch 严重的类上改善显著

## 亮点与洞察

1. **问题分析深入**：首次将伪标签不平衡归因于 concept mismatch 和 concept confusion 两种语义鸿沟表现形式，并给出定量统计（5% 类 mismatch、30% 类 confusion）
2. **迭代聚类检测**：用无监督方式自动发现 mismatch 类别，无需任何标注
3. **边距矩阵设计精巧**：联合类间相似度和预测倾向，自适应调节不同类对之间的决策边界
4. **双适配器隔离噪声**：main adapter 保持高精度不受动态伪标签噪声污染，架构简洁有效
5. **覆盖范式全面**：UL / SSL / TRZSL 三种学习范式统一框架

## 局限与展望

1. **LLM 依赖**：概念对齐需要调用 LLM 生成文本描述，增加了推理流程复杂度和成本
2. **超参数敏感性**：阈值 $t$（mismatch 检测）、$\tau$（置信度过滤）、边距尺度 $m$ 等需调优
3. **仅限分类任务**：框架设计围绕图像分类，对检测/分割等任务的适用性未探讨
4. **数据集规模有限**：6 个数据集均为中等规模，未验证 ImageNet 等大规模场景
5. **推理时关闭适配器**：训练时有适配器但推理时没有，训练-推理不一致可能限制上限

## 评分
- 新颖性: ⭐⭐⭐⭐ — 问题分析（mismatch vs confusion）有新意，迭代聚类检测 + 校准边距组合方案原创
- 实验充分度: ⭐⭐⭐⭐ — 6 数据集 × 3 范式，消融完整，但缺少大规模验证
- 写作质量: ⭐⭐⭐⭐ — 动机图示清晰、公式推导完整，整体可读性好
- 价值: ⭐⭐⭐⭐ — VLM 伪标签不平衡是实际痛点，方法可迁移到其他 VLM 微调场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization](cocoa-mix_confusion-and-confidence-aware_mixture_model_for_context_optimization.md)
- [\[CVPR 2026\] CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](../../CVPR2026/multimodal_vlm/capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)
- [\[ICLR 2026\] Unified Vision-Language Modeling via Concept Space Alignment](../../ICLR2026/multimodal_vlm/unified_vision-language_modeling_via_concept_space_alignment.md)
- [\[ICML 2025\] Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models](kernel-based_unsupervised_embedding_alignment_for_enhanced_visual_representation.md)
- [\[CVPR 2026\] Quota-Calibrated Fine-Grained Alignment with Context-Aware Marginals for Text-based Person Retrieval](../../CVPR2026/multimodal_vlm/quota-calibrated_fine-grained_alignment_with_context-aware_marginals_for_text-ba.md)

</div>

<!-- RELATED:END -->
