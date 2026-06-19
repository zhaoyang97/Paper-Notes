---
title: >-
  [论文解读] Exploring Contextual Attribute Density in Referring Expression Counting (CAD-GD)
description: >-
  [CVPR 2025][指代表达计数] 提出上下文属性密度（Contextual Attribute Density, CAD）概念来增强指代表达计数（Referring Expression Counting），通过 U 形密度估计器、CAD 注意力和动态查询初始化三个模块，在 REC-8K 数据集上相比 GroundingREC 降低了约 30% 的计数误差（MAE 从 6.80 降至 5.43）。
tags:
  - "CVPR 2025"
  - "指代表达计数"
  - "上下文属性密度"
  - "开放世界检测"
  - "GroundingDINO"
  - "密度图"
---

# Exploring Contextual Attribute Density in Referring Expression Counting (CAD-GD)

**会议**: CVPR 2025  
**arXiv**: [2503.12460](https://arxiv.org/abs/2503.12460)  
**代码**: [github.com/Xu3XiWang/CAD-GD](https://github.com/Xu3XiWang/CAD-GD)  
**领域**: 其他  
**关键词**: 指代表达计数, 上下文属性密度, 开放世界检测, GroundingDINO, 密度图

## 一句话总结

提出上下文属性密度（Contextual Attribute Density, CAD）概念来增强指代表达计数（Referring Expression Counting），通过 U 形密度估计器、CAD 注意力和动态查询初始化三个模块，在 REC-8K 数据集上相比 GroundingREC 降低了约 30% 的计数误差（MAE 从 6.80 降至 5.43）。

## 研究背景与动机

**领域现状**：指代表达计数（REC）是一种新兴的计数任务，需要根据细粒度文本描述（如"walking person"而非简单的"person"）来计数特定属性的对象。GroundingREC 是第一个基于 GroundingDINO 的 REC 基线。

**现有痛点**：GroundingREC 在处理细粒度属性时存在两类错误：（1）过度计数——过度关注类别信息而忽略细粒度属性，将错误属性的对象也计入；（2）遗漏计数——由于遮挡和尺度变化，漏掉具有指定属性的对象。

**核心矛盾**：REC 本质上是一个检测-计数流程（一对一匹配），缺乏对空间密度分布的感知。传统计数方法已证明"视觉密度"对尺度鲁棒的空间分布建模至关重要，但现有开放世界模型忽略了这一能力。

**切入角度**：类比"视觉密度"概念，定义"上下文属性密度"——度量某个细粒度属性在不同尺度的视觉区域中的信息强度。通过建模 CAD 来引导模型更准确地对齐属性信息与视觉模式。

**核心 idea**：为开放世界检测器引入属性级密度图监督，使其能感知细粒度文本描述对应的属性空间分布。

## 方法详解

### 整体框架

基于 GroundingDINO 构建 CAD-GD 框架：图像和文本分别通过 backbone 提取特征后，经 Feature Enhancer 得到多尺度视觉特征 $\{F_{vi}\}_{i=1}^{4}$ 和文本特征 $F_t$。然后通过三大模块注入 CAD 信息：CAD 生成模块产生密度特征、CAD 注意力模块增强视觉特征、CAD 动态查询模块初始化解码器查询。

### 关键设计

1. **CAD 生成模块（U-shape CADE）**：先将视觉特征投射到文本空间计算相似性 $S_i = \text{Proj}(F_{vi}) \cdot F_t$，然后将相似性特征与视觉特征一起送入 U 形估计器，生成多尺度 CAD 特征 $\{D_i\}_{i=1}^{4}$，最终输出密度图并用 $\ell_2$ 损失监督（GT 密度图由高斯核生成）

2. **CAD 注意力模块**：分两步——空间注意力用 CAD 特征的通道池化（max+avg）生成空间权重图增强前景区域；通道注意力通过共享 MLP 对空间注意力增强后的特征做通道级加权，跨尺度区分不同属性

3. **CAD 动态查询初始化**：先用文本特征动态初始化查询内容（Text Init，$\dot{Q} = (Q \times (F_t \times M)^\top) \times F_t$），再用 CAD 特征通过交叉注意力进一步细化（Density Init），使不同指代表达的查询在特征空间中易于区分

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{loc}} + \alpha \cdot \mathcal{L}_{\text{density}}$$

其中 $\mathcal{L}_{\text{loc}}$ 是标准的定位损失（匈牙利匹配 + L1/GIoU），$\mathcal{L}_{\text{density}} = \|D_{\text{pred}} - D_{\text{gt}}\|_2^2$ 是密度图的 $\ell_2$ 回归损失。训练冻结视觉和文本 backbone，使用 AdamW，学习率 1e-5，20 个 epoch，第 10 epoch 衰减 10 倍。

## 实验关键数据

| 方法 | Backbone | Val MAE↓ | Val RMSE↓ | Val F1↑ | Test MAE↓ | Test RMSE↓ | Test F1↑ |
|------|----------|----------|-----------|---------|-----------|------------|----------|
| GroundingDINO | Swin-T | 9.03 | 21.98 | 0.65 | 8.88 | 21.95 | 0.66 |
| GroundingREC | Swin-T | 6.80 | 18.13 | 0.68 | 6.50 | 19.79 | 0.69 |
| **CAD-GD** | Swin-T | **5.43** | **15.01** | **0.70** | **5.29** | **17.08** | **0.72** |
| GroundingREC* | Swin-B | 5.66 | 15.24 | 0.71 | 5.42 | 18.47 | 0.70 |
| **CAD-GD** | Swin-B | **4.83** | **13.52** | **0.75** | **4.94** | **14.65** | **0.76** |

### 消融实验

| 模块组合 | Val MAE | Val RMSE | Val F1 |
|---------|---------|----------|--------|
| Baseline (无 CAD) | 6.52 | 17.72 | 0.665 |
| +CAD 生成 | 6.17 | 16.38 | 0.673 |
| +空间注意力 | 5.88 | 16.43 | 0.691 |
| +通道注意力 | 5.61 | 16.28 | 0.690 |
| +Text Init | 5.67 | 14.43 | 0.690 |
| +Density Init | 5.43 | 15.01 | 0.700 |
| +密度推理策略 | **4.83** | **13.52** | 0.695 |

### 关键发现

- 密度图推理策略（用密度图估计数量代替阈值）可额外降低 11% MAE
- CAD 密度图能区分同一类别不同属性的空间分布（如"bluish pen" vs "greenish pen"）
- 零样本 FSC-147 计数上也超越 GroundingREC（MAE 9.30 vs 10.06）

## 亮点与洞察

- **概念创新**——首次将密度估计引入跨模态的指代表达计数，定义了"上下文属性密度"这一新概念
- **查询可视化有说服力**——t-SNE 可视化清楚展示 CAD 初始化后不同属性查询能有效分离
- **即插即用**——CAD 模块可增强任何基于 DETR 类结构的开放世界检测器

## 局限与展望

- GT 密度图使用固定大小高斯核（σ=15），未适应目标尺度
- 密度推理策略提升计数但略微降低定位指标，两者存在不匹配
- 仅在 REC-8K（~8000 张图）上验证，数据集规模较小
- 对无关属性的复杂语义组合（如否定表达"not in a bus"）仍有改进空间

### 零样本计数泛化（FSC-147）

| 方法 | Val MAE | Val RMSE | Test MAE | Test RMSE |
|------|---------|----------|----------|-----------|
| GroundingREC | 10.06 | 58.62 | 10.12 | 107.19 |
| CountGD | 12.14 | 47.51 | 12.98 | 98.35 |
| **CAD-GD** | **9.30** | **40.96** | **10.35** | **86.88** |

## 相关工作

- **GroundingDINO / GroundingREC**：开放世界检测→REC 基线
- **密度估计计数**：CounTR, LOCA, CACViT, DAVE 等——证明密度特征对计数任务的价值
- **文本引导计数**：CLIP-Count, CounTX, CountGD, VLCounter——跨模态计数
- **密度建模在其他任务**：DQ-DETR（小目标检测）、Cholakka（实例分割）——密度先验的普适性

## 评分

- 新颖性: ⭐⭐⭐⭐ CAD 概念新颖，密度+检测的融合有启发性
- 实验充分度: ⭐⭐⭐⭐ 消融详尽，含零样本泛化验证
- 写作质量: ⭐⭐⭐⭐ 清晰，可视化丰富
- 价值: ⭐⭐⭐⭐ 为开放世界计数提供了新视角

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](../../NeurIPS2025/others/contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[ACL 2025\] Entropy-UID: A Method for Optimizing Information Density](../../ACL2025/others/entropy-uid_a_method_for_optimizing_information_density.md)
- [\[CVPR 2026\] Neural Mixture Density Processes](../../CVPR2026/others/neural_mixture_density_processes.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](../../ACL2025/others/contextual_experience_replay_for_self-improvement_of_language_agents.md)

</div>

<!-- RELATED:END -->
