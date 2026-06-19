---
title: >-
  [论文解读] A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features
description: >-
  [ICML 2025][模型压缩][跨模态知识蒸馏] 提出 Semi-Clipped（基于 CLIP 的跨模态蒸馏方法）和 PEA（扰动嵌入增强），在弱配对数据场景下将显微镜图像的丰富形态学特征蒸馏到转录组学表征中，在保持基因表达可解释性的同时显著提升其预测能力。 核心问题 理解细胞对各种刺激的响应是生物发现和药物研发的基础…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "跨模态知识蒸馏"
  - "数据增强"
  - "转录组学"
  - "显微镜成像"
  - "弱配对学习"
---

# A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features

**会议**: ICML 2025  
**arXiv**: [2505.21317](https://arxiv.org/abs/2505.21317)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 跨模态知识蒸馏, 数据增强, 转录组学, 显微镜成像, 弱配对学习

## 一句话总结

提出 Semi-Clipped（基于 CLIP 的跨模态蒸馏方法）和 PEA（扰动嵌入增强），在弱配对数据场景下将显微镜图像的丰富形态学特征蒸馏到转录组学表征中，在保持基因表达可解释性的同时显著提升其预测能力。

## 研究背景与动机

### 核心问题

理解细胞对各种刺激的响应是生物发现和药物研发的基础。当前存在两种互补的生物数据模态：

- **转录组学（Transcriptomics）**：提供基因级别的可解释性洞察，但预测能力相对较弱
- **显微镜成像（Microscopy Imaging）**：具有丰富的视觉表型特征和强大的预测能力，但难以解释

两种模态各有优劣，理想情况是将它们结合以构建更全面的生物系统表征。然而，**收集样本级别配对的多模态数据代价极高**，在技术和成本上都不可行。

### 弱配对数据的挑战

实际场景中只能获得**弱配对数据集（Weakly Paired Datasets）**——来自不同模态的样本并非来自同一生物重复实验，但共享相同的细胞系和扰动条件等关键元数据。即便是弱配对数据集也极度稀缺，这严重限制了多模态学习的训练和推理。

### 动机

- 大多数跨模态蒸馏技术依赖监督目标函数，需要精确标签——但在生物模态中这些标签往往不可用
- 无监督对齐方法试图发现模态间的共享结构，但由于两种模态捕获的生物关系本质不同，这很有挑战性
- 需要一种能在数据稀缺条件下工作、仅需弱配对信息、且推理时只使用单模态的方法

## 方法详解

### 整体框架

本文提出的框架包含两个核心组件：

1. **Semi-Clipped**：跨模态知识蒸馏方法，将显微镜图像（教师模态 $T$）的知识转移到转录组学（学生模态 $S$）
2. **PEA（Perturbation Embedding Augmentation）**：生物学启发的数据增强技术，专为转录组学表征设计

**问题形式化**：给定教师模态数据集 $\mathcal{X}_T$ 和学生模态数据集 $\mathcal{X}_S$，样本 $x_T^{(i)}$ 和 $x_S^{(i)}$ 对应相同的生物扰动和细胞类型，但由于生物变异性不是强配对。每个样本标注有弱标签 $p$（扰动类型）和 $l$（细胞类型）。数据按实验批次（biological batches）组织，每个批次包含扰动样本和对照（未扰动）样本。

### 关键设计

#### Semi-Clipped：冻结教师的 CLIP 蒸馏

核心思想是利用冻结的预训练单模态编码器，通过可训练适配器实现单向知识传递：

1. **冻结编码器**：使用预训练的单模态编码器 $E_T$ 和 $E_S$，分别生成教师嵌入 $z_T^{(i)} = E_T(x_T^{(i)})$ 和学生嵌入 $z_S^{(i)} = E_S(x_S^{(i)})$
2. **可训练适配器**：学习映射函数 $f_S: \mathbb{R}^{d_S} \to \mathbb{R}^{d_T}$，将学生嵌入对齐到教师嵌入空间，生成 $h_S^{(i)} = f_S(z_S^{(i)})$
3. **CLIP 损失对齐**：在 $h_S$ 和 $z_T$ 之间优化 CLIP 损失

**关键区别于标准 CLIP**：

- 标准 CLIP 双向训练两个编码器，容易在共享信息有限时发生模态漂移
- Semi-Clipped **冻结教师端**，仅训练学生端的适配器 $f_S$
- 这确保了**单向知识转移**（教师→学生），避免了双向漂移问题
- 弱标签仅用于构建配对关系，而非作为学习目标

#### PEA：扰动嵌入增强

PEA 的核心创新是将**批次效应校正（batch correction）重新目的化为数据增强策略**：

**生物学背景**：生物数据集中的批次效应（实验条件差异引入的变异性）通常作为噪声需要被消除。传统批次校正技术通过将嵌入中心化到每个批次的对照样本上来降噪。

**PEA 的做法**：

1. 从预定义的批次校正变换集合 $\mathcal{A}$ 中随机选择一种变换 $A$
2. 对学生嵌入施加增强：$z_{S,A}^{(i)} = A(z_S^{(i)}, X_S^{(c)})$，其中 $X_S^{(c)}$ 为对照样本
3. 增强后的嵌入传入学生适配器 $f_S$ 进行跨模态蒸馏
4. 教师端使用固定的批次校正 $B$（TVN）

**PEA 的关键优势**：

- 利用不同批次校正方法产生的变换差异作为"增强"来源
- 每种校正方法消除的噪声模式不同，随机切换等价于引入有生物学意义的变异
- 增强后的数据保留了核心生物信息（扰动信号），同时丰富了训练数据多样性
- 特别适合表征级别（representation-level）的增强，区别于图像级的旋转/翻转

### 损失函数 / 训练策略

**损失函数**：采用标准 CLIP 对比损失，在 $h_S = f_S(z_{S,A})$（增强后经适配器映射的学生嵌入）和 $z_T$（经批次校正后的冻结教师嵌入）之间计算。正样本对由共享相同弱标签（扰动+细胞系）的跨模态样本构成。

**训练策略要点**：

- 教师编码器 $E_T$ 和学生编码器 $E_S$ 均**冻结**不参与训练
- 仅训练学生适配器 $f_S$（轻量级 MLP）
- PEA 增强在训练过程中在线随机应用
- 教师端统一使用 Typical Variation Normalization (TVN) 进行批次校正
- 推理时仅需学生模态（转录组学）数据

## 实验关键数据

### 主实验

论文在药物发现相关的生物学下游任务上进行评估，主要包括扰动检索（perturbation retrieval）和基因关系发现等任务。

| 方法 | 模态 | 弱配对依赖 | 检索性能 | 特点 |
|------|------|-----------|---------|------|
| scVI (baseline) | 转录组学 | 无 | 基线水平 | 单模态基础模型 |
| 标准 CLIP | 双模态 | 是 | 中等 | 双向漂移问题 |
| CSA | 双模态 | 是 | 较好 | few-shot 对齐 |
| **Semi-Clipped** | 蒸馏→单模态 | 是 | **SOTA** | 冻结教师，防漂移 |
| **Semi-Clipped + PEA** | 蒸馏→单模态 | 是 | **最优** | 增强进一步提升 |

### 消融实验

| 配置 | 关键指标变化 | 说明 |
|------|------------|------|
| Semi-Clipped（无增强） | 基线 | 仅蒸馏已达 SOTA |
| + 传统增强（Cutout/Mixup） | 无显著提升或下降 | 传统增强不适用于生物表征 |
| + Gaussian Noise | 轻微提升 | 随机噪声有一定正则化作用 |
| + PEA（单一校正） | 中等提升 | 单一校正方法的增强有限 |
| + **PEA（随机多校正）** | **最大提升** | 多种校正方法的组合效果最佳 |
| 双向 CLIP（两端均训练） | 性能下降 | 证实模态漂移问题 |
| 仅冻结学生、训练教师 | 性能大幅下降 | 蒸馏方向错误 |

### 关键发现

1. **Semi-Clipped 在数据稀缺条件下显著优于标准 CLIP 和其他对齐方法**——冻结教师端是成功的关键
2. **PEA 在各类增强方法中表现最优**——传统的 Cutout、Mixup、高斯噪声等增强方法在生物表征上效果有限甚至有害
3. **蒸馏后的转录组学表征在下游任务上超越了原始显微镜特征**——说明蒸馏不仅转移了教师知识，还与学生模态的原始信息形成了互补
4. **PEA 在发现新基因关系方面表现尤为突出**——大幅超越其他增强技术

## 亮点与洞察

1. **逆向利用批次效应作为增强**：传统上批次效应被视为需要消除的噪声，本文创造性地将其转化为增强来源。不同批次校正方法产生不同的"视角"，随机切换等价于让模型从多角度观察生物信号
2. **单向蒸馏策略**：通过冻结教师端，Semi-Clipped 巧妙解决了弱配对场景下的模态漂移难题。这一设计思路对其他弱配对跨模态学习场景也有启发
3. **推理时仅需单模态**：训练时利用多模态数据，推理时仅需更便宜、更可解释的转录组学数据——这种"训练-推理不对称"设计在实际应用中非常实用
4. **方法的通用性**：Semi-Clipped 和 PEA 的思路可推广到其他弱配对多模态场景（如蛋白质组学-转录组学、影像-基因组学等）

## 局限与展望

1. **增强方法集合的选择**：PEA 依赖预定义的批次校正方法集合，如何自动选择或学习最优的校正方法组合值得探索
2. **仅限于弱配对场景**：对强配对数据（若存在）是否还有优势，文中未充分讨论
3. **适配器架构**：目前使用简单 MLP 作为适配器，更复杂的架构（如 cross-attention）是否能带来进一步提升
4. **可扩展性**：当弱配对数据集规模增大时，Semi-Clipped 的冻结策略是否仍是最优选择
5. **生物学验证**：蒸馏后的表征发现的新基因关系仍需实验验证

## 相关工作与启发

- **CLIP (Radford et al., 2021)**：Semi-Clipped 的直接灵感来源，本文通过冻结一侧将其适配为蒸馏工具
- **CSA (Li et al., 2024)**：使用预训练单模态模型的 few-shot 对齐，但双向训练限制了其在稀缺数据下的效果
- **scVI (Lopez et al., 2018)**：转录组学领域的基础模型，作为学生编码器使用
- **VICReg (Bardes et al., 2022)**：自监督对齐方法，但需要模态间有显著共享信息
- **批次校正文献**：TVN (Ando et al., 2017) 等批次校正方法是 PEA 的技术基础

**启发**：本文的"将预处理/后处理步骤重新利用为训练时增强"的思路具有普遍意义——在其他领域中也可能存在类似的"被浪费的变换多样性"可以挖掘。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | Semi-Clipped 简洁有效，PEA 的"批次效应→增强"思路新颖 |
| 技术深度 | 3.5 | 方法本身不复杂，但问题建模和生物学结合做得好 |
| 实验充分度 | 4 | 多个下游任务和消融实验，有生物学意义的评估 |
| 写作质量 | 4 | 结构清晰，问题动机阐述充分 |
| 实用价值 | 4.5 | 解决了药物发现中的实际痛点，推理成本低 |
| **综合** | **4.0** | 实用性强的跨模态蒸馏工作，生物学洞察有价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](../../NeurIPS2025/model_compression/atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](../../AAAI2026/model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[ACL 2025\] Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](../../ACL2025/model_compression/data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)
- [\[AAAI 2026\] Asymmetric Cross-Modal Knowledge Distillation: Bridging Modalities with Weak Semantic Consistency](../../AAAI2026/model_compression/asymmetric_cross-modal_knowledge_distillation_bridging_modalities_with_weak_sema.md)
- [\[CVPR 2025\] Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](../../CVPR2025/model_compression/multi-modal_knowledge_distillation-based_human_trajectory_forecasting.md)

</div>

<!-- RELATED:END -->
