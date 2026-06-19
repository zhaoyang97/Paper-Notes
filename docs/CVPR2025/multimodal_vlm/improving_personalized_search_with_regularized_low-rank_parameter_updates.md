---
title: >-
  [论文解读] Improving Personalized Search with Regularized Low-Rank Parameter Updates
description: >-
  [CVPR 2025][多模态VLM][个性化检索] 本文提出POLAR方法，通过对CLIP文本编码器**最后一层**的value矩阵施加**rank-1的LoRA更新**加正则化，仅用少量样本即可学习个性化概念并保留通用知识，在DeepFashion2和ConCon-Chi基准上超越基于文本反转的先前方法4%~22%。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "个性化检索"
  - "LoRA微调"
  - "文本编码器"
  - "灾难性遗忘"
  - "视觉语言模型"
---

# Improving Personalized Search with Regularized Low-Rank Parameter Updates

**会议**: CVPR 2025  
**arXiv**: [2506.10182](https://arxiv.org/abs/2506.10182)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 个性化检索、LoRA微调、文本编码器、灾难性遗忘、视觉语言模型

## 一句话总结

本文提出POLAR方法，通过对CLIP文本编码器**最后一层**的value矩阵施加**rank-1的LoRA更新**加正则化，仅用少量样本即可学习个性化概念并保留通用知识，在DeepFashion2和ConCon-Chi基准上超越基于文本反转的先前方法4%~22%。

## 研究背景与动机

个性化视觉-语言检索（PerVL）旨在让预训练的双编码器模型（如CLIP）识别新的个人概念（如"我的狗Fido"），并在不同上下文中检索该概念（如"Fido在接飞盘"）。现有方法主要基于**文本反转**：学习一个伪文本token来代表新概念，插入到查询文本中。这种方法有两个关键问题：(1) 伪token影响整个编码过程，容易干扰语言编码器的通用知识；(2) 概念的表达能力局限于单个输入token。核心矛盾在于：**如何在极少样本下学到个性化概念，同时不遗忘模型的通用知识？** 本文的切入角度：与其在输入端插入伪token，不如直接对模型内部参数做极小的、正则化的低秩更新，在编码过程的最后阶段注入个性化信息。

## 方法详解

### 整体框架

POLAR（PersOnalized Low-rank Adaptation for Retrieval）在CLIP文本编码器的最后一层attention中，对value变换矩阵学习一个rank-1的LoRA更新。使用固定词汇token（如"sks"）作为个人概念的占位符。训练时用MSE损失拉近文本嵌入和图像嵌入，同时用 $L_2$ 正则化约束更新幅度以保留通用知识。多概念查询通过直接相加各概念的LoRA参数实现。

### 关键设计

1. **Rank-1 Value LoRA更新**:
    - 功能：以最少参数学习个性化概念
    - 核心思路：对文本编码器最后一层的value矩阵 $V_L$ 学习低秩更新 $V'_{L,c} = V_L + B_{L,c} A_{L,c}$，其中 $B \in \mathbb{R}^{d \times 1}$，$A \in \mathbb{R}^{1 \times d}$。每个概念只需存储 $2d$ 个参数
    - 设计动机：rank-1反映了"用极少样本学一个概念"的本质需求，最小化对原有表示的干扰。选择value矩阵（而非Q/K）是因为实验表明value在最后一层对最终表示影响最直接

2. **结构化正则化策略**:
    - 功能：防止灾难性遗忘通用知识
    - 核心思路：两个约束——(1) 对 $B_{L,c}$ 施加 $L_2$ 正则化 $\mathcal{L}_{\text{reg}} = |B_{L,c}|^2$，控制更新幅度；(2) 约束 $\|A_{L,c}\|_2 = 1$，让 $A$ 只学习"何时激活"的方向信息，更新大小完全由正则化的 $B$ 控制。总损失为 $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{reg}}$
    - 设计动机：利用rank-1分解的结构——$A \cdot x$ 可理解为检测输入与个性化方向的相似度，$B$ 控制更新的方向和大小。当 $BA = 0$ 时，编码器退化为原始CLIP，因此正则化 $B$ 直接控制偏离程度

3. **多概念参数合并**:
    - 功能：支持引用多个个性化概念的查询（如"Fido在玩Rex的飞盘"）
    - 核心思路：直接将多个概念的LoRA更新相加 $V'_{L,c_1+c_2} = V'_{L,c_1} + V'_{L,c_2}$，等价于构造一个rank-2的联合更新
    - 设计动机：参数加法是最简单且有效的合并策略，利用了低秩更新的可组合性

### 损失函数 / 训练策略

- **MSE损失**：将归一化后的文本嵌入和图像嵌入拉近：$\mathcal{L}_{\text{MSE}} = \frac{1}{N_c} \sum_i \left(\frac{\psi'_T(q_i)}{\|\psi'_T(q_i)\|_2} - \frac{\psi_I(I_i^c)}{\|\psi_I(I_i^c)\|_2}\right)^2$
- 训练500次迭代，学习率0.001，Adam优化器，50个epoch内收敛
- 仅反向传播通过最后一层，个性化过程在V100上不到1秒
- ConCon-Chi上 $\lambda=0.35$，DeepFashion2上 $\lambda=0.1$

## 实验关键数据

### 主实验（DeepFashion2，5张训练图）

| 方法 | 架构 | Context mRR | Context r@5 | Concept mRR | Concept mAP |
|------|------|------------|------------|------------|------------|
| PALAVRA | ViT-B/32 | 28.4 | 39.2 | - | - |
| SEARLE | ViT-B/32 | 21.90 | 27.15 | 25.97 | 12.74 |
| **POLAR（本文）** | ViT-B/32 | **34.82** | **44.88** | **59.26** | **28.75** |
| SEARLE | ViT-L/14 | 27.62 | 34.12 | 32.07 | 16.17 |
| **POLAR（本文）** | ViT-L/14 | **40.72** | **51.31** | **65.96** | **35.07** |

### 消融实验

| 配置 | Context mRR | Concept mAP | VLM cap r@10 | 说明 |
|------|------------|------------|-------------|------|
| 仅最后一层 Value (r=1) | **51.64** | **68.71** | 52.62 | 最优配置 |
| r=2 | 52.31 | 66.07 | 52.78 | 参数翻倍但无显著提升 |
| r=16 | 51.67 | 67.93 | 52.62 | 更多参数无益 |
| 所有层 | 43.23 | 63.77 | 52.45 | 早期层更新破坏通用知识 |
| 仅第1层 | 44.69 | 64.66 | 52.18 | 过早注入个性化信息效果差 |
| Q矩阵 | 16.65 | 10.91 | 51.84 | Q/K更新效果极差 |
| Prompt Tuning (1 tok) | 31.77 | 58.95 | **30.84** | 灾难性遗忘严重 |
| Textual Inversion | 42.45 | 64.71 | N/A | 上下文查询弱于POLAR |

### 关键发现

- **参数更新位置至关重要**：仅更新最后一层效果最佳，早期层更新会破坏编码过程中建立的通用表示
- **Value矩阵是最佳目标**：Q和K矩阵的更新几乎无效（mRR仅16%），Output和MLP也不如Value
- **VLM Caption指标揭示遗忘**：Prompt Tuning虽在概念检索上强，但VLM caption r@10从52.69暴降至30.84，说明通用知识严重遗忘；POLAR保持52.62基本不变
- **rank-1已足够**：增加rank不带来明显收益，反映了"学单个概念"任务的低复杂度

## 亮点与洞察

- **极简设计产生最佳效果**：rank-1、单层、单矩阵的"最小更新"策略反而优于更复杂的配置
- **正则化利用了LoRA结构**：将A和B的角色分离——A选择性激活、B控制更新幅度——是一个优雅的几何解释
- **新评估指标**：VLM Caption recall填补了评估通用知识保留的空白
- **个性化速度极快**：V100上<1秒完成，显著优于需要反向传播整个编码器的文本反转方法

## 局限与展望

- 仅在CLIP架构上验证，未测试更大的VLM（如LLaVA）
- $\lambda$ 需要在验证集上调整，不同数据集使用不同值
- 多概念合并可能在概念数量增多时出现干扰（rank累加可能饱和）
- 未探索图像编码器端的联合更新

## 相关工作与启发

- 受个性化图像生成（DreamBooth、Custom Diffusion）中参数微调策略的启发，但发现检索任务需要更保守的更新策略
- 与Perfusion（rank-1 U-net更新+key-locking）思路相关，但在判别式任务上需要不同的设计
- 验证了LoRA在极低rank（r=1）下的有效性，对其他个性化任务有参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将参数更新（而非文本反转）引入个性化检索，设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 消融全面（rank、层、参数类型、正则化），但仅两个数据集
- 写作质量: ⭐⭐⭐⭐⭐ 方法动机和设计选择解释清楚，消融逻辑严谨
- 价值: ⭐⭐⭐⭐ 4%~22%的提升显著，且方法极其轻量，工业可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Vision Graph Prompting via Semantic Low-Rank Decomposition](../../ICML2025/multimodal_vlm/vision_graph_prompting_via_semantic_low-rank_decomposition.md)
- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](../../ICCV2025/multimodal_vlm/sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)
- [\[CVPR 2025\] Dynamic Updates for Language Adaptation in Visual-Language Tracking](dynamic_updates_for_language_adaptation_in_visual-language_tracking.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](../../AAAI2026/multimodal_vlm/bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)

</div>

<!-- RELATED:END -->
