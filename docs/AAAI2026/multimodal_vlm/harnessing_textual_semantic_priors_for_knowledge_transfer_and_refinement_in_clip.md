---
title: >-
  [论文解读] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning
description: >-
  [AAAI 2026][多模态VLM][持续学习] 本文提出SECA框架，利用CLIP文本分支的稳定语义先验来指导骨干网络中语义相关的历史知识迁移（SG-AKT模块），并通过文本嵌入的类间语义关系精炼视觉原型构建混合分类器（SE-VPR模块），在ImageNetR/A和CIFAR100上超越现有SOTA。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "持续学习"
  - "CLIP"
  - "文本语义先验"
  - "知识蒸馏"
  - "模态差距"
---

# Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning

**会议**: AAAI 2026  
**arXiv**: [2508.01579](https://arxiv.org/abs/2508.01579)  
**代码**: [https://github.com/HHHLF/SECA_master](https://github.com/HHHLF/SECA_master)  
**领域**: 多模态VLM  
**关键词**: 持续学习, CLIP, 文本语义先验, 知识蒸馏, 模态差距

## 一句话总结
本文提出SECA框架，利用CLIP文本分支的稳定语义先验来指导骨干网络中语义相关的历史知识迁移（SG-AKT模块），并通过文本嵌入的类间语义关系精炼视觉原型构建混合分类器（SE-VPR模块），在ImageNetR/A和CIFAR100上超越现有SOTA。

## 研究背景与动机

### 领域现状
持续学习 (Continual Learning, CL) 的核心挑战是**稳定性-可塑性困境**——模型在学习新类别时必须保持已学知识（稳定性）同时适应新信息（可塑性）。随着CLIP等视觉语言模型的兴起，其强大的零样本能力使其成为持续学习的理想骨干，PEFT-based方法（如prompt tuning、adapter）在此方向取得了显著进展。

### 现有痛点
现有CLIP-based持续学习方法存在两个核心问题：

**骨干训练中的非选择性知识迁移**：大多数方法（如EwC、AFC）使用正则化或蒸馏来强制与最近任务的模型保持一致，但不区分知识的语义相关性。当学习"猫"类时，来自"狗"类的知识有利于迁移，而来自"车辆"类的知识则会造成干扰（语义干扰）

**分类器的模态差距**：纯文本分类器（基于CLIP文本编码器）泛化性强但可塑性不足；视觉分类器（基于原型）可以弥合模态差距，但粗糙的视觉原型缺乏丰富和精确的语义信息

### 核心矛盾
CLIP的文本分支在持续学习过程中保持一致的语义表示（因为冻结的文本编码器不随任务变化），但这一宝贵的**稳定语义先验**在现有方法中未被充分利用——既没有用来指导选择性知识迁移，也没有用来增强视觉分类器的语义结构。

### 切入角度
将CLIP文本分支的"抗遗忘"和"结构化"特性作为统一的指导信号：
- 在骨干端：用文本线索评估新图像与历史视觉知识的相关性，实现实例自适应的选择性蒸馏
- 在分类器端：用文本嵌入的类间关系精炼视觉原型，桥接模态差距

## 方法详解

### 整体框架
SECA基于CLIP构建，包含冻结的视觉和文本编码器、可学习的文本prompt和视觉adapter，以及两个核心模块：
1. **SG-AKT**：语义引导的自适应知识迁移
2. **SE-VPR**：语义增强的视觉原型精炼

### 关键设计

#### 1. **混合PEFT基线 (H-PEFT)**
- **文本侧**：为每个任务 $s$ 引入可学习prompt $\mathbf{P}^s \in \mathbb{R}^{M \times d_T}$，与类名拼接后输入冻结文本编码器
- **视觉侧**：在视觉编码器的每层transformer块中插入共享adapter $\mathcal{A}_l$
- **分类概率**：通过视觉特征和文本特征的余弦相似度计算，使用温度因子 $\tau$
- 基线训练损失：标准交叉熵 $\mathcal{L}_{ce-T}$

#### 2. **语义引导的自适应知识迁移 (SG-AKT)**
- **核心思路**：维护一个历史adapter池 $\mathcal{P} = \{\mathcal{A}^1, ..., \mathcal{A}^{|\mathcal{P}|}\}$，每个adapter编码了对应任务的视觉知识。对新图像，根据其文本语义向量评估与各adapter的相关性，加权聚合相关知识作为蒸馏信号

- **步骤一：提取多视角知识**
    - 视觉知识：将新图像通过所有历史adapter，得到 $\mathbf{V}_x = [\mathbf{V}_x^{(1)}, ..., \mathbf{V}_x^{(|\mathcal{P}|)}]$
    - 语义向量：用类标签与所有（历史+当前）task prompt组合，得到 $\mathbf{S}_y = [\mathbf{S}_y^{(1)}, ..., \mathbf{S}_y^{(s)}]$

- **步骤二：计算相关性得分**
  $$\alpha_x^{(p)} = \frac{1}{s} \sum_{i=1}^s [\phi(\mathbf{S}_y^{(i)})\mathbf{W}_S]^\top [\phi(\mathbf{V}_x^{(p)})\mathbf{W}_V]$$
  - 使用可学习投影器 $\mathbf{W}_S, \mathbf{W}_V$ 将文本和视觉映射到共享语义空间
  - LayerNorm $\phi(\cdot)$ 稳定训练

- **步骤三：自适应聚合与蒸馏**
  $$\mathbf{V}_x^{agg} = \sum_{p=1}^{|\mathcal{P}|} \frac{\exp(\lambda \alpha_x^{(p)})}{\sum_i \exp(\lambda \alpha_x^{(i)})} \mathbf{V}_x^{(p)}$$
  - 聚合特征作为teacher信号，通过KL散度蒸馏到当前模型
  - **设计动机**：相比无差别蒸馏（Vanilla-KD）或均匀聚合（Avg-KD），基于语义相关性的自适应聚合可以优先传递有用知识、抑制干扰知识

- **Adapter池管理**：固定池大小 $|\mathcal{P}|=5$，基于效用得分 $U^p$ 进行动量更新和裁剪——效用最高的adapter表示其知识已充分迁移到最新模型，可以安全移除

#### 3. **语义增强的视觉原型精炼 (SE-VPR)**
- **核心思路**：利用文本嵌入的类间语义关系来修正粗糙的CLIP视觉原型

- **步骤一：计算类间亲和矩阵**
  $$\mathbf{M}_{k,j} = \exp(-\gamma \|\phi(\mathbf{Z}_k)\mathbf{H}_{proj} - \phi(\mathbf{Z}_j)\mathbf{H}_{proj}\|_2^2)$$
  - 使用可学习投影器 $\mathbf{H}_{proj}$ 将类文本嵌入映射到更具表达力的潜空间
  - $\gamma$ 为缩放因子

- **步骤二：精炼视觉原型**
  $$\hat{\mathbf{c}}_{V,k} = \sum_{j \in \mathcal{Y}^{1:s}} \frac{\mathbf{M}_{k,j}}{\sum_i \mathbf{M}_{k,i}} \mathbf{c}_{V,k}$$
  - 原始原型 $\mathbf{c}_{V,k}$ 通过亲和矩阵加权得到精炼原型

- **原型一致性正则化**：防止新任务训练导致旧类原型漂移
  $$\mathcal{L}_{reg} = \frac{1}{|\mathcal{Y}^{1:s-1}|} \sum_{k \in \mathcal{Y}^{1:s-1}} \|\hat{\mathbf{c}}_{V,k}^s - \hat{\mathbf{c}}_{V,k}^{s-1}\|_2^2$$

- **设计动机**：单独的文本分类器因模态差距限制了可塑性；精炼后的视觉原型既保留了视觉侧的匹配优势，又继承了文本侧的语义结构

### 训练目标与推理策略
- **总损失**：$\mathcal{L} = \mathcal{L}_{ce-T} + \underbrace{\mathcal{L}_{agg} + \beta \mathcal{L}_{SG-AKT}}_{SG-AKT} + \underbrace{\mathcal{L}_{ce-V} + \mathcal{L}_{reg}}_{SE-VPR}$
- **推理**：混合分类——精炼视觉原型预测 + 所有任务文本分类器预测的平均

## 实验关键数据

### 主实验（ImageNetR & ImageNetA，CLIP ViT-B/16）

| 方法 | ER/FR | 10S-ImageNetR Last↑ | 10S-ImageNetR Avg↑ | 10S-ImageNetA Last↑ | 10S-ImageNetA Avg↑ |
|------|-------|--------------------|--------------------|--------------------|--------------------|
| ZS-CLIP | ✗ | 74.93 | 81.56 | 47.33 | 58.35 |
| VPT-NSP | ✓ | 82.48 | 87.94 | 61.42 | 71.76 |
| RAPF | ✓ | 79.62 | 86.28 | 55.37 | 67.32 |
| CLAP | ✓ | 79.98 | 85.77 | 58.66 | 69.35 |
| **SECA (本文, 无replay)** | **✗** | **83.18** | **88.58** | **65.09** | **74.45** |
| **SECA++ (本文, 有replay)** | **✓** | **83.41** | **88.75** | **65.77** | **74.65** |

### 消融实验

| 模块组合 | 10S-ImageNetA Last↑ | 10S-CIFAR100 Last↑ | 10S-ImageNetR Last↑ |
|----------|--------------------|--------------------|---------------------|
| ZS-CLIP (基线) | 47.33 | 67.19 | 74.93 |
| +H-PEFT | 55.78 | 73.97 | 80.57 |
| +H-PEFT+SG-AKT | 57.91 (+2.13) | 75.93 (+1.96) | 81.36 (+0.79) |
| +H-PEFT+SE-VPR | 62.62 (+6.84) | 77.13 (+3.16) | 81.30 (+0.73) |
| +H-PEFT+SG-AKT+SE-VPR (**SECA**) | **65.09** (+9.31) | **79.79** (+5.82) | **83.18** (+2.61) |

### 蒸馏策略对比（10S-ImageNetA，带SE-VPR）

| 蒸馏策略 | Last↑ | Avg↑ |
|----------|------|------|
| Sequential (无蒸馏) | 62.62 | 73.15 |
| CLIP-KD (全局教师) | 62.39 | 73.01 |
| Vanilla (最近任务蒸馏) | 63.51 | 73.89 |
| Avg-KD (均匀聚合) | 64.32 | 74.08 |
| **SG-AKT (本文)** | **65.09** | **74.45** |

### 分类器设计对比（10S-ImageNetA）

| 分类器 | Last↑ | Avg↑ |
|--------|------|------|
| Only Text | 57.91 | 68.56 |
| Centroid (CLIP) | 58.55 | 68.30 |
| Centroid (Adapted) | 63.16 | 72.89 |
| Linear | 51.07 | 62.76 |
| **SE-VPR (本文)** | **65.09** | **74.45** |

### 关键发现
1. **无replay版SECA已超越有replay的SOTA**：SECA在10S-ImageNetA上Last acc比VPT-NSP高3.67%，且不需要任何replay
2. **SE-VPR贡献最大**：在ImageNetA上，SE-VPR单独贡献+6.84%，远超SG-AKT的+2.13%
3. **两模块互补**：同时使用时SECA总提升+9.31%，远超单独使用之和
4. **语义引导蒸馏优于所有变体**：SG-AKT比Vanilla-KD高1.58%，比Avg-KD高0.77%
5. **Adapter池大小5即饱和**：|P|≥5时性能接近使用所有历史adapter的"ALL"方案
6. **线性分类器表现最差**：说明简单线性层无法有效利用adapted特征进行跨任务分类

## 亮点与洞察
- **文本先验的系统性挖掘**：首次从知识迁移和原型精炼两个互补角度同时利用CLIP文本语义
- **实例自适应蒸馏**：不同图像根据其语义相关性获得不同的知识聚合权重，比"一刀切"蒸馏更精细
- **效用得分池管理**：用动量更新跟踪adapter被利用的程度，效用最高者意味着知识已充分迁移，可以安全裁剪——一个优雅的满容量策略
- **模态差距的桥接方案**：不是简单地融合文本和视觉特征，而是用文本的关系结构来"雕刻"视觉原型
- **无replay即超SOTA**：说明语义引导的选择性迁移可以替代传统的样本/特征回放策略

## 局限与展望
- 仅在分类任务上验证，未扩展到检测、分割等更复杂的持续学习场景
- adapter池固定大小为5，对于任务数量极多（如100+任务）的场景可能不够
- 依赖于CLIP的视觉-文本对齐质量，对于CLIP对齐较弱的领域（如医学）效果未知
- 推理时需要所有历史task prompt和adapter池，存储和计算开销随任务增长
- SE-VPR使用原始CLIP编码器提取原型（非adapted），可能丢失task-specific的有用信息
- $\beta$ 作为任务依赖的超参数，其增长策略需要针对不同数据集调整

## 相关工作与启发
- L2P (Wang et al., CVPR 2022)、CODA (Smith et al., CVPR 2023)：prompt-based持续学习，但未利用CLIP文本语义
- RAPF (ECCV 2024)：轻量投影器+伪特征replay，SECA在无replay下已超越其replay版
- VPT-NSP (NeurIPS 2024)：展示了CLIP的强持续学习潜力，SECA进一步从文本语义角度挖掘
- PROOF (T-PAMI 2025)：跨注意力模块+权重平均策略，但分类器仍限于纯文本
- 启发：在多模态预训练模型的下游适应中，不同模态各具优势——文本的稳定语义可以指导视觉的选择性适应

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ （文本先验指导知识迁移+原型精炼的双模块设计非常有洞察力）
- 实验充分度: ⭐⭐⭐⭐⭐ （多基准+多消融+多对比策略+超参分析，极为充分）
- 写作质量: ⭐⭐⭐⭐ （公式推导完整，但符号较多需要反复对照）
- 价值: ⭐⭐⭐⭐⭐ （为CLIP-based持续学习提供了新范式，无replay超SOTA有重要意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models](../../ECCV2024/multimodal_vlm/select_and_distill_selective_dual-teacher_knowledge_transfer_for_continual_learn.md)
- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [\[ICML 2025\] LADA: Scalable Label-Specific CLIP Adapter for Continual Learning](../../ICML2025/multimodal_vlm/lada_scalable_label-specific_clip_adapter_for_continual_learning.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)
- [\[CVPR 2026\] Re-evaluating Continual VQA: Toward Fair and Robust Evaluation for Multimodal Continual Learning](../../CVPR2026/multimodal_vlm/re-evaluating_continual_vqa_toward_fair_and_robust_evaluation_for_multimodal_con.md)

</div>

<!-- RELATED:END -->
