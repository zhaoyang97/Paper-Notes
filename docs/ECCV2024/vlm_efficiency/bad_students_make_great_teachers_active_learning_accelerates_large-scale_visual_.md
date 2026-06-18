---
title: >-
  [论文解读] Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding
description: >-
  [ECCV 2024][多模态VLM][主动学习] 提出 ClassAct/ActiveCLIP 方法，利用小型廉价代理模型为数据点计算"可学习性"评分来优先选择训练数据，使大规模视觉分类器和多模态模型分别减少46%和51%的训练更新量，且总计算量节省高达25%，是首个在大规模预训练中实现计算正收益的主动学习方法。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "主动学习"
  - "数据选择"
  - "大规模预训练"
  - "CLIP"
  - "可学习性评分"
---

# Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding

**会议**: ECCV 2024  
**arXiv**: [2312.05328](https://arxiv.org/abs/2312.05328)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 主动学习, 数据选择, 大规模预训练, CLIP, 可学习性评分

## 一句话总结

提出 ClassAct/ActiveCLIP 方法，利用小型廉价代理模型为数据点计算"可学习性"评分来优先选择训练数据，使大规模视觉分类器和多模态模型分别减少46%和51%的训练更新量，且总计算量节省高达25%，是首个在大规模预训练中实现计算正收益的主动学习方法。

## 研究背景与动机

大规模视觉和语言模型的训练遵循幂律缩放定律——模型性能的增量改进需要数量级增长的计算量。这一规律的关键特征是训练数据是均匀采样的。主动数据选择通过优先在最相关的样本上进行训练来提高数据效率，但一直未被广泛采用，原因在于现有方法无法同时满足三个关键条件：

**跨模型和任务的通用性**：没有一种算法能在不同模型和任务上都有效

**大数据集的可扩展性**：许多方法在中等规模数据集（如ImageNet）上就已无法扩展

**计算正收益**：当考虑数据选择的开销时，总FLOPs是否真正节省

核心矛盾在于：基于模型的数据筛选方法虽然能显著提高学习效率，但筛选本身的计算开销往往与后续训练的节省相当甚至更多。本文的切入角度是：**用极小的代理模型来近似大模型的数据评分**，从而将筛选开销降到几乎可忽略的程度。

## 方法详解

### 整体框架

整体框架基于在线批量选择（Online Batch Selection）：首先从训练集均匀采样一个大批次（super-batch），用评分模型计算每个样本的分数，然后按分数进行非均匀采样得到小批次（sub-batch）用于训练。框架包含三个模型组件：
- **参考模型（Reference Model）**：预训练的小模型，提供基准损失
- **在线模型（Online Model）**：与参考模型同架构，随学习者并行训练
- **学习者模型（Learner Model）**：实际要训练的大模型

### 关键设计

1. **数据评分统计量的设计**:

    - 功能：定义如何为每个数据点计算优先级分数
    - 核心思路：比较了三种评分策略——
        - **难例优先**（hard）：$s^{hard}(x|\theta) = \ell(x|\theta)$，优先选loss高的样本
        - **易例优先**（easy）：$s^{easy}(x|\theta) = -\ell(x|\theta)$，优先选loss低的样本（去噪）
        - **可学习性**（learnability）：$s^{learn}(x|\theta^t, \theta^*) = \ell(x|\theta^t) - \ell(x|\theta^*)$，结合了两者优点
    - 设计动机：难例优先会引入噪声和不可学习样本；易例优先排除噪声但错过有价值的困难样本；可学习性评分选择那些"参考模型能做好但当前学习者做不好"的样本，即真正可以通过训练改善的样本

2. **跨尺度代理评分（ClassAct/ActiveCLIP 核心创新）**:

    - 功能：用极小的模型替代大学习者模型进行数据评分
    - 核心思路：引入第三个"在线模型"，与参考模型同架构同大小，替代学习者在可学习性公式中的角色。评分成本从 $F_{act} = F_{ref} + F_{learn}$ 降低到 $F_{act} = 2F_{ref}$
    - 设计动机：原始RHO-loss需要通过学习者模型做推理来评分，无法降低评分开销。实验发现可学习性评分对评分模型的缩小非常鲁棒——即使评分模型比学习者小1000倍（ViT-Mu vs ViT-L），仍能提供16%的加速
    - 关键发现：易例优先评分对模型缩小极敏感，而可学习性评分优雅退化

3. **多模态适配（ActiveCLIP/ActiveSigLIP）**:

    - 功能：将框架扩展到CLIP/SigLIP多模态预训练
    - 核心思路：学习者用对比损失训练，评分使用简化的图文点积相似度作为actor loss
    - 设计动机：对比损失计算开销大（需要全batch的softmax），用简单的点积相似度可以大幅降低评分成本

4. **在线参考模型训练**:

    - 功能：消除预训练参考模型的两步式流程
    - 核心思路：参考模型在super-batch（10倍大批次）上并行训练，学习率设为2倍
    - 设计动机：小模型可以在更大的batch上计算，能快速收敛到良好的评分策略

### 损失函数 / 训练策略

- 分类任务（ClassAct）：学习者和评分者都用标准交叉熵损失
- 多模态任务（ActiveCLIP）：学习者用对比损失，评分用图文embedding的点积
- 所有实验固定过滤比例为50%（$\rho=B/b=2$），每个super-batch中选一半
- 按Softmax概率进行非均匀采样：$\pi(x_i) = \text{Softmax}(\{s_i\})$

## 实验关键数据

### 主实验：大规模分类（JFT-300M）

| 配置 | 学习者加速 | 总计算加速 | 说明 |
|------|-----------|-----------|------|
| ViT-B评分→ViT-L（可学习性） | 31% | 计算负收益 | 评分模型太大 |
| ViT-S评分→ViT-L（可学习性） | 28% | ~20%正收益 | 计算正收益区间 |
| ViT-Ti评分→ViT-L（可学习性） | 26% | ~25%正收益 | 最佳总效率 |
| ViT-Mu评分→ViT-L（可学习性） | 16% | 正收益 | 1000x小仍有效 |
| ViT-S评分→ViT-L（易例优先） | <10% | 负收益 | 对缩小敏感 |

### 主实验：多模态预训练

| 方法 | 训练量 | IN-1K ZS Top-1 | COCO im2txt | COCO txt2im |
|------|-------|----------------|-------------|-------------|
| CLIP | 13B | 68.3 | 52.4 | 33.1 |
| OpenCLIP | 34B | 70.2 | 59.4 | 42.3 |
| ActiveCLIP | 3B | 71.3 | 57.7 | 43.0 |
| ActiveCLIP | 8B | 72.2 | 60.7 | 44.9 |
| SigLIP | 3B | 72.1 | 60.7 | 42.7 |
| ActiveSigLIP | 3B | 72.0 | 63.5 | 45.3 |

### 消融实验

| 配置 | 学习者加速 | 计算加速 | 说明 |
|------|-----------|---------|------|
| RHO（Tiny参考+B在线+B学习者） | 0% | -79% | 原始RHO失败 |
| ClassAct-HO（Tiny+Tiny+B，held-out） | 18% | 3% | 持有集参考 |
| ClassAct（Tiny+Tiny+B，in-domain） | 18% | 3% | 同域参考等效 |
| ClassAct-Online（Tiny+Tiny+B，在线） | 17% | 2% | 无需预训练参考 |

### 关键发现

- 可学习性评分对评分模型的缩放极其鲁棒，即使1000倍缩小仍有效；易例优先评分则非常敏感
- 数据选择策略可跨任务泛化：在高质量LTIP数据集上训练的参考模型，用于指导大规模ALIGN训练效果更好
- 缩放定律可推广到主动学习设置：不同计算预算下均能实现一致的加速
- 在线训练参考模型（无需预训练步骤）效果与预训练参考模型相当

## 亮点与洞察

- **首个计算正收益的大规模主动学习方法**：解决了领域长期存在的"数据筛选开销 >= 训练节省"的困境
- **"差学生造就好老师"的反直觉发现**：极小的模型（1000x小于学习者）虽然本身性能差，但能有效指导大模型的数据选择
- **Pareto前沿的发现**：揭示了评分计算开销与训练迭代节省之间的最优权衡曲面
- **与数据筛选方法互补**：ActiveCLIP与数据清洗（DataComp）和新训练目标（SigLIP）均互补，叠加后达到新SOTA

## 局限与展望

- 仅实验了50%过滤比例，更激进的过滤可能带来更大收益但也更高风险
- 仅验证了图像分类和多模态对比学习两个任务，未扩展到语言模型、视频、生成式建模
- 评分策略中的温度参数（Softmax）对最终效果的敏感度未深入分析
- 参考模型的训练数据质量对下游效果影响很大（LTIP>>ALIGN），如何自动选择最佳参考数据是开放问题

## 相关工作与启发

- **RHO-loss**（Mindermann et al., 2022）：提出可学习性概念但无法实现计算正收益，本文通过跨尺度代理解决
- **DoReMi**（Xie et al., 2023）：用小代理模型确定数据混合比例（用于语言模型），思路类似但粒度不同
- **DataComp**（Gadre et al., 2023）：静态数据筛选方法，与本文动态选择互补
- 启发：**小模型的"知识"可能比我们想象的更有价值**——不需要准确预测，只需要对数据质量有正确的排序即可

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在大规模预训练中实现计算正收益的主动学习，跨尺度代理思路巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ JFT-300M+ALIGN+Webli多数据集验证，缩放定律分析，丰富的消融
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，计算分析严谨，Pareto前沿图非常直观
- 价值: ⭐⭐⭐⭐⭐ 对大规模训练具有直接实用价值，能显著降低训练成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Active Data Curation Effectively Distills Large-Scale Multimodal Models](../../CVPR2025/multimodal_vlm/active_data_curation_effectively_distills_large-scale_multimodal_models.md)
- [\[ICML 2026\] Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](../../ICML2026/multimodal_vlm/bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)
- [\[ECCV 2024\] UniCode: Learning a Unified Codebook for Multimodal Large Language Models](unicode_learning_a_unified_codebook_for_multimodal_large_language_models.md)
- [\[ECCV 2024\] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [\[CVPR 2026\] Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset](../../CVPR2026/multimodal_vlm/towards_open-vocabulary_industrial_defect_understanding_with_a_large-scale_multi.md)

</div>

<!-- RELATED:END -->
