---
title: >-
  [论文解读] Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models
description: >-
  [ECCV 2024][多模态VLM][持续学习] 提出选择性双教师知识迁移框架（SND），通过衡量预训练VLM和最近微调VLM之间的特征差异，在无标签参考数据集上自适应选择合适的教师进行知识蒸馏，同时缓解灾难性遗忘并保持零样本分类能力。 大规模视觉语言模型（如CLIP）展现了强大的零样本泛化能力，但在对序列下游任务进行适配…
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "持续学习"
  - "知识蒸馏"
  - "视觉语言模型"
  - "灾难性遗忘"
  - "零样本迁移"
---

# Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2403.09296](https://arxiv.org/abs/2403.09296)  
**代码**: [项目页面](https://chuyu.org/research/snd)  
**领域**: 多模态VLM  
**关键词**: 持续学习, 知识蒸馏, 视觉语言模型, 灾难性遗忘, 零样本迁移

## 一句话总结
提出选择性双教师知识迁移框架（SND），通过衡量预训练VLM和最近微调VLM之间的特征差异，在无标签参考数据集上自适应选择合适的教师进行知识蒸馏，同时缓解灾难性遗忘并保持零样本分类能力。

## 研究背景与动机
大规模视觉语言模型（如CLIP）展现了强大的零样本泛化能力，但在对序列下游任务进行适配时面临两大核心挑战：

**灾难性遗忘**：微调到新任务后，模型遗忘了先前任务的知识

**零样本退化**：微调过程会破坏预训练VLM固有的零样本迁移能力

现有方法各有局限：
- **排练型方法**（iCaRL等）：需要存储先前数据，有隐私泄露和存储负担
- **数据无关方法**（DF-CL等）：主要针对闭集分类，无法处理VLM的开放词汇特性
- **ZSCL**：仅从预训练模型蒸馏知识来保持零样本能力，但无法有效保留先前任务的知识，因为预训练模型从未在先前任务上微调过

本文的核心洞察：**需要同时利用两个教师** —— 最近微调的VLM用于保留先前任务知识，原始预训练VLM用于维护零样本能力。关键问题在于：对于参考数据集中的每张图片，如何判断应该从哪个教师蒸馏知识？

## 方法详解

### 整体框架
给定当前任务 $\mathcal{T}^k$、最近微调模型 $g_{k-1}$、预训练模型 $g_0$ 和无标签参考数据集 $\mathcal{X}^{ref}$（如ImageNet），框架执行以下流程：
1. 对参考数据集中的每张图片，计算双教师间的特征差异
2. 根据差异大小选择合适的教师进行知识蒸馏
3. 结合当前任务的交叉熵损失进行联合训练

### 关键设计

1. **双教师差异度量（Dual-Teacher Discrepancy）**:

    - 功能：通过衡量 $g_{k-1}$ 和 $g_0$ 对同一参考图片的特征差异来判断教师选择
    - 核心思路：如果参考图片与先前微调数据的分布相似，$g_{k-1}$ 的特征会显著偏离 $g_0$ 的特征（因为微调改变了表征），产生大的差异 $d$；如果图片不属于先前数据分布，两个教师的表征应该相似（差异小），因为 $g_{k-1}$ 对这类图片的表征未被微调改变
    - 形式化表达：$\mathbb{E}_{\mathbf{x} \in \mathcal{X}^{1:k-1}}[d(g_{k-1}(\mathbf{x}), g_0(\mathbf{x}))] \geq \mathbb{E}_{\mathbf{x}' \notin \mathcal{X}^{1:k-1}}[d(g_{k-1}(\mathbf{x}'), g_0(\mathbf{x}'))]$
    - 差异度量使用欧氏距离
    - 设计动机：无需访问先前任务数据，仅通过双教师间的特征差异就能推断参考图片是否与先前训练数据相关

2. **选择评分函数（Selection Scoring Function）**:

    - 功能：将双教师差异映射为[0,1]的选择分数
    - 核心公式：$\eta(\mathbf{x}) = \sigma\left(\frac{d(g_{k-1}(\mathbf{x}), g_0(\mathbf{x})) - \delta}{\gamma}\right)$
    - 其中 $\sigma$ 是sigmoid函数，$\delta$ 和 $\gamma$ 是归一化超参数
    - $\eta > 0.5$：倾向选择 $g_{k-1}$ 作为教师（缓解灾难性遗忘）
    - $\eta < 0.5$：倾向选择 $g_0$ 作为教师（保持零样本能力）
    - 设计动机：使用sigmoid函数实现平滑的教师选择，而非硬切换

3. **选择性双教师知识蒸馏（Selective Dual-Teacher KD）**:

    - 功能：根据选择分数加权蒸馏两个教师的知识
    - 蒸馏损失：$\mathcal{L}_{KD}^{dual} = \sum_{\mathbf{x} \sim \mathcal{X}^{ref}} \eta(\mathbf{x}) \cdot \mathcal{L}_{KD}^{k-1} + (1 - \eta(\mathbf{x})) \cdot \mathcal{L}_{KD}^{0}$
    - 其中 $\mathcal{L}_{KD}^{k-1} = d(g_{k-1}(\mathbf{x}), g_k(\mathbf{x}))$ 保留先前任务知识
    - $\mathcal{L}_{KD}^{0} = d(g_0(\mathbf{x}), g_k(\mathbf{x}))$ 保持零样本能力
    - 设计动机：不是简单地选择一个教师，而是通过 $\eta$ 分数进行软加权，使两个教师的贡献平滑过渡

### 损失函数 / 训练策略
- **总损失**：$\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{KD}^{dual}$
- 训练时只更新图像编码器，文本编码器保持冻结
- 使用AdamW优化器，cosine学习率调度，初始学习率 $1 \times 10^{-5}$
- 权重衰减 $5 \times 10^{-4}$
- 文本prompt固定为"a photo of a \<CLASS\>"

### 评估协议设计
本文提出了更严格的评估方案：
- **多训练序列**：构建K个循环左移的训练序列，确保每个数据集都有机会作为第一个和最后一个任务
- **三个指标**：平均准确率、灾难性遗忘（最大性能下降均值）、零样本退化（未见任务的最大性能下降均值）

## 实验关键数据

### 主实验：MTIL基准（8个序列平均）

| 方法 | 灾难性遗忘(↓) | 零样本退化(↓) | 平均准确率(↑) |
|------|--------------|--------------|--------------|
| Continual FT | 12.04 | 21.15 | 75.17 |
| LwF | 8.56 | 10.03 | 78.81 |
| iCaRL | 8.04 | 12.92 | 78.55 |
| ZSCL | 3.64 | 3.00 | 82.92 |
| MoE-Adapters | 2.71 | 2.17 | 82.29 |
| **Ours (SND)** | **1.20** | **1.96** | **84.92** |

### 消融实验：教师选择策略（序列S1）

| 配置 | 遗忘(↓) | 退化(↓) | 平均准确率(↑) |
|------|---------|---------|--------------|
| 仅从 $g_0$ 蒸馏 | 5.26 | 2.51 | 81.35 |
| 仅从 $g_{k-1}$ 蒸馏 | 2.63 | 3.36 | 83.61 |
| **双教师选择性蒸馏** | **1.70** | **1.55** | **84.48** |

### 关键发现
- **遗忘大幅降低**：SND的灾难性遗忘（1.20）仅为ZSCL（3.64）的1/3，比Continual FT（12.04）降低10倍
- **零样本保持优秀**：零样本退化（1.96）优于ZSCL（3.00），接近MoE-Adapters（2.17）但在准确率上高出2.63%
- **MCIL更具挑战性但依然领先**：在类增量学习设置中，SND的遗忘（1.35）和退化（1.65）均最优
- **经验验证双教师差异假设**：Aircraft数据集上 $g_1$ 和 $g_0$ 的差异为1.059，远高于其他未微调数据集（0.067-0.170），验证了选择机制的有效性
- **可视化验证**：高选择分数 $\eta$ 的参考图片确实与先前微调任务视觉相似

## 亮点与洞察
- **问题定义清晰**：将VLM持续学习分解为两个可量化的子目标（抗遗忘+保零样本），比以往工作更全面
- **选择机制直觉简洁**：利用双教师特征差异自动识别参考图片的归属，无需任何标签信息
- **评估更严格**：K个循环序列的评估方案比以往只选1-2个序列更可靠
- **无需额外参数**：不像MoE-Adapters需要新增adapter模块，SND仅通过蒸馏损失的加权实现，不增加模型参数

## 局限与展望
- **参考数据集依赖**：如果参考数据集与先前微调任务差异过大（如医学图像），$g_{k-1}$ 很少被选为教师，灾难性遗忘可能无法有效缓解
- **超参数 $\delta$ 和 $\gamma$**：选择评分函数的归一化参数需要针对具体场景调整
- **仅在图像编码器端操作**：未探索文本编码器的持续学习
- **扩展性**：随着任务数增加，$g_{k-1}$ 可能逐渐偏离 $g_0$ 过远，差异度量的可靠性可能下降
- 可考虑使用更大规模的参考数据集（如CC12M、LAION 5B的子集）

## 相关工作与启发
- **ZSCL**：最直接的先驱工作，仅从预训练模型蒸馏知识保持零样本能力，本文在其基础上增加了先前微调模型作为第二教师
- **MoE-Adapters**：通过增量adapter作为专家混合，是另一种抗遗忘策略，但需要额外的模块和测试时选择器
- **iCaRL/LwF**：经典持续学习方法，在VLM场景下效果有限
- 启发：在无数据场景下，模型本身的特征差异就蕴含了丰富的"数据归属"信息，可作为多种蒸馏策略的选择依据

## 评分
- 新颖性: ⭐⭐⭐⭐ 双教师差异驱动的选择性蒸馏思路清晰且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 8个训练序列、MTIL和MCIL两个基准、详尽分析和可视化
- 写作质量: ⭐⭐⭐⭐ 论文结构工整，动机和方法推导逻辑顺畅
- 价值: ⭐⭐⭐⭐ VLM持续学习是重要方向，该方法简洁有效，具有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](../../CVPR2025/multimodal_vlm/synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[CVPR 2026\] Learning from Itself: Mining Internal Knowledge from Vision Language Models for Continual Learning](../../CVPR2026/multimodal_vlm/learning_from_itself_mining_internal_knowledge_from_vision_language_models_for_c.md)
- [\[CVPR 2026\] Understanding Task Transfer in Vision-Language Models](../../CVPR2026/multimodal_vlm/understanding_task_transfer_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
