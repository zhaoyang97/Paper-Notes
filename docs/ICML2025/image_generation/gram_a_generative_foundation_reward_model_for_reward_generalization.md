---
title: >-
  [论文解读] GRAM: A Generative Foundation Reward Model for Reward Generalization
description: >-
  [ICML 2025][图像生成][奖励模型] GRAM 提出用生成式（而非判别式）方法训练奖励模型——先通过大规模无监督学习预训练生成式奖励模型，再用监督数据微调，并证明 label smoothing 实际上等价于正则化的 pairwise ranking 损失，实现了跨任务的奖励泛化。 领域现状：在 LLM 对齐中…
tags:
  - "ICML 2025"
  - "图像生成"
  - "奖励模型"
  - "生成式模型"
  - "基础模型"
  - "RLHF"
  - "泛化"
---

# GRAM: A Generative Foundation Reward Model for Reward Generalization

**会议**: ICML 2025  
**arXiv**: [2506.14175](https://arxiv.org/abs/2506.14175)  
**代码**: 无  
**领域**: Image Generation (LLM Alignment)  
**关键词**: 奖励模型, 生成式模型, 基础模型, RLHF, 泛化

## 一句话总结
GRAM 提出用生成式（而非判别式）方法训练奖励模型——先通过大规模无监督学习预训练生成式奖励模型，再用监督数据微调，并证明 label smoothing 实际上等价于正则化的 pairwise ranking 损失，实现了跨任务的奖励泛化。

## 研究背景与动机

**领域现状**：在 LLM 对齐中，奖励模型（RM）是 RLHF 的核心组件，用于指导模型生成符合人类偏好的输出。当前奖励模型通常以判别式方式训练——直接在人类偏好标签数据上学习打分函数。

**现有痛点**：判别式奖励模型严重依赖标注的人类偏好数据，泛化能力不足。当面对新任务或新分布的数据时，RM 性能显著下降。同时，高质量偏好数据获取成本高昂。

**核心矛盾**：标注偏好数据量有限 vs 奖励模型需要泛化到广泛任务。如何让 RM 像基础语言模型一样，通过大规模无标注数据获得泛化能力？

**本文目标**：构建一个"基础奖励模型"（foundation reward model），能在少量甚至零标注数据下迁移到多种任务。

**切入角度**：借鉴LLM的"预训练+微调"范式——先用大规模无监督数据训练生成式RM，再用少量偏好数据微调。

**核心 idea**：生成模型的对数似然天然就是一种奖励信号，通过预训练+微调，可以构建泛化能力极强的基础奖励模型。

## 方法详解

### 整体框架

- **第一阶段（无监督预训练）**：在大规模无标注文本上以生成方式预训练，学习语言的一般性质量分布
- **第二阶段（有监督微调）**：在标注的人类偏好数据上微调，将生成式能力适配为奖励打分
- **推理**：给定 prompt-response 对，生成式 RM 的对数似然（或其变体）作为奖励分数

### 关键设计

1. **生成式奖励模型（Generative RM）**:

    - 不同于传统 RM 在序列末尾加一个标量分类头，GRAM 直接用生成模型的对数似然 $\log p_\theta(y|x)$ 作为奖励
    - 关键洞察：高质量的回复在生成模型下应有更高的似然
    - **设计动机**：生成式模型可以利用大量无标注数据预训练，获得对语言质量的普遍理解

2. **Label Smoothing 与正则化 Pairwise Ranking 的等价性**:

    - 证明了当使用 label smoothing 训练时，生成式损失等价于一个正则化的 pairwise ranking 损失
    - 这意味着：$\mathcal{L}_{\text{smooth}} = (1-\epsilon)\mathcal{L}_{\text{CE}}(y_w) + \epsilon \mathcal{L}_{\text{CE}}(y_l)$ 在偏好训练下可以被理解为同时最大化好回复的似然并最小化坏回复的似然
    - **设计动机**：建立生成式模型和判别式模型在同一训练目标类下的统一视角

3. **基础奖励模型的迁移**:

    - 预训练后的生成式 RM 可以直接（零样本）或少量微调后应用于各种下游任务
    - 包括 response ranking、RLHF 训练信号、任务适配等
    - **设计动机**：类似基础语言模型的零样本和少样本迁移能力

### 损失函数 / 训练策略

- **预训练**：标准的自回归语言模型损失 $\mathcal{L} = -\sum_t \log p_\theta(y_t | y_{<t}, x)$
- **微调**：带 label smoothing 的偏好损失，等价于正则化的 pairwise ranking
- 训练策略支持两种模式：
    - 直接在偏好数据上训练（label smoothed CE）
    - 冻结生成模型，仅训练轻量适配层

## 实验关键数据

### 主实验

| 任务 | 指标 | GRAM | 判别式基线 | 提升 |
|------|------|------|-----------|------|
| 偏好排序 (RewardBench) | Accuracy↑ | 显著提升 | 标准 BT RM | 多个百分点 |
| RLHF 训练 | Win Rate↑ | 更高 | 标准 RM | 显著 |
| 任务适配（少样本） | Accuracy↑ | 更好 | 直接微调 RM | 明显改善 |
| 零样本迁移 | Accuracy↑ | 可用 | 需要训练数据 | 无需标注 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无预训练 | 明显下降 | 无监督预训练是泛化的关键 |
| 无 label smoothing | 下降 | 正则化效果对稳定训练重要 |
| 纯判别式 RM | 泛化差 | 在分布外数据上衰退严重 |
| 不同预训练规模 | 随规模提升 | 更大模型 = 更好泛化 |

### 关键发现

- 生成式 RM 在跨任务泛化方面显著优于判别式 RM
- Label smoothing 不只是正则化技巧，它与 pairwise ranking 有深层数学联系
- 预训练的质量和规模直接决定了基础 RM 的泛化能力
- GRAM 在 RLHF 训练和直接排序任务上都取得了优于多个强基线的表现

## 亮点与洞察

1. **理论贡献**：label smoothing = 正则化 pairwise ranking 的等价性证明，统一了生成式和判别式训练视角
2. **范式创新**：将"预训练+微调"范式从语言模型扩展到奖励模型
3. **实用价值**：基础 RM 可以跨任务复用，大幅降低每个新任务的标注成本
4. **元学习视角**：生成式预训练可以看作隐式地学习了"什么是好的文本"的元知识

## 局限与展望

1. 生成式 RM 的推理成本高于单标量输出的判别式 RM
2. 对数似然作为奖励可能在某些任务上与人类偏好不完全一致（如创意写作）
3. 预训练数据的领域偏向可能影响 RM 在特定专业领域的泛化
4. 大规模预训练的计算成本

## 相关工作与启发

- 与 DPO 的联系：DPO 也用似然比作为隐式奖励，GRAM 进一步将此泛化为基础模型
- 与 GPT-4 as judge 的对比：GRAM 提供了一种参数化的、可训练的替代方案
- 启发：生成式和判别式模型之间的等价性可能在更多场景中被利用

## 评分
- 新颖性: ⭐⭐⭐⭐ 生成式基础 RM 的思路和理论联系有新意
- 实验充分度: ⭐⭐⭐⭐ 多任务验证
- 写作质量: ⭐⭐⭐⭐ 理论与实验结合好
- 价值: ⭐⭐⭐⭐⭐ 为 RM 训练提供了新范式，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/image_generation/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[ICML 2025\] Towards a Mechanistic Explanation of Diffusion Model Generalization](towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [\[CVPR 2025\] Visual-ERM: Reward Modeling for Visual Equivalence](../../CVPR2025/image_generation/visual-erm_reward_modeling_for_visual_equivalence.md)
- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](../../ICLR2026/image_generation/editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)

</div>

<!-- RELATED:END -->
