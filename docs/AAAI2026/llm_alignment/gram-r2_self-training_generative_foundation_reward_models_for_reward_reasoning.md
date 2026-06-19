---
title: >-
  [论文解读] GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning
description: >-
  [AAAI 2026][LLM对齐][奖励模型] 本文提出 GRAM-R²，一个通过自训练方式在无标签数据上引发奖励推理能力的生成式基础奖励模型，能够同时产生偏好标签和推理理由，在响应排序、任务适配和 RLHF 等多个下游任务中一致超越判别式和生成式基线。 领域现状：奖励建模（reward modeling）在 LLM 对齐…
tags:
  - "AAAI 2026"
  - "LLM对齐"
  - "奖励模型"
  - "自训练"
  - "生成式奖励"
  - "偏好推理"
  - "RLHF"
---

# GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning

**会议**: AAAI 2026  
**arXiv**: [2509.02492](https://arxiv.org/abs/2509.02492)  
**代码**: 无  
**领域**: LLM对齐  
**关键词**: 奖励模型, 自训练, 生成式奖励, 偏好推理, RLHF

## 一句话总结

本文提出 GRAM-R²，一个通过自训练方式在无标签数据上引发奖励推理能力的生成式基础奖励模型，能够同时产生偏好标签和推理理由，在响应排序、任务适配和 RLHF 等多个下游任务中一致超越判别式和生成式基线。

## 研究背景与动机

**领域现状**：奖励建模（reward modeling）在 LLM 对齐中至关重要，近年来从任务特定设计走向通用主义奖励模型的趋势明显。好的奖励模型需要能跨任务、跨域地评估 LLM 输出的质量，并且最好能解释自己的判断理由。

**现有痛点**：（1）开发有效奖励模型的根本挑战在于对大规模标注偏好数据的严重依赖——人类偏好标注成本极高且难以规模化；（2）预训练可以利用丰富的无标签数据，但现有预训练方法无法为奖励模型注入显式的推理能力——模型给出偏好判断但无法解释为什么；（3）判别式奖励模型只输出分数，缺乏可解释性；生成式奖励模型虽然可以输出理由，但需要大量带推理标注的训练数据。

**核心矛盾**：标注数据稀缺 vs 推理能力需求——既要从少量（甚至无）标注数据中学习，又要模型能够产生有推理链支持的偏好判断。

**本文目标**：（1）利用无标签数据通过自训练引发奖励模型的推理能力；（2）构建可作为基础模型使用的生成式奖励模型；（3）最小化或消除对标注偏好数据的依赖。

**切入角度**：作者利用自训练（self-training）范式——让模型自身生成伪标签和推理理由，然后在这些自生成数据上继续训练，形成迭代增强的循环。

**核心 idea**：通过自训练让生成式奖励模型在无标签数据上自举（bootstrap），同时生成偏好标签和奖励理由（reward rationale），形成具有推理能力的基础奖励模型。

## 方法详解

### 整体框架

GRAM-R² 的训练 pipeline：（1）从预训练 LLM 出发，少量种子偏好数据初始化奖励推理能力；（2）在大量无标签数据上，模型自身生成偏好标签和推理理由；（3）筛选高质量自生成数据用于进一步训练；（4）多轮迭代逐步增强推理能力。训练完成后，GRAM-R² 可直接用于多种下游任务（响应排序、奖励微调、RLHF）而无需或仅需少量额外适配。

### 关键设计

1. **自训练引发推理（Self-Training for Reward Reasoning）**:

    - 功能：从无标签数据中自举获取奖励推理能力。
    - 核心思路：给定一对响应（response A vs response B），模型生成：（a）偏好标签（A 更好/B 更好/相当）；（b）推理理由（rationale，解释为什么做出此判断）。自训练过程中，使用当前模型对未标注数据进行推理，筛选高置信度的结果作为伪训练数据，然后在这些数据上进一步训练模型。关键在于理由和标签同时生成并同时用于训练。
    - 设计动机：纯标签的自训练容易陷入确认偏差（模型强化自己的错误）。同时要求生成理由增加了自监督信号的丰富度——不仅要判断对错，还要说明原因，这迫使模型进行更深层的思考。

2. **质量筛选与课程策略**:

    - 功能：确保自生成的伪训练数据质量，防止噪声累积。
    - 核心思路：对自生成的（标签，理由）对进行多维质量筛选：（a）标签置信度过滤——只保留模型非常确定的判断；（b）一致性检查——同一对响应多次采样，保留判断一致的；（c）理由质量评估——检查理由的逻辑连贯性和与结论的一致性。训练采用课程策略，早期轮次使用更严格的筛选以避免早期噪声累积。
    - 设计动机：自训练的核心风险是"垃圾进垃圾出"——如果伪标签质量差，训练会越跑越偏。多重筛选和课程策略是关键的质量保障。

3. **基础模型架构与多任务适配**:

    - 功能：使模型成为可泛化的奖励基础模型。
    - 核心思路：GRAM-R² 基于预训练 LLM 构建，以生成式方式（而非回归分数的方式）输出偏好判断和理由。这种生成式设计使得模型可以自然地适配多种下游格式——排序任务只需比较多个响应的偏好输出，评分任务可以从理由中提取分数，RLHF 可以将偏好判断转化为奖励信号。
    - 设计动机：判别式奖励模型（输出标量分数）灵活性有限，难以提供推理理由。生成式设计统一了判断和解释，且天然适合多任务使用。

### 损失函数 / 训练策略

训练使用标准的自回归语言建模损失：$\mathcal{L} = -\sum_t \log p_\theta(y_t | y_{<t}, x)$，其中 $y$ 是（理由 + 标签）的序列，$x$ 是输入的响应对。自训练通过多轮迭代进行，每轮用当前模型生成伪数据并筛选后加入训练集。

## 实验关键数据

### 主实验

在三个任务上评估：响应排序、任务适配、RLHF。

| 任务 | 指标 | GRAM-R² | 判别式基线 | 生成式基线 | 说明 |
|------|------|---------|-----------|-----------|------|
| 响应排序 | 排序准确率 | 最佳 | 中等 | 次优 | 一致超越 |
| 任务适配 | 适配后准确率 | 最佳 | -- | 次优 | 少样本适配 |
| RLHF | 下游LLM质量 | 最佳 | 中等 | 次优 | 更好的对齐效果 |

### 消融实验

| 配置 | 性能 | 说明 |
|------|------|------|
| 完整自训练 | 最佳 | 理由+标签联合自训练 |
| 仅标签自训练 | 下降 | 缺少推理理由的监督信号 |
| 无质量筛选 | 显著下降 | 噪声伪标签累积 |
| 单轮自训练 | 不如多轮 | 多轮迭代逐步提升 |
| 无课程策略 | 下降 | 早期噪声污染严重 |

### 关键发现

- 自训练+推理理由生成的组合是 GRAM-R² 成功的核心——仅自训练标签效果有限，同时生成理由显著提升了推理能力。
- 质量筛选至关重要——无筛选的自训练会退化。
- 作为基础奖励模型，GRAM-R² 在零样本或少样本设置下跨任务表现出色，证明了基础模型范式的有效性。
- 生成式设计相比判别式在可解释性和灵活性上有明显优势。

## 亮点与洞察

- **"奖励推理"的概念化**很有前瞻性——不仅给出偏好判断，还要解释判断依据。这对构建可信赖的 AI 对齐系统至关重要。
- **自训练引发推理**的方法论可以推广到其他需要推理能力的任务——不依赖大量标注推理链数据就能学习推理。
- 作为基础模型的设计使得 GRAM-R² 可以被广泛复用，降低了奖励建模的门槛。

## 局限与展望

- 自训练的起始质量取决于种子数据的选择和初始模型的能力。
- 多轮自训练的计算成本较高——每轮都需要在大量无标签数据上进行推理。
- 自生成理由的真实质量难以客观评估——是否真的"在推理"还是"在编造合理化"？
- 可以与 Constitutional AI 结合，用明确的准则来约束推理理由的内容。

## 相关工作与启发

- **vs 传统判别式奖励模型（如 Bradley-Terry）**: 判别式模型只输出分数无法解释，GRAM-R² 提供了推理理由，更加透明。
- **vs LLM-as-Judge**: LLM-as-Judge 使用通用 LLM 做评判，但缺乏专门的奖励推理训练。GRAM-R² 专门训练了推理能力。
- **vs 自训练方法（如 Self-Play）**: 自训练在策略优化中常见，本文将其创新性地应用于奖励模型的推理能力引发。

## 评分

- 新颖性: ⭐⭐⭐⭐ 自训练+奖励推理的组合方案新颖，基础奖励模型的概念有前瞻性
- 实验充分度: ⭐⭐⭐⭐ 三个任务维度的全面评估+消融实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法阐述逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 对RLHF和AI对齐领域有重要的方法论贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](../../ACL2026/llm_alignment/consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)
- [\[CVPR 2026\] Thinking with Frames: Generative Video Distortion Evaluation via Frame Reward Model](../../CVPR2026/llm_alignment/thinking_with_frames_generative_video_distortion_evaluation_via_frame_reward_mod.md)
- [\[AAAI 2026\] LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward](laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)
- [\[AAAI 2026\] DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF](decorl_decoupling_reasoning_chains_via_parallel_sub-step_gen.md)
- [\[CVPR 2026\] Unlocking Token Rewards via Training-Free Reward Attribution](../../CVPR2026/llm_alignment/unlocking_token_rewards_via_training-free_reward_attribution.md)

</div>

<!-- RELATED:END -->
