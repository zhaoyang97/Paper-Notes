---
title: >-
  [论文解读] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback
description: >-
  [AAAI 2026 Oral][LLM推理][Chain-of-Thought推理] 提出 InTRO 框架，通过将模型的生成策略与其answer-conditioned后验对齐（KL散度最小化），在单次前向传播中实现token级探索和自生成反馈，从而在不依赖外部监督的情况下提升LLM推理的准确性与简洁性。
tags:
  - "AAAI 2026 Oral"
  - "LLM推理"
  - "Chain-of-Thought推理"
  - "token级探索"
  - "自反馈"
  - "KL散度对齐"
  - "数学推理"
---

# In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.09865](https://arxiv.org/abs/2511.09865)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: Chain-of-Thought推理, token级探索, 自反馈, KL散度对齐, 数学推理

## 一句话总结

提出 InTRO 框架，通过将模型的生成策略与其answer-conditioned后验对齐（KL散度最小化），在单次前向传播中实现token级探索和自生成反馈，从而在不依赖外部监督的情况下提升LLM推理的准确性与简洁性。

## 研究背景与动机

当前训练LLM进行Chain-of-Thought (CoT)推理面临一个根本性的两难困境：

**SFT的局限性**：监督微调依赖单一"golden"推理路径，会惩罚同样正确的替代推理方式，导致泛化能力差。获取高质量的逐步人工标注也非常昂贵。

**RL方法的挑战**：以GRPO为代表的强化学习方法使用序列级稀疏奖励，面临"维度灾难"——有效推理序列空间随长度指数增长，使得信用分配极为困难，且计算开销巨大。

**过程监督的代价**：引入外部验证器模型或人工标注来评估单个推理步骤虽能缓解奖励稀疏性，但带来了标注成本高、计算开销大、验证器噪声等新问题。

本文提出一个核心问题：**模型能否在token级别自主探索，同时自生成反馈信号，完全摆脱对外部监督的依赖？**

InTRO给出了肯定回答。其核心洞察是：最优推理策略的生成分布 $\pi_\theta(z|x)$ 应当自然地重视正确且逻辑一致的推理路径——而这些路径恰好是模型在已知正确答案时会生成的内容。因此，可以用模型自身的answer-conditioned后验 $\pi_\theta(z|x,y)$ 作为理想的"教师"分布。

## 方法详解

### 整体框架

InTRO的训练流程分为三个阶段：
1. 策略 $\pi_\theta$ 对查询 $x$ 生成多条推理路径，保留答案正确的路径
2. 对每条保留路径的每个位置，采样 $n$ 个候选token，分别计算前向策略和answer-conditioned后验的概率，得到token级correction factor
3. 用加权梯度更新策略，强化与答案最相关的token

### 关键设计

#### 1. **从不可行优化到KL散度对齐**

最优推理的理想目标是最大化边际对数似然：

$$\mathcal{L}_{\text{marg.}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[-\log\sum_{z\in\mathcal{Z}_y}\pi_\theta(z,y|x)\right]$$

但直接优化该目标在计算上不可行（需遍历所有有效推理路径）。InTRO转而最小化前向KL散度：

$$\min_\theta D_{\text{KL}}(\pi_\theta(z|x,y) \| \pi_\theta(z|x))$$

选择前向KL的原因是它鼓励策略扩大支持集，更好地捕获多样的有效解。

**关键理论保证（Proposition 2.1）**：在 $y=f(z)$ 为确定性映射的假设下，边际对数似然的梯度与KL散度最小化的梯度完全等价：

$$\nabla_\theta \log \pi_\theta(y|x) = \mathbb{E}_{z\sim\pi_\theta(z|x,y)}[\nabla_\theta \log \pi_\theta(z|x)]$$

这意味着通过KL对齐，InTRO实际上在执行对不可行的边际似然目标的梯度上升。

#### 2. **Estimated Posterior与Correction Factor**

直接从 $\pi_\theta(z|x,y)$ 采样仍不可行，因此InTRO构造一个estimated posterior $\pi_\theta(\cdot|x \oplus y)$，即将问题和答案拼接后让模型生成推理过程。这利用了LLM强大的in-context learning能力——给定已知答案，模型生成的推理更倾向于正确路径。

通过重要性采样，核心训练目标变为：

$$\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\frac{1}{|z|\cdot n}\sum_{t=1}^{|z|}\sum_{i=1}^{n} w_t^i \cdot \log\pi_\theta(z_t^i|x, z_{<t})\right]$$

其中correction factor为：

$$w_{t,i} = \frac{\pi_\theta(z_t^i|x\oplus y, z_{<t})}{\pi_\theta(z_t^i|x, z_{<t})}$$

- $w_t^i > 1$：后验概率高于前向策略，该token对达到正确答案有正面贡献，应被强化
- $w_t^i < 1$：该token对正确答案贡献小，应被抑制
- 实践中将 $w_t^i$ 裁剪到 $[0, 200]$ 保证训练稳定性

#### 3. **Token级探索机制**

在每个时间步采样 $n$ 个候选token（包括原始token），直接鼓励token级别的探索。这与传统RL方法在序列级别探索有本质区别：

- **传统RL**（如GRPO）：在序列层面采样多条完整推理路径，依赖稀疏的结果奖励
- **InTRO**：在每个token位置进行细粒度探索，通过correction factor获得密集的自反馈信号

### 损失函数 / 训练策略

- 基于OpenRLHF框架实现，使用80GB A100 GPU
- 训练数据：MATH数据集难度3-5级，约9.2k样本
- 批大小128，学习率5e-7
- 每个查询生成4条候选推理路径（$G=4$），每步采样5个候选token（$n=5$）
- 二值奖励：正确answer=1.0，错误=0.0
- 不使用prompt模板（遵循Qwen系列最佳实践）

## 实验关键数据

### 主实验

| 模型 | MATH500 | Minerva | Olympiad | College | AMC23 | AIME25 | Avg | 提升(%) |
|------|---------|---------|----------|---------|-------|--------|-----|--------|
| Qwen2.5-1.5B base | 50.4 | 11.4 | 14.1 | 36.7 | 23.2 | 0.6 | 22.7 | - |
| GRPO | 50.6 | 9.6 | 17.9 | 37.0 | 24.5 | 1.1 | 23.5 | +3.5 |
| **InTRO** | **54.2** | **9.6** | **19.6** | **38.4** | **25.5** | **1.8** | **24.9** | **+9.7** |
| Qwen2.5-7B base | 66.4 | 15.1 | 30.4 | 42.4 | 41.9 | 5.0 | 33.5 | - |
| GRPO | 71.8 | 17.6 | 33.9 | 44.7 | 46.2 | 5.1 | 36.6 | +9.3 |
| **InTRO** | **72.6** | **19.9** | **35.3** | **45.0** | **47.0** | **5.6** | **37.6** | **+12.2** |
| Qwen3-4B base | 69.0 | 12.5 | 31.3 | 30.3 | 45.5 | 8.9 | 32.9 | - |
| GRPO | 73.8 | 15.8 | 34.4 | 34.2 | 50.9 | 7.9 | 36.2 | +10.0 |
| **InTRO** | **74.8** | **17.6** | **39.4** | **35.1** | **58.3** | **12.6** | **39.6** | **+20.4** |
| Qwen3-8B base | 65.8 | 11.8 | 34.7 | 29.8 | 53.4 | 10.0 | 34.3 | - |
| GRPO | 74.4 | 14.3 | 36.3 | 33.1 | 55.8 | 10.9 | 37.5 | +9.3 |
| **InTRO** | **75.2** | **18.8** | **38.7** | **35.2** | **56.7** | **12.4** | **39.5** | **+15.2** |

### 消融实验

**Token探索数量 $n$ 的影响（Qwen2.5-1.5B）**：

| 采样token数 $n$ | 1 | 2 | 5 | 10 | 20 | 40 |
|----------------|---|---|---|----|----|-----|
| 平均准确率 | 20.1 | 24.4 | 24.9 | 25.6 | 25.0 | 23.8 |

**OOD泛化（Qwen3-4B）**：

| 任务 | LiveCodeBench | BigCodeBench | GPQA | HumanEval | IFEval | Avg |
|------|--------------|-------------|------|-----------|--------|-----|
| Base | 4.1 | 27.7 | 22.1 | 81.7 | 45.6 | 36.2 |
| GRPO | 6.2 | 27.8 | 21.9 | 83.5 | 40.6 | 36.0 |
| **InTRO** | **22.6** | **35.4** | **38.6** | **89.2** | **50.4** | **47.2** |

### 关键发现

1. **规模效应**：更强的基础模型（如Qwen3系列）获得更大的提升，Qwen3-4B在数学推理上提升高达20.4%
2. **推理简洁性**：InTRO生成的推理路径显著更短，尤其在困难问题上，同时保持更高准确率
3. **跨领域泛化**：仅在数学数据上训练，但在代码、知识、指令跟随等任务上均获得显著提升
4. **Answer-conditioned推理增强**：在困难benchmark（AIME25、Olympiad）上，conditioning on answer大幅提升性能，但在简单任务上收益有限
5. **计算效率**：InTRO仅需两次前向传播（策略+后验），而GRPO需 $2G$ 次

## 亮点与洞察

- **理论优雅**：将不可行的边际似然优化等价转化为KL散度最小化，再通过estimated posterior实现实际可行的训练——每一步都有清晰的理论支撑
- **自反馈机制**：完全不依赖外部奖励模型或验证器，模型通过比较"有答案"和"无答案"时的生成概率来自我评估每个token的贡献
- **简洁性涌现**：InTRO并未显式优化推理长度，但correction factor自然抑制了冗余token，使推理更简洁
- **OOD泛化的解释**：token级信息差异最小化强化了因果链接，使得逻辑驱动的泛化成为可能

## 局限与展望

- estimated posterior $\pi_\theta(\cdot|x\oplus y)$ 与真实后验有偏差（实验测得KL约2.3），在弱模型上效果受限
- correction factor的裁剪范围 $[0, 200]$ 是经验值，缺乏理论指导
- 仅在数学推理上训练，虽然展示了OOD能力，但直接在多任务上训练可能效果更好
- 对基座模型的推理能力有较强依赖，Llama等较弱模型的提升不如Qwen显著

## 相关工作与启发

- 与LaTRO同样利用后验分布，但LaTRO用 $\log p_\theta(y|x\oplus z)$ 作奖励，InTRO则直接用后验概率比作correction factor
- 与GRPO的本质区别：GRPO在序列级探索+序列级奖励，InTRO在token级探索+token级反馈
- Correction factor的设计思路可能启发其他token级信号的构造方式

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — token级自反馈的理论框架非常有创意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多模型多基准+消融+OOD+效率分析
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，但符号较多
- **价值**: ⭐⭐⭐⭐⭐ — 为LLM推理训练提供了全新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] The Role of Feedback Alignment in Self-Distillation](../../ICML2026/llm_reasoning/the_role_of_feedback_alignment_in_self-distillation.md)
- [\[AAAI 2026\] SAPO: Self-Adaptive Process Optimization Makes Small Reasoners Stronger](sapo_self-adaptive_process_optimization_makes_small_reasoners_stronger.md)
- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](../../ICML2025/llm_reasoning/improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)
- [\[NeurIPS 2025\] Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning](../../NeurIPS2025/llm_reasoning/martingale_score_an_unsupervised_metric_for_bayesian_rationality_in_llm_reasonin.md)
- [\[AAAI 2026\] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)

</div>

<!-- RELATED:END -->
