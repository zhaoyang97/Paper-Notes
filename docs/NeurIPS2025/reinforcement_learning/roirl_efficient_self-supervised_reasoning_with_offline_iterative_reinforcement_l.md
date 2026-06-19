---
title: >-
  [论文解读] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][离线强化学习] 提出RoiRL——一种基于离线迭代强化学习的轻量级自监督推理框架，通过加权对数似然目标函数替代在线RL（如TTRL），在不需要参考模型和真实标签的情况下实现LLM推理能力的自我提升，训练速度提高2.5倍且性能更优。 强化学习（RL）在提升大语言模型（LLM）推理能力方…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "离线强化学习"
  - "LLM推理"
  - "自监督学习"
  - "多数投票"
  - "加权似然"
---

# RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.02892](https://arxiv.org/abs/2510.02892)  
**代码**: 暂无  
**领域**: 强化学习  
**关键词**: 离线强化学习, LLM推理, 自监督学习, 多数投票, 加权似然

## 一句话总结

提出RoiRL——一种基于离线迭代强化学习的轻量级自监督推理框架，通过加权对数似然目标函数替代在线RL（如TTRL），在不需要参考模型和真实标签的情况下实现LLM推理能力的自我提升，训练速度提高2.5倍且性能更优。

## 研究背景与动机

强化学习（RL）在提升大语言模型（LLM）推理能力方面扮演核心角色，但传统RL方法依赖真实标签作为奖励信号，这在大规模场景下存在明显瓶颈。Test-Time Reinforcement Learning（TTRL）通过多数投票（majority vote）作为弱监督信号，消除了对真实标签的依赖，但面临两个关键问题：

**计算开销大**：TTRL需要在训练中维护一个参考模型并计算其logits，结合反复的CoT采样，GPU显存消耗快速饱和，难以扩展到更大模型

**在线RL不稳定**：基于GRPO的在线训练对超参数高度敏感，性能波动大，实际部署困难

核心问题是：**能否用一种像监督微调一样简单稳定的方法，达到与TTRL相同的优化目标？** RoiRL对此给出了肯定的回答。

## 方法详解

### 整体框架

RoiRL采用迭代式离线学习范式，每轮迭代包含两个步骤：**生成阶段**和**离线更新阶段**。在生成阶段，当前策略 $\pi_{m-1}$ 为每个问题采样 $k$ 个候选解并用多数投票打分；在离线更新阶段，基于加权对数似然目标优化策略参数，无需在线交互或参考模型。

### 关键设计

1. **多数投票奖励信号**：对每个问题 $x_i$，模型生成 $k$ 个候选答案 $\{y_i^\ell\}_{\ell \in [k]}$，通过多数投票确定伪标签 $\tilde{y}_i^k(\theta) = \text{maj}_{\ell \in [k]}(y_i^\ell)$，奖励函数为 $\tilde{r}_k(y, x_i, \theta) = \mathbb{1}[y = \tilde{y}_i^k(\theta)]$。这完全不依赖真实标签，利用模型自身的一致性信号。

2. **加权对数似然目标**：核心优化目标为：
$$\theta_m = \arg\max_{\theta} \sum_{i=1}^{n} \mathbb{E}_{(c,y) \sim \pi_{m-1}(\cdot|x_i)} \left[ g_m(\tilde{r}_k(y, x_i, \theta_{m-1})) \log \pi_\theta(c, y | x_i) \right]$$
其中 $g_m: \mathbb{R} \to \mathbb{R}$ 是递增的奖励变换函数。这本质上是对正确答案的加权监督微调，比在线RL更稳定。

3. **两种奖励变换实例**：

    - **恒等变换 $g_I(r) = r$**：退化为简单的正确答案SFT，只保留与多数投票一致的样本进行训练
    - **指数变换 $g_\beta(r) = \exp(r/\beta)$**：模拟KL正则化目标，与TTRL瞄准相同的理论最优解

### 损失函数 / 训练策略

**理论保证**：论文证明RoiRL的解析解为 $\pi_m(c,y|x) \propto \left(\prod_{j=1}^{m} g_j(\tilde{r}_k(y,x,\theta_{j-1}))\right) \pi_0(c,y|x)$，当选择 $g_j(r) = \exp(r/\beta)$ 时，该解与KL正则化RL目标的闭式解形式一致（Proposition 3.1）。

**关键优势**：
- 无需维护参考模型 $\pi_0$，大幅降低显存占用
- 离线更新类似SFT，训练稳定
- 迭代式设计处理非平稳奖励问题

## 实验关键数据

### 主实验

在MATH500 Train（无标签）上训练，评估三个基准的推理准确率：

| 模型 | 解码 | MATH500 Train | MATH500 Test | AMC | AIME |
|------|------|--------------|--------------|-----|------|
| Qwen2.5-Math-1.5B (Base) | maj₁ | 0.244 | 0.239 | 0.170 | 0.036 |
| TTRL | maj₁ | 0.307 | 0.298 | 0.214 | 0.026 |
| **RoiRL $g_I$** | maj₁ | **0.686** | 0.587 | 0.337 | **0.083** |
| **RoiRL $g_\beta$** | maj₁ | 0.670 | **0.604** | **0.340** | 0.070 |
| Phi4-mini-4B (Base) | maj₁ | 0.210 | 0.160 | 0.071 | 0.000 |
| **RoiRL $g_I$** | maj₁ | **0.660** | **0.511** | **0.246** | **0.016** |
| Llama-3.2-3B (Base) | maj₁₀ | 0.495 | 0.480 | 0.253 | 0.033 |
| TTRL | maj₁₀ | **0.510** | 0.490 | **0.313** | 0.167 |
| **RoiRL $g_I$** | maj₁₀ | 0.508 | 0.520 | **0.313** | **0.200** |

### 消融实验

| 配置 | 训练速度 | 参考模型 | 稳定性 | 说明 |
|------|---------|---------|--------|------|
| TTRL (GRPO在线) | 1× | 需要 | 差 | 超参敏感，显存饱和 |
| RoiRL $g_\beta$ | ~2× | 不需要 | 好 | 模拟KL正则化目标 |
| RoiRL $g_I$ | **~2.5×** | 不需要 | **最好** | 简单SFT式更新，性能最优 |

### 关键发现

- RoiRL $g_I$ 在大多数情况下取得最佳性能，说明**简单恒等变换可能比KL正则化更有效**
- RoiRL是真正的自改进方法而非多数投票蒸馏：训练后模型的 maj₁ 解码可超过基础模型的 maj₁₀，maj₁₀ 可超过基础模型的 maj₁₂₈
- 在三个不同架构的模型（Qwen、Phi4、Llama）上一致优于TTRL，验证了方法的鲁棒性

## 亮点与洞察

- 将在线RL问题转化为离线加权SFT，大幅降低了自监督推理的技术门槛
- 理论上证明了RoiRL与TTRL可瞄准相同最优策略，但实践中更简单的目标反而效果更好
- 非平稳奖励问题（多数投票随策略变化）在迭代框架下自然解决

## 局限与展望

- 仅在小规模模型（1.5B-4B）和有限计算预算下验证，需要在更大LLM上进一步评估
- 多数投票作为伪标签的质量上限受限于模型自身能力，可能导致错误强化
- 奖励变换函数 $g$ 的选择对性能有显著影响，但尚缺乏自适应选择机制
- 仅在数学推理任务上实验，泛化到代码生成等其他推理场景有待验证

## 相关工作与启发

- **TTRL**：开创性地使用多数投票替代真实标签进行RL训练，但在线方式成本高
- **GRPO/DPO**：KL正则化RL的经典方法，RoiRL证明可以用更简单的离线方式实现同等效果
- **离线RL (AWR/REPS)**：RoiRL的加权似然目标直接受到离线RL文献的启发
- **启发**：在LLM训练中，简单方法往往比复杂方法更有效——这与SFT优于PPO的实践经验一致

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将离线RL思想应用于自监督推理是自然但有效的创新
- **实验充分度**: ⭐⭐⭐⭐ 三种模型、多个基准、训练效率分析，但缺少大模型实验
- **写作质量**: ⭐⭐⭐⭐⭐ 理论推导清晰，动机到方法逻辑连贯
- **价值**: ⭐⭐⭐⭐ 为自监督LLM推理提供了更实用的训练范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)
- [\[ICLR 2026\] Attention as a Compass: Efficient Exploration for Process-Supervised RL in Reasoning Models](../../ICLR2026/reinforcement_learning/attention_as_a_compass_efficient_exploration_for_process-supervised_rl_in_reason.md)
- [\[ICCV 2025\] Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement](../../ICCV2025/reinforcement_learning/progressor_a_perceptually_guided_reward_estimator_with_self-supervised_online_re.md)
- [\[NeurIPS 2025\] TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning](trico_triadic_game-theoretic_co-training_for_robust_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
