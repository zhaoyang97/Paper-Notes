---
title: >-
  [论文解读] Training a Generally Curious Agent (Paprika)
description: >-
  [ICML 2025][模型压缩][in-context RL] 提出 Paprika 框架，通过在多种文本决策任务上微调 LLM，使模型学会通用的信息收集和决策能力，并能零样本迁移到完全未见的任务。 - LLM 作为自主 agent 需要与环境交互并收集信息以达成目标 - 直接在真实世界部署收集数据代价高昂且有风险 - 合…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "in-context RL"
  - "curious agent"
  - "curriculum learning"
  - "sequential decision making"
  - "Paprika"
---

# Training a Generally Curious Agent (Paprika)

**会议**: ICML 2025  
**arXiv**: [2502.17543](https://arxiv.org/abs/2502.17543)  
**代码**: -  
**领域**: 模型压缩  
**关键词**: in-context RL, curious agent, curriculum learning, sequential decision making, Paprika  

## 一句话总结

提出 Paprika 框架，通过在多种文本决策任务上微调 LLM，使模型学会通用的信息收集和决策能力，并能零样本迁移到完全未见的任务。

## 研究背景与动机

- LLM 作为自主 agent 需要与环境交互并收集信息以达成目标
- 直接在真实世界部署收集数据代价高昂且有风险
- 合成数据生成无法覆盖所有任务，但 LLM 的 **in-context learning** 能力支持从少量任务学习泛化策略
- **核心思路**：不是训练模型做所有任务，而是训练模型**学会做任务的通用过程**（即 in-context RL）
- 类似于 SFT/RLHF 阶段仅需少量示例即可产生能广泛回答的模型

## 方法详解

### 1. 任务设计（10 个任务组）

所有任务都是纯文本、多轮交互、部分可观察的：

| 任务组 | 训练任务数 | 最大轮次 | 环境反馈 |
|--------|----------|---------|---------|
| 20 Questions | 1499 | 20 | LLM 生成 |
| Guess My City | 500 | 20 | LLM 生成 |
| Wordle | 1515 | 6 | 硬编码 |
| Cellular Automata | 1000 | 6 | 硬编码 |
| Customer Service | 628 | 20 | LLM 生成 |
| Murder Mystery | 203 | 20 | LLM 生成 |
| Mastermind | 1000 | 12 | 硬编码 |
| Battleship | 1000 | 20 | 硬编码 |
| Minesweeper | 1000 | 20 | 硬编码 |
| Bandit Best Arm | 81 | 21 | 硬编码 |

### 2. 数据集构建

- 使用 Min-p 采样（温度 1.5，p=0.3）生成多样化轨迹
- 每个任务生成 $n_\text{sample}=20$ 条轨迹
- 构建偏好对 $(h_w, h_l)$：$h_w$ 为最高分轨迹，$h_l$ 从低分轨迹中随机采样

### 3. 优化目标

**SFT**：在获胜轨迹上最大化似然：

$$\mathcal{L}_\text{SFT} = -\mathbb{E}\left[\frac{1}{\sum_t |a_t^w|}\sum_t \log \pi_\theta(a_t^w | h_{:t}^w)\right]$$

**多轮 DPO**：

$$\mathcal{L}_\text{DPO} = -\mathbb{E}\left[\log\sigma\left(\sum_t \beta\log\frac{\pi_\theta(a_t^w|h_{:t}^w)}{\pi_\text{ref}(a_t^w|h_{:t}^w)} - \sum_t \beta\log\frac{\pi_\theta(a_t^l|h_{:t}^l)}{\pi_\text{ref}(a_t^l|h_{:t}^l)}\right)\right]$$

仅在 agent 动作 token 上计算损失（排除环境生成的 token）。

**RPO**：结合 SFT + DPO 缓解 DPO 的"非预期反对齐"：

$$\mathcal{L}_\text{RPO} = \mathcal{L}_\text{DPO} + \alpha \mathcal{L}_\text{SFT}$$

### 4. 课程学习：可扩展任务选择

**学习潜力度量**：使用变异系数衡量任务的学习信号强度：

$$\nu_\pi(\tau) = \frac{\sqrt{\sigma^2_\pi(\tau)}}{R_\pi(\tau)}$$

高方差→有可能采到好坏两种轨迹→DPO 有信号；按平均奖励归一化使跨任务可比。

**UCB 算法选择任务组**：将每个任务组视为一个臂，用 UCB 平衡探索与利用。

## 实验结果

### 主实验：全任务训练

Paprika 在所有 10 个任务组上提升 Llama-3.1-8B-Instruct 的平均成功率 **47%**（相对提升），仅使用约 22,500 条轨迹。

### 零样本迁移 (Leave-One-Out)

| 任务组 | 基线 | Paprika (LOO) | Paprika (全部) | 单任务训练 |
|--------|------|--------------|--------------|----------|
| Bandit | 42.25% | **62.25%** | 65.0% | 58.0% |
| 20 Questions | ~25% | ~38% | ~40% | ~35% |
| Murder Mystery | ~15% | ~28% | ~32% | ~30% |

- LOO 模型在 **9/10 任务组**上优于初始模型
- 在 **7/10 任务组**上，训练全部任务优于单任务训练
- 表明跨任务策略迁移确实发生

### 课程学习效果

- 课程学习比均匀采样在平均成功率上提升 **1.4%**，pass@4 提升 **3.3%**
- 主要在中等难度任务上获益

## 亮点

- 验证了 LLM 可以通过文本决策任务学到**可迁移的通用探索策略**
- 无需已知最优算法（如 UCB）来生成训练数据，模型自己的多样化采样即可
- 变异系数作为学习潜力度量直观有效
- 任务设计多样性好，涵盖推理、搜索、规划等多种策略
- 与 meta-RL 的联系构建了 LLM agent 训练的理论框架

## 局限性

- 数据生成是主要瓶颈（比模型更新更耗资源）
- Wordle 出现负迁移，表明并非所有策略都能跨任务迁移
- 仅在 8B 和 12B 模型上实验，更大或更小模型的效果未知
- 任务环境部分使用 LLM 生成反馈，可能引入噪声
- 缺少与 online RL 方法的对比（仅使用 offline DPO）

## 评分

⭐⭐⭐⭐ — 令人兴奋的研究方向，证明了通用决策能力可以通过合成数据训练获得并零样本迁移，但瓶颈在数据生成效率。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles](moragent_parameter_efficient_agent_tuning_with_mixture-of-roles.md)
- [\[ICML 2025\] Towards an Optimal Control Perspective of ResNet Training](towards_an_optimal_control_perspective_of_resnet_training.md)
- [\[ACL 2025\] CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration](../../ACL2025/model_compression/cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[ICLR 2026\] Automated Stateful Specialization for Adaptive Agent Systems](../../ICLR2026/model_compression/automated_stateful_specialization_for_adaptive_agent_systems.md)

</div>

<!-- RELATED:END -->
