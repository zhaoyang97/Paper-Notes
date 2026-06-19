---
title: >-
  [论文解读] Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning
description: >-
  [ICML 2025][强化学习][Visual RL] 本文提出 Sliding Puzzles Gym (SPGym)，一个将经典 8-拼图改造为视觉 RL 任务的基准，通过独立调节图片池大小来精确控制视觉表征学习的复杂度，实验揭示当前方法在视觉多样性增大时的根本性记忆化局限。 视觉强化学习（Visual RL）要求智能…
tags:
  - "ICML 2025"
  - "强化学习"
  - "Visual RL"
  - "Benchmark"
  - "Representation Learning"
  - "Sliding Puzzle"
  - "Generalization"
---

# Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning

**会议**: ICML 2025  
**arXiv**: [2410.14038](https://arxiv.org/abs/2410.14038)  
**代码**: [bryanoliveira/sliding-puzzles-gym](https://github.com/bryanoliveira/sliding-puzzles-gym)  
**领域**: 视觉强化学习 / 表征学习基准  
**关键词**: Visual RL, Benchmark, Representation Learning, Sliding Puzzle, Generalization  

## 一句话总结

本文提出 Sliding Puzzles Gym (SPGym)，一个将经典 8-拼图改造为视觉 RL 任务的基准，通过独立调节图片池大小来精确控制视觉表征学习的复杂度，实验揭示当前方法在视觉多样性增大时的根本性记忆化局限。

## 研究背景与动机

视觉强化学习（Visual RL）要求智能体从原始像素输入中提取任务相关特征并用于决策。评估表征学习能力是关键挑战，但现有基准存在核心问题：

**耦合问题**：Atari、DM Control 等经典基准的性能指标**混杂了表征学习、策略优化和环境动力学**，无法单独评估表征学习

**ProcGen**：同时改变视觉和任务难度，无法分离表征学习的影响

**Distracting Control Suite**：引入与任务无关的视觉干扰，智能体可以安全忽略

**COOM**：关注灾难性遗忘而非单任务内的表征学习

核心需求：一个能**精确、独立地调节视觉复杂度**的基准，同时保持任务动力学不变。

## 方法详解

### 整体框架

SPGym 将经典滑块拼图转化为 POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \mathcal{S}_0, \Omega, \mathcal{O})$：
- **状态空间** $\mathcal{S}$：所有可解的拼图配置（3×3 网格约 $1.81 \times 10^5$ 种）
- **动作空间** $\mathcal{A}$：上、下、左、右四个方向
- **观测函数** $\mathcal{O}(s, i)$：从数据池 $\mathcal{I}$ 中选择图片 $i$，按拼图状态 $s$ 排列图片块
- 智能体**无法直接访问状态**，必须从 84×84 像素重建原图

### 关键设计：双正交缩放机制

1. **视觉多样性缩放**：调节图片池大小 $p$（1 到 100+张），改变 $|\Omega|$，但 $\mathcal{P}$、$\mathcal{A}$、$\mathcal{R}$ 不变
2. **搜索复杂度缩放**：调节网格尺寸（3×3 → 4×4），改变 $|\mathcal{S}|$ 和视觉复杂度

### 奖励函数

基于归一化曼哈顿距离的稠密奖励：

$$\mathcal{R}(s) = \begin{cases} -D & \text{if action valid} \\ -1 & \text{if action invalid} \\ +1 & \text{if solved} \end{cases}$$

$$D = \frac{\sum_{i,j} |u_{i,j} - u^*_{i,j}| + |v_{i,j} - v^*_{i,j}|}{\sum_{i,j} \max(i, H-i) + \max(j, W-j)}$$

## 实验关键数据

### 主实验：样本效率（百万步达到 80% 成功率）

| Agent | Pool 1 | Pool 5 | Pool 10 |
|-------|--------|--------|---------|
| PPO | 1.75±0.44 | 7.80±1.08 | 9.73±0.36 |
| PPO + PT (ID) | **0.95±0.21** | **5.55±1.22** | **9.17±1.10** |
| SAC | 0.33±0.07 | 0.91±0.12 | 2.03±0.38 |
| SAC + RAD | **0.24±0.03** | **0.42±0.06** | **0.82±0.18** |
| SAC + CURL | 0.46±0.10 | 1.56±0.31 | 5.24±1.92 |
| SAC + SPR | 2.09±0.81 | 3.68±1.68 | 10.00±0.00 |
| DreamerV3 | **0.42±0.06** | **1.23±0.20** | **1.44±0.58** |
| DreamerV3 w/o dec | 1.13±0.12 | 1.79±0.61 | 2.57±0.91 |

### 退化模式对比

| Agent | 退化阈值（池大小） | 完全失败阈值 |
|-------|-----------------|------------|
| PPO | 10 | 20 |
| SAC | 30 | 50 |
| DreamerV3 | 50 | 100 |

### Easy OOD 泛化结果（成功率）

| Agent | Pool 1 | Pool 5 | Pool 10 |
|-------|--------|--------|---------|
| SAC + AE | 0.78 | 0.64 | 0.55 |
| SAC + SB | **0.89** | 0.65 | 0.06 |
| SAC + RAD | 0.62 | 0.42 | 0.30 |
| SEFA | 0.76 | 0.44 | 0.37 |

### Hard OOD 泛化结果

所有方法在完全未见图片上**几乎 0% 成功率**——揭示端到端 RL 方法依赖记忆而非真正的视觉理解。

### 网格尺寸影响（百万步，池大小 1）

| Grid | PPO | SAC | DreamerV3 |
|------|-----|-----|-----------|
| 3×3 | 1.75 | 0.33 | 0.42 |
| 4×4 | 24.46 | 8.14 | **2.26** |

### 消融关键发现

1. **表征质量与性能强相关**：线性探测准确率与样本效率的 Pearson $r = -0.81$，$p = 1.1 \times 10^{-13}$
2. **Easy OOD 成功率与样本效率强相关**：$r = -0.81$，$p = 2.5 \times 10^{-12}$
3. **数据增强 (RAD) 一致最优**：复杂辅助目标（CURL, SPR, DBC, VAE）往往不如简单数据增强
4. **DreamerV3 的世界模型最稳健**：去掉解码器梯度后性能下降，证明重建目标重要
5. **DiffusionDB 上的结果与 ImageNet 一致**：排除了数据集特异性

## 亮点与洞察

1. **精确隔离表征学习**：SPGym 设计巧妙——视觉理解是任务成功的**必要条件**（需要正确拼图），而非可忽略的干扰
2. **反直觉发现**：训练池越大，Easy OOD 泛化反而越差——小池训练的模型学到了更强的任务特定不变性
3. **记忆化暴露**：Hard OOD 的全面失败证明现有端到端 RL 方法本质上在**记忆视觉模式**而非学习泛化表征
4. **方法假设不匹配**：CURL（对比学习要求增强观测相似）、DBC（要求视觉相似=动力学相似）等方法的隐含假设与 SPGym 的离散、高变异观测不匹配
5. **持续训练显著改善解质量**：DreamerV3 持续训练后平均步数从 126 步降至 23 步，接近理论最优 22 步

## 局限性

1. **未充分调优**：以"开箱即用"为目标，可能未展现各方法的峰值能力
2. **统计鲁棒性**：仅 5 个随机种子，鉴于图片池的高随机性，可能不足
3. **任务简单性**：3×3 拼图本身是较简单的搜索问题，主要难度来自视觉表征
4. **离散动作空间**：与连续控制的视觉 RL（DM Control 等）有本质不同

## 相关工作

- **传统 RL 基准**：Atari, DM Control Suite, DeepMind Lab, CARLA
- **专用视觉 RL 基准**：ProcGen, Distracting Control Suite, COOM
- **拼图基准**：Estermann et al. 2024（离散状态空间）
- **滑块拼图求解**：A*, IDA*, DRL 方法（Agostinelli et al., Moon & Cho）
- **表征学习方法**：RAD, CURL, SPR, DBC, DreamerV3, SAC-AE

## 评分

⭐⭐⭐⭐ (4/5)

SPGym 设计精巧，成功隔离了视觉表征学习挑战。"所有方法都在记忆化"的发现具有重要警示意义。但基准本身偏简单（拼图任务），且开箱即用的评估可能低估了方法潜力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](../../ICLR2026/reinforcement_learning/reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [\[CVPR 2026\] Saliency-Guided Representation with Consistency Policy Learning for Visual Unsupervised Reinforcement Learning](../../CVPR2026/reinforcement_learning/saliency-guided_representation_with_consistency_policy_learning_for_visual_unsup.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](../../NeurIPS2025/reinforcement_learning/reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[ICML 2025\] A Theoretical Study of (Hyper) Self-Attention through the Lens of Interactions: Representation, Training, Generalization](a_theoretical_study_of_hyper_self-attention_through_the_lens_of_interactions_rep.md)

</div>

<!-- RELATED:END -->
