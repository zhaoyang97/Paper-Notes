---
title: >-
  [论文解读] Reward-Aware Proto-Representations in Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][后继表示] 系统发展了默认表示（DR）的理论基础——推导了 DP 和 TD 学习算法、分析了特征空间结构、提出了默认特征进行函数逼近——并在奖励塑形、期权发现、探索和迁移学习四个场景中展示了 DR 相比后继表示（SR）的奖励感知优势。 领域现状 领域现状：后继表示（SR）通过编码转移…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "后继表示"
  - "默认表示"
  - "奖励感知表征"
  - "期权发现"
  - "迁移学习"
---

# Reward-Aware Proto-Representations in Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.16217](https://arxiv.org/abs/2505.16217)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 后继表示, 默认表示, 奖励感知表征, 期权发现, 迁移学习

## 一句话总结
系统发展了默认表示（DR）的理论基础——推导了 DP 和 TD 学习算法、分析了特征空间结构、提出了默认特征进行函数逼近——并在奖励塑形、期权发现、探索和迁移学习四个场景中展示了 DR 相比后继表示（SR）的奖励感知优势。

## 研究背景与动机

### 领域现状

**领域现状**：后继表示（SR）通过编码转移动态来表示状态间的时间关系，已广泛应用于奖励塑形、探索、迁移学习等。但 SR 是奖励无关的——它只编码到达各状态的转移次数

**现有痛点**：在有"应避免"的低奖励区域的环境中，SR 无法区分高奖励路径和低奖励路径。Piray 和 Daw 提出的默认表示（DR）是奖励感知的，但缺乏高效的在线学习算法和理论分析

**核心矛盾**：SR 编码的是 $\gamma^{\eta(\tau)}$（折扣后的步数），DR 编码的是 $\exp(r(\tau)/\lambda)$（累积奖励的指数）——后者自然地整合了奖励信息

**核心 idea**：完善 DR 的理论工具箱，使其可以像 SR 一样便捷地应用于 RL

## 方法详解

### 整体框架
在线性可解 MDP 框架下，DR 定义为 $\mathbf{Z} = [\text{diag}(\exp(-\mathbf{r}/\lambda)) - \mathbf{P}^{\pi_d}]^{-1}$。本文从三个层面推进：学习算法、理论分析、函数逼近。

### 关键设计

1. **DP 和 TD 学习算法**：

    - DP 更新：$\mathbf{Z}_{k+1} = \mathbf{R}^{-1} + \mathbf{R}^{-1}\mathbf{P}^{\pi_d}\mathbf{Z}_k$，证明了收敛性（利用 Neumann 级数）
    - TD 更新：$\mathbf{Z}(s,j) \leftarrow \mathbf{Z}(s,j) + \alpha[\exp(r/\lambda)(\mathbb{1}_{s=j} + \mathbf{Z}(s',j)) - \mathbf{Z}(s,j)]$
    - 对比 SR 的 TD 更新（将 $\gamma$ 替换为 $\exp(r/\lambda)$），区别在于奖励感知的折扣因子

2. **特征空间分析**：

    - 定理3.1：当奖励在所有状态上**恒定**时，SR 和 DR 共享相同的特征向量
    - 当奖励**不同**时，DR 的特征向量反映低奖励区域位置（见图2），而 SR 只反映转移距离

3. **默认特征 (Default Features)**：

    - 类似后继特征（SF），分解为：$\exp(\mathbf{v}^*_N/\lambda) = \boldsymbol{\zeta}(s)^\top \mathbf{w}$
    - TD 更新：$\boldsymbol{\zeta}(s) \leftarrow \boldsymbol{\zeta}(s) + \alpha(\exp(r/\lambda)\boldsymbol{\zeta}(s') - \boldsymbol{\zeta}(s))$
    - 不需要访问转移动态即可计算不同终端奖励下的最优策略

## 实验关键数据

### 主实验 — 奖励塑形（四个含低奖励区域的环境）

| 环境 | DR-pot | SR-pot | SR-pri | 无塑形 |
|------|--------|--------|--------|--------|
| Grid Task | **最优** | 次优 | 最差 | 慢收敛 |
| Four Rooms | **最优** | 走次优路径 | 走次优路径 | 极慢 |

### 消融/对比 — 探索（Count-based, 值×10³）

| 环境 | Sarsa | +SR | +DR |
|------|-------|-----|-----|
| RiverSwim | 25 | 1,206 | **2,964** |
| SixArms | 265 | 1,066 | **3,518** |

### 关键发现
- DR 在有低奖励障碍的环境中显著优于 SR——SR 总是选最短路径（可能经过低奖励区），DR 绕行避开
- 当所有状态奖励相同时，DR 和 SR 表现几乎一致（理论预期）
- DR 的范数作为密度模型在 RiverSwim 上甚至显著优于 SR——可能因为编码了奖励信息后学习更快

## 亮点与洞察
- **DR 是 SR 的奖励感知推广**：统一了"去哪"和"去那划不划算"两个维度
- 默认特征实现了**不需要环境动态就能做迁移学习**——SR 特征虽然更灵活（任意奖励变化）但只能恢复到参考策略水平

## 局限与展望
- DR 中 $\exp(-r/\lambda)$ 在大负奖励时可能产生数值不稳定
- 只在表格环境中验证，未扩展到深度 RL
- DR 只能在终端奖励变化时做迁移，不如 SF 灵活

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统完善了一个有潜力但被低估的表征
- 实验充分度: ⭐⭐⭐⭐ 覆盖奖励塑形/探索/期权发现/迁移四个场景
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，实验与理论预测一致
- 价值: ⭐⭐⭐⭐ 为奖励感知表征奠定了理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)
- [\[ICML 2026\] From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](../../ICML2026/reinforcement_learning/from_reward-free_representations_to_preferences_rethinking_offline_preference-ba.md)
- [\[NeurIPS 2025\] DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](learning_from_demonstrations_via_capability-aware_goal_sampling.md)

</div>

<!-- RELATED:END -->
