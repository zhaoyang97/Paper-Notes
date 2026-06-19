---
title: >-
  [论文解读] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][鲁棒强化学习] 首次给出鲁棒平均奖励 MDP 策略评估的有限样本复杂度分析：通过构造精巧的半范数证明鲁棒 Bellman 算子具有收缩性质，结合截断 Multi-Level Monte Carlo 估计器实现有限期望样本复杂度，最终达到阶最优的 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 样本复杂度。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "鲁棒强化学习"
  - "平均奖励MDP"
  - "策略评估"
  - "有限样本分析"
  - "半范数收缩"
---

# Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2502.16816](https://arxiv.org/abs/2502.16816)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 鲁棒强化学习, 平均奖励MDP, 策略评估, 有限样本分析, 半范数收缩

## 一句话总结
首次给出鲁棒平均奖励 MDP 策略评估的有限样本复杂度分析：通过构造精巧的半范数证明鲁棒 Bellman 算子具有收缩性质，结合截断 Multi-Level Monte Carlo 估计器实现有限期望样本复杂度，最终达到阶最优的 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 样本复杂度。

## 研究背景与动机

**领域现状**：鲁棒 RL 通过在转移概率的不确定集上做最坏情况优化，解决 sim-to-real gap 等问题。在折扣奖励设置下，鲁棒 Bellman 算子因折扣因子 $\gamma < 1$ 自然具有 sup-norm 下的收缩性质，有限样本分析已较成熟。

**现有痛点**：平均奖励设置更适合需要长期持续高效的应用（排队系统、库存管理、网络控制），但**即使非鲁棒的平均奖励 Bellman 算子在任何范数下都不具备收缩性质**，使得标准不动点迭代分析不可用。因此现有鲁棒平均奖励 RL 工作仅有渐近收敛保证（基于 ODE 分析），无法给出有限样本复杂度。

**核心矛盾**：折扣设置中 $\gamma < 1$ 提供的自然收缩机制在平均奖励中完全缺失，而鲁棒性引入的 min 运算进一步加剧了分析难度——需要在不确定集的所有转移模型上证明收缩。

**本文目标**：(a) 鲁棒平均奖励 Bellman 算子在什么意义下具有收缩性？(b) 如何用有限样本估计涉及非线性最坏情况转移效应的支撑函数？(c) 最终的样本复杂度是什么？

**切入角度**：将所有不确定集内的最坏情况转移矩阵视为一族线性映射，利用其联合谱半径严格小于 1 的性质构造极值范数，进而构建能"一步收缩"的半范数。

**核心 idea**：通过极值范数+商空间修正构造半范数证明鲁棒 Bellman 算子的收缩性，配合截断 MLMC 实现有限样本策略评估。

## 方法详解

### 整体框架
给定策略 $\pi$ 和名义模型 $\tilde{\mathsf{P}}$（不确定集的中心），目标是仅使用名义模型的样本估计鲁棒价值函数 $V$ 和鲁棒平均奖励 $g$。方法分两阶段：(1) 随机近似迭代估计 $V_T$；(2) 用 $V_T$ 估计 $g_T$。核心技术挑战是构造鲁棒 Bellman 算子的估计器（需有限样本+可控偏差/方差）。

### 关键设计

1. **半范数收缩构造（核心理论贡献）**:

    - 功能：证明鲁棒 Bellman 算子 $\mathbf{T}_g(V)(s) = \sum_a \pi(a|s)[r(s,a) - g + \sigma_{\mathcal{P}_s^a}(V)]$ 在某半范数下是收缩映射
    - 核心思路（非鲁棒版本）：对单一转移矩阵 $\mathsf{P}^{\pi}$，利用遍历性得到唯一平稳分布 $d^\pi$，定义波动矩阵 $Q^\pi = \mathsf{P}^\pi - \mathbf{e}^\top d^\pi$，其特征值严格在单位圆内。通过离散 Lyapunov 方程构造范数 $\|\cdot\|_Q$ 使得 $\|Q^\pi x\|_Q \leq \alpha \|x\|_Q$（$\alpha < 1$）。半范数定义为 $\|x\|_{\mathsf{P}} = \|Q^\pi x\|_Q + \epsilon \inf_{c \in \mathbb{R}} \|x - c\mathbf{e}\|_Q$，核为常向量空间
    - 核心思路（鲁棒版本）：对不确定集 $\mathcal{P}$ 中所有 $\mathsf{P}$ 的波动矩阵族 $\{Q_\mathsf{P}^\pi\}$，利用其联合谱半径 $< 1$，通过 Berger-Wang 定理构造极值范数 $\|\cdot\|_{\text{ext}}$ 使得所有 $Q_\mathsf{P}^\pi$ 统一收缩因子 $\alpha$。最终半范数 $\|x\|_{\mathcal{P}} = \sup_{\mathsf{P} \in \mathcal{P}} \|Q_\mathsf{P}^\pi x\|_{\text{ext}} + \epsilon \inf_{c \in \mathbb{R}} \|x - c\mathbf{e}\|_{\text{ext}}$，保证 $\|\mathbf{T}_g(V_1) - \mathbf{T}_g(V_2)\|_{\mathcal{P}} \leq \gamma \|V_1 - V_2\|_{\mathcal{P}}$，$\gamma = \alpha + \epsilon < 1$
    - 设计动机：这是唯一能克服平均奖励缺乏折扣因子收缩的理论路径

2. **截断 MLMC 估计器（技术贡献）**:

    - 功能：为 TV 和 Wasserstein 不确定集的支撑函数 $\sigma_{\mathcal{P}_s^a}(V)$ 构造有限样本无偏/低偏估计器
    - 核心思路：标准 MLMC 从几何分布 $\text{Geom}(\Psi)$ 采样层数 $N$，需要 $2^{N+1}$ 个样本，当 $\Psi < 0.5$ 时期望样本数为无穷。关键创新是设 $\Psi = 0.5$ 并截断 $N' = \min\{N, N_{\max}\}$，使得期望样本数 $\mathbb{E}[M] = N_{\max} + 2 = \mathcal{O}(N_{\max})$（线性增长而非指数增长）
    - 偏差以 $2^{-N_{\max}/2}$ 指数衰减，方差与 $N_{\max}$ 线性增长
    - 设计动机：之前的 MLMC 方法有无穷期望样本复杂度，只能给出渐近收敛保证

3. **三类不确定集的处理**:

    - **污染模型**（Contamination）：$\sigma_{\mathcal{P}_s^a}(V) = (1-\delta)(\tilde{\mathsf{P}}_s^a)^\top V + \delta \min_s V(s)$，对名义转移线性，直接用一个样本无偏估计即可
    - **全变差距离（TV）**：支撑函数涉及 span 半范数的对偶优化，需要 MLMC。Lipschitz 性质：$|\sigma_{\mathcal{P}_{TV}}(V) - \sigma_{\mathcal{Q}_{TV}}(V)| \leq (1+1/\delta)\|V\|_{\text{sp}}\|p-q\|_1$
    - **Wasserstein 距离**：支撑函数涉及 inf-sup 双层优化，同样需要 MLMC。Lipschitz 常数更紧：$|\sigma_{\mathcal{P}_W}(V) - \sigma_{\mathcal{Q}_W}(V)| \leq \|V\|_{\text{sp}}\|p-q\|_1$

4. **鲁棒平均奖励 TD 学习（Algorithm 2）**:

    - 第一阶段（迭代 $T$ 步）：$V_{t+1}(s) \leftarrow V_t(s) + \eta_t(\hat{\mathbf{T}}_{g_0}(V_t)(s) - V_t(s))$，然后中心化 $V_{t+1}(s) = V_{t+1}(s) - V_{t+1}(s_0)$
    - 第二阶段（迭代 $T$ 步）：用 $V_T$ 估计 $g_T \leftarrow g_t + \beta_t(\bar{\delta}_t - g_t)$

### 样本复杂度结论
- 污染不确定集：$\mathcal{O}\left(\frac{SAt_{\text{mix}}^2}{\epsilon^2(1-\gamma)^2}\right)$
- TV 和 Wasserstein：$\tilde{\mathcal{O}}\left(\frac{SAt_{\text{mix}}^2}{\epsilon^2(1-\gamma)^2}\right)$
- 关于 $\epsilon$ 均为阶最优 $\tilde{\mathcal{O}}(\epsilon^{-2})$

## 实验关键数据

本文为纯理论工作，无数值实验。主要结果为定理形式的样本复杂度界。

### 主要理论结果对比

| 结果 | 不确定集 | 样本复杂度 | 相比前人 |
|------|---------|-----------|---------|
| Theorem 6.1（策略评估） | 污染 | $\mathcal{O}(SA t_{\text{mix}}^2 / (\epsilon^2(1-\gamma)^2))$ | 首个非渐近结果 |
| Theorem 6.1（策略评估） | TV / Wasserstein | $\tilde{\mathcal{O}}(SA t_{\text{mix}}^2 / (\epsilon^2(1-\gamma)^2))$ | 首个非渐近结果 |
| Theorem 6.2（平均奖励估计） | 所有三类 | $\tilde{\mathcal{O}}(SA t_{\text{mix}}^2 / (\epsilon^2(1-\gamma)^2))$ | 首个非渐近结果 |

### 技术引理对比

| 引理 | 内容 | 关键数值 |
|------|------|---------|
| Theorem 5.1 | 截断 MLMC 期望样本数 | $\mathbb{E}[M] = N_{\max} + 2$ |
| Theorem 5.2 | 偏差衰减率 | $\leq 6(1+1/\delta) \cdot 2^{-N_{\max}/2} \|V\|_{\text{sp}}$ (TV) |
| Theorem 5.4 | 方差界 | $\leq 3\|V\|_{\text{sp}}^2 + 144(1+1/\delta)^2 \|V\|_{\text{sp}}^2 N_{\max}$ (TV) |

### 关键发现
- 半范数收缩是平均奖励鲁棒 RL 有限样本分析的突破口
- 截断 MLMC 的 $\Psi=0.5$ 选择是精确消除指数样本增长的关键——$\Psi < 0.5$ 会导致无穷期望样本
- 结果在 $\epsilon$ 意义下阶最优，但关于 $S$, $A$, $\gamma$ 的紧致性仍为开放问题

## 亮点与洞察
- **半范数构造的优美性**：从单个转移矩阵的 Lyapunov 范数出发，经过"联合谱半径→极值范数→商空间修正"三步推广到鲁棒情形，每步都有清晰的数学动机。这一构造手法可能可以应用于其他缺乏自然收缩的随机近似问题
- **截断 MLMC 的工程巧妙**：$\Psi=0.5$ 让几何分布的指数增长恰好被样本数的指数增长抵消，得到线性期望复杂度——这是一个简洁而深刻的 trick
- **统一处理三类不确定集**：通过 Lipschitz 引理统一了不同度量下的分析

## 局限与展望
- **纯理论无实验**：没有验证算法在实际 MDP 上的表现，也未比较常数因子
- **需要遍历性假设**（Assumption 3.1），且要求不确定集半径足够小使得所有 $\mathsf{P} \in \mathcal{P}$ 下链都是遍历的
- **关于 $S$, $A$, $\gamma$ 的依赖未必紧致**，作者也承认进一步收紧这些依赖是开放问题
- **未考虑函数近似**，所有分析基于 tabular 设置
- **策略优化（而非评估）的有限样本分析**尚未解决

## 相关工作与启发
- **vs wang2023model**：提出鲁棒 RVI TD/Q-learning 但仅有 ODE 渐近收敛保证，用了无截断 MLMC 导致无穷样本复杂度；本文通过半范数+截断 MLMC 首次得到有限样本界
- **vs zhang2021finite**：非鲁棒平均奖励 TD 的有限样本分析，本文将其框架推广到鲁棒设置（额外需要处理不确定集带来的偏差和非线性）
- **vs wang2022policy, zhou2024natural**：鲁棒折扣 RL 的策略评估，利用 $\gamma<1$ 的 sup-norm 收缩，技术路线与本文完全不同——本文的半范数方法处理了 $\gamma=1$ 的本质困难
- 半范数收缩的构造方法可能对控制论中的稳定性分析有启发

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 半范数收缩构造和截断 MLMC 是两个独立且优美的理论创新
- 实验充分度: ⭐⭐ 纯理论工作无实验，缺少数值验证
- 写作质量: ⭐⭐⭐⭐ 证明思路清晰、proof sketch 非常好，但公式密集对非理论读者门槛高
- 价值: ⭐⭐⭐⭐⭐ 填补了鲁棒平均奖励 RL 有限样本分析的重要空白，半范数构造方法有广泛理论影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](risk-averse_total-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
