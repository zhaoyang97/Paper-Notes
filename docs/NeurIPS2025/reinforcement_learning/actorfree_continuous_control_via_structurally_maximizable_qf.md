---
title: >-
  [论文解读] Actor-Free Continuous Control via Structurally Maximizable Q-Functions
description: >-
  [NeurIPS 2025][强化学习][无actor Q-learning] 提出 Q3C（Q-learning for Continuous Control with Control-points），通过学习一组控制点来逼近 Q 函数并保证最大值恰好在控制点上取到，配合动作条件化 Q 值生成、控制点多样性损失和尺度归一化等关键改进，在标准基准上匹配 TD3，在受限动作空间中显著超越所有 actor-critic 方法。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "无actor Q-learning"
  - "连续控制"
  - "控制点"
  - "wire-fitting"
  - "结构化最大化"
---

# Actor-Free Continuous Control via Structurally Maximizable Q-Functions

**会议**: NeurIPS 2025  
**arXiv**: [2510.18828](https://arxiv.org/abs/2510.18828)  
**代码**: [https://github.com/USC-Lira/Q3C](https://github.com/USC-Lira/Q3C)  
**领域**: 强化学习  
**关键词**: 无actor Q-learning, 连续控制, 控制点, wire-fitting, 结构化最大化

## 一句话总结

提出 Q3C（Q-learning for Continuous Control with Control-points），通过学习一组控制点来逼近 Q 函数并保证最大值恰好在控制点上取到，配合动作条件化 Q 值生成、控制点多样性损失和尺度归一化等关键改进，在标准基准上匹配 TD3，在受限动作空间中显著超越所有 actor-critic 方法。

## 研究背景与动机

**领域现状**：连续动作空间的 RL 通常使用 actor-critic 方法（如 DDPG、TD3、SAC），其中 critic 估计 Q 值，actor 通过梯度上升在 Q 函数景观中寻找最优动作。纯基于值的方法（如 DQN）在离散空间中表现出色，但因无法在连续空间中穷举动作，被认为不适用于连续控制。

**现有痛点**：actor-critic 方法存在根本性问题：(1) actor 和 critic 的耦合训练导致不稳定；(2) 梯度上升只能找到局部最优动作，在 Q 函数非凸时（如受限动作空间）失败；(3) 额外的 actor 网络增加超参数和计算开销。之前的无 actor 尝试（如 NAF 限制 Q 为二次型、RBF-DQN 用径向基函数）要么表达能力不足，要么最大值不保证在基点上。

**核心矛盾**：连续 Q-learning 的核心困难是 Bellman 方程中的 $\max_a Q(s,a)$ 操作——在连续空间中无法精确求解。actor 是一种近似方案但引入了新的问题。需要一种 Q 函数表示方式，使得最大化操作可以精确且高效地完成。

**本文目标** 设计一种结构化可最大化的 Q 函数表示，使得无需 actor 就能在连续空间中精确找到最优动作。

**切入角度**：重新审视 wire-fitting 框架——用一组"控制点"锚定 Q 函数逼近，使得 Q 的最大值结构性地出现在某个控制点上。此方向因早期在深度 RL 中效果不佳而被放弃，但作者发现结合现代深度 RL 技术可以使其焕发生机。

**核心 idea**：用控制点插值构造结构化可最大化的 Q 函数，配合一系列架构和算法创新使其在深度 RL 中达到 SOTA 水平。

## 方法详解

### 整体框架

Q3C 由三个组件构成：(1) 控制点生成器 $g_\phi(s)$ 输出 $N$ 个候选动作 $\hat{a}_i(s)$；(2) Q 估计器 $h_\psi(s, \hat{a}_i)$ 对每个控制点估计 Q 值 $\hat{Q}_i(s)$；(3) wire-fitting 插值器根据控制点位置和 Q 值计算任意动作 $a$ 的 Q 值。最优动作直接通过 $\arg\max_i \hat{Q}_i$ 在 $N$ 个标量中选取，无需梯度上升。

### 关键设计

1. **Wire-fitting 插值保证结构化最大值**:

    - 功能：构造 Q 函数使最大值必然在某个控制点上取到
    - 核心思路：$Q(s,a) = \frac{\sum_i \hat{Q}_i w_i}{\sum_i w_i}$，其中权重 $w_i = \frac{1}{|a - \hat{a}_i|^2 + c_i(\hat{Q}_{\max} - \hat{Q}_i)}$。当 $a$ 趋近最高 Q 值的控制点时，对应权重趋于无穷大，$Q$ 趋近该点的值。作者证明这种插值保持了万能逼近能力（Proposition）
    - 设计动机：相比 NAF 的二次型限制和 RBF-DQN 最大值不保证在基点上的问题，wire-fitting 既有充足的表达能力又提供结构化最大值保证

2. **动作条件化 Q 值生成（Action-Conditioned Q-value）**:

    - 功能：确保 Q 值估计与控制点位置一致
    - 核心思路：将架构分解为两个阶段——控制点生成器 $g_\phi(s)$ 输出 $N$ 个动作，然后用独立的 Q 估计器 $h_\psi(s, \hat{a}_i)$ 为每个控制点评估 Q 值。所有控制点共用同一个 Q 估计器，确保相同/相近动作得到一致的 Q 值
    - 设计动机：原始 wire-fitting 中 Q 值与控制点位置独立预测，可能给相同位置的控制点分配完全不同的 Q 值，导致训练不稳定

3. **控制点多样性与尺度归一化**:

    - 功能：防止控制点聚集和跨任务的尺度不一致
    - 核心思路：添加分离损失 $L_{\text{sep}} = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{\|\hat{a}_i - \hat{a}_j\|_2 + \epsilon}$ 鼓励控制点均匀分布；归一化 wire-fitting 权重中的 Q 值差异项 $\tilde{Q}_i = (\hat{Q}_i - \hat{Q}_{\min})/(\hat{Q}_{\max} - \hat{Q}_{\min})$ 并指数衰减平滑系数 $c_i$，使方法对不同奖励尺度和动作范围具有鲁棒性
    - 设计动机：无约束时控制点倾向于聚集在动作空间边界（实验观察到的现象），破坏 Q 函数的表达能力

### 损失函数 / 训练策略

基于 TD3 框架：双 Q 网络避免过估计、目标网络稳定学习目标、高斯噪声探索。总损失 = Bellman 损失 + $\lambda \cdot L_{\text{sep}}$。学习率采用延迟指数衰减调度，最终学习率为初始的 10%。默认 $N=20, k=10$（即取最近 10 个控制点计算 Q 值）。

## 实验关键数据

### 主实验

| 环境 | TD3 | NAF | Wire-Fitting | RBF-DQN | **Q3C** |
|------|-----|-----|-------------|---------|---------|
| Pendulum | -144.6 | -252.4 | -351.5 | -143.9 | **-159.5** |
| Swimmer | 300.7 | 20.6 | 313.6 | 92.4 | **316.4** |
| Hopper | 3113.4 | 500.8 | 1987.5 | 2189.4 | **3206.1** |
| Walker2d | **4770.8** | 2179.6 | 2462.3 | 781.6 | 3977.4 |
| HalfCheetah | **9984.7** | 3531.5 | 7546.2 | 6175.6 | 9468.7 |
| Ant | **5167.7** | -18.1 | 1154.6 | 1674.0 | 3698.4 |

受限环境（非凸 Q 函数）：

| 环境 | TD3 | NAF | Wire-Fitting | RBF-DQN | **Q3C** |
|------|-----|-----|-------------|---------|---------|
| InvPendulumBox | 782.8 | 909.7 | 386.4 | 862.0 | **1000.0** |
| HalfCheetahBox | 2276.7 | **4867.1** | -2139.8 | 2238.4 | 4357.8 |
| HopperBox | 1406.8 | 461.5 | 169.8 | 1641.2 | **1974.3** |

### 消融实验

| 配置 | Hopper | BipedalWalker | HalfCheetah |
|------|--------|---------------|-------------|
| **Q3C (full)** | **3206** | **290** | **9469** |
| - CondQ | 2330 | 286 | 8386 |
| - Ranking | 3037 | 180 | 8961 |
| - Div | 1921 | -68 | 5283 |
| - Norm | 2915 | 262 | 8746 |
| Wire-Fitting | 1988 | 70 | 7546 |

### 关键发现

- Q3C 在标准环境中与 TD3 表现相当，但在受限/非凸 Q 函数环境中显著更优——InvPendulumBox 上 Q3C 得分 1000（完美），TD3 仅 783
- 控制点多样性（Div）是最关键组件——去掉后 BipedalWalker 从 290 暴跌到 -68，Hopper 跌 40%
- 原始 wire-fitting 在深度 RL 中效果很差，Q3C 的改进使其性能提升 2-5 倍
- 在 26 维动作空间的 Adroit 任务上 Q3C 能匹配 TD3，表明方法可扩展到高维

## 亮点与洞察

- **结构化最大值**的巧妙优势：将连续空间的 $\max$ 转化为 $N$ 个标量的 $\arg\max$，完全消除了梯度上升找最优动作的局部最优问题。在 Q 函数非凸时这一优势尤为突出
- **无 actor 的简洁性**：Q3C 同时充当 actor 和 critic，减少了超参数（无需独立调 actor 学习率、更新频率等），训练更稳定
- **控制点数量不需要随动作维度线性增长**（26 维空间只需 70 个），因为动作条件化 Q 估计器的并行化设计使参数量不随 $N$ 线性增长

## 局限与展望

- 在标准环境的 Ant-v4 和 Walker2d 上仍落后于 TD3，大约差 20-30%
- 探索策略直接沿用 TD3 的高斯噪声，缺乏专门设计（如基于控制点 Q 值的 Boltzmann 探索）
- 仅在确定性策略上验证，未扩展到随机策略（如 SAC 风格的软 Q）
- 离线 RL 场景值得探索——控制点插值对 Q 值的约束可能天然缓解过估计问题

## 相关工作与启发

- **vs TD3**: TD3 是确定性 actor-critic 的 SOTA，但 actor 的梯度上升只能找局部最优。Q3C 在标准环境持平、受限环境显著优于 TD3，代价是在 Ant 等高维环境中稍逊
- **vs NAF (Gu et al. 2016)**: NAF 将 Q 函数限制为动作的二次型，允许解析求最大值但表达能力严重受限。Q3C 通过控制点插值保持万能逼近能力
- **vs RBF-DQN (Asadi et al. 2021)**: RBF 插值不保证最大值在基点上，且需要大量中心（~100）。Q3C 的控制点插值结构性保证最值在控制点上，且效率更高

## 评分

- 新颖性: ⭐⭐⭐⭐ 重新激活被放弃的 wire-fitting 方向，关键创新在于使其在深度 RL 中可用
- 实验充分度: ⭐⭐⭐⭐⭐ 标准+受限环境、多基线、详尽消融、高维测试、可视化分析
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机和贡献阐述充分，理论和实验结合紧密
- 价值: ⭐⭐⭐⭐ 在受限动作空间场景下有明确优势，为连续 Q-learning 开辟了新路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[ICML 2026\] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](../../ICML2026/reinforcement_learning/debiased_model-based_representations_for_sample-efficient_continuous_control.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [\[NeurIPS 2025\] A Differential and Pointwise Control Approach to Reinforcement Learning](a_differential_and_pointwise_control_approach_to_reinforceme.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)

</div>

<!-- RELATED:END -->
