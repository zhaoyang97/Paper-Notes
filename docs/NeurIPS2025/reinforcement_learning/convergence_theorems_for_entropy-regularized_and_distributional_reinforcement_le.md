---
title: >-
  [论文解读] Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][熵正则化] 提出 温度解耦策略（temperature decoupling gambit）：，证明在熵正则化强化学习中，通过解耦评估温度和行为温度，可以在温度趋于零时保证策略和回报分布收敛到一个可解释的、保持多样性的最优策略。 领域现状：标准RL中存在多个最优策略…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "熵正则化"
  - "分布式强化学习"
  - "收敛性"
  - "温度解耦"
  - "最优策略"
---

# Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.08526](https://arxiv.org/abs/2510.08526)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 熵正则化, 分布式强化学习, 收敛性, 温度解耦, 最优策略  
**arXiv**: [2510.08526](https://arxiv.org/abs/2510.08526)  
**代码**: 无  
**领域**: 强化学习  

## 一句话总结

提出 **温度解耦策略（temperature decoupling gambit）**，证明在熵正则化强化学习中，通过解耦评估温度和行为温度，可以在温度趋于零时保证策略和回报分布收敛到一个可解释的、保持多样性的最优策略。

## 研究背景与动机

**领域现状**：标准RL中存在多个最优策略，策略优化方法虽能收敛到某个最优策略，但无法控制学到的是哪一个——不同最优策略可能访问不同状态、执行不同动作、获得不同回报分布。

**现有痛点**：熵正则化RL（ERL）通过对策略加KL散度惩罚来诱导唯一性，对每个正温度 $\tau$ 存在唯一最优策略 $\pi^{\tau,\star}$。但当 $\tau \to 0$（试图恢复RL最优性）时，策略的收敛性在非表格（连续）MDP中是未知的。

**核心矛盾**：ERL为每个 $\tau > 0$ 给出唯一策略，但该策略对RL是次优的；当 $\tau \to 0$ 试图恢复RL最优性时，我们重新陷入最优策略不确定的困境。

**本文目标**：(1) 提供一种在 $\tau \to 0$ 极限下保证策略收敛的方案；(2) 刻画收敛到的极限策略的性质；(3) 建立分布式RL中首个收敛的最优回报分布估计算法。

**切入角度**：借鉴国际象棋"弃子战术"的思路——短期牺牲 $\tau$-ERL的最优性，换取长期的收敛保证。

**核心 idea**：用一个比评估温度 $\sigma$ 更大的行为温度 $\tau$（要求 $\sigma/\tau \to 0$）构造 Boltzmann-Gibbs 策略，保证收敛到"最优性过滤参考策略"。

## 方法详解

### 整体框架

1. 证明 $\tau$-ERL的最优Q函数 $q_\tau^\star \to q_{\text{ref}}^\star$（参考最优值函数），但策略不一定收敛
2. 引入温度解耦策略 $\pi^{\tau,\sigma} := \mathcal{G}_\tau q_\sigma^\star$（在温度 $\tau$ 下用 $\sigma$-最优Q函数的BG策略）
3. 证明 $\pi^{\tau,\sigma} \to \pi^{\text{ref},\star}$（最优性过滤参考策略）
4. 扩展到分布式RL，建立收敛的回报分布估计

### 关键设计 1：Bellman 参考最优算子

- **功能**：定义一个新的Bellman算子，其不动点捕获了ERL在 $\tau \to 0$ 时实际能达到的最优值。
- **核心思路**：
$$(\mathcal{B}_{\text{ref}}^\star q)(x,a) := r(x,a) + \gamma \int \text{ess}\sup_{\pi_x^{\text{ref}}} q(x', \cdot)\, dP_{x,a}(x')$$
  用 $\text{ess}\sup$（本质上确界）替代 $\sup$，只在参考策略支撑集上取最大值。
- **关键结论（Theorem 3）**：$q_\tau^\star \to q_{\text{ref}}^\star$ 单调收敛当 $\tau \to 0$。
- **设计动机**：当最优动作是零测度集时（连续动作空间），$q_{\text{ref}}^\star < q^\star$，此时**任何**参数化策略（高斯、扩散等）都以概率1无法采样到最优动作。$q_{\text{ref}}^\star$ 是实际可达的"天花板"。

### 关键设计 2：温度解耦策略（Definition 3.4）

- **功能**：构造一种在 $\tau \to 0$ 时具有收敛保证的策略。
- **核心思路**：给定温度 $\tau > 0$，选择 $\sigma = \sigma(\tau)$（如 $\sigma = \tau^2$），令 $\pi^{\tau,\sigma} := \mathcal{G}_\tau q_\sigma^\star$。要求 $\sigma/\tau \to 0$ 当 $\tau \to 0$。
- **关键不等式**：
$$\lim_{\tau \to 0} \sup_x \|(\mathcal{G}_\tau q_\sigma^\star)_x - (\mathcal{G}_\tau q_{\text{ref}}^\star)_x\|_{\text{TV}} \lesssim -\lim_{\tau \to 0} \frac{\sigma}{\tau} \log p_{\text{ref}}$$
  $\sigma/\tau \to 0$ 的条件确保右端为零。
- **设计动机**：正常ERL中BG策略的log概率被 $\tau^{-1}$ 放大，使得 $\|q_{\text{ref}}^\star - q_\tau^\star\| \sim O(\tau)$ 的误差在策略空间中不消失。解耦后，用更慢趋零的 $\tau$ 作为平滑参数，更快趋零的 $\sigma$ 作为Q估计精度。

### 关键设计 3：最优性过滤参考策略（Definition 3.5）

- **功能**：刻画温度解耦策略收敛到的极限策略。
- **核心思路**：
$$\pi_x^{\text{ref},\star} \propto \pi_x^{\text{ref}} \odot \chi_{\mathsf{N}^\star_{\text{ref}}(x)}, \quad \mathsf{N}^\star_{\text{ref}}(x) := \{a : q^\star(x,a) = \text{ess}\sup_{\pi_x^{\text{ref}}} q^\star(x,\cdot)\}$$
  即参考策略限制在最优动作集合上的归一化版本。
- **关键性质**：当 $\pi^{\text{ref}}$ 为均匀策略时，$\pi^{\text{ref},\star}$ 在所有最优动作上均匀分布——这是"最多样"的最优策略。
- **与正常ERL极限的区别**：即使在表格MDP中 $\pi^{\tau,\star}$ 也收敛（Theorem 2.2），但极限不同！正常极限通过最小化长期占据度量的KL散度来选择最优策略，倾向于状态间的平均多样性；解耦极限 $\pi^{\text{ref},\star}$ 最大化每个状态的动作多样性。

### 关键设计 4：分布式ERL（DERL）

- **软分布Bellman评估算子**（Definition 4.1）：
$$(\mathcal{T}_\tau^\pi \bar{\zeta})_{x,a} := (\mathtt{b}_{r(x,a),\gamma} \circ \mathtt{proj}^{\mathbb{R}} - \gamma\tau \mathtt{kl}[\pi] \circ \mathtt{proj}^{\mathsf{X}})_\# (\bar{\zeta}_{\_,\_} \otimes \check{P}_{x,a}^\pi)$$
  是 $\gamma$-压缩映射（Theorem 4.2），有唯一不动点 $\bar{\zeta}^{\pi,\tau}$。
- **软分布最优算子**（Definition 4.2）：用当前分布的均值构造BG策略再评估，$\mathcal{Q}\mathcal{T}_\tau^\star = \mathcal{B}_\tau^\star \mathcal{Q}$（交换性引理）。
- **收敛速率**（Theorem 4.2）：$\bar{d}_p(\bar{\zeta}^n, \bar{\zeta}^{\tau,\star}) \leq C_{p,\tau,\gamma} n \gamma^{n/p}$——迭代收敛！与标准分布式RL中 $\mathcal{T}^\star$ 的迭代不收敛形成鲜明对比。

### 损失函数

理论工作，无显式损失函数。算法基于动态规划迭代。

## 实验关键数据

### 数值演示 1：三状态MDP（Figure 3.1）

- 三个状态 $\{x_0, x_1, x_2\}$，两个动作（蓝、绿），$\gamma = 0.9$，$\pi^{\text{ref}} = \mathcal{U}(\mathsf{A})$
- **标准ERL** $\hat{\pi}^{\tau,\star}$：在 $x_0$ 处收敛到 $\delta_{a_1}$（退化为确定性策略）
- **温度解耦** $\hat{\pi}^{\tau,\sigma}$：在 $x_0$ 处收敛到 $\mathcal{U}(\{a_1, a_2\})$（保持多样性）
- 两者在 $x_1, x_2$ 处行为相同，差异仅在 $x_0$——验证了理论预测

### 数值演示 2：回报分布收敛（Figure 4.3, 4.4）

- 说明性MDP，从 $x_1$ 出发蓝色动作给确定性回报 $2\gamma/(1-\gamma)$，绿色动作给 Bernoulli 回报 $4\gamma/(1-\gamma) \cdot \text{Bernoulli}(1/2)$
- **温度解耦估计** $\hat{\eta}^{\tau,\sigma}$：随 $\tau \to 0$ 收敛到 $\pi^{\text{ref},\star}$ 的回报分布（包含两个模式）
- **标准ERL估计** $\hat{\eta}^{\tau,\star}$：也收敛，但到不同的回报分布
- **分布最优算子稳定性**：$\mathcal{T}_\tau^\star$ 的迭代稳定收敛（图4.1下行），而标准 $\mathcal{T}^\star$ 的迭代发散（图4.1上行）
- **32位精度影响**：64位精度下两种方法都收敛；32位精度下温度解耦更稳定

### 关键发现

- 温度解耦策略和标准ERL策略收敛到**不同的**最优策略，即使在相同的表格MDP中
- 前者保持逐状态的动作多样性（"不歧视最优动作"），后者优化长期占据度量的KL散度
- 分布式Bellman最优算子 $\mathcal{T}^\star$ 的不收敛问题可通过熵正则化完美解决

## 亮点与洞察

1. **优雅的理论贡献**：温度解耦策略是一个精妙的构造——牺牲ERL短期最优性换取RL长期收敛保证，正如象棋弃子
2. **$\pi^{\text{ref},\star}$ 的可解释性**：极限策略有清晰的数学定义（参考策略在最优动作集上的投影），且在均匀参考下最大化逐状态多样性
3. **连接三大领域**：将策略优化（ERL）、分布式RL（DRL）、信息论（KL正则化）统一在一个收敛理论框架中
4. **首个收敛的最优回报分布估计**：Theorem 4.2是DRL中的里程碑结果——此前分布式Bellman最优算子的迭代不收敛
5. **BG策略的TV距离界**（Theorem 3.3）：独立有价值的技术工具

## 局限与展望

1. **定性收敛**：策略收敛是TV/弱收敛意义下的，缺乏具体的有限温度近似误差界——不知道具体需要多小的 $\tau$
2. **Assumption 3.2 的可验证性**：连续动作空间中，确保参考策略在最优动作邻域有下界质量并非trivial
3. **缺乏深度RL实验**：数值演示仅限于玩具MDP，未在高维任务（如MuJoCo）上验证实际效果
4. **计算代价**：温度解耦需要估计 $q_\sigma^\star$（$\sigma$ 可能非常小），在实践中可能需要高精度算术
5. **与实际算法的差距**：如何将温度解耦嵌入SAC等实际ERL算法是重要的开放问题

## 相关工作与启发

- **与 SAC (Haarnoja et al., 2018) 的关系**：SAC使用固定温度的MaxEnt RL，温度解耦策略为其自适应温度方案提供了新的理论视角
- **与分布式RL (Bellemare et al., 2017) 的互补**：标准DRL中控制设定下分布不收敛是已知的open problem，本文通过ERL给出了首个解决方案
- **对安全RL的启发**：收敛的回报分布估计对于风险敏感应用至关重要——我们第一次能准确估计某个确定的最优策略的回报分布
- **对探索的启发**：$\pi^{\text{ref},\star}$ 在所有最优动作上保持均匀分布，天然有利于探索性能
- **Ziegler et al. (2019) / KL-constrained RLHF**：温度解耦的思路可能对RLHF中KL惩罚的温度退火有指导意义

## 评分

⭐⭐⭐⭐⭐ (5/5)

理论深度极高的工作，解决了ERL和DRL中长期存在的开放问题。温度解耦策略的构造既简洁又深刻，$\pi^{\text{ref},\star}$ 的刻画提供了"最多样最优策略"的精确数学定义。主要不足是缺乏大规模实验验证和定量收敛率。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Distributional Inverse Reinforcement Learning](../../ICML2026/reinforcement_learning/distributional_inverse_reinforcement_learning.md)
- [\[NeurIPS 2025\] Last Iterate Convergence in Monotone Mean Field Games](last_iterate_convergence_in_monotone_mean_field_games.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](../../ICLR2026/reinforcement_learning/entropy-preserving_reinforcement_learning.md)
- [\[ICML 2025\] On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning](../../ICML2025/reinforcement_learning/on_the_dynamic_regret_of_following_the_regularized_leader_optimism_with_history_.md)

</div>

<!-- RELATED:END -->
