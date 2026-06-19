---
title: >-
  [论文解读] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][一般效用强化学习] 本文为一般效用强化学习（RLGU）中的策略梯度方法建立了全局最优性理论保证：在表格设定下通过新的梯度支配不等式证明了全局收敛，在大规模状态-动作空间下提出基于最大似然估计（MLE）的占据度量近似算法 PG-OMA，样本复杂度仅依赖函数近似类的维度 $m$ 而非状态-动作空间大小。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "一般效用强化学习"
  - "策略梯度"
  - "梯度支配不等式"
  - "占据度量估计"
  - "最大似然估计"
---

# On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2410.04108](https://arxiv.org/abs/2410.04108)  
**代码**: 无  
**领域**: 强化学习理论  
**关键词**: 一般效用强化学习, 策略梯度, 梯度支配不等式, 占据度量估计, 最大似然估计

## 一句话总结

本文为一般效用强化学习（RLGU）中的策略梯度方法建立了全局最优性理论保证：在表格设定下通过新的梯度支配不等式证明了全局收敛，在大规模状态-动作空间下提出基于最大似然估计（MLE）的占据度量近似算法 PG-OMA，样本复杂度仅依赖函数近似类的维度 $m$ 而非状态-动作空间大小。

## 研究背景与动机

**领域现状**：一般效用强化学习（RLGU，又称凸RL）提供了一个统一框架，可以涵盖标准期望回报 RL 之外的多种问题，包括模仿学习、纯探索、安全 RL、技能发现和实验设计等。标准 RL 的目标是占据度量的线性函数，而 RLGU 将目标推广为占据度量的一般（可能非线性）泛函。

**现有痛点**：尽管策略梯度方法在标准 RL 中已有成熟的全局最优性理论（如 Agarwal 等人 2021 证明的梯度支配不等式），但这些结果无法直接推广到 RLGU。现有 RLGU 工作主要局限于两个方面：（1）表格设定下的全局最优性证明依赖"隐藏凸性"技术，与标准 RL 的策略梯度分析缺乏联系；（2）绝大多数算法仅适用于表格设定，使用基于计数的 Monte Carlo 估计器来估计占据度量，无法扩展到大规模状态-动作空间。

**核心矛盾**：RLGU 中的目标函数关于策略参数是非凹的，且需要估计未知的占据度量来计算"伪奖励"。在大规模空间中，逐状态-动作对计数的方式计算和存储代价过高，而已有的均方误差（MSE）近似方法在理论保证上存在对状态空间大小的依赖性问题。

**本文目标**（1）能否将标准 RL 中的梯度支配结构推广到 RLGU？（2）能否设计在大规模空间中样本复杂度不依赖状态-动作空间大小的策略梯度算法？

**切入角度**：作者注意到 RLGU 的策略梯度可以表示为标准 RL 策略梯度在"伪奖励" $\nabla_\lambda F(\lambda(\theta))$ 处的求值（公式5），这一关键身份将 RLGU 梯度与标准 RL 梯度联系起来。利用这一洞察，结合凸性假设，可以将标准 RL 的梯度支配推广到 RLGU。在扩展性方面，采用 MLE 而非 MSE 来估计占据度量，因为 MLE 的全变差误差界只依赖参数维度。

**核心 idea**：通过将 RLGU 策略梯度还原为标准 RL 梯度在伪奖励处的求值来建立梯度支配不等式，并用 MLE 近似占据度量使算法可扩展到大空间。

## 方法详解

### 整体框架

考虑折扣 MDP $(S, A, P, F, \rho, \gamma)$，其中 $F$ 是定义在占据度量空间上的一般效用函数。目标是 $\max_\theta F(\lambda^{\pi_\theta})$。算法分两个层次：表格设定下建立理论结构性质（梯度支配），大规模设定下设计实用的 Actor-Critic 风格算法 PG-OMA。

### 关键设计

1. **RLGU 梯度支配不等式（表格设定）**:

    - 功能：证明 RLGU 目标在直接策略参数化下满足梯度支配性质，从而任何驻点都是全局最优解
    - 核心思路：利用链式法则将 RLGU 梯度分解为 $\nabla_\theta F(\lambda(\theta)) = [\nabla_\theta \lambda(\theta)]^T \nabla_\lambda F(\lambda(\theta))$，发现右端恰好等于标准 RL 策略梯度在伪奖励 $r_\theta = \nabla_\lambda F(\lambda(\theta))$ 处的求值。然后利用标准 RL 的梯度支配结果（Agarwal 2021 Lemma 4）对伪奖励应用，再结合 $F$ 的凹性将值函数差异转化为效用差异：$V^{\pi^*(r_\theta)}(r_\theta) - V^{\pi_\theta}(r_\theta) \geq \langle r_\theta, \lambda^{\pi^*} - \lambda^{\pi_\theta} \rangle \geq F(\lambda(\theta^*)) - F(\lambda(\theta))$
    - 设计动机：将 RLGU 的优化结构与标准 RL 的已有理论直接对接，打开了在 RLGU 框架中分析 softmax 等参数化的通道

2. **基于 MLE 的占据度量近似（PG-OMA 算法）**:

    - 功能：在大规模状态-动作空间中可扩展地估计占据度量
    - 核心思路：将归一化状态占据度量 $d^{\pi_\theta}$ 视为概率分布，在参数化分布族 $\Lambda = \{p_\omega : \omega \in \Omega \subseteq \mathbb{R}^m\}$ 中通过最大似然估计来近似：$\omega^* = \arg\max_\omega \frac{1}{n}\sum_{i=1}^n \log p_\omega(s_i)$。关键理论结果是 MLE 的全变差误差界 $\|\hat\lambda - \lambda\|_1 \leq O(\sqrt{m/n})$ 仅依赖近似类维度 $m$，不依赖状态-动作空间大小
    - 设计动机：与 MSE 方法相比，MLE 天然适合概率分布估计，不受空间大小影响。作者通过一个简单但深刻的反例说明：当真实分布均匀时，MSE 的误差在大空间中趋近于零变得无法区分好坏估计，而 MLE 的 TV 误差始终有效

3. **PG-OMA 算法的两阶段迭代**:

    - 功能：在每次迭代中先估计占据度量（Critic），再更新策略参数（Actor）
    - 核心思路：每轮迭代分两步：（i）用 MLE 从当前策略采样的状态来近似占据度量 $\hat\lambda_t$，计算伪奖励 $\hat r_t = \nabla_\lambda F(\hat\lambda_t)$；（ii）用 REINFORCE 估计器在伪奖励 $\hat r_t$ 下做随机策略梯度上升更新。整个过程无需估计转移核，是 model-free 的
    - 设计动机：将占据度量估计解耦为独立的统计学习子问题，利用 MLE 的统计效率；伪奖励只需在当前轨迹访问的状态-动作对上计算（Remark 4），进一步节省空间

### 损失函数 / 训练策略

策略梯度通过 REINFORCE 估计器近似，步长满足 $\alpha_t \leq 1/(2L_\theta)$。对非凹效用，保证一阶驻点；对凹效用，结合隐藏凸性技术和策略过参数化假设（Assumption 4.3）给出末次迭代的全局最优性。总样本复杂度中关键参数为函数近似维度 $m$、精度 $\epsilon$ 和折扣因子 $\gamma$。

## 实验关键数据

### 主实验

本文为纯理论工作，以下为与最相关工作的理论对比：

| 方法 | 一阶平稳复杂度 | 全局最优复杂度 | 超越表格 | 无 $|S \times A|$ 依赖 |
|------|--------------|--------------|---------|----------------------|
| Zhang et al. 2020 | $\tilde{O}(\epsilon^{-2})$* | $\tilde{O}(\epsilon^{-1})$* | ✘ | ✘ |
| Zhang et al. 2021 | $\tilde{O}(\epsilon^{-3})$ | $\tilde{O}(\epsilon^{-2})$ | ✘ | ✘ |
| Barakat et al. 2023 (sec.5) | $\tilde{O}(\epsilon^{-4})$ | ✘ | ✓ | ✘ |
| **本文 PG-OMA** | $\tilde{O}(m\epsilon^{-4})$ | $\tilde{O}(m\epsilon^{-4})$ | ✓ | ✓ |

*确定性设定，仅报告迭代数

### 消融实验

| 对比维度 | 本文 | Barakat et al. 2023 | Huang & Jiang 2024 |
|---------|------|--------------------|--------------------|
| 全局收敛保证 | ✓ (凹效用) | ✘ (仅一阶) | ✘ (仅一阶) |
| 占据度量估计 | MLE | MSE | MLE+递归回归 |
| 无 $|S|$ 依赖 | ✓ | ✘ (隐含 $1/\rho_{min}$) | 未明确 |
| last-iterate 保证 | ✓ | ✓ | ✘ (best-iterate) |

### 关键发现

- MLE 估计占据度量在理论上严格优于 MSE：MSE 在大空间中无法检测分布差异（均匀分布反例），而 MLE 的 TV bound 与空间大小无关
- 梯度支配不等式中的分布不匹配系数依赖伪奖励 $\nabla_\lambda F(\lambda(\theta))$，当 $F$ 为线性时退化为标准 RL 的常数系数
- 证明中仅在一个点 $\lambda^{\pi^*}$ 用到了凹性，暗示结果可推广到更弱的局部凹性条件

## 亮点与洞察

- **策略梯度-伪奖励等价性**是核心贡献：$\nabla_\theta F(\lambda(\theta))$ 恰好等于标准策略梯度在 $r = \nabla_\lambda F(\lambda(\theta))$ 处的求值。这将标准 RL 的全套分析工具（梯度支配、方差缩减等）引入 RLGU，具有方法论意义
- **MLE vs MSE 的可扩展性分析**是独立的理论贡献：反例清晰展示了 MSE 在概率分布估计中的根本缺陷——均匀分布上估计非均匀分布时，MSE 损失为 $O(1/|X|^2)$ 随空间增大趋零无法区分
- 证明技术可推广到 softmax 策略参数化，作者明确指出可利用 Mei et al. 2020 的 Lemma 8，为后续工作铺平道路

## 局限与展望

- 策略过参数化假设（Assumption 4.3）在实际神经网络策略下难以验证
- 理论分析限于有限状态-动作空间，连续空间的推广是主要开放问题
- 凹效用的 $\tilde{O}(m\epsilon^{-4})$ 复杂度与表格最优 $\tilde{O}(\epsilon^{-2})$ 有两阶差距
- 纯理论工作缺乏任何数值实验验证
- 假设 MLE 子问题可解到全局最优，实际中是非凸优化

## 相关工作与启发

- **vs Zhang et al. 2021**: 用隐藏凸性直接证明全局最优，无法连接到标准 RL 分析。本文通过梯度支配建立了联系
- **vs Barakat et al. 2023**: MSE 估计导致 $|S|$ 依赖且仅一阶保证。本文 MLE 方案彻底消除空间依赖
- **vs Huang & Jiang 2024**: 需要额外估计对数梯度占据度量，本文仅需一步 MLE

## 评分

- 新颖性: ⭐⭐⭐⭐ 梯度支配推广和 MLE 替代方案有显著理论创新
- 实验充分度: ⭐⭐ 纯理论无实验
- 写作质量: ⭐⭐⭐⭐ 结构严谨，前人对比详尽
- 价值: ⭐⭐⭐⭐ 为 RLGU 可扩展算法奠定理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[ICML 2026\] ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation](../../ICML2026/reinforcement_learning/prorl_effective_reinforcement_learning_for_proactive_recommendation_via_rectifie.md)
- [\[ICML 2026\] Reinforcement Learning for Reachability: Guaranteeing Asymptotic Optimality](../../ICML2026/reinforcement_learning/reinforcement_learning_for_reachability_guaranteeing_asymptotic_optimality.md)
- [\[NeurIPS 2025\] Near-Optimal Quantum Algorithms for Computing (Coarse) Correlated Equilibria of General-Sum Games](near-optimal_quantum_algorithms_for_computing_coarse_correlated_equilibria_of_ge.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)

</div>

<!-- RELATED:END -->
