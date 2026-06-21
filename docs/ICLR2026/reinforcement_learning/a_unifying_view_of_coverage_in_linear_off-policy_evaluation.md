---
title: >-
  [论文解读] A Unifying View of Coverage in Linear Off-Policy Evaluation
description: >-
  [ICLR 2026][强化学习][离策略评估] 提出了一种新的覆盖性参数——特征-动态覆盖：（feature-dynamics coverage），通过工具变量视角对经典算法 LSTDQ 进行新颖的有限样本分析，统一了线性离策略评估中各种不同覆盖性定义，解决了该领域长期存在的碎片化问题。 离策略评估（Off-Policy…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "离策略评估"
  - "覆盖性"
  - "线性函数逼近"
  - "LSTDQ"
  - "特征-动态覆盖"
---

# A Unifying View of Coverage in Linear Off-Policy Evaluation

**会议**: ICLR 2026  
**arXiv**: [2601.19030](https://arxiv.org/abs/2601.19030)  
**代码**: 无  
**领域**: 强化学习 / 离策略评估  
**关键词**: 离策略评估, 覆盖性, 线性函数逼近, LSTDQ, 特征-动态覆盖

## 一句话总结
提出了一种新的覆盖性参数——**特征-动态覆盖**（feature-dynamics coverage），通过工具变量视角对经典算法 LSTDQ 进行新颖的有限样本分析，统一了线性离策略评估中各种不同覆盖性定义，解决了该领域长期存在的碎片化问题。

## 研究背景与动机
离策略评估（Off-Policy Evaluation, OPE）是强化学习中的基础问题：给定由行为策略（behavior policy）收集的数据，评估一个不同的目标策略（target policy）的价值。这在无法进行在线交互的场景（如医疗、推荐系统）中至关重要。

在线性 OPE 的经典设置中，有限样本保证通常采用如下形式：

$$\text{评估误差} \leq \text{poly}(C^\pi, d, 1/n, \log(1/\delta))$$

其中 $d$ 是特征维度，$n$ 是样本数，$C^\pi$ 是**覆盖性参数**——描述数据分布对目标策略访问的特征空间的覆盖程度。

**核心矛盾/碎片化问题**：

在更强的假设下（如 Bellman 完备性），覆盖性参数的定义是清晰的，多种经典算法的保证也很好理解。但在**最小假设设定**（仅要求目标值函数线性可实现）下，情况变得非常混乱：
- 对于"正确的"覆盖性概念没有共识
- 不同分析使用的覆盖性定义相互矛盾，性质也不理想（如不是分布无关的、不能恢复特殊情况下的标准定义）
- 各种定义之间缺乏联系，导致理论理解碎片化

**本文的目标**：提出一个统一的覆盖性概念，在最小假设下给出紧的有限样本保证，并在更强假设下能优雅地退化为已知的标准覆盖性定义。

## 方法详解

### 整体框架
本文不提出新算法，而是为经典的 LSTDQ（Least-Squares Temporal Difference for Q-values）补一套新的有限样本理论：把它重新读成一个工具变量回归，由此自然导出新的覆盖性参数 feature-dynamics coverage，并证明在它之下的评估误差界。这套界的好处是，更强假设下它会优雅退化为人们已经熟悉的标准覆盖性，从而把过去碎片化的各种定义统一进同一个框架。

### 关键设计

**1. 工具变量视角：把 LSTDQ 读成 IV 回归，绕开内生性**

线性 OPE 的难点在于 Bellman 方程里有内生性——直接对 $Q(s,a)\approx\phi(s,a)^\top w$ 做最小二乘时，回归量 $\phi(s,a)$ 会和 TD 误差项相关，普通最小二乘因此有偏。工具变量（Instrumental Variable）正是计量经济学里处理这类问题的标准工具：引入一个与误差无关、却与内生回归量相关的"工具"，就能得到一致估计。本文的关键观察是，Bellman 方程的结构天生适配 IV 解释——当前状态-动作特征 $\phi(s,a)$ 是内生变量，而经环境转移后映射出的特征 $\mathbb{E}[\phi(s',a')]$ 恰好可以充当工具。一旦把 LSTDQ 摆进这个框架，它的统计行为就有了清晰的计量经济学解释，也直接指明了"误差到底被什么放大"。

**2. 特征-动态覆盖（Feature-Dynamics Coverage）：把"动态如何放大覆盖不足"显式化**

IV 视角带来的副产物，是一个新的覆盖性参数 $C^\pi_{\mathrm{FD}}$。它可以被理解为在一个由特征演化动态所诱导的线性系统中的覆盖性度量：不只看行为策略对当前特征 $\phi(s,a)$ 的覆盖，还看它对"特征经过环境转移后演化到的位置"的覆盖。直观上，OPE 的难度不仅取决于数据分布本身，还取决于环境动态如何沿轨迹放大覆盖的薄弱处——$C^\pi_{\mathrm{FD}}$ 把这层放大效应显式地装进了定义里。它是分布相关的，但形式自然，并且保留了一个理论上很关键的性质：在特殊情形下能塌缩回大家公认的标准覆盖性，这正是统一性的支点。

**3. 统一化与误差界：一个参数串起所有旧定义**

基于上述定义，本文给出 LSTDQ 的主误差界：评估误差随 $C^\pi_{\mathrm{FD}}$ 和特征维度 $d$ 多项式增长，随样本数 $n$ 以 $1/\sqrt{n}$ 速率下降，并以高概率成立（依赖 $\log(1/\delta)$），可写成 $\text{误差}\lesssim \mathrm{poly}(C^\pi_{\mathrm{FD}},d)\cdot n^{-1/2}\cdot\sqrt{\log(1/\delta)}$。真正的价值在于它的退化行为：在 Bellman 完备性假设下，$C^\pi_{\mathrm{FD}}$ 退化为该设定下公认的标准线性覆盖性（standard linear coverage）；在状态抽象（state abstraction，线性函数逼近的一个特例）下，它自然涵盖了聚合集中系数（aggregated concentrability）的 $\chi^2$ 版本；而在仅要求线性可实化的最小假设下，它给出比以往候选参数（如取 $C^\pi=1/\sigma_{\min}(A)$）更紧、更可解释的界。这一点很关键，因为本文同时指出 $1/\sigma_{\min}(A)$ 作为覆盖性参数有三处毛病：缺乏尺度不变性（把特征整体放缩就能任意改变它）、缺乏离策略刻画（只在受限的同策略情形下有已知的有界性解释）、且无法统一到其他分析（无法恢复状态抽象下公认的聚合集中系数）。换句话说，过去那些彼此矛盾、甚至依赖特定算法的"奇怪"覆盖性定义，原来都是 $C^\pi_{\mathrm{FD}}$ 在特殊情形下的投影。

## 实验关键数据

### 主实验
本文是理论贡献为主的工作，但包含了数值验证来支持理论结果。

| 设定 | 指标 | 主要发现 |
|------|------|----------|
| 合成MDP（线性可实化）| MSE vs n | LSTDQ的误差符合理论预测的 $1/\sqrt{n}$ 速率 |
| 合成MDP（Bellman完备）| MSE vs C^π | 误差与 feature-dynamics coverage 的关系符合理论 |
| 不同覆盖度的数据分布 | 各定义的比较 | feature-dynamics coverage 比之前定义更紧 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Bellman完备 + feature-dynamics | 误差界 | 退化为集中系数界，验证统一性 |
| 仅线性可实化 + 之前覆盖性定义 | 误差界 | 之前定义给出更松的界 |
| 仅线性可实化 + feature-dynamics | 误差界 | 本文给出更紧的界 |
| 不同维度d | 误差界 | 多项式依赖关系得到验证 |

### 关键发现
- Feature-dynamics coverage 是比之前提出的各种覆盖性定义更自然、更紧的参数
- 在 Bellman 完备性假设下，它完美退化为已知的最优覆盖性参数
- LSTDQ 算法在工具变量视角下有更清晰的统计学解释
- 之前分析中的一些"奇怪"的覆盖性定义（如依赖于特定算法的定义）是 feature-dynamics coverage 在特殊情况下的表现

## 亮点与洞察
- **理论统一的优雅性**：长期以来，线性 OPE 中存在多种看似不兼容的覆盖性定义，本文通过一个统一的概念串联起来，是"拨开迷雾见月明"式的贡献
- **工具变量视角的新颖性**：将强化学习的 OPE 问题与计量经济学的 IV 理论联系起来，开辟了新的分析工具
- **最小假设下的理解**：在仅要求线性可实化的最小假设下给出了紧的分析，此前这个设定下的理解非常有限
- **概念性贡献**：feature-dynamics coverage 的"诱导动态系统"解释非常有启发性——它暗示了 OPE 的难度不仅取决于数据分布，还取决于环境动态如何"放大"覆盖的不足

## 局限与展望
- **纯理论工作**：缺少在真实 RL 任务上的实验验证
- **集中于线性设定**：现代 RL 更多使用非线性函数逼近（如神经网络），理论框架能否扩展需要进一步研究
- **关注 OPE 而非 OPL**：离策略评估（evaluation）与离策略学习（learning）在技术上有本质区别，本文的覆盖性概念是否适用于后者
- **计算可行性**：feature-dynamics coverage 在实际中是否可以高效估计？如果不能，理论保证的实际指导意义会打折扣
- **单策略评估**：能否扩展到多策略同时评估或策略优化场景

## 相关工作与启发
- **线性 OPE 经典算法**：LSTD、LSTDQ、FQE（Fitted Q Evaluation）等——本文重新分析了最经典的 LSTDQ
- **覆盖性/集中系数**：集中系数（concentrability coefficient）是 OPE 理论的核心概念，本文给出了其在一般设定下的正确推广
- **工具变量**：经济学中用于处理内生性问题，本文将这一思想引入 RL 理论
- **计量经济学与 RL 的交叉**：近年来越来越多的工作从因果推断视角理解 RL，本文是这一趋势的重要贡献
- **启发**：是否可以利用 feature-dynamics coverage 来设计自适应的数据收集策略，使得 OPE 的效率最大化？

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Unifying View of Linear Function Approximation in Off-Policy RL Through Matrix Splitting and Preconditioning](../../NeurIPS2025/reinforcement_learning/a_unifying_view_of_linear_function_approximation_in_offpolic.md)
- [\[ICLR 2026\] Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?](is_pure_exploitation_sufficient_in_exogenous_mdps_with_linear_function_approxima.md)
- [\[ICLR 2026\] Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration](off-policy_safe_reinforcement_learning_with_cost-constrained_optimistic_explorat.md)
- [\[ICLR 2026\] Primal-Dual Policy Optimization for Linear CMDPs with Adversarial Losses](primal-dual_policy_optimization_for_linear_cmdps_with_adversarial_losses.md)
- [\[ICLR 2026\] Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](enhancing_generative_auto-bidding_with_offline_reward_evaluation_and_policy_sear.md)

</div>

<!-- RELATED:END -->
