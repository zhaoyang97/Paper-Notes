---
title: >-
  [论文解读] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization
description: >-
  [ICML 2025][强化学习][robust offline RL] 提出 d-rectangular linear RRMDP (d-RRMDP) 框架，将潜在线性结构同时引入转移核和 f-散度正则化，设计 R2PVI 算法在离线数据下学习鲁棒策略，证明了 instance-dependent 的次优性上界，并通过信息论下界验证算法接近最优。
tags:
  - "ICML 2025"
  - "强化学习"
  - "robust offline RL"
  - "f-divergence regularization"
  - "linear structure"
  - "d-rectangular"
  - "pessimistic value iteration"
---

# Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization

**会议**: ICML 2025  
**arXiv**: [2411.18612](https://arxiv.org/abs/2411.18612)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: robust offline RL, f-divergence regularization, linear structure, d-rectangular, pessimistic value iteration

## 一句话总结

提出 d-rectangular linear RRMDP (d-RRMDP) 框架，将潜在线性结构同时引入转移核和 f-散度正则化，设计 R2PVI 算法在离线数据下学习鲁棒策略，证明了 instance-dependent 的次优性上界，并通过信息论下界验证算法接近最优。

## 研究背景与动机

**领域现状**：离线强化学习中，agent 从预收集的数据集中学习策略，面临训练环境与部署环境可能存在动态差异的问题。Robust Regularized MDP (RRMDP) 框架通过在值函数中对转移动态添加正则化项来学习对环境扰动鲁棒的策略，已成为处理分布偏移的重要工具。

**现有痛点**：已有 RRMDP 方法大多使用无结构 (unstructured) 正则化——对转移核施加整体性的 f-散度约束，允许状态转移在所有维度上同时发生最坏情况的偏移。这种做法虽然提供了最强的鲁棒性保证，但往往导致策略过度保守：在不现实的极端转移场景下最优化，进而在正常部署环境中性能大幅下降。

**核心矛盾**：鲁棒性与保守性之间的根本 trade-off。无结构正则化考虑了所有可能的扰动方向（包括现实中不可能发生的联合极端扰动），导致策略过于悲观。但如果完全放弃鲁棒性，策略又无法应对实际中必然存在的环境差异。

**本文目标** 设计一种结构化的正则化方式，让鲁棒性约束更符合实际场景中动态变化的模式——只考虑"合理的"扰动方向，而非所有理论上可能的扰动。具体分解为三个子问题：(1) 如何将线性结构引入 RRMDP？(2) 如何在此框架下设计高效的离线算法？(3) 算法的样本复杂度是否接近最优？

**切入角度**：作者观察到线性 MDP 假设下，转移核天然具有 $d$ 维的分解结构（$P_h(s'|s,a) = \langle \phi(s,a), \mu_h(s') \rangle$），可以自然地将 f-散度正则化也按维度分解——即 d-rectangular 结构。这种分解使得每个潜在维度上独立施加约束，避免了跨维度的联合极端扰动。

**核心 idea**：在线性 MDP 的潜在空间中施加维度分解的 f-散度正则化，使得鲁棒策略只需对付每个维度上的独立扰动而非联合扰动，从而在保持鲁棒性的同时大幅降低保守性。

## 方法详解

### 整体框架

输入是离线数据集（从名义环境收集的 $(s, a, r, s')$ 转移元组）和 d-RRMDP 的定义（包含特征映射 $\phi$、正则化半径 $\rho$ 和 f-散度类型）。R2PVI 算法通过 $H$ 轮值迭代后向推导鲁棒值函数，每一步利用线性回归估计值函数参数，并加入悲观惩罚项处理数据覆盖不足的区域。输出为鲁棒策略 $\hat\pi$。

### 关键设计

1. **d-Rectangular Linear RRMDP**:

    - 功能：定义结构化的鲁棒 MDP 框架，使得环境扰动的不确定性集合具有可控的结构
    - 核心思路：在线性 MDP 中，转移核可分解为 $P_h(\cdot|s,a) = \Phi(s,a)^\top \mu_h(\cdot)$，其中 $\Phi \in \mathbb{R}^d$ 是特征向量。d-rectangular 正则化对每个维度 $i \in [d]$ 独立施加 f-散度约束：$D_f(\xi^{(i)}_h \| \mu^{(i)}_h) \leq \rho$ 对所有 $i$ 成立。这意味着每个潜在维度的测度偏移是独立约束的，而非对联合分布施加单一约束
    - 设计动机：(s,a)-rectangular 的无结构正则化允许每个状态-动作对有独立的最坏扰动，导致不确定性集合过大。d-rectangular 利用线性结构将不确定性限制在 $d$ 维潜在空间中，大幅缩小了需要考虑的扰动范围

2. **R2PVI (Robust Regularized Pessimistic Value Iteration)**:

    - 功能：在 d-RRMDP 框架下从离线数据学习鲁棒策略
    - 核心思路：算法从时间步 $H$ 到 $1$ 后向迭代。在每步 $h$：(a) 用线性回归从数据估计鲁棒 Q 函数的参数 $\hat w_h = \Lambda_h^{-1} \sum_{i} \phi(s_h^i, a_h^i) [r_h^i + \hat V_{h+1}^{\text{rob}}(s_{h+1}^i)]$，其中 $\Lambda_h = \sum_i \phi \phi^\top + \lambda I$ 是正则化 Gram 矩阵；(b) 利用 f-散度的凸对偶形式计算鲁棒值函数 $\hat V_h^{\text{rob}}$——对偶化将 min-max 问题转化为凸优化；(c) 加入悲观惩罚 $\beta \|\phi(s,a)\|_{\Lambda_h^{-1}}$ 确保对数据覆盖不足区域的值函数偏保守
    - 设计动机：线性函数逼近使得在大状态空间中可行；悲观惩罚结合鲁棒正则化形成双重保护——前者处理离线数据的分布偏移，后者处理环境动态的偏移

3. **Instance-Dependent 理论分析**:

    - 功能：为 R2PVI 的次优性提供精细的上界，并通过信息论下界证明接近最优
    - 核心思路：上界表达为 $\text{SubOpt}(\hat\pi) \leq \tilde O(\sum_h \sqrt{d / n} \cdot C_h^{\text{rob}})$，其中 $C_h^{\text{rob}}$ 刻画了数据集覆盖最优鲁棒策略下"鲁棒可达"状态-动作空间的程度。这个覆盖条件比标准离线 RL 更强——不仅要覆盖名义环境下的最优路径，还要覆盖所有鲁棒可容许转移下的路径
    - 设计动机：instance-dependent 分析揭示了算法性能与问题"硬度"的精细关系——简单问题上自动获得更紧的界

### 损失函数

R2PVI 的核心计算涉及鲁棒 Bellman 方程的求解。对于 f-散度（如 KL 散度或 $\chi^2$ 散度），鲁棒值函数通过对偶形式计算：$V_h^{\text{rob}}(s) = \max_a [\hat Q_h(s,a) - \beta \|\phi(s,a)\|_{\Lambda_h^{-1}} - \rho \cdot \text{DualPenalty}]$，其中 DualPenalty 依赖于具体的 f-散度类型——KL 散度时为 log-sum-exp 形式，$\chi^2$ 散度时为二次惩罚形式。

## 实验关键数据

### 合成 MDP 实验：结构化 vs 无结构正则化

| 方法 | 次优性 Gap | 计算效率 | 保守性 |
|------|-----------|---------|--------|
| 无结构 RRMDP | 较高 | 慢 | 过度保守 |
| d-RRMDP (KL 散度) | 显著更低 | 快 | 适度 |
| d-RRMDP ($\chi^2$ 散度) | 显著更低 | 快 | 适度 |
| 非鲁棒基线 | 最低（名义环境）| 最快 | 无鲁棒性 |

### 算法关键理论指标对比

| 指标 | R2PVI (本文) | 已有无结构方法 | 非鲁棒离线 RL |
|------|------------|-------------|-------------|
| 不确定性维度 | $d$ (潜在空间) | $|\mathcal{S}|$ (状态空间) | — |
| 样本复杂度阶数 | $\tilde O(d H^2 / \epsilon^2)$ | $\tilde O(|\mathcal{S}| H^2 / \epsilon^2)$ | $\tilde O(d H^2 / \epsilon^2)$ |
| 是否接近下界 | 是 | 未知 | 是 |
| 处理动态偏移 | 是（结构化） | 是（无结构） | 否 |

### 关键发现

1. d-rectangular 结构利用线性 MDP 的潜在低维性，将鲁棒性约束的维度从 $|\mathcal{S}|$（可能很大）降低到 $d$（特征维度），在保持鲁棒性的同时大幅降低了保守性
2. 信息论下界证明 R2PVI 的样本复杂度阶数是 minimax 最优的——无法在不增加假设的情况下进一步改进
3. 不同类型的 f-散度（KL 和 $\chi^2$）在实验中表现相近，说明框架对散度选择不敏感
4. 正则化强度 $\rho$ 控制鲁棒性-性能 trade-off：$\rho$ 过大导致过度保守，$\rho$ 过小则无法抵御动态偏移

## 亮点与洞察

1. **将问题结构注入鲁棒性约束**：利用线性 MDP 的低秩分解自然导出 d-rectangular 结构——不是人为设计的约束，而是数据生成过程的内在结构
2. **理论完整性**：上界和下界同时给出，确认了算法的最优性——这在鲁棒离线 RL 的线性函数逼近设定下是首次
3. **f-散度框架的通用性**：统一覆盖 KL 散度、$\chi^2$ 散度、总变差距离等多种距离度量，为实际选择提供了灵活性
4. **凸对偶化的计算优势**：将鲁棒 Bellman 方程中的 min-max 问题转化为凸优化，使得 R2PVI 在计算效率上优于需要显式求解对抗博弈的方法

## 局限性

- 线性 MDP 假设在复杂实际任务中可能不成立——非线性函数逼近（如神经网络）的扩展是重要开放问题
- 实验仅在合成 MDP 上验证，缺乏 MuJoCo、D4RL 等标准离线 RL benchmark 的实证
- 正则化半径 $\rho$ 需要先验知识设定——自适应选择 $\rho$ 的方法未探索
- d-rectangular 结构虽然比无结构更合理，但仍可能与实际环境扰动的真实模式不完全匹配

## 相关工作与启发

- 经典 RMDP (Nilim & El Ghaoui, 2005) 和 RRMDP (Panaganti & Kalathil, 2022) 奠定了鲁棒 MDP 理论基础
- 离线 RL 悲观原则 (Jin et al., 2021) 与鲁棒正则化形成互补——二者分别处理数据分布偏移和环境动态偏移
- 启发：结构化不确定性集合的思想可推广到其他鲁棒决策问题（如分布式鲁棒优化中的因子分解不确定性集合）

## 评分

⭐⭐⭐⭐ 理论贡献扎实——d-rectangular 线性结构化正则的首次系统研究，上下界齐全证明了接近最优性。实验规模有限是主要扣分项，但作为理论工作整体质量高，是鲁棒离线 RL 函数逼近方向的重要推进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](learning_to_trust_bellman_updates_selective_state-adaptive_regularization_for_of.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ACL 2025\] Learning to Generate Structured Output with Schema Reinforcement Learning](../../ACL2025/reinforcement_learning/learning_to_generate_structured_output_with_schema_reinforcement_learning.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
