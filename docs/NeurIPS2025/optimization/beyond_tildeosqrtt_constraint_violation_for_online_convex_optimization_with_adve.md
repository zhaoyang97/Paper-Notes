---
title: >-
  [论文解读] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints
description: >-
  [NeurIPS 2025][优化/理论][在线凸优化] 研究带对抗约束的在线凸优化 (COCO)，通过引入可调参数 $\beta$ 实现 $\tilde{O}(T^\beta)$ 遗憾与 $\tilde{O}(T^{1-\beta})$ 约束违反之间的精确权衡，突破了此前 $\tilde{O}(\sqrt{T})$ 约束违反的已知最优界。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "在线凸优化"
  - "对抗约束"
  - "约束违反"
  - "遗憾-违反权衡"
  - "安全约束"
---

# Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints

**会议**: NeurIPS 2025  
**arXiv**: [2505.06709](https://arxiv.org/abs/2505.06709)  
**代码**: 无  
**领域**: 在线优化 / 约束优化  
**关键词**: 在线凸优化, 对抗约束, 约束违反, 遗憾-违反权衡, 安全约束

## 一句话总结

研究带对抗约束的在线凸优化 (COCO)，通过引入可调参数 $\beta$ 实现 $\tilde{O}(T^\beta)$ 遗憾与 $\tilde{O}(T^{1-\beta})$ 约束违反之间的精确权衡，突破了此前 $\tilde{O}(\sqrt{T})$ 约束违反的已知最优界。

## 研究背景与动机

**问题设定**：在带对抗约束的在线凸优化 (Constrained Online Convex Optimization, COCO) 中，每一轮：
1. 学习者从凸决策集中选择一个动作
2. 对手揭示一个凸代价函数和一个凸约束函数
3. 学习者的目标是最小化累积遗憾 (regret) 和累积约束违反 (CCV)

**已有成果**：此前最优策略可达 $O(\sqrt{T})$ 遗憾和 $\tilde{O}(\sqrt{T})$ CCV。

**核心动机**：在安全关键应用中（如自动驾驶、医疗决策），满足安全约束是不可协商的。因此，以遗憾为代价换取更小的约束违反具有重要实际意义。现有 $\tilde{O}(\sqrt{T})$ 的 CCV 界意味着即使经过大量轮次，约束仍可能被显著违反，这在安全场景中不可接受。

**本文贡献**：提出可调参数 $\beta \in [0, 1]$ 实现遗憾与 CCV 之间的平滑权衡：
- $\beta = 1/2$：退化为已知最优的 $O(\sqrt{T})$ 遗憾和 $\tilde{O}(\sqrt{T})$ CCV
- $\beta \to 1$：CCV 趋近于常数，但遗憾接近线性增长
- 实际选择 $\beta > 1/2$ 可在遗憾可控的前提下大幅降低约束违反

## 方法详解

### 整体框架

本文采用分层策略设计。首先解决一个特殊的"受约束专家 (Constrained Expert)"问题，然后通过覆盖论证 (covering argument) 将一般的 COCO 问题规约到该特殊问题。

### 关键设计

**1. 受约束专家问题 (Constrained Expert)**：
- 决策集为概率单纯形 $\Delta_N$（$N$ 个专家上的分布）
- 代价和约束函数均为线性
- 这是 COCO 的一个重要特殊情形，也是构建一般解法的基础

**策略核心**：利用新提出的自适应小损失遗憾界 (adaptive small-loss regret bound)，设计计算高效的策略。关键思想是：当约束函数值较小时（即接近可行），算法可以更积极地优化代价；当约束违反较大时，算法转而专注于减少违反。

**遗憾界**：$O(\sqrt{T \ln N} + T^\beta)$ 遗憾和 $\tilde{O}(T^{1-\beta} \ln N)$ CCV

**2. 从受约束专家到一般 COCO 的规约**：
- 对凸决策集进行 $\epsilon$-网覆盖，得到 $N$ 个"专家"（离散化的动作）
- 在这些专家上运行受约束专家算法
- 覆盖精度 $\epsilon$ 的选择决定了离散化误差与专家数量的权衡
- 最终结果：$\tilde{O}(\sqrt{dT} + T^\beta)$ 遗憾和 $\tilde{O}(dT^{1-\beta})$ CCV，其中 $d$ 是决策集维度

**3. 光滑情形的一阶方法**：
- 附加 $M$-光滑性假设（代价和约束函数梯度 Lipschitz 连续）
- 设计计算高效的一阶策略
- 实现 $O(\sqrt{MT} + T^\beta)$ 遗憾和 $\tilde{O}(MT^{1-\beta})$ CCV

### 损失函数 / 训练策略

- **自适应学习率**：算法的学习率根据累积约束违反程度动态调整
- **小损失遗憾界**：遗憾界与学习者实际遭受的累积损失成比例，而非最坏情况的 $T$。这一性质在约束违反较小时自动给出更紧的界
- **参数 $\beta$ 的选择策略**：在实际应用中，$\beta$ 可根据对安全约束的严格程度调整。安全要求越高，$\beta$ 越大

## 实验关键数据

### 主实验：遗憾-CCV 权衡的理论界比较

| 方法 | 遗憾上界 | CCV 上界 | 参数 | 计算效率 |
|------|---------|---------|------|---------|
| 先前最优 | $O(\sqrt{T})$ | $\tilde{O}(\sqrt{T})$ | 无 | 多项式时间 |
| **本文 (一般凸)** | $\tilde{O}(\sqrt{dT} + T^\beta)$ | $\tilde{O}(dT^{1-\beta})$ | $\beta \in [0,1]$ | 通过覆盖论证 |
| **本文 (专家)** | $O(\sqrt{T\ln N} + T^\beta)$ | $\tilde{O}(T^{1-\beta}\ln N)$ | $\beta \in [0,1]$ | 计算高效 |
| **本文 ($M$-光滑)** | $O(\sqrt{MT} + T^\beta)$ | $\tilde{O}(MT^{1-\beta})$ | $\beta \in [0,1]$ | 一阶方法 |

### 消融实验：不同 $\beta$ 值的性能

| $\beta$ | 遗憾量级 | CCV 量级 | 遗憾-CCV 乘积 | 适用场景 |
|---------|---------|---------|--------------|---------|
| 0 | $\tilde{O}(\sqrt{dT})$ | $\tilde{O}(dT)$ | $\tilde{O}(d^{3/2}T^{3/2})$ | 纯遗憾最小化 |
| 1/4 | $\tilde{O}(\sqrt{dT} + T^{1/4})$ | $\tilde{O}(dT^{3/4})$ | — | 中等安全要求 |
| 1/2 | $\tilde{O}(\sqrt{dT} + \sqrt{T})$ | $\tilde{O}(d\sqrt{T})$ | $\tilde{O}(d^{3/2}T)$ | 经典设定 |
| 3/4 | $\tilde{O}(\sqrt{dT} + T^{3/4})$ | $\tilde{O}(dT^{1/4})$ | — | 高安全要求 |
| 1 | $\tilde{O}(\sqrt{dT} + T)$ | $\tilde{O}(d)$ | $\tilde{O}(d^{3/2}T)$ | 严格约束满足 |

### 关键发现

1. **Pareto 最优权衡**：遗憾和 CCV 不能同时被改善，$\beta$ 提供了在该 Pareto 前沿上的平滑移动机制
2. **维度依赖**：一般凸情形的界依赖于决策集维度 $d$，而非更大的函数族复杂度
3. **自适应小损失界的关键作用**：这一技术创新使得在不牺牲遗憾的情况下精确控制约束违反成为可能
4. **计算效率**：受约束专家算法和一阶光滑方法均具有多项式时间复杂度

## 亮点与洞察

1. **安全关键应用的理论基础**：首次在理论上明确了在线学习中遗憾与约束违反的可调权衡
2. **分层规约的优雅设计**：先解决简单的专家问题，再通过覆盖论证推广到一般凸问题，方法论上值得借鉴
3. **自适应小损失界的新技术**：该技术超越了标准的在线学习工具，可能在其他在线优化问题中有广泛应用
4. **$\beta$ 的实际指导意义**：为工程师在部署在线学习系统时提供了明确的安全-性能调节旋钮

## 局限与展望

1. **下界缺失**：尚不清楚所提出的 $\tilde{O}(\sqrt{dT} + T^\beta)$ 遗憾和 $\tilde{O}(dT^{1-\beta})$ CCV 是否为 Pareto 最优
2. **对抗假设的现实性**：完全对抗的约束设定可能过于保守，实际约束可能具有某种统计规律性
3. **凸性假设**：对非凸代价/约束函数的扩展尚未探讨
4. **覆盖论证的效率**：通过 $\epsilon$-网规约在高维决策集上可能导致计算量爆炸
5. **数值实验不足**：文中未提供具体的数值模拟以验证理论界的紧密性

## 相关工作与启发

- **在线凸优化**：Hazan (2016) 的经典框架是本文的理论基础
- **约束在线学习**：Mahdavi et al. (2012), Yu et al. (2020) 研究了 COCO 的基本结果
- **安全在线学习**：本文的 $\beta$ 调节机制为安全增强学习 (safe RL) 的理论分析提供了新视角

## 评分

| 维度 | 评分 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 5 |
| 实验充分性 | 2 |
| 写作质量 | 4 |
| 总评 | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](online_two-stage_submodular_maximization.md)

</div>

<!-- RELATED:END -->
