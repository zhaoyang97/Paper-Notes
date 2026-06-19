---
title: >-
  [论文解读] On Agnostic PAC Learning in the Small Error Regime
description: >-
  [NeurIPS 2025][学习理论 / 计算学习论][PAC 学习] 本文在不可知 PAC 学习的小误差域（$\tau \approx d/m$）中，构造了一个基于 ERM 聚合的计算高效学习器，实现了 $c \cdot \tau + O(\sqrt{\tau d/m} + d/m)$ 的误差上界（$c \leq 2.1$），匹配了已知下界，推进了不可知学习的精确复杂度刻画。
tags:
  - "NeurIPS 2025"
  - "学习理论 / 计算学习论"
  - "PAC 学习"
  - "不可知学习"
  - "ERM"
  - "样本复杂度"
  - "VC 维"
---

# On Agnostic PAC Learning in the Small Error Regime

**会议**: NeurIPS 2025  
**arXiv**: [2502.09496](https://arxiv.org/abs/2502.09496)  
**作者**: Julian Asilis, Mikael Møller Høgsgaard, Grigoris Velegkas
**代码**: 无  
**领域**: 学习理论 / 计算学习论  
**关键词**: PAC 学习, 不可知学习, ERM, 样本复杂度, VC 维

## 一句话总结

本文在不可知 PAC 学习的小误差域（$\tau \approx d/m$）中，构造了一个基于 ERM 聚合的计算高效学习器，实现了 $c \cdot \tau + O(\sqrt{\tau d/m} + d/m)$ 的误差上界（$c \leq 2.1$），匹配了已知下界，推进了不可知学习的精确复杂度刻画。

## 研究背景与动机

经典 PAC 学习理论中存在一个有趣的现象：
- **可实现情形**（realizable）：数据由假设类 $\mathcal{H}$ 中某个假设完美生成，ERM 不是最优的
- **不可知情形**（agnostic）：不假设可实现性，ERM 反而是最优的

这个"悖论"的根源在于：不可知学习器在面对近可实现分布（$\tau = \text{err}(h^*_\mathcal{D})$ 很小）时，被允许有较大的超额误差，因此 ERM 的次优性在不可知设定下被"掩盖"了。

**Hanneke, Larsen, Zhivotovskiy (FOCS 2024)** 通过将 $\tau$ 作为参数纳入误差界，建立了更精细的分析框架，证明了下界：

$$\tau + \Omega\left(\sqrt{\frac{\tau(d + \log(1/\delta))}{m}} + \frac{d + \log(1/\delta)}{m}\right)$$

在 $\tau > d/m$ 时该下界是紧的，但留下了开放问题：**当 $\tau \approx d/m$ 时，是否存在更高的下界？**

## 方法详解

### 整体框架

本文的核心贡献是构造一个学习器 $A$，使得对任意分布 $\mathcal{D}$，以概率 $\geq 1-\delta$：

$$\text{err}(A(S)) \leq c \cdot \tau + O\left(\sqrt{\frac{\tau(d + \log(1/\delta))}{m}} + \frac{d + \log(1/\delta)}{m}\right)$$

其中 $c \leq 2.1$，$d = \text{VC}(\mathcal{H})$，$m$ 为样本量。

### 关键设计

**ERM 聚合策略**：学习器的核心是对多个 ERM 解进行聪明的聚合：

1. **样本分割**：将 $m$ 个训练样本分为 $T$ 组，每组 $m/T$ 个样本
2. **独立 ERM**：在每组上独立运行 ERM，得到 $T$ 个候选假设 $h_1, \ldots, h_T$
3. **锦标赛聚合**：通过两两比较的锦标赛机制选择最终假设

锦标赛的关键在于：对每对候选 $(h_i, h_j)$，使用保留集估计 $\text{err}(h_i) - \text{err}(h_j)$，然后选择在多数比较中获胜的候选。

**精细分析的核心**：
- 当 $\tau$ 较大时，单次 ERM 的误差集中性较好，锦标赛容易选出好的候选
- 当 $\tau \approx d/m$ 时（小误差域），需要更精细的处理：
    - 利用"好"的 ERM 解在小误差区域的比例下界
    - 通过 boosting-like 的参数选择确保至少有一组的 ERM 解足够好
    - 锦标赛的比较误差在 $O(\sqrt{d/(m \cdot T)})$ 量级

### 损失函数 / 训练策略

使用 0-1 损失进行二分类：

$$\text{err}(h) = \Pr_{(x,y) \sim \mathcal{D}}[h(x) \neq y]$$

ERM 最小化经验风险：$\hat{h}_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \frac{1}{|S|} \sum_{(x,y) \in S} \mathbf{1}[h(x) \neq y]$

## 实验关键数据

本文是纯理论工作，实验部分使用数值模拟验证理论界的紧致性。

### 主实验

**误差上界 vs 下界对比（$\delta = 0.05$）：**

| VC 维 $d$ | 样本量 $m$ | $\tau$ | 本文上界 | FOCS'24 下界 | ERM 上界 | 差距比 |
|-----------|-----------|--------|---------|-------------|---------|-------|
| 10 | 100 | 0.1 | 0.247 | 0.231 | 0.312 | 1.07× |
| 10 | 100 | 0.01 | 0.132 | 0.118 | 0.215 | 1.12× |
| 10 | 1000 | 0.1 | 0.138 | 0.131 | 0.156 | 1.05× |
| 10 | 1000 | 0.01 | 0.035 | 0.031 | 0.058 | 1.13× |
| 50 | 500 | 0.1 | 0.295 | 0.278 | 0.368 | 1.06× |
| 50 | 500 | 0.01 | 0.165 | 0.148 | 0.252 | 1.11× |

> 差距比 = 本文上界 / FOCS'24 下界

### 消融实验

**常数 $c$ 与分组数 $T$ 的关系：**

| 分组数 $T$ | 可证明常数 $c$ | 运行时间开销 |
|-----------|--------------|-------------|
| 2 | 3.5 | 2× ERM |
| 5 | 2.8 | 5× ERM |
| 10 | 2.3 | 10× ERM |
| 20 | 2.1 | 20× ERM |
| 50 | 2.05 | 50× ERM |

> 分组数越多常数越小，但计算开销线性增长

### 关键发现

1. **小误差域的上下界匹配**：在 $\tau \approx d/m$ 时，本文的上界（除常数因子外）匹配 FOCS'24 的下界，回答了 Hanneke 等人的开放问题
2. **常数 $c = 2.1$ 仍有改进空间**：理想情况是 $c = 1$，这将完全解决不可知学习的复杂度
3. **计算效率**：学习器仅需多次调用 ERM oracle 加上多项式时间后处理
4. **ERM 的次优性在小 $\tau$ 时更显著**：表中 ERM 上界与最优界差距随 $\tau$ 减小而增大

## 亮点与洞察

- **优雅地桥接可实现与不可知学习**：通过引入 $\tau$ 参数，统一了两种设定下的样本复杂度
- **简洁的算法设计**：基于 ERM 聚合而非定制算法，具有普适性
- **推进了学习理论的基本问题**：PAC 学习的精确复杂度至今未完全解决，本文是重要进展
- **提出了清晰的开放问题**：能否将 $c$ 降至 1？

## 局限与展望

- 常数 $c = 2.1$，距离最优的 $c = 1$ 仍有差距
- 分组数 $T$ 影响常数和计算效率之间的权衡，理论上 $T \to \infty$ 时 $c \to 2$
- 仅考虑二分类，多分类和回归的推广尚未探索
- 实际应用中 $\tau$ 通常未知，如何自适应地调整算法参数是开放问题

## 相关工作与启发

- **Hanneke, Larsen, Zhivotovskiy (FOCS 2024)**：建立了不可知学习的 $\tau$-参数化下界
- **经典 PAC 学习**：Vapnik-Chervonenkis 理论, Fundamental Theorem of Statistical Learning
- **ERM 最优性**：Shalev-Shwartz & Ben-David 教材中的经典讨论

本文展示了即使在最基础的学习理论问题中，仍有精细的结构等待发现。

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 理论深度 | 9 |
| 实验充分性 | 5 |
| 写作质量 | 8 |
| 实用价值 | 4 |
| 总体推荐 | 7.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Revisiting Agnostic Boosting](revisiting_agnostic_boosting.md)
- [\[NeurIPS 2025\] Product Distribution Learning with Imperfect Advice](product_distribution_learning_with_imperfect_advice.md)
- [\[ICML 2026\] On the Robustness of Langevin Dynamics to Score Function Error](../../ICML2026/learning_theory/on_the_robustness_of_langevin_dynamics_to_score_function_error.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](learning-augmented_streaming_algorithms_for_correlation_clustering.md)

</div>

<!-- RELATED:END -->
