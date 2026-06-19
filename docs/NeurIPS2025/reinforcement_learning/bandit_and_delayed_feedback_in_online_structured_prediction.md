---
title: >-
  [论文解读] Bandit and Delayed Feedback in Online Structured Prediction
description: >-
  [NeurIPS 2025][强化学习][在线结构化预测] 首次研究在线结构化预测中赌臂反馈和延迟反馈场景，通过设计新的伪逆矩阵梯度估计器，实现了不显式依赖输出集大小 $K$ 的 $O(T^{2/3})$ 替代遗憾上界。 在线结构化预测是序贯预测具有复杂结构输出的任务，涵盖在线分类、多标签分类和排序等问题…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "在线结构化预测"
  - "赌臂反馈"
  - "延迟反馈"
  - "替代遗憾"
  - "Fenchel-Young 损失"
---

# Bandit and Delayed Feedback in Online Structured Prediction

**会议**: NeurIPS 2025  
**arXiv**: [2502.18709](https://arxiv.org/abs/2502.18709)  
**代码**: 无  
**领域**: 在线学习 / 结构化预测  
**关键词**: 在线结构化预测, 赌臂反馈, 延迟反馈, 替代遗憾, Fenchel-Young 损失

## 一句话总结

首次研究在线结构化预测中赌臂反馈和延迟反馈场景，通过设计新的伪逆矩阵梯度估计器，实现了不显式依赖输出集大小 $K$ 的 $O(T^{2/3})$ 替代遗憾上界。

## 研究背景与动机

在线结构化预测是序贯预测具有复杂结构输出的任务，涵盖在线分类、多标签分类和排序等问题。给定输入空间 $\mathcal{X}$ 和有限输出空间 $\mathcal{Y}$（$K = |\mathcal{Y}|$），学习者在每轮需要根据输入预测输出，并承受目标损失 $L(\hat{y}_t; y_t)$。

**替代损失框架**：由于直接优化结构化输出空间中的目标损失通常很困难，实际中采用替代损失（surrogate loss）如 Fenchel-Young 损失来间接学习。替代遗憾（surrogate regret）衡量学习者的累积目标损失超出最优替代损失的额外部分。

**现有局限**：
- 先前工作 (Sakaue et al., 2024) 仅在全信息反馈下实现了有限的替代遗憾界
- 全信息反馈要求立即访问完整的复杂输出结构，在实际中往往不现实
- 例如，在线广告排列中只能观察到点击率（赌臂反馈）而非具体点击了哪些广告

**核心挑战**：
1. 赌臂反馈下无法观察真实输出 $y_t$，导致无法计算替代损失的真实梯度
2. 当输出集极大时（如多标签分类中 $K = \binom{d}{m}$，排序中 $K = m!$），依赖 $K$ 的界极度不理想
3. 延迟反馈增加了算法设计的额外复杂性

## 方法详解

### 整体框架

本文基于 Fenchel-Young 损失的在线结构化预测框架，发展了四类算法以应对不同的反馈和延迟场景组合。核心流程为：每轮接收输入 → 线性估计器计算分数向量 → 随机解码生成预测 → 接收反馈 → 估计梯度更新参数。

### 关键设计

**1. 带均匀探索的随机解码 (RDUE)**：
- 以概率 $q$ 从 $\mathcal{Y}$ 中均匀随机选择输出
- 以概率 $1-q$ 使用标准随机解码
- 确保每个输出被选中的概率不小于 $q/K$，为逆加权估计提供保障
- 损失满足：$\mathbb{E}[L(\psi_\Omega(\theta); y)] \leq \frac{4\gamma}{\lambda\nu}(1-q)S_\Omega(\theta; y) + q\frac{K-1}{K}$

**2. 逆加权梯度估计器**（$O(\sqrt{KT})$ 算法）：
$$\tilde{G}_t = \frac{\mathbb{1}[\hat{y}_t = y_t]}{p_t(y_t)} G_t$$
该估计器是无偏的，但其方差与 $K/q$ 成正比，导致遗憾界依赖于 $K$。

**3. 伪逆矩阵梯度估计器**（$O(T^{2/3})$ 算法，核心贡献）：

当目标损失属于结构化编码损失函数 (SELF) 的特殊子类时，构造：
$$\tilde{y}_t = V^{-1} P_t^+ \hat{y}_t \langle \hat{y}_t, Vy_t \rangle$$
其中 $P_t = \mathbb{E}[\hat{y}_t \hat{y}_t^\top]$ 是二阶矩矩阵。基于此定义梯度估计器：
$$\tilde{G}_t = (\hat{y}_\Omega(\theta_t) - \tilde{y}_t) x_t^\top$$

关键性质：$\mathbb{E}_t[\|\tilde{G}_t\|_F^2] \leq 2bS_t(W_t) + 2C_x^2 \omega / q$，其中 $\omega = \text{tr}(V^{-1}Q^+(V^{-1})^\top)$ 不依赖 $K$。

**4. 延迟反馈处理**：
- 使用 ODAFTRL（Optimistic Delayed Adaptive FTRL）算法处理固定 $D$ 步延迟
- 获得 AdaGrad 型遗憾界，充分利用梯度的自适应结构

### 损失函数 / 训练策略

- **替代损失**：Fenchel-Young 损失 $S_\Omega(\theta; y) = \Omega^*(\theta) + \Omega(y) - \langle\theta, y\rangle$
- **目标损失**：结构化编码损失函数 (SELF)，$L(y_t; \hat{y}_t) = \langle\hat{y}_t, Vy_t + b\rangle + c(y_t)$
- **更新策略**：自适应在线梯度下降（赌臂场景）或 ODAFTRL（延迟场景），学习率 $\eta_t = B / \sqrt{2\sum_{i=1}^t \|\tilde{G}_i\|_F^2}$

## 实验关键数据

### 主实验：替代遗憾界汇总

| 问题设置 | 反馈类型 | 目标损失 | 替代遗憾上界 | 来源 |
|---------|---------|---------|------------|------|
| 多类分类 | 赌臂 | 0-1 损失 | $O(\sqrt{KT})$ | Van der Hoeven et al. (2021) |
| 结构化预测 | 全信息 | SELF | 有限界 | Sakaue et al. (2024) |
| **结构化预测** | **赌臂** | **SELF** | $O(\sqrt{KT})$ | **本文 (Thm 3.5, 3.6)** |
| **结构化预测** | **赌臂** | **SELF*** | $O(T^{2/3})$ | **本文 (Thm 3.8)** |
| **结构化预测** | **全信息+延迟** | **SELF** | $O(D^{2/3}T^{1/3})$ | **本文 (Thm 4.3, 4.4)** |
| **结构化预测** | **赌臂+延迟** | **SELF** | $O(\sqrt{DKT})$ | **本文 (Thm 5.1)** |
| **结构化预测** | **赌臂+延迟** | **SELF*** | $O(D^{1/3}T^{2/3})$ | **本文 (Thm 5.2)** |

### 消融实验：赌臂反馈的数值比较 (MNIST)

| 算法 | 替代损失类型 | 错误分类率（中位数）| 是否专门设计用于多类分类 |
|------|-----------|---------------|---------------------|
| Gaptron | Hinge loss | 较高 | 是 |
| Gappletron | Logistic loss | 中等 | 是 |
| **本文算法** | **FY loss** | **最低** | **否（通用结构化预测）** |

在 MNIST 数据集上（$K=10$），尽管本文算法是为通用结构化预测设计的，仍在多类分类任务上优于专门设计的算法。

### 关键发现

1. **$K$-无关界**：伪逆矩阵估计器使得遗憾界中消除了对 $K$ 的显式依赖。在多标签分类（$K = \binom{d}{m}$）和排序（$K = m!$）中，这一改进尤为显著
2. **$O(T^{2/3})$ 的代价**：消除 $K$ 依赖的代价是 $T$ 的指数从 $1/2$ 增加到 $2/3$。在 $K$ 极大时，这一权衡是有利的
3. **延迟的影响**：固定延迟 $D$ 引入乘法因子 $D^{1/3}$（伪逆估计器）或 $\sqrt{D}$（逆加权估计器）
4. **合成数据实验**：当 $K \leq 24$ 时本文方法具有竞争力；$K = 48, 96$ 时，专门化的算法因使用不同替代损失而具有优势
5. **高概率界**：Theorem 3.6 和 4.4 给出了高概率遗憾界，结合了 Bernstein 不等式

## 亮点与洞察

1. **伪逆矩阵估计器的精巧设计**：不同于标准逆加权估计器，伪逆估计器在构造上巧妙利用了 SELF 的结构，使梯度估计器的二阶矩不依赖 $K$
2. **理论框架的完整性**：系统覆盖了赌臂、延迟、赌臂+延迟三种场景，以及 SELF 和 SELF* 两类损失函数
3. **实际意义**：在线广告排列、推荐系统等场景中，通常只有有限的（赌臂）或延迟的反馈，本文方法直接适用

## 局限与展望

1. **下界缺失**：赌臂反馈和延迟反馈下替代遗憾的下界尚未建立，最优依赖于 $d$ 的形式未知
2. **固定延迟假设**：仅考虑了固定延迟 $D$，而实际中延迟通常是变化的
3. **$O(T^{2/3})$ 的最优性**：$T^{2/3}$ 是否可以改进为 $\sqrt{T}$（代价为某种对 $K$ 的多项式依赖）尚不清楚
4. **线性估计器限制**：假设模型为线性估计器，对深度学习场景的扩展值得探索
5. **SELF* 的限制性**：$O(T^{2/3})$ 界仅对 SELF 的子类成立，一般 SELF 仍需要 $K$ 依赖

## 相关工作与启发

- **在线多类分类**：Van der Hoeven (2020) 引入替代遗憾概念，Gaptron 和 Gappletron 是代表性算法
- **组合赌臂**：Cesa-Bianchi & Lugosi (2012) 的伪逆估计器启发了本文的设计
- **延迟在线学习**：Flaspohler et al. (2021) 的 ODAFTRL 提供了延迟反馈的核心算法基础

## 评分

| 维度 | 评分 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 5 |
| 实验充分性 | 3 |
| 写作质量 | 4 |
| 总评 | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback](../../AAAI2026/reinforcement_learning/bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)
- [\[ICML 2026\] Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](../../ICML2026/reinforcement_learning/parameter-free_dynamic_regret_time-varying_movement_costs_delayed_feedback_and_m.md)
- [\[ICLR 2026\] Online Prediction of Stochastic Sequences with High Probability Regret Bounds](../../ICLR2026/reinforcement_learning/online_prediction_of_stochastic_sequences_with_high_probability_regret_bounds.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Thompson Sampling for Multi-Objective Linear Contextual Bandit](thompson_sampling_for_multi-objective_linear_contextual_bandit.md)

</div>

<!-- RELATED:END -->
