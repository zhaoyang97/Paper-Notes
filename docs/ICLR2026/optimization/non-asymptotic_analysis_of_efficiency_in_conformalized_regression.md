---
title: >-
  [论文解读] Non-Asymptotic Analysis of Efficiency in Conformalized Regression
description: >-
  [ICLR 2026][优化/理论][保形预测] 首次建立保形分位数回归（CQR）和保形中位数回归（CMR）在 SGD 训练下的非渐近效率界，明确刻画了预测集长度偏差与训练样本量 $n$、校准样本量 $m$ 和误覆盖率 $\alpha$ 的联合依赖关系。 保形预测（Conformal Prediction）是一种无分布假设的…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "保形预测"
  - "分位数回归"
  - "非渐近分析"
  - "预测集效率"
  - "不确定性量化"
---

# Non-Asymptotic Analysis of Efficiency in Conformalized Regression

**会议**: ICLR 2026  
**arXiv**: [2510.07093](https://arxiv.org/abs/2510.07093)  
**代码**: 无  
**领域**: 优化 / 统计学习理论  
**关键词**: 保形预测, 分位数回归, 非渐近分析, 预测集效率, 不确定性量化

## 一句话总结

首次建立保形分位数回归（CQR）和保形中位数回归（CMR）在 SGD 训练下的非渐近效率界，明确刻画了预测集长度偏差与训练样本量 $n$、校准样本量 $m$ 和误覆盖率 $\alpha$ 的联合依赖关系。

## 研究背景与动机

保形预测（Conformal Prediction）是一种无分布假设的框架，能为黑盒模型提供具有覆盖率保证的预测集。在回归任务中，保形预测的**效率**（efficiency）通常以预测区间的期望长度来衡量——在满足覆盖率条件 $\mathbb{P}[Y \in \mathcal{C}(X)] \geq 1-\alpha$ 的前提下，预测集越小越好。

现有的效率分析主要有两类问题：

**渐近分析**: 证明预测集在样本趋于无穷时收敛到 oracle 集，但不提供有限样本保证

**已有非渐近界**: 通常将 $\alpha$ 视为常数，只考虑校准集大小 $m$ 的影响，忽略训练集大小 $n$ 和 $\alpha$ 取值的影响

在实际安全关键应用中（医疗、金融、自动驾驶），$\alpha$ 通常取得很小以确保高覆盖率，此时效率的行为尚不清楚。本文填补了这一理论空白，首次给出了效率关于 $(n, m, \alpha)$ 三者的非渐近刻画。

## 方法详解

### 整体框架

本文在 split conformal prediction 框架下分析两种回归变体：保形分位数回归（CQR）用线性模型估计条件上下分位数、构造自适应的非对称区间，保形中位数回归（CMR）只估计条件中位数、以残差绝对值为非一致性分数、构造对称区间。两者都用 SGD 训练，分析目标是预测集长度与 oracle 区间长度之间的期望绝对偏差 $\mathbb{E}[\,||\mathcal{C}(X)| - |\mathcal{C}^*(X)|\,|]$，并把这个偏差显式地拆解到训练样本量 $n$、校准样本量 $m$ 和误覆盖率 $\alpha$ 上。

### 关键设计

**1. CQR-SGD 的四项效率界：把长度偏差拆到 $(n,m,\alpha)$ 三个维度**

以往非渐近分析往往把 $\alpha$ 当常数、只盯校准集大小 $m$，于是在 $\alpha$ 很小的安全关键场景里完全失灵。Theorem 3.2 给出了完整的界 $\mathbb{E}[\,||\mathcal{C}(X)| - |\mathcal{C}^*(X)|\,|] \leq \mathcal{O}(n^{-1/2} + (\alpha^2 n)^{-1} + m^{-1/2} + \exp(-\alpha^2 m))$，四项各有清晰来源：$n^{-1/2}$ 是分位数回归的标准训练误差，$(\alpha^2 n)^{-1}$ 是 $\alpha$ 变小时训练误差被放大的部分，$m^{-1/2}$ 是校准集的有限样本效应，$\exp(-\alpha^2 m)$ 则刻画 $\alpha$ 变小时校准的指数衰减。这一界成立的前提是线性模型良定义（Assumption 3.1）、协方差有界（Assumption 3.2）以及条件密度正则性（Assumption 3.3）。正是 $(\alpha^2 n)^{-1}$ 与 $\exp(-\alpha^2 m)$ 这两项第一次把 $\alpha$ 的作用显式写进了效率界。

**2. CMR-SGD 的等阶界：对称假设换来更简洁的分析**

CMR 在同方差任务下的预测集长度不随输入变化、是个常数，分析理应更简单，但它只估中位数、缺少对上下分位数的直接控制。Theorem 4.1 表明，在额外的分位数对称假设（Assumption 4.2，即分位数关于中位数对称）下，CMR 能取得与 CQR 完全相同阶的 $(n,m,\alpha)$ 效率界。这说明三元组刻画并非 CQR 特有，而是 SGD 训练下保形回归的共性，对称性是把单点中位数估计「补」成区间控制的关键代价。

**3. 相位转变分析：按 $\alpha$ 大小指导训练/校准数据分配**

安全关键应用里 $\alpha$ 取多小才会改变效率的主导项？Section 3.2.1 取 $n = \Theta(m)$ 让界简化为 $\mathcal{O}(n^{-1/2} + (\alpha^2 n)^{-1})$，再按 $\alpha$ 分三段刻画其行为：当 $\alpha = \Omega(n^{-1/4})$ 时界为 $\mathcal{O}(n^{-1/2})$、与 $\alpha$ 无关；当 $n^{-1/2} \ll \alpha \ll n^{-1/4}$ 时界过渡到由 $\mathcal{O}((\alpha^2 n)^{-1})$ 主导；当 $\alpha = \Theta(n^{-1/2})$ 这一极端情形 $\alpha$ 完全主导界的行为。由此得到可操作的数据分配指导——$\alpha$ 足够大时训练集与校准集取同阶即可，$\alpha$ 较小时则必须把更多预算投向训练数据以压住被放大的训练误差项。

**4. 三步证明框架：把参数偏差逐层传到经验分位数**

把长度偏差落到 $(n,m,\alpha)$ 上的难点在于，SGD 学到的参数误差要先影响 score 分布、再影响经验分位数、最后才影响区间长度。证明沿三个 Proposition 逐层传递：先用 Prop B.5 控制学习参数偏差对 score 分布总体分位数的影响，再用 Prop B.7 控制有限样本分位数 $(1-\alpha)_m$ 与总体 $(1-\alpha)$ 分位数之间的差距，最后用 Prop B.11 借 DKW 不等式把经验分位数集中到总体分位数。其中两项核心技术分别撑起首尾：利用 pinball loss 的强凸性得到 SGD 的 $\mathcal{O}(1/n)$ 中心误差界（Theorem 3.1），以及利用条件密度正则性保证 score 分布在分位数附近有下界密度，从而让分位数的微小平移不会被无限放大。

## 实验关键数据

### 合成数据实验

实验使用分段仿射条件密度，验证理论界的三个维度：

| 实验维度 | 理论预测 | 实验观察 |
|---------|---------|---------|
| 训练集大小 $n$ | log Δ vs log n 斜率从 -1 到 -0.5 | 随 α 增大，斜率确实从 -1 过渡到 -0.5 |
| 校准集大小 $m$ | log-log 斜率趋近 -0.5 | 实测斜率接近 -0.5 |
| 误覆盖率 α | Δ ~ α^{-2} | 拟合系数 b₁ = -2.24，接近理论值 -2 |

### 真实数据实验

使用 MEPS Panel 19/20 数据集：
- 增大校准集 $m$ 一致减小长度偏差
- 固定样本量下，较大的 $\alpha$ 导致更小偏差和更低方差
- CQR 和 CMR 均展示与理论预测一致的趋势

### 关键发现

- 在 $\mathcal{O}((\alpha^2 n)^{-1})$ 主导的区间内，实测 Δ vs $1/(n\alpha^2)$ 的 log-log 回归斜率约为 0.92，接近理论值 1
- 效率界的相位转变在实验中得到清晰验证

## 亮点与洞察

1. **首次三元组 $(n, m, \alpha)$ 效率界**: 以往工作要么只考虑 $m$，要么将 $\alpha$ 视为常数；本文首次揭示 $\alpha$ 的关键作用
2. **直接假设于数据分布**: 不同于先前工作对 score 分布的假设，本文假设直接施加在数据分布上，更自然且可验证
3. **优化器无关的分析框架**: 虽以 SGD 为例，但只需替换训练误差界即可推广到其他优化器（如 SAGA/SVRG 可获得指数收敛率）
4. **实用的数据分配指导**: 根据 $\alpha$ 的大小，可以指导如何在训练集和校准集之间分配数据以最小化预测集冗余长度

## 局限与展望

1. **线性模型假设**: 理论结果限于线性分位数回归，推广到非线性模型（如神经网络）需要新的技术工具
2. **条件密度正则性**: 要求条件密度上下有界（$f_{\min} \leq f_{Y|X} \leq f_{\max}$），排除了重尾分布和密度趋零的情况
3. **CMR 的对称假设**: Assumption 4.2 要求分位数关于中位数对称，限制了 CMR 分析的适用范围
4. **缺乏下界**: 仅给出上界，无法判断界的紧性；是否存在匹配的下界是重要的开放问题

## 相关工作与启发

- **Romano et al. (2019)**: CQR 的提出者，本文从理论角度刻画其效率
- **Lei et al. (2018)**: 将训练误差纳入效率分析，但将 $\alpha$ 视为常数
- **Bars & Humbert (2025)**: 对体积最小化保形方法的非渐近分析，假设有限函数类，$\alpha$ 固定时与本文结果一致
- **Rakhlin et al. (2012)**: 提供强凸目标下 SGD 最优收敛率，是 Theorem 3.1 的理论基础

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次建立 $(n, m, \alpha)$ 三元非渐近界
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实数据，多角度验证理论预测
- 写作质量: ⭐⭐⭐⭐⭐ — 理论清晰，证明框架结构化，实验直观
- 价值: ⭐⭐⭐⭐ — 填补保形预测效率理论的重要空白，提供实用数据分配指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting](hinge_regression_tree_a_newton_method_for_oblique_regression_tree_splitting.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](../../ICML2026/optimization/taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[ICLR 2026\] Distributionally Robust Linear Regression with Block Lewis Weights](distributionally_robust_linear_regression_with_block_lewis_weights.md)
- [\[ICLR 2026\] SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration](sgd_with_adaptive_preconditioning_unified_analysis_and_momentum_acceleration.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)

</div>

<!-- RELATED:END -->
