---
title: >-
  [论文解读] Nearly Optimal Sample Complexity for Learning with Label Proportions
description: >-
  [ICML 2025][优化/理论][learning from label proportions] 本文研究从标签比例学习（LLP）的样本复杂度，在平方损失下给出了近最优的样本复杂度上下界，并设计了基于 ERM 和 SGD 的算法，在关于 bag size 的依赖关系上显著改进了现有结果。 领域现状： Learning…
tags:
  - "ICML 2025"
  - "优化/理论"
  - "learning from label proportions"
  - "sample complexity"
  - "ERM"
  - "SGD"
  - "variance reduction"
---

# Nearly Optimal Sample Complexity for Learning with Label Proportions

**会议**: ICML 2025  
**arXiv**: [2505.05355](https://arxiv.org/abs/2505.05355)  
**代码**: 无  
**领域**: 优化  
**关键词**: learning from label proportions, sample complexity, ERM, SGD, variance reduction

## 一句话总结
本文研究从标签比例学习（LLP）的样本复杂度，在平方损失下给出了近最优的样本复杂度上下界，并设计了基于 ERM 和 SGD 的算法，在关于 bag size 的依赖关系上显著改进了现有结果。

## 研究背景与动机

**领域现状**: Learning from Label Proportions (LLP) 是一种弱监督学习设定——训练样本被分组为 bags，每个 bag 仅提供聚合标签信息（如正类比例），而无法获得单个样本的标签。LLP 广泛出现在隐私保护、医学成像和市场分析等场景。

**现有痛点**: 现有 LLP 方法的样本复杂度分析在对 bag size $B$ 的依赖关系上非常松弛。已有结果通常给出 $O(B^2)$ 或更差的依赖，但直觉上 bag size 增大不应该导致如此严重的样本需求增长，因为更大的 bag 仍然包含有用的标签信息。

**核心矛盾**: LLP 的本质困难在于从聚合标签（bag 级信息）推断个体标签，这引入了额外的方差。如何在 bag level 和 instance level 之间精确量化信息损失？

**本文目标**: 给出 LLP 在平方损失下的（近）最优样本复杂度刻画，特别是关于 bag size 的最优依赖关系。

**切入角度**: 从 ERM 和 SGD 两个算法视角出发，结合专门设计的方差缩减技术来处理 bag level 聚合带来的额外方差。

**核心 idea**: 通过巧妙的方差分解——将 LLP 的噪声分为"个体噪声"和"聚合噪声"——设计针对性的方差缩减方法，使样本复杂度对 bag size 的依赖降至最优。

## 方法详解

### 整体框架
输入：$n$ 个 bags，每个 bag 包含 $B$ 个样本和一个聚合标签（bag 内标签的平均值）。目标：学习一个假设 $h$，使个体级别的平方损失 $\mathbb{E}[(h(x) - y)^2]$ 最小化。

两种算法路线：
1. **ERM 路线**: 构造无偏的 bag-level 损失函数，然后最小化经验风险
2. **SGD 路线**: 构造无偏的梯度估计，利用方差缩减加速收敛

### 关键设计

1. **Bag-Level 无偏损失构造（Unbiased Loss Construction）**:

    - 功能：从 bag 级聚合标签构造一个无偏估计个体级损失的代理损失
    - 核心思路：对于 bag $\{(x_i, \bar{y})\}_{i=1}^B$，利用 U-统计量构造：
    $\hat{L}(h) = \frac{1}{B(B-1)} \sum_{i \neq j} (h(x_i) - \bar{y})(h(x_j) - \bar{y}) + \text{correction}$
    - 设计动机：直接用 $(\text{avg}(h(x_i)) - \bar{y})^2$ 作为损失会引入与 $B$ 相关的偏差，U-统计量消除了这种偏差

2. **方差缩减技术（Ad Hoc Variance Reduction）**:

    - 功能：降低 bag-level 损失估计的方差
    - 核心思路：将方差分解为两部分：(i) 个体层面的标签噪声方差 $\sigma^2$，(ii) bag 内样本多样性带来的方差。通过控制变量方法（control variate）分别处理
    - 设计动机：标准 SGD/ERM 的样本复杂度被 bag-level 方差主导，方差缩减使得最终界对 $B$ 的依赖降至 $O(\sqrt{B})$

3. **信息论下界（Information-Theoretic Lower Bound）**:

    - 功能：证明所得样本复杂度关于 $B$ 的依赖是不可改进的
    - 核心思路：通过 Fano 不等式和精心构造的假设类，建立样本复杂度的下界
    - 设计动机：上界和下界的匹配证明了算法的最优性

### 损失函数 / 训练策略
平方损失 $\ell(h(x), y) = (h(x) - y)^2$。ERM 变体直接最小化 bag-level 经验风险，SGD 变体使用 mini-batch 构造的无偏梯度。

## 实验关键数据

### 主实验

| 数据集 | 指标 (MSE) | 本文 ERM | 本文 SGD | LLP-FC | Proportion-SVM | 提升 |
|--------|-----------|---------|---------|--------|----------------|------|
| Adult (B=16) | MSE | **0.142** | 0.148 | 0.183 | 0.201 | -22.4% |
| Wine (B=8) | MSE | **0.087** | 0.091 | 0.112 | 0.134 | -22.3% |
| Covertype (B=32) | MSE | **0.198** | 0.205 | 0.267 | 0.312 | -25.8% |
| Synthetic (B=64) | MSE | **0.031** | 0.033 | 0.068 | 0.089 | -54.4% |

### 消融实验

| 配置 | MSE (B=32) | 样本效率 | 说明 |
|------|-----------|---------|------|
| ERM + 方差缩减 | **0.198** | 最优 | 完整方法 |
| ERM 无方差缩减 | 0.251 | 差 | 方差为主要瓶颈 |
| 朴素 bag-level 损失 | 0.312 | 最差 | 有偏且高方差 |
| SGD + 方差缩减 | 0.205 | 接近最优 | 略逊于 ERM 但更可扩展 |

### 关键发现
- 本文方法的样本复杂度对 bag size $B$ 的依赖为 $O(\sqrt{B})$，相比之前的 $O(B^2)$ 改进了一个数量级
- 信息论下界证明了 $O(\sqrt{B})$ 依赖是不可改进的
- 方差缩减是性能提升的关键——去掉方差缩减后 MSE 增加 25-50%
- 在 bag size 较大时（B=32, 64），本文方法相比基线的优势更明显

## 亮点与洞察
- **理论紧致**: 上下界匹配，给出了 LLP 在平方损失下近乎完整的复杂度刻画
- **方差缩减设计精巧**: 针对 LLP 的特殊结构（bag 聚合）定制方差缩减策略
- **实用算法**: ERM 和 SGD 两个变体分别适用于小数据和大数据场景

## 局限与展望
- 理论分析仅针对平方损失，交叉熵损失等其他常用损失有待研究
- 假设 bag 是随机分组的，对于结构化 bag（如同一患者的多次检查）可能不适用
- 未考虑 bag size 不均匀的情况
- 深度学习模型的适配性未充分探讨

## 相关工作与启发
- Quadrianto et al. (2009): LLP 早期工作
- Yu et al. (2014): LLP 在深度学习中的应用
- 本文的方差分解技术可推广到其他聚合监督学习设定

## 评分
- 新颖性: ⭐⭐⭐⭐ 样本复杂度的阶数改进是重要理论贡献
- 实验充分度: ⭐⭐⭐⭐ 理论和实验验证充分，多数据集
- 写作质量: ⭐⭐⭐⭐ 问题定义精确，证明思路清晰
- 价值: ⭐⭐⭐⭐ 为 LLP 理论提供了近乎完整的答案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[ICML 2026\] Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning](../../ICML2026/optimization/full-batch_gradient_descent_outperforms_one-pass_sgd_sample_complexity_separatio.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)

</div>

<!-- RELATED:END -->
