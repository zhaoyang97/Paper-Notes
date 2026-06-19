---
title: >-
  [论文解读] Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent
description: >-
  [ICML 2025][优化/理论][coordinate descent] 本文首次理论证明了在正定二次函数的坐标下降中，随机排列坐标下降（RPCD）的收缩率严格优于均匀随机坐标下降（RCD），从而解决了一个长期悬而未决的理论问题。 领域现状： 坐标下降（CD）是大规模优化中的基础算法，广泛应用于 LASSO、SVM、矩阵…
tags:
  - "ICML 2025"
  - "优化/理论"
  - "coordinate descent"
  - "random permutation"
  - "convergence rate"
  - "quadratic optimization"
  - "contraction rate"
---

# Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent

**会议**: ICML 2025  
**arXiv**: [2505.23152](https://arxiv.org/abs/2505.23152)  
**代码**: 无  
**领域**: 优化  
**关键词**: coordinate descent, random permutation, convergence rate, quadratic optimization, contraction rate

## 一句话总结
本文首次理论证明了在正定二次函数的坐标下降中，随机排列坐标下降（RPCD）的收缩率严格优于均匀随机坐标下降（RCD），从而解决了一个长期悬而未决的理论问题。

## 研究背景与动机

**领域现状**: 坐标下降（CD）是大规模优化中的基础算法，广泛应用于 LASSO、SVM、矩阵分解等问题。两种主要变体为：随机坐标下降（RCD，每次均匀随机选择一个坐标更新）和随机排列坐标下降（RPCD，每个 epoch 对坐标做随机排列，依次更新所有坐标）。

**现有痛点**: 大量实验表明 RPCD 比 RCD 收敛更快，但理论上证明这一差距异常困难。即使在最简单的正定二次函数（$f(x) = \frac{1}{2} x^\top A x - b^\top x$）加上置换不变 Hessian 的条件下，之前的工作也未能给出二者之间的可证明差距。

**核心矛盾**: RPCD 的分析难度在于排列引入了坐标更新之间的复杂依赖关系。RCD 由于独立采样性质更易分析，但 RPCD 的"不重复采样"直觉上应该更高效（每个 epoch 保证每个坐标恰好更新一次）。

**本文目标**: 在正定二次函数类上严格证明 RPCD 优于 RCD。

**切入角度**: 聚焦于具有置换不变结构的 Hessian 矩阵类，利用其对称性简化分析。

**核心 idea**: 对于置换不变 Hessian 的二次函数类，RPCD 的收缩率上界严格小于 RCD 的收缩率下界——对每个具体问题实例都成立。

## 方法详解

### 整体框架
考虑二次优化问题 $\min_x f(x) = \frac{1}{2} x^\top A x - b^\top x$，其中 $A \succ 0$。

- **RCD**: 每步随机选 $i \in \{1,\ldots,d\}$，更新 $x_i \leftarrow x_i - \frac{1}{A_{ii}}(\nabla_i f(x))$
- **RPCD**: 每个 epoch 生成随机排列 $\sigma$，依次更新 $x_{\sigma(1)}, x_{\sigma(2)}, \ldots, x_{\sigma(d)}$

收敛速率用收缩率 $\rho$ 衡量：$\mathbb{E}[\|x^{t+1} - x^*\|_A^2] \leq \rho \cdot \|x^t - x^*\|_A^2$。

### 关键设计

1. **RCD 收缩率下界（RCD Contraction Rate Lower Bound）**:

    - 功能：给出 RCD 在置换不变 Hessian 类上的精确收缩率下界
    - 核心思路：利用 Hessian 的特征结构，将 $d$ 维问题分解为独立的 1 维问题，从而精确计算每步的期望能量递减
    - 关键公式：$\rho_{RCD} \geq 1 - \frac{2}{d} \cdot \frac{\lambda_{\min}}{\lambda_{\min} + \lambda_{\max}} + \frac{1}{d^2} \cdot g(\lambda_{\min}, \lambda_{\max})$
    - 设计动机：精确的下界是证明差距的必要条件

2. **RPCD 收缩率上界（RPCD Contraction Rate Upper Bound）**:

    - 功能：给出 RPCD 在同一函数类上的收缩率上界
    - 核心思路：RPCD 一个 epoch 的更新等价于将 $d$ 次坐标更新按随机排列顺序复合。利用矩阵指数和排列群的对称性，得到比 RCD 更紧的上界
    - 关键结论：对于每个问题实例，$\rho_{RPCD} < \rho_{RCD}$（严格小于）
    - 设计动机："不重复"采样特性使得 RPCD 在每个 epoch 内更均匀地覆盖所有坐标

3. **最坏情况猜想（Worst-Case Conjecture）**:

    - 功能：猜想置换不变 Hessian 类包含了 RPCD 在所有正定二次函数上的最坏情况
    - 核心思路：通过数值实验支持此猜想——在大量随机生成的正定矩阵上，RPCD 的收缩率均不超过在置换不变矩阵上的值
    - 设计动机：若猜想成立，则 RPCD 优于 RCD 的结论推广到所有正定二次函数

### 损失函数 / 训练策略
标准正定二次函数损失，无需特殊训练策略。

## 实验关键数据

### 主实验

| 问题维度 $d$ | 条件数 $\kappa$ | $\rho_{RCD}$ (理论下界) | $\rho_{RPCD}$ (理论上界) | 实际 $\rho_{RPCD}$ | gap |
|-------------|----------------|----------------------|------------------------|-------------------|-----|
| 10 | 10 | 0.836 | **0.791** | 0.783 | 5.4% |
| 10 | 100 | 0.982 | **0.965** | 0.961 | 1.7% |
| 50 | 10 | 0.965 | **0.947** | 0.944 | 1.9% |
| 50 | 100 | 0.9965 | **0.9932** | 0.9928 | 0.33% |

### 消融实验

| 矩阵类型 | RPCD 优于 RCD? | gap 大小 | 说明 |
|----------|---------------|---------|------|
| 置换不变 Hessian | ✅ 已证明 | 1-5% | 理论保证 |
| 对角 Hessian | ✅ 已证明 | 较大 | 特殊情况，此前已知 |
| 一般正定 Hessian (数值) | ✅ 观测到 | 1-10% | 支持猜想 |
| 非正定 Hessian | 不适用 | — | 超出分析范围 |

### 关键发现
- 首次在理论上严格证明了 RPCD 优于 RCD（在置换不变 Hessian 类上）
- 差距（gap）在低维和高条件数时更为显著
- 数值实验强烈支持"最坏情况猜想"——RPCD 在一般正定二次函数上也优于 RCD
- 理论上界与实际收缩率非常接近，说明分析是紧致的

## 亮点与洞察
- **解决了重要开放问题**: RCD vs RPCD 的理论差距是坐标下降领域长期未解的问题
- **分析技术新颖**: 利用置换不变结构和矩阵指数理论，为 RPCD 分析提供了全新工具
- **猜想有启发性**: 最坏情况猜想为后续完整证明指明了方向

## 局限与展望
- 结果限于正定二次函数，非凸和非二次情况尚未解决
- 置换不变 Hessian 是较特殊的矩阵类，一般正定矩阵的理论证明仍是开放问题
- gap 在高维时较小，实际收益可能有限
- 未考虑加速坐标下降或带动量的变体

## 相关工作与启发
- Gürbüzbalaban et al. (2021): RCD 和 RPCD 的渐近分析
- Recht & Ré (2013): 提出 RPCD 优于 RCD 的猜想
- 本文的分析技术可能推广到 mini-batch CD、随机 Gauss-Seidel 等变体

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 解决了领域内重要的开放问题
- 实验充分度: ⭐⭐⭐⭐ 数值实验充分验证理论，支持猜想
- 写作质量: ⭐⭐⭐⭐ 68 页长文，理论严谨，组织清晰
- 价值: ⭐⭐⭐⭐⭐ 对优化理论有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](../../ICML2026/optimization/on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2025\] Random Feature Representation Boosting](random_feature_representation_boosting.md)
- [\[ICML 2025\] Provable In-Context Vector Arithmetic via Retrieving Task Concepts](provable_in-context_vector_arithmetic_via_retrieving_task_concepts.md)
- [\[ICML 2025\] Learning Mixtures of Experts with EM: A Mirror Descent Perspective](learning_mixtures_of_experts_with_em_a_mirror_descent_perspective.md)
- [\[ICLR 2026\] A Block Coordinate Descent Method for Nonsmooth Composite Optimization under Orthogonality Constraints](../../ICLR2026/optimization/a_block_coordinate_descent_method_for_nonsmooth_composite_optimization_under_ort.md)

</div>

<!-- RELATED:END -->
