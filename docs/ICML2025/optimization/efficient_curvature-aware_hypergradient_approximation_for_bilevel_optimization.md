---
title: >-
  [论文解读] Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization
description: >-
  [ICML2025][优化/理论][bilevel optimization] 提出 NBO 框架，利用双层优化中超梯度的内在结构（下层问题求解与 Hessian 逆向量积共享同一 Hessian），通过非精确 Newton 方法高效融合曲率信息来逼近超梯度，在确定性场景下将梯度计算复杂度相比 SOTA 改善了 $\kappa \log \kappa$ 倍。
tags:
  - "ICML2025"
  - "优化/理论"
  - "bilevel optimization"
  - "hypergradient"
  - "inexact Newton"
  - "curvature information"
  - "meta-learning"
---

# Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization

**会议**: ICML2025  
**arXiv**: [2505.02101](https://arxiv.org/abs/2505.02101)  
**代码**: 待确认  
**领域**: 优化  
**关键词**: bilevel optimization, hypergradient, inexact Newton, curvature information, meta-learning

## 一句话总结
提出 NBO 框架，利用双层优化中超梯度的内在结构（下层问题求解与 Hessian 逆向量积共享同一 Hessian），通过非精确 Newton 方法高效融合曲率信息来逼近超梯度，在确定性场景下将梯度计算复杂度相比 SOTA 改善了 $\kappa \log \kappa$ 倍。

## 研究背景与动机
双层优化广泛应用于超参数优化、元学习、对抗训练、神经架构搜索等机器学习任务。其核心挑战在于估计超梯度（implicit gradient）：

$$\nabla \Phi(x) = \nabla_1 f(x, y^*(x)) - \nabla_{12}^2 g(x, y^*(x)) u^*(x)$$

其中 $u^*(x) = [\nabla_{22}^2 g(x,y^*(x))]^{-1} \nabla_2 f(x,y^*(x))$。现有方法通常将**求解下层问题**和**计算 Hessian 逆向量积**视为独立任务，分别处理。然而两者共享同一 Hessian $\nabla_{22}^2 g$，这一内在结构尚未被充分利用。

SHINE（Ramzi et al., 2022）首次尝试利用该结构，用拟 Newton 矩阵同时处理两个子问题，但仅提供了渐近收敛分析，缺少收敛速率和计算复杂度的精确刻画。本文旨在**系统地挖掘并利用超梯度结构的收益**。

## 方法详解

### 框架：NBO（Newton-based Bilevel Optimization）

**核心思想**：利用非精确 Newton 方法同时求解下层最优解 $y^*(x)$ 和 Hessian 逆向量积 $u^*(x,y)$，两者共享同一 Hessian $H_k = \nabla_{22}^2 g(x^k, y^k)$。

**Step 1：曲率感知超梯度逼近**。给定 $(x, y)$，同时求解线性系统：

$$[\nabla_{22}^2 g(x,y)](v, u) = (\nabla_2 g(x,y), \nabla_2 f(x,y))$$

得到非精确 Newton 方向 $v$（逼近下层 Newton 步）和 $u$（逼近 Hessian 逆向量积）。

**Step 2：计算逼近超梯度**：

$$d_x = \nabla_1 f(x, y-v) - \nabla_{12}^2 g(x, y-v) u$$

**关键公式——单步 Newton 的二次衰减**：经典 Newton 步 $y^+ = y - [\nabla_{22}^2 g]^{-1} \nabla_2 g$ 可以将超梯度逼近误差从线性衰减提升为二次衰减：

$$\|\hat{\nabla}\Phi(x, y^+) - \nabla\Phi(x)\| \leq \frac{C L_{g,2}}{2\mu} \|y^*(x) - y\|^2$$

**将线性系统转化为二次规划**：子问题 (8)(10) 为强凸二次规划，用 GD 或 CG 近似求解。

### 算法实例

| 算法 | 设定 | 子问题求解器 | 特点 |
|------|------|-------------|------|
| **NBO-GD** | 确定性 | 梯度下降（T+1 步） | 当 T=0 时退化为 SOBA/Dagréou 框架 |
| **NBO-CG** | 确定性 | 共轭梯度 | Hessian-向量积复杂度更优 |
| **NSBO-SGD** | 随机 | 随机梯度下降 | 适用大规模场景 |

### 收敛与复杂度（确定性设定）

在标准假设（下层强凸参数 $\mu$、梯度 Lipschitz 等）下，NBO-GD 的收敛保证为：

$$\min_{0 \leq k \leq K-1} \|\nabla\Phi(x^k)\|^2 \leq O(\kappa^3 / K)$$

| 方法 | 梯度计算 | Hessian-向量积 |
|------|---------|---------------|
| AmIGO-GD | $O(\kappa^4 \log\kappa \cdot \epsilon^{-1})$ | $O(\kappa^4 \log\kappa \cdot \epsilon^{-1})$ |
| AID-BiO | $O(\kappa^4 \log\kappa \cdot \epsilon^{-1})$ | $O(\kappa^{3.5} \log\kappa \cdot \epsilon^{-1})$ |
| **NBO-GD** | $O(\kappa^3 \cdot \epsilon^{-1})$ | $O(\kappa^4 \cdot \epsilon^{-1})$ |
| **NBO-CG** | $O(\kappa^3 \cdot \epsilon^{-1})$ | $O(\kappa^{3.5} \log\kappa \cdot \epsilon^{-1})$ |

梯度复杂度改善了 $\kappa \log\kappa$ 倍。

### 随机设定（NSBO-SGD）

总样本复杂度 $O(\kappa^9 \epsilon^{-2})$，对比 AmIGO 的 $O(\kappa^9 \log\kappa \cdot \epsilon^{-2})$ 改善 $\log\kappa$ 倍。

### Lyapunov 分析

定义 Lyapunov 函数 $V_k = \Phi(x^k) - \Phi^* + b_y \|y^k - y^*(x^k)\|^2 + b_u \|u^k - u^*(x^k)\|^2$，证明非精确 Newton 步下 $\|y^{k+1} - y^*(x^{k+1})\|$ 满足二次衰减+线性扰动的下降条件，利用归纳法保证迭代点始终落在 Newton 局部收敛区域内。

## 实验关键数据

### 合成实验（超参数优化 + 逻辑回归）

| 对比 | 结果 |
|------|------|
| NBO-GD (T=1) vs AmIGO (Q=10) | 目标函数下降速度相当 |
| NBO-GD (T=1) vs AmIGO (Q=1) | NBO-GD 显著更优 |
| 按运行时评估 | NBO-GD (T=1) 最快 |

**发现**：NBO-GD 即使仅用 T=1 步近似 Newton 方向，效果也与 AmIGO 的 Q=10 步 GD 持平，说明曲率信息比多步 GD 更高效。

### 超参数优化（IJCNN1 + Covtype）

- **IJCNN1**：NSBO-SGD 收敛速度显著优于 SOBA、SABA、StoBiO、AmIGO、SHINE、F2SA、MA-SABA 等全部基线
- **Covtype**：NSBO-SGD 同样最快收敛，达到最低测试误差

### 数据清洗（MNIST + FashionMNIST，50% 噪声标签）

- NSBO-SGD 和 AmIGO 达到最低测试误差
- NSBO-SGD 收敛速度最快，体现了融合曲率信息的效率优势

### 方差缩减与动量扩展

- NSBO-SAGA 优于 SABA
- MA-NSBO-SAGA 优于 MA-SABA
- NBO 框架兼容 SAGA、STORM、PAGE 等方差缩减技术和动量技术

## 亮点与洞察
1. **优雅的结构洞察**：超梯度中下层问题和 Hessian 逆向量积共享 Hessian 这一结构被系统利用，是非常自然但被忽视的观察
2. **Newton 步带来二次衰减**：单步 Newton 即可将超梯度估计误差从 $O(\|y-y^*\|)$ 提升到 $O(\|y-y^*\|^2)$
3. **理论与实验一致**：确定性场景梯度复杂度提升 $\kappa \log\kappa$ 倍，实验也确认了显著加速
4. **T=1 即有效**：内层仅需 1 步近似 Newton 方向就能超越 10 步 GD，实际部署成本低
5. **框架通用性**：NBO 可轻松结合 CG、方差缩减（SAGA）、动量（MA）等技术

## 局限与展望
1. **需要初始点在 Newton 局部收敛域内**（BOX 1/BOX 2 条件），实际中可能需要额外预热步骤
2. **随机设定下改善有限**：NSBO-SGD 仅改善 $\log\kappa$ 倍，不如确定性设定显著
3. **下层需强凸**：虽然文中讨论了通过聚合函数扩展到凸情形（BAMM+NBO），但理论保证仅限强凸
4. **未考虑非光滑/约束情形**：对含约束或非光滑下层问题的扩展尚未探讨
5. **Hessian-向量积在深度网络中代价高**：对于大规模神经网络，计算精确 Hessian-向量积的实际开销仍然可观

## 相关工作与启发
- **SHINE**（Ramzi et al., 2022）：同样利用超梯度结构，但用拟 Newton 方法且缺少非渐近分析
- **AmIGO**（Arbel & Mairal, 2022a）：多步 GD 求解子问题的代表性框架
- **SOBA/SABA**（Dagréou et al., 2022）：单循环方法，NBO-GD 在 T=0 时退化为此
- **F2SA**（Kwon et al., 2023）：全一阶方法，无需 Hessian 但复杂度更高
- **启发**：共享 Hessian 的思路可推广至多级优化或含多个耦合子问题的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ — 结构洞察简洁优雅，非精确 Newton 融入双层优化是自然但新颖的组合
- 实验充分度: ⭐⭐⭐⭐ — 涵盖合成、超参数优化、数据清洗及多种扩展，基线全面
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、理论严谨、表格对比一目了然
- 价值: ⭐⭐⭐⭐ — 为双层优化提供了更高效的超梯度估计方法，理论和实践均有贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](sassha_sharpness-aware_adaptive_second-order_optimization_with_stable_hessian_ap.md)
- [\[ACL 2025\] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](tilted_sharpness-aware_minimization.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](../../NeurIPS2025/optimization/a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)

</div>

<!-- RELATED:END -->
