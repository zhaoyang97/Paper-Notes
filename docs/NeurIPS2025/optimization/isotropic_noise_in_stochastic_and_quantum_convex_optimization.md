---
title: >-
  [论文解读] Isotropic Noise in Stochastic and Quantum Convex Optimization
description: >-
  [NeurIPS 2025][优化/理论][stochastic convex optimization] 本文引入各向同性随机梯度预言机（ISGO）概念——噪声在每个方向上都以高概率有界——并设计随机切平面算法达到 $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$ 的查询复杂度，较 SGD 在某些参数区间改进 $d$ 倍，作为推论获得了 sub-exponential 噪声下的新 SOTA 复杂度，并通过量子各向同性化子程序改进了量子随机凸优化的维度依赖。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "stochastic convex optimization"
  - "isotropic noise"
  - "cutting plane"
  - "sub-exponential distribution"
  - "quantum optimization"
---

# Isotropic Noise in Stochastic and Quantum Convex Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.20745](https://arxiv.org/abs/2510.20745)  
**代码**: 无  
**领域**: 优化  
**关键词**: stochastic convex optimization, isotropic noise, cutting plane, sub-exponential distribution, quantum optimization

## 一句话总结
本文引入各向同性随机梯度预言机（ISGO）概念——噪声在每个方向上都以高概率有界——并设计随机切平面算法达到 $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$ 的查询复杂度，较 SGD 在某些参数区间改进 $d$ 倍，作为推论获得了 sub-exponential 噪声下的新 SOTA 复杂度，并通过量子各向同性化子程序改进了量子随机凸优化的维度依赖。

## 研究背景与动机
**经典结果**：SGD 在 $\sigma_B$-有界随机梯度预言机（BSGO）下达到 $O(R^2\sigma_B^2/\epsilon^2)$，这是最优的。当噪声以方差 $\sigma_V^2$ 刻画（VSGO）时，SGD 达到 $O(R^2(\sigma_V^2 + L^2)/\epsilon^2)$，其中 $R^2\sigma_V^2/\epsilon^2$ 项不可改进，但 $R^2L^2/\epsilon^2$ 项来自非随机部分，在精度足够高时可以用切平面方法改进到 $\tilde{O}(d)$。

**Open Problem**：能否在 VSGO 下达到 $\tilde{O}(R^2\sigma_V^2/\epsilon^2 + d)$？即将随机与非随机贡献解耦？

**本文贡献**：定义 ISGO 噪声模型，在该模型下肯定回答了上述问题；对 sub-exponential 噪声达到完全匹配的复杂度；对一般 VSGO 达到 $\tilde{O}(dR^2\sigma_V^2/\epsilon^2 + d)$。

## 方法详解

### 噪声模型层次

**BSGO（有界随机梯度）**：$\mathbb{E}\|\mathcal{O}_B(x)\|^2 \leq \sigma_B^2$

**VSGO（方差有界）**：$\mathbb{E}\|\mathcal{O}_V(x) - \nabla f(x)\|^2 \leq \sigma_V^2$

**ISGO（各向同性噪声，本文新定义）**：

$$\Pr[|\langle \mathcal{O}_I(x) - \nabla f(x), u \rangle| \geq \sigma_I/\sqrt{d}] \leq \delta, \quad \forall \|u\|=1$$

$1/\sqrt{d}$ 归一化确保 $\sigma_I$ 与 $\sigma_V$ 量级可比（$(σ,0)$-ISGO 是 $\sigma$-VSGO）。

**ESGO（sub-exponential 噪声）**：$\Pr[|\langle \mathcal{O}_E(x) - \nabla f(x), u \rangle| \geq t] \leq 2\exp(-t\sqrt{d}/\sigma_E)$

**层次关系**：ESGO → ISGO → VSGO（每步有对数或 $\sqrt{d}$ 因子的转换代价）

### 核心算法：随机切平面方法

#### Step 1: 边际近似梯度预言机（MAGO）
定义 MAGO 为在指定方向 $u$ 上误差有界 $\eta$，在 $\ell_2$ 范数上误差有界 $\Gamma$ 的梯度估计器：

$$\|\tilde{g}(x,u) - \nabla f(x)\| \leq \Gamma, \quad |\langle \tilde{g}(x,u) - \nabla f(x), u/\|u\| \rangle| \leq \eta$$

**关键引理（Lemma 1）**：用 $K = O(\frac{\sigma_I^2}{d\eta^2}\log(2d/\xi) + 1)$ 次 ISGO 查询可实现 $(\eta, \eta\sqrt{d})$-MAGO，且**不需要知道方向 $u$**。

证明思路：对 ISGO 输出取平均，利用 Hoeffding 不等式在标准正交基每个方向上控制误差，再由并联不等式推出。

#### Step 2: 切平面方法 + MAGO
运行切平面方法（如重心法），用 $-\tilde{g}(x_t, x_t - x^\star)$ 作为半空间预言机。关键分析：

$$f(x_t) \leq f(z) + \underbrace{\langle g_t, x_t - z \rangle}_{< 0} + \underbrace{\langle \nabla f(x_t) - g_t, x_t - x^\star \rangle}_{\leq 4R\eta} + \underbrace{\langle \nabla f(x_t) - g_t, x^\star - z \rangle}_{\leq \Gamma r \leq \epsilon}$$

第二项只需在 $x_t - x^\star$ 方向控制误差 → 省去 $\sqrt{d}$ 因子；第三项利用 $r$ 的选取（$r = \min\{\epsilon/(2L), \epsilon/(4\Gamma)\}$）吸收大的 $\ell_2$ 误差。

#### Step 3: 后处理选择最优候选
切平面返回 $T = O(d\log(d + RL/\epsilon))$ 个候选点，保证至少一个 $\epsilon$-最优。通过二分搜索 + ISGO 查询找到最优点，总复杂度 $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$。

### 主要定理

**Theorem 1（ISGO 上界）**：$\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$ 次查询解决 SCO。

**Corollary 2（ISGO 下界）**：$\tilde{\Omega}(R^2\sigma_I^2/(\epsilon^2\log^2(1/\delta)) + \min\{R^2L^2/\epsilon^2, d\})$。

**Corollary 3（ESGO）**：$\tilde{O}(R^2\sigma_E^2/\epsilon^2 + d)$，匹配下界 $\tilde{\Omega}(R^2\sigma_E^2/\epsilon^2 + d)$（Theorem 5）。

**Corollary 4（VSGO）**：$\tilde{O}(dR^2\sigma_V^2/\epsilon^2 + d)$，多一个 $d$ 因子。

### 量子扩展

**Theorem 6（量子各向同性化）**：用 $\tilde{O}(\sigma_V\sqrt{d}\log^7(1/\delta)/\sigma_I)$ 次 QVSGO 查询构造 ISGO。

**Theorem 7（量子 SCO）**：$\tilde{O}(dR\sigma_V/\epsilon)$ 次查询，优于先前的 $\tilde{O}(d^{3/2}\sigma_V R/\epsilon)$，改进了 $\sqrt{d}$ 因子。

量子各向同性化的核心技巧：用无偏相位估计算法（boosted unbiased phase estimation）替代标准相位估计，避免了偏差在某些方向上的累积（标准方法会引入 $\sqrt{d}$ 额外开销）。然后通过 MLMC（多级蒙特卡罗）去偏差并保持各向同性噪声性质。

## 实验关键数据

本文为纯理论工作，无实验。核心结果以复杂度界汇总：

| 噪声模型 | SGD 复杂度 | 本文复杂度 | 改进条件 |
|----------|-----------|-----------|---------|
| ISGO | $O(R^2\sigma_I^2/\epsilon^2 + R^2L^2/\epsilon^2)$ | $\tilde{O}(R^2\sigma_I^2/\epsilon^2 + d)$ | $L \gg \max\{\sigma_I, \epsilon\sqrt{d}/R\}$ |
| ESGO | 同 VSGO | $\tilde{O}(R^2\sigma_E^2/\epsilon^2 + d)$ | 同上 |
| VSGO | $O(R^2\sigma_V^2/\epsilon^2 + R^2L^2/\epsilon^2)$ | $\tilde{O}(dR^2\sigma_V^2/\epsilon^2 + d)$ | $L \gg \sqrt{d}\max\{\sigma_V, \epsilon/R\}$ |

| 量子 SCO | 先前 SOTA | 本文 |
|----------|----------|------|
| QVSGO | $\tilde{O}(d^{3/2}\sigma_V R/\epsilon)$ | $\tilde{O}(dR\sigma_V/\epsilon)$ |

## 亮点与洞察
- **ISGO 噪声模型的价值**：捕捉了噪声"形状"的信息（不仅是矩），允许在切平面方法中只控制一个方向的误差，自然获得 $d$ 因子改进
- **MAGO 不需要知道方向的妙处**：虽然理论上需要控制朝 $x^\star$ 方向的误差（而 $x^\star$ 未知），但通过 ISGO 的各向同性性，取平均后自动满足所有方向的控制
- **sub-exponential 噪声的完整解决**：上下界匹配（多对数因子内），有效地解决了 Open Problem 1 在 sub-exponential 特例下的情形
- **量子各向同性化的独立价值**：作为通用子程序，可将方差有界量子预言机转为各向同性预言机，可能在其他量子优化问题中有应用

## 局限与展望
- **Open Problem 1 在一般 VSGO 下未解决**：VSGO 情形多了 $d$ 因子，$\tilde{O}(R^2\sigma_V^2/\epsilon^2 + d)$ 是否可达仍是开放问题
- **切平面方法的实用性**：重心法等切平面方法的计算复杂度高（维护凸体表示），实际中不如 SGD 实用
- **polylog 因子隐藏**：$\tilde{O}$ 记号隐藏了 $\log(1/\epsilon), \log d, \log(1/\delta)$ 等因子，实际常数可能不小
- **ISGO 假设的实际适用性**：实际问题中随机梯度噪声是否满足各向同性条件需要逐案验证
- **量子结果的实用门槛**：量子随机凸优化的实际量子硬件实现仍遥远

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ ISGO 噪声模型和 MAGO 概念是全新的，将噪声"形状"引入 SCO 复杂度分析
- 理论深度: ⭐⭐⭐⭐⭐ 上下界匹配、量子各向同性化、MLMC 去偏差保各向同性——技术功底极强
- 实验充分度: ⭐ 纯理论工作
- 写作质量: ⭐⭐⭐⭐ 开放问题驱动的叙事清晰，技术概览有效；但符号和定义较多
- 价值: ⭐⭐⭐⭐ 对 SCO 的基础理论有重要推进，量子部分达到新 SOTA

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise](unveiling_m-sharpness_through_the_structure_of_stochastic_gradient_noise.md)
- [\[ICML 2025\] Enhancing Parallelism in Decentralized Stochastic Convex Optimization](../../ICML2025/optimization/enhancing_parallelism_in_decentralized_stochastic_convex_optimization.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](../../ICML2025/optimization/quantum_optimization_via_gradient-based_hamiltonian_descent.md)

</div>

<!-- RELATED:END -->
