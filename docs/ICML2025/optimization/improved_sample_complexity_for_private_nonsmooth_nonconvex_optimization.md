---
title: >-
  [论文解读] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization
description: >-
  [ICML2025][优化/理论][差分隐私] 在差分隐私约束下研究非光滑非凸（NSNC）随机优化，通过改进梯度估计器的有效灵敏度，将已知最优样本复杂度降低了 $\Omega(\sqrt{d})$ 倍，并首次证明 Goldstein 稳定性可从经验损失泛化到总体损失。 - 问题设定： 考虑随机/经验目标函数 $F(x) =…
tags:
  - "ICML2025"
  - "优化/理论"
  - "差分隐私"
  - "非光滑非凸优化"
  - "Goldstein 稳定性"
  - "样本复杂度"
  - "零阶优化"
  - "方差缩减"
---

# Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization

**会议**: ICML2025  
**arXiv**: [2410.05880](https://arxiv.org/abs/2410.05880)  
**代码**: 无（理论工作）  
**领域**: 优化  
**关键词**: 差分隐私, 非光滑非凸优化, Goldstein 稳定性, 样本复杂度, 零阶优化, 方差缩减

## 一句话总结

在差分隐私约束下研究非光滑非凸（NSNC）随机优化，通过改进梯度估计器的有效灵敏度，将已知最优样本复杂度降低了 $\Omega(\sqrt{d})$ 倍，并首次证明 Goldstein 稳定性可从经验损失泛化到总体损失。

## 研究背景与动机

- **问题设定**: 考虑随机/经验目标函数 $F(x) = \mathbb{E}_{\xi \sim \mathcal{P}}[f(x;\xi)]$，其中分量函数 $f(\cdot;\xi)$ 既不光滑也不凸，这在深度学习中极为常见（ReLU、MaxPool 等）。
- **稳定性概念**: 非光滑情况下梯度范数最小化存在指数维度依赖的下界（Kornowski & Shamir, 2022），因此采用 **Goldstein 稳定性**——若 $x$ 的 $\alpha$-邻域内梯度的凸组合范数 $\leq \beta$，则 $x$ 是 $(\alpha,\beta)$-稳定点。
- **隐私需求**: 模型训练数据的隐私保护日益重要，差分隐私（DP）是标准形式化框架。核心问题是：在 $(ε,δ)$-DP 约束下，找到 $(\alpha,\beta)$-稳定点所需的最小样本量 $n$ 是多少？
- **已有结果**: Zhang et al. (2024) 给出了唯一的 NSNC DP 优化结果，样本复杂度为 $n = \widetilde{\Omega}\!\left(\frac{d}{\alpha\beta^3} + \frac{d^{3/2}}{\varepsilon\alpha\beta^2}\right)$，其中隐私代价项含 $d^{3/2}$，维度依赖偏高。
- **本文动机**: 能否降低维度依赖？能否用 ERM 泛化到总体损失？能否用一阶信息减少 oracle 复杂度？

## 方法详解

### 核心思想：改进梯度估计器的灵敏度

本文的关键洞察在于零阶梯度估计器的 **有效灵敏度** 可显著降低。标准估计器为：

$$\widetilde{\nabla} f_\alpha(x;\xi) = \frac{1}{m}\sum_{j=1}^{m} \frac{d}{2\alpha}\bigl(f(x+\alpha y_j;\xi) - f(x-\alpha y_j;\xi)\bigr) y_j$$

其中 $y_j \sim \text{Unif}(\mathbb{S}^{d-1})$。Zhang et al. 取 $m=d$，此时 mini-batch 灵敏度上界为 $\frac{Ld}{B}$。但本文观察到：当 $m$ 足够大时，亚高斯集中不等式保证 $\widetilde{\nabla} f_\alpha \approx \nabla f_\alpha$，从而灵敏度可降至 $\frac{L}{B}$，比原来小 $d$ 倍。这意味着隐私化所需的噪声大幅减少。

### 算法框架：O2NC + Tree Mechanism

算法基于 Cutkosky et al. (2023) 的 **Online-to-Non-Convex (O2NC)** 转换框架：

1. 维护增量 $\Delta_t$ 和迭代点 $x_t = x_{t-1} + \Delta_t$
2. 在随机插值点 $z_t = x_{t-1} + s_t \Delta_t$ 处查询梯度 oracle
3. 使用裁剪更新 $\Delta_{t+1} = \min\{1, D/\|\Delta_t - \eta \tilde{g}_t\|\} \cdot (\Delta_t - \eta \tilde{g}_t)$
4. 输出 $K$ 个窗口平均的随机选择

隐私通过 **树机制（Tree Mechanism）** 保障——用相关高斯噪声替代独立噪声，将误差从 $O(\sqrt{\Sigma})$ 降至 $O(\log \Sigma)$。

### 贡献 1：改进的单遍算法（Theorem 3.1）

单遍扫描数据集，$(ε,δ)$-DP，产出 $(\alpha,\beta)$-稳定点：

$$n = \widetilde{\Omega}\!\left(\frac{\sqrt{d}}{\alpha\beta^3} + \frac{d}{\varepsilon\alpha\beta^2}\right)$$

两项均比 Zhang et al. 降低了 $\Omega(\sqrt{d})$ 倍。特别是非隐私项为 $\sqrt{d}/(\alpha\beta^3)$，具有亚线性维度依赖（此前被错误声称不可能）。

### 贡献 2：多遍 ERM 算法（Theorem 4.1）

允许多次扫描数据，设计 $(ε,δ)$-DP 的 ERM 算法，产出经验损失的 $(\alpha,\beta)$-稳定点：

$$n = \widetilde{\Omega}\!\left(\frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}\right)$$

这是首个在 NSNC 目标下具有亚线性维度依赖样本复杂度的私有 ERM 算法。

### 贡献 3：ERM 到总体的泛化（Proposition 5.1）

首次证明 Goldstein 稳定性可从经验损失泛化到总体损失：经验损失的 $(\alpha,\hat\beta)$-稳定点以高概率是总体损失的 $(\alpha,\beta)$-稳定点，其中 $\beta = \hat\beta + \widetilde{O}(\sqrt{d/n})$。结合 ERM 算法，随机目标的样本复杂度为：

$$n = \widetilde{\Omega}\!\left(\frac{d}{\beta^2} + \frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}\right)$$

### 贡献 4：一阶算法降低 Oracle 复杂度（Theorem 6.1）

用一阶（梯度）oracle 替代零阶（函数值）oracle，保持相同样本复杂度，但 oracle 调用次数减少 $\widetilde{\Omega}(d^2)$ 倍。

## 理论结果对比

| 方法 | 类型 | 经验样本复杂度 | 随机样本复杂度 |
|------|------|---------------|---------------|
| Zhang et al. (2024) | 单遍 | $\frac{d}{\alpha\beta^3} + \frac{d^{3/2}}{\varepsilon\alpha\beta^2}$ | 同左 |
| **本文 Thm 3.1** | 单遍 | $\frac{\sqrt{d}}{\alpha\beta^3} + \frac{d}{\varepsilon\alpha\beta^2}$ | 同左 |
| **本文 Thm 4.1** | 多遍 | $\frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}$ | $\frac{d}{\beta^2} + \frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}$ |

> 表中省略了 Lipschitz 常数 $L$、初始化间隙 $\Phi$ 及对数因子。

## 亮点与洞察

- **灵敏度改进是核心**: 将零阶梯度估计器的有效灵敏度从 $O(Ld/B)$ 降至 $O(L/B)$，靠的是增大采样数 $m$ 后利用亚高斯集中，在高概率下"裁剪"即可获得更紧的灵敏度界。这一观察简洁有力。
- **纠正先前错误声明**: 之前工作声称非隐私项不可能低于线性维度依赖 $d$，本文证明可以达到 $\sqrt{d}$。
- **首次 Goldstein 稳定性泛化保证**: 以往 NSNC 文献缺少从 ERM 到总体的泛化结果，本文填补了这一空白，使 ERM 框架在隐私场景下有实际意义。
- **理论完整**: 从单遍到多遍、从零阶到一阶、从经验到总体，提供了一套系统化的改进。

## 局限与展望

- **纯理论工作**: 没有实验验证，所有结果仅在理论层面成立。实际深度学习场景中的效果未知。
- **常数和对数因子**: 样本复杂度中隐藏了 poly-log 因子和 Lipschitz 常数 $L$、初始间隙 $\Phi$ 的依赖，实际所需样本量可能较大。
- **下界缺失**: 未给出匹配的下界，不知道当前结果是否接近最优。
- **多遍算法需多次数据访问**: ERM 算法需要对数据进行多项式次扫描，在部分隐私场景中（如联邦学习、流式数据）可能不可行。
- **仅 Lipschitz 假设**: 虽然不要求光滑性，但仍需全局 Lipschitz 条件，这对某些实际模型可能不成立。
- **Goldstein 稳定性的实际意义**: 相比光滑情况下的梯度范数，$(\alpha,\beta)$-Goldstein 稳定性在实际中的可解释性较弱。

## 相关工作与启发

- **非光滑非凸优化**: Zhang et al. (2020) 开创了 Goldstein 稳定性视角；Cutkosky et al. (2023) 提出 O2NC 框架；Davis et al. (2022), Kornowski & Shamir (2024) 等进一步发展。
- **差分隐私优化**: 凸/光滑情况已有丰富文献（Bassily et al., 2014, 2019; Feldman et al., 2020）；NSNC 情况仅 Zhang et al. (2024) 有结果。
- **树机制**: Dwork et al. (2010), Chan et al. (2011) 提出的经典隐私机制，用于累积和的隐私化。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 灵敏度改进的洞察简洁而有效，泛化结果填补了理论空白
- 实验充分度: ⭐⭐ — 纯理论工作，无实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，从单遍到多遍逐步递进，符号一致性好
- 价值: ⭐⭐⭐⭐ — 系统性改进 NSNC DP 优化的样本复杂度，理论贡献扎实

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization](improved_last-iterate_convergence_of_shuffling_gradient_methods_for_nonsmooth_co.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](../../NeurIPS2025/optimization/private_zeroth-order_optimization_with_public_data.md)
- [\[ICML 2025\] Nearly Optimal Sample Complexity for Learning with Label Proportions](nearly_optimal_sample_complexity_for_learning_with_label_proportions.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] Lower Complexity Bounds for Nonconvex-Strongly-Convex Bilevel Optimization with First-Order Oracles](../../ICML2026/optimization/lower_complexity_bounds_for_nonconvex-strongly-convex_bilevel_optimization_with_.md)

</div>

<!-- RELATED:END -->
