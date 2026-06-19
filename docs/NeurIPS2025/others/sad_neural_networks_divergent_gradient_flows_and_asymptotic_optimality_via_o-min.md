---
title: >-
  [论文解读] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures
description: >-
  [NeurIPS 2025][梯度流] 利用 o-minimal 结构的数学工具，证明了使用常见光滑激活函数（sigmoid、tanh、softplus、GELU 等）的全连接网络的梯度流存在二元性：要么收敛到临界点，要么发散到无穷大且损失收敛到渐近临界值。特别地，对多项式目标函数，证明了损失无法精确取零但可任意接近零，从而导致参数必然发散。
tags:
  - "NeurIPS 2025"
  - "梯度流"
  - "o-minimal 结构"
  - "发散现象"
  - "SAD 激活函数"
  - "渐近最优性"
---

# SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures

**会议**: NeurIPS 2025  
**arXiv**: [2505.09572](https://arxiv.org/abs/2505.09572)  
**代码**: [GitHub](https://github.com/deeplearningmethods/sad)  
**领域**: 深度学习理论 / 优化理论  
**关键词**: 梯度流, o-minimal 结构, 发散现象, SAD 激活函数, 渐近最优性

## 一句话总结

利用 o-minimal 结构的数学工具，证明了使用常见光滑激活函数（sigmoid、tanh、softplus、GELU 等）的全连接网络的梯度流存在二元性：要么收敛到临界点，要么发散到无穷大且损失收敛到渐近临界值。特别地，对多项式目标函数，证明了损失无法精确取零但可任意接近零，从而导致参数必然发散。

## 研究背景与动机

深度学习实践中，梯度优化方法常能达到接近零的训练损失，但理论上对非凸损失景观的收敛性理解仍不充分。核心问题是：

> 在什么条件下梯度流收敛到一个好的局部最小值？

现有理论多依赖强假设：凸性、过参数化、特定初始化、Łojasiewicz 不等式等，且几乎都隐含或显式假设参数轨迹有界。然而，越来越多的工作表明这一假设并不总成立——最简单的例子就是带 logistic 激活的线性分类器参数会发散到无穷。

本文的动机是弥合这一差距：放弃有界性假设，直接分析梯度流在无界情况下的行为。作者注意到一个关键的数学结构——神经网络训练中出现的函数都可以在 o-minimal 结构中"定义"，这赋予了它们强大的有限性和刚性。

## 方法详解

### 整体框架

本文有两个层次的结果：
1. **通用二元性定理**（Theorem 2.8）：适用于所有 $C^1$ 可定义激活函数的梯度流
2. **多项式目标的发散定理**（Corollary 3.6）：对 SAD 激活函数和多项式目标函数的特化

### 关键设计

1. **O-minimal 结构的应用**：O-minimal 结构是数学模型论中的概念，直觉上指可以用一阶逻辑和"良好操作"（加减乘除、指数、对数、导数、反导数）定义的集合/函数。关键性质是 $\mathcal{S}_1$ 中的集合仅为有限个点和区间的并——这赋予了极强的有限性保证。

   常见激活函数都是 Pfaffian 函数（偏导数是自身和多项式的复合），而 Wilkie (1999) 证明所有 Pfaffian 函数生成的结构是 o-minimal 的。这涵盖了 sigmoid、tanh、softplus、swish、GELU、Mish、ELU、softsign 等。

2. **二元性定理（Theorem 2.8）**：对于 $C^1$ 可定义的激活函数和损失函数，梯度流 $\Theta'(t) = -\nabla\mathcal{L}(\Theta(t))$ 有唯一全局解，且恰好满足以下之一：

    - **(a)** $\lim_{t\to\infty}\Theta(t)$ 存在且为临界点
    - **(b)** $\lim_{t\to\infty}\|\Theta(t)\|=\infty$ 且 $\lim_{t\to\infty}\mathcal{L}(\Theta(t))$ 为渐近临界值

   关键推论：存在 $\varepsilon > 0$ 使得任何损失初始值在最优值 $+\varepsilon$ 以内的梯度流最终损失都收敛到最优值。这一"吸引阈值"在深度学习文献中似乎未被注意到。

3. **SAD（Sublinear Analytic Definable）激活函数类**：定义三个性质：

    - **(S) 次线性**：$\limsup_{t\to\infty}\|f(tx)\|/t < \infty$
    - **(A) 解析性**：$f$ 是解析函数
    - **(D) 可定义性**：$f$ 在某个 o-minimal 结构中可定义

   sigmoid、tanh、softplus、swish、GELU、Mish 都是 SAD。ReLU 满足 (S)(D) 但不满足 (A)。SAD 函数具有良好的永久性：SAD 激活构成的神经网络仍然是 SAD。

4. **多项式目标函数的不可精确表示性（Theorem 3.4）**：对任何 $\deg(f)\geq 2$ 的多项式目标、SAD 激活和足够大的架构/数据集，证明 $\mathcal{L}(\theta) > 0$ 对所有 $\theta$ 成立。核心论证：

    - SAD 网络的次线性性阻止它在全局上等同于 $\geq 2$ 次多项式
    - 解析性和可定义性提供足够的"刚性"来在有限数据上检测这一性质
    - 但解析性保证有足够的非平凡 Taylor 系数，允许任意好的近似（Theorem 3.5：$\inf_\theta \mathcal{L}(\theta) = 0$）

   结合二元性定理：损失不能为零 + 可以任意接近零 → 最优值只能渐近实现 → 梯度流必然发散。

### 损失函数 / 优化设置

理论适用于经验损失（有限数据集）和连续分布上的期望损失（紧支撑密度函数）。覆盖的损失函数包括均方误差、二元交叉熵、Huber 损失等所有 $C^1$ 可定义损失。

## 实验关键数据

### 多项式目标函数实验

| 维度 | 激活函数 | 优化器 | 损失→0 | 参数范数→∞ |
|------|---------|--------|--------|-----------|
| 1D | sigmoid/tanh/softplus/swish/GELU | GD | ✓ | ✓（缓慢） |
| 2D | 同上 | GD | ✓ | ✓ |
| 4D | 同上 | GD | ✓ | ✓ |
| 1D | 同上 | Adam | ✓ | ✓（更快） |
| 2D | 同上 | Adam | ✓ | ✓ |
| 4D | 同上 | Adam | ✓ | ✓ |

### 复杂任务扩展实验

| 任务 | 激活 | 损失下降 | 参数增长 | 说明 |
|------|------|---------|---------|------|
| Heat PDE | GELU | ✓ | ✓ | Deep Kolmogorov 方法 |
| Black-Scholes PDE | GELU | ✓ | ✓ | 同上 |
| MNIST 分类 | GELU | ✓ | ✓ | 交叉熵损失 |

### 消融观察

| 对比 | 参数增长速度 | 损失收敛速度 |
|------|-------------|-------------|
| GD vs Adam | Adam 增长快得多 | Adam 收敛快得多 |
| 理论预测 $O(\sqrt{t})$ | 实际观察更接近对数增长 | 与 Lyu & Li 的对数结果一致 |

### 关键发现

- 梯度下降的参数增长非常缓慢（看起来像对数增长），这解释了实践中为何通常"感觉"参数有界
- Adam 由于自适应步长能维持更大的更新幅度，参数增长更明显
- 发散现象不限于多项式目标，在 PDE 求解和 MNIST 分类中同样观察到

## 亮点与洞察

- **将数理逻辑（o-minimal 理论）引入深度学习**：这是一个非常不寻常的跨领域连接。o-minimal 结构的"有限性定理"和"均匀有限性定理"是关键技术工具
- **SAD 函数类的定义优雅且实用**：精炼出次线性+解析+可定义三个性质，具有良好的封闭性，为未来光滑网络的理论研究提供便利框架
- **"吸引阈值" $\varepsilon$ 的存在**（Theorem 2.8(v)）：这意味着足够好的初始化必然导致损失收敛到最优值，不需要额外的收敛条件——这是一个之前被忽视的结果
- **ReLU 的特殊性**：理论明确指出 ReLU 不满足发散结论——浅层 ReLU 网络对多项式目标存在全局最小值。这解释了不同激活函数行为的根本差异

## 局限与展望

- **仅适用于梯度流**（连续时间），目前方法无法直接推广到（随机）梯度下降：关键障碍是缺乏对 Hessian 的控制
- 数值实验相对有限，主要作为概念验证
- 要求激活函数至少 $C^1$（部分结果需解析性），不适用于 ReLU
- L2 正则化或 weight decay（如 AdamW）会阻止发散——这意味着实践中这种现象可能较少出现
- 参数发散的方向性收敛（directional convergence）在基于指数函数的 o-minimal 结构中仍是开放问题

## 相关工作与启发

- Lyu & Li (2020)、Vardi et al. (2022) 的齐次网络发散结果是特例，本文提供了统一视角
- Kurdyka 的 Łojasiewicz 不等式在有界情况下保证收敛是经典结果；本文扩展到无界情况
- 对参数范数研究（implicit bias、weight norm 增长）有理论支撑意义
- 启发：实践中使用 weight decay 的一个合理性解释——它阻止了理论上必然的参数发散

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ o-minimal 结构在深度学习中的系统应用极具原创性
- 实验充分度: ⭐⭐⭐ 实验主要是概念验证，规模和多样性有限
- 写作质量: ⭐⭐⭐⭐ 数学严谨，但对不熟悉模型论的读者门槛较高
- 价值: ⭐⭐⭐⭐ 提供了理解梯度动力学的新理论视角，SAD 框架有后续研究价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Hessian-guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points](hessian-guided_perturbed_wasserstein_gradient_flows_for_escaping_saddle_points.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)

</div>

<!-- RELATED:END -->
