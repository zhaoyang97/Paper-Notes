---
title: >-
  [论文解读] Grokking at the Edge of Linear Separability
description: >-
  [ICML2025][优化/理论][grokking] 在最简单的逻辑回归二分类任务中揭示了 grokking（延迟泛化）的根本原因：当数据维度与样本数之比 $\lambda = d/N$ 接近临界点 $\lambda_c = 1/2$ 时，训练动力学会在过拟合解附近停留任意长时间后才收敛到泛化解，类似于物理学中的"临界减速"现象。
tags:
  - "ICML2025"
  - "优化/理论"
  - "grokking"
  - "延迟泛化"
  - "逻辑回归"
  - "线性可分性"
  - "临界现象"
  - "插值阈值"
  - "梯度下降隐式偏置"
---

# Grokking at the Edge of Linear Separability

**会议**: ICML2025  
**arXiv**: [2410.04489](https://arxiv.org/abs/2410.04489)  
**代码**: 无公开代码  
**领域**: 优化  
**关键词**: grokking, 延迟泛化, 逻辑回归, 线性可分性, 临界现象, 插值阈值, 梯度下降隐式偏置

## 一句话总结

在最简单的逻辑回归二分类任务中揭示了 grokking（延迟泛化）的根本原因：当数据维度与样本数之比 $\lambda = d/N$ 接近临界点 $\lambda_c = 1/2$ 时，训练动力学会在过拟合解附近停留任意长时间后才收敛到泛化解，类似于物理学中的"临界减速"现象。

## 研究背景与动机

**Grokking 现象**：Power et al. (2022) 在 Transformer 训练模运算任务时首次观察到——模型先达到完美训练精度但零泛化，继续训练后突然泛化。这违反了传统"过拟合不可逆"的认知。

**已有解释的不足**：先前工作将 grokking 归因于正则化、表示学习、电路形成等机制，但缺乏严格的数学分析框架。大多数可解模型过于复杂，难以精确定义"记忆"和"泛化"。

**本文动机**：在最简洁的逻辑回归模型中复现 grokking，给出严格的数学证明，揭示其与临界现象（phase transition）的深层联系。核心问题：**为什么 grokking 恰好发生在系统接近临界点时？**

## 方法详解

### 问题设定

考虑 $N$ 个训练样本 $\{\mathbf{x}_i\}_{i=1}^N \subset \mathbb{R}^d$，服从高斯分布 $\mathbf{x}_i \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_d)$，所有标签相同 $y_i = -1$。模型为带偏置的线性分类器：

$$f(\mathbf{x}_i) = \mathbf{S}^T \mathbf{x}_i + b$$

优化交叉熵损失：

$$\mathcal{L}(\mathbf{S}, b) = \frac{1}{N} \sum_{i=1}^{N} \log(1 + e^{\mathbf{S}^T \mathbf{x}_i + b})$$

关键参数：$\lambda = d/N$（维度与样本数之比）。

### 核心理论：临界点 $\lambda_c = 1/2$

**定义（从原点可分）**：数据集从原点线性可分，当且仅当存在 $\mathbf{S}$ 使得 $\mathbf{S}^T \mathbf{x}_i > 0$ 对所有 $i$ 成立。

**命题 3.3**（基于 Wendel 定理）：当 $N, d \to \infty$ 时：

- $\lambda > 1/2$：数据几乎必然从原点可分
- $\lambda < 1/2$：数据几乎必然不可分

**命题 3.2**（可分性决定泛化）：

- **不可分**（$\lambda < 1/2$）：$b(t) \to -\infty$，$\|\mathbf{S}\|$ 收敛到有限值 $\mathbf{S}_\infty$，模型完美泛化
- **可分**（$\lambda > 1/2$）：$b$ 和 $\|\mathbf{S}\|$ 同时发散，泛化精度受限于分离边际 $M$

$$\lim_{t \to \infty} \mathcal{A}_{\text{gen}} = \frac{1}{2}\left[1 + \text{erf}\left(\frac{1}{\sigma M \sqrt{2}}\right)\right] < 1$$

### Grokking 的动力学机制

通过引入"共形时间" $\tau(t) = \int_0^t e^{b(t')} dt'$，将带偏置的动力学映射到无偏置动力学：

$$\frac{\partial \mathbf{S}}{\partial \tau} = -\frac{\eta}{N} \sum_i e^{\mathbf{S}^T \mathbf{x}_i} \mathbf{x}_i$$

**命题 3.4**：当 $\lambda \to (1/2)^-$ 时，$\|\mathbf{S}_\infty\| \to \infty$。

**命题 3.5**：当 $\sigma$ 足够大时，$\mathbf{S}(t)$ 可以任意快地接近 $\mathbf{S}_\infty$。

**Grokking 机制**：在 $\lambda$ 略小于 $1/2$ 且 $\sigma$ 大的条件下：

1. $\|\mathbf{S}_\infty\|$ 极大 → $\mathbf{S}(t)$ 快速增长到很大值
2. $b(t)$ 的增长速率有界 → $|b|/\|\mathbf{S}\|$ 在很长时间内保持较小
3. 泛化精度 $\mathcal{A}_{\text{gen}} \approx \frac{1}{2}[1 - \text{erf}(b/(\sigma\|\mathbf{S}\|\sqrt{2}))]$ 长时间停滞在低水平
4. 最终 $\mathbf{S}$ 饱和、$b$ 持续增长才实现泛化 → 延迟泛化 = grokking

### 简化一维模型

构造 $N=2$ 的极简模型（$x_1 = -1, x_2 = 1 - 2\lambda$），完全解析求解三个动态区域：

| 区域 | $b(t \gg 1)$ | $\|\mathbf{S}\|(t \gg 1)$ |
|------|-------------|--------------------------|
| $\lambda < 1/2$（不可分） | $-\log(t)$ | $\frac{1}{2(1-\lambda)}\log\frac{1}{1-2\lambda}$（有限） |
| $\lambda = 1/2$（临界） | $-\log(t)$ | $\log(\log(t))$（对数发散） |
| $\lambda > 1/2$（可分） | $-\frac{1}{1+M^2}\log(t)$ | $\frac{M}{1+M^2}\log(t)$（发散） |

## 实验关键数据

| 实验设定 | 参数 | 关键发现 |
|----------|------|----------|
| 主实验 | $N=4\times10^4$, $\sigma=5$, $\eta=0.01$ | $\lambda=0.48$ 时出现显著 grokking；$\lambda=0.52$ 时永不泛化 |
| 泛化精度 | 不同 $\lambda$ | $\lambda > 0.5$ 时终态精度与边际 $M$ 的理论预测完美吻合 |
| Grokking 时间 | $\lambda \to 0.5^-$ | grokking 时间发散，符合临界减速预测 |
| $\|\mathbf{S}_\infty\|$ 发散 | $\lambda \to 0.5^+$ | $\|\mathbf{S}_\infty\|$ 在 $\lambda = 0.5$ 处发散，确认临界点 |
| 简化模型 | 变化 $\sigma$ 和 $1-2\lambda$ | grokking 需要 $\lambda \to 1/2$ **且** $\sigma \to \infty$ 两个条件 |

## 亮点与洞察

1. **极简模型中的深刻发现**：在最简单的逻辑回归中复现 grokking，证明它不需要复杂网络、正则化或特殊数据结构
2. **临界现象视角**：将 grokking 与物理学临界减速（critical slowing down）精确类比，提供全新解释框架
3. **严格数学证明**：对泛化/过拟合的分界条件、grokking 时间发散等均给出rigorous proof
4. **共形时间技巧**：通过时间重参数化将有偏置问题映射到无偏置问题，技术上优雅
5. **可分性作为统一判据**：将"从原点可分性"确立为泛化的充要条件，概念简洁有力

## 局限与展望

1. **模型过于简单**：逻辑回归 → 真实神经网络的推广尚不明确，非线性模型中临界点的形式可能完全不同
2. **高斯数据假设**：真实数据分布远比高斯复杂，临界点可能不存在简洁的 $\lambda_c = 1/2$ 闭式解
3. **无正则化设定**：现实中 grokking 通常伴随 weight decay 等正则化，本文的无正则化分析需要扩展
4. **缺少对 Transformer/模运算任务的直接联系**：与原始 grokking 场景（Power et al.）的桥梁需要进一步构建
5. **无有限宽度网络分析**：仅考虑线性模型，非线性激活函数和深层网络中的表示学习效应未探讨
6. **SGD/Adam 分析不完整**：主要结果基于梯度流极限，附录中的 Adam 讨论仅为初步观察
7. **实际指导意义有限**：虽然理论优雅，但对实践中如何预测和利用 grokking 缺乏直接建议

## 相关工作与启发

- **Power et al. (2022)**：首次发现 grokking
- **Soudry et al. (2018)**：梯度下降在可分数据上的隐式偏置 → 本文的理论基础
- **Levi et al. (2023)**：线性回归中 grokking 的完整动力学解
- **Gromov (2023)**：可解模型中的 grokking 与潜空间结构
- **Wendel (1962)**：随机点集可分性的经典定理 → 本文的 $\lambda_c = 1/2$ 的来源

## 评分

- 新颖性: ⭐⭐⭐⭐ 临界现象视角解释 grokking 是全新且优雅的
- 实验充分度: ⭐⭐⭐⭐ 理论与实验相互验证，简化模型有完整解析解
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，物理直觉到位，结构逻辑性强
- 价值: ⭐⭐⭐⭐ 对理解 grokking 机制有重要贡献，但向深度网络推广仍需后续工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product](emergence_in_non-neural_models_grokking_modular_arithmetic_via_average_gradient_.md)
- [\[ICLR 2026\] Generalization Below the Edge of Stability: The Role of Data Geometry](../../ICLR2026/optimization/generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)
- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[ICML 2025\] Compelling ReLU Networks to Exhibit Exponentially Many Linear Regions at Initialization and During Training](compelling_relu_networks_to_exhibit_exponentially_many_linear_regions_at_initial.md)
- [\[ICML 2026\] Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization](../../ICML2026/optimization/conflicting_biases_at_the_edge_of_stability_norm_versus_sharpness_regularization.md)

</div>

<!-- RELATED:END -->
