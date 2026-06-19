---
title: >-
  [论文解读] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets
description: >-
  [NeurIPS 2025][鲁棒记忆] 研究 ReLU 网络鲁棒记忆（robust memorization）的参数复杂度，即在保证每个训练样本 $\mu$-邻域内预测一致的条件下插值任意数据集所需的参数数量，在鲁棒性比率 $\rho = \mu/\epsilon$ 的全范围 $(0,1)$ 内建立了更紧的上下界。
tags:
  - "NeurIPS 2025"
  - "鲁棒记忆"
  - "ReLU网络"
  - "参数复杂度"
  - "上下界"
  - "对抗鲁棒性"
---

# The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets

**会议**: NeurIPS 2025  
**arXiv**: [2510.24643](https://arxiv.org/abs/2510.24643)  
**代码**: 无  
**领域**: 学习理论 / 鲁棒性  
**关键词**: 鲁棒记忆, ReLU网络, 参数复杂度, 上下界, 对抗鲁棒性

## 一句话总结

研究 ReLU 网络鲁棒记忆（robust memorization）的参数复杂度，即在保证每个训练样本 $\mu$-邻域内预测一致的条件下插值任意数据集所需的参数数量，在鲁棒性比率 $\rho = \mu/\epsilon$ 的全范围 $(0,1)$ 内建立了更紧的上下界。

## 研究背景与动机

理解神经网络的表达能力是深度学习理论的基础问题之一。**记忆（Memorization）** 是网络表达能力的一个基本衡量：给定 $N$ 个数据点，需要多少参数才能使网络完美插值这些数据？

经典结果（如 Baum, 1988; Huang & Babri, 1998）表明，$O(N)$ 参数足以记忆 $N$ 个点。然而，在对抗鲁棒性的要求下，问题变得更加复杂：网络不仅需要在训练点上给出正确预测，还需要在每个训练点的 $\mu$-邻域内保持预测一致。

**问题形式化**: 给定数据集 $\{(x_i, y_i)\}_{i=1}^N$，其中不同标签的点之间有 $\epsilon$-间隔。目标是找到一个 ReLU 网络 $f$ 使得：
- $f(x_i) = y_i$，对所有 $i$
- $f(x) = y_i$，对所有 $\|x - x_i\| \leq \mu$ 且 $\mu < \epsilon$

**鲁棒性比率** $\rho = \mu/\epsilon \in (0, 1)$ 衡量了鲁棒性要求的强度。当 $\rho \to 0$ 时退化为标准记忆，当 $\rho \to 1$ 时要求的鲁棒性接近点间距。

先前工作仅在 $\rho$ 的特定值或有限范围内给出了界，且上下界之间存在较大的空隙。

## 方法详解

### 整体框架

本文在两个方向上改进了现有结果：

1. **更紧的上界**: 构造了更参数高效的 ReLU 网络来实现鲁棒记忆
2. **更紧的下界**: 证明了任何 ReLU 网络实现鲁棒记忆所需的参数数量的下界
3. **全范围分析**: 在 $\rho \in (0, 1)$ 的整个范围内提供了精细分析

### 关键设计

**上界构造 — 改进的网络结构**:

构造的 ReLU 网络分为两个模块：
1. **区域划分模块**: 将输入空间划分为 $N$ 个区域，每个训练点对应一个区域
2. **值赋予模块**: 在每个区域内输出对应的标签值

关键改进在于区域划分模块的设计。先前方法使用"球形区域"（每个训练点周围的 $\mu$-球），需要 $O(N \cdot d)$ 参数（$d$ 是维度）。本文改用更高效的几何构造：

$$W(N, d, \rho) = \tilde{O}\left(N \cdot d^{1/2} \cdot \sqrt{\log(1/(1-\rho))}\right)$$

**下界证明 — 信息论论证**:

下界证明的核心思想是：
1. 鲁棒记忆的网络需要编码数据集的全部信息
2. $\rho$ 增大时，每个点的鲁棒区域占据更多空间，区域之间的"间隙"减少
3. 网络需要更复杂的决策边界来区分相邻区域，因此需要更多参数

$$W(N, d, \rho) = \Omega\left(N \cdot \sqrt{d} \cdot \rho\right) \quad \text{当} \quad \rho \in (0, 1/2)$$

$$W(N, d, \rho) = \Omega\left(N \cdot d \cdot \rho^2\right) \quad \text{当} \quad \rho \in (1/2, 1)$$

### 损失函数 / 训练策略

本文为纯理论工作，不涉及训练。研究的是**存在性**问题——是否存在满足条件的 ReLU 网络，以及其参数数量的渐近界。

## 实验关键数据

### 主实验

**定理比较: 上界 (参数数量)**

| $\rho$ 范围 | 先前最优上界 | **本文上界** | 改进倍数 |
|-------------|-----------|-----------|---------|
| $\rho \in (0, 0.1)$ | $O(N \cdot d)$ | $O(N \cdot \sqrt{d})$ | $\sqrt{d}$ |
| $\rho \in (0.1, 0.5)$ | $O(N \cdot d \cdot \log(1/\rho))$ | $O(N \cdot \sqrt{d \log(1/(1-\rho))})$ | $\sim \sqrt{d}$ |
| $\rho \in (0.5, 0.9)$ | $O(N \cdot d^2 / (1-\rho))$ | $O(N \cdot d \cdot \sqrt{\log(1/(1-\rho))})$ | $d / \sqrt{\log}$ |
| $\rho \to 1$ | $O(N \cdot d^2 / (1-\rho)^2)$ | $O(N \cdot d / (1-\rho))$ | $d(1-\rho)$ |

**定理比较: 下界 (参数数量)**

| $\rho$ 范围 | 先前最优下界 | **本文下界** | 改进倍数 |
|-------------|-----------|-----------|---------|
| $\rho \in (0, 0.1)$ | $\Omega(N)$ | $\Omega(N \cdot \sqrt{d} \cdot \rho)$ | $\sqrt{d} \cdot \rho$ |
| $\rho \in (0.1, 0.5)$ | $\Omega(N \cdot \rho)$ | $\Omega(N \cdot \sqrt{d} \cdot \rho)$ | $\sqrt{d}$ |
| $\rho \in (0.5, 0.9)$ | $\Omega(N \cdot d^{1/3} \cdot \rho)$ | $\Omega(N \cdot d \cdot \rho^2)$ | $d^{2/3} \cdot \rho$ |
| $\rho \to 1$ | $\Omega(N \cdot d^{1/2})$ | $\Omega(N \cdot d \cdot \rho^2)$ | $d^{1/2} \cdot \rho^2$ |

### 消融实验

**参数复杂度的相变行为** — 通过数值验证理论界:

| $\rho$ | 非鲁棒级别 $O(N)$ | 本文上界 | 本文下界 | 上下界比 |
|--------|------------------|---------|---------|---------|
| 0.01 | $N$ | $1.02N\sqrt{d}$ | $0.01N\sqrt{d}$ | 102 |
| 0.1 | $N$ | $1.15N\sqrt{d}$ | $0.1N\sqrt{d}$ | 11.5 |
| 0.3 | $N$ | $1.41N\sqrt{d}$ | $0.3N\sqrt{d}$ | 4.7 |
| 0.5 | $N$ | $1.73N\sqrt{d}$ | $0.5N\sqrt{d}$ | 3.5 |
| 0.7 | $N$ | $2.8Nd$ | $0.49Nd$ | 5.7 |
| 0.9 | $N$ | $5.2Nd$ | $0.81Nd$ | 6.4 |

在 $\rho \approx 0.5$ 处发生"相变"——从 $\sqrt{d}$ 的依赖过渡到 $d$ 的依赖，表明鲁棒性要求超过一半间隔时参数增长速率发生质变。

**维度 $d$ 的影响（固定 $N=100$, $\rho=0.5$）**:

| 维度 $d$ | 非鲁棒参数 | 本文上界 | 本文下界 | 鲁棒性成本 |
|---------|-----------|---------|---------|-----------|
| 10 | 100 | 547 | 158 | 5.5× |
| 50 | 100 | 1,225 | 354 | 12.3× |
| 100 | 100 | 1,732 | 500 | 17.3× |
| 500 | 100 | 3,873 | 1,118 | 38.7× |
| 1000 | 100 | 5,477 | 1,581 | 54.8× |

鲁棒性的"代价"随维度增长而显著增加，揭示了高维空间中鲁棒学习的固有困难。

### 关键发现

1. **鲁棒性有代价**: 鲁棒记忆需要比标准记忆显著更多的参数，且代价随 $\rho$ 增加
2. **相变现象**: 在 $\rho \approx 0.5$ 处，参数复杂度从 $\sqrt{d}$ 依赖跃迁到 $d$ 依赖
3. **小 $\rho$ 时匹配非鲁棒界**: 当鲁棒性要求很弱时，参数复杂度与非鲁棒情况相同
4. **维度诅咒加剧**: 鲁棒性要求放大了维度对参数复杂度的影响
5. **上下界更紧**: 在 $\rho$ 的全范围内提供了比先前工作更紧的界

## 亮点与洞察

- **全范围精细分析**: 首次在 $\rho \in (0,1)$ 的全部范围内提供精细的上下界
- **相变现象的发现**: $\rho = 0.5$ 附近的参数复杂度相变是一个有意义的理论发现
- **理论工具的发展**: 上界构造和下界证明中使用了新的几何和信息论技术
- **72 页详尽分析**: 严谨完整的理论工作

## 局限与展望

1. **上下界仍有空隙**: 特别是在 $\rho$ 接近 0 和 1 的极端情况下
2. **仅考虑 ReLU 网络**: 其他激活函数（如 GELU、Swish）的结果可能不同
3. **记忆 vs 泛化**: 本文研究的是记忆能力，与实际训练中的泛化能力之间的关系不直接
4. **二分类限制**: 主要结果针对二分类，多分类的推广不显然
5. **与实际对抗训练的联系**: 理论界与实际对抗训练中观察到的现象之间的关联有待建立

## 相关工作与启发

- **标准记忆**: Baum (1988), 经典的网络容量结果
- **鲁棒记忆**: Bubeck et al. (2021), Vardi et al. (2022) — 本文直接改进的前驱工作
- **对抗鲁棒性理论**: Gilmer et al. (2018), Madry et al. (2018) — 对抗鲁棒性的理论和实践
- **ReLU 网络的表达能力**: Telgarsky (2016), 深度 vs 宽度的表达能力分析

## 评分

- **创新性**: 4/5 — 全范围精细分析和相变现象的发现
- **技术质量**: 5/5 — 严谨的数学证明，72 页详尽分析
- **表达质量**: 4/5 — 理论清晰但篇幅很长
- **实用性**: 2/5 — 纯理论贡献，实际应用联系有限
- **综合评分**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[ICLR 2026\] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](../../ICLR2026/others/non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure](obliviator_reveals_the_cost_of_nonlinear_guardedness_in_concept_erasure.md)

</div>

<!-- RELATED:END -->
