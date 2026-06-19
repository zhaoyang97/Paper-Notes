---
title: >-
  [论文解读] Global Convergence and Rich Feature Learning in $L$-Layer Infinite-Width Neural Networks under $\mu$P Parametrization
description: >-
  [ICML2025][优化/理论][μP参数化] 证明了在 $\mu$P (Maximal Update Parametrization) 下，$L$ 层无限宽 MLP 用 SGD 训练时，各层特征在整个训练过程中保持线性独立且发生实质性演化，从而保证训练收敛点必为全局最小值——首次同时解决"丰富特征学习"和"全局收敛"两个理论目标。
tags:
  - "ICML2025"
  - "优化/理论"
  - "μP参数化"
  - "无限宽网络"
  - "特征学习"
  - "全局收敛"
  - "高斯过程"
  - "张量程序"
  - "线性独立性"
---

# Global Convergence and Rich Feature Learning in $L$-Layer Infinite-Width Neural Networks under $\mu$P Parametrization

**会议**: ICML2025  
**arXiv**: [2503.09565](https://arxiv.org/abs/2503.09565)  
**代码**: 无  
**领域**: 优化理论 / 神经网络理论  
**关键词**: μP参数化, 无限宽网络, 特征学习, 全局收敛, 高斯过程, 张量程序, 线性独立性

## 一句话总结

证明了在 $\mu$P (Maximal Update Parametrization) 下，$L$ 层无限宽 MLP 用 SGD 训练时，各层特征在整个训练过程中保持线性独立且发生实质性演化，从而保证训练收敛点必为全局最小值——首次同时解决"丰富特征学习"和"全局收敛"两个理论目标。

## 研究背景与动机

深度学习理论的核心问题之一是：神经网络如何在非凸优化中既学到有意义的特征、又能全局收敛？

**NTK 参数化的局限**：在 NTK (Neural Tangent Kernel) 参数化下，无限宽网络的训练等价于线性模型，特征在训练过程中始终停留在初始化附近（$Z^{z_t}(\xi) = Z^{z_0}(\xi)$），无法进行真正的特征学习。虽然 NTK 能证明全局收敛，但本质上是"随机特征+线性回归"，无法解释实际网络的表示学习能力。

**Mean Field 参数化的问题**：Mean Field 方法在两三层网络上效果不错，但对超过 4 层的网络，特征向量和梯度退化为零向量，出现 feature collapse。即使 Integrable Parametrization (IP) 部分缓解了这一问题，深层网络在无限宽极限下仍从驻点出发，难以实现丰富的特征学习。

**标准参数化 (SP) 的限制**：SP 在宽度趋于无穷时需要学习率以 $O(1/\text{width})$ 衰减，同样无法在无限宽极限下进行特征学习。

本文的核心问题：**能否找到一种参数化方案，使深度网络同时实现有意义的特征学习和全局收敛？**

## 方法详解

### 整体框架

本文在张量程序 (Tensor Programs) 框架下研究 $L$ 层 MLP 在 $\mu$P 参数化下的训练动态。核心思路：

1. 利用 $Z$ 随机变量刻画无限宽极限下各层隐含状态的逐元素分布
2. 分析前向传播和反向传播诱导的两族高斯过程的结构不变量
3. 通过协方差保持性质和 GOOD 激活函数条件，逐层逐步归纳证明特征非退化性

### $\mu$P 参数化设计

$\mu$P 的关键区别在于各层的初始化方差和学习率缩放：

| 层             | 初始化方差        | 学习率              |
|:---------------|:-----------------|:-------------------|
| 输入层 $W^1$    | $1$              | $\eta \cdot n$     |
| 隐含层 $W^l$    | $n^{-1}$         | $\eta$             |
| 输出层 $W^{L+1}$| $n^{-2}$         | $\eta \cdot n^{-1}$ |

与 NTK/SP 相比，$\mu$P 的输入层学习率为 $\eta n$（远大于 NTK 的 $\eta$），使得各层参数获得 **最大化更新**，在宽度趋于无穷时仍保留非平凡的特征演化。

### 关键理论工具

**Z 随机变量表示**：在 $n \to \infty$ 时，隐含层向量 $h^l, x^l$ 的各分量趋于 i.i.d.，用标量随机变量 $Z^{h^l}, Z^{x^l}$ 刻画其渐近行为。两个向量的内积关系由期望编码：

$$\lim_{n\to\infty} x^\top y / n = \mathbb{E}[Z^x Z^y]$$

**特征演化的分解**：对任意特征 $z \in \{x^l, h^l\}$，有：

$$Z^{z_t}(\xi) = Z^{z_0}(\xi) + \underbrace{Z^{\delta z_1}(\xi) + \cdots + Z^{\delta z_t}(\xi)}_{\text{特征学习项}}$$

**高斯过程家族**：训练诱导两族高斯过程——前向过程 $\{\hat{Z}^{W_0^l \delta x_s^{l-1}(\xi_i)}\}$ 追踪特征演化，反向过程 $\{\hat{Z}^{W_0^{l\top} dh_s^l(\xi_i)}\}$ 描述梯度流动。

### 核心证明思路

**协方差保持性质**（关键发现）：前向和反向高斯过程的协方差结构在训练过程中保持不变：

$$\text{Cov}(\hat{Z}^{W_0^l \delta x_s^{l-1}(\xi)}, \hat{Z}^{W_0^l \delta x_t^{l-1}(\zeta)}) = \mathbb{E}[Z^{\delta x_s^{l-1}(\xi)} Z^{\delta x_t^{l-1}(\zeta)}]$$

这意味着相邻层之间的特征相关性遵循一致的模式，即使单个特征在大幅演化。

**GOOD 函数条件**：要求激活函数 $\phi$ 满足（sigmoid、tanh、SiLU 均满足）：
- 二阶连续可微，$\phi', \phi''$ 有界
- 对满足特定条件的参数 $\{a_i, b_i, c_i\}$，$f(x)=\sum_i a_i \phi(b_i x + c_i)$ 不是常函数
- $(r_1 + \phi(x))(r_2 + \phi'(x))$ 不是几乎处处常函数

**四步归纳证明**：
1. **第一隐含层特征** $\hat{Z}^{W_0^2 \delta x_s^1}$：基例，仅依赖输入
2. **其余层特征** $\hat{Z}^{W_0^l \delta x_s^{l-1}}$：利用前层非退化性逐层推进
3. **最后一层梯度** $\hat{Z}^{W_0^{L\top} dh_s^L}$：基于已建立的特征性质
4. **其余层梯度** $\hat{Z}^{W_0^{l\top} dh_s^l}$：完成反向传播分析

## 主要理论结果

### 定理 4.5（特征非退化性）

在假设 4.1（输入几何条件：$|\langle\xi_i,\xi_j\rangle| \neq |\langle\xi_i,\xi_k\rangle|$ 且 $|\langle\xi_i,\xi_j\rangle| \neq 0$）和假设 4.3（GOOD 激活函数）下：

> 对无限宽 $L$ 层 MLP 在梯度下降训练的任意时刻 $t$，每一层 $l \in [L]$ 的 **pre-activation 特征** $\{Z^{h_t^l(\xi)}\}_{\xi \in S}$ 和 **post-activation 特征** $\{Z^{x_t^l(\xi)}\}_{\xi \in S}$ 均保持线性独立。

### 推论 4.6（全局收敛）

> 若模型在时间 $T$ 收敛（权重不再变化），则对所有后续 mini-batch 中的样本，误差信号必定消失：$\mathring{\chi}_{T,i} = 0$，即收敛到训练目标的全局最小值。

### 理论意义

这是首次在允许实质性特征演化的条件下证明全局收敛。NTK 能证明收敛但无特征学习，Mean Field 有特征学习但深层退化——$\mu$P 是唯一同时实现两者的参数化方案。

## 实验验证

实验在 3 隐含层 MLP 上用 CIFAR-10 数据集验证，对比 SP、NTP、IP（Mean Field）和 $\mu$P：

| 参数化方案 | 特征变化量 | 特征多样性（最小特征值） | 时空特征联合非退化 |
|:----------|:---------|:---------------------|:----------------|
| SP        | 小（接近初始化）| 丰富 | 随宽度递减 |
| NTP       | 小（接近初始化）| 丰富 | 随宽度递减 |
| IP (Mean Field) | 大 | 低（feature collapse）| 随宽度递减 |
| **$\mu$P** | **大且稳定** | **丰富** | **随宽度稳定** |

**关键发现**：
- **特征变化量**：$\|h(x)-h^0(x)\|_2 / \|h^0(x)\|_2$ 衡量，仅 $\mu$P 在宽度增大时保持稳定的非零特征变化
- **特征多样性**：特征 Gram 矩阵 $K_{ij} = \langle h(\xi_i), h(\xi_j)\rangle$ 的最小特征值，$\mu$P 保持非退化
- **时空联合分析**：拼接初始和最终特征 $[h_1^0, h_1^T, \ldots, h_N^0, h_N^T]$ 后的 Gram 矩阵最小特征值，$\mu$P 在不同宽度下都保持较高值，其他方案均随宽度衰减

## 亮点与洞察

1. **首次统一两大目标**：在同一理论框架下同时证明丰富特征学习和全局收敛，解决了深度学习理论的一个基础性开放问题
2. **协方差不变量的发现**：揭示了前向和反向高斯过程的二阶统计量在训练中保持一致性，这是一个此前未被利用的结构性质
3. **GOOD 函数概念**：提出了优雅的激活函数正则性条件，涵盖 sigmoid、tanh、SiLU 等主流激活函数
4. **双层 filtration 框架**：$\mathcal{F}_t$ 和 $\mathcal{G}_t$ 的设计精巧地分离了前向和反向传播中的新随机性与历史信息
5. **对参数化方案的系统比较**：Table 1 清晰展示了不同方案在特征学习和特征丰富性上的权衡，$\mu$P 是唯一两者兼得的方案

## 局限与展望

1. **仅适用于 MLP**：未涉及 Transformer、CNN 等现代架构，尤其是注意力机制的特征学习分析
2. **ReLU 不满足条件**：最常用的 ReLU 激活函数不满足假设 4.3（$\phi''$ 不存在），是理论的重要遗漏
3. **仅证明收敛性，无收敛速率**：定理仅说明"若收敛则为全局最小值"，未提供收敛速率的定量刻画
4. **无限宽假设**：结论严格依赖 $n \to \infty$ 极限，有限宽度下的逼近质量未被讨论
5. **实验规模有限**：仅在 CIFAR-10 上用 3 层 MLP 验证，未在更大规模数据和更深网络上测试
6. **未讨论泛化性**：全局收敛保证的是训练误差，泛化误差的分析留作未来工作

## 相关工作与启发

- **NTK 理论** (Jacot et al., 2018; Du et al., 2019)：建立无限宽网络的线性化分析，但无法解释特征学习
- **Mean Field 分析** (Mei et al., 2018; Chizat & Bach, 2018)：两层网络的全局收敛，深层退化
- **张量程序** (Yang, 2019; Yang & Hu, 2020)：$\mu$P 的提出和无限宽极限的统一框架，本文核心技术基础
- **$\mu$P 实践** (Yang et al., 2021, 2023)：超参数迁移和谱分析，本文提供理论保障
- 本文的协方差不变量思路可能启发对连续深度网络 (Neural ODE) 和 Transformer 的类似分析

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次在 $\mu$P 下统一证明特征学习+全局收敛
- 实验充分度: ⭐⭐⭐ — 仅 CIFAR-10 + 3 层 MLP，规模有限但足以验证理论
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，技术路线阐述到位，符号记法稍显密集
- 价值: ⭐⭐⭐⭐⭐ — 对深度学习基础理论有重要推进意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](../../NeurIPS2025/optimization/quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[ICML 2025\] Random Feature Representation Boosting](random_feature_representation_boosting.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](../../NeurIPS2025/optimization/learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[NeurIPS 2025\] From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning](../../NeurIPS2025/optimization/from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)

</div>

<!-- RELATED:END -->
