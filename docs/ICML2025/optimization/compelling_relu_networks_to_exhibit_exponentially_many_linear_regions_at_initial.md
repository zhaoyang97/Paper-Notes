---
title: >-
  [论文解读] Compelling ReLU Networks to Exhibit Exponentially Many Linear Regions at Initialization and During Training
description: >-
  [ICML2025][优化/理论][ReLU网络] 提出一种基于非对称三角波的 ReLU 网络重参数化方法，使深度为 $d$ 的 4 神经元宽网络在初始化时即产生 $2^d$ 个线性区域，并在预训练中保持该指数级表达能力，在一维函数逼近任务上将误差降低了 3 个数量级：。 ReLU 网络的输出是分段线性函数…
tags:
  - "ICML2025"
  - "优化/理论"
  - "ReLU网络"
  - "线性区域"
  - "网络初始化"
  - "重参数化"
  - "三角波函数"
  - "指数表达能力"
---

# Compelling ReLU Networks to Exhibit Exponentially Many Linear Regions at Initialization and During Training

**会议**: ICML2025  
**arXiv**: [2311.18022](https://arxiv.org/abs/2311.18022)  
**代码**: 待确认  
**领域**: ReLU网络理论  
**关键词**: ReLU网络, 线性区域, 网络初始化, 重参数化, 三角波函数, 指数表达能力

## 一句话总结

提出一种基于非对称三角波的 ReLU 网络重参数化方法，使深度为 $d$ 的 4 神经元宽网络在初始化时即产生 $2^d$ 个线性区域，并在预训练中保持该指数级表达能力，在一维函数逼近任务上将误差降低了 **3 个数量级**。

## 研究背景与动机

ReLU 网络的输出是分段线性函数，理论上线性区域数量可随深度指数增长（Montufar et al., 2014），这是深度网络优于浅层网络的核心论据之一。然而实际中存在几个关键问题：

**随机初始化的低效性**：Hanin & Rolnick (2019) 证明随机初始化网络的平均线性区域数量与深度无关，仅取决于总神经元数，深度带来的指数级优势完全被浪费

**Dying ReLU 现象**：随机初始化可能导致整层神经元输出为负值，被 ReLU 截断后梯度完全消失；随着深度增加该现象概率递增（Lu et al., 2020）

**梯度下降的局限**：线性区域数量不是参数空间的局部性质，梯度下降难以直接优化该数量（如图 5 所示，相邻层的神经元输出若都不过零，微小扰动无法创造新区域）

**网络冗余**：Frankle & Carlin (2019) 表明约 95% 的权重可被剪枝而不显著降低精度

本文的核心动机是：能否通过数学构造而非随机初始化，强制 ReLU 网络从一开始就具有指数数量的线性区域？

## 方法详解

### 核心思路：非对称三角波重参数化

将深度 $d$、宽度 4 的 ReLU 网络的权重用三角函数的峰值位置 $a_i \in (0,1)$ 来参数化，取代直接操作原始权重。关键洞察：无论 $a_i$ 取何值，该参数化都保证产生 $2^d$ 个线性区域。

**三角函数定义**：每层关联一个非对称三角函数

$$T_i(x) = \begin{cases} \frac{x}{a_i} & 0 \leq x \leq a_i \\ 1 - \frac{x - a_i}{1 - a_i} & a_i \leq x \leq 1 \end{cases}$$

其中 $a_i$ 为峰值位置。每个三角函数由 2 个 ReLU 神经元实现。

**三角波的复合**：深层网络中，各层的三角函数进行复合：

$$W_i(x) = T_i \circ T_{i-1} \circ \cdots \circ T_0(x)$$

每复合一层，线性区域数翻倍，$W_i$ 具有 $2^i$ 个线性区域。

**网络输出**：将各层三角波加权求和作为最终输出：

$$F(x) = \sum_{i=0}^{\infty} s_i W_i(x)$$

其中 $s_i$ 为缩放系数。额外引入第 3 个"sum"神经元逐层累加，类似残差连接。第 4 个"bias"神经元用于替代指数衰减的偏置项，避免数值条件问题。

### 权重矩阵的具体形式

每个隐藏层的权重矩阵为 $4 \times 4$：

$$\begin{bmatrix} 1 & \pm S_i/a_i & -S_i/(a_i - a_i^2) & 0 \\ 0 & S_i/a_i & -S_i/(a_i - a_i^2) & 0 \\ 0 & S_i/a_i & -S_i/(a_i - a_i^2) & -S_i a_{i+1} \\ 0 & 0 & 0 & S_i \end{bmatrix}$$

其中 $S_i = s_i / s_{i-1}$ 为相邻缩放系数之比。行对应 sum、$t_1$、$t_2$、bias 四个神经元。

### 可微性正则化（定理 3.1）

为防止输出退化为分形曲线，要求 $F(x)$ 在无穷深度极限下可微。这导出了缩放系数的递推关系：

$$s_{i+1} = s_i (1 - a_{i+1}) \cdot a_{i+2}$$

即峰值位置 $a_i$ 唯一确定了缩放系数 $s_i$，减少了自由参数，同时引导优化器找到更光滑的解。

### 三阶段训练流程

1. **重参数化与初始化**：用三角峰值参数 $a_i$ 设置网络权重，保证 $2^d$ 个线性区域
2. **预训练**：在三角波参数空间中用 Adam 优化器训练，梯度通过原始权重回传更新 $a_i$，全程保持指数级线性区域
3. **标准训练**：切换回原始权重参数空间，释放约束，用梯度下降微调

### 扩展到非凸函数与高维

- **非凸函数**：用两个构造网络取差值 $f(x) = g_1(x) - g_2(x)$（差凸分解），二阶导有界的函数均可如此分解
- **多维函数**：将构造网络作为更大网络的激活函数，多个子网络沿不同维度操作后线性组合
- **稀疏块对角结构**：多个子网络组合时，隐藏层矩阵为块大小 4 的稀疏块对角矩阵

## 实验关键数据

### 一维凸函数逼近（500 点密集采样，4×5 网络，30 次取 min/mean）

| 方法 | Min MSE ($x^3$) | Min MSE ($x^{11}$) | Mean MSE ($x^3$) | Mean MSE ($x^{11}$) |
|------|-----------------|---------------------|-------------------|----------------------|
| Kaiming 初始化 | $2.11 \times 10^{-5}$ | $2.19 \times 10^{-5}$ | $7.20 \times 10^{-2}$ | $2.82 \times 10^{-2}$ |
| RAAI 初始化 | $2.14 \times 10^{-5}$ | $4.40 \times 10^{-5}$ | $3.97 \times 10^{-2}$ | $4.12 \times 10^{-2}$ |
| 跳过预训练 | $7.63 \times 10^{-7}$ | $1.86 \times 10^{-5}$ | $3.89 \times 10^{-5}$ | $3.56 \times 10^{-4}$ |
| 预训练(无正则) | $1.64 \times 10^{-7}$ | $3.20 \times 10^{-6}$ | $1.02 \times 10^{-5}$ | $3.73 \times 10^{-5}$ |
| **预训练+定理3.1** | **$7.86 \times 10^{-8}$** | **$8.86 \times 10^{-7}$** | **$5.27 \times 10^{-7}$** | **$7.87 \times 10^{-6}$** |

**关键观察**：完整方法相比 Kaiming 初始化，最小误差降低约 **3 个数量级**，平均误差降低 **5 个数量级**。

### 稀疏数据泛化能力（10 个训练点，10 个测试点）

| 方法 | Min MSE ($x^3$) | Min MSE ($x^{11}$) | Min MSE ($\sin x$) | Min MSE ($\tanh 3x$) |
|------|-----------------|---------------------|---------------------|----------------------|
| Kaiming 初始化 | $2.41 \times 10^{-4}$ | $2.14 \times 10^{-3}$ | $2.27 \times 10^{-5}$ | $1.60 \times 10^{-4}$ |
| **预训练+定理3.1** | **$5.65 \times 10^{-6}$** | **$6.53 \times 10^{-4}$** | **$7.92 \times 10^{-7}$** | **$5.09 \times 10^{-6}$** |

稀疏数据下依然有 1–2 个数量级优势，表明更多线性区域增强了对未见数据的预测能力。

### VGG-16 ImageNet 分类

替换 VGG-16 的全连接分类器（宽度 4096），仅增加约 0.5% 计算量：
- 早期训练具有优势，但最终精度与原始方法相当（均超过 PyTorch 报告的 73.3%）
- 分类任务中精确决策边界的作用可能不如回归任务关键

## 亮点与洞察

1. **优雅的数学构造**：将 ReLU 网络的权重空间约束到"三角波峰值位置"这一低维流形上，任意点都保证指数级线性区域，是一个非常巧妙的重参数化
2. **三个数量级的改进**：在一维函数逼近上取得了令人印象深刻的误差降低
3. **理论与实践的桥梁**：将 Telgarsky/Yarotsky 的理论构造（$x^2$ 逼近）推广为可训练的参数族
4. **可微性条件的自然推导**：定理 3.1 给出缩放系数的唯一确定关系，将正则化内化为网络结构
5. **与 KAN 的联系**：类似 Kolmogorov-Arnold 网络，用一维构造作为更大网络的激活函数实现多维逼近

## 局限与展望

1. **宽度限制**：核心构造仅适用于宽度 4 的网络，扩展到任意宽度尚缺理论框架
2. **分类任务收益有限**：VGG-16 实验表明在分类任务上优势不显著，实用价值需更多验证
3. **一维凸函数限制**：直接数学保证仅对一维凸函数成立，非凸和高维依赖启发式扩展
4. **可扩展性**：块对角结构在大规模网络中的效果未充分探索
5. **缺少与 KAN 等方法的直接比较**：作为同样利用一维构造的方法，缺乏与 KAN/spline 方法的实验对比
6. **预训练到标准训练的切换时机**：何时切换参数化缺乏理论指导

## 相关工作与启发

- **Telgarsky (2015)**：深度 ReLU 网络可产生指数数量线性段的对称三角波，本文推广到非对称三角波
- **Yarotsky (2017)**：用三角波复合逼近 $x^2$，本文将此构造推广为可训练的参数族
- **Hanin & Rolnick (2019)**：随机初始化网络的线性区域数量与深度无关的负面结果，是本文的直接动机
- **KAN (Liu et al., 2024)**：基于 Kolmogorov-Arnold 定理用一维激活函数构建多维逼近，思路相通
- **Elbrächter et al. (2019)**：探索条件良好的 ReLU 网络参数空间，与本文的重参数化思路互补

## 评分

- 新颖性: ⭐⭐⭐⭐ — 三角波重参数化保证指数线性区域的想法新颖优雅
- 实验充分度: ⭐⭐⭐ — 一维实验充分但大规模实验（ImageNet）仅初步展示
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，图示直观
- 价值: ⭐⭐⭐⭐ — 为 ReLU 网络初始化和表达能力提供了新的理论视角与实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Training Dynamics of In-Context Learning in Linear Attention](training_dynamics_of_in-context_learning_in_linear_attention.md)
- [\[NeurIPS 2025\] Better NTK Conditioning: A Free Lunch from ReLU Nonlinear Activation in Wide Neural Networks](../../NeurIPS2025/optimization/better_ntk_conditioning_a_free_lunch_from_relu_nonlinear_activation_in_wide_neur.md)
- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)
- [\[NeurIPS 2025\] Training Robust Graph Neural Networks by Modeling Noise Dependencies](../../NeurIPS2025/optimization/training_robust_graph_neural_networks_by_modeling_noise_dependencies.md)
- [\[ICLR 2026\] Conditioned Initialization for Attention](../../ICLR2026/optimization/conditioned_initialization_for_attention.md)

</div>

<!-- RELATED:END -->
