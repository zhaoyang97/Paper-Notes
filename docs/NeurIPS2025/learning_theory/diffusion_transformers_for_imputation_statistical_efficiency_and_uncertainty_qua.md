---
title: >-
  [论文解读] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification
description: >-
  [NeurIPS 2025][时间序列 / 生成模型理论][扩散模型] 本文从统计学习角度分析了条件扩散Transformer（DiT）在时间序列插补任务中的样本复杂度和不确定性量化性能，并提出混合掩码训练策略提升插补效果。 时间序列数据在金融、医疗、交通、气象等领域无处不在，但常因传感器故障、数据传输错误等原因存在大量缺失…
tags:
  - "NeurIPS 2025"
  - "时间序列 / 生成模型理论"
  - "扩散模型"
  - "Transformer"
  - "时间序列插补"
  - "不确定性量化"
  - "统计学习理论"
---

# Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification

**会议**: NeurIPS 2025  
**arXiv**: [2510.02216](https://arxiv.org/abs/2510.02216)  
**代码**: [有](https://github.com/liamyzq/DiT_time_series_imputation)  
**领域**: 时间序列 / 生成模型理论  
**关键词**: 扩散模型, Transformer, 时间序列插补, 不确定性量化, 统计学习理论

## 一句话总结

本文从统计学习角度分析了条件扩散Transformer（DiT）在时间序列插补任务中的样本复杂度和不确定性量化性能，并提出混合掩码训练策略提升插补效果。

## 研究背景与动机

时间序列数据在金融、医疗、交通、气象等领域无处不在，但常因传感器故障、数据传输错误等原因存在大量缺失值。缺失值会严重影响下游任务性能，因此准确的插补至关重要。

传统统计方法（均值插补、插值、卡尔曼滤波、ARIMA等）依赖线性和平稳性等强假设，难以处理复杂非线性数据。近年来，基于扩散模型的生成式插补方法（如CSDI）展现出优越的经验性能，但仍存在两个关键问题：

1. 扩散模型在不同数据集上表现差异大，性能不稳定
2. 插补质量受缺失模式（missing pattern）影响显著

本文的核心问题：**扩散模型能多好地捕捉缺失值的条件分布？缺失模式如何影响插补性能？**

## 方法详解

### 整体框架

本文以高斯过程（GP）数据为理论分析对象，研究DiT在插补任务中的统计效率。核心思路是将插补建模为条件分布估计问题：给定观测序列 $x_{\text{obs}}$，估计缺失值的条件分布 $P(x_{\text{miss}} | x_{\text{obs}})$。

**数据模型**：考虑 $d$ 维高斯过程，序列长度 $H$，联合分布为 $\mathcal{N}(\mu, \Gamma \otimes \Lambda)$，其中 $\Gamma$ 表示时间相关性，$\Lambda$ 表示空间依赖性。

### 关键设计

**1. 条件得分函数的Transformer逼近理论（定理1）**

作者提出了一种新的DiT构造性证明，利用算法展开（algorithm unrolling）技术，证明Transformer能有效逼近高斯过程的条件得分函数。关键步骤包括：
- 利用正交基分解条件得分函数
- 通过注意力机制实现位置嵌入和时间依赖性捕获
- MLP层实现非线性变换

**2. 统计样本复杂度（定理2）**

建立了DiT学习条件分布的样本复杂度上界：
$$\tilde{O}\left(\frac{\sqrt{Hd^2\kappa^5}}{\sqrt{n}}\right)$$
其中 $n$ 是训练样本量，$H$ 是序列长度，$d$ 是维度，$\kappa$ 是由缺失模式决定的条件协方差矩阵的条件数。关键发现：
- 收敛速率为 $n^{-1/2}$，对序列长度 $H$ 仅有温和的多项式依赖
- 条件数 $\kappa$ 直接刻画缺失模式对插补难度的影响

**3. 不确定性量化（推论1）**

利用训练好的DiT生成大量缺失值样本，构建置信区间（CR）。证明覆盖概率以 $\tilde{O}(n^{-1/2})$ 速率收敛到期望水平。

### 损失函数 / 训练策略

**混合掩码训练策略（Mixed-Masking Training Strategy）**

受理论分析启发，提出混合不同缺失模式的训练策略：
- **S1**: 100% 随机缺失（16×1）
- **S2**: 50% 随机 + 50% 弱分组（8×2）
- **S3**: 33% 随机 + 33% 弱分组 + 33% 中分组（4×4）
- **S4**: 25% 随机 + 25% 弱分组 + 25% 中分组 + 25% 强分组（1×16）

核心思想：训练时引入从易到难的多种缺失模式，缩小训练与测试分布之间的分布偏移。

## 实验关键数据

### 主实验

**高斯过程数据上的置信区间覆盖率（表1-2）**

| 序列长度 H | 16 | 32 | 64 | 96 | 128 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CR 覆盖率(%) | 92.67 | 88.63 | 82.14 | 80.25 | 77.81 |

不同训练策略在不同缺失模式下的CR覆盖率（%）：

| 策略 | P1 (κ=415) | P2 (κ=30) | P3 (κ=9.5) | P4 (κ=3.0) |
|:---:|:---:|:---:|:---:|:---:|
| S1 (纯随机) | 34.58 | 58.46 | 72.42 | 80.25 |
| S4 (混合) | 57.27 | 79.00 | 74.38 | 82.74 |

**潜在高斯过程上的MSE比较（表3）**

| 模型 | P1-S4 | P2-S4 | P3-S4 | P4-S4 |
|:---:|:---:|:---:|:---:|:---:|
| DiT | **0.67** | **0.62** | **0.58** | **0.53** |
| CSDI | 0.68 | 0.63 | 0.61 | 0.58 |
| GPVAE | 5.28 | 4.84 | 4.59 | 4.45 |

**真实数据集MAE结果（表6，附录）**

| 模型 | ETT_m1 10% | ETT_m1 50% | BeijingAir 10% | BeijingAir 50% |
|:---:|:---:|:---:|:---:|:---:|
| DiT | **0.1269** | **0.1543** | **0.1753** | **0.2057** |
| CSDI | 0.1448 | 0.1650 | 0.1780 | 0.2141 |
| GP-VAE | 0.2786 | 0.4666 | 0.4152 | 0.5265 |

### 消融实验

混合掩码策略的消融：
- 仅使用单一模式（8×2、4×4、1×16）的性能均不如混合策略
- 分布偏移系数分析：S4 的分布偏移系数比 S1 低约 **47.93 倍**，提供强理论支持

### 关键发现

1. **条件数是关键指标**：条件数 $\kappa$ 越低（缺失点间隔越远），插补越容易，所需样本越少
2. **混合掩码训练一致性优势**：混合策略在所有缺失模式上均优于纯随机掩码训练
3. **DiT一致优于CSDI**：在MSE和CR覆盖率上均领先，说明Transformer架构更适合此任务
4. **理论预测与实验吻合**：序列长度增加导致覆盖率下降，低条件数模式更易估计

## 亮点与洞察

1. **首次为扩散Transformer插补提供端到端统计保证**：不仅分析分布估计，还涵盖不确定性量化
2. **理论驱动方法设计**：混合掩码策略直接由分布偏移理论结果启发，非经验调参
3. **算法展开构造证明**：创新性地用算法展开技术构造Transformer逼近条件得分函数
4. **置信区间构造自然**：利用生成模型的采样能力直接构建置信区间，方法简洁有效

## 局限与展望

1. 理论分析局限于高斯过程数据，对重尾分布（如金融数据）的适用性待研究
2. 最优混合掩码比例是实例相关的，目前没有自适应选择方法
3. 实验主要在合成数据和小规模真实数据上验证，大规模真实场景的验证不足
4. 仅考虑了块缺失（block-missing）设定，随机散点缺失的分析未涉及

## 相关工作与启发

- **CSDI** [Tashiro et al., 2021]：首个条件扩散时间序列插补方法，本文的主要对比基线
- **DiT** [Peebles and Xie, 2022]：扩散Transformer架构，本文的骨干网络
- **GPVAE** [Fortuin et al., 2020]：基于VAE的生成式插补，实验表明远不如扩散方法
- 扩散模型理论 [Chen et al., 2023; Fu et al., 2024]：为本文理论分析提供基础

## 评分

- **新颖性**: ★★★★☆ — 理论贡献扎实，混合掩码策略虽简单但有理论支撑
- **技术深度**: ★★★★★ — 涉及深度统计学习理论，技术含量高
- **实验充分性**: ★★★☆☆ — 合成数据实验充分，真实数据实验偏少
- **写作质量**: ★★★★☆ — 论文结构清晰，理论和实验平衡良好
- **实用性**: ★★★☆☆ — 偏理论，混合掩码策略有一定实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Epistemic Uncertainty Quantification To Improve Decisions From Black-Box Models](../../ICLR2026/learning_theory/epistemic_uncertainty_quantification_to_improve_decisions_from_black-box_models.md)
- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](a_highdimensional_statistical_method_for_optimizing_transfer.md)
- [\[ICLR 2026\] Transformers as Unsupervised Learning Algorithms: A study on Gaussian Mixtures](../../ICLR2026/learning_theory/transformers_as_unsupervised_learning_algorithms_a_study_on_gaussian_mixtures.md)
- [\[ICLR 2026\] A Unification of Discrete, Gaussian, and Simplicial Diffusion](../../ICLR2026/learning_theory/a_unification_of_discrete_gaussian_and_simplicial_diffusion.md)
- [\[ICLR 2026\] A Statistical Theory of Overfitting for Imbalanced Classification](../../ICLR2026/learning_theory/a_statistical_theory_of_overfitting_for_imbalanced_classification.md)

</div>

<!-- RELATED:END -->
