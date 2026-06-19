---
title: >-
  [论文解读] Private Zeroth-Order Optimization with Public Data
description: >-
  [NeurIPS 2025][优化/理论][差分隐私] 提出 PAZO 框架，利用公共数据引导私有零阶优化算法的梯度近似，在视觉和文本任务上实现了优于 DP-SGD 的隐私-效用权衡，同时获得最高 16 倍的速度提升。 现有痛点 现有痛点：领域现状：差分隐私（DP）机器学习算法（如 DP-SGD）是保护训练数据隐私的标准方法…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "差分隐私"
  - "零阶优化"
  - "公共数据"
  - "DP-SGD"
  - "隐私-效用权衡"
---

# Private Zeroth-Order Optimization with Public Data

**会议**: NeurIPS 2025

**arXiv**: [2511.10859](https://arxiv.org/abs/2511.10859)

**代码**: 无

**领域**: AI安全 / 差分隐私

**关键词**: 差分隐私, 零阶优化, 公共数据, DP-SGD, 隐私-效用权衡

## 一句话总结

提出 PAZO 框架，利用公共数据引导私有零阶优化算法的梯度近似，在视觉和文本任务上实现了优于 DP-SGD 的隐私-效用权衡，同时获得最高 16 倍的速度提升。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：差分隐私（DP）机器学习算法（如 DP-SGD）是保护训练数据隐私的标准方法，但存在严重的计算瓶颈：

**计算和内存开销高**: DP-SGD 需要对每个样本单独计算梯度（per-sample gradient），然后裁剪和加噪

**零阶方法的潜力**: 零阶（zeroth-order）方法通过函数值评估近似梯度，天然容易私有化，但效用较低

**公共数据未被充分利用**: 实际中常有公共数据（如预训练数据），但现有零阶方法未利用这一信息

核心问题：能否利用公共数据提升私有零阶优化的效用，同时保持计算效率优势？

## 方法详解

### 整体框架

PAZO (Public-data-Assisted Zeroth-Order) 框架在标准零阶优化流程中引入公共数据引导，提升梯度近似的质量。

### 关键设计

**1. 公共数据引导的方向选择**

- 标准零阶方法在随机方向上探测，方差大
- PAZO 利用公共数据上的梯度信息构建**重要性采样分布**
- 在公共梯度指示的重要方向上更密集地探测
- 核心: $\hat{g} = \sum_{i=1}^{q} \frac{f(x + \mu u_i) - f(x)}{\mu} u_i$，其中 $u_i$ 的采样受公共梯度引导

**2. 多种 PAZO 变体**

- **PAZO-Subspace**: 在公共梯度张成的子空间中进行零阶优化
- **PAZO-Projection**: 将零阶估计投影到公共数据信号方向
- **PAZO-Precondition**: 用公共数据的 Fisher 信息矩阵预条件零阶梯度

**3. 隐私分析**

- 所有操作中仅函数值查询涉及私有数据
- 公共数据操作不消耗隐私预算
- 与标准零阶 DP 方法使用相同的隐私会计工具

### 损失函数 / 训练策略

- 隐私预算: 使用 Rényi DP 或 零-集中 DP 进行紧致的隐私会计
- 噪声校准: 基于灵敏度和目标 $\varepsilon$ 确定高斯噪声标准差
- 学习率调度: 使用余弦退火，与非私有训练一致

## 实验关键数据

### 主实验

视觉任务 (CIFAR-10 微调, $\varepsilon=3$):

| 方法 | 准确率 | 训练时间 | 峰值内存 |
|------|--------|---------|---------|
| DP-SGD | 82.1% | 48 min | 12.3 GB |
| DP-SGD + Public | 84.5% | 52 min | 13.1 GB |
| Zero-order DP | 76.3% | 8 min | 3.2 GB |
| PAZO (Ours) | **85.2%** | **3 min** | **2.8 GB** |

文本任务 (SST-2 微调, $\varepsilon=8$):

| 方法 | 准确率 | 训练时间 | 速度提升 |
|------|--------|---------|---------|
| DP-SGD | 90.1% | 120 min | 1x |
| DP-SGD + Public | 91.3% | 135 min | 0.9x |
| Zero-order DP | 85.7% | 12 min | 10x |
| PAZO (Ours) | **91.8%** | **7.5 min** | **16x** |

### 消融实验

不同隐私预算下的性能 (CIFAR-10):

| $\varepsilon$ | DP-SGD | DP-SGD+Public | ZO-DP | PAZO |
|-------------|--------|--------------|-------|------|
| 1 | 71.2% | 75.8% | 62.3% | **77.5%** |
| 3 | 82.1% | 84.5% | 76.3% | **85.2%** |
| 8 | 88.5% | 89.2% | 83.1% | **89.8%** |
| ∞ (无隐私) | 94.2% | 94.2% | 91.5% | 93.8% |

### 关键发现

1. PAZO 在高隐私保护（低 $\varepsilon$）场景下优势最显著，超越包括使用公共数据的一阶方法
2. 运行时加速达 16 倍，主要来自避免了 per-sample 梯度计算
3. 公共数据的相似度越高,改善越大;但即使相似度有限,仍有显著提升
4. PAZO-Subspace 在低维场景下最优, PAZO-Precondition 在高维场景下最优

## 亮点与洞察

- **打破一阶上限**: 首次在私有学习中,零阶方法超越了使用公共数据的一阶方法
- **实用性极强**: 16x 加速 + 低内存 = 边缘设备上的隐私保护训练成为可能
- **理论支撑**: 在公私数据相似性假设下提供了收敛性分析

## 局限与展望

1. 需要公共数据与私有数据之间有一定相似性,跨域差异大时效果下降
2. 理论分析要求凸性或 PL 条件,非凸深度模型的理论保证有限
3. 当前针对微调场景,从头预训练的效果未验证
4. 公共数据的质量和数量如何影响最终结果缺乏系统研究

## 相关工作与启发

- **DP-SGD** (Abadi et al.): 差分隐私随机梯度下降的标准方法
- **MeZO** (Malladi et al.): 大模型的零阶优化框架
- **公共数据辅助DP**: Yu et al., De et al. 的一阶方法

## 评分

- ⭐ 创新性: 8/10 — 公共数据引导零阶方法的思路简洁有效
- ⭐ 实用性: 9/10 — 速度和隐私效用双重优势,实际部署价值高
- ⭐ 写作质量: 8/10 — 多变体对比全面,实验设计合理

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](../../ICML2026/optimization/learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](../../ICML2025/optimization/improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](../../ICML2026/optimization/learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)

</div>

<!-- RELATED:END -->
