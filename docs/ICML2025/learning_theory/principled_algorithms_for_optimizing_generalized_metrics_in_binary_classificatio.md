---
title: >-
  [论文解读] Principled Algorithms for Optimizing Generalized Metrics in Binary Classification
description: >-
  [ICML 2025][学习理论 / 分类优化][generalized metrics] 本文提出了优化广义分类指标（如 $F_\beta$、Jaccard、加权准确率等）的有原则算法 METRO，基于 $H$-一致性界和代理损失理论，将指标优化重新表述为广义代价敏感学习问题，具有有限样本泛化保证。
tags:
  - "ICML 2025"
  - "学习理论 / 分类优化"
  - "generalized metrics"
  - "F-measure"
  - "H-consistency"
  - "surrogate loss"
  - "cost-sensitive learning"
---

# Principled Algorithms for Optimizing Generalized Metrics in Binary Classification

**会议**: ICML 2025  
**arXiv**: [2512.23133](https://arxiv.org/abs/2512.23133)  
**代码**: 无  
**领域**: 学习理论 / 分类优化  
**关键词**: generalized metrics, F-measure, H-consistency, surrogate loss, cost-sensitive learning

## 一句话总结
本文提出了优化广义分类指标（如 $F_\beta$、Jaccard、加权准确率等）的有原则算法 METRO，基于 $H$-一致性界和代理损失理论，将指标优化重新表述为广义代价敏感学习问题，具有有限样本泛化保证。

## 研究背景与动机
**领域现状**：在类别不平衡或非对称代价场景中，$F_\beta$-score、AM measure、Jaccard 系数等广义指标比标准 0-1 损失更合适。然而，这些指标通常是不可分解的（non-decomposable），无法写成单样本损失之和。

**现有痛点**：现有方法通常依赖 Bayes 最优分类器的刻画，先估计类概率再寻找最优阈值。这导致：(1) 算法未针对受限假设集裁剪；(2) 缺乏有限样本性能保证。

**核心矛盾**：理论上需要直接优化目标指标，但这些指标非凸非可分解，难以直接优化。

**本文目标**：为广义指标的优化提供有原则的算法，同时具有 $H$-一致性和有限样本泛化界。

**切入角度**：将指标优化重新表述为广义代价敏感学习，设计具有可证明 $H$-一致性的代理损失。

**核心idea**：对每种广义指标，存在一个等价的代价敏感学习问题，可以用设计好的代理损失来高效优化。

## 方法详解

### 整体框架
输入：训练数据 $\{(x_i, y_i)\}_{i=1}^n$，目标广义指标 $\Psi$（如 $F_\beta$）
输出：优化该指标的分类器 $h \in H$

Pipeline：
1. 将广义指标 $\Psi$ 分解为关于混淆矩阵元素（TP, FP, FN, TN）的函数
2. 构造等价的代价敏感学习问题，其中代价参数依赖于未知的先验分布
3. 设计具有 $H$-一致性的代理损失函数
4. 使用 METRO 算法交替优化代价参数和分类器

### 关键设计

1. **广义指标到代价敏感学习的归约**:

    - 功能：证明优化 $\Psi$ 等价于求解一个特定的代价敏感分类问题
    - 核心思路：对于 $F_\beta = \frac{(1+\beta^2) \text{TP}}{(1+\beta^2)\text{TP} + \beta^2 \text{FN} + \text{FP}}$ 等指标，Bayes 最优解可以表示为阈值分类器 $h(x) = \mathbb{1}[\eta(x) > c^*]$，其中阈值 $c^*$ 是代价参数的函数。将此推广为代价敏感学习：$\min_h \mathbb{E}[c \cdot \mathbb{1}[h(x) \neq y]]$，代价 $c$ 依赖于指标和类先验
    - 设计动机：代价敏感学习有成熟的理论基础，归约后可以利用已有的代理损失理论

2. **$H$-一致性代理损失**:

    - 功能：设计新的代理损失函数，满足对受限假设集 $H$ 的一致性
    - 核心思路：$H$-一致性要求：以代理损失最小化得到的分类器，其目标指标也收敛到最优。本文证明了一类以代价为参数的凸代理损失具有此性质：
    $\ell_c(h(x), y) = c \cdot \phi(y \cdot h(x))$
      其中 $\phi$ 是适当设计的凸递减函数
    - 设计动机：标准交叉熵损失不一定对广义指标具有 $H$-一致性；需要专门设计的损失

3. **METRO 算法**:

    - 功能：交替优化代价参数 $c$ 和分类器 $h$
    - 核心思路：
        - Step 1：固定分类器 $h$，更新代价参数 $c$ 以匹配当前分类器下的最优代价
        - Step 2：固定代价 $c$，用代理损失训练分类器 $h$
        - 重复直到收敛
    - 设计动机：代价参数依赖于分布信息（如 $P(Y=1)$），需要从数据中估计。交替优化自然地处理了这种互相依赖关系

### 损失函数 / 训练策略
- 代理损失：$\hat{R}_\ell(h) = \frac{1}{n} \sum_{i=1}^n c_i \cdot \phi(y_i h(x_i))$
- 有限样本泛化界：$\Psi(h) \leq \Psi(h^*) + O(\sqrt{\text{complexity}(H)/n})$

## 实验关键数据

### 主实验

| 数据集 | 指标 | METRO | 阈值搜索法 | 直接优化基线 | 标准CE |
|---|---|---|---|---|---|
| 信用违约 | $F_1$ | **0.523** | 0.498 | 0.486 | 0.451 |
| 医疗诊断 | $F_2$ | **0.714** | 0.688 | 0.670 | 0.632 |
| 欺诈检测 | Jaccard | **0.381** | 0.355 | 0.342 | 0.310 |
| 合成不平衡 | AM | **0.862** | 0.841 | 0.830 | 0.795 |

### 消融实验

| 配置 | $F_1$ | 说明 |
|---|---|---|
| METRO（完整） | **0.523** | 代价敏感 + 一致性损失 |
| 无代价更新（固定 $c$） | 0.495 | 代价需自适应 |
| 标准交叉熵 + 阈值搜索 | 0.498 | 传统方法 |
| METRO + 非一致性代理损失 | 0.507 | 一致性保证有帮助 |

### 关键发现
- METRO 在所有指标和数据集上均优于基线，尤其在高度不平衡场景中优势更大
- $H$-一致性代理损失相比普通代理损失有明显提升，验证了理论分析
- 代价参数的自适应更新很重要（消融显示固定代价显著劣于自适应）
- 有限样本泛化界紧密，实际性能与理论界吻合

## 亮点与洞察
- 理论完备性：$H$-一致性 + 有限样本界，现有方法首次同时具备
- 实用性：METRO 算法简洁高效，只需在标准训练循环中加入代价更新步骤
- 适用范围广：统一处理 $F_\beta$、Jaccard、AM measure、加权准确率等多种指标

## 局限与展望
- 目前限于二分类，多分类推广是重要的未来方向
- 代价参数估计在极端不平衡下可能不稳定
- 与深度学习的结合（如与预训练模型联合微调）需要进一步探索

## 相关工作与启发
- 与 Koyejo et al. (2014)、Narasimhan et al. (2014) 的阈值方法互补
- $H$-一致性概念来自 Awasthi et al. (2022)，本文是首次将其应用于广义指标优化
- 对不平衡分类实践有直接指导价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 归约框架和一致性分析有新意
- 实验充分度: ⭐⭐⭐⭐ 多指标多数据集验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论表述清晰，算法描述简洁
- 价值: ⭐⭐⭐⭐ 解决了实际中的重要问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data](../../ICLR2026/learning_theory/a_generalized_geometric_theoretical_framework_of_centroid_discriminant_analysis_.md)
- [\[ICML 2025\] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)
- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](../../NeurIPS2025/learning_theory/a_highdimensional_statistical_method_for_optimizing_transfer.md)
- [\[ICLR 2026\] A Statistical Theory of Overfitting for Imbalanced Classification](../../ICLR2026/learning_theory/a_statistical_theory_of_overfitting_for_imbalanced_classification.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)

</div>

<!-- RELATED:END -->
