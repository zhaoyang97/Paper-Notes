---
title: >-
  [论文解读] Learning-Augmented Hierarchical Clustering
description: >-
  [ICML 2025][算法 / 聚类][hierarchical clustering] 本文研究借助分裂预言机（splitting oracle）的辅助信息来突破层次聚类的近似硬度障碍，获得 Dasgupta 目标的 $O(1)$ 常数近似和 Moseley-Wang 目标的 $(1-o(1))$ 近似，并推广到流式和并行计算场景。
tags:
  - "ICML 2025"
  - "算法 / 聚类"
  - "hierarchical clustering"
  - "learning-augmented algorithms"
  - "splitting oracle"
  - "Dasgupta objective"
  - "approximation algorithms"
---

# Learning-Augmented Hierarchical Clustering

**会议**: ICML 2025  
**arXiv**: [2506.05495](https://arxiv.org/abs/2506.05495)  
**代码**: 无  
**领域**: 算法 / 聚类  
**关键词**: hierarchical clustering, learning-augmented algorithms, splitting oracle, Dasgupta objective, approximation algorithms

## 一句话总结
本文研究借助分裂预言机（splitting oracle）的辅助信息来突破层次聚类的近似硬度障碍，获得 Dasgupta 目标的 $O(1)$ 常数近似和 Moseley-Wang 目标的 $(1-o(1))$ 近似，并推广到流式和并行计算场景。

## 研究背景与动机
**领域现状**：层次聚类（HC）是数据分析的基础技术，将数据集递归划分为树状结构。Dasgupta (STOC'16) 提出了经典的层次聚类目标函数，Moseley-Wang (NeurIPS'17) 提出了对偶目标。

**现有痛点**：对于这两个目标函数，存在很强的近似硬度障碍。在 Small Set Expansion Hypothesis (SSEH) 下，Dasgupta 目标不存在多项式时间的常数近似算法，Moseley-Wang 目标不存在 $(1-C)$ 近似。

**核心矛盾**：理论上不可能的高质量近似 vs 实际应用对高质量聚类的迫切需求。

**本文目标**：利用自然的辅助信息（oracle）来突破近似硬度。

**切入角度**：Learning-Augmented Algorithms 范式——允许算法访问（可能有噪声的）预言机。

**核心idea**：分裂预言机（splitting oracle）提供三元组关系信息，可以指导层次聚类突破传统硬度限制。

## 方法详解

### 整体框架
输入：加权图 $G = (V, E, w)$，分裂预言机（splitting oracle）
输出：层次聚类树 $T$

分裂预言机定义：给定三元组 $(u, v, w)$，返回哪个顶点在最优树中与另外两个"分裂开来"（即其最近公共祖先包含所有三个顶点）。

### 关键设计

1. **Dasgupta 目标的 $O(1)$ 近似算法**:

    - 功能：利用分裂预言机构建低代价层次聚类树
    - 核心思路：递归地使用预言机来确定最优分裂点。对于节点集合 $S$，选择一个枢轴（pivot），通过查询预言机将其他节点分为与枢轴同侧或异侧，从而递归划分
    - 设计动机：预言机提供的三元组关系可以恢复最优树的拓扑结构，即使有错误（错误率有界时），仍能保证常数近似

2. **Moseley-Wang 目标的 $(1-o(1))$ 近似**:

    - 功能：在近线性时间内构建几乎最优的层次聚类树
    - 核心思路：利用 Moseley-Wang 目标与 Dasgupta 目标的对偶关系。先用预言机确定粗粒度结构，再在局部精细调整
    - 设计动机：Moseley-Wang 目标关注的是"好的分裂"（相似点在同一子树），预言机直接提供分裂信息

3. **推广到亚线性设置**:

    - 功能：设计流式（streaming）和并行（PRAM）算法
    - 核心思路：
        - 流式版本：使用有限空间维护关键的三元组关系，在一遍扫描后构建聚类树
        - PRAM版本：将递归分裂过程并行化，利用预言机的独立查询特性
    - 设计动机：大规模数据集无法全部载入内存，需要亚线性空间/时间的解决方案

### 损失函数 / 训练策略
本文为纯理论算法工作。
- Dasgupta 目标：$\min_T \sum_{(i,j) \in E} w_{ij} |T[i \vee j]|$，其中 $T[i \vee j]$ 是 $i, j$ 的最近公共祖先下的叶子数
- Moseley-Wang 目标：$\max_T \sum_{(i,j) \in E} w_{ij} (n - |T[i \vee j]|)$

## 实验关键数据

### 主实验（理论保证）

| 目标函数 | 指标 | 本文方法 | 无 Oracle 的 SOTA | 硬度下界 |
|---|---|---|---|---|
| Dasgupta | 近似比 | $O(1)$ | $O(\sqrt{\log n})$ | 无常数近似(SSEH) |
| Moseley-Wang | 近似比 | $1-o(1)$ | $\frac{2}{3}+\delta$ | $1-C$(SSEH) |
| Dasgupta (流式) | 空间/近似 | $\tilde{O}(n)$/O(1) | 无已知 | — |
| Dasgupta (PRAM) | 深度/近似 | $\text{polylog}(n)$/O(1) | 无已知 | — |

### 消融实验（Oracle 错误率影响）

| Oracle 错误率 | Dasgupta 近似比 | Moseley-Wang 近似比 | 说明 |
|---|---|---|---|
| 0%（完美Oracle） | $O(1)$ 最优 | $1-o(1)$ | 理想情况 |
| $< 1/3$ | $O(1)$ | $(1-o(1))$ | 容错范围内 |
| $\to 1/2$（随机猜测） | 退化 | 退化 | 接近无用信息 |

### 关键发现
- 分裂预言机是一种自然且强大的辅助信息形式，能将"不可能"的近似变为可能
- 算法对预言机的错误具有一定的鲁棒性（错误率 $< 1/3$ 时仍保持性能保证）
- 流式和并行设置下的推广表明方法具有实践可部署性

## 亮点与洞察
- 理论贡献突出：首次证明用自然预言机可以突破层次聚类的近似硬度
- 分裂预言机的定义非常自然，可以由领域专家或预训练模型提供
- Learning-Augmented Algorithms 范式的典范：即使辅助信息不完美也能获得强保证

## 局限与展望
- 分裂预言机的实际构造方式需要进一步探索（如用机器学习模型来学习）
- 目前的近似比常数可能还有优化空间
- 加权图上的结果是否能进一步改进有待研究

## 相关工作与启发
- Dasgupta [STOC'16] 和 Moseley-Wang [NeurIPS'17] 的经典目标函数
- Learning-augmented algorithms 是近年来算法设计的热门方向
- 可以探索其他类型的预言机（如成对比较预言机）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用预言机突破近似硬度是重要理论突破
- 实验充分度: ⭐⭐⭐ 主要是理论工作，实验相对有限
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，证明完整
- 价值: ⭐⭐⭐⭐ 对算法设计范式有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2025\] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)
- [\[ICML 2025\] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems](near-optimal_consistency-robustness_trade-offs_for_learning-augmented_online_kna.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/learning_theory/learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICML 2025\] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)

</div>

<!-- RELATED:END -->
