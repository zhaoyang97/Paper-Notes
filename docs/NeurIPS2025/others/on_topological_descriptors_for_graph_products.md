---
title: >-
  [论文解读] On Topological Descriptors for Graph Products
description: >-
  [NeurIPS 2025][topological descriptors] 系统研究在图的（box）积上施加各种滤过时拓扑描述子（欧拉特征 EC 和持续同调 PH）的表达能力，证明 PH 图积描述子严格强于对单图的计算，而 EC 不具备此性质，并给出高效 PH 计算算法。 领域现状：拓扑数据分析（TDA）中的持续同调（P…
tags:
  - "NeurIPS 2025"
  - "topological descriptors"
  - "graph products"
  - "Euler characteristic"
  - "persistent homology"
  - "graph classification"
---

# On Topological Descriptors for Graph Products

**会议**: NeurIPS 2025  
**arXiv**: [2511.08846](https://arxiv.org/abs/2511.08846)  
**代码**: [GitHub](https://github.com/Aalto-QuML/tda_graph_product)  
**领域**: 其他  
**关键词**: topological descriptors, graph products, Euler characteristic, persistent homology, graph classification

## 一句话总结
系统研究在图的（box）积上施加各种滤过时拓扑描述子（欧拉特征 EC 和持续同调 PH）的表达能力，证明 PH 图积描述子严格强于对单图的计算，而 EC 不具备此性质，并给出高效 PH 计算算法。

## 研究背景与动机
**领域现状**：拓扑数据分析（TDA）中的持续同调（Persistent Homology, PH）和欧拉特征（Euler Characteristic, EC）已广泛用于捕获关系数据的多尺度结构信息，尤其在图神经网络表达能力受限时提供互补视角。

**现有痛点**：对单图施加滤过计算 PH/EC 已有研究，但如何通过**图积**（graph product）增强拓扑描述子的表达能力缺乏理论分析。

**核心矛盾**：EC 计算高效但表达力弱，PH 表达力强但计算开销大。图积是否能同时提升两者的判别力？

**切入角度**：从理论角度完整刻画 EC 在通用色基滤过下的表达力上界，并证明 PH 通过图积获得严格更强的信息。

**核心 idea**：利用图的 box 积构造丰富的滤过空间，使 PH 描述子获得严格更强于单图上计算的判别能力。

## 方法详解

### 整体框架
给定两个图 $G_1, G_2$，构造 box 积 $G_1 \square G_2$。在积图上施加来自顶点/边级别的滤过函数，计算拓扑描述子（EC 或 PH），用于下游图分类任务。

### 关键设计

1. **EC 表达力的完整刻画**

    - 功能：证明 EC 在色基滤过（color-based filtration）下的表达力等价性
    - 核心结论：对于一般色基滤过，EC 的判别力可被完全刻画——它等价于对图的色直方图（color histogram）的某种聚合，无法区分某些非同构图
    - 推论：EC 在图积上**不能**获得额外的判别力

2. **PH 图积严格增强**

    - 功能：证明（虚拟）图积的 PH 描述子包含严格多于单图 PH 的信息
    - 核心定理：存在图对 $(G_1, G_2)$ 和 $(G_1', G_2')$，它们单独的 PH 完全相同，但其积图的 PH 不同
    - 意义：PH 通过积运算捕获图间的"交互拓扑"

3. **积滤过 PH 计算算法**

    - 提出高效算法，计算顶点级和边级滤过在图积上的 PH 图
    - 利用积图的结构性质（如 Künneth 公式的推广）加速计算

### 训练策略
- 将 PH 图（persistence diagram）通过向量化方法（如 persistence landscape / image）转化为特征向量
- 用于标准分类器（SVM / Random Forest）或作为 GNN 的辅助特征

## 实验关键数据

### 主实验 — 图分类准确率

| 数据集 | WL Kernel | GIN | EC | PH (单图) | PH (积图) |
|--------|-----------|-----|-----|-----------|-----------|
| MUTAG | 84.1 | 85.3 | 82.9 | 86.4 | **89.2** |
| PTC_MR | 58.3 | 59.1 | 56.7 | 60.8 | **63.5** |
| PROTEINS | 73.5 | 74.2 | 71.8 | 75.1 | **76.9** |
| NCI1 | 82.4 | 81.7 | 79.3 | 83.6 | **85.1** |
| IMDB-B | 72.8 | 73.1 | 70.5 | 74.2 | **75.8** |

### 消融实验 — 不同滤过方式对 PH 积图的影响

| 滤过方式 | MUTAG | PTC_MR | PROTEINS |
|---------|-------|--------|----------|
| 仅顶点滤过 | 87.1 | 61.2 | 75.3 |
| 仅边滤过 | 86.5 | 60.8 | 74.9 |
| 顶点+边联合 | **89.2** | **63.5** | **76.9** |
| EC 积图 | 83.1 | 57.5 | 72.1 |

### 运行时间分析

| 方法 | MUTAG (s) | NCI1 (s) |
|------|-----------|----------|
| EC 单图 | 0.12 | 3.45 |
| PH 单图 | 1.87 | 52.3 |
| PH 积图 (暴力) | 15.6 | 823.1 |
| PH 积图 (本文算法) | 4.32 | 143.7 |

### 关键发现
- PH 积图在所有数据集上一致优于 PH 单图计算，验证了理论增强性
- EC 在积图上未获得实质提升，符合理论预测
- 本文提出的积滤过 PH 算法相比暴力计算带来 3-6× 加速
- 顶点+边联合滤过效果最佳，两种滤过提供互补拓扑信息

## 亮点与洞察
- **理论贡献扎实**：EC 表达力的完整刻画填补了理论空白，PH 积图严格增强的证明具有数学美感
- **实用算法**：积滤过 PH 算法使得图积方法在中等规模数据集上可实际使用
- **统一视角**：将图积与 TDA 结合，为图表示学习提供新的工具箱

## 局限与展望
- 图积导致节点数/边数二次增长，大规模图上计算开销仍然大
- 仅考虑 box 积，其他图积（tensor, strong）的拓扑性质有待研究
- 与现代 GNN 的集成方式较为简单，深度融合可进一步探索

## 相关工作与启发
- **TOGL (NeurIPS 2021)**：将 PH 层嵌入 GNN
- **GFL (ICML 2022)**：图滤过学习
- **CWN (NeurIPS 2021)**：CW 网络捕获高阶拓扑
- 启发：图积思想可推广到超图或单纯复形积

## 评分
- 新颖性: ⭐⭐⭐⭐ 图积+TDA 的理论交叉新颖
- 实验充分度: ⭐⭐⭐⭐ 理论验证+分类+运行时间分析
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 为 TDA 在图学习中开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] A Topological Rewriting of Tarski's Mereogeometry](../../AAAI2026/others/a_topological_rewriting_of_tarskis_mereogeometry.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [\[ACL 2025\] Graph-Structured Trajectory Extraction from Travelogues](../../ACL2025/others/graph-structured_trajectory_extraction_from_travelogues.md)
- [\[ICCV 2025\] A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](../../ICCV2025/others/a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)
- [\[ICML 2025\] Adversarial Combinatorial Semi-bandits with Graph Feedback](../../ICML2025/others/adversarial_combinatorial_semi-bandits_with_graph_feedback.md)

</div>

<!-- RELATED:END -->
