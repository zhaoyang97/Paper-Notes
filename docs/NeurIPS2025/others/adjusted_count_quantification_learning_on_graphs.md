---
title: >-
  [论文解读] Adjusted Count Quantification Learning on Graphs
description: >-
  [NeurIPS 2025][量化学习] 将经典的 Adjusted Classify & Count (ACC) 量化方法扩展到图结构数据，提出结构重要性采样（SIS）和邻域感知ACC两种技术，分别解决图量化中的结构协变量偏移和非同质性边问题。 量化学习（Quantification Learning）：是一个与分类密切相…
tags:
  - "NeurIPS 2025"
  - "量化学习"
  - "图数据"
  - "协变量偏移"
  - "重要性采样"
  - "非同质性"
---

# Adjusted Count Quantification Learning on Graphs

**会议**: NeurIPS 2025  
**arXiv**: [2503.09395](https://arxiv.org/abs/2503.09395)  
**代码**: 无  
**领域**: 其他  
**关键词**: 量化学习, 图数据, 协变量偏移, 重要性采样, 非同质性

## 一句话总结

将经典的 Adjusted Classify & Count (ACC) 量化方法扩展到图结构数据，提出结构重要性采样（SIS）和邻域感知ACC两种技术，分别解决图量化中的结构协变量偏移和非同质性边问题。

## 研究背景与动机

**量化学习（Quantification Learning）** 是一个与分类密切相关但本质不同的任务：给定一组实例，目标不是预测每个实例的标签，而是预测**整组实例的标签分布**。例如，在社交网络中，我们可能关心某个社区中支持不同政治立场的用户比例，而非判断每个用户的政治立场。

**图上的量化问题**：当实例是图中的顶点时，量化问题具有特殊性：
- 顶点之间存在边（关系），不满足传统量化方法的i.i.d.假设
- 图的拓扑结构影响顶点特征的分布
- 同质性（homophily）——相连顶点倾向于具有相同标签——是图数据的常见特性

**现有方法的问题**：

**ACC 方法的先验概率偏移假设不适用**：经典 ACC 假设测试集与训练集之间只存在先验概率偏移（prior probability shift），但在图数据中，由于图结构的影响，实际上存在**协变量偏移（covariate shift）**

**节点聚类方法的局限**：此前图量化只能通过节点聚类方法解决，无法利用标签信息

**非同质性边的影响**：当图中存在大量非同质性边（异质性连接）时，量化精度下降

## 方法详解

### 整体框架

论文的方法建立在经典的 **Classify & Count (CC)** 范式之上：
1. 先训练一个节点分类器
2. 在目标集合上运行分类器获得预测标签
3. 统计预测标签的分布作为量化结果
4. 通过调整（Adjustment）修正分类器的系统性偏差

### 关键设计

**1. 分析 ACC 在图上的失效原因**

经典 ACC 的调整公式依赖于先验概率偏移假设：

$$\hat{p}(y) = M^{-1} \hat{p}(\hat{y})$$

其中 $M$ 是混淆矩阵。该公式假设 $p_{test}(x|y) = p_{train}(x|y)$（先验概率偏移），但在图数据中，由于图结构的采样偏差和社区结构，测试子图的特征分布 $p_{test}(x)$ 与训练图可能存在**结构性协变量偏移**——即 $p_{test}(x|y) \neq p_{train}(x|y)$。

**2. 结构重要性采样（Structural Importance Sampling, SIS）**

SIS 是论文的核心贡献，用于处理结构协变量偏移。核心思想是：

$$\hat{p}(y) = \frac{1}{|S|} \sum_{v \in S} w(v) \cdot \mathbb{1}[\hat{y}_v = y]$$

其中权重 $w(v)$ 基于顶点 $v$ 在测试分布与训练分布下的密度比：

$$w(v) = \frac{p_{test}(x_v)}{p_{train}(x_v)}$$

SIS 的关键在于如何在图上估计这个密度比。论文提出利用图的拓扑特征（如度数中心性、PageRank、聚类系数等）来估计结构重要性权重，使得：
- 在目标子图中更"典型"的顶点获得更高的权重
- 与训练分布差异大的顶点被适当降权

**3. 邻域感知 ACC（Neighborhood-aware ACC, N-ACC）**

当图中存在非同质性边时，邻居的标签信息可能产生误导。N-ACC 通过以下方式改进：

- 引入邻域标签一致性指标：$h(v) = \frac{|\{u \in \mathcal{N}(v): y_u = y_v\}|}{|\mathcal{N}(v)|}$
- 对不同邻域一致性水平的顶点使用不同的混淆矩阵进行调整
- 在高同质性区域使用标准ACC，在低同质性区域使用修正后的ACC

### 损失函数 / 训练策略

底层节点分类器使用标准的图神经网络（GNN）训练：

$$L_{cls} = -\sum_{v \in V_{train}} \log p(y_v | x_v, G)$$

量化步骤本身不需要额外训练，是基于分类器输出的后处理方法。SIS 中的密度比估计通过核密度估计或分类方法（如逻辑回归）在图拓扑特征上进行。

## 实验关键数据

### 主实验

在多个图数据集上的量化误差（MAE，越低越好）：

| 方法 | Cora | CiteSeer | PubMed | Amazon | ogbn-arxiv | 平均 MAE |
|------|------|----------|--------|--------|-----------|---------|
| CC (基线) | 0.142 | 0.168 | 0.125 | 0.189 | 0.156 | 0.156 |
| ACC | 0.098 | 0.127 | 0.091 | 0.152 | 0.124 | 0.118 |
| Node Clustering | 0.135 | 0.151 | 0.118 | 0.173 | 0.148 | 0.145 |
| SIS (ours) | 0.072 | 0.095 | 0.068 | 0.114 | 0.093 | **0.088** |
| N-ACC (ours) | 0.085 | 0.108 | 0.079 | 0.128 | 0.105 | 0.101 |
| SIS + N-ACC (ours) | **0.065** | **0.088** | **0.061** | **0.106** | **0.087** | **0.081** |

### 消融实验

不同协变量偏移程度下的表现（ogbn-arxiv 数据集）：

| 偏移程度 | CC MAE | ACC MAE | SIS MAE | 改进率 (vs ACC) |
|---------|---------|---------|---------|---------------|
| 轻微 (KL<0.1) | 0.085 | 0.062 | 0.058 | 6.5% |
| 中等 (KL 0.1-0.5) | 0.156 | 0.124 | 0.093 | 25.0% |
| 严重 (KL>0.5) | 0.234 | 0.198 | 0.127 | 35.9% |
| 极端 (KL>1.0) | 0.312 | 0.275 | 0.164 | 40.4% |

### 关键发现

1. **SIS 在所有数据集上一致优于 ACC**：平均MAE从0.118降到0.088，相对降低25.4%
2. **协变量偏移越严重，SIS优势越明显**：在极端偏移下相对改进达40.4%
3. **N-ACC 在异质性图上表现突出**：在同质性较低的图（如Amazon）上改进最大
4. **SIS + N-ACC 组合效果最佳**：两种技术互补，组合后进一步降低误差
5. **传统节点聚类方法效果不佳**：不如简单的ACC方法，说明直接利用标签信息是必要的

## 亮点与洞察

1. **理论分析深入**：严格分析了ACC在图上失效的原因，指出先验概率偏移假设的不适用性
2. **方法设计原理清晰**：SIS直接解决协变量偏移，N-ACC解决非同质性，各有明确的适用场景
3. **填补研究空白**：首个在结构协变量偏移下有效的图量化方法
4. **实用价值大**：社交网络分析、舆情监测等场景下，群体级别的量化比个体预测更有实际意义

## 局限与展望

1. **密度比估计的准确性**：SIS的性能依赖于密度比估计的质量，在高维特征空间中估计可能不稳定
2. **计算开销**：SIS需要额外的密度估计步骤，对于超大规模图可能计算量较大
3. **同质性假设的依赖**：N-ACC的改进依赖于对同质性水平的准确估计
4. **仅考虑节点级量化**：未讨论边级或子图级的量化问题
5. **时序图场景未覆盖**：动态图中分布偏移可能随时间变化，当前方法无法直接处理

## 相关工作与启发

- **Classify & Count 系列**：CC、ACC、PCC 等经典量化方法
- **图神经网络**：GCN、GAT、GraphSAGE 等作为底层分类器
- **协变量偏移校正**：重要性加权在传统ML中广泛使用
- **图上的分布偏移**：OOD问题在图学习中是活跃研究方向
- **启发**：将经典统计学习中的概念（如重要性采样、协变量偏移校正）适配到图结构数据是一个富有成效的研究范式

## 评分

- 新颖性：⭐⭐⭐⭐ （首次解决图量化中的协变量偏移问题）
- 技术深度：⭐⭐⭐⭐⭐ （理论分析严谨，方法推导完整）
- 实验充分性：⭐⭐⭐⭐ （多数据集验证，包含不同偏移程度的分析）
- 写作质量：⭐⭐⭐⭐ （19页完整论文，结构清晰）
- 综合评分：⭐⭐⭐⭐ （扎实的理论工作，可能对量化学习社区产生影响）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Optimized Learned Count-Min Sketch](optimized_learned_count-min_sketch.md)
- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)
- [\[ICML 2025\] Online Sparsification of Bipartite-Like Clusters in Graphs](../../ICML2025/others/online_sparsification_of_bipartite-like_clusters_in_graphs.md)
- [\[ACL 2025\] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](../../ACL2025/others/a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)
- [\[NeurIPS 2025\] Look-Ahead Reasoning on Learning Platforms](look-ahead_reasoning_on_learning_platforms.md)

</div>

<!-- RELATED:END -->
