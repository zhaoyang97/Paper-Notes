---
title: >-
  [论文解读] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization
description: >-
  [NeurIPS 2025][图学习][图神经网络] 提出了分层自我图神经网络（Hierarchical Ego GNNs，HEGNNs），通过层次化的节点个体化机制泛化了子图GNN，形成表达力递增的模型层级；在有界度图上，证明HEGNN节点分类器的区分能力等价于分级杂合逻辑（graded hybrid logic），从而统一了多种GNN变体的表达力分析。
tags:
  - "NeurIPS 2025"
  - "图学习"
  - "图神经网络"
  - "表达力"
  - "分层节点个体化"
  - "同构测试"
  - "杂合逻辑"
---

# Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization

**会议**: NeurIPS 2025

**arXiv**: [2506.13911](https://arxiv.org/abs/2506.13911)

**代码**: 无

**领域**: 图学习 / 图神经网络理论

**关键词**: 图神经网络, 表达力, 分层节点个体化, 同构测试, 杂合逻辑

## 一句话总结

提出了分层自我图神经网络（Hierarchical Ego GNNs，HEGNNs），通过层次化的节点个体化机制泛化了子图GNN，形成表达力递增的模型层级；在有界度图上，证明HEGNN节点分类器的区分能力等价于分级杂合逻辑（graded hybrid logic），从而统一了多种GNN变体的表达力分析。

## 研究背景与动机

标准消息传递GNN（MPNN）的表达力受限于1阶Weisfeiler-Leman（1-WL）颜色细化算法，无法区分许多非同构图。为提升GNN表达力，已有多种方法：

- **高阶GNN**（$k$-WL GNN）：计算复杂度高
- **子图GNN**（Subgraph-GNN）：为每个节点构建以其为中心的子图，增加区分能力
- **局部同态计数特征**：在节点特征中加入子图出现次数
- **个体化-细化**（Individualization-Refinement, IR）：图同构测试中的经典范式

**核心问题**：这些方法之间的关系是什么？能否建立一个统一的层次化框架来精确刻画它们的表达力？

传统分析缺乏将GNN表达力与逻辑形式化语言精确关联的工具，导致不同GNN变体之间的比较缺乏统一基准。

## 方法详解

### 整体框架

HEGNN受Individualization-Refinement（IR）范式启发，通过**层次化节点个体化**扩展标准GNN：

- **第0层**：标准GNN（等价于1-WL）
- **第1层**：对每个节点 $v$，在以 $v$ 为"个体化"节点的子图上运行GNN（等价于子图GNN）
- **第$k$层**：递归地对 $k$ 层嵌套的节点进行个体化
- **极限**：$k \to \infty$ 时可以区分所有非同构图

形式化定义：

$$\text{HEGNN}_k(v) = \text{GNN}\left(G, v, \{\text{HEGNN}_{k-1}(u) : u \in V\}\right)$$

### 关键设计

#### 层次化节点个体化

1. **个体化操作**：选择一个节点赋予唯一标记，打破节点间的对称性
2. **层次化**：递归应用个体化，每层增加一个"锚点"节点
3. **聚合**：跨层次聚合信息，综合不同层级的区分能力

**与子图GNN的关系**：
- 子图GNN ≈ HEGNN-1（仅一层个体化）
- HEGNN-$k$ 严格强于 HEGNN-$(k-1)$

#### 逻辑表征定理

**核心定理**（在有界度图上）：

$$\text{HEGNN}_k \text{的区分能力} = \text{分级杂合逻辑} \mathcal{HL}_k$$

其中分级杂合逻辑 $\mathcal{HL}_k$ 是模态逻辑的扩展：
- **分级模态算子**：$\langle \geq n \rangle \phi$ 表示"至少有$n$个邻居满足$\phi$"
- **命名机制**（nominals）：可以引用特定节点
- **层级$k$**：允许$k$层嵌套的命名

#### 与其他GNN变体的关系

通过逻辑表征，建立了统一比较：

| GNN变体 | 等价逻辑 | HEGNN层级 |
|---------|---------|-----------|
| 标准MPNN | 分级模态逻辑 $\mathcal{ML}$ | HEGNN-0 |
| 子图GNN | $\mathcal{HL}_1$ | HEGNN-1 |
| $k$-WL GNN | $\mathcal{C}^k$ ($k$-变量计数逻辑) | 不直接对应 |
| HEGNN-$k$ | $\mathcal{HL}_k$ | HEGNN-$k$ |
| HEGNN-$\infty$ | 完全区分（同构测试） | 极限 |

### 损失函数 / 训练策略

HEGNN的训练采用标准图学习流程：
- **节点分类**：交叉熵损失
- **图分类**：在readout后使用交叉熵
- **优化**：标准SGD/Adam

关键实现考量：
- 第$k$层HEGNN需要 $O(n^k)$ 的计算（每层个体化引入一个额外的循环）
- 实际应用中通常使用 $k=1$ 或 $k=2$

## 实验关键数据

### 主实验

#### 图同构判别任务

在多个标准图同构测试数据集上进行评估：

| 模型 | EXP | CSL | SR25 | 4×4 Rook | 判别率(%) |
|------|:---:|:---:|:----:|:--------:|:---------:|
| GCN | 50.0 | 10.0 | 0.0 | 0.0 | 15.0 |
| GIN | 50.0 | 10.0 | 0.0 | 0.0 | 15.0 |
| PPGN (3-WL) | 100.0 | 100.0 | 66.7 | 100.0 | 91.7 |
| DS (子图GNN) | 100.0 | 100.0 | 0.0 | 16.7 | 54.2 |
| **HEGNN-1** | 100.0 | 100.0 | 33.3 | 66.7 | 75.0 |
| **HEGNN-2** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |

- HEGNN-2在所有数据集上完全区分，与3-WL GNN相当甚至更优
- HEGNN-1已经显著优于标准GNN和子图GNN

#### 图分类任务

在TU数据集和OGB数据集上的图分类准确率：

| 模型 | MUTAG | PTC | PROTEINS | IMDB-B | 平均 |
|------|:-----:|:---:|:--------:|:------:|:----:|
| GCN | 85.6 | 64.2 | 76.0 | 74.1 | 75.0 |
| GIN | 89.4 | 64.6 | 76.2 | 75.1 | 76.3 |
| 子图GNN | 89.8 | 66.1 | 76.8 | 75.8 | 77.1 |
| **HEGNN-1** | 90.2 | 67.3 | 77.1 | 76.2 | 77.7 |
| **HEGNN-2** | **91.1** | **68.5** | **77.5** | **76.8** | **78.5** |

### 消融实验

#### 局部同态计数特征的影响

| 模型 | 无计数特征 | +三角形计数 | +4-环计数 | 全部计数 |
|------|:---------:|:----------:|:---------:|:-------:|
| GCN | 75.0 | 76.2 | 76.5 | 77.0 |
| GIN | 76.3 | 77.5 | 77.8 | 78.1 |
| **HEGNN-1** | 77.7 | 78.2 | 78.3 | 78.5 |
| **HEGNN-2** | **78.5** | **78.8** | **78.9** | **79.0** |

**关键发现**：
- HEGNN-2即使不使用同态计数特征，也接近带有全部计数特征的GIN
- 同态计数特征对HEGNN的提升幅度小于对基础GNN的提升，说明HEGNN已经隐式捕获了部分子结构信息

#### 计算复杂度分析

| 模型 | 时间复杂度 | 空间复杂度 | 实际推理时间（EXP） |
|------|:---------:|:---------:|:------------------:|
| GCN | $O(n \cdot d \cdot L)$ | $O(n \cdot d)$ | 1× |
| GIN | $O(n \cdot d \cdot L)$ | $O(n \cdot d)$ | 1.2× |
| HEGNN-1 | $O(n^2 \cdot d \cdot L)$ | $O(n^2 \cdot d)$ | 8× |
| HEGNN-2 | $O(n^3 \cdot d \cdot L)$ | $O(n^3 \cdot d)$ | 50× |

### 关键发现

1. HEGNN形成了严格的表达力层级：HEGNN-0 $\subset$ HEGNN-1 $\subset$ HEGNN-2 $\subset \ldots$
2. 逻辑表征建立了GNN表达力与数学逻辑之间的精确桥梁
3. HEGNNs在实践中可行，但高阶版本的计算成本限制了可扩展性
4. 在大部分任务中，HEGNN-1或HEGNN-2就足够了

## 亮点与洞察

1. **精确的逻辑刻画**：将GNN表达力与分级杂合逻辑精确对应，是图学习理论的重要进展
2. **统一框架**：提供了比较不同GNN变体表达力的统一语言
3. **IR范式的优雅迁移**：将图同构测试中的经典IR范式自然地转化为GNN架构
4. **理论指导实践**：逻辑刻画可以帮助实践者选择合适复杂度的GNN变体

## 局限与展望

1. **有界度假设**：逻辑表征定理要求图的度有界，对无界度图（如幂律图）不直接适用
2. **计算可扩展性**：HEGNN-$k$ 的 $O(n^{k+1})$ 复杂度限制了高阶版本的实际使用
3. **缺乏大规模实验**：在OGB等大规模基准上的实验有限
4. **潜在方向**：
    - 设计近似HEGNN-$k$ 的高效算法（如采样策略）
    - 放松有界度假设
    - 探索与Transformer架构的结合

## 相关工作与启发

- **WL层级**：$k$-WL测试是GNN表达力分析的基准，本文提供了正交的层级结构
- **子图GNN**：Bevilacqua等 (2022), Frasca等 (2022)的工作，HEGNN是自然推广
- **GNN与逻辑**：Barceló等 (2020)建立了GNN与FOC$_2$的联系，本文扩展到杂合逻辑
- **局部同态计数**：Bouritsas等 (2023)的工作，本文通过逻辑框架统一了与其的关系

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 精确逻辑刻画GNN层级的表达力是开创性工作
- **技术深度**: ⭐⭐⭐⭐⭐ — 结合了图论、模态逻辑和深度学习的深度理论分析
- **实验充分度**: ⭐⭐⭐ — 实验展示了可行性但规模较小
- **实用性**: ⭐⭐⭐ — 理论贡献巨大，实际应用受计算限制
- **总体**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](../../AAAI2026/graph_learning/enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)

</div>

<!-- RELATED:END -->
