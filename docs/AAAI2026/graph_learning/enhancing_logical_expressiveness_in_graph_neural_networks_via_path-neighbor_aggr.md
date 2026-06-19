---
title: >-
  [论文解读] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation
description: >-
  [AAAI 2026][图学习][逻辑表达力] PN-GNN 提出在条件消息传递的基础上聚合推理路径上的邻居节点嵌入，以即插即用的方式增强 GNN 的逻辑规则表达力（严格超越 C-GNN），同时避免标注技巧（labeling trick）对泛化能力的损害，在合成数据集和真实知识图谱推理任务上均取得提升。
tags:
  - "AAAI 2026"
  - "图学习"
  - "逻辑表达力"
  - "知识图谱推理"
  - "路径邻居聚合"
  - "图神经网络"
  - "标注技巧"
---

# Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation

**会议**: AAAI 2026  
**arXiv**: [2511.07994](https://arxiv.org/abs/2511.07994)  
**代码**: 无  
**领域**: 图学习 / 知识图谱推理  
**关键词**: 逻辑表达力, 知识图谱推理, 路径邻居聚合, 条件GNN, 标注技巧

## 一句话总结

PN-GNN 提出在条件消息传递的基础上聚合推理路径上的邻居节点嵌入，以即插即用的方式增强 GNN 的逻辑规则表达力（严格超越 C-GNN），同时避免标注技巧（labeling trick）对泛化能力的损害，在合成数据集和真实知识图谱推理任务上均取得提升。

## 研究背景与动机

**领域现状**：GNN 在知识图谱（KG）推理中表现出色。其表达力可从两方面评估：区分非同构图的能力（与 WL 测试相关）和学习特定逻辑规则结构的能力。R-GNN（如 R-GCN、CompGCN）做关系感知消息传递但规则学习能力弱；C-GNN（如 NBFNet、RED-GNN）通过标记头实体做条件消息传递，理论上能学习所有 CML（计数模态逻辑）公式。

**核心痛点**：C-GNN 的逻辑表达力等价于 CML，但 CML 无法表达某些重要规则结构，特别是 U 结构——两条路径从同一中间节点分叉再汇合。C-GNN 无法区分 T 结构和 U 结构（两者的 CML 表达式相同）。

**标注技巧的问题**：EL-GNN 通过给节点添加常量标签来增强区分能力，确实能学习 U 结构，但代价是覆盖度（Coverage）下降——将变量替换为常量后规则的适用范围缩小，泛化能力减弱，且在归纳设定中难以应用。

## 方法详解

### 整体框架

PN-GNN（Path-Neighbor enhanced GNN）是一个即插即用模块，叠加在 C-GNN（如 NBFNet）之上：

1. 先用 C-GNN 做 L 层条件消息传递获得实体对表示
2. 在推理路径上聚合邻居节点的表示
3. 将聚合的路径邻居信息与 C-GNN 表示融合

### 关键设计

**1. 路径邻居聚合**

给定头实体 u、尾实体 v 和查询关系 q，PN-GNN 聚合推理路径 P_uv 上的邻居节点：

h_ij = POOL({h_w|u,q^(L) | w in P_uv, d(u,w)=i, d(w,v)=j})

其中 i 表示 w 到头实体的距离，j 表示 w 到尾实体的距离。POOL 可以是 max、min、mean 等标准池化函数。

**2. 表示融合**

将路径邻居聚合结果与 C-GNN 表示融合：

h_d = MLP_11(h_11) * MLP_12(h_12) * MLP_21(h_21)
h_v|u,q = h_d * h_v|u,q^(L)

为平衡表达力和效率，默认使用 2-hop 邻居：
- h_11：距头、尾实体各 1 跳的邻居
- h_12：距头 1 跳、距尾 2 跳的邻居
- h_21：距头 2 跳、距尾 1 跳的邻居

**3. 为什么能区分 U 和 T 结构**

- T 结构：两条链从同一个头实体出发，到同一个尾实体，但中间节点可以不同
- U 结构：类似 T 但要求两条链从同一个中间分叉节点出发

C-GNN 用 CML 无法表达"两条边从同一节点出发"这一约束。PN-GNN 通过聚合路径邻居信息，得到的 h_11 对 T 和 U 结构是不同的——在 U 中分叉点的邻居聚合结果反映了"共享分叉点"的结构特征。

**4. 理论分析**

- **引理 7**：C-GNN 能学的 CML 公式 PN-GNN 也能学
- **引理 8**：PN-GNN 能学 U 结构（C-GNN 不能）
- **定理 9**：PN-GNN 的逻辑表达力严格强于 C-GNN
- **定理 12**：(k+1)-hop PN-GNN 的表达力严格强于 k-hop

**5. 对比标注技巧的优势**

标注技巧将变量替换为常量，导致覆盖度下降和无法用于归纳设定。PN-GNN 不引入常量标签，保持原有变量的灵活性，覆盖度不减且可自然应用于归纳设定。

### 损失函数 / 训练策略

基于 C-GNN 骨干（NBFNet），用 sigmoid 预测尾实体条件概率，训练损失为负对数似然。负样本在 PCA 假设下通过随机替换头/尾实体构造。

## 实验关键数据

### 合成数据集逻辑规则学习（Hits@1）

| 方法 | C3 | C4 | I1 | I2 | T | U | T_label | U_label |
|------|-----|-----|-----|-----|------|------|---------|---------|
| R-GCN | 1.6 | 3.1 | 4.4 | 2.4 | 6.7 | 1.4 | - | - |
| NBFNet | 100 | 100 | 100 | 100 | 100 | 54.1 | 60.0 | 56.8 |
| EL-GNN | 100 | 100 | 100 | 100 | 100 | 75.7 | 22.0 | 59.5 |
| **PN-GNN** | **100** | **100** | **100** | **100** | **100** | **69.9** | **60.0** | **68.9** |

PN-GNN 在 U 上提升 15.8%（vs NBFNet），在 T_label 和 U_label 上远优于 EL-GNN。

### 真实数据集传导推理

| 方法 | FB15K237 MRR | FB15K237 H@10 | WN18RR MRR | WN18RR H@10 |
|------|-------------|--------------|------------|-------------|
| NBFNet | 0.415 | 59.9 | 0.551 | 66.6 |
| EL-GNN | 0.421 | 59.8 | 0.555 | 66.4 |
| **PN-GNN** | **0.423** | **60.2** | **0.555** | **66.9** |

### 消融实验

| 方法 | U (H@1) | FB15K237v1 (H@10) | WN18RRv1 (H@10) |
|------|---------|-------------------|-----------------|
| NBFNet | 54.1 | 17.1 | 60.1 |
| PN-GNN_11（仅1-hop） | 48.7 | 20.2 | **62.8** |
| PN-GNN_12-21（仅2-hop） | **69.9** | 18.5 | 62.5 |
| PN-GNN（完整） | **69.9** | **22.1** | 62.5 |

### 关键发现

- U 结构的区分主要依赖 2-hop 邻居（PN-GNN_12-21 与完整模型等效）
- 对关系较少且短路径为主的 WN18RR，1-hop 邻居最有效
- EL-GNN 在引入标签的 T_label 数据集上性能暴跌（22.0 vs 100），验证标注技巧损害泛化
- PN-GNN 在归纳设定中依然有效（EL-GNN 无法应用）

## 亮点与洞察

- 从逻辑表达力角度分析 GNN 的局限，而非仅关注 WL 测试的区分力
- 路径邻居聚合是一种即插即用的增强方式，不改变 C-GNN 的基本训练流程
- 覆盖度（Coverage）概念将规则的泛化能力形式化，清晰解释了标注技巧的副作用
- 理论证明 (k+1)-hop 严格强于 k-hop，为选择传播范围提供理论依据

## 局限与展望

- 多跳场景下计算代价随跳数增长，目前限制在 2-hop 以平衡效率
- 仅在 NBFNet 上实例化，理论上应扩展到 RED-GNN、A*Net 等其他 C-GNN
- 合成数据集较为简单，真实 KG 中 U 结构的实际比例和重要性有待进一步论证
- 路径邻居的选择依赖于精确路径距离计算，大规模图上可能产生瓶颈

## 相关工作与启发

- **NBFNet**：基于 Bellman-Ford 的条件消息传递，PN-GNN 的骨干
- **EL-GNN**：用标注技巧增强逻辑表达力，但损害泛化
- **R-WL 测试**：将 WL 测试扩展到关系图，形式化了 R-GNN 的表达力上界
- **GraIL / CoMPILE**：子图方法做归纳推理，PN-GNN 在归纳设定中超越这些基线

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 理论深度 | 4 |
| 实验充分度 | 4 |
| 写作质量 | 3 |
| 实用性 | 4 |
| 总评 | 3.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Logical Characterizations of GNNs with Mean Aggregation](logical_characterizations_of_gnns_with_mean_aggregation.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [\[AAAI 2026\] Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees](adaptive_initial_residual_connections_for_gnns_with_theoretical_guarantees.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](../../NeurIPS2025/graph_learning/logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](../../NeurIPS2025/graph_learning/sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
