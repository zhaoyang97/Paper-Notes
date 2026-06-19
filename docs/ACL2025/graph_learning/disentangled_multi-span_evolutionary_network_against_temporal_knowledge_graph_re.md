---
title: >-
  [论文解读] Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning
description: >-
  [ACL 2025 Findings][图学习][时序知识图谱] 提出 DiMNet，通过多跨度演化策略和跨时间解耦机制，分离节点语义的活跃/稳定特征，显著提升时序知识图谱（TKG）外推推理性能，在四个基准数据集上取得 SOTA。 时序知识图谱（Temporal Knowledge Graph, TKG）以四元组 $(s…
tags:
  - "ACL 2025 Findings"
  - "图学习"
  - "时序知识图谱"
  - "外推推理"
  - "图神经网络"
  - "解耦表示"
  - "多跨度演化"
---

# Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning

**会议**: ACL 2025 Findings  
**arXiv**: [2505.14020](https://arxiv.org/abs/2505.14020)  
**代码**: 未提供  
**领域**: 图学习  
**关键词**: 时序知识图谱, 外推推理, 图神经网络, 解耦表示, 多跨度演化

## 一句话总结

提出 DiMNet，通过多跨度演化策略和跨时间解耦机制，分离节点语义的活跃/稳定特征，显著提升时序知识图谱（TKG）外推推理性能，在四个基准数据集上取得 SOTA。

## 研究背景与动机

时序知识图谱（Temporal Knowledge Graph, TKG）以四元组 $(s, r, o, t)$ 表示带时间戳的事实，其推理任务旨在基于历史子图序列预测未来缺失的事实。现有基于演化的方法（如 RE-GCN、TiPNN 等）通常对历史子图序列进行逐步建模，但存在以下不足：

**忽略多跨度语义演化**：现有方法在每个时间步仅捕获局部结构语义，未能感知不同时间跨度之间邻居特征的中间更新，导致无法学习细粒度的多跨度演化模式。

**未解耦活跃与稳定特征**：节点的语义在演化过程中同时包含频繁变化的"活跃"成分和相对稳定的"稳定"成分，现有方法未对此进行区分，限制了对语义变化模式的精确建模。

**推理阶段拓扑不确定性**：在预测未来事实时，未来时间步的图拓扑结构未知，直接使用历史演化的表示可能不够准确。

## 方法详解

### 整体架构

DiMNet 由三个核心组件构成：**多跨度演化模块**、**跨时间解耦模块**和**采样虚拟子图解码器**。

### 1. 多跨度演化（Multi-span Evolution）

给定历史子图序列 $\{G_{t-m}, ..., G_{t-1}\}$，DiMNet 在每个时间步 $t_i$ 内执行 $\omega$ 层 GNN 消息传递，捕获局部结构语义。关键创新在于：在当前时间步 $t_i$ 的第 $l$ 层消息聚合中，不仅使用当前子图的邻居信息，还感知前一时间步 $t_{i-1}$ 第 $l$ 层的邻居特征，从而实现跨时间步的多跨度语义感知：

$$h_v^{(l)} = \text{Agg}\left(h_v^{(l-1)},\ \{h_u^{(l-1)}\}_{u \in \mathcal{N}(v, t_i)},\ \{h_u^{(l)}|_{t_{i-1}}\}\right)$$

这使得模型能够在演化过程中捕获不同层次和不同跨度的语义更新。

### 2. 跨时间解耦（Cross-time Disentanglement）

在相邻时间步之间，DiMNet 自适应地将节点表示解耦为**活跃因子** $\mathcal{A}$ 和**稳定因子** $\mathcal{B}$：

- **活跃因子**：捕获在相邻时间步之间发生显著变化的语义成分，通过注意力机制从两个时间步的表示差异中提取
- **稳定因子**：捕获在时间演化中保持相对不变的语义成分

使用 GRU 对活跃因子进行迭代更新，建模活跃语义在时间上的持续变化模式。解耦损失 $\mathcal{L}_{dis}$ 通过最小化活跃因子与稳定因子之间的相似度来鼓励两者正交分离：

$$\mathcal{L}_{dis} = \sum \text{sim}(\mathcal{A}, \mathcal{B})$$

### 3. 采样虚拟子图解码器（Sampling Virtual Subgraph Decoder）

为应对推理阶段未来拓扑的不确定性，DiMNet 设计了虚拟子图 $G_{\text{INF}}$：

1. 根据演化后的实体表示对所有候选三元组进行初步评分
2. 采样 Top-$k$ 个最高分三元组构建虚拟子图
3. 在虚拟子图上再次进行 GNN 消息传递，利用预测的拓扑结构增强表示
4. 基于增强后的表示进行最终评分

### 训练损失

总损失为交叉熵损失与解耦损失的加权组合：

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{dis}$$

## 实验

### 主实验：整体性能对比

在 ICEWS14、ICEWS05-15、ICEWS18 和 GDELT 四个数据集上，与 15+ 个基线方法对比（包括传统 KG 方法、插值 TKG 方法和外推 TKG 方法），使用时间感知过滤的 MRR 和 Hits@{1,3,10} 指标：

| 模型 | ICEWS14 MRR | ICEWS05-15 MRR | ICEWS18 MRR | GDELT MRR |
|------|------------|----------------|------------|----------|
| RE-GCN | 41.78 | 48.03 | 30.58 | 19.64 |
| CEN | 42.17 | 46.84 | 30.84 | 20.18 |
| TiPNN | 41.79 | 45.35 | 32.17 | 21.17 |
| DaeMon | 40.68 | 44.50 | 31.85 | 20.73 |
| **DiMNet** | **45.72** | **58.93** | **34.13** | **21.93** |

DiMNet 在四个数据集上的 MRR 分别超越现有最优方法 **8.4%、22.7%、6.1% 和 3.6%**，特别是在 ICEWS05-15 上取得了大幅领先。

### 消融实验

| 变体 | ICEWS14 MRR | ICEWS05-15 MRR | ICEWS18 MRR | GDELT MRR |
|------|------------|----------------|------------|----------|
| DiMNet (完整) | 45.72 | 58.93 | 34.13 | 21.93 |
| w/o Multi-span | 40.75 | 51.17 | 30.74 | 20.99 |
| w/o Disentangle | 34.34 | 53.22 | 33.60 | 20.51 |
| w/o 两者 | 36.09 | 50.36 | 30.88 | 20.71 |
| w/o $G_{\text{INF}}$ | 36.10 | 45.45 | 30.02 | 19.81 |

- 去除多跨度策略后性能显著下降，证明跨时间步的邻居感知对细粒度演化建模至关重要
- 去除解耦组件后在 ICEWS14 上降幅最大（MRR 从 45.72 降至 34.34），说明活跃/稳定特征分离是核心贡献
- 虚拟子图解码器的移除同样导致较大性能损失，验证了推理阶段拓扑增强的必要性

### 参数敏感性分析

- **历史序列长度 $m$**：在 ICEWS14 和 ICEWS18 上对不同 $m$ 值（2-15）进行测试，性能变化不大，尤其 ICEWS18 趋于稳定，验证了模型的鲁棒性
- **GNN 层数 $\omega$**：两个数据集在 $\omega=3$ 时达到最优，层数增加后性能趋于稳定
- **采样数 $k$**：在 ICEWS14 上 $k=50$ 最优，$k$ 过大（如80）引入噪声反而降低性能

## 亮点

- **多跨度演化机制新颖**：在 GNN 消息传递中引入跨时间步的邻居特征感知，突破了现有方法逐步独立建模的范式，能捕获更丰富的中间演化特征
- **解耦设计有理有据**：将节点语义分解为活跃和稳定因子，直觉清晰且消融实验充分验证了其有效性
- **虚拟子图解码器**：通过构建预测拓扑减轻推理阶段的不确定性，是一个实用的工程设计
- **实验全面**：四个数据集、15+ 基线、消融实验和参数敏感性分析覆盖完整

## 局限性

- **计算复杂度未讨论**：多跨度演化需要维护跨时间步的中间特征，虚拟子图需要对所有候选评分+采样+再推理，训练和推理开销可能较大
- **仅在事件型 TKG 数据集上验证**：ICEWS 和 GDELT 均为政治事件数据集，未在其他类型（如 Wikidata、YAGO 等知识型 TKG）上验证泛化性
- **解耦的可解释性有限**：虽然提出活跃/稳定因子的概念，但缺乏对解耦结果的可视化分析或语义解释
- **代码未开源**：无法复现验证

## 相关工作

- **传统 KG 嵌入**：DistMult、ComplEx、ConvE、RotatE — 不考虑时间维度
- **插值 TKG 推理**：TTransE、TA-DistMult、DE-SimplE、TNTComplEx — 对已知时间范围内的事实进行补全
- **外推 TKG 推理**：RE-NET、RE-GCN、CEN、TiPNN、DaeMon — 基于历史序列预测未来事实，但缺乏多跨度感知和特征解耦
- **图解耦学习**：DisenGCN、IPGDN 等 — 在静态图上解耦表示，未扩展到时序图

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐：多跨度演化和跨时间解耦的组合是对现有 TKG 推理范式的有意义改进
- **实验** ⭐⭐⭐⭐⭐：基线覆盖全面，消融设计合理，参数分析充分
- **写作** ⭐⭐⭐⭐：结构清晰，实验展示规范
- **影响力** ⭐⭐⭐：TKG 推理是 KG 领域的重要子方向，但受众相对有限；Findings 级别

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](a_generative_adaptive_replay_continual_learning_model_for_temporal_knowledge_gra.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](../../NeurIPS2025/graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)

</div>

<!-- RELATED:END -->
