---
title: >-
  [论文解读] TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation
description: >-
  [ICML 2025][图学习][图神经网络] 提出 TINED，将 GNN 中特征变换（FT）的参数直接注入 MLP（Teacher Injection），并用 Dirichlet 能量蒸馏传递 GNN 层中 FT 与图传播（GP）的对立平滑特性，在 7 个数据集上超越 GNN 教师，推理速度提升 94 倍。
tags:
  - "ICML 2025"
  - "图学习"
  - "图神经网络"
  - "Teacher Injection"
  - "Dirichlet Energy"
  - "推理加速"
---

# TINED: GNNs-to-MLPs by Teacher Injection and Dirichlet Energy Distillation

**会议**: ICML 2025  
**arXiv**: [2412.11180](https://arxiv.org/abs/2412.11180)  
**代码**: [https://github.com/scottjiao/TINED_ICML25/](https://github.com/scottjiao/TINED_ICML25/)  
**领域**: 图学习  
**关键词**: GNN蒸馏, GNN-to-MLP, Teacher Injection, Dirichlet Energy, 推理加速

## 一句话总结

提出 TINED，将 GNN 中特征变换（FT）的参数直接注入 MLP（Teacher Injection），并用 Dirichlet 能量蒸馏传递 GNN 层中 FT 与图传播（GP）的对立平滑特性，在 7 个数据集上超越 GNN 教师，推理速度提升 94 倍。

## 研究背景与动机

### 现有痛点

**现有痛点**：GNN 消息传递需多跳邻域数据，在延迟敏感场景中难以部署。

### 2. 现有蒸馏的不足

GLNN 仅用软标签蒸馏，忽略 GNN 层内部的细粒度知识。

### 3. 关键观察

- FT 在计算上等价于 MLP 的 FC 层
- FT 和 GP 在平滑效果上呈对立：GP 激进平滑，FT 保守甚至多样化

## 方法详解

### 整体框架

1. **Teacher Injection**：将 GNN FT 的参数直接移植到 MLP FC 层，再微调
2. **Dirichlet Energy Distillation**：用 DE ratio 传递 FT/GP 的对立平滑特性到 MLP

### 关键设计

#### 1. Teacher Injection

- FT 和 FC 数学形式相同：$h' = \sigma(Wh + b)$
- 直接复制参数，用另一个 FC 层模拟 GP
- 理论证明 GP 可被 FC 近似，误差与图拉普拉斯特征值相关

#### 2. Dirichlet Energy Distillation

- DE ratio > 1 表示保守（多样化），< 1 表示激进（平滑）
- 蒸馏损失让 MLP 各层 DE ratio 匹配 GNN 对应层

## 实验关键数据

### 主实验：节点分类

| 方法 | Citeseer | Cora | PubMed | 速度 |
|------|---------|------|--------|-----|
| GCN Teacher | 73.1% | 81.5% | 79.0% | 1x |
| MLP | 61.2% | 60.0% | 71.4% | 94x |
| GLNN | 74.0% | 81.6% | 79.8% | 94x |
| NOSMOG | 75.5% | 82.3% | 80.5% | 94x |
| **TINED** | **77.0%** | **83.2%** | **81.3%** | **94x** |

### 消融实验

| 配置 | Citeseer | 说明 |
|------|---------|------|
| TINED 完整 | 77.0% | TI + DE |
| w/o TI | 74.8% | 退化为软标签 |
| w/o DE | 75.6% | 失去平滑传递 |
| 仅软标签 (GLNN) | 74.0% | 基线 |

### 关键发现

- Teacher Injection 贡献 +2.2%，DE Distillation 贡献 +1.4%
- MLP 通过蒸馏可以超越 GNN 教师——"又快又好"
- 推理速度提升 94 倍

## 亮点与洞察

- **参数移植的巧妙**：发现 FT = FC 等价性，直接移植而非间接蒸馏
- **对立平滑的发现**：此前未被注意到的 GNN 内部结构性质
- **理论保证**：GP -> FC 近似的误差界
- **超越教师**：MLP 学生在多个数据集上超越 GNN 教师

## 局限与展望

- 仅测试节点分类任务，图级和边级任务待验证
- 对 GAT 等注意力 GNN 的适配性待研究
- 深层 GNN 中 DE ratio 的表现未充分探索
- 可结合 VQGraph 等结构感知方法进一步提升

## 相关工作与启发

- **vs GLNN**：仅软标签蒸馏，TINED 加入参数注入和能量蒸馏
- **vs NOSMOG**：考虑图结构但仍整体蒸馏，TINED 做逐层蒸馏
- **vs VQGraph**：学习结构感知 tokenizer，TINED 直接移植

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ FT=FC 等价性 + 对立平滑 + 层级蒸馏
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个数据集 x 多种教师
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨、观察新颖
- 价值: ⭐⭐⭐⭐⭐ GNN 加速的高效方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality](balancing_efficiency_and_expressiveness_subgraph_gnns_with_walk-based_centrality.md)
- [\[ICML 2026\] ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs](../../ICML2026/graph_learning/eralign_energy-based_representation_alignment_of_gnns_and_llms_on_text-attribute.md)
- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)
- [\[ICML 2025\] Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes](machines_and_mathematical_mutations_using_gnns_to_characterize_quiver_mutation_c.md)
- [\[ICML 2026\] ProMoS: Generalist Graph Anomaly Detection via Prototype-Based Distillation](../../ICML2026/graph_learning/generalist_graph_anomaly_detection_via_prototype-based_distillation.md)

</div>

<!-- RELATED:END -->
