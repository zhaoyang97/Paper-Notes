---
title: >-
  [论文解读] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering
description: >-
  [AAAI 2026][图学习][层次聚类] 提出 HypCSE，将离散结构熵（SE）松弛为双曲空间中的连续结构熵（CSE），结合图结构学习和对比学习，实现端到端可微的层次聚类，在 7 个数据集上全面超越离散和连续层次聚类方法。 层次聚类将数据点组织为树状图（dendrogram），是经典的无监督学习任务…
tags:
  - "AAAI 2026"
  - "图学习"
  - "层次聚类"
  - "结构熵"
  - "双曲空间"
  - "图结构学习"
  - "对比学习"
---

# Hyperbolic Continuous Structural Entropy for Hierarchical Clustering

**会议**: AAAI 2026  
**arXiv**: [2512.00524](https://arxiv.org/abs/2512.00524)  
**代码**: [GitHub](https://github.com/SELGroup/HypCSE)  
**领域**: Graph Learning / 层次聚类  
**关键词**: 层次聚类, 结构熵, 双曲空间, 图结构学习, 对比学习  

## 一句话总结

提出 HypCSE，将离散结构熵（SE）松弛为双曲空间中的连续结构熵（CSE），结合图结构学习和对比学习，实现端到端可微的层次聚类，在 7 个数据集上全面超越离散和连续层次聚类方法。

## 背景与动机

层次聚类将数据点组织为树状图（dendrogram），是经典的无监督学习任务。现有方法面临两大挑战：

1. **大多数方法缺乏全局目标函数**：凝聚式方法逐步合并最近的簇对，分裂式方法逐步切分——都是贪心策略，容易产生次优树
2. **基于图的连续方法优化目标定义在完全图或静态预定义图上**：完全图被噪声边污染，预定义图无法在训练中学习——未充分利用数据特征

结构熵（Structural Entropy）从信息论角度衡量树的质量，量化图上随机游走的最小编码长度，是优秀的全局目标。但 SE 定义在离散树上，不可微分。

## 核心问题

如何将离散结构熵松弛为可微目标，并在自适应学习的图结构上进行端到端优化，实现高质量层次聚类？

## 方法详解

### 整体框架

HypCSE 包含两个模块：(1) 双曲层次聚类模块：构建图 → 双曲编码 → 最小化 CSE → 解码为分区树；(2) 图结构学习（GSL）模块：通过对比学习引导的图学习器动态更新图结构。

### 关键设计

1. **CSE 目标函数**：将 SE 重写为基于最低公共祖先（LCA）的形式：
$$\mathcal{H}^\mathcal{T}(G) = \frac{2}{\mathcal{V}(G)} \sum_{(v_i,v_j)\in E} \mathbf{W}_{ij} \log_2 \mathcal{V}_{\mathcal{T}_i \vee \mathcal{T}_j} - \frac{1}{\mathcal{V}(G)} \sum_{v_i \in V} \mathcal{V}_{\mathcal{T}_i} \log_2 \mathcal{V}_{\mathcal{T}_i}$$
利用双曲空间中测地线与树中最短路径的对应关系，用 Poincaré 模型中的双曲 LCA 近似离散 LCA，非可微的指示函数用 scaled softmax 松弛。

2. **双曲图编码**：使用 3 层 Lorentz 卷积（LConv）将图编码到 Lorentz 模型 $\mathbb{L}^{\kappa,d}$，然后投影到 Poincaré 模型以计算 CSE。

3. **图结构学习**：采用锚图-学习器图双视图设计。锚图 $G_a$ 提供稳定引导，学习器图 $G_l$ 通过 GSL 编码器从数据中学习。锚图通过 bootstrapping 更新：$E_a \leftarrow S_\tau(E_a) + S_{1-\tau}(E_l)$。

4. **双曲对比学习**：在双曲空间中最小化同一节点两个视图的距离、最大化不同节点间距离，引导图学习器产生更具判别力的特征。

5. **总目标**：$\mathcal{L}_{HypCSE} = \mathcal{L}_{cse} + \eta_1 \mathcal{L}_{con} + \eta_2 \mathcal{L}_{cen}$，其中 $\mathcal{L}_{cen}$ 确保叶子嵌入分散在原点周围。

## 实验关键数据

### 层次聚类质量（Dendrogram Purity %）

| 方法 | Zoo | Iris | Wine | Br.Cancer | OptDigits | Spambase | PenDigits |
|------|-----|------|------|-----------|-----------|----------|-----------|
| SingleLinkage | 97.6 | 81.2 | 67.9 | 85.1 | 73.3 | 58.9 | 70.0 |
| HCSE | 97.3 | 89.7 | 71.1 | 94.2 | 81.5 | 55.2 | 76.9 |
| HypHC | 96.8 | 76.0 | 88.7 | 96.5 | 33.5 | 75.4 | 11.7 |
| FPH | 89.9 | 85.3 | 92.8 | 92.6 | 81.0 | 54.8 | 69.6 |
| **HypCSE** | **97.9** | **95.1** | **93.9** | **96.8** | **86.4** | **75.5** | **81.4** |

HypCSE 在全部 7 个数据集上 DP 指标排名第一，SE 指标在 4 个数据集最优。

### 灵活性验证（相似性分类，ACC %）

| 方法 | Zoo | Iris | Glass | Seg. |
|------|-----|------|-------|------|
| HypHC-ETE | 87.9 | 85.6 | 54.4 | 67.7 |
| **HypCSE** | **88.3** | **91.8** | **54.7** | **78.8** |

## 亮点

- **信息论目标 + 双曲几何的精妙结合**：CSE 通过双曲 LCA 桥接离散树与连续嵌入
- **GSL 解决静态图瓶颈**：对比学习引导的图结构更新有效消除噪声边
- **端到端可微且可扩展**：树解码复杂度可降至 $O(n \log n)$
- **理论基础扎实**：证明了 SE 与图导纳率的下界关系（Lemma 3），保证优化方向正确

## 局限与展望

- Spambase 数据集上对超参（$\tau$、$\eta_1$）敏感，说明层次结构模糊时 GSL 不稳定
- CSE 目标中的 softmax 松弛引入近似误差，温度参数 $t_1$ 的最优选择依赖数据集
- 仅在 UCI 小规模数据集上验证，缺乏大规模图数据的实验

## 与相关工作的对比

HypHC 也在双曲空间做层次聚类但优化 Dasgupta cost 在完全图上，对噪声敏感且大规模数据崩溃（PenDigits 仅 11.7%）。UFit/FPH 松弛 ultrametric 或 Dasgupta cost 但使用静态图。HCSE 优化离散 SE 但无法端到端学习。HypCSE 统一了 SE 全局目标、双曲几何优势和图结构自适应学习。

## 启发与关联

- CSE 的松弛思路可推广到其他组合优化目标（如 normalized cut）的连续化
- 双曲嵌入 + 结构熵的结合可用于社交网络社区检测、知识图谱层次发现
- GSL 模块是通用的，可集成到任何需要图结构输入的下游任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将结构熵松弛到双曲连续空间，创意出色
- 实验充分度: ⭐⭐⭐⭐ 消融完整、参数敏感性分析详尽，但缺大规模实验
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，图示清晰
- 价值: ⭐⭐⭐⭐ 为层次聚类提供了新的信息论 + 几何学范式
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CORDS - Continuous Representations of Discrete Structures](../../ICLR2026/graph_learning/cords_-_continuous_representations_of_discrete_structures.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](../../ICML2025/graph_learning/hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)
- [\[ACL 2026\] GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion](../../ACL2026/graph_learning/gs-quant_granular_semantic_and_generative_structural_quantization_for_knowledge_.md)

</div>

<!-- RELATED:END -->
