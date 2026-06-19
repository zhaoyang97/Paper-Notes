---
title: >-
  [论文解读] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily
description: >-
  [AAAI2026][图学习][图神经网络] 提出 AD-GNN，通过理论分析节点级别的同配/异配特性，为每个节点自适应分配不同的聚合深度，在统一框架中同时处理同配和异配图上的节点分类任务。 传统 GNN 建立在同配假设（homophily assumption）之上——相连节点倾向于拥有相同标签。然而在社交网络、网页图等真…
tags:
  - "AAAI2026"
  - "图学习"
  - "图神经网络"
  - "adaptive depth"
  - "heterophily"
  - "node classification"
  - "homophily"
---

# Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily

**会议**: AAAI2026  
**arXiv**: [2511.06608](https://arxiv.org/abs/2511.06608)  
**代码**: 待确认  
**领域**: 图学习  
**关键词**: GNN, adaptive depth, heterophily, node classification, homophily  

## 一句话总结
提出 AD-GNN，通过理论分析节点级别的同配/异配特性，为每个节点自适应分配不同的聚合深度，在统一框架中同时处理同配和异配图上的节点分类任务。

## 背景与动机
传统 GNN 建立在同配假设（homophily assumption）之上——相连节点倾向于拥有相同标签。然而在社交网络、网页图等真实场景中，大量存在异配图（heterophilic graphs），相连节点往往属于不同类别，导致性能显著下降。

现有改进工作存在两个关键不足：

1. **固定聚合深度**：绝大多数 GNN 对所有节点使用统一的层数，忽视了不同节点因局部同配程度和邻域结构不同，所需的信息传播深度也不同
2. **缺乏统一框架**：现有方法往往只针对同配或异配设计，无法在一个架构中自然适应两种模式

作者的核心观察是：即便在同一张图中，最优的信息传播策略对不同节点也是不同的。这一观察促使他们从理论角度建立节点级别的结构-标签特性与信息传播动态之间的联系。

## 核心问题
如何根据每个节点的局部邻域结构（度、同标签/异标签邻居比例）自适应地决定 GNN 聚合深度，使同一模型能同时高效处理同配和异配图？

## 方法详解

### 理论基础：信号保留因子

对每个节点 $v$，定义其 profile 为 $(d_v^+, d_v^-, d_v)$，分别表示同标签邻居数、异标签邻居数和总度。在 CSBM（Contextual Stochastic Block Model）假设下，定义信号保留因子（Signal Preservation Factor）：

$$\alpha_v = \frac{1 + d_v^+ - d_v^-}{d_v + 1}$$

**定理 1（标签聚合效应）**：经过一层聚合后，节点 $v$ 的分类质量为：

$$Q_v = \frac{\alpha_v^2 (d_v + 1) \Delta^2}{\sigma_{\text{intra}}^2}$$

其中 $\Delta^2$ 为类间信号方差，$\sigma_{\text{intra}}^2$ 为类内噪声方差。由此得出三个关键推论：

- **强同配**（$d_v^+ \gg d_v^-$）：$\alpha_v \approx 1$，聚合总是有益，分类质量随度线性增长
- **强异配**（$d_v^- \gg d_v^+$）：低度节点发生信号抵消（$Q_v \approx 0$），但高度节点可通过学习反向关系恢复性能
- **混合情况**（$d_v^+ \approx d_v^-$）：$\alpha_v \approx \frac{1}{d_v+1}$，度越高信号抵消越严重

**定理 2（多层聚合效应）**：经过 $n$ 层后，分类质量为：

$$Q_v^n = \frac{\alpha_v^{2n} (d_v+1)^n \Delta^2}{\sigma_{\text{intra}}^2}$$

若 $|\alpha_v| < 1$，信号退化随深度指数恶化。

### AD-GNN 架构

基于理论分析，提出 Depth Benefit Metric（深度收益指标）：

$$\varepsilon_v^n = \frac{Q_v^n}{Q_v^0} = (\alpha_v^2 \cdot (d_v + 1))^n$$

$\varepsilon_v^n > 1$ 表示深层聚合有益，$< 1$ 表示有害。

**停止深度分配**：设最大允许深度为 $t_{\max}$，通过可学习的单调递增阈值函数 $\tau_\theta(t) = \lambda + (1-\lambda) \cdot \theta(t)$ 确定每个节点的停止层 $T(v)$。随层数增加，深度收益较低的节点被逐步过滤，不再参与更深层聚合。

**消息传递**：对 $t \leq T(v)$ 正常执行聚合-更新操作；对 $t > T(v)$，节点表示保持不变。该机制可即插即用地增强 GCN、GAT、GraphSAGE 等主流 GNN 骨架。

### 深度收益指标计算

由于半监督训练中标签不完整，通过可学习函数 $f_\delta(\mathbf{h}_u, \mathbf{h}_v) \in [0,1]$ 估计相邻节点同标签概率，进而估算 $\hat{\alpha}_v$ 和 $\hat{\varepsilon}_v$。训练目标为：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{reg}}$$

其中正则项 $\mathcal{L}_{\text{reg}}$ 利用训练集中已知标签边约束 $f_\delta$ 学习有意义的相似度。

### 快速变体 AD-GNN_fast

用静态的基于度的近似替代可学习相似度函数。利用高度节点倾向连接高度节点（度同配性），用 $p_{uv} = \frac{d_u \times d_v}{\max_{(i,j) \in E}(d_i \times d_j)}$ 估计同标签概率。消除了学习 $f_\delta$ 的计算开销和正则化项，复杂度从 $\mathcal{O}(|E| \times d + t_{\max} \times (|E|+|V|))$ 降至 $\mathcal{O}(t_{\max} \times (|E|+|V|))$。

## 实验关键数据

在 11 个数据集上评估（5 个同配 + 6 个异配），使用 6 种 GNN 骨架：

| 骨架 | 基线（Cornell） | AD-版本 | 基线（Texas） | AD-版本 |
|------|----------------|---------|--------------|---------|
| GCN | 55.14 | **88.51** | 60.00 | **92.30** |
| GAT | 53.64 | **86.17** | 61.21 | **90.49** |
| GraphSAGE | 75.95 | **89.57** | 82.43 | **92.95** |
| MixHop | 73.51 | **90.21** | 77.84 | **94.43** |

核心发现：

- 在异配图上提升最为显著（如 Cornell 上 GCN 从 55.14% → 88.51%，提升 33 个百分点）
- 在同配图上也有稳定提升（如 Photo 从 89.30% → 94.10%）
- 快速变体性能接近甚至偶尔超过完整版
- 可扩展性实验（ogbn-arxiv）：AD-GCN_fast 参数量与 GCN 相同，运行时开销仅增加约 5%，在 depth=8 时准确率从 68.39% 提升至 70.42%
- AD-GNN 有效缓解过平滑问题，在深层网络中保持稳定性能

## 亮点
1. **理论驱动的架构设计**：从信号保留因子出发建立完整的节点级理论框架，推导出强同配/强异配/混合场景下的不同行为，理论与实验结论高度一致
2. **即插即用设计**：AD-GNN 可无缝集成到任意现有 GNN 骨架（GCN/GAT/GraphSAGE/MixHop/GATv2/DirGNN），无需修改骨架架构
3. **兼顾同配和异配**：单一统一框架同时处理两种模式，无需预先判断图的类型
4. **快速变体实用性强**：AD-GNN_fast 以极低的额外开销实现接近完整版的性能
5. **副产品：缓解过平滑**：自适应深度机制天然地阻止了冗余聚合层对信号的过度平滑

## 局限与展望
1. **理论假设较强**：CSBM 结构 + 高斯特征 + 二分类 + 层间独立假设，与真实图存在差距
2. **快速变体的度同配性假设**：AD-GNN_fast 依赖度同配性（高度连高度），在度与标签无关的图上可能失效
3. **超参数 $\lambda$**：需针对同配/异配图分别调整（同配图 $\lambda=0$ 最优，异配低度图需 $\lambda > 0$），未来可考虑数据驱动学习
4. **停止深度粒度**：当前基于全局阈值函数做停止判断，未充分利用节点局部拓扑的动态变化
5. **仅验证节点分类**：未探索图分类、链路预测等其他任务上的效果

## 与相关工作的对比
- **异配感知 GNN**（MixHop、DirGNN 等）：AD-GNN 是正交的补充，可在这些方法之上进一步提升
- **深度自适应 GNN**（wu2024depth 用强化学习搜索深度、ADMP-GNN 用中心性启发式）：AD-GNN 首次从异配理论出发分析邻域标签组成对传播深度的影响
- **图重连方法**（改变图结构增加同配性）：AD-GNN 不修改图结构，而是自适应调整每个节点的聚合层数

## 启发与关联
- "信号保留因子"是一个简洁有力的理论工具，可以推广到其他需要分析聚合效果的场景
- 自适应深度的思想可以扩展到 Transformer 中的自适应层数（early exit），特别是在图 Transformer 场景
- 混合同配/异配节点的信号抵消问题提示：在异构图学习中，可能需要节点级别的传播策略而非全局策略

## 评分
- 新颖性: ⭐⭐⭐⭐ （理论驱动的自适应深度思路新颖，信号保留因子的形式化清晰）
- 实验充分度: ⭐⭐⭐⭐⭐ （11个数据集 × 6个骨架，消融、可扩展性、过平滑分析齐全）
- 写作质量: ⭐⭐⭐⭐ （理论推导清晰，结构完整，推论与实验对应良好）
- 价值: ⭐⭐⭐⭐ （即插即用、理论扎实，对图学习社区实用价值高）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](../../NeurIPS2025/graph_learning/making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)
- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](posterior_label_smoothing_for_node_classification.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[ACL 2026\] Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval](../../ACL2026/graph_learning/autonomous_knowledge_graph_exploration_with_adaptive_breadth-depth_retrieval.md)

</div>

<!-- RELATED:END -->
