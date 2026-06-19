---
title: >-
  [论文解读] GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers
description: >-
  [ICML2025][图学习][Transformer] 提出 GrokFormer，通过傅里叶级数参数化的 Kolmogorov-Arnold 可学习激活函数，在图 Laplacian 的多阶谱上自适应学习滤波器基，同时具备 **谱阶自适应** 和 **谱自适应** 能力，是目前唯一在两个维度上都可学习的图 Transformer 滤波器。
tags:
  - "ICML2025"
  - "图学习"
  - "Transformer"
  - "Kolmogorov-Arnold Network"
  - "傅里叶级数"
  - "谱图滤波器"
  - "节点分类"
  - "图分类"
---

# GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers

**会议**: ICML2025  
**arXiv**: [2411.17296](https://arxiv.org/abs/2411.17296)  
**代码**: [https://github.com/GGA23/GrokFormer](https://github.com/GGA23/GrokFormer)  
**领域**: 图Transformer / 谱图神经网络  
**关键词**: Graph Transformer, Kolmogorov-Arnold Network, 傅里叶级数, 谱图滤波器, 节点分类, 图分类

## 一句话总结

提出 GrokFormer，通过傅里叶级数参数化的 Kolmogorov-Arnold 可学习激活函数，在图 Laplacian 的多阶谱上自适应学习滤波器基，同时具备 **谱阶自适应** 和 **谱自适应** 能力，是目前唯一在两个维度上都可学习的图 Transformer 滤波器。

## 研究背景与动机

- **Graph Transformer 的低通瓶颈**：自注意力本质上是低通滤波器，只保留节点间的低频相似性信号，无法捕获高频差异信号，这在异配图 (heterophilic graph) 上尤为致命。
- **多项式滤波器的局限**：ChebyNet、GPRGNN、BernNet、JacobiConv 等方法用 $K$ 阶多项式逼近滤波器，虽然阶自适应但每个基的频率响应固定在预定义谱上，感受野也局限于 $K$ 跳邻居，灵活性不足。
- **Specformer 的局限**：Specformer 实现了谱自适应（可任意学习频率响应在一阶谱上），但仅限于一阶 Laplacian 谱，且滤波器学习复杂度为 $O(N^2)$，难以扩展到高阶谱信息。
- **核心问题**：能否设计一个 GT 滤波器，同时在 **谱阶**（从 1 到 $K$ 阶）和 **谱位置**（每个特征值处的响应）上都自适应学习？

| 模型类型 | 代表方法 | 阶自适应 | 谱自适应 |
|---------|---------|---------|---------|
| 多项式 GNN | ChebyNet, GPRGNN, BernNet, JacobiConv | ✓ | ✗ |
| 多项式 GT | FeTA, PolyFormer | ✓ | ✗ |
| 谱 GT | Specformer | ✗ | ✓ |
| **GrokFormer** | **本文** | **✓** | **✓** |

## 方法详解

### 核心思想：Graph Fourier KAN

受 Kolmogorov-Arnold Network (KAN) 启发，GrokFormer 用可学习的激活函数替代固定基函数。但 KAN 原始的 B-spline 分段函数训练困难，因此改用 **傅里叶级数** 参数化每个可学习函数，得到如下滤波器：

$$\phi_h(\lambda) = \sum_{k=1}^{K} \sum_{m=0}^{M} \left( \cos(m\lambda^k) \cdot a_{km} + \sin(m\lambda^k) \cdot b_{km} \right)$$

其中 $K$ 为最高阶数，$M$ 为频率分量数（网格大小），$a_{km}$、$b_{km}$ 为可训练傅里叶系数。

### 阶自适应与谱自适应

将上式分解为 $K$ 个阶别的滤波基 $b_k(\lambda)$：

$$b_k(\lambda) = \sum_{m=0}^{M} \left( \cos(m\lambda^k) \cdot a_m + \sin(m\lambda^k) \cdot b_m \right)$$

再引入可学习阶系数 $\alpha_k$ 自适应组合各阶：

$$h(\lambda) = \sum_{k=1}^{K} \alpha_k \, b_k(\lambda)$$

- $\alpha_k$ 控制谱阶：决定哪些阶的图 Laplacian 对任务更重要
- $a_{km}$, $b_{km}$ 控制谱位置：在特定阶 $k$ 上学习任意频率响应

最终的谱图卷积为：

$$\mathbf{X}_F^{(l)} = \mathbf{U} \, \text{diag}(h(\lambda)) \, \mathbf{U}^\top \mathbf{X}^{(l-1)}$$

### 网络架构

GrokFormer 每层由三部分组成：

1. **高效多头自注意力 (EMHA)**：将 $(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ 改为 $\mathbf{Q}(\mathbf{K}^\top \mathbf{V})$，复杂度从 $O(N^2 d)$ 降至 $O(Nd^2)$
2. **Graph Fourier KAN**：上述谱图卷积模块，捕捉全局谱域信息
3. **融合**：将空间域（EMHA）与谱域信号相加后经 LayerNorm + FFN

$$\mathbf{X}'^{(l)} = \text{EMHA}(\text{LN}(\mathbf{X}^{(l-1)})) + \mathbf{X}^{(l-1)} + \mathbf{X}_F^{(l)}$$

$$\mathbf{X}^{(l)} = \text{FFN}(\text{LN}(\mathbf{X}'^{(l)})) + \mathbf{X}'^{(l)}$$

### 理论性质

- **命题 4.2**：多项式滤波器 $h(\lambda) = \sum_{k=0}^{K} \alpha_k \lambda^k$ 是 GrokFormer 滤波器的特例
- **命题 4.3**：Specformer 滤波器也是 GrokFormer 滤波器的特例
- **命题 4.4**：GrokFormer 滤波器可以逼近任意连续函数，且构造排列等变的谱图卷积

### 复杂度

- 前向传播：$O(Nd(N + 2d) + KNM)$
- 大图可用稀疏特征分解 (SGE) 计算 $q \ll N$ 个特征值，复杂度降至 $O(2Nd^2 + KqM + Nqd)$

## 实验关键数据

### 节点分类（11 个数据集，精度 %）

| 方法 | Cora | Citeseer | Pubmed | Chameleon | Squirrel | Actor | Texas |
|------|------|----------|--------|-----------|----------|-------|-------|
| GCN | 87.14 | 79.86 | 86.74 | 59.61 | 46.78 | 33.23 | 77.38 |
| JacobiConv | 88.98 | 80.78 | 89.62 | 74.20 | 57.38 | 41.17 | 93.44 |
| Specformer | 88.57 | 81.49 | 90.61 | 74.72 | 64.64 | 41.93 | 88.23 |
| PolyFormer | 87.67 | 81.80 | 90.68 | 60.17 | 44.98 | 41.51 | 89.02 |
| **GrokFormer** | **89.57** | **81.92** | **91.39** | **75.58** | **65.12** | **42.98** | **94.59** |

- 同配图：在 6 个数据集上均取得最佳或第二，Physics 达 98.31%
- 异配图：5 个数据集全部最优，Squirrel 比 Specformer 高 0.48%，Texas 高 6.36%

### 图分类（5 个数据集，精度 %）

| 方法 | PROTEINS | MUTAG | PTC-MR | IMDB-B | IMDB-M |
|------|----------|-------|--------|--------|--------|
| Specformer | 70.9 | 96.3 | 82.9 | 86.6 | 58.5 |
| **GrokFormer** | **78.2** | **99.5** | **94.8** | **88.5** | **62.2** |

- MUTAG 达 99.5%，PTC-MR 达 94.8%，均大幅领先其他 GT 方法

### 滤波器拟合能力

在 6 种标准滤波器（低通/高通/带通/带阻/梳状/低频梳状）上，GrokFormer 的 SSE 最低、$R^2$ 最高。特别是在复杂的低频梳状滤波器上，多项式方法完全失败而 GrokFormer 可近似完美拟合。

## 亮点与洞察

1. **双维自适应**：首次在图 Transformer 中同时实现谱阶和谱位置的自适应学习，理论上统一了多项式滤波和 Specformer 滤波
2. **傅里叶 KAN 设计精妙**：用 $\sin$/$\cos$ 正交基替代 B-spline，既保留了 KAN 的表达力又大幅简化训练，且理论上有最优逼近收敛率
3. **异配图优势显著**：在 Chameleon/Squirrel/Texas 等异配图上提升明显，说明多阶谱信息对捕捉节点差异模式至关重要
4. **理论完备**：严格证明了多项式滤波器和 Specformer 是特例，滤波器可逼近任意连续函数且保持排列等变性

## 局限与展望

1. **谱分解开销**：需要离线特征分解 $O(N^3)$，虽然可用稀疏近似但对超大图仍有瓶颈
2. **大规模可扩展性**：Penn94 (约 42K 节点) 是实验中最大的图，未在百万级图上验证
3. **归纳学习场景缺失**：特征分解依赖完整图结构，归纳式 (inductive) 场景中新节点加入需重新分解
4. **超参数敏感性**：$K$（最高阶）和 $M$（傅里叶分量数）需要调优，文中未充分讨论其交互影响
5. **图分类实验数据集偏小**：仅用了 TU 数据集，未在 OGB 等大规模基准上评估

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 KAN 的傅里叶参数化引入图 Transformer 谱滤波器，双维自适应是实质贡献
- 实验充分度: ⭐⭐⭐⭐ — 11 节点分类 + 5 图分类 + 滤波器拟合实验 + 消融，但缺大规模图
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、理论推导完整、图表质量高
- 价值: ⭐⭐⭐⭐ — 为谱图 Transformer 提供了新的滤波器设计范式，对异配图学习有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Graph Meta-Network for Learning on Kolmogorov–Arnold Networks](../../ICLR2026/graph_learning/a_graph_meta-network_for_learning_on_kolmogorovarnold_networks.md)
- [\[ICML 2025\] Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification](mitigating_over-squashing_in_graph_neural_networks_by_spectrum-preserving_sparsi.md)
- [\[NeurIPS 2025\] Relieving the Over-Aggregating Effect in Graph Transformers](../../NeurIPS2025/graph_learning/relieving_the_over-aggregating_effect_in_graph_transformers.md)
- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](../../NeurIPS2025/graph_learning/unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [\[ACL 2025\] Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors](../../ACL2025/graph_learning/fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)

</div>

<!-- RELATED:END -->
