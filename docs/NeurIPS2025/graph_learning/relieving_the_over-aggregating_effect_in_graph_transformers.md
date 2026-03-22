# Relieving the Over-Aggregating Effect in Graph Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2510.21267](https://arxiv.org/abs/2510.21267)  
**代码**: https://github.com/sunjss/over-aggregating (有)  
**领域**: 图学习 / Graph Transformer  
**关键词**: Over-Aggregating, Graph Transformer, Attention Entropy, Wideformer, Linear Attention

## 一句话总结
发现了 Graph Transformer 中的 over-aggregating 现象——大量节点以近均匀注意力分数被聚合导致关键信息被稀释，提出 Wideformer 通过分割聚合+引导注意力来缓解，作为即插即用模块在 13 个数据集上一致提升骨干模型性能。

## 研究背景与动机

1. **领域现状**：Graph Transformer 通过全局注意力机制学习节点间的长距离依赖，克服了传统 GNN 的 over-smoothing 和 over-squashing 问题。为了处理大规模图，主要有 sparse attention 和 linear attention 两条路线。
2. **现有痛点**：Linear attention 方法（如 Performer、SGFormer、Polynormer）保持全局感受野但产生严重的信息稀释——当所有节点参与聚合时，注意力分数趋于均匀（高 attention entropy），目标节点无法区分哪些消息是重要的。
3. **核心矛盾**：全局注意力计算中，节点数越多，注意力分数越均匀（Theorem 3.1 证明 entropy 下界随 $n$ 单调递增），关键消息被稀释（over-aggregating）。而 sparse attention 虽然缓解了此问题，却缩小了感受野。
4. **本文要解决什么？**：在保持全局感受野的同时缓解 over-aggregating。
5. **切入角度**：不减少输入节点数，而是将聚合分成多个并行子过程（cluster-wise aggregation），增加输出维度来保留更多信息。
6. **核心 idea 一句话**：将全局注意力的 all-to-one 聚合拆分为 cluster-to-one 的多路并行聚合，再通过引导机制让目标节点聚焦信息量最大的子集。

## 方法详解

### 整体框架
Wideformer 是一个即插即用模块，包含两步：
1. **Dividing** (分割聚合)：将源节点分成 $m$ 个 cluster，每个 cluster 独立聚合
2. **Guiding** (引导注意力)：对 $m$ 个聚合结果排序加权，让目标节点聚焦最有信息量的 cluster

输入：标准的 Q, K, V 特征  
输出：每个目标节点的增强表示

### 关键设计

1. **Cluster Center Selection (基于 K-Means++ 的变体)**:
   - 做什么：在 query 空间中选择 $m$ 个代表性的 cluster center
   - 核心思路：用 Algorithm 1，初始选 query 特征和最大的节点为第一个 center，然后贪心选择与已有 center 最不相似的节点作为新 center
   - 设计动机：选出的 center 之间差异最大化，使 cluster 分割有意义

2. **Source Node Assignment**:
   - 做什么：将源节点分配到最相似的 cluster $\mathbf{k}_i = \arg\max_j [(\mathbf{KC}^\top)_{i,j}]$
   - 核心思路：用 key 和 center 的相似度做 hard assignment
   - 设计动机：高相似度节点聚在一起，每个 cluster 内的节点语义更一致，聚合更有区分度

3. **Cluster-wise Aggregation + Guiding**:
   - 做什么：每个 cluster 独立做 attention 聚合，得到 $m$ 个输出向量；按 cluster 与目标节点的注意力分数排序加权
   - 核心思路：聚合输入量从 $n$ 减到 $n/m$，attention entropy 自然降低；再对 $m$ 个输出做二级注意力，保留全局信息
   - 设计动机：小输入量聚合 + 二级排序 = 信息保留与区分度兼得

### 损失函数 / 训练策略
- 直接嵌入到骨干模型的训练流程中，不引入额外损失
- 复杂度依然是 $O(n)$（线性）
- 与 GraphGPS、SGFormer、Polynormer 三种骨干兼容

## 实验关键数据

### 主实验
| 数据集类型 | 骨干 | 原始 | + Wideformer | 提升 |
|-----------|------|------|-------------|------|
| Cora | Polynormer | 86.03 | **86.23** | +0.20 |
| Citeseer | Polynormer | 77.96 | **78.19** | +0.23 |
| Amazon Photo | Polynormer | 95.47 | **95.52** | +0.05 |
| Minesweeper | Polynormer | 97.13 | **97.20** | +0.07 |

### 消融实验
| 配置 | 关键发现 |
|------|---------|
| 直接 entropy 正则化 | 有效但需显式计算 $O(n^2)$ 注意力矩阵，不可扩展 |
| Only Dividing | 降低了 entropy，验证了分割的有效性 |
| Only Guiding | 排序加权改善了信息聚焦 |
| Dividing + Guiding (Wideformer) | 两者配合效果最优 |

### 关键发现
- Over-aggregating 是 linear attention graph transformer 的普遍问题，attention entropy 随节点数增加而增加（见 Theorem 3.1）
- 梯度分析（Eq. 3-4）解释了为什么 over-aggregating 难以通过训练自行缓解：小注意力分数导致弱梯度信号
- Wideformer 在 13 个数据集上一致降低了 attention entropy 并提升了分类性能

## 亮点与洞察
- **发现新现象 (Over-Aggregating)**：与 over-smoothing（GNN 层间）和 over-squashing（瓶颈边）不同，over-aggregating 发生在单步全局聚合中，是 graph transformer 特有的问题。
- **理论+梯度双重分析**：不仅用 Theorem 3.1 证明了 entropy 下界随 $n$ 单调增，还通过梯度分析解释了为什么模型训练无法自行缓解。
- **即插即用设计**：Wideformer 不改变骨干架构，直接作为模块插入，实用性强。
- **与 over-smoothing/over-squashing 的区分论述**很清晰，帮助社区理解图上的不同信息损失机制。

## 局限性 / 可改进方向
- **提升幅度相对温和**：在一些数据集上提升不到 1%，说明 over-aggregating 不是所有场景的主要瓶颈
- **Cluster 数量 $m$ 的选择**：需要调参，且对不同数据集的最优值可能不同
- **Hard assignment**：当前用 argmax 做 hard cluster 分配，可能不够灵活
- **可改进**：探索 soft assignment；自适应调整 $m$；与 sparse attention 结合

## 相关工作与启发
- **vs Over-smoothing**: over-smoothing 是 GNN 层间的表示趋同，over-aggregating 是单步全局聚合中的信息稀释，两者正交
- **vs Over-squashing**: over-squashing 关注瓶颈边导致的信息压缩，over-aggregating 关注全局聚合中的无差别混合
- **vs Sparse Attention**: sparse attention 通过缩小感受野缓解问题，Wideformer 保持全局感受野但分割聚合

## 评分
- 新颖性: ⭐⭐⭐⭐ 发现新现象+理论分析扎实
- 实验充分度: ⭐⭐⭐⭐ 13 个数据集，3 个骨干
- 写作质量: ⭐⭐⭐⭐ 问题定义和分析链路清晰
- 价值: ⭐⭐⭐⭐ 实用的即插即用模块，但提升幅度有限
