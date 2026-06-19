---
title: >-
  [论文解读] ComRecGC: Global Graph Counterfactual Explainer through Common Recourse
description: >-
  [ICML 2025][计算生物][反事实解释] 本文首次形式化了图神经网络的**公共补救 (Common Recourse)** 全局反事实解释问题，证明该问题是 NP-hard 的，并提出了 ComRecGC 算法——通过多头顶点增强随机游走 (Multi-head VRRW) 寻找反事实图，再用 DBScan 聚类提取公共补救，在 NCI1、Mutagenicity、AIDS、Proteins 四个真实数据集上，覆盖率全面超越现有基线 10%–30%。
tags:
  - "ICML 2025"
  - "计算生物"
  - "反事实解释"
  - "图神经网络"
  - "公共补救 (Common Recourse)"
  - "全局解释"
  - "子模优化"
---

# ComRecGC: Global Graph Counterfactual Explainer through Common Recourse

**会议**: ICML 2025  
**arXiv**: [2505.07081](https://arxiv.org/abs/2505.07081)  
**代码**: [GitHub](https://github.com/ssggreg/COMRECGC)  
**领域**: 计算生物  
**关键词**: 反事实解释, 图神经网络, 公共补救 (Common Recourse), 全局解释, 子模优化

## 一句话总结

本文首次形式化了图神经网络的**公共补救 (Common Recourse)** 全局反事实解释问题，证明该问题是 NP-hard 的，并提出了 ComRecGC 算法——通过多头顶点增强随机游走 (Multi-head VRRW) 寻找反事实图，再用 DBScan 聚类提取公共补救，在 NCI1、Mutagenicity、AIDS、Proteins 四个真实数据集上，覆盖率全面超越现有基线 10%–30%。

## 研究背景与动机

图神经网络 (GNN) 已广泛应用于药物发现、分子属性预测、蛋白质功能分类等领域，但其预测是黑盒的。**反事实解释 (Counterfactual Explanation, CE)** 是一种重要的可解释性方法：给定一个被 GNN 分为"拒绝"类的图 $G$，找到一个结构相似但被分为"接受"类的图 $H$，从而揭示决策边界。

现有工作的主要局限包括：

**局部反事实解释** (如 CF-GNNExplainer、CFF、RCExplainer)：为每个输入图生成一个独立的反事实，解释数量多且难以在模型层面归纳理解。

**全局反事实解释** (如 GCFExplainer)：生成一小组反事实图覆盖所有输入图，但同一个反事实对不同输入图意味着截然不同的变换 (recourse)，不适用于提供可操作的指导。

**GLOBE-CE**：通过学习翻译方向生成全局解释，但尚未应用于图数据。

**关键观察**：如果一个变换（补救）能同时将多个"拒绝"图转为"接受"图，那么这种**公共补救 (Common Recourse)** 就能同时提供全局可解释性和实例级的可操作性。例如，在毒性预测中，"移除 NO₂ 基团"可能是一个适用于多种致突变分子的公共补救，具有很高的实用价值。

## 方法详解

### 整体框架

ComRecGC 分为三个阶段：

1. **图嵌入**：使用 GREED 算法将图映射到 $\mathbb{R}^l$（$l=64$），以近似归一化图编辑距离 (GED)
2. **多头顶点增强随机游走 (Multi-head VRRW)**：在图编辑空间中执行随机游走，搜索高质量反事实图
3. **聚类与贪心选择**：对所有补救向量用 DBScan 聚类，再贪心选取 $R$ 个公共补救以最大化覆盖率

整体算法伪代码（Algorithm 2）非常简洁：
```
ComRecGC(Φ, G, k, M, τ, R, n):
  S ← Multi-head VRRW(Φ, G, k, M, τ)       // 阶段1: 搜索反事实
  S ← top n frequently visited in S          // 筛选高频访问的反事实
  return CR_clustering(G, S, R)               // 阶段2: 聚类+贪心
```

### 关键设计

#### 1. 问题形式化

给定被 GNN $\Phi$ 分为"拒绝"类的输入图集合 $\mathbb{G}$，一个图 $H$ 是 $G$ 的**反事实**当且仅当 $\Phi(H)=1$ 且 $d(G,H) \leq \theta$（归一化 GED 距离不超过阈值）。

**补救 (Recourse)** 定义为嵌入空间中的向量 $\vec{r} = \overrightarrow{z(G)z(H)}$，即从输入图嵌入到反事实图嵌入的位移。两个补救 $r_1, r_2$ 构成**公共补救**，当存在中心向量 $\vec{v}$ 使得 $\|\vec{v} - \vec{r_i}\|_2 \leq \Delta$。

**Finding Common Recourse (FCR) 问题**：给定预算 $R$，找到大小为 $R$ 的公共补救集合 $\mathbb{F}$，最大化覆盖率：

$$\max_{\mathbb{F}} \text{coverage}(\mathbb{F}) \quad \text{s.t.} \quad |\mathbb{F}| \leq R$$

其中 coverage 定义为至少有一个公共补救能将其转化为反事实的输入图比例。

**理论结果**：
- **Theorem 1**: FCR 问题是 NP-hard 的（通过从最大覆盖问题归约证明）
- coverage 函数 $f$ 是**单调且子模的** → 贪心算法具有 $(1 - 1/e)$ 近似保证

#### 2. 图嵌入 (GREED 算法)

使用 GREED 算法学习图到 $\mathbb{R}^{64}$ 的嵌入，使得嵌入空间中的距离近似归一化图编辑距离：

$$\widehat{\text{GED}}(G_1, G_2) = \frac{\text{GED}(G_1, G_2)}{|V_1| + |V_2| + |E_1| + |E_2|}$$

这样可以统一比较不同大小图之间的距离。

#### 3. 多头顶点增强随机游走 (Multi-head VRRW)

这是 ComRecGC 最核心的创新，在图编辑空间中执行随机游走搜索反事实：

- **$k$ 个头**同时从不同输入图出发（实验中 $k=5$）
- 每步随机选一个**领头者 (lead)**，根据转移概率移动：

$$p(u_\ell, v) \propto p_\phi(v) \cdot C(v)$$

其中 $p_\phi(v)$ 是 GNN 预测 $v$ 为"接受"类的概率，$C(v)$ 是 $v$ 的访问次数加一（顶点增强机制，鼓励探索高频反事实区域）

- **非领头者**跟随领头者移动，选择使其补救向量最接近领头者补救向量的邻居：

$$\arg\min_{v \in \mathcal{N}(u_i) \cup \{u_i\}} \|\overrightarrow{z(G_\ell)z(u_\ell)} - \overrightarrow{z(G_i)z(v)}\|_2$$

这个设计关键在于让多个头**同步执行相似的补救**，从而天然促进公共补救的形成。

- **传送 (Teleportation)**：以概率 $\tau=0.05$ 重置所有头到输入图，传送概率偏向访问次数少的输入图：

$$p_\tau(G) = \frac{\exp(-t(G))}{\sum_{G' \in \mathbb{G}} \exp(-t(G'))}$$

其中 $t(G)$ 是从 $G$ 出发的游走次数，确保均匀探索。

- 执行 $M=50000$ 步后终止，选择高频访问的反事实作为候选。

#### 4. 聚类与贪心选择

对所有候选反事实产生的补救向量 $\{\overrightarrow{z(G)z(v)}\}$ 使用 DBScan 进行半径为 $\Delta$ 的聚类，每个聚类代表一个公共补救。然后贪心选择 $R=100$ 个公共补救，每步选择边际覆盖增益最大的：

```
CR_clustering(G, S, R):
  C ← Δ-clusterize {z(G)z(v)→ : v∈S, G∈G}
  F ← ∅
  for i = 1 to R:
    r ← argmax_{r∈C} gain(r; F)    // 贪心选最大边际增益
    F ← F ∪ {r}
  return F
```

### 损失函数 / 训练策略

ComRecGC 本身**不需训练**，它是一个后置解释方法。底层 GNN（GCN）的训练配置：

- 架构：3 层图卷积 + max pooling + 全连接层
- 优化器：Adam，学习率 0.001
- 训练 1000 epochs
- 数据划分：80%/10%/10%（训练/验证/测试）

**关键超参数**：
- $\theta = 0.1$（反事实距离阈值，Proteins 数据集为 0.15）
- $\Delta = 0.02$（公共补救相似度阈值）
- $k = 5$（随机游走头数）
- $\tau = 0.05$（传送概率）
- $M = 50000$（随机游走步数）
- $R = 100$（公共补救数量）

**复杂度分析**：$O(Mhk + n^2|\mathbb{G}|^2)$，其中 $h$ 是图编辑空间最大度数，$n$ 是筛选的高频反事实数量。

## 实验关键数据

### 主实验

**数据集统计**：

| 数据集 | 图数量 | 节点数 | 边数 | 节点标签数 | 分类任务 |
|--------|--------|--------|------|-----------|---------|
| NCI1 | 3978 | 118714 | 128663 | 10 | 抗癌活性 |
| Mutagenicity | 4308 | 130719 | 132707 | 10 | 致突变性 |
| AIDS | 1837 | 28905 | 29985 | 9 | 抗 HIV 活性 |
| Proteins | 1113 | 43471 | 81044 | 3 | 酶/非酶 |

**FCR 问题主结果** (Coverage ↑ / Cost ↓)：

| 方法 | NCI1 Cov. | NCI1 Cost | Mutag. Cov. | Mutag. Cost | AIDS Cov. | AIDS Cost | Prot. Cov. | Prot. Cost |
|------|-----------|-----------|-------------|-------------|-----------|-----------|------------|------------|
| GCFExplainer | 21.4% | 5.75 | 20.6% | 6.91 | 14.2% | 6.97 | 32.8% | 10.65 |
| ComRecGC 2-head | 40.5% | 5.12 | 45.9% | 5.74 | 32.8% | 6.71 | 45.9% | 11.44 |
| ComRecGC 3-head | 44.5% | 5.07 | **52.6%** | 5.61 | 33.6% | **6.62** | 45.9% | 11.51 |
| **ComRecGC 5-head** | **42.9%** | **4.95** | 51.8% | **5.63** | **34.7%** | 6.74 | **46.4%** | 11.59 |

ComRecGC 在四个数据集上覆盖率分别超越 GCFExplainer **+21.5%、+31.2%、+20.5%、+13.6%**。

**FC 问题结果**（限制反事实数量 $T=|\mathbb{G}|$）：

| 方法 | NCI1 | Mutagenicity | AIDS | Proteins |
|------|------|-------------|------|----------|
| Dataset CF | 8.52% | 10.4% | 0.41% | 29.0% |
| LocalRWExplainer | 19.0% | 18.2% | 12.9% | 22.1% |
| GCFExplainer | 14.7% | 11.9% | 14.2% | 29.8% |
| **ComRecGC** | **33.4%** | **46.7%** | **24.3%** | **39.6%** |

### 消融实验

| 配置 | NCI1 Cov. | Mutag. Cov. | AIDS Cov. | Prot. Cov. | 说明 |
|------|-----------|-------------|-----------|------------|------|
| ComRecGC (完整) | 42.9% | 51.8% | 34.7% | 46.4% | 全部组件 |
| w/o 聚类 | 10.1% | 8.2% | 13.6% | 33.9% | 不做 DBScan 聚类，覆盖率大幅下降 |
| w/o 多头 | 21.4% | 20.6% | 14.2% | 32.8% | 退化为单头，性能降至 GCFExplainer 水平 |

聚类对大规模数据集（NCI1、Mutagenicity）至关重要；多头机制是形成公共补救的核心。

### 关键发现

1. **公共补救 vs 全局反事实**（10 个解释的覆盖率）：ComRecGC 的公共补救解释在 Proteins 上覆盖率 18.0%，显著超过 GCFExplainer 的 10.9%，在其他数据集上可比或更优
2. **可解释性案例**：在 Mutagenicity 数据集上，ComRecGC 发现"移除 NO₂ 基团"是一个公共补救，将致突变分子转为非致突变分子——这与化学领域的已知知识一致
3. **GNN 架构泛化性**：在 GAT、GraphSAGE、GIN 上也始终优于 GCFExplainer
4. **运行时间**：Mutagenicity 上总耗时约 199 分钟（VRRW 172 分钟 + 聚类 27 分钟）

## 亮点与洞察

1. **问题定义的优雅性**：将"公共补救"形式化为嵌入空间中的向量聚类问题，巧妙避开了图结构空间中定义"相同变换"的困难，同时保留了可操作性
2. **多头 VRRW 的协同设计**：非领头者跟随领头者执行相似补救的机制，是天然促进公共补救形成的关键设计——这比先独立生成反事实再后处理聚类要高效得多
3. **理论与实践结合**：NP-hard 证明 + 子模性分析 + $(1-1/e)$ 近似保证，理论扎实；实验提升也显著
4. **实用价值**：在药物发现中，一个公共补救就像一个"通用药方"，告诉研究者"这类变换通常能让分子从有毒变为无毒"，直接指导分子设计

## 局限与展望

1. **物理可行性未考虑**：生成的反事实图可能不对应真实存在的分子或蛋白质，缺乏领域约束
2. **参数敏感性**：$\theta$ 和 $\Delta$ 的选择依赖经验，论文在 Proteins 上特别调整了 $\theta=0.15$，缺乏自动调参机制
3. **可扩展性瓶颈**：当 $\theta$ 增大时，补救数量爆炸式增长（Mutagenicity 上增长 100 倍），DBScan 变得不可行
4. **仅支持二分类**：当前框架限于"拒绝/接受"二分类，多类别场景需要扩展
5. **嵌入质量依赖**：整个方法建立在嵌入空间能良好近似 GED 的假设上，对嵌入算法的选择敏感
6. **运行开销较大**：50000 步随机游走耗时超过 2 小时，实际部署需要加速

## 相关工作与启发

- **GCFExplainer** (Huang et al., WSDM 2023)：全局反事实解释基线，但不考虑补救的一致性
- **GLOBE-CE** (Ley et al., ICML 2023)：学习翻译方向的全局解释，但未应用于图数据
- **CF-GNNExplainer** (Lucic et al., AISTATS 2022)：局部反事实解释，基于子图生成
- **GREED** (Ranjan et al., 2023)：图编辑距离的神经近似，提供嵌入基础
- **VRRW** (Pemantle, 2004)：顶点增强随机游走理论基础

**启发**：公共补救的思想可以迁移到其他领域——例如在医学影像中，找到"公共编辑操作"解释为什么一组影像被判为异常；在推荐系统中，找到"通用偏好变化"解释一类用户的推荐结果。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ - 首次提出并形式化公共补救问题，多头 VRRW 设计原创
- 实验充分度: ⭐⭐⭐⭐ - 4 个数据集、多种 GNN 架构、完整消融；但缺少大规模图数据
- 写作质量: ⭐⭐⭐⭐ - 逻辑清晰，理论与实验结合好；符号较多需适应
- 价值: ⭐⭐⭐⭐ - 对分子设计等领域有直接应用价值，但物理可行性是实际落地的障碍

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](global_context-aware_representation_learning_for_spatially_resolved_transcriptom.md)
- [\[ICML 2025\] PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models](polyconf_unlocking_polymer_conformation_generation_through_hierarchical_generati.md)
- [\[ICML 2025\] Supercharging Graph Transformers with Advective Diffusion](supercharging_graph_transformers_with_advective_diffusion.md)
- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)

</div>

<!-- RELATED:END -->
