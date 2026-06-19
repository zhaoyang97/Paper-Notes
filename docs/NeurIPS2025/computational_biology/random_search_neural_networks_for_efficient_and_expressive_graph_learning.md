---
title: >-
  [论文解读] Random Search Neural Networks for Efficient and Expressive Graph Learning
description: >-
  [NeurIPS 2025][计算生物][图神经网络] 提出随机搜索神经网络（RSNN），用随机深度优先搜索（DFS）替代随机游走来采样图结构，在稀疏图上仅需$O(\log|V|)$次搜索即可实现完整边覆盖，配合通用序列模型可达到通用逼近能力，在分子和蛋白质基准上以最多16倍更少的采样量持续超越RWNN。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "图神经网络"
  - "随机游走"
  - "深度优先搜索"
  - "图表示学习"
  - "通用逼近"
  - "同构不变性"
---

# Random Search Neural Networks for Efficient and Expressive Graph Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.22520](https://arxiv.org/abs/2510.22520)  
**作者**: Michael Ito, Danai Koutra, Jenna Wiens (University of Michigan)
**代码**: [MLD3/RandomSearchNNs](https://github.com/MLD3/RandomSearchNNs)  
**领域**: 计算生物  
**关键词**: 图神经网络, 随机游走, 深度优先搜索, 图表示学习, 通用逼近, 同构不变性

## 一句话总结

提出随机搜索神经网络（RSNN），用随机深度优先搜索（DFS）替代随机游走来采样图结构，在稀疏图上仅需$O(\log|V|)$次搜索即可实现完整边覆盖，配合通用序列模型可达到通用逼近能力，在分子和蛋白质基准上以最多16倍更少的采样量持续超越RWNN。

## 研究背景与动机

### 问题背景
随机游走神经网络（RWNN）是新兴的图学习范式，通过将图表示为随机游走序列并用序列模型处理，克服了消息传递神经网络（MPNN）和图Transformer的局限性。然而，RWNN在实际采样约束下面临严重的表达力瓶颈——随机游走的节点覆盖时间可达$O(|V||E|)$，即使采用非回溯游走或最小度局部规则也仍为$O(|V|^2)$，导致在中小型图上也无法保证完整覆盖。

### 已有工作的不足
- **覆盖不完整**：随机游走在有限步数下容易遗漏关键结构（如环上的侧链），导致图重建不完整
- **表达力受限**：本文首次证明部分覆盖下的RWNN严格弱于MPNN，这意味着即使序列模型很强大，信息缺失也会根本性地限制模型能力
- **采样代价高**：RWNN需要$O(|V|)$条长度为$O(|V|)$的游走才能接近完整覆盖，计算开销大
- 现有改进（非回溯游走CRaWl、MDLR游走）虽然降低了覆盖时间，但仍保持二次复杂度

### 核心动机
设计一种新的图采样策略，能高效保证完整的节点和边覆盖，从而在实际采样预算下实现最大理论表达力。关键洞察：DFS产生的子图是生成树，天然保证节点全覆盖，将问题简化为跨多棵树的边覆盖。

## 方法详解

### RWNN表达力理论分析

**定理3.1（RWNN-MPNN等价性）**：当RWNN拥有长度达到边覆盖时间$C_E(G)$的完整游走多重集时，在注入性序列模型和聚合函数下，RWNN与MPNN表达力等价：$f_{\text{RWNN}}^{FC} \simeq f_{\text{MPNN}}$。

**推论3.2（部分覆盖下的RWNN）**：当RWNN仅获得部分节点/边覆盖时，其表达力严格弱于MPNN：$f_{\text{RWNN}}^{PC} \prec f_{\text{MPNN}}$。这一结果通过引入Walk Weisfeiler-Lehman（WWL）颜色细化算法建立了RWNN与WL层级的桥梁。

### RSNN架构设计

RSNN的四个核心组件：
1. **采样策略**：随机深度优先搜索（DFS），而非随机游走。均匀采样DFS序列$S \sim \mathbb{U}(\mathcal{S}_{\text{DFS}}(G))$
2. **记录函数**：$f_{\text{emb}}[i] = h_V(w_i) + \text{proj}(h_{\text{PE}}[i])$，包含位置编码以标记序列中的不连续点（回溯点）
3. **序列模型**：$f_{\text{seq}}: \mathbb{R}^{\ell \times d} \to \mathbb{R}^{\ell \times d}$，支持GRU/LSTM/Transformer
4. **节点聚合**：跨多个搜索序列对同一节点的表示取平均

### 对数采样覆盖定理

**引理4.1**：对于满足$|E| \leq C|V|$且最大度有界的稀疏连通图，$m$次独立随机搜索的生成树并集包含所有边的概率至少为$1-\delta$，只需：

$$m \geq \frac{\ln(C|V|/\delta)}{\ln(d_{\max}/(d_{\max}-1))}$$

这是$O(\log|V|)$级别的，相比RWNN的$O(|V|)$条游走有指数级改进。

### 通用逼近与同构不变性

**定理4.2（通用逼近）**：在稀疏有界度图空间上，若搜索次数$m$满足引理4.1的条件，则RSNN配合通用序列模型可逼近任意连续图函数至任意精度$\epsilon$。

**定理4.3（概率同构不变性）**：RSNN的随机DFS过程满足概率不变性，即对所有同构$G \cong H$，$f_{\text{RSNN}}(G) \stackrel{d}{=} f_{\text{RSNN}}(H)$，期望预测器$\Phi(G) = \mathbb{E}[f_{\text{RSNN}}(G)]$是图同构不变函数。

**推论4.4**：SGD训练收敛到不变目标函数的最优解，即使每次前向传播仅采样$m=1$个搜索。

### 运行时复杂度
- RSNN：单次DFS在稀疏图上为$O(|V|)$，$m$次搜索为$O(m|V|)$
- RWNN：$m$条长度$\ell$的游走为$O(m\ell)$
- 当$\ell \ll |V|$时RWNN采样更快，但短游走无法捕获全局结构；RSNN以略高的采样代价换取完整覆盖和更强表达力

## 实验关键数据

### 实验1：分子和蛋白质基准

在4个小规模分子数据集（MoleculeNet）和4个蛋白质数据集（ProteinShake）上评估，所有方法使用相同的采样数$m$和游走长度$\ell=|V|$。

| 模型 | m | CLINTOX (AUC) | BBBP (AUC) | TOX21 (AUC) | SC Family (ACC) | EC Subclass (ACC) |
|------|---|--------------|------------|-------------|-----------------|-------------------|
| GCN | — | 62.4 | 73.9 | 67.5 | 3.9 | 31.2 |
| GIN | — | 59.7 | 75.3 | 66.9 | 10.4 | 37.2 |
| Fingerprint | — | 66.5 | 86.2 | 79.1 | — | — |
| CRAWL | 1 | 70.0 | 77.6 | 71.7 | 5.2 | 28.7 |
| **RSNN** | **1** | **88.1** | **87.5** | **79.8** | **13.9** | **36.8** |
| CRAWL | 16 | 89.1 | 87.0 | 80.9 | 15.5 | 48.7 |
| **RSNN** | **16** | **88.5** | **89.4** | **82.2** | **19.0** | **50.0** |

关键发现：RSNN在$m=1$时即超越所有RWNN变体在$m=16$时的分子基准性能；蛋白质图上RSNN在所有$m$下均保持显著优势，$m=16$时SC Family准确率19.0%远超CRAWL的15.5%。

### 实验2：大规模分子基准与稠密图

在OGB大规模数据集（16-24万图）和稠密脑图数据集上评估：

| 数据集 | 规模 | RWNN-mdlr (m=1) | CRAWL (m=1) | **RSNN (m=1)** |
|--------|------|-----------------|-------------|----------------|
| PCBA-1030 | 160K图 | 63.5 | 64.2 | **78.8** |
| PCBA-1458 | 195K图 | 76.2 | 77.0 | **87.0** |
| PCBA-4467 | 240K图 | 75.4 | 75.6 | **85.2** |

| 数据集 | Avg.V | Avg.E | CRAWL m=4 | **RSNN m=4** | CRAWL m=16 | **RSNN m=16** |
|--------|-------|-------|-----------|--------------|------------|---------------|
| NeuroGraph | 1000 | 7029 | 77.5 | **80.4** | 68.3 | **86.5** |

RSNN在大规模分子基准上$m=1$时AUC提升超过10个百分点；在稠密脑图上$m=16$时RSNN(86.5%)大幅超越CRAWL(68.3%)，证明RSNN不仅限于稀疏图。

## 亮点

- **理论贡献深刻**：首次严格证明部分覆盖下RWNN弱于MPNN，建立了Walk WL与经典WL的等价桥梁，统一了两类看似不同的图模型的表达力分析
- **$O(\log|V|)$覆盖效率**：利用DFS生成树天然全节点覆盖的性质，将边覆盖所需采样量从$O(|V|)$降至$O(\log|V|)$，是指数级改进
- **三位一体理论保证**：同时证明高效覆盖、通用逼近能力、概率同构不变性，理论体系完整
- **16倍采样效率**：实验中RSNN $m=1$即可达到RWNN $m=16$的性能，实际采样效率提升显著

## 局限与展望

- **稀疏图假设**：理论分析依赖$|E|=O(|V|)$和有界最大度假设，对稠密图（如社交网络、知识图谱）覆盖效率退化
- **DFS计算开销**：完整DFS遍历代价为$O(|V|)$，对超大规模图（数百万节点）不可行，截断搜索的理论分析缺失
- **序列长度限制**：DFS序列长度为$O(|V|)$（包含回溯步），大图上序列过长影响序列模型处理效率
- **领域适用性**：主要在分子和蛋白质上验证，未测试社交网络、推荐系统等其他重要图学习场景
- **训练稳定性**：仅展示了中位数结果，部分数据集的min-max跨度较大（如NeuroGraph的CRAWL），可能存在训练不稳定问题

## 与相关工作的对比

- **CRaWl (Tönshoff et al. 2023)**：使用非回溯游走+节点级聚合，是最强RWNN基线，但覆盖效率仍为$O(|V|^2)$，在低采样预算下显著弱于RSNN
- **RWNN-mdlr (Kim et al. 2025)**：最小度局部规则游走实现最优一阶游走覆盖$O(|V|^2)$，但二次复杂度仍不够高效
- **CRW (Chen et al. 2025)**：学习长程依赖的随机游走方法，使用非回溯+可学习游走策略
- **Wang & Cho (2024)**：Non-convolutional GNN，均匀随机游走+节点匿名化，表达力受限于部分覆盖
- **GIN (Xu et al. 2019)**：WL等价的MPNN，RSNN在理论上可达到且超越其表达力
- **Graph Transformer (Dwivedi & Bresson 2020)**：全图注意力机制，在小数据集上表现不稳定

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将DFS搜索引入图神经网络，实现从游走到搜索的范式转变
- 实验充分度: ⭐⭐⭐⭐ — 涵盖分子/蛋白质/大规模/稠密图多类设定，但缺少社交/知识图谱实验
- 写作质量: ⭐⭐⭐⭐⭐ — 理论-方法-实验三者衔接紧密，分析层层递进，图示清晰直观
- 价值: ⭐⭐⭐⭐ — 为图表示学习提供了高效且有理论保证的新范式，但稀疏图假设限制了应用范围

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[NeurIPS 2025\] Autoencoding Random Forests](autoencoding_random_forests.md)
- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](../../ICML2025/computational_biology/neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [\[CVPR 2026\] Hyperbolic Busemann Neural Networks](../../CVPR2026/computational_biology/hyperbolic_busemann_neural_networks.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](../../ICML2025/computational_biology/geometric_generative_modeling_with_noise-conditioned_graph_networks.md)

</div>

<!-- RELATED:END -->
