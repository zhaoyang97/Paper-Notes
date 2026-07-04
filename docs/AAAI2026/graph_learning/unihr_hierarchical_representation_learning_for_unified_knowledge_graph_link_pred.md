---
title: >-
  [论文解读] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction
description: >-
  [AAAI2026 Oral][图学习][知识图谱] 提出UniHR框架，通过Hierarchical Data Representation (HiDR)将超关系/时序/嵌套等多类KG统一转换为三元组形式，并设计Hierarchical Structure Learning (HiSL)模块在事实内部和事实间进行两阶段消息传递，在9个数据集5种KG类型上取得最优或竞争性的link prediction结果。
tags:
  - "AAAI2026 Oral"
  - "图学习"
  - "知识图谱"
  - "Link Prediction"
  - "超关系KG"
  - "时序KG"
  - "嵌套KG"
  - "统一表示学习"
  - "层次化消息传递"
---

# UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction

**会议**: AAAI2026 Oral  
**arXiv**: [2411.07019](https://arxiv.org/abs/2411.07019)  
**作者**: Zhiqiang Liu, Yin Hua, Mingyang Chen, Yichi Zhang, Zhuo Chen, Lei Liang, Wen Zhang (ZJU)  
**代码**: [zjukg/UniHR](https://github.com/zjukg/UniHR)  
**领域**: 图学习  
**关键词**: 知识图谱, Link Prediction, 超关系KG, 时序KG, 嵌套KG, 统一表示学习, 层次化消息传递  

## 一句话总结

提出UniHR框架，通过Hierarchical Data Representation (HiDR)将超关系/时序/嵌套等多类KG统一转换为三元组形式，并设计Hierarchical Structure Learning (HiSL)模块在事实内部和事实间进行两阶段消息传递，在9个数据集5种KG类型上取得最优或竞争性的link prediction结果。

## 背景与动机

### 真实KG的异构性

现实世界的大规模知识图谱（如Wikidata、DBpedia）不仅包含标准三元组 $(h, r, t)$，还包含更复杂的事实形式：超关系事实附带辅助key-value对，时序事实附带时间戳，嵌套事实表达事实之间的关系。例如"Oppenheimer在哈佛获得化学学士学位"无法用简单三元组表达，需要超关系形式 $((h,r,t), \{(k_i:v_i)\}_{i=1}^m)$。这些丰富的表示形式因其增强的语义表达能力受到广泛关注。

### 现有方法的两大局限

已有研究主要存在两个问题：(1) 大多只针对特定类型的KG（如HKG、TKG或NKG）设计方法，难以泛化到包含多种事实类型的真实场景；(2) 由于beyond-triple表示的复杂性，难以实现可泛化的层次化建模（事实内部语义+事实间结构信息）。例如StarE为HKG定制了GNN但无法灵活捕获key-value信息，NestE只在事实内部评分而忽略全局结构，ECEformer只捕获TKG的事实间语义。虽然HAHE开始捕获HKG的层次语义，但其异构表示限制了向其他KG类型的扩展。

### 统一方法的价值

建立一个统一的层次化表示学习方法，能够同时处理多种事实类型并进行层次化语义建模，不仅具有理论价值，更是实际应用（如Wikidata本身就包含混合事实类型）的迫切需求。统一表示还为KG预训练模型、跨类型联合训练等方向铺平道路。

## 核心问题

如何设计一个统一框架，将超关系KG、时序KG和嵌套KG转换为统一表示，并在其上进行有效的层次化结构学习（事实内部+事实间），以实现跨KG类型的通用link prediction？

## 方法详解

### 整体框架

UniHR包含三个步骤：(1) HiDR模块将任意类型KG统一转换为三元组形式 $\mathcal{G}^{\text{HiDR}}$；(2) HiSL模块在 $\mathcal{G}^{\text{HiDR}}$ 上进行事实内和事实间两阶段消息传递增强节点嵌入；(3) 使用Transformer解码器进行link prediction。

### HiDR: Hierarchical Data Representation

HiDR将节点分为三类（atomic node $\mathcal{V}_a$、relation node $\mathcal{V}_r$、fact node $\mathcal{V}_f$），关系分为三类（atomic relation $\mathcal{R}_a$、nested relation $\mathcal{R}_n$、connected relation $\mathcal{R}_c = \{\text{has\_relation, has\_head\_entity, has\_tail\_entity}\}$），事实分为三类（atomic facts $\mathcal{F}_a$、connected facts $\mathcal{F}_c$、nested facts $\mathcal{F}_n$）。

- **HKG → HiDR**: 将主三元组 $(h,r,t)$ 拆为connected facts $(f, \text{has\_relation}, e_r)$、$(f, \text{has\_head}, h)$、$(f, \text{has\_tail}, t)$ 和atomic fact $(h,r,t)$，key-value对转为 $(f, k_i, v_i)$
- **NKG → HiDR**: 原子事实保持不变，嵌套事实 $(f_1, R, f_2)$ 转为nested facts
- **TKG → HiDR**: 先将时间戳转为辅助key-value对（begin:$\tau_b$, end:$\tau_e$），再按HKG方式转换

### HiSL: Hierarchical Structure Learning

**表示初始化**：atomic node嵌入 $\mathbf{H}_a \in \mathbb{R}^{|\mathcal{V}_a| \times d}$，relation node嵌入通过投影矩阵从relation edge生成：$\mathbf{H}_r = \mathbf{E}_a \cdot \mathbf{W}_r$。Fact node嵌入由主三元组初始化：
$$\mathbf{h}_f = f_m([\mathbf{h}_h; \mathbf{h}_r; \mathbf{h}_t])$$
其中 $f_m: \mathbb{R}^{3d} \rightarrow \mathbb{R}^d$ 为1层MLP。时间戳用Time2Vec编码。

**事实内消息传递（Intra-fact MP）**：对每个fact node $f_k$ 构建其一跳邻居子图，使用graph attention聚合局部语义信息：
$$\alpha_{i,j}^l = \frac{\exp(\mathbf{W}^l(\sigma(\mathbf{W}_{in}^l \mathbf{h}_i^l + \mathbf{W}_{out}^l \mathbf{h}_j^l)))}{\sum_{j' \in \mathcal{N}_i} \exp(\mathbf{W}^l(\sigma(\mathbf{W}_{in}^l \mathbf{h}_i^l + \mathbf{W}_{out}^l \mathbf{h}_{j'}^l)))}$$

**事实间消息传递（Inter-fact MP）**：在整个 $\mathcal{G}^{\text{HiDR}}$ 上进行消息传递，使用circular correlation算子 $\phi(\mathbf{h}_j, \mathbf{e}_r) = \mathbf{h}_j \star \mathbf{e}_r$ 结合边方向 $\lambda(r)$ 和类型 $\tau(r)$ 的细粒度聚合：
$$\mathbf{h}_i^{l+1} = \sum_{(r,j) \in \mathcal{N}(i)} \sigma(\omega_{\tau(r)}^l) \mathbf{W}_{\lambda(r)}^l \phi(\mathbf{h}_j^l, \mathbf{e}_r^l) + \mathbf{W}_{self}^l \mathbf{h}_i^l$$

**解码器**：使用Transformer对序列化的嵌入进行解码，用交叉熵损失训练。

## 实验关键数据

### HKG Link Prediction (WikiPeople, WD50K)

| 模型 | WikiPeople MRR | WikiPeople H@10 | WD50K MRR | WD50K H@10 |
|------|---------------|----------------|-----------|------------|
| StarE | 0.458 | 0.611 | 0.309 | 0.452 |
| GRAN | 0.477 | 0.596 | 0.329 | 0.465 |
| ShrinkE | 0.485 | 0.601 | 0.345 | 0.482 |
| HAHE | 0.498 | 0.610 | 0.343 | 0.484 |
| HyperSAT | 0.493 | 0.610 | 0.345 | 0.489 |
| **UniHR** | **0.496** | **0.619** | **0.348** | 0.482 |

### NKG Triple Prediction (FBH, FBHE, DBHE)

| 模型 | FBH MRR | FBH MR | FBHE MRR | DBHE MRR |
|------|---------|--------|----------|----------|
| BiVE | 0.855 | 6.20 | 0.711 | 0.687 |
| NestE | 0.922 | 3.34 | 0.851 | 0.862 |
| GRADATE | 0.780 | 18.15 | 0.603 | 0.654 |
| **UniHR** | **0.946** | **2.46** | 0.793 | **0.862** |

### TKG Link Prediction (wikidata12k)

| 模型 | MRR | H@1 | H@3 | H@10 |
|------|-----|-----|-----|------|
| TGeomE+ | 0.333 | 0.232 | 0.361 | 0.546 |
| 5EL | 0.311 | 0.237 | 0.355 | 0.546 |
| **UniHR** | **0.334** | **0.242** | **0.368** | 0.527 |

### Hyper-relational TKG (WIKI-hy, YAGO-hy)

| 模型 | WIKI-hy MRR | WIKI-hy H@10 | YAGO-hy MRR | YAGO-hy H@10 |
|------|------------|-------------|------------|-------------|
| HypeTKG | 0.687 | 0.789 | 0.832 | 0.857 |
| **UniHR** | **0.692** | **0.792** | **0.841** | **0.862** |

### 消融实验 (FBHE / DB15K / wikidata12k)

| 变体 | FBHE MRR | DB15K MRR | wikidata12k MRR |
|------|----------|-----------|----------------|
| w/o intra-fact MP | 0.754 | 0.341 | 0.321 |
| w/o inter-fact MP | 0.776 | 0.338 | 0.319 |
| UniHR (full) | **0.793** | **0.348** | **0.334** |

## 亮点

- **首个统一KG表示学习框架**：统一处理HKG、TKG、NKG三种KG类型，HiDR无信息损失地将多种事实形式转为三元组
- **参数高效的层次化学习**：HiSL的训练参数量不随图规模增长，relation node和fact node的嵌入从atomic元素派生，避免参数膨胀
- **超越统一方法本身的潜力验证**：展示了联合训练、组合型KG（超关系时序KG）、多任务学习等场景的性能提升，wikimix联合训练使MR在HKG和TKG上分别改进17.1%和39.7%
- **在5种KG类型9个数据集上达到SOTA或竞争性结果**，特别是在NKG的FBH数据集上triple prediction MRR达0.946，超越NestE的0.922

## 局限与展望

- **HiDR转换引入额外节点和边**：虽然作者声称存储开销小，但在超大规模KG上（如完整Wikidata）的可扩展性仍需验证
- **解码端依赖Transformer**：序列化嵌入后用Transformer解码，对于长序列（如含大量key-value对的超关系事实）可能存在效率瓶颈
- **TKG上H@10略低于TGeomE+**：说明纯粹通过图结构学习时间信息在某些指标上还不如专门的temporal embedding方法
- **未涉及大规模预训练**：虽然讨论了统一表示对预训练的潜力，但实验中未真正进行大规模跨KG预训练

## 与相关工作的对比

- **vs StarE**: StarE为HKG定制GNN但无法灵活捕获key-value信息，UniHR通过HiDR+HiSL在WD50K上MRR提升12.6%
- **vs NestE**: NestE仅在事实内部评分，UniHR通过inter-fact MP补充全局结构信息，FBH上triple prediction MRR从0.922提升至0.946
- **vs HAHE**: HAHE虽实现HKG层次建模但设计不可泛化，UniHR在WikiPeople上H@10超越HAHE的0.610达到0.619
- **vs HypeTKG**: 针对组合型超关系时序KG，UniHR无需复杂模块堆叠即在WIKI-hy和YAGO-hy上超越专门模型

## 启发与关联

- 统一表示的思路可迁移到其他异构数据建模任务，如多模态知识图谱
- 层次化消息传递（intra-fact + inter-fact）的设计模式对其他需要同时建模局部和全局信息的图学习任务有启发
- 联合训练不同KG类型的实验验证了"数据多样性有益于表示学习"的直觉

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次提出统一框架处理三种beyond-triple KG，HiDR设计简洁优雅
- 实验充分度: ⭐⭐⭐⭐⭐ — 9个数据集5种KG类型+消融+联合训练+效率分析，覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，HiDR转换规则定义严谨
- 价值: ⭐⭐⭐⭐ — 填补了统一KG表示学习的空白，联合训练实验展示了实际应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](../../ICML2026/graph_learning/unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](../../ICML2026/graph_learning/t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](../../ICML2026/graph_learning/generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](../../NeurIPS2025/graph_learning/tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)

</div>

<!-- RELATED:END -->
