---
title: >-
  [论文解读] Break the Tie: Learning Cluster-Customized Category Relationships for Categorical Data Clustering
description: >-
  [AAAI 2026][categorical data clustering] 提出 DISC 方法，为每个聚类簇学习定制化的属性类别关系（而非全局统一距离），通过关系树建模与聚类联合优化，在 12 个数据集上以平均排名 1.25 大幅超越现有最佳方法（5.21）。 类别数据无固有距离：数值数据有天然的欧氏距离…
tags:
  - "AAAI 2026"
  - "categorical data clustering"
  - "distance metric learning"
  - "subspace clustering"
  - "category relationship"
  - "minimum spanning tree"
---

# Break the Tie: Learning Cluster-Customized Category Relationships for Categorical Data Clustering

**会议**: AAAI 2026  
**arXiv**: [2511.09049](https://arxiv.org/abs/2511.09049)  
**代码**: [GitHub - ZHAO-Mingjie/SCOF](https://github.com/ZHAO-Mingjie/SCOF)  
**领域**: LLM评测  
**关键词**: categorical data clustering, distance metric learning, subspace clustering, category relationship, minimum spanning tree  

## 一句话总结

提出 DISC 方法，为每个聚类簇学习定制化的属性类别关系（而非全局统一距离），通过关系树建模与聚类联合优化，在 12 个数据集上以平均排名 1.25 大幅超越现有最佳方法（5.21）。

## 研究背景与动机

**类别数据无固有距离**：数值数据有天然的欧氏距离，但类别属性（如"职业"中的"律师"与"司机"）缺乏明确定义的距离关系，这是类别数据聚类的根本挑战。

**现有距离度量与任务无关**：基于 Hamming 距离、信息熵、统计特征的传统方法无法根据不同聚类任务自适应调整，表达能力受限。

**已有学习方法仅学全局统一关系**：距离学习方法（相似度学习、核空间映射、图学习）虽能联合优化距离与聚类，但对所有簇使用统一的类别关系，忽略了子空间特异性。

**子空间聚类仅调权重不调类别关系**：现有子空间方法只学习属性重要性权重，属性值之间的关系仍然是固定的，无法捕捉同一属性在不同簇中的差异。

**同一属性在不同簇中应有不同关系**：以肺炎数据为例，"程序员"和"建筑工人"在 COVID-19 簇中差距小（高传染性），但在尘肺病簇中差距很大（粉尘暴露差异），证明簇定制化关系的必要性。

**实验验证了定制关系的优势**：作者在 k-modes 上对比了随机定制化、统一、加权三种策略，发现 Customized 策略始终优于 Uniform 和 Weighted，且 Uniform 也优于单纯的属性加权，表明簇定制化类别关系是准确聚类的关键。

## 方法详解

### 整体框架 (DISC)

DISC（DIStance learning from Cluster）的核心思想是为每个聚类簇学习定制化的类别关系距离。整体流程为：

1. 初始化聚类划分 → 2. 为每个簇的每个属性建立全连接图 → 3. 推断最小生成树作为关系树 → 4. 基于关系树定义子空间距离 → 5. 联合优化距离与聚类划分 → 迭代直至收敛

目标函数为最小化簇内不相似度：$z(\mathbf{H}, M, \mathcal{T}) = \sum_{j=1}^{k} \sum_{i=1}^{n} h_{i,j} \cdot \sum_{r=1}^{l} \mathbf{D}_{j,r}(u,s)$

### 关键设计 1：基于条件概率分布的全连接图建模

对每个簇 $C_j$ 的每个属性 $\mathbf{a}_r$，构建全连接图 $G_{j,r}$，节点为属性的所有可能取值。边的权重定义为两个取值在该簇内条件概率分布的差异：

$$\mathbf{W}_{j,r}(u,s) = |p(v_r^u | C_j) - p(v_r^s | C_j)|$$

条件概率分布能反映取值在簇内的分布模式，分布相似的取值之间边权更小，表示更相似。

### 关键设计 2：最小生成树推断关系树

全连接图存在多路径表示不确定性的问题。为此，从全连接图中提取最小生成树（MST）$\mathcal{T}_{j,r}$ 作为关系树，确保任意两个取值间只有唯一路径，距离定义确定且唯一。

关键定理：推断出的关系树定义了确定性的、与欧氏距离兼容的距离度量。这意味着学到的类别距离可以自然地与数值属性的欧氏距离结合，支持混合数据集聚类。

### 关键设计 3：广义属性加权机制

关系树隐式捕获了属性重要性：若某属性在簇内取值分布差异大（条件概率差大），则其距离值也大，贡献更高。这等价于将传统的属性加权分解为更细粒度的取值级距离学习，表达能力更强。

### 损失函数与训练策略

采用三步交替优化：

1. **固定 $M$ 和 $\mathcal{T}$，更新 $\mathbf{H}$**：将每个样本分配到距离最近的聚类中心
2. **固定 $\mathcal{T}$ 和 $\mathbf{H}$，更新 $M$**：按 k-modes 方式计算各簇各属性的众数作为中心
3. **步骤 1-2 收敛后，固定 $\mathbf{H}$ 和 $M$，重新推断 $\mathcal{T}$**：根据当前划分重新计算条件概率分布并构建关系树

三步循环迭代直至划分矩阵 $\mathbf{H}$ 不再变化。收敛性有理论保证（Theorem 3），时间复杂度为 $O(nlk\mathcal{I}E)$，对样本数、属性数和簇数均为线性。

## 实验关键数据

### 表 1：聚类性能对比（ACC 指标，12 个 UCI 数据集，11 种方法）

| 数据集 | KMD | CoForest | HARR | SigDT | **DISC** |
|--------|------|----------|------|-------|----------|
| CA | 0.380 | 0.402 | 0.368 | 0.534 | **0.583** |
| DT | 0.597 | 0.676 | 0.711 | 0.844 | **0.859** |
| AV | 0.625 | 0.667 | 0.644 | 0.638 | **0.726** |
| OB | 0.340 | 0.383 | 0.373 | 0.381 | **0.425** |
| BM | 0.640 | 0.609 | 0.617 | 0.249 | **0.729** |
| ZO | 0.697 | 0.724 | 0.726 | 0.723 | **0.805** |
| **平均排名** | 8.38 | 5.42 | 5.21 | 6.83 | **1.25** |

DISC 在 ACC 上平均排名 1.25，远超次优方法 HARR 的 5.21；ARI 平均排名 1.42，CMP 平均排名 1.58。

### 表 2：混合数据集上的扩展实验（ACC）

| 数据集 | KPT | WOCIL | HARR | DISC (仅类别) | **DISC (mixed)** |
|--------|------|-------|------|---------------|------------------|
| CC | 0.398 | 0.386 | 0.430 | 0.443 | **0.449** |
| AP | 0.516 | 0.572 | 0.576 | 0.641 | **0.656** |
| DT | 0.531 | 0.690 | 0.615 | 0.859 | **0.862** |
| AV | 0.539 | 0.673 | 0.681 | 0.726 | **0.776** |
| BM | 0.587 | 0.572 | 0.522 | 0.729 | **0.742** |

验证了学到的类别距离与欧氏距离的兼容性，混合数据集上进一步提升。

## 亮点

- **新颖的问题视角**：首次明确提出为不同聚类簇学习定制化的类别关系，而非统一距离，极大增强了聚类算法的拟合能力
- **优雅的数学建模**：通过条件概率分布 → 全连接图 → 最小生成树的层级建模，将模糊的类别关系转化为确定性的、可度量的距离
- **严格的理论保证**：证明了关系树距离的确定性、欧氏兼容性、收敛性，不是启发式方法
- **实验效果突出**：ACC 平均排名 1.25 vs 次优 5.21，显著性检验通过，12 个数据集全面领先
- **自然支持混合数据**：欧氏兼容性使得方法可无缝扩展到数值+类别混合数据集

## 局限性

- **仅在 UCI 小规模数据集上实验**：最大数据集 BM 仅 45K 样本，缺乏大规模（百万级）数据集验证
- **依赖真实簇数 $k^*$**：所有实验均使用真实标签给定的簇数，未讨论 $k$ 未知时的自动确定策略
- **基于 k-modes 框架**：方法绑定在 k-modes 的硬划分框架上，未探索软聚类或更灵活的聚类范式
- **类别取值数多时 MST 计算开销增大**：尽管整体线性复杂度，但单个属性取值数 $o_r$ 很大时全连接图和 MST 构建可能成为瓶颈
- **缺少与深度聚类方法的对比**：未与基于深度学习的类别数据聚类方法比较

## 相关工作

- **类别数据距离度量**：Hamming 距离（Hamming 1950）、信息熵方法（Lin 1998; Zhang & Cheung 2022b）、统计方法（Ahmad & Dey 2007）——均为任务无关，无法自适应聚类
- **距离学习**：概率分布相似度（Cheung & Jia 2013）、核空间映射（Zhu et al. 2022）、图学习（Zhang & Cheung 2022a, 2023; Zhao et al. 2024/CoForest）——学全局统一关系
- **子空间聚类**：属性加权方法（Bai et al. 2011; Cao et al. 2013; Jia & Cheung 2017/WOCIL）——仅调权重不调类别关系
- **SOTA 对比方法**：HARR（Zhang et al. 2025b）、SigDT（Hu et al. 2025c）、MCDC（Cai et al. 2024）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 簇定制化类别关系是一个清晰且有价值的新视角
- 实验充分度: ⭐⭐⭐⭐ — 12 个数据集、11 种对比方法、消融实验、混合数据扩展、收敛与效率分析齐全，但缺少大规模和深度方法对比
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，理论推导完整，图表直观
- 价值: ⭐⭐⭐⭐ — 对类别数据聚类领域有明确推进，理论与实践兼备

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](approximation_algorithm_for_constrained_k-center_clustering_.md)

</div>

<!-- RELATED:END -->
