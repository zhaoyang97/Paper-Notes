---
title: >-
  [论文解读] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion
description: >-
  [NeurIPS 2025][不完整多视图聚类] 提出HSACC框架通过双层语义空间设计（低层互信息一致性+高层自适应加权融合）和协同优化的隐式缺失视图恢复，在五个基准数据集上显著超越现有不完整多视图聚类方法。 不完整多视图数据（某些样本在特定视图上完全缺失）在实际中广泛存在——传感器限制、遮挡、数据采集条件差异等都会导致视…
tags:
  - "NeurIPS 2025"
  - "不完整多视图聚类"
  - "层级语义对齐"
  - "动态加权融合"
  - "MMD"
  - "协同补全"
---

# Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion

**会议**: NeurIPS 2025  
**arXiv**: [2510.13887](https://arxiv.org/abs/2510.13887)  
**代码**: [GitHub](https://github.com/XiaojianDing/2025-NeurIPS-HSACC)  
**领域**: LLM评测  
**关键词**: 不完整多视图聚类, 层级语义对齐, 动态加权融合, MMD, 协同补全

## 一句话总结

提出HSACC框架通过双层语义空间设计（低层互信息一致性+高层自适应加权融合）和协同优化的隐式缺失视图恢复，在五个基准数据集上显著超越现有不完整多视图聚类方法。

## 研究背景与动机

不完整多视图数据（某些样本在特定视图上完全缺失）在实际中广泛存在——传感器限制、遮挡、数据采集条件差异等都会导致视图缺失。这破坏了跨视图关联，放大噪声干扰并引入偏差。

现有深度不完整多视图聚类（IMVC）方法的两个关键局限：

1. **融合策略不足**：依赖静态融合（如均匀加权）无法适应视图间的分布差异；动态融合方法缺乏层级语义分离，无法区分低层一致性对齐和高层语义融合，导致多粒度信息丢失
2. **两阶段流水线误差传播**：先补全后聚类导致补全阶段的误差传播到聚类阶段；虽有联合优化尝试，但忽视了不同视图在融合表示中的重要性差异，补全未能充分利用高质量视图信息

## 方法详解

### 整体框架

HSACC包含三个模块：

1. **视图重建**：视图特定自编码器提取并重建特征
2. **多视图表示学习**：低层互信息对齐 → 高层自适应加权融合 → 分布差异最小化
3. **数据恢复与聚类**：隐式缺失视图补全 → 联合优化

### 关键设计

1. **双层语义空间对齐**:
    - 做什么：在低层语义空间通过互信息最大化确保跨视图一致性，在高层语义空间进行自适应加权融合
    - 核心思路：低层——计算跨视图特征相似性矩阵 $P_{(m,n)} = \frac{1}{N}\sum_{i=1}^N z_{i,m}^1 \cdot z_{i,n}^2$，从联合和边缘分布计算互信息损失 $L_{MMI}$。高层——先拼接得到初始融合 $\mathbf{R}$，用MMD衡量各视图与 $\mathbf{R}$ 的分布差异，通过softmax分配权重 $W^v = \frac{\exp(-D(\mathbf{Z}^v, \mathbf{R}))}{\sum_v \exp(-D(\mathbf{Z}^v, \mathbf{R}))}$
    - 设计动机：低层对齐确保共享模式被捕获，高层加权融合根据各视图与融合表示的一致程度动态分配贡献

2. **协同补全机制**:
    - 做什么：通过MLP将已对齐的潜在表示投影到高维语义空间，隐式恢复缺失视图
    - 核心思路：$\mathbf{Q}^1 = f_\text{MLP1}(\mathbf{Z}^1)$ 预测视图2的潜在表示。推断一致性损失 $L_{INF} = \frac{1}{N}\sum_i \|\mathbf{z}_i^2 - \mathbf{q}_i^1\|_2^2 + \frac{1}{N}\sum_i \|\mathbf{z}_i^1 - \mathbf{q}_i^2\|_2^2$ 确保推断结果与真实潜在表示一致
    - 设计动机：直接在潜在空间补全而非输入空间，避免生成低质量的原始特征；联合优化使补全和聚类相互增强

3. **分布对齐损失**:
    - 做什么：用RKHS中的MMD最小化高层共享表示 $\mathbf{H}$ 与各视图 $\mathbf{Z}^v$ 的分布差异
    - 核心思路：$L_{MMD} = \sum_{v=1}^V \text{MMD}^2(\mathcal{P}_{\mathbf{Z}^v}, \mathcal{Q}_\mathbf{H})$，用核矩阵近似计算
    - 设计动机：促进全局表示和各视图的信息交互，增强跨视图表示的一致性和互补性

### 损失函数 / 训练策略

总损失：$L = \lambda_1 L_{REC} + \lambda_2 L_{INF} + \lambda_3 L_{MMI} + \lambda_4 L_{MMD}$

- $L_{REC}$：重建损失（MSE）
- $L_{INF}$：推断一致性损失（从第 $E_1$ 个epoch开始引入）
- $L_{MMI}$：跨视图互信息损失
- $L_{MMD}$：分布对齐损失

训练策略：前 $E_1$ epoch训练表示学习，之后引入补全和聚类联合优化。最终用k-means对拼接的完整视图表示进行聚类。

## 实验关键数据

### 主实验（表格）

五个数据集、不同缺失率下的ACC对比（部分结果）：

| 缺失率 | 数据集 | HSACC | 次优方法 | 提升 |
|--------|--------|-------|---------|------|
| 0.5 | Caltech101-20 | **最优** | DCP/DSIMVC | ACC +5.3%, ARI +8.57% |
| 0.3→0.7 | Noisy MNIST | 下降6.92% | ICMVC下降35.19% | **鲁棒性强5倍** |
| 0.3 | Hdigit | **最优** | — | — |
| 0.5 | LandUse_21 | **最优** | — | — |
| 0.5 | 100leaves | **最优** | — | — |

### 消融实验

- 去除层级对齐（$L_{MMI}$）：所有数据集性能显著下降
- 去除动态加权（固定均匀权重）：ACC下降2-5%
- 去除协同补全（$L_{INF}$）：高缺失率下退化明显
- 去除 $L_{MMD}$：跨视图融合效果变差

### 关键发现

- HSACC在所有5个数据集的所有缺失率下均为最优或接近最优
- 在高缺失率（0.7）下鲁棒性远优于对比方法——Noisy MNIST上ACC仅下降6.92%，而ICMVC下降35.19%
- 层级对齐和动态加权的组合贡献最大
- 超参数分析表明模型对 $\lambda_1 \sim \lambda_4$ 的变化不敏感

## 亮点与洞察

- **层级语义分离**：明确区分低层一致性对齐和高层语义融合，多粒度信息保留更完整
- **自适应视图权重**：基于分布亲和度动态分配，避免低质量视图主导融合
- **消除误差传播**：联合优化补全和聚类，补全过程利用判别性特征引导
- 在5个数据集上全面超越9个SOTA方法，统计显著

## 局限与展望

- 仅实验了两视图场景，多视图（>2）的泛化性需进一步验证
- MLP用于跨视图推断的非线性映射可能不足以处理复杂关系
- 线性核的MMD计算可能对复杂分布差异不够敏感
- 自编码器架构（全连接1024-1024-1024）固定，未探索不同架构的影响
- 缺少与大规模数据集（如ImageNet子集）的实验
- 图结构信息未被利用

## 相关工作与启发

- **COMP (2023)**：通过对比学习+双预测实现恢复和一致性学习，但缺少层级语义分离
- **DCP**：对比预测框架，本文在此基础上引入了动态权重和MMD对齐
- **DSIMVC**：双层优化框架动态补全，但未考虑视图权重差异
- 层级语义对齐思想可推广到其他多模态融合问题

## 评分

⭐⭐⭐⭐ — 方法设计系统且有效，在多个数据集上的优势显著且一致，消融实验充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](../../AAAI2026/others/deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](../../AAAI2026/others/hierarchical_semantic_alignment_for_image_clustering.md)
- [\[CVPR 2026\] Plug-and-Play Incomplete Multi-View Clustering via Janus-Faced Affinity Learning with Topology Harmonization](../../CVPR2026/others/plug-and-play_incomplete_multi-view_clustering_via_janus-faced_affinity_learning.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)

</div>

<!-- RELATED:END -->
