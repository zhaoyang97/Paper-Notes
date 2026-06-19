---
title: >-
  [论文解读] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs
description: >-
  [NeurIPS 2025][计算生物][蛋白质表示学习] 提出SSHG（Secondary Structure-based Hierarchical Graph）框架，基于蛋白质二级结构motif构建两级层次化图表示（残基级内部图+motif级全局图），用两阶段GNN分别学习局部和全局特征，理论证明保持最大表达力的同时在酶分类和配体亲和力预测上同时提升精度和降低计算成本。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "蛋白质表示学习"
  - "图神经网络"
  - "多尺度"
  - "二级结构"
  - "层次化图"
---

# Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs

**会议**: NeurIPS 2025  
**arXiv**: [2602.00862](https://arxiv.org/abs/2602.00862)  
**代码**: 暂无  
**领域**: 计算生物  
**关键词**: 蛋白质表示学习, 图神经网络, 多尺度, 二级结构, 层次化图

## 一句话总结
提出SSHG（Secondary Structure-based Hierarchical Graph）框架，基于蛋白质二级结构motif构建两级层次化图表示（残基级内部图+motif级全局图），用两阶段GNN分别学习局部和全局特征，理论证明保持最大表达力的同时在酶分类和配体亲和力预测上同时提升精度和降低计算成本。

## 研究背景与动机

图神经网络已成为蛋白质结构学习的有力工具，能在残基级别捕捉空间关系。但现有GNN方法面临两个核心挑战：

**多尺度建模不足。** 蛋白质具有天然的层次化组织：一级结构（氨基酸序列）→二级结构（α-螺旋、β-折叠等motif）→三级/四级结构。现有残基级GNN无法有效捕捉二级结构层面的关键特征。一个经典反例是朊蛋白：正常形态PrPC和病态形态PrPSc具有完全相同的一级序列，但二级结构截然不同——正常形态富含α-螺旋，而病态形态转变为β-折叠富集结构，导致致命的神经退行性疾病。纯残基级GNN无法区分这两种状态。

**长程依赖建模效率低。** 为捕获全局上下文，现有方法通常使用大半径截断（如16Å），这会生成极其稠密的图（边数可达~15K），带来巨大的计算和内存开销。多尺度方法如HoloProt使用表面建模，但计算成本同样高昂。

本文的核心idea是利用领域知识——蛋白质二级结构——作为天然的多尺度层次化节点：每个二级结构motif(如一段α-螺旋)就是一个高级节点。这种设计同时利用了生物学先验知识和几何信息，能以极少的边数（总边数<3N，其中N为残基数）实现高效的多尺度蛋白质建模，并具有理论保证的最大表达力。

## 方法详解

### 整体框架
SSHG是一个模块化的两阶段GNN框架：
1. 用DSSP算法将蛋白质序列划分为二级结构motif（α-螺旋、β-链、环等）
2. 构建两级层次化图：内部结构图（每个motif内的残基级图）+ 结构间图（motif间的全局图）
3. 第一阶段GNN在每个motif内学习局部特征，第二阶段GNN在motif间学习全局特征
可灵活选择任意GNN作为backbone（如GVP-GNN、ProNet、Mamba等）

### 关键设计

1. **DSSP二级结构分割与层次化图构建**: 

    - 功能：将蛋白质序列基于二级结构分割为motif子序列，构建两级图
    - 核心思路：DSSP算法为每个残基分配二级结构类型token（H=α-螺旋, E=β-链, T=转角等9种），连续相同token的残基被归为同一子序列 $S_i$。对每个 $S_i$ 构建内部结构图 $\mathcal{G}_i$（残基为节点，用SCHull方法基于α-碳坐标确定边）；对所有 $S_i$ 的几何中心构建结构间图 $\mathcal{G}$（motif为节点），边特征包含encoding相对朝向的local frame乘积 $g_i^\top g_j$
    - 设计动机：二级结构是蛋白质中经过验证的功能单元，用它作层次化节点既有生物学支撑又能大幅减少边数。SCHull图保证几何完备性和稀疏性。

2. **两阶段GNN消息传递**: 

    - 功能：先在motif内学局部交互，再在motif间学全局关系
    - 第一阶段：对每个内部图 $\mathcal{G}_i$ 独立做 $T_1$ 轮消息传递，最后通过readout得到motif嵌入 $\mathbf{s}_i = \text{readout}_1(\{\!\!\{\mathbf{f}_k^{(T_1)} | k \in \mathcal{V}(\mathcal{G}_i)\}\!\!\})$
    - 第二阶段：在结构间图 $\mathcal{G}$ 上以 $\mathbf{s}_i$ 为初始节点特征做 $T_2$ 轮消息传递，输出全局特征 $\mathbf{s}_{\text{global}} = \text{readout}_2(\{\!\!\{\mathbf{s}_i^{(T_2)} | i \in \mathcal{V}(\mathcal{G})\}\!\!\})$
    - 设计动机：两阶段设计使得第一阶段GNN可以用很少层数处理小图（每个motif通常只有几十个残基），而第二阶段可以用一层就捕获长程依赖（因为motif图的节点数远少于残基图）

3. **理论保证：最大表达力定理（Theorem 4.2）**: 

    - 在UPD、AGG、readout满足单射性假设下，SSHG两阶段GNN能区分任意在刚体运动下不等价的蛋白质结构
    - 稀疏性保证（Proposition 3.2）：层次化图的总边数 $|\mathcal{E}| + \sum_i |\mathcal{E}_i| < 3N$，其中 $N$ 为残基数
    - 设计动机：证明层次化设计不会丢失关键结构信息——这是整个框架的理论基石

### 损失函数 / 训练策略
根据下游任务选择：酶分类任务使用交叉熵损失，配体亲和力预测使用MSE损失。数据增强包括对坐标添加高斯噪声(std=0.1)、各向异性缩放(0.9~1.1)、随机mask氨基酸类型(概率0.1~0.2)。

## 实验关键数据

### 主实验

**酶反应分类（EC）**

| 方法 | 测试准确率(%) | 训练时间(s/epoch) | 参数量 |
|------|-------------|-------------------|--------|
| GCN | 66.5 | 186 | - |
| GCN+SSHG | **71.2** | **150** | - |
| GVP-GNN | 68.5 | 334 | 1.0M |
| GVP-GNN+SSHG | **73.6** | **236** | 1.0M |
| IEConv | 87.2 | - | 9.8M |
| ProNet-Backbone | 86.4 | 210 | 1.3M |
| ProNet+SSHG | **87.2** | **140** | 1.3M |
| Mamba+SSHG | **88.4** | **157** | 1.5M |

**配体结合亲和力预测（LBA）**

| 方法 | RMSE↓ | Pearson↑ | Spearman↑ | 训练时间(s/epoch) |
|------|-------|----------|-----------|-------------------|
| HoloProt-Full | 1.464 | 0.509 | 0.500 | 45 |
| ProNet-Backbone | 1.458 | 0.546 | 0.550 | 32 |
| ProNet+SSHG | **1.435** | **0.579** | **0.591** | **24** |
| Mamba+SSHG | **1.399** | **0.614** | **0.610** | 29 |

### 消融实验

**图构建策略效率对比（ProNet作为backbone）**

| 配置 | 平均边数 | 训练时间(s/epoch) | 内存(MiB) | 准确率(%) |
|------|----------|-------------------|-----------|----------|
| cutoff=4 | 1,034 | 138 | 1,290 | 78.1 |
| cutoff=10 | 11,316 | 210 | 14,548 | 86.4 |
| cutoff=16 | 14,881 | 247 | 17,768 | 87.0 |
| **+SSHG** | **1,593** | **140** | **1,818** | **87.2** |

**两阶段GNN参数分配**

| 第一阶段参数 | 第二阶段参数 | 训练时间 | 准确率(%) |
|-------------|-------------|----------|----------|
| 0.69M (均分) | 0.69M | 140 | 87.2 |
| 1.03M (偏重局部) | 0.34M | 136 | **87.4** |
| 0.34M (偏重全局) | 1.03M | 142 | 87.1 |

### 关键发现
- SSHG使得每个对比backbone都获得了准确率提升+训练加速的双重收益，这是极为难得的"双赢"
- 在边数仅1,593（vs cutoff=16时14,881）的情况下，SSHG达到了更高准确率，证实层次化稀疏图的有效性
- 内存使用从17,768 MiB降至1,818 MiB（降低90%），对大规模蛋白质的实际应用至关重要
- 将更多参数分配给第一阶段（局部motif建模）比分配给第二阶段略好，说明精细的局部表示更重要

## 亮点与洞察
- 框架设计思路非常优雅：利用蛋白质二级结构这一已有的生物学知识作为自然的层次化分割，不需要任何学习即可完成层次化图构建。这种"领域知识驱动的图构建"思路值得在其他结构化数据上推广。
- 理论与实验的完美结合：既证明了最大表达力和稀疏性的理论保证，又在实验中展示了一致的精度提升和效率改善。Proposition 3.2的边数上界 $< 3N$ 使得复杂度分析简洁而有力。

## 局限与展望
- 目前仅验证了酶分类和配体亲和力两个任务，蛋白质折叠分类、蛋白质-蛋白质交互等任务有待验证
- readout函数使用mean pooling而非理论要求的单射操作，更强的聚合方案可能进一步提升
- 未与预训练蛋白质语言模型（如ESM-2）结合，这是一个有前景的方向
- 二级结构的定义依赖DSSP算法的准确性，对预测结构（如AlphaFold输出）的鲁棒性有待验证

## 相关工作与启发
- **vs HoloProt**: HoloProt用大半径图+表面建模实现多尺度，训练时间300 s/epoch vs SSHG的157 s/epoch，准确率HoloProt 78.9% vs Mamba+SSHG 88.4%——SSHG大幅领先
- **vs IEConv**: IEConv通过层次化池化获取多尺度特征但需要多层反复池化，参数量9.8M vs SSHG的1.3M，SSHG在参数量1/7的情况下达到相当准确率
- **vs ProNet**: 作为SSHG的backbone之一，ProNet+SSHG在保持相同参数量的情况下训练时间降低33%，准确率持平或提升

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 利用二级结构构建层次化图的想法直觉自然又理论扎实，是蛋白质GNN领域的重要创新
- 实验充分度: ⭐⭐⭐⭐ 两个benchmark任务、多个backbone、效率分析完整，但任务覆盖面可更广
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，图示清晰，从生物学动机到方法设计到理论保证的逻辑链非常完整
- 价值: ⭐⭐⭐⭐⭐ 提供了一个通用的即插即用框架，任何GNN都能从中受益，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)
- [\[NeurIPS 2025\] Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning](generative_distribution_embeddings_lifting_autoencoders_to_the_space_of_distribu.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](../../ICML2025/computational_biology/geometric_generative_modeling_with_noise-conditioned_graph_networks.md)

</div>

<!-- RELATED:END -->
