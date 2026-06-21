---
title: >-
  [论文解读] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?
description: >-
  [ICLR 2026][图学习][oversmoothing] 指出广泛使用的Dirichlet energy指标无法在实际场景中正确捕获GNN过平滑现象，提出以特征表征的数值秩/有效秩（effective rank）作为替代度量。在深度2–24、各深度独立训练的设定下，Erank与准确率的平均相关性达0.91（且方向一致为正），而Dirichlet energy平均仅−0.72、相关方向在数据集间反复横跳（在大规模OGB-Arxiv等场景下尤其失效）；并从理论上证明对线性及非负权重的非线性GNN族其特征矩阵数值秩收敛到1（秩坍塌），从而把过平滑重新定义为秩坍塌而非特征向量对齐。
tags:
  - "ICLR 2026"
  - "图学习"
  - "oversmoothing"
  - "图神经网络"
  - "Dirichlet energy"
  - "numerical rank"
  - "effective rank"
  - "rank collapse"
---

# Are We Measuring Oversmoothing in Graph Neural Networks Correctly?

**会议**: ICLR 2026  
**arXiv**: [2502.04591](https://arxiv.org/abs/2502.04591)  
**领域**: 图神经网络 / 理论分析  
**关键词**: oversmoothing, graph neural networks, Dirichlet energy, numerical rank, effective rank, rank collapse, GNN depth  

## 一句话总结

指出广泛使用的Dirichlet energy指标无法在实际场景中正确捕获GNN过平滑现象，提出以特征表征的数值秩/有效秩（effective rank）作为替代度量。在深度2–24、各深度独立训练的设定下，Erank与准确率的平均相关性达0.91（且方向一致为正），而Dirichlet energy平均仅−0.72、相关方向在数据集间反复横跳（在大规模OGB-Arxiv等场景下尤其失效）；并从理论上证明对线性及非负权重的非线性GNN族其特征矩阵数值秩收敛到1（秩坍塌），从而把过平滑重新定义为秩坍塌而非特征向量对齐。

## 研究背景与动机

**过平滑是GNN的核心瓶颈**：随着层数加深，GNN节点表征趋于一致，丧失区分性——这被认为是GNN无法像其他深度网络一样从深度中获益的主要原因。

**Dirichlet energy的主导地位**：当前文献几乎统一使用Dirichlet energy（衡量相邻节点表征差异的平方和）作为过平滑的度量标准，基于此设计了大量缓解方法。

**度量与性能的脱节**：实践中发现Dirichlet energy的变化方向有时与模型性能变化不一致——Dirichlet energy降低不一定意味着性能下降，反之亦然。

**理论理解的不足**：现有过平滑理论主要分析Dirichlet energy在特定图拓扑下的收敛行为，但未质疑该度量本身的合理性。

**异构图上的失效**：在异构图（heterophilic graph）中相邻节点标签往往不同，Dirichlet energy 高未必是坏事，反而可能意味着模型学到了有区分性的表征——这让「能量越低越过平滑」的朴素解读彻底失灵。

**需要更好的度量**：GNN社区需要一个与下游任务性能更一致的过平滑度量，以正确指导架构设计和正则化策略。

## 方法详解

### 整体框架

本文不提新模型，而是从线性代数视角重新定义过平滑：把它看作节点特征矩阵的秩坍塌（rank collapse）——层数加深时特征矩阵的有效秩趋近 1，所有节点表征被压进同一个低维子空间。围绕这个定义，论文用秩度量（数值秩 / 有效秩）替换主流的 Dirichlet energy，先在多架构多数据集上证明它与性能更吻合，再从理论上证明在线性及非负权重的非线性 GNN 族里秩坍塌是必然结果。

### 关键设计

**1. 用秩的连续松弛（数值秩 / 有效秩）替代 Dirichlet energy：直接刻画表征空间的维度塌缩**

Dirichlet energy 测的是相邻节点表征差的平方和，本质上是衡量特征对消息传递矩阵 $A$ 主特征子空间的偏离——它低只能说明特征在向主特征向量对齐，看不出整个表征空间是否真的塌缩；而且它只有精确收敛到 0（特征完全对齐、或矩阵整体趋零）时才真正等价于过平滑，现实中「降了两个数量级但远未到 0」根本无法判读。本文改用矩阵秩的连续松弛来度量。秩本身是离散的奇异值非零计数、不可微，于是用三种松弛逼近：给定奇异值 $\sigma_1>\sigma_2>\dots$、记 $p_k = \sigma_k / \sum_i \sigma_i$，

$$\text{StabRank}(X)=\frac{\|X\|_*^2}{\|X\|_F^2},\quad \text{NumRank}(X)=\frac{\|X\|_F^2}{\|X\|_2^2},\quad \text{Erank}(X)=\exp\!\left(-\sum_k p_k\log p_k\right).$$

其中有效秩 Erank 是奇异值谱归一化熵的指数，直观地数出特征矩阵里「有意义的维度个数」：所有节点表征坍塌到一条方向时趋于 1（彻底过平滑），谱越平铺、表征越多样则越高。相比 Dirichlet energy，秩度量有三个本质优势：(a) 尺度不变，特征整体趋零或爆炸都仍有意义；(b) 不依赖预先给定的固定特征子空间，能捕捉向**任意**低维子空间的收敛（Dirichlet 必须先知道主特征向量、且要求各层不变）；(c) 不需要精确到秩 1，就能在浅层网络里诊断出低秩，即提前预警过平滑。

**2. 在「按深度独立训练」的真实设定下做相关性诊断：揭穿 Dirichlet energy 的失效**

一个合格的过平滑度量，应当与下游性能强相关、方向一致。本文先指出过往证据多是在很深的**未训练 / 欠训练**网络上、沿单个网络的层逐层去看 Dirichlet 衰减，而那种衰减只是小初始化导致的幅度收缩，并不反映训练后模型的过平滑。本文换用更真实的设定：对每个深度 $l=2,\dots,24$ 各自独立训练一个 GNN，再算各度量（取对数）与节点分类准确率的相关系数（Erank、NumRank 先减 1，使其也在过平滑时趋零，便于同向比较）。结果是 Erank 平均相关性约 0.91、方向一致为正；Dirichlet energy 平均仅约 $-0.72$，方向与 Erank 相反，而且在不同数据集间反复横跳（如 Cora-GCN 上为 $-0.79$、OGB-Arxiv-GCN 上却翻成 $+0.77$），在异构图（Squirrel / Chameleon / Amazon-Ratings）和大规模图上尤其不可靠。这说明 Dirichlet energy 已不能再当过平滑的可信代理指标。

**3. 秩坍塌的理论收敛证明：把「深层必过平滑」钉成定理（在可证明的架构族内）**

为给「秩坍塌 = 过平滑」这个新定义打地基，本文证明：对线性 GCN（Thm 5.1），以及权重逐元素非负、且激活函数与消息传递矩阵共享主特征向量的非线性 GNN（Thm 5.3，借助非线性 Perron–Frobenius 理论），当层数 $l\to\infty$ 时特征矩阵的数值秩收敛到 1。直觉是反复作用归一化邻接矩阵 $A$ 会让特征沿主特征向量 $u$ 方向增长最快、其余方向被相对压低，于是非对齐分量 $\|(I-\mathcal{P})X\|_F / \|X\|_2 \to 0$（$\mathcal{P}=uu^\top$），数值秩 $\to 1$。关键在于这个收敛**与权重幅度无关**——即便特征不趋零、Dirichlet 类度量「看不出动静」，秩仍会坍塌，这也解释了为何单纯缩放 / 归一化类技巧无法根治过平滑。

> ⚠️ 该定理限定在线性 GNN 或「非负权重 + 共享特征向量」的非线性 GNN，并非对任意 GNN 都成立（一般非线性情形的反例与讨论见原文附录 D）。

## 实验关键数据

### 主实验：度量与准确率的相关性（深度 2–24，各深度独立训练）

相关系数衡量准确率与各度量对数值的相关性，期望方向为正（度量越低越过平滑、准确率越低）。Erank / NumRank 已减 1 以便同向比较。下表取 GCN 行的代表性数据集，平均行为全部 7 数据集 × {GCN, GAT} 共 14 组。

| 数据集（GCN） | $E_{\mathrm{Dir}}$ | MAD | Erank | NumRank |
|--------|------|------|-------|---------|
| Cora | −0.79 | −0.25 | **0.97** | 0.59 |
| Citeseer | −0.84 | −0.72 | **0.97** | 0.68 |
| Pubmed | −0.91 | 0.62 | **0.95** | 0.93 |
| Squirrel（异构） | −0.78 | −0.82 | 0.63 | **0.96** |
| Chameleon（异构） | −0.92 | −0.88 | **0.94** | 0.90 |
| Amazon-Ratings（异构） | −0.93 | 0.92 | **0.93** | 0.80 |
| OGB-Arxiv | **+0.77**（方向反了） | 0.28 | **0.97** | 0.91 |
| **平均（7 数据集 × GCN/GAT）** | **−0.72** | 0.16 | **0.91** | 0.84 |

### 深度退化：2→24 层的准确率保留比

保留比 = 24 层准确率 / 2 层准确率，越低说明加深掉点越狠（即过平滑越严重）。

| 数据集（GCN） | 准确率保留比（24 层 / 2 层） |
|--------|------|
| OGB-Arxiv | 0.09（退化最严重） |
| Cora | 0.27 |
| Citeseer | 0.44 |
| Pubmed | 0.52 |
| Chameleon（异构） | 0.62 |
| Squirrel（异构） | 0.85 |
| Amazon-Ratings（异构） | 0.86 |

### 关键发现

1. **Dirichlet energy 方向不一致、平均还反了**：其与准确率的相关平均为 $-0.72$，与期望（正向，如 Erank）相反，且符号在数据集间横跳（Cora-GCN $-0.79$ vs OGB-Arxiv-GCN $+0.77$），在异构图与大规模图上尤其失效。
2. **Erank 一致性强**：在几乎所有数据集与架构上都保持高正相关，平均 0.91；NumRank 次之（平均 0.84），两者都远胜 Dirichlet energy 与 MAD（平均仅 0.16）。
3. **过往证据设定有偏**：以往多在未训练 / 欠训练的极深网络上沿层观察 Dirichlet 衰减，那只是小初始化的幅度收缩；改为各深度独立训练后，Dirichlet 与性能脱钩，而秩度量仍紧跟性能。
4. **退化与图同构性相关**：同构图（OGB-Arxiv / Cora）加深后准确率保留比低至 0.09–0.27，部分异构图（Squirrel / Amazon-Ratings ≈0.85）反而更抗深。
5. **度量选择影响研究结论**：用 Dirichlet energy 当判据，可能让研究者误判某架构「没有过平滑」，从而导向错误的改进方向。

## 亮点与洞察

1. **挑战基础假设**：质疑了GNN社区广泛接受的Dirichlet energy度量，属于"皇帝的新衣"式的重要贡献。
2. **替代方案优越**：Erank在理论和实证上均显著优于Dirichlet energy，且实现简单。
3. **概念重定义**：将过平滑从"相邻节点特征趋同"（特征向量对齐）重新定义为"特征空间秩坍塌"，更加本质。
4. **实际指导价值**：正确的度量将引导社区开发更有效的过平滑缓解策略。
5. **大规模数据集的关键性**：在小数据集上两种度量差异不大，但在大规模真实图上差异巨大。

## 局限与展望

1. **因果关系未完全建立**：高相关性不等于因果性，秩坍塌是否是性能下降的唯一原因尚待商榷。
2. **理论与实证的覆盖边界**：收敛定理只覆盖线性 GNN 与「非负权重 + 共享特征向量」的非线性 GNN，一般非线性架构是否必然秩坍塌仍是开放问题；实证虽含异构图，但 Erank 在异构图上偶有偏低（如 Squirrel-GCN 仅 0.63），其行为机理未深入展开。
3. **缓解策略未提出**：论文聚焦于度量诊断，未基于新的秩视角提出具体的缓解算法。
4. **图级任务未涉及**：所有实验针对节点分类，图级分类任务中过平滑的表现未探讨。

## 相关工作与启发

- **过平滑理论**：Li et al. (2018) 首次形式定义过平滑；Cai & Wang (2020) 的Dirichlet energy框架
- **过平滑缓解**：DropEdge (Rong et al., 2020), PairNorm (Zhao & Akoglu, 2020), DeeperGCN (Li et al., 2020)
- **有效秩**：Roy & Vetterli (2007) 提出的effective rank定义
- **深层GNN**：GCNII (Chen et al., 2020), RevGNN (Li et al., 2021) 等深层架构设计

## 评分

- 新颖性: ⭐⭐⭐⭐ 挑战基础度量假设，提出秩坍塌视角
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多架构、理论+实证的全面验证
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑清晰，实验设计精当
- 价值: ⭐⭐⭐⭐⭐ 对GNN社区的基础度量进行了必要的修正

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](../../ICML2025/graph_learning/on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[ICLR 2026\] Learning from Historical Activations in Graph Neural Networks](learning_from_historical_activations_in_graph_neural_networks.md)
- [\[ICLR 2026\] WATS: Wavelet-Aware Temperature Scaling for Reliable Graph Neural Networks](wats_wavelet-aware_temperature_scaling_for_reliable_graph_neural_networks.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](cooperative_sheaf_neural_networks.md)
- [\[ICLR 2026\] LogicXGNN: Grounded Logical Rules for Explaining Graph Neural Networks](logicxgnn_grounded_logical_rules_for_explaining_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
