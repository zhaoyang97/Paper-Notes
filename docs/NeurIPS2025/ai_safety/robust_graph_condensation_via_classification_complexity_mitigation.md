---
title: >-
  [论文解读] Robust Graph Condensation via Classification Complexity Mitigation
description: >-
  [NeurIPS 2025 Spotlight][AI安全][图凝缩] 本文揭示图凝缩（GC）本质上是降低分类复杂度的过程，而对抗攻击恰好破坏了这一特性，据此提出MRGC框架，通过内在维度正则化、曲率感知流形平滑和类间流形解耦三个流形约束模块来增强GC的鲁棒性，首次在结构、特征和标签均可能被篡改的条件下系统研究GC鲁棒性。
tags:
  - "NeurIPS 2025 Spotlight"
  - "AI安全"
  - "图凝缩"
  - "对抗鲁棒性"
  - "分类复杂度"
  - "流形学习"
  - "图神经网络"
---

# Robust Graph Condensation via Classification Complexity Mitigation

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.26451](https://arxiv.org/abs/2510.26451)  
**代码**: [GitHub](https://github.com/RingBDStack/MRGC)  
**领域**: AI安全  
**关键词**: 图凝缩, 对抗鲁棒性, 分类复杂度, 流形学习, 图神经网络

## 一句话总结

本文揭示图凝缩（GC）本质上是降低分类复杂度的过程，而对抗攻击恰好破坏了这一特性，据此提出MRGC框架，通过内在维度正则化、曲率感知流形平滑和类间流形解耦三个流形约束模块来增强GC的鲁棒性，首次在结构、特征和标签均可能被篡改的条件下系统研究GC鲁棒性。

## 研究背景与动机

图凝缩（GC）通过将大图压缩为小图来提升GNN的训练效率，已在神经架构搜索和图持续学习等场景中得到应用。然而，现有GC方法**假设原始图是干净的**。当原始图被对抗攻击篡改时，凝缩图的质量大幅下降。

作者通过系统调研提出三个关键问题并逐一回答：

**Q1: 现有鲁棒图学习技术能否改善GC鲁棒性？** 答案是否定的。作者将鲁棒GNN（如MedianGCN）集成为GC骨干或应用于凝缩图上训练，发现两种策略都不如标准GC方法。原因是大多数鲁棒GNN依赖注意力机制，而这类机制在GC场景中表现不佳。

**Q2: 攻击破坏了GC的什么关键特性？** 通过分类复杂度理论的视角，作者发现GC本质上是一个**降低分类复杂度**的过程——凝缩后三个复杂度指标（内在维度、Fisher判别率、超球覆盖比例）平均下降89.25%。但对抗攻击使这些指标在凝缩图中平均上升547.54%，即攻击**逆转了GC的复杂度降低特性**。

**Q3: 如何设计通用防御策略？** 需要从三个互补方向保护GC的分类复杂度降低能力：降低内在维度、简化类边界复杂度、消除类间模糊性。

## 方法详解

### 整体框架

MRGC是一个即插即用的框架，在现有GC方法（如GCond）的基础上添加三个流形学习正则化模块。总损失为：
$$\mathcal{L} = \mathcal{L}_{GC} + \alpha\mathcal{L}_{dim} + \beta\mathcal{L}_{cur} + \gamma\mathcal{L}_{sep}$$

### 关键设计

1. **内在维度流形正则化（$\mathcal{L}_{dim}$）**: 首先通过Theorem 1证明GC是内在维度降低过程：$\mathrm{ID}(\mathcal{G}') < \mathrm{ID}(\mathcal{G})$；Theorem 2进一步证明攻击会增加凝缩图的内在维度：$\mathrm{ID}(\mathcal{G}') < \mathrm{ID}(\mathcal{G}'_*)$。

   为约束内在维度，使用节点表示 $\mathbf{Z}' = (\mathbf{A}')^2\mathbf{X}'$（两轮消息传播后的嵌入），通过Laplacian近似求解流形维度：
    $\mathcal{L}_{dim} = \sqrt{\det(\mathbf{\Sigma}_{Z'} + \mathbf{I})} \cdot \sum_{i=1}^{d'} S_{\mathbf{Z}'}(\alpha_i)$
   其中第一项通过协方差矩阵的行列式估计流形体积，第二项通过图Laplacian算子估计坐标函数的梯度积分。整个过程是可微的，可加入端到端训练。

2. **曲率感知流形平滑（$\mathcal{L}_{cur}$）**: 类边界的复杂度由类流形的曲率决定。对每个节点 $i$，通过在其局部邻域拟合二次超曲面 $f_\Theta(\mathbf{o}) = \mathbf{o}^\top\Theta\mathbf{o}$ 来估计高斯曲率 $K(i) = 2\det(\mathrm{Mat}(\mathbf{Q}^{-1}\mathbf{p}))$。具体步骤：估计法向量 $\mathbf{u}_i$（最小特征值对应的特征向量）→ 将邻居投影到切空间 → 拟合超曲面获得曲率。

   为区分不同节点的重要性，使用Ollivier Ricci曲率 $\kappa(i,j) = 1 - \mathcal{W}(m_i^\alpha, m_j^\alpha)/\mathcal{D}(i,j)$ 对边界桥接节点赋予更高权重：
    $\mathcal{L}_{cur} = \sum_c \sum_{i \in \mathcal{V}_c} \mathrm{Norm}(-\kappa(i)) \cdot |K(i)|$

3. **类间流形解耦（$\mathcal{L}_{sep}$）**: 通过最小化各类流形体积之和与整体数据流形体积之差来消除类间重叠：
    $\mathcal{L}_{sep} = \left(\sum_c |\mathcal{M}(\mathcal{G}'_c)| - |\mathcal{M}(\mathcal{G}')|\right)^2$
   当各类流形完全不重叠时，体积之和等于总体积，$\mathcal{L}_{sep}=0$。流形体积仍通过协方差矩阵行列式估计。

### 复杂度分析

- 内在维度模块：$\mathcal{O}(n'd' + (d')^3)$
- 曲率计算：$\mathcal{O}(n'((d')^6 + k))$（高斯曲率）+ $\mathcal{O}((n')^2)$（Ricci曲率）
- 类间解耦：$\mathcal{O}(c(d')^3)$
- 实际中 $n'$ 很小，且使用PCA降维使 $d'$ 可控

## 实验关键数据

### 主实验——不同凝缩比下的鲁棒性（攻击：S.1%/F.10%/L.20%）

| 数据集 | 比例 | GCond | SGDD | SFGC | GEOM | RobGC | MRGC |
|--------|------|-------|------|------|------|-------|------|
| Cora | 1.30% | 70.69 | 68.28 | 39.70 | 40.43 | 71.60 | **77.43** |
| Cora | 2.60% | 71.39 | 68.48 | 50.10 | 54.00 | 72.19 | **76.72** |
| CiteSeer | 0.90% | 60.77 | 47.13 | 36.23 | 36.70 | 59.88 | **65.12** |
| CiteSeer | 1.80% | 61.03 | 56.12 | 48.69 | 47.17 | 61.77 | **63.87** |
| PubMed | 0.08% | 70.81 | 48.75 | 47.26 | 50.17 | — | **74.85** |

### 不同攻击预算下的鲁棒性（Cora，比例1.30%）

| 攻击类型 | GCond | SGDD | RobGC | MRGC | 提升 |
|---------|-------|------|-------|------|------|
| 结构 S.5% | 60.59 | 59.16 | 62.03 | **65.57** | +3.54 |
| 结构 S.10% | 59.09 | 53.68 | 61.24 | **64.02** | +2.78 |
| 特征 F.20% | 69.36 | 68.54 | 71.13 | **74.79** | +3.66 |
| 特征 F.30% | 65.22 | 58.34 | 69.15 | **72.36** | +3.21 |
| 标签 L.30% | 64.97 | 68.02 | 65.99 | **70.47** | +4.48 |
| 标签 L.40% | 57.49 | 61.97 | 57.81 | **63.89** | +6.08 |

### 分类复杂度指标变化

| 指标 | GC后变化 | 攻击后凝缩图变化 |
|------|---------|----------------|
| 内在维度 | ↓89.25%（平均） | ↑547.54%（平均） |
| Fisher判别率 | 明显下降 | 大幅上升 |
| 超球覆盖比例 | 明显下降 | 大幅上升 |

### 关键发现

- MRGC在Cora、CiteSeer、PubMed、DBLP四个数据集上的几乎所有凝缩比和攻击配置下均取得最佳性能
- 在Ogbn-arxiv上，轨迹匹配方法（SFGC/GEOM）因在干净数据集上基础性能更优而保持优势，这是骨干方法差异而非鲁棒性差异
- 图去噪预处理（Jaccard/SVD/KNN）效果有限，因为它们假设特征和标签是干净的
- RobGC仅对结构攻击有效，在特征和标签攻击下表现不佳
- MRGC在标签攻击30%/40%预算下提升最为显著（+4.48/+6.08），说明类间流形解耦模块在应对标签攻击时特别有效

## 亮点与洞察

- **理论视角独特**: 将分类复杂度理论引入GC鲁棒性研究，通过三个复杂度因子（内在维度、边界复杂度、类模糊性）建立了清晰的分析框架
- **两个定理的关系构建巧妙**: Theorem 1说明GC天然降低复杂度，Theorem 2说明攻击逆转这一过程，由此确定了防御的目标
- **即插即用设计**: 作为正则化模块可与大多数GC方法兼容，不限于GCond

## 局限与展望

- 高斯曲率计算的复杂度为 $\mathcal{O}(n'(d')^6)$，虽然 $n'$ 小且用PCA降维，但在大规模图上可能仍有开销
- 仅使用GCond作为骨干的实验，与轨迹匹配方法（SFGC/GEOM）的结合效果未知
- 三个超参数 $\alpha, \beta, \gamma$ 需要网格搜索（1e-3到1e2），调参成本较高
- 防御假设了已知攻击存在，在未知攻击类型时的自适应性有待验证

## 相关工作与启发

- 与RobGC相比，MRGC是首个能同时防御结构、特征和标签三种攻击的GC鲁棒化方法
- 分类复杂度理论（Ho & Basu, 2002）在图学习中的应用是新颖的跨领域迁移
- 流形正则化的思路可推广到其他数据蒸馏场景（如数据集蒸馏、模型压缩）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 分类复杂度视角和流形约束设计都是全新的研究路线
- **实验充分度**: ⭐⭐⭐⭐ 五个数据集、多种攻击类型/预算、多种凝缩比、与10种基线对比
- **写作质量**: ⭐⭐⭐⭐ Q&A驱动的论文结构引人入胜，理论和实验衔接紧密
- **价值**: ⭐⭐⭐⭐ 对GC在对抗环境下的实际部署有重要意义，推动了鲁棒GC的研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[CVPR 2025\] Lyapunov Stable Graph Neural Flow](../../CVPR2025/ai_safety/lyapunov_stable_graph_neural_flow.md)
- [\[NeurIPS 2025\] DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning](design_encrypted_gnn_inference_via_server-side_input_graph_pruning.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[NeurIPS 2025\] Improved Balanced Classification with Theoretically Grounded Loss Functions](improved_balanced_classification_with_theoretically_grounded_loss_functions.md)

</div>

<!-- RELATED:END -->
