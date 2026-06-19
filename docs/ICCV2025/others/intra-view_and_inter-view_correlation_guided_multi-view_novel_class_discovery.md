---
title: >-
  [论文解读] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery
description: >-
  [ICCV 2025][新类发现] 提出 IICMVNCD 框架，首次将新类发现（NCD）扩展到多视图设定，通过视图内矩阵分解捕捉已知/新类的分布一致性，以及视图间权重学习传递已知类的视图关系到新类，避免了对伪标签的依赖。 新类发现（NCD）：是一个重要的学习范式：给定一个有标注的已知类数据集和一个无标注的新类数据集（类别不…
tags:
  - "ICCV 2025"
  - "新类发现"
  - "多视图学习"
  - "矩阵分解"
  - "视图加权"
  - "聚类"
---

# Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery

**会议**: ICCV 2025  
**arXiv**: [2507.12029](https://arxiv.org/abs/2507.12029)  
**代码**: 无  
**领域**: 其他  
**关键词**: 新类发现, 多视图学习, 矩阵分解, 视图加权, 聚类

## 一句话总结

提出 IICMVNCD 框架，首次将新类发现（NCD）扩展到多视图设定，通过视图内矩阵分解捕捉已知/新类的分布一致性，以及视图间权重学习传递已知类的视图关系到新类，避免了对伪标签的依赖。

## 研究背景与动机

**新类发现（NCD）** 是一个重要的学习范式：给定一个有标注的已知类数据集和一个无标注的新类数据集（类别不重叠），目标是利用已知类的知识对新类进行聚类。这模拟了人类利用先验知识理解新概念的认知过程——例如了解手机和平板电脑的孩子，可以正确区分电脑和智能手表。

现有 NCD 方法存在两大局限：

**局限一：仅考虑单视图数据**。实际应用中多视图数据越来越普遍，如医学诊断中需要基因表达和影像等多组学特征协同判断。单视图数据可能无法提供足够信息，而现有 NCD 方法无法有效处理多视图场景下的信息融合。

**局限二：依赖伪标签**。现有 NCD 方法通常使用伪标签监督新类聚类，但伪标签质量受数据噪声、特征维度等因素影响，导致性能不稳定。在多视图场景下，伪标签生成更加困难。

**本文核心思路**：从视图内和视图间两个层面解决多视图 NCD：

**视图内**：利用已知类和新类之间数据分布的相似性，学习共享的特征基矩阵

**视图间**：利用已知类上的监督信号学习视图权重，迁移到新类的视图融合

## 方法详解

### 整体框架

IICMVNCD 是一个端到端的一阶段方法，包含三个核心组件：
1. 视图内信息提取（矩阵分解）
2. 视图间信息提取（视图加权融合 + 标签预测）
3. 联合优化目标（含类别分离约束）

### 关键设计

1. **视图内共享基矩阵分解（Intra-view）**:

    - 功能：为每个视图学习已知类和新类共享的基矩阵，提升特征表示质量
    - 核心思路：NCD 的核心假设是已知类和新类的数据分布相似。基于此假设，对每个视图 $v$ 的特征矩阵 $\mathbf{X}_v = [\mathbf{X}_v^l, \mathbf{X}_v^u]$（拼接已知类和新类数据）进行矩阵分解：
    $\min_{\mathbf{W}_v, \mathbf{Z}_v} \|\mathbf{X}_v - \mathbf{W}_v \mathbf{Z}_v\|_F^2 \quad \text{s.t.} \quad \mathbf{W}_v^\top \mathbf{W}_v = \mathbf{I}_k$
      其中 $\mathbf{W}_v \in \mathbb{R}^{d_v \times k}$ 是视图特有的共享基矩阵，$\mathbf{Z}_v \in \mathbb{R}^{k \times n}$ 是因子矩阵，$k = k_l + k_u$ 为总类别数
    - 设计动机：共享基矩阵 $\mathbf{W}_v$ 捕捉两个数据集间的分布一致性，正交约束防止冗余并稳定优化。因子矩阵 $\mathbf{Z}_v$ 编码样本间的关系，为后续标签预测提供基础

2. **视图间权重学习与标签预测（Inter-view）**:

    - 功能：利用已知类的监督信号学习最优视图权重，融合多视图信息生成一致的预测标签
    - 核心思路：引入可学习的视图权重 $\boldsymbol{\alpha}$，将因子矩阵进一步分解为视图特有中心矩阵 $\mathbf{A}_v$ 和一致预测标签 $\mathbf{Y}$：
    $\min_{\boldsymbol{\alpha}, \mathbf{W}_v, \mathbf{A}_v, \mathbf{Y}} \sum_{v=1}^V \alpha_v^2 \|\mathbf{X}_v - \mathbf{W}_v \mathbf{A}_v \mathbf{Y}\|_F^2 + \lambda_1 \|\mathbf{Y}_l - \mathbf{G}_l\|_F^2$
      约束 $\boldsymbol{\alpha}^\top \mathbf{1} = 1, \boldsymbol{\alpha} \geq \mathbf{0}$。视图权重根据重构误差自动调整：
    $\alpha_v = \frac{1/r_v^2}{\sum_{v=1}^V 1/r_v^2}$
      其中 $r_v^2 = \|\mathbf{X}_v - \mathbf{W}_v \mathbf{A}_v \mathbf{Y}\|_F^2$
    - 设计动机：不同视图的质量和重要性不同，固定权重无法适应具体数据。通过已知类的真实标签 $\mathbf{G}_l$ 约束预测标签 $\mathbf{Y}_l$ 的学习，间接优化视图权重，然后将学到的权重应用于新类

3. **类别分离约束**:

    - 功能：防止新类样本被错误归入已知类
    - 核心思路：在最终目标函数中添加排斥项，最大化新类预测标签与已知类真实标签之间的距离：
    $\mathcal{L} = \sum_v \alpha_v^2 \|\mathbf{X}_v - \mathbf{W}_v \mathbf{A}_v \mathbf{Y}\|_F^2 + \lambda_1 \|\mathbf{Y}_l - \mathbf{G}_l\|_F^2 - \lambda_2 \sum_{\mathbf{g}^i \in \mathbf{G}_l} \sum_{\mathbf{y}^j \in \mathbf{Y}_u} \|\mathbf{g}^i - \mathbf{y}^j\|_F^2$
    - 设计动机：由于已知类和新类分布相似，联合学习时新类样本容易被错误分配到已知类的聚类中。排斥项鼓励新类标签远离已知类标签

### 损失函数 / 训练策略

- **优化方法**：四步交替优化，每步固定其余变量优化一个变量
    - $\mathbf{W}_v$：通过 SVD 闭式求解 $\mathbf{W}_v = \mathbf{S}_v \mathbf{V}_v^\top$
    - $\mathbf{A}_v$：对导数置零直接求解 $\mathbf{A}_v = \mathbf{W}_v^\top \mathbf{X}_v \mathbf{Y}^\top (\mathbf{Y}\mathbf{Y}^\top)^{-1}$
    - $\mathbf{Y}$：逐样本离散优化（one-hot 约束）
    - $\boldsymbol{\alpha}$：基于 Cauchy-Schwarz 不等式闭式更新
- **收敛性保证**：目标函数在每步迭代中单调递减，且有下界 $\mathcal{J} \geq -\lambda_2 n_l n_u \sqrt{2}$
- **时间复杂度**：$\mathcal{O}(d(nk + k^2) + Vk^3)$，对样本数 $n$ 线性，保证可扩展性

## 实验关键数据

### 主实验

**8 个数据集上的 ACC 对比**（vs 多视图聚类和 NCD 方法）：

| 数据集 | AEVC (MVC) | CKD (NCD) | IICMVNCD | 提升 |
|--------|-----------|-----------|-----------|------|
| BRCA | 86.67 | 84.32 | **98.79** | +12.12 |
| uci-digit | 92.60 | 92.50 | **95.30** | +2.70 |
| Cora | 63.20 | 35.23 | **76.36** | +13.16 |
| Wiki | 37.30 | 61.05 | **65.42** | +4.37 |
| STL10 | 98.74 | 96.11 | **99.02** | +0.28 |
| YTB10 | 91.59 | 93.01 | **94.55** | +1.54 |

### 消融实验

**NMI 指标对比**：

| 数据集 | 最佳 MVC | 最佳 NCD | IICMVNCD | 说明 |
|--------|---------|---------|-----------|------|
| BRCA | 87.81 | 86.93 | **90.45** | 大幅超越 |
| Cora | 29.96 | 2.12 | **44.59** | NCD 方法严重退化 |
| Wiki | 49.98 | 35.10 | **66.44** | 显著提升 |
| CCV | 17.81 | 16.83 | **19.24** | 小幅提升 |

### 关键发现

1. **多视图 NCD 与单视图 NCD 差异显著**：现有单视图 NCD 方法（如 CKD）在多视图数据上表现不稳定，甚至不如传统多视图聚类方法（如 Cora 上 ACC 仅 35.23% vs AEVC 的 63.20%）
2. **视图权重学习至关重要**：不同数据集上各视图的最优权重差异很大，自适应权重比固定均等权重性能明显更好
3. **无需伪标签的设计**有效避免了噪声标签问题，在高维多视图数据上尤为显著
4. 方法在医学多组学数据（BRCA、KIPAN）上表现尤其出色（98.79%、92.51% ACC），验证了多视图 NCD 在生物医学中的应用价值
5. 理论收敛性保证使得方法在各种数据条件下都能稳定运行

## 亮点与洞察

- **首次提出多视图 NCD 任务**，填补了重要的研究空白。现有 NCD 方法仅限单视图，而多视图数据在生物信息学等领域极为普遍
- **彻底摆脱伪标签**的优雅设计：通过矩阵分解 + 视图加权直接预测标签矩阵，从根本上避免了伪标签噪声
- **闭式解的交替优化**使方法高效且收敛有保证，这在 NCD 领域少见
- 视图权重学习机制将已知类的视图关系"迁移"到新类，利用了 NCD 中有标注数据的独特优势

## 局限与展望

- 矩阵分解假设线性特征空间，对于复杂非线性特征分布可能不够表达
- 需要预知新类数量 $k_u$，这在实际应用中可能不切实际
- 数据集分割方式简单（前半/后半类别），真实场景中已知类和新类的关系可能更复杂
- 缺少与深度学习基线（如 DINOv2 特征 + 聚类）的对比
- 类别分离约束使用简单的距离排斥，可能导致新类标签分布被过度推远

## 相关工作与启发

- AutoNovel (ICLR 2021) 和 UNO 等 NCD 先驱工作集中在图像单视图，本文的多视图扩展有方法论意义
- 与 Multi-view NMF 聚类方法（如 DiNMF, OPMC）经典方法有直接关联，但加入了 NCD 的监督信号
- 多组学数据的成功应用提示了与生物信息学的交叉研究方向
- 视图权重学习策略可推广到多模态 few-shot 学习等场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首创多视图 NCD 设定，方法设计干净优雅
- 实验充分度: ⭐⭐⭐⭐ 8 个数据集覆盖面广，但缺少深度特征基线
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，理论分析完整
- 价值: ⭐⭐⭐⭐ 开辟新方向，对生物信息学等多视图场景有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](../../ECCV2024/others/mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)
- [\[CVPR 2025\] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](../../CVPR2025/others/which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)

</div>

<!-- RELATED:END -->
