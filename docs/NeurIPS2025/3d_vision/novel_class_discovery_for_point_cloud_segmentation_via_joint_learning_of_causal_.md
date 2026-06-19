---
title: >-
  [论文解读] Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning
description: >-
  [NeurIPS 2025][3D视觉][新类发现] 本文首次将因果学习引入3D点云新类发现（3D-NCD），通过结构因果模型（SCM）分析基类中的混杂因子和基-新类间的因果关系，提出因果表示原型学习（通过对抗网络消除混杂因子）和基于图的因果推理（GCN生成伪标签），在SemanticKITTI和SemanticPOSS上取得了SOTA结果。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "新类发现"
  - "点云语义分割"
  - "因果学习"
  - "结构因果模型"
  - "图卷积网络"
---

# Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2510.13307](https://arxiv.org/abs/2510.13307)  
**代码**: 暂无  
**领域**: 3D视觉  
**关键词**: 新类发现, 点云语义分割, 因果学习, 结构因果模型, 图卷积网络

## 一句话总结

本文首次将因果学习引入3D点云新类发现（3D-NCD），通过结构因果模型（SCM）分析基类中的混杂因子和基-新类间的因果关系，提出因果表示原型学习（通过对抗网络消除混杂因子）和基于图的因果推理（GCN生成伪标签），在SemanticKITTI和SemanticPOSS上取得了SOTA结果。

## 研究背景与动机

传统点云语义分割采用"封闭世界"假设——只能分割训练集中出现过的类别。**3D新类发现（3D-NCD）** 旨在仅利用已标注的基类监督，学习一个能分割未标注新类的模型。这对自动驾驶和机器人等需要应对未知类别的场景至关重要。

该任务的关键在于：

**准确建立点表示与基类标签的关联**

**建立基类与新类之间的表示关联**

现有方法面临两个核心问题：

**问题一：快捷特征（Shortcut Features）干扰基类学习**。基类分类器本质上是统计模型，容易学到表面层的虚假关联而非内在因果机制。例如，模型可能将"圆形支撑结构"这一视觉显著但非因果的特征与"凳子"关联，导致新类推理时将有相似结构的"椅子"误判为"凳子"。如果改用因果学习，应该发现"椅腿"才是区分这些类别的因果特征。

**问题二：缺乏基-新类因果关系建模**。新类通常是基类在某种因果机制下的变体（如"骑手"是"人"在"自行车"语境下的因果变异，"卡车"与"汽车"共享"载具"的因果先验）。现有方法仅依赖统计相似性，未显式建模这种因果传导路径。

作者引入SCM将3D-NCD重新形式化：四个因果变量（原始点云$X$、基类$B$、新类$N$、混杂因子$U$），路径 $U \to B$ 表示混杂因子影响基类学习，$B \to N$ 表示基类对新类的因果影响。目标是消除$U$、建立$B \to N$的因果推理。

## 方法详解

### 整体框架

方法包含三个核心模块：
1. **因果表示原型学习（CRP）**：通过对抗机制消除混杂因子，获取基类的因果表示原型
2. **因果推理图构建（CRG）**：用图结构显式建模基-新类的因果关系
3. **基于GCN的伪标签生成（GCPL）**：通过图卷积网络生成新类伪标签

### 关键设计

1. **因果表示原型学习（CRP）**：核心目标是学习特征表示 $Z$，使其与混杂因子 $U$ 独立（$Z \perp U$），即最小化互信息 $I(Z;U)$。这符合独立因果机制（ICM）原则——生成基类的真实因果机制应独立于产生混杂因子的机制。通过类似GAN的对抗训练实现：

$$\min_\theta \max_\phi \mathcal{L}_{ADV} = \mathcal{L}_{cls}(f_\theta(X_B), Y_B) - \lambda_{adv} \mathcal{L}_{adv}(g_\phi(Z), U)$$

特征提取器 $f_\theta$ 尝试提取去除 $U$ 的因果特征，对抗网络 $g_\phi$ 尝试从 $Z$ 恢复 $U$（如果恢复成功则说明 $Z$ 仍含混杂信息）。$\mathcal{L}_{cls}$ 为交叉熵分类损失，$\mathcal{L}_{adv}$ 为二元交叉熵对抗损失。训练后，基于软分配权重 $W_{ij} = \frac{\exp(\text{sim}(Z_j, C_i))}{\sum_k \exp(\text{sim}(Z_j, C_k))}$ 迭代更新原型 $C_i^{(t+1)} = \frac{\sum_j W_{ij} \cdot Z_j}{\sum_j W_{ij}}$。原型匹配损失为：$\mathcal{L}_{PRO} = -\sum_i \sum_j W_{ij} \cdot \text{sim}(Z_j, C_i) + \lambda \|C_i\|_2^2$。

2. **因果推理图构建（CRG）**：受因果贝叶斯网络（CBN）和因果马尔可夫原则启发，构建有向图来建模 $B \to N$ 的因果路径。图节点包括$M$个基类因果原型 $C$ 和 $K$ 个新类原型 $N$。引入**因果自适应邻接矩阵** $A = [A_{ij}]_{M \times K}$，权重通过自注意力机制动态调整：$w_{ij} = \text{softmax}(\text{Attention}(c_i, n_j)/\tau)$。两个关键约束保证因果有效性：

    - **推理方向一致性约束**：确保信息沿因果方向（基→新）流动：$\mathcal{L}_{direction} = \sum_{(c_i, n_j) \in E} (w_{ij} \cdot (1 - \mathbb{I}(c_i \to n_j)))^2$
    - **因果剪枝约束**：移除因果权重低于可学习阈值 $\theta$ 的边：$\mathcal{L}_{pruning}(\theta) = \sum_{(c_i, n_j) \in E} \mathbb{I}(w_{ij} < \theta) \cdot w_{ij}^2$

3. **基于GCN的伪标签生成**：现有方法直接用相似度匹配生成伪标签，忽略了类间的高阶依赖关系。本文使用图卷积网络聚合邻居信息：

$$\mathbf{n}_j^{(t+1)} = \sigma\left(\sum_{i=1}^M \frac{w_{ij}}{\sqrt{d_i d_j}} \cdot \mathbf{c}_i^{(t)} + \sum_{k=1}^K \frac{w_{jk}}{\sqrt{d_j d_k}} \cdot \mathbf{n}_k^{(t)}\right)$$

经过多层GCN后，用余弦相似度匹配生成伪标签：$\hat{y}_j = \arg\min_i \text{sim}(\mathbf{n}_j^{\text{final}}, \mathbf{c}_i)$。

### 损失函数 / 训练策略

- 骨干网络：MinkowskiUNet-34C
- 优化器：AdamW，初始lr=1e-3，每5个epoch衰减至最低1e-5
- $\lambda_{adv}$ 和剪枝阈值 $\theta$ 初始均为0.5，训练中动态调整
- 温度参数 $\tau = 0.06$，正则化系数 $\lambda = 0.02$
- GCN层数设为3
- 新类仅提取因果表示原型，不进行因果去混杂（因为缺乏可靠标签）

## 实验关键数据

### 主实验

**SemanticPOSS数据集（4个split的平均）：**

| 方法 | Split 0 Novel | Split 0 All | Split 1 Novel | Split 1 All | Split 2 Novel | Split 3 Novel |
|------|---------|--------|---------|--------|---------|---------|
| EUMS | 17.4 | 20.3 | 21.0 | 35.6 | 8.3 | 13.0 |
| NOPS | 35.7 | 29.4 | 30.0 | 36.4 | 9.0 | 10.9 |
| SNOPS | 51.2 | 33.6 | 32.1 | 37.2 | 16.9 | 20.1 |
| DASL | 48.4 | 41.2 | 36.2 | 44.0 | 12.6 | 17.7 |
| **Ours** | **51.3** | **41.7** | **37.3** | **45.8** | **13.6** | **27.9** |

**SemanticKITTI数据集（4个split）：**

| 方法 | Split 0 Novel↑ | Split 0 All↑ | Split 1 Novel↑ | Split 2 Novel↑ | Split 3 Novel↑ |
|------|---------|--------|---------|---------|---------|
| NOPS | 37.1 | 29.3 | 25.4 | 16.5 | 12.4 |
| SNOPS | 45.9 | 31.2 | 27.2 | 17.6 | 14.9 |
| DASL | 45.7 | 36.8 | 28.7 | 20.1 | 12.6 |
| **Ours** | **46.9** | **36.9** | **33.6** | **18.5** | **15.1** |

### 消融实验

**组件分析（SemanticPOSS Split 0 和平均）：**

| Baseline | CRP | CRG | GCPL | Split0 Novel↑ | 全Split平均↑ | 说明 |
|----------|-----|-----|------|---------|---------|------|
| ✓ | | | | 38.3 | 24.7 | 基线 |
| ✓ | ✓ | | | 41.6 | 25.9 | +因果表示原型 |
| ✓ | | ✓ | | 45.5 | 28.8 | +因果推理图（传统聚类原型） |
| ✓ | ✓ | ✓ | | 47.4 | 30.1 | CRP+CRG |
| ✓ | ✓ | ✓ | ✓ | **51.2** | **32.5** | 完整方法 |

**2D NCD扩展实验（PASCAL-5i / COCO-20i）：**

| 方法 | PASCAL-5i Avg | COCO-20i Avg |
|------|-------------|-------------|
| EUMS | 59.1 | 26.81 |
| **Ours** | **61.2** | **27.00** |

### 关键发现

- 因果推理图（CRG）对性能提升最大（Novel从38.3→45.5），说明建模基-新类因果关系比去混杂更重要
- CRP和CRG组合后效果叠加（47.4），进一步加GCN提升到51.2，说明三个组件互补
- 在最具挑战性的Split 3中，本方法在cone-stone类上达到36.2%（DASL和NOPS均为0），展示了因果推理对困难新类的特殊优势
- 方法同样适用于2D NCD（PASCAL和COCO），验证了因果学习的通用性
- Grad-CAM可视化显示本方法聚焦于更精确的因果区域，而EUMS产生分散的特征图

## 亮点与洞察

- **第一个将因果学习引入3D NCD的工作**，为该领域提供了新的理论视角
- 用SCM重新形式化NCD问题非常清晰：混杂因子$U$对应快捷特征，$B \to N$对应类间知识迁移
- 因果剪枝+方向一致性约束的设计很巧妙，确保了图中信息流符合因果方向
- 对抗去混杂的方法不需要显式定义混杂因子是什么，比后门调整更实用
- 在困难类别上的表现尤其突出，说明因果推理对处理歧义性高的类别有独特优势

## 局限与展望

- 新类不进行因果去混杂处理（因为无标签），可能导致新类原型仍含噪声
- 混杂因子$U$的具体形式未明确定义，对抗训练只是间接近似$Z \perp U$
- GCN只用3层，对复杂的多跳因果关系可能不够充分
- 实验仅在两个相对小的数据集上验证，未在大规模点云（如nuScenes完整版）上测试
- 因果图的方向一致性约束中，指示函数$\mathbb{I}(c_i \to n_j)$的具体实现细节不够明确

## 相关工作与启发

- **NOPS/SNOPS/DASL**是3D NCD的直接前驱方法，本文在此基础上引入因果框架
- CausalPC在3D点云鲁棒性中使用SCM处理对抗扰动，本文将SCM应用于NCD任务，思路有几分借鉴
- 因果表示学习（Schölkopf等）提供了理论基础：ICM原则和do-calculus
- 方法可扩展到2D NCD（已验证），也可能适用于其他需要跨类知识迁移的分割任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将因果学习引入3D NCD，理论动机清晰
- **实验充分度**: ⭐⭐⭐⭐ 两个3D数据集+两个2D数据集，有消融和可视化分析
- **写作质量**: ⭐⭐⭐⭐ 因果分析部分写得清晰，但部分符号使用可更统一
- **价值**: ⭐⭐⭐⭐ 为NCD提供了因果学习的新范式，对3D开放世界感知有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Geometric-Aware Hypergraph Reasoning for Novel Class Discovery in Point Cloud Segmentation](../../CVPR2026/3d_vision/geometric-aware_hypergraph_reasoning_for_novel_class_discovery_in_point_cloud_se.md)
- [\[ECCV 2024\] Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation](../../ECCV2024/3d_vision/dual-level_adaptive_self-labeling_for_novel_class_discovery_in_point_cloud_segme.md)
- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)
- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](../../CVPR2025/3d_vision/p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)

</div>

<!-- RELATED:END -->
