---
title: >-
  [论文解读] Improve Representation for Imbalanced Regression through Geometric Constraints
description: >-
  [CVPR 2025][物理/科学计算][不平衡回归] 本文首次研究深度不平衡回归（DIR）中的表征空间均匀性问题，提出包络损失（enveloping loss）和同质性损失（homogeneity loss）两种几何约束来确保回归表征在超球面上均匀分布，并设计代理驱动表征学习（SRL）框架将全局几何约束整合到mini-batch训练中，在年龄估计等多个DIR任务上达到SOTA。
tags:
  - "CVPR 2025"
  - "物理/科学计算"
  - "不平衡回归"
  - "几何约束"
  - "表征均匀性"
  - "包络损失"
  - "代理驱动学习"
---

# Improve Representation for Imbalanced Regression through Geometric Constraints

**会议**: CVPR 2025  
**arXiv**: [2503.00876](https://arxiv.org/abs/2503.00876)  
**代码**: 有  
**领域**: 表征学习 / 人体理解  
**关键词**: 不平衡回归, 几何约束, 表征均匀性, 包络损失, 代理驱动学习

## 一句话总结
本文首次研究深度不平衡回归（DIR）中的表征空间均匀性问题，提出包络损失（enveloping loss）和同质性损失（homogeneity loss）两种几何约束来确保回归表征在超球面上均匀分布，并设计代理驱动表征学习（SRL）框架将全局几何约束整合到mini-batch训练中，在年龄估计等多个DIR任务上达到SOTA。

## 研究背景与动机

1. **领域现状**：不平衡数据集在各领域普遍存在。在不平衡分类中，表征空间的均匀性（uniformity）已被证明是有效学习欠表示类别的关键。方法包括解耦训练、对比学习、类级别均匀化等。

2. **现有痛点**：分类方法将特征聚成离散簇，但回归任务要求表征是连续且有序的。现有DIR方法主要关注训练无偏回归器（如LDS、FDS、BalancedMSE），或建模标签空间和特征空间的关系（如RankSim、RNC、SupReMix），但忽略了表征如何在整个特征空间中分布的问题。

3. **核心矛盾**：在不平衡回归中，vanilla训练的特征空间被多数样本主导，少数样本区域的表征被"压缩"到很小的空间中，导致预测精度差。但分类中衡量均匀性的方法（如类中心分散度）不适用于连续有序的回归场景。

4. **本文目标**：定义并量化回归表征空间中的"均匀性"概念，设计损失函数促进回归表征在超球面上的均匀分布。

5. **切入角度**：将回归表征的轨迹类比为"绕球缠绕毛线"——毛线（latent trace）要尽可能覆盖球面（包络性），同时要紧密平滑不松散（同质性）。

6. **核心 idea**：用包络损失让回归表征轨迹充分占满超球面，用同质性损失确保表征沿轨迹均匀分布且平滑，通过代理（surrogate）机制将全局几何约束应用到batch训练中。

## 方法详解

### 整体框架
输入：不平衡分布的回归数据对$(x_i, y_i)$。输出：经过特征提取器$f(\cdot)$的归一化表征$z_i = f(x_i)$，加回归头预测。核心是将特征表征约束在单位超球面上，通过SRL框架计算全局几何损失。训练流程：每个mini-batch中(1)编码样本到latent space；(2)对同bin样本取平均作为centroid；(3)用上一epoch的centroid填补当前batch缺失的bin；(4)在完整surrogate上计算几何损失；(5)epoch结束时更新surrogate（动量式更新）。

### 关键设计

1. **包络损失（Enveloping Loss）**:

    - 功能：鼓励回归表征的轨迹（latent trace）尽可能覆盖超球面表面
    - 核心思路：定义trace $l$ 的管状邻域$T(l, \epsilon)$为超球面上到trace距离小于$\epsilon$的所有点。包络损失$\mathcal{L}_{\text{env}} = -\text{vol}(T(l,\epsilon))/\text{vol}(\mathcal{U})$。实际计算时采用连续到离散的近似：在超球面上均匀采样N个点（蒙特卡洛方法），计算落在管状邻域内的点的比例。为保证可微性，不使用硬阈值判断，而是最大化每个采样点与trace上最近点的余弦相似度（软化版本）。
    - 设计动机：回归表征的trace是一条线，直接计算其体积为零。通过管状邻域扩展，将trace的"空间覆盖程度"转化为可优化的损失。

2. **同质性损失（Homogeneity Loss）**:

    - 功能：确保表征沿trace均匀分布且trace平滑无折叠
    - 核心思路：通过惩罚trace的弧长来实现。离散形式为$\mathcal{L}_{\text{homo}} = \sum_{k=1}^{K-1}\frac{\|l(y_{k+1}) - l(y_k)\|^2}{y_{k+1} - y_k}$，即相邻centroid之间的距离平方除以标签差。定理1证明：给定trace形状固定时，同质性损失最小当且仅当表征沿trace均匀分布（即$\|\nabla_y l(y)\| = c$为常数）。
    - 设计动机：只有包络损失可能导致表征沿trace分布不均（多数样本区域密集，少数区域稀疏）或产生锯齿形trace。同质性损失作为正则化，同时解决分布均匀性和平滑性两个问题。单独使用同质性损失会导致特征坍缩到圆或点。

3. **代理驱动表征学习框架（SRL Framework）**:

    - 功能：将全局几何约束整合到基于mini-batch的SGD训练中
    - 核心思路：一个mini-batch通常不包含所有标签bin。SRL维护一个代理（surrogate）$\mathcal{S}$——包含所有K个bin的centroid。每个batch中：对出现的bin取centroid，缺失的bin用上一epoch保存的centroid填充。几何损失在完整surrogate上计算。Epoch间通过动量更新$\mathcal{S}^{e+1} \leftarrow \alpha \cdot \mathcal{S}^e + (1-\alpha) \cdot \hat{\mathcal{S}}^e$。还加入对比损失$\mathcal{L}_{\text{con}}$将个体表征拉近其对应centroid、推远其他centroid。
    - 设计动机：几何约束需要看到完整标签范围的表征分布才有意义，但batch采样是随机的。Surrogate机制巧妙地用历史信息补全了全局视角。

### 损失函数 / 训练策略
总损失$\mathcal{L}_\theta = \mathcal{L}_{\text{reg}} + \mathcal{L}_G + \mathcal{L}_{\text{con}}$，其中$\mathcal{L}_{\text{reg}}$是MSE回归损失，$\mathcal{L}_G = \lambda_e \mathcal{L}_{\text{env}} + \lambda_h \mathcal{L}_{\text{homo}}$是几何约束，$\mathcal{L}_{\text{con}}$是centroid对比损失。第一个epoch仅用MSE训练（此时surrogate尚未初始化），使用AdamW优化器带动量更新。

## 实验关键数据

### 主实验
在AgeDB-DIR年龄估计任务上（以MAE和GM为指标）：

| 方法 | All MAE↓ | Many MAE↓ | Med MAE↓ | Few MAE↓ | All GM↓ |
|------|---------|----------|---------|---------|---------|
| Vanilla | 7.67 | 6.66 | 9.30 | 12.61 | 4.85 |
| LDS+FDS | 7.55 | 7.03 | 8.46 | 10.52 | 4.86 |
| RankSim | 7.41 | 6.49 | 8.73 | 12.47 | 4.71 |
| ConR | 7.41 | 6.51 | 8.81 | 12.04 | 4.70 |
| **SRL (ours)** | **7.22** | 6.64 | **8.28** | **9.81** | **4.50** |

UCI-Airfoil数据集上（MAE）：

| 方法 | All↓ | Many↓ | Med↓ | Few↓ |
|------|------|-------|------|------|
| Vanilla | 5.66 | 5.11 | 5.03 | 6.75 |
| RankSim | 5.23 | 5.05 | 4.91 | 5.72 |
| **SRL (ours)** | **5.10** | **4.83** | **4.75** | **5.69** |

### 消融实验

| 配置 | 效果观察 | 说明 |
|------|---------|------|
| Baseline (MSE only) | 特征坍缩到多数样本区域 | 少数样本表征被压缩 |
| SRL w/o $\mathcal{L}_{\text{env}}$ | 特征坍缩为平凡形状 | 缺少包络约束，不能充分利用特征空间 |
| SRL w/o $\mathcal{L}_{\text{homo}}$ | 特征沿trace不均匀分布 | 有包络但分布不平滑 |
| **SRL (full)** | **均匀且平滑地填满特征空间** | 两个损失互补 |

### 关键发现
- 在Few-shot区域改进最为显著（AgeDB: 12.61→9.81，降低22%），证明均匀性对少数样本至关重要
- 包络损失和同质性损失缺一不可：t-SNE可视化清楚展示了单独去掉任一损失的退化模式
- 代理机制对于将全局约束引入batch训练是必要的——直接在batch上计算几何损失无效
- 首创的不平衡算子学习（IOL）任务验证了方法在函数空间映射中的有效性

## 亮点与洞察
- **毛线绕球的类比**：将回归表征均匀性问题用直觉化的几何类比来解释，使得包络和同质性两个概念非常好理解。这种从几何直觉出发设计损失函数的方法论可以推广到其他连续空间学习问题。
- **Surrogate机制的巧妙设计**：用历史centroid填补batch中缺失的bin，用动量更新保持稳定性。这种"记忆库"式的设计有效解决了全局约束与batch训练的兼容性问题，可迁移到其他需要全局信息的batch训练场景。
- **定理1的理论支撑**：证明了在trace形状固定时，弧长最小等价于均匀分布，为同质性损失提供了坚实的数学基础。

## 局限与展望
- 包络损失需要蒙特卡洛采样大量点来近似超球面，高维空间中采样效率可能受限
- Surrogate的动量更新引入了延迟，训练初期（第一个epoch只用MSE）的表征质量可能影响后续收敛
- bin划分对方法有影响，但论文中bin划分的敏感性分析不够深入
- 可以探索将几何约束扩展到多维标签回归（多目标回归）场景
- 与基于对比学习的DIR方法（如ConR、RNC）相比，理论上两者可以结合使用

## 相关工作与启发
- **vs RankSim**: RankSim关注标签空间和特征空间的顺序一致性，但不关心特征在整个空间中的覆盖程度。本文的包络损失直接优化空间利用率。
- **vs ConR**: ConR用对比学习建模全局和局部标签相似性，但不显式约束特征空间的均匀性。两者关注的是不同层面的问题，可能互补。
- **vs SupReMix/RNC**: 这些方法学习连续有序的表征，是本文的基础假设之一，但它们没有解决表征如何与整个特征空间交互的问题。
- 将均匀性概念从分类推广到回归的思路很有启发性，未来可以继续推广到其他连续预测任务（如深度估计、密度估计）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将表征均匀性概念从分类推广到回归，几何损失设计有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+可视化分析丰富，但缺少大规模数据集实验
- 写作质量: ⭐⭐⭐⭐⭐ 类比直觉、数学严谨、可视化出色
- 价值: ⭐⭐⭐⭐ 为DIR领域开辟了表征学习的新视角，几何约束思路有广泛适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] From Geometry to Dynamics: Learning Overdamped Langevin Dynamics from Sparse Observations with Geometric Constraints](../../ICML2026/physics/from_geometry_to_dynamics_learning_overdamped_langevin_dynamics_from_sparse_obse.md)
- [\[ICML 2026\] From Generalist to Specialist Representation](../../ICML2026/physics/from_generalist_to_specialist_representation.md)
- [\[ICLR 2026\] Deep Learning for Subspace Regression](../../ICLR2026/physics/deep_learning_for_subspace_regression.md)
- [\[NeurIPS 2025\] Latent Representation Learning in Heavy-Ion Collisions with MaskPoint Transformer](../../NeurIPS2025/physics/latent_representation_learning_in_heavy-ion_collisions_with_maskpoint_transforme.md)
- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/physics/towards_universal_neural_operators_through_multiphysics_pretraining.md)

</div>

<!-- RELATED:END -->
