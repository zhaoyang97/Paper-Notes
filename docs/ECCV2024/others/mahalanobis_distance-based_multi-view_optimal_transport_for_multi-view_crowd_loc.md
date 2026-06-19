---
title: >-
  [论文解读] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization
description: >-
  [ECCV 2024][多视角人群定位] 提出基于马氏距离的多视角最优传输损失（M-MVOT），通过视线方向和目标到相机的距离自适应调整传输代价，首次将点监督最优传输引入多视角人群定位任务，显著超越基于密度图MSE损失的方法。 多视角人群定位旨在融合多个相机视角的信息来预测场景地面上所有人的位置，广泛应用于人群分析、自动驾驶…
tags:
  - "ECCV 2024"
  - "多视角人群定位"
  - "最优传输"
  - "马氏距离"
  - "密度图"
  - "点监督"
---

# Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization

**会议**: ECCV 2024  
**arXiv**: [2409.01726](https://arxiv.org/abs/2409.01726)  
**代码**: 有 (项目页面)  
**领域**: 其他  
**关键词**: 多视角人群定位, 最优传输, 马氏距离, 密度图, 点监督

## 一句话总结

提出基于马氏距离的多视角最优传输损失（M-MVOT），通过视线方向和目标到相机的距离自适应调整传输代价，首次将点监督最优传输引入多视角人群定位任务，显著超越基于密度图MSE损失的方法。

## 研究背景与动机

多视角人群定位旨在融合多个相机视角的信息来预测场景地面上所有人的位置，广泛应用于人群分析、自动驾驶和公共交通管理。

**现有方法的问题**：

**密度图监督的局限**：现有方法依赖固定大小高斯核的密度图作为监督信号，使用MSE损失训练。在拥挤区域，当高斯核显著重叠时，局部峰值被平滑掉，导致无法准确定位每个人。

**单视图OT未扩展到多视图**：最优传输（OT）损失在单图人群定位中已展现出显著优势（如GL方法），但尚未被探索用于多视角人群定位。

**多视角特有挑战**：特征从相机视角投影到地面平面时，因未知物体3D高度，会沿视线方向产生条纹伪影，影响密度图定位精度。

## 方法详解

### 整体框架

模型架构遵循标准多视角人群定位流程：单视角特征提取 → 投影到地面平面 → 多视角融合与解码。创新在于将地面平面密度图的MSE损失替换为提出的M-MVOT损失。

### 关键设计

**1. 马氏距离传输代价（替代欧氏距离）**

标准OT使用欧氏距离作为传输代价：$C_{ij} = \exp(\|\mathbf{x}_i - \mathbf{y}_j\|)$

本文使用马氏距离，定义椭圆形等值线的代价函数：
$$C_{ij} = \exp(\sqrt{(\mathbf{x}_i - \mathbf{y}_j)^T \mathbf{S}^{-1} (\mathbf{x}_i - \mathbf{y}_j)})$$

协方差矩阵 $\mathbf{S} = \mathbf{R} \boldsymbol{\Sigma} \mathbf{R}^{-1}$，其中旋转矩阵 $\mathbf{R}$ 由视线方向确定。

**2. 视线方向引导（MV-OT）**

椭圆的短轴沿视线方向（$\sigma_1^2=1$），长轴垂直于视线方向（$\sigma_2^2=1.2$）。这意味着预测偏离视线方向的误差会被更重地惩罚，以对抗投影造成的条纹伪影。

**3. 距离自适应调整（ED-OT）**

远离相机的点倾向于有更大的预测误差。通过目标到相机的距离调整方差：
$$\sigma_1^2 = \sigma_2^2 = 1/\exp(\alpha \cdot \text{MinMaxNorm}(d_{cam}))$$

距离越远，方差越小，惩罚越大。

**4. 联合视线+距离（M-OT）**

结合两种机制：
$$\sigma_1^2 = 1, \quad \sigma_2^2 = \exp(\alpha \cdot \text{MinMaxNorm}(d_{cam}))$$

远处的点在视线方向上被更重惩罚，同时在垂直于视线的方向上适当放松。

**5. 多视角扩展（M-MVOT）**

基于距离选择策略，每个ground truth点的传输代价由最近相机的M-OT计算：
$$C_{ij} = \sum_{k=1}^K \mathbb{1}(d_{cam}^k) \exp(\sqrt{(\mathbf{x}_i - \mathbf{y}_j)^T \mathbf{S}_k^{-1} (\mathbf{x}_i - \mathbf{y}_j)})$$

### 损失函数 / 训练策略

采用非平衡最优传输（UOT）公式，使用Sinkhorn迭代求解。总损失为M-MVOT损失加上辅助的2D密度图损失。超参数 $\tau$ 在CVCS和Wildtrack上设为1，MultiviewX上设为20；$\alpha$ 在CVCS上为1，其余为0.05。

## 实验关键数据

### 主实验（表格）

**CVCS数据集（跨场景）**：

| 方法 | MODA↑ | MODP↑ | Precision↑ | Recall↑ | F1↑ |
|------|-------|-------|------------|---------|-----|
| **M-MVOT (Ours)** | **43.5** | 74.1 | 85.5 | **52.3** | **64.9** |
| E-MVOT (Ours) | 43.1 | 74.3 | 85.6 | 51.8 | 64.5 |
| SHOT | 31.7 | 72.1 | 94.5 | 33.6 | 49.6 |
| MVDeTr | 24.9 | **79.6** | **98.1** | 25.4 | 40.4 |
| 3DROM | 20.1 | 74.2 | 84.1 | 23.7 | 37.0 |
| MVDet | 14.2 | 59.3 | 85.0 | 17.3 | 28.7 |

**MultiviewX数据集**：

| 方法 | MODA↑ | MODP↑ | F1↑ |
|------|-------|-------|-----|
| **M-MVOT (Ours)** | **96.7** | 86.1 | **98.3** |
| 3DROM | 95.0 | 84.9 | 97.5 |
| MVDeTr | 93.7 | **91.3** | 97.8 |

### 消融实验（表格）

| 损失函数 | 视线方向 | 距离调整 | MODA↑ | F1↑ |
|----------|----------|----------|-------|-----|
| MSE | ✗ | ✗ | 14.2 | 28.7 |
| E-MVOT | ✗ | ✗ | 43.1 | 64.5 |
| MV-MVOT | ✓ | ✗ | 42.2 | 63.4 |
| ED-MVOT | ✗ | ✓ | 38.7 | 62.3 |
| **M-MVOT** | **✓** | **✓** | **43.5** | **64.9** |

### 关键发现

1. 从MSE到OT的转换带来了巨大的提升（MODA从14.2到43.1），证明点监督优于密度图监督
2. M-MVOT始终优于E-MVOT，验证了马氏距离代价函数的有效性
3. 在跨场景数据集CVCS上提升最显著，表明方法具有良好的泛化能力
4. 可视化显示M-MVOT在拥挤区域和远处区域均有更好的定位效果
5. M-MVOT减少了误检和条纹伪影

## 亮点与洞察

- **首次将OT点监督引入多视角定位**：填补了研究空白，从密度图到点监督的转换带来了质的飞跃
- **物理驱动的代价函数设计**：马氏距离的椭圆等值线巧妙对应了投影伪影的方向特性
- **视线方向+距离**的联合建模，充分利用了多视角系统的几何信息
- 方法是即插即用的损失函数，可与任何现有多视角定位模型结合

## 局限与展望

- 在小规模单场景数据集Wildtrack上不如3DROM（专门的数据增强方法），可能存在过拟合问题
- 最近相机的选择策略较为简单，可考虑加权多相机的贡献
- 超参数 $\alpha$ 对不同数据集需要不同设置，增加了调参难度
- 未探索与检测框架（如DETR风格）的结合
- 可考虑学习自适应的协方差矩阵而非手动设定

## 相关工作与启发

- GL方法证明了MSE损失和Bayesian损失是非平衡OT损失的特殊次优情况，为本文提供了理论基础
- 马氏距离在传统统计学中广泛使用，本文将其与相机几何巧妙结合
- P2PNet的一对一匹配范式提供了另一种替代密度图的思路
- 3DROM的数据增强与本文的损失改进是正交的，二者可以结合

## 评分

- **创新性**: ★★★★☆ — 首次将OT引入多视角定位，马氏距离的设计有物理直觉
- **实用性**: ★★★★☆ — 即插即用损失函数，易于集成
- **实验完整性**: ★★★★★ — 三个数据集+详尽消融+可视化分析
- **写作质量**: ★★★★☆ — 数学推导清晰，层层递进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance](improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)
- [\[CVPR 2026\] Multi-view Crowd Tracking Transformer with View-Ground Interactions Under Large Real-World Scenes](../../CVPR2026/others/multi-view_crowd_tracking_transformer_with_view-ground_interactions_under_large_.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](../../ICCV2025/others/intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[CVPR 2026\] DF²-VB: Dual-level Fuzzy Fusion with View-specific Boosting for Multi-view Multi-label Classification](../../CVPR2026/others/df2-vb_dual-level_fuzzy_fusion_with_view-specific_boosting_for_multi-view_multi-.md)
- [\[ECCV 2024\] FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information](fisherrf_active_view_selection_and_mapping_with_radiance_fields_using_fisher_inf.md)

</div>

<!-- RELATED:END -->
