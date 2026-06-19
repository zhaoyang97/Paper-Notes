---
title: >-
  [论文解读] CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection
description: >-
  [ICCV 2025][3D视觉][图像-点云配准] 提出 CA-I2P，通过 Channel Adaptive Adjustment Module (CAA) 增强并过滤图像-点云特征的通道差异，并用 Global Optimal Selection (GOS) 基于最优传输替代 top-k 选择减少多对一匹配误差，在 RGB-D Scenes V2 和 7-Scenes 上实现图像-点云配准 SOTA。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "图像-点云配准"
  - "跨模态特征匹配"
  - "通道自适应"
  - "最优传输"
  - "检测无关方法"
---

# CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection

**会议**: ICCV 2025  
**arXiv**: [2506.21364](https://arxiv.org/abs/2506.21364)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 图像-点云配准, 跨模态特征匹配, 通道自适应, 最优传输, 检测无关方法

## 一句话总结

提出 CA-I2P，通过 Channel Adaptive Adjustment Module (CAA) 增强并过滤图像-点云特征的通道差异，并用 Global Optimal Selection (GOS) 基于最优传输替代 top-k 选择减少多对一匹配误差，在 RGB-D Scenes V2 和 7-Scenes 上实现图像-点云配准 SOTA。

## 研究背景与动机

图像-点云配准（I2P）估计从点云坐标系到相机坐标系的刚体变换，是 3D 重建、SLAM 和视觉定位的基础。然而，图像（2D 规则密集网格）和点云（3D 无序稀疏不规则数据）之间存在巨大的域差异，使得跨模态配准极具挑战。

现有检测无关方法（如 2D3D-MATR）采用粗到细流程：先提取图像/点云特征，进行 patch 级匹配，再精化到像素-点对应。但存在两个核心问题：

**通道级特征不一致**：LiDAR 和相机的成像范围差异导致两种模态特征在不同通道上的注意力分布不同。2D 编码器和 3D 编码器输出的特征在通道维度上存在系统性偏差，会导致感受野不对齐和遮挡区域的错误匹配。如图所示，两种模态的通道颜色分布明显不同，经过模块处理后才趋于一致。

**局部选择导致冗余对应**：场景中常有相似结构（如重复的家具、对称的建筑），传统 top-k 选择策略会导致多个相似结构被错误匹配到同一个跨模态物体上，产生冗余的多对一对应关系，降低配准精度。

## 方法详解

### 整体框架

CA-I2P 使用 FPN-ResNet 提取图像特征、KPFCNN 提取点云特征，经 Fourier 位置编码和 Transformer 增强后，送入 Channel Adaptive Adjustment Module (CAA) 进行通道级优化，计算余弦相似度得到分数图后，经 Global Optimal Selection (GOS) 获取精确的 patch 级匹配，最终精化为像素-点密集匹配并用 PnP+RANSAC 估计变换。

### 关键设计

1. **Intra-Modal Enhancement Stage (IME)**：针对两种模态的特征分别设计增强单元：

    - **Image Channel Enhancement Unit (ICE)**：设计三个并行分支，分别捕获 $(H,W)$、$(C,H)$、$(C,W)$ 维度间的交叉依赖关系。通过沿不同轴旋转张量实现维度交互，每个分支使用 GO-Pool（Max+Avg 池化拼接）降维后经卷积+BN+Sigmoid 产生注意力权重，三分支结果取平均：
    $F'_I = \frac{1}{3}\sum_{y=1}^{3} F_{Iy} \sigma(\text{Conv}(\text{GO-Pool}(F_{Iy})))$
    - **Point Channel Enhancement Unit (PCE)**：对点云特征使用通道自注意力机制，通过 Q/K/V 线性投影计算通道间相关性并自适应调整：
    $A = \text{Softmax}\left(\frac{QK^T S}{\sqrt{C}}\right), \quad F'_P = AV$
    - 增强后的特征与原始特征通过可学习权重融合：$F_I = \alpha F_I + \beta F'_I$

2. **Cross-Modal Channel Filtering Stage (CMCF)**：

    - 先对每个样本做 Instance Normalization（IN），使特征独立于训练集统计量
    - 计算图像和点云特征的协方差矩阵 $\mathbf{V}^x_I$ 和 $\mathbf{V}^x_P$，再计算两者之间的协方差矩阵 $\mathbf{Cov}_x$
    - $\mathbf{Cov}_x(i,j)$ 衡量第 $i$ 和第 $j$ 通道在不同模态间的敏感度
    - 高方差通道包含更多关注非对应区域的注意力，通过选择性掩码 $M_x$ 过滤
    - 仅优化上三角部分，防止模型过拟合特定模态的统计信息
    - 过滤后的特征与原始特征融合以保持信息完整性
    - 过滤损失：$L_f = \frac{1}{X}\sum_{x=1}^X (\|\hat{\mathbf{V}}^x_I \odot M_x\|_1 + \|\hat{\mathbf{V}}^x_P \odot M_x\|_1)$

3. **Global Optimal Selection Module (GOS)**：

    - 将 patch 级匹配问题建模为最优传输（OT）问题
    - 相似度代价矩阵为 $(1-S)$，通过 Sinkhorn 迭代求解全局最优传输计划：
    $T^* = \min_{T \in \mathcal{T}} \text{Tr}(T^T(1-S)) - \epsilon H(\mathcal{T})$
    - 约束 $\mathcal{T} = \{T | T\mathbf{1} = \frac{1}{N}\mathbf{1}, T^T\mathbf{1} = \frac{1}{N}\mathbf{1}\}$ 确保均匀分布
    - 约 10 次 Sinkhorn 迭代即可收敛，高效可 GPU 并行
    - 从全局视角优化匹配，有效减少多对一错误

### 损失函数 / 训练策略

总损失包含三个部分：
$$L_{\text{total}} = \lambda_1 L_f + \lambda_2 L_{ic} + \lambda_3 L_{if}$$
- $L_f$：CMCF 的通道过滤损失
- $L_{ic}$：粗匹配 Circle Loss
- $L_{if}$：细匹配 Circle Loss

Circle Loss 通过对正负对的自适应加权平衡困难样本的学习。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 2D3D-MATR | 提升 |
|--------|------|------|-----------|------|
| RGB-D Scenes V2 | IR↑ | 35.5 | 32.4 | +3.1pp |
| RGB-D Scenes V2 | FMR↑ | 93.6 | 90.8 | +2.8pp |
| RGB-D Scenes V2 | RR↑ | 63.3 | 56.4 | +6.9pp |
| 7-Scenes | IR↑ | 51.6 | 50.1 | +1.5pp |
| 7-Scenes | FMR↑ | 92.4 | 92.1 | +0.3pp |
| 7-Scenes | RR↑ | 79.5 | 75.8 | +3.7pp |

与 FreeReg 比较：RR 提升 6pp（63.3 vs 57.3），FMR 提升 12pp。

### 消融实验

| 配置 | PIR | IR | FMR | RR | 说明 |
|------|-----|-----|-----|-----|------|
| 基线 (M1) | 48.5 | 32.5 | 91.0 | 55.8 | 无任何增强 |
| +ICE (M2) | 56.3 | 34.6 | 92.4 | 56.9 | 图像通道增强 |
| +PCE (M3) | 54.7 | 33.6 | 93.2 | 56.0 | 点云通道增强 |
| +ICE+PCE (M4) | 56.3 | 34.6 | 92.3 | 58.2 | 联合增强 |
| +CMCF (M5) | 59.2 | 35.4 | 91.4 | 59.7 | 跨模态过滤 |
| +CAA (M6) | 59.1 | 35.1 | 93.3 | 61.8 | 完整通道自适应 |
| +GOS (M7) | 56.3 | 35.3 | 91.7 | 58.1 | 全局最优选择 |
| 完整 CA-I2P (M8) | 58.3 | 35.5 | 93.6 | 63.3 | 全部组件 |

CMCF 贡献最大的 PIR 提升（+10.7pp），GOS 贡献最大的 IR 提升（+2.8pp），整体 RR 提升 +7.5pp。

### 关键发现

- 模态内特征增强和跨模态通道过滤各自贡献显著且互补
- CMCF 的 Instance Normalization + 协方差矩阵分析有效揭示了通道级模态差异
- GOS 的最优传输策略在存在重复结构的室内场景中尤为有效
- 在挑战性最大的 Heads（近距离小误差放大）和 Stairs（重复模式）场景上提升最为显著
- 方法在室内场景下优于额外引入重叠区域检测器的方案

## 亮点与洞察

- 从通道维度切入跨模态特征对齐是一个新颖且有效的视角，此前工作更关注空间维度
- ICE 的三分支旋转设计巧妙地在保持轻量的同时捕获了通道-空间的交叉依赖
- 协方差矩阵分析揭示了哪些通道对模态变化敏感，提供了可解释的特征过滤依据
- 用最优传输替代 top-k 选择，将局部贪心问题提升为全局优化

## 局限与展望

- 仅在室内数据集上评估，未验证对大规模室外场景（如 KITTI）的适用性
- CMCF 中掩码 $M_x$ 的三组分档策略可能不够灵活，可考虑可学习的软掩码
- Sinkhorn 迭代虽然高效，但在极大规模匹配时仍有计算瓶颈
- 作者坦言室内和室外方法通常不能直接迁移，泛化性有待进一步验证

## 相关工作与启发

- 2D3D-MATR 的粗到细框架提供了良好的基线，CA-I2P 在其上做了通道级增强
- LoFTR/GeoTransformer 等检测无关方法启发了 I2P 任务的 detector-free 设计
- 最优传输在 SuperGlue 中已用于 2D 匹配，本文首次引入 I2P 的 patch 级匹配

## 评分

- 新颖性: ⭐⭐⭐⭐ 通道自适应和全局最优选择的组合设计新颖
- 实验充分度: ⭐⭐⭐⭐ 两个基准详细消融，引入了 PIR 新指标
- 写作质量: ⭐⭐⭐ 技术描述尚可但部分符号使用不够统一
- 价值: ⭐⭐⭐⭐ 对跨模态配准问题提出了有效的通道级解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Rethinking 2D-3D Registration: A Novel Network for High-Value Zone Selection and Representation Consistency Alignment](../../CVPR2026/3d_vision/rethinking_2d-3d_registration_a_novel_network_for_high-value_zone_selection_and_.md)
- [\[ICCV 2025\] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)
- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [\[CVPR 2026\] Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs](../../CVPR2026/3d_vision/hg-i2p_bridging_modalities_for_generalizable_image-to-point-cloud_registration_v.md)
- [\[ICML 2026\] FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth](../../ICML2026/3d_vision/fs-i2pa_hierarchical_focus-sweep_registration_network_with_dynamically_allocated.md)

</div>

<!-- RELATED:END -->
