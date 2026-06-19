---
title: >-
  [论文解读] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation
description: >-
  [ICCV 2025][3D视觉][3D anomaly detection] 提出 PASDF 框架，通过姿态感知的签名距离函数（SDF）实现连续几何表征，统一了3D异常检测与修复任务，在 Real3D-AD 和 Anomaly-ShapeNet 上取得 SOTA。 1. 3D异常检测的重要性：在制造业质量控制、机器人操作…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D anomaly detection"
  - "signed distance function"
  - "pose alignment"
  - "anomaly repair"
  - "点云"
---

# Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation

**会议**: ICCV 2025  
**arXiv**: [2505.24431](https://arxiv.org/abs/2505.24431)  
**代码**: [https://github.com/ZZZBBBZZZ/PASDF](https://github.com/ZZZBBBZZZ/PASDF)  
**领域**: 3D视觉 / 异常检测 / 点云  
**关键词**: 3D anomaly detection, signed distance function, pose alignment, anomaly repair, point cloud

## 一句话总结

提出 PASDF 框架，通过姿态感知的签名距离函数（SDF）实现连续几何表征，统一了3D异常检测与修复任务，在 Real3D-AD 和 Anomaly-ShapeNet 上取得 SOTA。

## 研究背景与动机

1. **3D异常检测的重要性**：在制造业质量控制、机器人操作等领域，即使微小的3D异常（缺失特征、变形、形状不规则）也可能导致整个组件失效，因此需要鲁棒的3D异常检测技术。

2. **现有方法的局限性**：
    - **体素方法**：离散化导致精细几何细节丢失，且存在立方级内存增长问题
    - **点云方法**：采样稀疏导致密度不一致和表面覆盖不完整
    - **投影方法**：遮挡区域信息丢失和视角相关的畸变
    - 这些方法本质上都是离散表征，会引入量化伪影，不利于细粒度异常定位

3. **从检测到修复的需求**：在3D打印和先进制造中，检测异常只是第一步，原位修复同样重要。传统方法依赖间接特征映射，无法提供显式形状重建来指导修复。即使是重建方法（如 IMRNet、R3D-AD），由于依赖离散点云表征，也无法生成连续、高保真的修复模板。

4. **核心动机**：利用 SDF 的连续性和平滑性来桥接异常检测与修复两个任务，同时通过姿态解耦解决任意姿态下的检测难题。

## 方法详解

### 整体框架

PASDF 包含三个核心阶段：

1. **姿态对齐模块（PAM）**：将输入点云对齐到规范化坐标系，消除姿态变化的影响
2. **SDF 网络**：学习连续的签名距离函数表征，隐式捕获物体形状
3. **异常评分模块**：基于测试样本的 SDF 值偏差计算异常分数

理论建模上，将异常检测建模为评估测试样本符合正常形状分布的似然度。通过在 SE(3) 群上积分来实现姿态不变性，并通过对齐到规范姿态来近似这一积分。

### 关键设计

#### 1. 姿态对齐模块（PAM）

PAM 采用粗到精的两阶段配准策略：

- **粗对齐**：对原始点云进行体素下采样，提取 FPFH（Fast Point Feature Histogram）特征，使用 RANSAC 进行全局粗配准
- **精细对齐**：利用 ICP（Iterative Closest Point）算法在粗对齐基础上进一步精细化
- **迭代优化**：引入 Chamfer Distance 驱动的反馈机制，动态调整损失阈值τ，避免局部最小值。累积变换矩阵迭代更新：$T^{(k)} = T_{icp}^{(k)} \cdot T_{ransac}^{(k)} \cdot T^{(k-1)}$

PAM 的关键参数：损失阈值 τ=0.016，增量 Δτ=0.001，最大迭代次数 K=10。

#### 2. SDF 网络

对齐后的点云通过神经网络参数化的 SDF 表征形状：

- 采样查询点分为表面点（10k）、包围盒内部点（10k）和单位体积点（3k），共 23000 个
- 对查询点坐标进行正弦位置编码 $\gamma(\mathbf{x}_i)$
- 网络架构：8层 MLP + 权重归一化，中间层使用 ReLU 激活和 0.2 dropout，第四层有 skip connection
- SDF 预测值通过截断 L1 损失训练

#### 3. 异常评分计算

- **点级异常分数**：$A(\mathbf{x}_j) = |f_\theta(\mathbf{x}_j)|$，即 SDF 值的绝对值
- **物体级异常分数**：选取 top-K（K=1000）最高异常分数点的均值

#### 4. 异常修复

利用训练好的 SDF 网络隐式表征的"正常"形状流形：
1. 将异常输入通过 PAM 对齐到规范姿态
2. 使用 Marching Cubes 算法提取 SDF 的零等值面
3. 从生成的三角网格上采样点云作为修复结果

### 损失函数 / 训练策略

- **截断 L1 损失**：$\mathcal{L}_{SDF} = \frac{1}{N_q} \sum_{i=1}^{N_q} |\text{clamp}(\hat{s}_i, -d_{max}, d_{max}) - s_i|$
- 截断距离 $d_{max} = 0.1$
- 训练 2000 个 epoch，学习率 $1\times10^{-5}$
- 数据预处理：坐标归一化到 [0,1]³，非流形检测与 Poisson 重建保证水密性

## 实验关键数据

### 主实验

**数据集**：
- Real3D-AD：高分辨率真实世界数据集，12类，每类4个正常训练样本和100个测试样本
- Anomaly-ShapeNet：合成数据集，40类，1600+样本

#### Real3D-AD 结果

| 方法 | O-AUROC ↑ | P-AUROC ↑ |
|------|-----------|-----------|
| BTF(Raw) | 0.635 | 0.571 |
| BTF(FPFH) | 0.603 | 0.733 |
| M3DM(PointMAE) | 0.552 | 0.637 |
| PatchCore(FPFH+Raw) | 0.682 | 0.680 |
| RegAD | 0.704 | 0.705 |
| IMRNet | 0.725 | - |
| Group3AD | 0.751 | - |
| **PASDF (Ours)** | **0.802** | **0.745** |

PASDF 在 O-AUROC 上比 Group3AD 高 5.1%，尤其在 Seahorse (1.000)、Car (0.959)、Fish (0.989) 等类别表现突出。

#### Anomaly-ShapeNet 结果

| 方法 | O-AUROC Mean ↑ | P-AUROC Mean ↑ |
|------|----------------|----------------|
| BTF(Raw) | 0.493 | 0.550 |
| M3DM | 0.552 | 0.616 |
| CPMF | 0.559 | - |
| RegAD | 0.572 | 0.668 |
| IMRNet | 0.661 | 0.650 |
| R3D-AD | 0.749 | - |
| **PASDF (Ours)** | **0.900** | **0.897** |

PASDF 在 40 类中的 37 类取得最佳 O-AUROC，且在多个类别达到 1.000 的完美分数。

### 消融实验

#### PAM 对不同基线方法的增强效果（Anomaly-ShapeNet）

| 方法 | PAM | O-AUROC ↑ | P-AUROC ↑ |
|------|-----|-----------|-----------|
| BTF(FPFH) | ✗ | 0.528 | 0.628 |
| BTF(FPFH) | ✓ | 0.579 | 0.683 |
| PatchCore(FPFH) | ✗ | 0.568 | 0.580 |
| PatchCore(FPFH) | ✓ | 0.814 | 0.867 |
| PatchCore(PointMAE) | ✗ | 0.562 | 0.577 |
| PatchCore(PointMAE) | ✓ | 0.626 | 0.681 |
| PASDF (Full) | ✓ | 0.900 | 0.897 |

PatchCore(FPFH) 引入 PAM 后 O-AUROC 提升 24.6%，P-AUROC 提升 28.7%，效果极为显著。

#### 组件消融（Anomaly-ShapeNet）

| 方法 | O-AUROC ↑ | P-AUROC ↑ |
|------|-----------|-----------|
| w/o RANSAC | 0.711 | 0.739 |
| w/o ICP | 0.727 | 0.836 |
| w/o 迭代优化 | 0.871 | 0.884 |
| w/o 位置编码 | 0.887 | 0.783 |
| PASDF (Full) | 0.900 | 0.897 |

#### 异常修复质量评估

| 方法 | Real3D-AD CD ↓ | Real3D-AD EMD ↓ | Anomaly-ShapeNet CD ↓ | Anomaly-ShapeNet EMD ↓ |
|------|---------|---------|---------|---------|
| w/o PE | 0.0255 | 0.0133 | 0.0575 | 0.0276 |
| with PE | 0.0203 | 0.0110 | 0.0445 | 0.0228 |

### 关键发现

1. **姿态对齐是关键**：移除 RANSAC 导致 O-AUROC 从 0.900 暴跌至 0.711，证明全局粗对齐的必要性
2. **位置编码的双重作用**：移除 PE 后 P-AUROC 下降 11.4%（0.897→0.783），同时修复质量也显著下降
3. **PAM 的通用性**：PAM 作为即插即用模块可显著提升多种基线方法的性能
4. **连续表征的优势**：PASDF 在37/40个类别上超越所有方法，而其他方法在不同类别间表现不稳定

## 亮点与洞察

1. **统一框架**：首次将3D异常检测与修复统一到连续 SDF 表征中，检测和修复共享同一个学习到的形状表征
2. **姿态-形状解耦**：通过 PAM 显式分离姿态和形状，使 SDF 网络专注于内在形状变化
3. **连续 vs 离散**：用连续的 SDF 替代离散的体素/点云/投影表征，避免量化伪影，保留精细几何细节
4. **PAM 的即插即用特性**：不仅服务于 PASDF，还能显著提升其他方法的性能，具有很好的通用性

## 局限性 / 可改进方向

1. **计算开销**：PAM 在初始姿态困难情况下的配准计算较为昂贵，可探索基于学习的或层次化配准方法
2. **单类假设**：当前假设单一正常物体类别，扩展到多类检测将提升实用性
3. **输入质量依赖**：性能受输入点云质量影响，需要增强对噪声和离群点的鲁棒性
4. **缺少上下文信息**：可考虑整合场景上下文信息，提升复杂真实环境中的检测能力
5. **训练效率**：2000 epoch 的训练较长，可探索更高效的训练策略

## 相关工作与启发

- **DeepSDF**：SDF 的神经网络参数化思想来源于 DeepSDF，PASDF 将其创造性地应用于异常检测
- **ICP + RANSAC**：经典配准方法的组合在 PAM 中仍然非常有效
- **Marching Cubes**：经典算法在修复阶段用于从 SDF 提取等值面
- 启发：连续表征（隐式函数）相比离散表征在需要精细几何感知的任务中有天然优势

## 评分

- **新颖性**: ⭐⭐⭐⭐ — SDF 用于3D异常检测+修复的统一框架思路新颖
- **技术质量**: ⭐⭐⭐⭐ — 方法设计完整，消融充分，PAM 的通用性验证令人信服
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集、多个基线、详细消融和定性结果
- **实用价值**: ⭐⭐⭐⭐ — 检测+修复的统一框架在制造业场景中有直接应用价值
- **综合评分**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [\[ICCV 2025\] TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction](tar3d_creating_high-quality_3d_assets_via_next-part_prediction.md)
- [\[ICCV 2025\] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](../../CVPR2025/3d_vision/hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)

</div>

<!-- RELATED:END -->
