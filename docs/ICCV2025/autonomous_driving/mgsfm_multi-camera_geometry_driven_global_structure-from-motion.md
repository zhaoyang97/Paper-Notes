---
title: >-
  [论文解读] MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion
description: >-
  [ICCV 2025][自动驾驶][Structure-from-Motion] 提出 MGSfM，一个面向多相机系统的全局 Structure-from-Motion (SfM) 框架，通过**解耦旋转平均 (DMRA)** 和**混合平移平均 (MGP)** 两个核心模块，充分利用多相机刚性约束，在大规模场景中实现与增量式 SfM 媲美甚至更优的精度，同时速度提升约 10 倍。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "Structure-from-Motion"
  - "多相机系统"
  - "全局运动平均"
  - "旋转平均"
  - "平移平均"
  - "三维重建"
---

# MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion

**会议**: ICCV 2025  
**arXiv**: [2507.03306](https://arxiv.org/abs/2507.03306)  
**代码**: [3dv-casia/MGSfM](https://github.com/3dv-casia/MGSfM)  
**领域**: 自动驾驶  
**关键词**: Structure-from-Motion, 多相机系统, 全局运动平均, 旋转平均, 平移平均, 三维重建, 自动驾驶

## 一句话总结

提出 MGSfM，一个面向多相机系统的全局 Structure-from-Motion (SfM) 框架，通过**解耦旋转平均 (DMRA)** 和**混合平移平均 (MGP)** 两个核心模块，充分利用多相机刚性约束，在大规模场景中实现与增量式 SfM 媲美甚至更优的精度，同时速度提升约 10 倍。

## 研究背景与动机

### 多相机系统在自动驾驶中的重要性

自动驾驶和机器人平台越来越多地采用多相机系统（立体相机、四目环视等）来实现环境感知。这些相机之间的**刚性相对位姿约束**（同一刚体上相机之间的相对位姿在所有帧中保持不变）是一个天然的强几何先验，可以显著提升 SfM 的精度和鲁棒性。

### 现有方法的局限

- **增量式 SfM**（如 COLMAP、MCSfM）：逐帧添加相机和三维点，精度较高但计算量大，且存在**尺度漂移**（scale drift）问题——误差随序列长度单调累积，尤其在大规模场景和缺乏回环检测时特别严重
- **全局 SfM**（如 GLOMAP）：一次性联合估计所有相机位姿，误差分布更均匀，但鲁棒性不足——容易受到外点匹配的影响，尤其是在仅使用 camera-to-point 约束时
- **多相机全局方法**（如 MMA）：利用多相机约束进行平移平均，但仅使用相对平移方向，丢弃了 feature track 中的丰富信息

### 核心问题

如何在全局 SfM 框架中同时利用：(1) 多相机刚性约束来消除尺度歧义；(2) camera-to-camera 和 camera-to-point 两种互补约束来提升鲁棒性？

## 方法详解

MGSfM 基于 GLOMAP 构建，输入 COLMAP 数据库，输出 COLMAP 兼容的稀疏重建结果。核心包含两个阶段：

### 1. 解耦旋转平均（Decoupled Multi-camera Rotation Averaging, DMRA）

传统旋转平均将所有相机的旋转统一求解，但多相机系统中同一刚体单元内部相机的相对旋转应保持一致。DMRA 采用**分层策略**：

1. **刚体内部旋转估计**：首先利用冗余的多帧观测，通过中值旋转平均（Median Rotation Averaging）鲁棒地估计同一刚体单元内各相机之间的相对旋转 $\{R_{ij}^{rig}\}$
2. **全局刚体旋转估计**：将每个多相机单元视为一个刚体节点，利用步骤 1 的结果将所有相机的相对旋转转换为刚体单元之间的相对旋转，再进行标准的全局旋转平均
3. **反投影恢复**：从全局刚体旋转和刚体内部相对旋转，恢复每个相机的全局旋转

这种解耦策略的优势在于：内部旋转估计充分利用了多帧冗余性，对外点鲁棒；全局旋转平均的问题规模大幅缩小（节点数等于刚体单元数而非相机总数）。

### 2. 混合平移平均（Multi-camera Geometry driven Position estimation, MGP）

平移平均是全局 SfM 的核心难点，因为相对平移仅有方向信息（无尺度），而多相机系统提供了天然的尺度约束。MGP 融合两类互补约束：

#### Camera-to-Camera 约束（相对平移）

从两帧之间的本质矩阵分解得到的相对平移方向 $\hat{t}_{ij}$，提供帧间运动的方向约束。在多相机系统中，同一帧的不同相机对之间形成重叠图像对，可以恢复刚体单元间的相对尺度。

#### Camera-to-Point 约束（Feature Tracks）

每条 feature track 将 3D 点与多个相机中心关联，提供了额外的角度约束。Feature tracks 的优势是数量多、覆盖面广，但容易受外点影响。

#### 两阶段优化

1. **初始化阶段**：利用相对平移的 camera-to-camera 约束，通过**凸的距离基目标函数**求解相机位置和 3D 点的初始估计。凸优化保证收敛到全局最优，避免局部极值
2. **精化阶段**：以初始化结果为起点，构造融合 camera-to-camera 和 camera-to-point 约束的**无偏非双线性角度基目标函数**（unbiased non-bilinear angle-based objective），通过非线性优化精化所有相机位置和 3D 点

角度基目标函数的关键特性：

- **无偏性**：不受 3D 点到相机距离的影响，避免近距离点主导优化
- **非双线性**：相比双线性形式，计算中无需引入辅助变量，优化更高效
- 对外点 feature tracks 具有更好的鲁棒性

### 3. 整体流程

1. 特征提取与匹配（COLMAP）
2. DMRA 估计全局旋转
3. MGP 估计全局平移
4. Bundle Adjustment 联合精化

## 实验关键数据

### KITTI Odometry（室外立体相机）

| 方法 | 类型 | 多相机约束 | 关键表现 |
|------|------|-----------|---------|
| COLMAP | 增量 | ✗ | 基线方法，尺度漂移明显 |
| GLOMAP | 全局 | ✗ | 误差分布均匀但整体偏大 |
| MMA | 全局 | ✓ | 仅用相对平移，精度优于 GLOMAP |
| MCSfM | 增量 | ✓ | 精度好但速度慢 |
| **MGSfM** | **全局** | **✓** | **精度最优，速度远超增量方法** |

- 在挑战性序列 08（无完整回环）上，MGSfM 的轨迹最接近 ground truth
- 在序列 01（特征少、外点多）上，仅用 feature tracks 的方法容易陷入局部极值，MGSfM 的混合策略表现鲁棒

### KITTI-360（大规模室外多相机）

- MGSfM 速度约为 MCSfM 的 **10 倍**
- 重建质量优于 COLMAP、GLOMAP 和 MMA
- 内部相机位姿（刚体内部相对旋转和平移）的估计精度与 MCSfM 的 BA 精化结果相当，验证了 DMRA 的鲁棒性

### ETH3D-SLAM（室内场景）

| 场景 | GLOMAP AUC@0.1m | MGSfM AUC@0.1m | GLOMAP 时间(s) | MGSfM 时间(s) |
|------|-----------------|----------------|---------------|--------------|
| ceiling_1 | 18.5 | **59.7** | 240 | **34** |
| desk_3 | 86.4 | **95.8** | 462 | **89** |
| large_loop_1 | 70.0 | **87.9** | 250 | **30** |
| motion_1 | 25.7 | **46.5** | 885 | **158** |
| reflective_1 | 79.1 | **91.3** | 3239 | **335** |
| repetitive | 66.9 | **91.4** | 90 | **23** |

MGSfM 在所有场景上均大幅优于 GLOMAP，精度提升 10-40 个百分点，速度提升 4-10 倍。

### 自采数据集

- **CAMPUS**（29,000+ 图像，520,000 m²）：MGSfM 耗时 66 分钟 vs COLMAP 1588 分钟 vs GLOMAP 580 分钟 vs MCSfM 401 分钟。COLMAP、GLOMAP、MMA 的重建结果错误，MCSfM 有局部错误结构
- **STREET**（四目相机，12,000+ 图像，500,000 m²）：仅 MGSfM 正确重建了道路轨迹

### 消融实验

在 KITTI Odometry 上比较六种配置（仅相对平移 / 仅 feature tracks / 混合，双线性 / 非双线性）：

- 混合策略始终优于单一约束，尤其在外点多的序列上优势明显
- 非双线性目标函数在合理初始化下，鲁棒性和精度优于双线性形式
- MGSfM（Hybrid-Non-Bilinear）在除"仅相对平移"外的所有配置中速度最快，说明良好的初始化对效率至关重要

## 亮点与洞察

1. **分层解耦的旋转平均**：将多相机旋转估计拆分为刚体内部和刚体间两个层次，既利用了多帧冗余性，又减小了全局优化的问题规模，是一个优雅的工程设计
2. **混合约束的平移平均**：将 camera-to-camera（稀疏但鲁棒）和 camera-to-point（密集但易受外点影响）两类约束统一到角度基框架中，通过两阶段优化（凸初始化 + 非线性精化）兼顾鲁棒性和精度
3. **多相机刚性约束消除尺度歧义**：这是多相机系统最重要的几何先验——同一刚体上的多相机提供冗余观测，可以约束相邻帧间的相对尺度，从根本上缓解单目/立体 SfM 的尺度漂移问题
4. **实际工程价值高**：基于 GLOMAP 构建，输入输出兼容 COLMAP 生态，代码已开源，支持单相机和多相机配置

## 局限与展望

1. **仅支持单一刚体配置（single rig）**：目前假设所有帧来自同一多相机系统，不支持多种不同相机配置混合的场景（作者在 GitHub 中提到 multi-rig 支持是未来计划）
2. **依赖 COLMAP 的特征匹配**：特征提取和匹配仍使用传统 COLMAP 流程，未集成学习型特征（如 SuperPoint + LightGlue），在纹理贫乏或重复纹理场景下匹配质量仍是瓶颈
3. **序列式图像假设**：主要针对有时序关系的序列图像设计（如自动驾驶行车序列），对无序图像集合的适用性未充分验证
4. **室内小场景提升空间**：虽然在 ETH3D-SLAM 上有不错表现，但多相机刚性约束的优势在小场景中不如大规模场景明显
5. **缺少与端到端学习方法的对比**：未与近年基于深度学习的 SfM 方法（如 DUSt3R、MASt3R）进行比较

## 相关工作与启发

- **COLMAP** [Schönberger & Frahm, CVPR 2016]：增量式 SfM 的标准方法，精度高但速度慢
- **GLOMAP** [Pan et al., ECCV 2024]：全局 SfM，MGSfM 的代码基础
- **MMA** [Cui et al.]：利用多相机几何的全局平移平均，启发了 MGSfM 的混合策略
- **MCSfM**：增量式多相机 SfM，在重建质量上是 MGSfM 的主要对比对象
- **HETA** [Tao et al., CVPR 2024]：作者前序工作，利用 feature tracks 重新审视全局平移估计

**对研究的启发**：多相机系统的刚性约束思路可推广到其他多传感器融合场景；两阶段优化（凸初始化 + 非凸精化）是处理非凸优化问题的通用范式。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 分层解耦旋转 + 混合平移的全局框架设计有新意
- 实验充分度: ⭐⭐⭐⭐⭐ — 多个数据集、消融实验、定量定性全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐⭐ — 代码开源，兼容 COLMAP 生态，直接可用于自动驾驶三维重建

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration](../../ICML2025/autonomous_driving/geometry-to-image_synthesis-driven_generative_point_cloud_registration.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)
- [\[ICCV 2025\] Language Driven Occupancy Prediction (LOcc)](language_driven_occupancy_prediction.md)
- [\[ICCV 2025\] SAM4D: Segment Anything in Camera and LiDAR Streams](sam4d_segment_anything_in_camera_and_lidar_streams.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)

</div>

<!-- RELATED:END -->
