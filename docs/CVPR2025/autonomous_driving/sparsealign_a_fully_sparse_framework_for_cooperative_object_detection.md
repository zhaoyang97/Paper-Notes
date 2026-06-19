---
title: >-
  [论文解读] SparseAlign: A Fully Sparse Framework for Cooperative Object Detection
description: >-
  [CVPR 2025][自动驾驶][协同感知] SparseAlign提出首个全稀疏的协同目标检测框架，通过坐标可扩展稀疏卷积解决中心特征缺失和孤立卷积域问题，在减少98%通信带宽的同时超越基于稠密BEV的SOTA方法。 协同感知通过多智能体共享感知信息来扩大视野、减少遮挡，是自动驾驶安全的关键。现有协同目标检测方法主要在稠…
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "协同感知"
  - "稀疏检测"
  - "LiDAR"
  - "目标检测"
  - "通信效率"
---

# SparseAlign: A Fully Sparse Framework for Cooperative Object Detection

**会议**: CVPR 2025  
**arXiv**: [2503.12982](https://arxiv.org/abs/2503.12982)  
**代码**: 无  
**领域**: Autonomous Driving  
**关键词**: 协同感知, 稀疏检测, LiDAR, 目标检测, 通信效率

## 一句话总结

SparseAlign提出首个全稀疏的协同目标检测框架，通过坐标可扩展稀疏卷积解决中心特征缺失和孤立卷积域问题，在减少98%通信带宽的同时超越基于稠密BEV的SOTA方法。

## 研究背景与动机

协同感知通过多智能体共享感知信息来扩大视野、减少遮挡，是自动驾驶安全的关键。现有协同目标检测方法主要在稠密BEV特征图上操作，存在两个核心问题：

- **计算复杂度随感知范围二次增长**：BEV特征图大小与感知距离成正比，远距检测计算代价高
- **通信带宽消耗大**：共享稠密BEV特征图需要大量传输资源

全稀疏框架利用点云的稀疏性，计算复杂度仅与点数线性相关，但构建竞争力的全稀疏框架面临两个技术挑战：

- **中心特征缺失（CFM）**：LiDAR扫描产生的点在物体中心区域通常缺失，而中心点对目标表示最为关键
- **孤立卷积域（ICF）**：远距区域不同激光束扫描的点之间连通性差，导致体素块相互孤立，感受野无法扩展

## 方法详解

### 整体框架

SparseAlign由增强的稀疏3D骨干（SUNet）、基于query的时序上下文学习模块（TAM）、位姿对齐模块（PAM）和空间对齐模块（SAM）组成。所有智能体共享网络权重，以广播方式共享Object Query特征作为CPM。

### 关键设计

**1. 坐标可扩展稀疏卷积（CEC）解决CFM+ICF**

- **功能**：同时解决中心特征缺失和孤立卷积域两个问题，构建有效的全稀疏3D骨干
- **核心思路**：在3D稀疏卷积的 $4\times$ 和 $8\times$ 降采样层使用CEC扩展体素连通性，增大感受野覆盖。在2D BEV稀疏特征上使用CEC扩展坐标，确保所有被扫描物体的中心位置都有特征覆盖
- **设计动机**：标准稀疏卷积的感受野只覆盖单个车辆的LiDAR点（如图3c），CEC扩展后可覆盖相邻车辆的点（如图3d），使得遮挡物体和远距物体也能聚合邻域信息

**2. 位姿无关的邻域图特征匹配（PAM）**

- **功能**：纠正协同智能体间的相对位姿误差，不依赖初始位姿精度
- **核心思路**：为每个检测框嵌入其K=8个最近邻的**相对几何特征**（相对方向 $\nu_a$、相对边缘方向 $\epsilon_a$、欧氏距离 $\epsilon_d$、邻居尺寸 $\nu_{dim}$）。这些特征都是位姿无关的相对量。通过自注意力聚合后用匈牙利算法匹配两个BBox集合，再用PGO优化对齐
- **设计动机**：现有方法需要初始位姿误差较小才能正确匹配，本方法的相对特征本身与全局坐标系无关，即使位姿误差大也能鲁棒匹配

**3. 空间对齐模块（SAM）融合稀疏Query**

- **功能**：将协同智能体的稀疏query特征精确融合到自车坐标系
- **核心思路**：先用MLP条件化旋转矩阵 $R$ 做特征空间变换 $F^c = MLP([F^c; F^R])$；再将旋转后的协同query坐标合并到最近的自车网格点；最后通过K近邻聚合 $Q^c \cup Q^e$ 的特征（含相对位置编码），用mean+max池化生成融合特征
- **设计动机**：稀疏query的坐标在旋转后不对齐网格，需要特殊处理。KNN聚合+位置编码能灵活处理不规则的点位置

### 损失函数

Focal Loss（前背景分类）+ Smooth L1 Loss（BBox回归，含位置偏移、尺寸和CompassRose方向编码）。CompassRose使用4个锚点角度编码方向，确保至少一个锚点能单调到达目标角度。

## 实验关键数据

### 主实验：OPV2V数据集

| 方法 | 通信带宽(Mb)↓ | AP@0.5↑ | AP@0.7↑ |
|------|-----------|---------|---------|
| V2VNet (稠密BEV) | 72.08 | 0.917 | 0.822 |
| CoBEVT (稠密BEV) | 72.08 | 0.927 | 0.830 |
| V2X-ViT (稠密BEV) | 72.08 | 0.926 | 0.844 |
| **SparseAlign** | **~1.5** | **0.935** | **0.860** |

### 消融实验：各模块贡献（OPV2V）

| 组件 | AP@0.7 |
|------|--------|
| MinkUNet baseline | 0.790 |
| + CEC (解决ICF) | 0.825 |
| + CEC (解决CFM) | 0.842 |
| + TAM | 0.850 |
| + PAM + SAM | **0.860** |

### 关键发现

- SparseAlign在**通信带宽减少98%**的情况下仍超越所有稠密BEV方法
- CEC同时解决ICF和CFM分别带来3.5%和1.7%的AP@0.7提升
- 在DairV2X（真实数据集）和时间对齐任务（OPV2Vt/DairV2Xt）上同样SOTA
- CompassRose方向编码相比标准sin/cos编码提升约0.5% AP
- Free Space Augmentation在远距稀疏区域有效缓解ICF问题

## 亮点与洞察

1. **全稀疏框架首次超越稠密BEV方法**在协同感知領域是重要突破，证明了稀疏处理在多智能体场景的可行性和优越性
2. **位姿无关的图匹配**是一个优雅的工程设计——利用拓扑结构而非绝对坐标进行跨智能体匹配
3. 通信效率提升98%对实际V2X部署有重要意义——从72Mb降至1.5Mb使得现有蜂窝网络即可支持

## 局限与展望

- 当前仅处理LiDAR协同检测，未扩展到相机融合或语义分割
- CEC增加了部分计算开销（虽然仍远低于稠密方法）
- 对于极端稀疏（超远距）的匹配鲁棒性有待验证
- 未来可探索自适应CEC扩展策略和更高效的query压缩方案

## 相关工作与启发

- **与V2X-ViT/CoBEVT的关系**：这些方法在稠密BEV上做注意力融合，SparseAlign证明稀疏query也能实现甚至更好的融合
- **与FPVRCNN的关系**：FPVRCNN也用稀疏特征共享，但SparseAlign的CEC backbone和SAM融合更强大
- **启发**：在多智能体协同中，"少而精"的稀疏query比"大而全"的稠密特征图更高效

## 评分

⭐⭐⭐⭐

首个在协同检测中超越稠密BEV的全稀疏框架，通信带宽减少98%。系统性地解决了CFM和ICF两个稀疏backbone的核心问题。PAM的位姿无关匹配设计精巧。对V2X实际部署有重要价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PAP: A Prediction-as-Perception Framework for 3D Object Detection](a_prediction-as-perception_framework_for_3d_object_detection.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)
- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](../../ECCV2024/autonomous_driving/fully_sparse_3d_occupancy_prediction.md)
- [\[CVPR 2025\] Cubify Anything: Scaling Indoor 3D Object Detection](cubify_anything_scaling_indoor_3d_object_detection.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)

</div>

<!-- RELATED:END -->
