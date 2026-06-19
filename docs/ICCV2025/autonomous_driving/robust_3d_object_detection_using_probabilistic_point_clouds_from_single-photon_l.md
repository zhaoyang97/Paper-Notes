---
title: >-
  [论文解读] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs
description: >-
  [ICCV2025][自动驾驶][概率点云] 提出概率点云(PPC)表示——将单光子LiDAR原始时间直方图中的测量置信度作为概率属性附加到每个3D点上，配合轻量级NPD滤波和FPPS采样方法，实现低信噪比(SBR)下鲁棒的3D目标检测，在SUN RGB-D和KITTI上大幅超越点云去噪基线，且几乎不增加计算开销。
tags:
  - "ICCV2025"
  - "自动驾驶"
  - "概率点云"
  - "单光子LiDAR"
  - "3D目标检测"
  - "SPAD"
  - "传感器不确定性传播"
  - "噪声鲁棒性"
---

# Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs

**会议**: ICCV2025  
**arXiv**: [2508.00169](https://arxiv.org/abs/2508.00169)  
**代码**: [项目页面](https://bhavyagoyal.github.io/ppc)  
**领域**: 自动驾驶  
**关键词**: 概率点云, 单光子LiDAR, 3D目标检测, SPAD, 传感器不确定性传播, 噪声鲁棒性

## 一句话总结
提出概率点云(PPC)表示——将单光子LiDAR原始时间直方图中的测量置信度作为概率属性附加到每个3D点上，配合轻量级NPD滤波和FPPS采样方法，实现低信噪比(SBR)下鲁棒的3D目标检测，在SUN RGB-D和KITTI上大幅超越点云去噪基线，且几乎不增加计算开销。

## 研究背景与动机

**LiDAR在挑战场景下的问题**：现代LiDAR(尤其SPAD单光子LiDAR)通过时间直方图估计场景深度。理想条件下直方图有清晰峰值→深度估计准确。但在远距离物体、低反射率表面、强环境光等挑战场景下，信号峰被背景噪声淹没→深度估计严重失真→点云稀疏且含大量噪声点。

**传统处理管线的根本缺陷**：传统管线对原始直方图做阈值滤波→要么过度滤波丢失真实场景点(如远处行人)，要么保留过多噪声点。关键问题是：**从原始传感器数据到点云的转换过程中，不确定性信息被完全丢弃**。现有点云数据集(KITTI等)已经是滤波后的"干净"数据，掩盖了这个问题。

**现有解决方案不足**：
   - 点云去噪网络(PointCleanNet/Score-Denoising等)：假设局部各向同性高斯噪声，但LiDAR噪声是各向异性的(沿相机射线方向分布)→去噪效果有限
   - 直方图去噪(3D-CNN等)：需要完整直方图读出→带宽代价极高(~10s GB/s)，不适合实时应用
   - 两类方法都增加显著计算开销

**核心洞察**：原始SPAD直方图编码了丰富的场景信息，传统管线只用峰值位置估计深度→丢弃了大量有用信息。论文提出**不丢弃不确定性，而是将其传播到下游推理**。

## 方法详解

### 概率点云(PPC)表示
- 对每个LiDAR像素(i,j)的时间直方图，定义点概率：

$$Pr(p^{ij}) = \frac{h_{i,j}[m]}{\sum_{n=1}^{N} h_{i,j}[n]}, \quad m = \arg\max_n h_{i,j}[n]$$

- 含义：峰值bin的光子计数占总光子计数的比例→高SBR时接近1，低SBR时趋近于均匀分布值1/N
- 将此概率附加为每个3D点的额外属性→形成概率点云(PPC)
- **计算极轻量**：仅需峰值查找和归一化，可在传感器芯片上完成

### NPD滤波(Neighbor Probability Density)
- 核心观察：噪声点通常概率低**且**空间密度低；真实场景点即使概率低，也有高局部密度(同一表面邻近点多)
- NPD分数计算：

$$NPD(p_i) = \frac{\sum_{p_j \in \mathcal{BQ}_{L,r}(p_i)} Pr(p_j)}{L}$$

- $\mathcal{BQ}_{L,r}$：半径r内最多L个邻居的球查询
- 巧妙归一化设计：邻居数>L的密集点取平均概率；邻居数<L的稀疏点被额外惩罚(分母仍为L)
- 阈值α=0.003即可滤掉大量噪声点而保留绝大多数真实点
- **不依赖任何物体/表面假设**，纯统计方法

### FPPS(Farthest Probable Point Sampling)
- 问题：PointNet++等网络的FPS(最远点采样)在噪声环境下失效——噪声点分布在远离物体的位置→被FPS优先采样→采到大量噪声关键点
- 解决：先构建概率>β(=0.01)的高置信候选集→在候选集上做FPS
- 关键设计：低概率点**不被丢弃**，仍保留在点云中参与特征聚合→只是不作为采样中心
- 仅适用于含FPS操作的网络(PointNet++/Point Transformer等)

### PPC直接集成到模型(进阶)
以VoteNet为例，探索三种集成方式：
- **(A) 概率作为点属性输入**：让网络学习利用概率特征
- **(B) 概率加权特征向量**：PointNet++逐点特征×邻域平均概率→高置信点特征被放大
- **(C) 概率加权objectness分数**：proposal的objectness×框内点平均概率→高置信proposal优先
- 三者联合额外提升~2% AP@25(在已强的PPC基线之上)

## 实验关键数据

### SUN RGB-D室内检测(VoteNet, AP@25/AP@50)

| 方法 | Clean | SBR=0.1 | SBR=0.05 | SBR=0.02 | SBR=0.01 |
|------|-------|---------|----------|----------|----------|
| Matched Filtering | 51.3/27.5 | 42.4/20.5 | 38.8/17.6 | 17.0/5.1 | 11.3/2.7 |
| Thresholding | 57.1/33.2 | 51.3/28.6 | 46.4/24.9 | 29.6/14.8 | 16.5/6.5 |
| PointCleanNet | 54.6/31.9 | 45.7/26.4 | 40.2/19.2 | 18.2/8.1 | 12.8/3.0 |
| Score Denoising | 57.4/34.0 | 53.2/29.5 | 48.6/25.8 | 26.4/13.7 | 14.6/4.7 |
| **PPC(Ours)** | **58.6/35.0** | **54.3/31.2** | **52.5/30.2** | **38.5/16.5** | **29.4/13.2** |

- **SBR=0.01极低信噪比下PPC vs Score Denoising：AP@25提升14.9%绝对值(29.4 vs 14.6)**
- **SBR=0.02下AP@50提升4.4%(30.2 vs 25.8)**

### KITTI户外检测(PV-RCNN, mAP Moderate)

| SBR | Car/Ped/Cyc(基线) | Car/Ped/Cyc(PPC) |
|-----|-------------------|-----------------|
| 0.05 | 73.1/55.8/61.8 | 73.0/**59.1**/**64.1** |
| 0.02 | 68.2/50.0/52.9 | 68.4/**59.0**/**53.2** |
| 0.01 | 60.0/47.1/43.7 | 60.3/**55.4**/**47.8** |
| 0.005 | 50.7/37.0/35.0 | 51.3/**49.5**/**36.4** |

- **行人类别在SBR=0.005下提升12.5%(49.5 vs 37.0)**——对小目标/远距目标增益最大

### 计算效率

| 方法 | 推理时间(ms/scene) |
|------|------------------|
| Matched Filtering | 87 |
| PointCleanNet | 755 |
| Score Denoising | **1345** |
| PathNet | 867 |
| **PPC(Ours)** | **95** |

- PPC仅增加8ms(87→95)，去噪网络增加10-15倍开销

### 真实硬件验证
- 室内：HORIBA FLIMera SPAD相机(192×128, 4096 bins)，200-800 lux环境光
- 户外：Adaps ADS6311商用单光子LiDAR(256×192, 672 bins)
- 基线方法在真实场景下漏检大量远距/小目标→PPC检测到绝大多数物体且bbox准确

## 亮点与洞察

- **"保留不确定性"而非"消除噪声"的哲学**：传统管线试图先去噪再推理，本文反其道→将不确定性作为有用信号传递给下游模型。这一思路启发了从传感器到推理的端到端设计。
- **物理意义明确的置信度**：概率定义直接来自光子统计model→不是学出来的黑箱特征，而是有明确物理含义的信噪比指标。
- **即插即用设计**：NPD滤波和FPPS不修改任何网络架构→作为预处理模块可接入任意3D检测器(VoteNet/PV-RCNN/ImVoteNet/Uni3DETR/PointPillars均验证)。
- **极致的效率-精度权衡**：95ms vs 1345ms(Score Denoising)，同时精度还更高→实际可部署。
- **数据带宽友好**：PPC仅需每像素1个额外标量(概率)→~10s MB/s，远小于完整直方图读出的~10s GB/s。

## 局限与展望

1. **概率定义过于简单**：仅用峰值bin比例→忽略了直方图形状信息(如多峰、峰值宽度)。论文提及Cramér-Rao不确定性估计是未来方向。
2. **阈值超参依赖**：NPD的α和FPPS的β需要手动选择→不同传感器/场景可能需要不同值。
3. **仅验证SPAD LiDAR**：论文讨论了可扩展到立体视觉/结构光/iToF等其他深度传感器，但未实验验证。
4. **SBR极低时仍有瓶颈**：SBR=0.01时AP@25仍仅29.4(Clean为58.6)→下降50%，说明PPC也有上限。
5. **训练策略**：当前用所有SBR级别联合训练→针对特定SBR的专门模型可能更优。

## 相关工作与启发

- **VoteNet/PointNet++**：PPC的验证主体→证明了不确定性传播对经典点云网络的价值
- **PV-RCNN**：体素化+点特征融合→PPC在体素化管线中同样有效
- **Score-based Point Cloud Denoising**(Luo & Hu, 2021)：当前SOTA去噪→但假设局部高斯噪声不适合LiDAR真实噪声
- **Single-Photon 3D Imaging**(Lindell et al., 2018; Heide et al., 2018)：从直方图重建3D→PPC是互补的(可在重建后再叠加PPC)
- **启发**：任何传感器到推理的管线，都应考虑不确定性传播而非简单二值滤波。这一思想可扩展到雷达、超声等其他传感器模态。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将SPAD传感器不确定性传播到3D推理，PPC概念简洁有力
- 实验充分度: ⭐⭐⭐⭐⭐ 仿真+真实硬件、室内+户外、5种检测器、5种SBR等级、多种基线
- 写作质量: ⭐⭐⭐⭐ 从传感器物理到推理方法的叙事流畅，图示清晰
- 价值: ⭐⭐⭐⭐ 对自动驾驶/机器人的恶劣条件3D感知有实际意义，方法可直接部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Free-running vs. Synchronous: Single-Photon Lidar for High-flux 3D Imaging](free-running_vs_synchronous_single-photon_lidar_for_high-flux_3d_imaging.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](../../ECCV2024/autonomous_driving/graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [\[AAAI 2026\] DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving](../../AAAI2026/autonomous_driving/driveflow_rectified_flow_adaptation_for_robust_3d_object_detection_in_autonomous.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](../../CVPR2025/autonomous_driving/reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)

</div>

<!-- RELATED:END -->
