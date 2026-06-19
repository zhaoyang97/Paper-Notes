---
title: >-
  [论文解读] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting
description: >-
  [ICCV 2025][自动驾驶][3D高斯溅射] 提出显式运动分解（EMD）模块，通过可学习运动嵌入和双尺度形变框架为每个 Gaussian 基元建模其运动特性，作为即插即用模块可无缝集成到自监督和监督街景高斯溅射方法中，在 Waymo 和 KITTI 数据集上达到自监督设置的 SOTA 性能。 问题定义 动态街景的新视角…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "3D高斯溅射"
  - "动态场景重建"
  - "运动建模"
  - "自监督"
  - "街景仿真"
---

# EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting

**会议**: ICCV 2025  
**arXiv**: [2411.15582](https://arxiv.org/abs/2411.15582)  
**代码**: [qingpowuwu.github.io/emd](https://qingpowuwu.github.io/emd)  
**领域**: 自动驾驶  
**关键词**: 3D高斯溅射, 动态场景重建, 运动建模, 自监督, 街景仿真

## 一句话总结

提出显式运动分解（EMD）模块，通过可学习运动嵌入和双尺度形变框架为每个 Gaussian 基元建模其运动特性，作为即插即用模块可无缝集成到自监督和监督街景高斯溅射方法中，在 Waymo 和 KITTI 数据集上达到自监督设置的 SOTA 性能。

## 研究背景与动机

### 问题定义

动态街景的新视角合成是闭环自动驾驶仿真的核心技术。基于 3DGS/4DGS 的方法通过将场景分解为静态和动态成分来处理街景重建，但现有方法未能有效建模动态物体的运动模式差异。

### 已有方法的不足

**监督方法**（StreetGaussian、OmniRe）：
- 使用 3D 框监督将场景元素二分为"静态"或"动态"
- 忽略了运动的**连续谱**特性（如行人运动速度远低于车辆）
- 虽然通过优化 3D 框来缓解动态误差，但本质上仍未建模运动模式

**自监督方法**（S3Gaussian、DeSiRe-GS）：
- 通过内在运动线索整体优化 4D 街景表示
- 忽略了物体间运动速度的差异（如行人 vs 车辆）
- 缺乏有效的运动建模机制，导致动态物体重建模糊

### 核心动机

街景中不同类型物体的运动模式存在本质差异：车辆做快速全局运动，行人做慢速局部运动。需要一种**显式的运动建模机制**来区分和处理这些不同尺度的运动模式。关键洞察是：通过为每个 Gaussian 赋予可学习的运动嵌入，再用双尺度形变网络分别处理快速全局运动和慢速局部运动，可以显著提升动态场景重建质量。

## 方法详解

### 整体框架

EMD 是一个即插即用的模块，包含两个核心组件：
1. **Motion-aware Feature Encoding**：融合空间、时间和 Gaussian 特定信息的运动感知特征编码
2. **Dual-scale Deformation Framework**：分层处理快速全局运动和慢速局部形变的双尺度形变框架

给定静态 3D Gaussian 基元集 $\mathbb{G} = \{(\mu_k, \mathbf{s}_k, \mathbf{q}_k, \alpha_k, \mathbf{c}_k)\}_{k=1}^K$ 和时间戳 $t$，目标是学习形变场 $\mathcal{D}$ 将每个 Gaussian 的参数从规范状态映射到时刻 $t$ 的形变状态。

### 关键设计

#### 1. **运动感知特征编码（Motion-aware Feature Encoding）**

- **功能**：将 Gaussian 基元的空间位置、时间信息和个体运动特性融合为综合特征表示。
- **核心思路**：

  聚合特征由三部分拼接组成：
  $$\mathbf{F}_{aggr}(\mu, t) = [\mathbf{F}_{pos}(\mu), \mathbf{F}_{temp}(t), \mathbf{F}_{gauss}]$$

  **空间编码** $\mathbf{F}_{pos}$：多频率位置编码（$P=10$ 个频带）
  $$\mathbf{F}_{pos}(\mu) = [\mu, \{\sin(2^i\pi\mu), \cos(2^i\pi\mu)\}_{i=0}^{P-1}]$$

  **自适应时间嵌入** $\mathbf{F}_{temp}$：通过可学习嵌入矩阵 $\mathbf{W} \in \mathbb{R}^{N_{max} \times D}$ 和渐进式时间采样实现：
  $$\mathbf{F}_{temp}(t) = \text{Interp}(\mathbf{W}, t, N(i))$$
  其中 $N(i)$ 从 $N_{min}=30$ 渐进增长到 $N_{max}=150$，训练迭代 $i$ 控制分辨率：
  $$N(i) = N_{min} + (N_{max} - N_{min}) \cdot \min(i, T) / T$$
  $D=4$ 为时间嵌入维度，$T=25000$ 控制渐进采样的持续时间。

  **Gaussian 嵌入** $\mathbf{F}_{gauss}$：每个 Gaussian $k$ 分配可学习隐变量 $\mathbf{z}_k \in \mathbb{R}^M$（$M=32$），编码其个体运动特征。

- **设计动机**：
    - 多频率空间编码捕捉从精细几何到全局结构的多层次信息
    - 渐进式时间采样从粗到细学习时间动态，避免早期训练时过拟合高频时间变化
    - Gaussian 嵌入实现了个体级运动特征表示——属于同一运动物体的 Gaussian 应学到相似的嵌入

#### 2. **双尺度形变框架（Dual-scale Deformation Framework）**

- **功能**：将形变分解为粗尺度（快速全局运动）和细尺度（慢速局部形变）两级。
- **核心思路**：

  $$\mathcal{D}(\mu, t) = \mathcal{D}_{coarse}(\mathbf{F}_{aggr}(\mu, t)) + \mathcal{D}_{fine}(\mathbf{F}_{aggr}(\mu + \Delta\mu_{coarse}, t))$$

  最终形变参数组合两个尺度的预测：
  $$\mu_t = \mu + \Delta\mu_{coarse} + \Delta\mu_{fine}$$
  $$\mathbf{s}_t = \mathbf{s} + \Delta\mathbf{s}_{coarse} + \Delta\mathbf{s}_{fine}$$
  $$\mathbf{q}_t = \mathbf{q} \otimes \Delta\mathbf{q}_{coarse} \otimes \Delta\mathbf{q}_{fine}$$

  关键点：细尺度形变网络的输入使用**经过粗尺度位移后的位置** $\mu + \Delta\mu_{coarse}$ 重新编码空间特征。

- **设计动机**：
    - $\mathcal{D}_{coarse}$ 专注于车辆平移等大尺度运动，提供主要的形变方向
    - $\mathcal{D}_{fine}$ 在粗形变基础上捕捉关节运动等局部细节
    - 先粗后细的级联设计让两个网络各司其职，避免单一网络同时学习大位移和小形变的困难

#### 3. **与现有框架的集成**

- **功能**：将 EMD 无缝集成到自监督和监督方法中。
- **核心思路**：

  **自监督集成**（S3Gaussian、DeSiRe-GS）：
  - 为每个 Gaussian 添加可学习嵌入 $\mathbf{z}_k$
  - 将原始解码器改造为双尺度框架
  - 保留原有的自监督形变设置

  **监督集成**（StreetGaussian、OmniRe）：
  - 在跟踪框优化中引入双尺度：$R'_t = \Delta R_t^f \cdot \Delta R_t^c \cdot R_t$，$T'_t = T_t + (\Delta T_t^c + \Delta T_t^f)$
  - 对 OmniRe 的非刚体 SMPL 模型也做双尺度精炼：$\theta'_t = \Delta\theta_t^f \cdot \Delta\theta_t^c \cdot \theta_t$

- **设计动机**：即插即用设计使 EMD 可以增强任何现有的街景高斯方法，不需要重新设计整个管线。

### 损失函数 / 训练策略

- 使用与基线相同的重建损失（光度损失等）
- 额外引入 **Gaussian 嵌入局部平滑正则化**，鼓励空间相邻的 Gaussian 有相似嵌入：
  $$\mathcal{L}_{\mathbf{z}_k} = \frac{1}{d|\mathcal{U}|}\sum_{i \in \mathcal{U}}\sum_{j \in \text{KNN}_{i;d}} (e^{-\lambda_w\|\mu_j - \mu_i\|_2} \|\mathbf{z}_{k_i} - \mathbf{z}_{k_j}\|_2)$$
  其中 $\lambda_w = 2000$，$d=20$（KNN 邻居数）
- 粗/细形变值正则化：约束形变值接近零，防止过大的形变

## 实验关键数据

### 主实验

自监督设置 — S3Gaussian 对比（Waymo-D32，场景重建）：

| 方法 | Full PSNR↑ | Full SSIM↑ | Full LPIPS↓ | Vehicle PSNR↑ | Vehicle SSIM↑ |
|------|-----------|-----------|-----------|-------------|-------------|
| EmerNeRF | 28.16 | 0.806 | 0.228 | 24.32 | 0.682 |
| 3DGS | 28.47 | 0.876 | 0.136 | 23.26 | 0.716 |
| S3Gaussian | 30.69 | 0.900 | 0.121 | 26.23 | 0.804 |
| **S3Gaussian+EMD** | **32.50** | **0.933** | **0.082** | **29.04** | **0.879** |

自监督设置 — DeSiRe-GS 对比（Waymo，场景重建）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS |
|------|-------|-------|--------|-----|
| PVG | 32.46 | 0.910 | 0.229 | 50 |
| DeSiRe-GS | 33.61 | 0.919 | 0.204 | 36 |
| **DeSiRe-GS+EMD** | **34.15** | **0.948** | **0.130** | 32 |

监督设置 — OmniRe 对比（Waymo，3 前视相机，新视角合成）：

| 方法 | Full PSNR↑ | Human PSNR↑ | Vehicle PSNR↑ |
|------|-----------|-------------|-------------|
| OmniRe | 32.57 | 24.36 | 27.57 |
| **OmniRe+EMD** | **33.89** | **25.97** | **27.82** |

### 消融实验

各组件贡献（Waymo-D32，S3Gaussian 基线）：

| 配置 | Full PSNR↑ | Vehicle PSNR↑ | 说明 |
|------|-----------|-------------|------|
| **完整模型** | **32.50** | **29.04** | — |
| w/o Gaussian 嵌入 | 32.21 (-0.29) | 28.80 | 未捕捉个体运动特征 |
| w/o 时间嵌入 | 32.23 (-0.27) | 28.08 | 时间变化建模不足 |
| w/o 粗尺度形变 | 29.40 (-3.10) | 24.54 | 大位移无法处理 |
| w/o 细尺度形变 | 32.45 (-0.05) | 28.80 | 局部细节丢失 |

新轨迹合成（FID↓，Waymo）：

| 方法 | 0.5m偏移 | 1.0m偏移 | 1.5m偏移 |
|------|---------|---------|---------|
| S3Gaussian | 83.48 | 110.11 | 134.38 |
| **S3Gaussian+EMD** | **45.11** | **70.26** | **90.20** |

### 关键发现

1. **EMD 对车辆区域提升最显著**：Vehicle PSNR 从 26.23 提升至 29.04（+2.81 dB），说明运动建模直接改善动态物体重建
2. **粗尺度形变是最关键组件**：移除后 PSNR 从 32.50 暴降至 29.40（-3.10），证明大尺度运动建模是核心
3. **新轨迹合成提升巨大**：FID 从 83.48 降至 45.11（0.5m 偏移），说明 EMD 的运动建模改善了车道变换等仿真场景
4. **即插即用的通用性**：在 S3Gaussian、DeSiRe-GS、StreetGaussian、OmniRe 四种方法上都有提升
5. **对人体建模也有效**：OmniRe+EMD 的 Human PSNR 从 24.36 提升至 25.97，说明双尺度框架也适用于非刚体运动
6. **FPS 损失可接受**：DeSiRe-GS+EMD 为 32 FPS（原始 36 FPS），速度损失仅 11%

## 亮点与洞察

1. **运动连续谱的洞察**：街景中物体的运动不是简单的"动/静"二分，而是存在从静止到高速的连续谱
2. **双尺度设计的优雅性**：先粗后细的级联设计自然对应了大位移（车辆）和小形变（行人、车轮旋转）
3. **Gaussian 嵌入 + 平滑正则化**：使同一物体的 Gaussian 自动聚类为相似运动模式，无需显式物体分割
4. **渐进式时间采样**：避免了高频时间信号对早期训练的干扰，是一种简单有效的课程学习策略
5. **即插即用的工程实用性**：只需添加运动嵌入和替换解码器，即可提升现有方法

## 局限与展望

1. **计算开销增加**：双尺度形变网络和嵌入平滑正则化增加了训练时间
2. **未显式建模物体关系**：每个 Gaussian 独立建模运动，未考虑物体间的运动关联
3. **超参数固定**：$N_{min}$、$N_{max}$、$T$ 等渐进采样参数未做场景自适应
4. **仅关注外观重建**：未将运动建模用于物体检测或跟踪等下游任务
5. **缺少多于 3 相机的实验**：Waymo 支持 5 相机，但实验主要在 1-3 相机设置下进行

## 相关工作与启发

- 与 **S3Gaussian** 的区别：S3Gaussian 整体优化场景但不区分运动模式，EMD 通过显式运动嵌入补充了这一缺失
- 与 **StreetGaussian** 的区别：StreetGaussian 通过 3D 框优化处理动态误差，EMD 在此基础上进一步精化运动建模
- **Gaussian 嵌入的思路**类似于 NeRF 中的 latent code，但应用于运动建模而非仅外观
- 双尺度形变的思路可能启发其他需要处理多尺度运动的场景（如室内机器人、体育场景等）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 显式运动建模的思路新颖，但双尺度形变和嵌入的技术手段相对常见
- **实验充分度**: ⭐⭐⭐⭐⭐ — 四种基线方法、两个数据集、自监督+监督两种设置、新轨迹合成评估
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，图表丰富，可视化效果有说服力
- **价值**: ⭐⭐⭐⭐ — 即插即用设计实用性极强，为街景重建提供了运动建模的标准组件

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad_gs_object_aware_bspline_gaussian_splatting_self_supervised_autonomous_driving.md)
- [\[ICCV 2025\] DAMap: Distance-aware MapNet for High Quality HD Map Construction](damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)

</div>

<!-- RELATED:END -->
