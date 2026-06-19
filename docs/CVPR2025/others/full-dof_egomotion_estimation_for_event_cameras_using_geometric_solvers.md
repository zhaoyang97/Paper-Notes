---
title: >-
  [论文解读] Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers
description: >-
  [CVPR 2025][事件相机] 提出首个仅用事件流估计完整6-DoF自运动（角速度+线速度）的几何求解器方法，通过建立事件扇形流形上的线段几何约束——入射关系和新颖的共面关系，设计最少仅需8个事件的稀疏求解器，无需IMU即可解耦旋转和平移估计。 领域现状 领域现状：事件相机因高时间分辨率和高动态范围在机器人导航中广泛应用…
tags:
  - "CVPR 2025"
  - "事件相机"
  - "自运动估计"
  - "几何求解器"
  - "6-DoF位姿"
  - "共面约束"
---

# Full-DoF Egomotion Estimation for Event Cameras Using Geometric Solvers

**会议**: CVPR 2025  
**arXiv**: [2503.03307](https://arxiv.org/abs/2503.03307)  
**代码**: [https://github.com/jizhaox/relpose-event](https://github.com/jizhaox/relpose-event)  
**领域**: 其他  
**关键词**: 事件相机、自运动估计、几何求解器、6-DoF位姿、共面约束

## 一句话总结
提出首个仅用事件流估计完整6-DoF自运动（角速度+线速度）的几何求解器方法，通过建立事件扇形流形上的线段几何约束——入射关系和新颖的共面关系，设计最少仅需8个事件的稀疏求解器，无需IMU即可解耦旋转和平移估计。

## 研究背景与动机

### 领域现状

**领域现状**：事件相机因高时间分辨率和高动态范围在机器人导航中广泛应用。现有事件相机运动估计方法大多假设旋转位移已知（由IMU提供），只估计平移部分，或者仅估计旋转。

**现有痛点**：（1）依赖IMU提供旋转先验限制了系统的独立性和轻量化；（2）仅估计平移或仅估计旋转无法满足完整6-DoF运动估计的需求；（3）现有方法缺乏从纯事件流恢复完整运动的理论基础。

**核心矛盾**：事件流（异步像素级亮度变化）与传统帧不同，无法直接使用经典的对极几何方法。如何从事件流的时空结构中提取6-DoF运动信号是开放问题。

**本文目标** 仅从事件流中恢复完整6-DoF自运动（3个角速度+3个线速度分量），无需IMU或其他外部传感器。

**切入角度**：利用事件扇形流形（eventail manifold）上的线段几何——每个事件在时空中形成射线，一组共线事件构成线段。通过线段间的入射关系（共享点）和共面关系（共享平面法向量）建立运动约束方程。

**核心 idea**：将事件流建模为时空线段集合，用线段间的入射关系和共面关系建立方程组，用最少8个事件的几何求解器恢复完整6-DoF自运动。

## 方法详解

### 整体框架
输入事件流$\{(x_i, y_i, t_i, p_i)\}$，在时空中构建eventail流形上的线段。通过两种几何约束建立关于运动参数$(\omega_x, \omega_y, \omega_z, v_x, v_y, v_z)$的方程组：（1）入射关系（线上点约束）建立的线性方程；（2）共面关系（法向量约束）建立的双线性方程。使用Adam优化器和一阶旋转近似高效求解。特殊处理纯旋转退化情况。

### 关键设计

1. **Eventail流形几何建模**:

    - 功能：将事件流转化为可利用的几何结构
    - 核心思路：每个事件$(x, y, t)$在3D时空中定义一条射线，同一边缘上的连续事件形成线段。这些线段的方向与像素速度成正比，像素速度由场景深度和自运动决定。通过聚合多个事件形成可靠的线段方向估计
    - 设计动机：单个事件信息太少（仅一个亮度变化触发），但时空中的线段结构编码了运动信息

2. **入射关系求解器（Incidence Relation）**:

    - 功能：通过线上点约束建立线性方程
    - 核心思路：如果一个事件点落在某条线段上，其坐标满足线段方程。将运动参数代入线段方向表达式后得到关于$(\omega, v)$的线性约束。最少需要8个约束（8个事件-线段对）构成线性方程组Ax=0
    - 设计动机：线性方程求解快速稳定，是最小求解器的首选

3. **共面关系求解器（Coplanarity Relation）**:

    - 功能：通过法向量约束提供额外约束
    - 核心思路：如果两条线段共面，它们的方向向量和连接向量满足三重积为零的约束。这给出了关于运动参数的双线性方程$n_1^T \cdot d_{12} = 0$，其中$n_1$是法向量，$d_{12}$是两线段间的连接。这种约束不需要共享点，适用于空间中不相交的线段
    - 设计动机：入射关系需要线段和点有明确的归属关系，共面关系更灵活——任意两条线段都可以产生约束

### 损失函数 / 训练策略
非学习方法，使用Adam优化器最小化几何残差。一阶旋转近似$R \approx I + [\omega]_\times \Delta t$简化非线性方程。对纯旋转退化情况有专门的理论分析和处理。

## 实验关键数据

### 主实验

在 VECtor 数据集的真实事件相机序列上评估，分割为 0.3 秒非重叠区间：

| 序列 | IncBat $\varepsilon_{ang}$ | IncBat $\varepsilon_{lin}$(°) | CopBat $\varepsilon_{ang}$ | CopBat $\varepsilon_{lin}$(°) |
|------|-----------|-----------|-----------|-----------|
| desk-normal | 0.232 | 23.0 | 0.236 | 25.1 |
| mountain-normal | 0.195 | 17.5 | 0.216 | 18.7 |
| sofa-normal | 0.229 | 21.1 | 0.221 | 20.6 |

其中 $\varepsilon_{ang} \in [0,1]$（越小越好），$\varepsilon_{lin}$ 为角度误差（°）。

### 消融实验

运行时间与数值稳定性（M=5线段，N=100事件/线段，无噪声合成数据，1000场景）：

| 求解器 | 旋转参数化 | SR1 (阈值0.01) | SR2 (阈值0.05) | 中位运行时间 |
|--------|-----------|---------------|---------------|-------------|
| IncBat | +cascad | 97.3% | — | 17.1 ms |
| CopBat | +cascad | — | — | 16.7~48.7 ms |
| IncBat | +exact | 较低 | — | 较慢 |
| IncBat | +approx | 中等 | — | 较快 |

关键变量分析（M=10, N=8~1000，pixel噪声0.5pix，时间戳抖动0.5ms）：
- 事件数<100时IncBat优于CopBat；事件数增大后两者趋于一致
- 线段数从1增至50时误差显著下降；M=1时旋转-平移模糊导致所有方法失败
- 噪声增大时误差单调增大；无噪声时求解器几乎100%成功

### 关键发现
- **cascade旋转参数化**效果最好——先用一阶近似快速初始化，再用精确参数化精调，兼顾效率和精度
- **共面关系**在线段不相交场景中提供了关键补充约束，是理论创新点
- 纯旋转是退化情况（平移不可估计），实际部署需要运动分类检测
- 实际误差水平（$\varepsilon_{ang} \approx 0.2$, $\varepsilon_{lin} \approx 20°$）足以集成到VIO/SLAM管道中

## 亮点与洞察
- **理论贡献突出**：首次证明仅用事件流可以恢复完整6-DoF运动，并给出了最少事件数的理论下界（8个）
- **共面关系的新颖性**：传统线段几何主要用入射关系；共面关系不需要线段显式相交，大幅扩展了可用约束的数量
- **无需外部传感器**：去除了IMU依赖，使纯事件相机的运动估计成为可能

## 局限与展望
- 一阶旋转近似仅在小运动下有效，快速旋转场景需要高阶展开
- 实际事件流中的噪声会影响线段方向估计的精度
- 缺少与基于学习的事件相机运动估计方法的定量对比
- 计算效率分析不够详细

## 相关工作与启发
- **vs CMax方法**: CMax系列仅估计旋转；本文同时估计旋转和平移
- **vs 传统帧方法（五点法等）**: 帧方法需要特征匹配；本文直接从事件时空结构提取运动
- **vs 事件+IMU融合**: 去除了IMU依赖，使系统更轻量

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个纯事件6-DoF求解器，共面关系的理论贡献重大
- 实验充分度: ⭐⭐⭐ 理论严谨但实验规模偏小，缺少大规模定量对比
- 写作质量: ⭐⭐⭐⭐ 几何推导清晰，理论框架完整
- 价值: ⭐⭐⭐⭐ 对事件相机SLAM和自主导航有基础性贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Event Ellipsometer: Event-based Mueller-Matrix Video Imaging](event_ellipsometer_event-based_mueller-matrix_video_imaging.md)
- [\[CVPR 2025\] EBS-EKF: Accurate and High Frequency Event-based Star Tracking](ebs-ekf_accurate_and_high_frequency_event-based_star_tracking.md)
- [\[CVPR 2025\] Order-One Rolling Shutter Cameras](order-one_rolling_shutter_cameras.md)
- [\[CVPR 2025\] Practical Solutions to the Relative Pose of Three Calibrated Cameras](practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)
- [\[AAAI 2026\] Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras](../../AAAI2026/others/spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)

</div>

<!-- RELATED:END -->
