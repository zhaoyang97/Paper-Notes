---
title: >-
  [论文解读] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation
description: >-
  [ICCV 2025][视频理解][事件相机] 本文提出 EMoTive，一个基于事件相机的 3D 运动估计框架，通过 Event Kymograph 编码精细时序演化信息，并使用事件密度引导的非均匀 NURBS 参数曲线建模时空轨迹，从轨迹中导出光流和深度运动场，在自建 CarlaEvent3D 数据集和真实世界基准上取得 SOTA 性能。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "事件相机"
  - "3D运动估计"
  - "非均匀参数曲线"
  - "光流"
  - "深度运动"
---

# EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation

**会议**: ICCV 2025  
**arXiv**: [2503.11371](https://arxiv.org/abs/2503.11371)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 事件相机, 3D运动估计, 非均匀参数曲线, 光流, 深度运动

## 一句话总结
本文提出 EMoTive，一个基于事件相机的 3D 运动估计框架，通过 Event Kymograph 编码精细时序演化信息，并使用事件密度引导的非均匀 NURBS 参数曲线建模时空轨迹，从轨迹中导出光流和深度运动场，在自建 CarlaEvent3D 数据集和真实世界基准上取得 SOTA 性能。

## 研究背景与动机
视觉 3D 运动估计（推断 2D 像素在 3D 空间中的运动）是空间智能的核心能力。然而，现有方法面临一个根本性矛盾：

**核心痛点：深度变化导致的时空运动不一致性**。当物体沿深度方向运动时，其在图像空间的投影会发生局部变形（缩放），这破坏了传统光流方法所依赖的**局部空间运动平滑**或**时间运动不变**假设。例如，一辆驶近相机的车辆，其不同部位在图像平面上的运动方向和速度是异质的。

**现有方法的局限**：
- 帧间对应方法（RAFT 等）依赖局部平滑假设，无法处理深度变化引起的运动异质性
- 双空间解耦方法（ScaleFlow 等）试图在特征空间和尺度空间分别估计平面运动和深度运动，但两个空间源自同一像素域，矛盾依然存在
- 传统帧相机受限于固定采样频率，时间观测不足

**事件相机的机遇**：事件相机以微秒级时间分辨率异步报告像素级亮度变化，提供了前所未有的精细时间观测。关键洞察是：事件流可以通过 x-t 和 y-t 解耦投影形成 **Event Kymograph**，以微秒级精度捕获时间演化，从而实现对异质时空运动的非平稳建模。

**核心 idea：用事件引导的非均匀参数曲线建模时空轨迹**，通过轨迹的多时间采样得到光流，通过轨迹的时间梯度推导深度运动。

## 方法详解

### 整体框架
EMoTive 的流程：(1) 将事件流分别投影为 Event Voxel（空间特征）和 Event Kymograph（时间特征）；(2) 构建时空双代价体（dual cost volumes）进行运动表征；(3) 通过密度感知自适应机制融合时空特征，更新 NURBS 曲线控制点；(4) 对轨迹进行多时间采样得到光流，通过时间梯度推导深度运动场。

### 关键设计
1. **Event Kymograph（事件波纹图）**:

    - 功能：将事件流投影到 x-t 和 y-t 解耦平面上，编码精细的时间演化信息
    - 核心思路：与 Event Voxel 使用三角核进行粗时间量化不同，Kymograph 使用连续高斯时间投影核：
    $K_x = \sum_i p_i k(x - x_i) g(t - t_i | \sigma), \quad g(a|\sigma) = \exp(-(a/\sigma)^2)$
      其中 $\sigma$ 控制时间平滑尺度，可以保持 10μs 级别的时间精度
    - 设计动机：传统 Event Voxel 通过时间量化不可避免地丢失精细时间线索。空间轴解耦使得 x 方向和 y 方向的运动可以被独立分析，而连续高斯核保留了精细时间分辨率

2. **时空双代价体（Dual Cost Volumes）**:

    - 功能：在空间和时间两个维度上分别构建代价金字塔，提供多尺度运动匹配信息
    - 核心思路：
        - **空间代价体**：从 Event Voxel 提取空间特征 $f_{hw}$，构建多分辨率内积金字塔
        - **时间代价体**：将 Kymograph 按时间锚点分块，1D 卷积提取时间特征 $f_{ht}, f_{wt}$，跨子块交叉关联后通过张量积融合：$C_t^m(n,i,k,j,l) \doteq C_{ht}^m(n,i,j) \otimes C_{wt}^m(n,k,l)$
    - 设计动机：空间代价体捕获帧间位移信息，时间代价体则利用 Kymograph 的精细时间分辨率捕获运动动态。两者互补，前者关注"在哪"，后者关注"如何变化"

3. **密度感知自适应 NURBS 轨迹（Density-aware Adaptive NURBS）**:

    - 功能：通过事件密度引导非均匀 B 样条曲线的节点分布和权重，自适应建模异质运动
    - 核心思路：轨迹定义为 NURBS 曲线：
    $\mathcal{T}(t,x,y) = \frac{\sum_i^n N_{i,p}(t) w_i \mathbf{P}_i(x,y)}{\sum_i^n N_{i,p}(t) w_i}$
      密度自适应三阶段：
        - 从 Kymograph 计算时空密度分布 $D_s$
        - 提取 top-n 时间索引作为关键参数
        - 通过滑动窗口平均计算可调节点，通过密度归一化得到权重
    - 设计动机：均匀 B 样条假设运动在时间上均匀分布，无法表达加速、减速等异质运动。事件密度天然反映运动强度——高密度区域意味着激烈运动，应配置更多节点和更高权重以增强曲线表达力

4. **从轨迹到 3D 运动**:

    - 功能：通过轨迹的多时间采样和时间梯度分析，统一获取光流和深度运动
    - 核心思路：
        - 光流：$\mathcal{O}(x,y) = \mathcal{T}(\tau, x, y)$，直接查询轨迹在目标时间的位移
        - 深度运动（motion in depth）：$\mathcal{M} = \frac{v_0 \Delta t + \Delta x}{v_1 \Delta t + \Delta x}$，通过轨迹梯度估计瞬时速度，推导出深度变化比
        - 多视角融合：$\mathcal{M}_k = \frac{1}{k} \sum_i \frac{t_k}{t_i}(\mathcal{M}_i - 1) + 1$，聚合多个时间观测提升鲁棒性
    - 设计动机：轨迹是运动的统一表示，光流是轨迹在特定时间的离散采样，深度运动是轨迹的微分性质——这种分析框架天然将 2D 和 3D 运动统一起来

### 损失函数 / 训练策略
- 多任务损失：$L = L_{\text{flow}} + L_{\text{depth}} + \lambda L_t$，$\lambda = 10^{-7}$
- 光流损失：指数加权 L1 损失 $L_{\text{flow}} = \sum_k \gamma^{N-k} (|\mathcal{O}_x^{(k)}|_1 + |\mathcal{O}_y^{(k)}|_1)$
- 深度运动损失：类似指数加权 L1 损失
- 时间梯度正则化：$L_t = \sum_i |\mathcal{T}'(t_{i+1}) - \mathcal{T}'(t_i)|_1$，防止高阶轨迹畸变
- AdamW 优化器 + OneCycle 学习率策略，训练 60,000 次迭代

## 实验关键数据

### 主实验
在 CarlaEvent3D 数据集上的验证结果（dense labels）：

| 模型 | Flow EPE↓ | Flow F1↓ | Mid log-mid↓ | 参数量(M) | 推理时间(s) |
|------|----------|---------|-------------|----------|-----------|
| E-RAFT | 2.781 | 24.604 | - | 5.04 | 0.049 |
| Expansion | 7.821 | 57.653 | 171.237 | 12.13 | 0.300 |
| ScaleFlow | 4.518 | 42.885 | 268.050 | 10.70 | 0.090 |
| Scale++ | 5.242 | 40.081 | 260.165 | 42.96 | 0.119 |
| EMoTive (Uniform) | 2.669 | 24.607 | 122.023 | 5.67 | 0.040 |
| **EMoTive** | **2.547** | **22.866** | **113.593** | 5.61 | 0.040 |

### 消融实验

| 配置 | Flow EPE↓ | Mid log-mid↓ | 说明 |
|------|----------|-------------|------|
| EMoTive (Uniform B-spline) | 2.669 | 122.023 | 均匀基线 |
| EMoTive (NURBS, 无密度自适应) | ~2.6 | ~118 | 非均匀但无事件引导 |
| EMoTive (完整) | **2.547** | **113.593** | 密度自适应NURBS |
| 无 Event Kymograph | ~3.0+ | ~140+ | 仅用 Voxel |
| 无多视角融合 | ~2.55 | ~130+ | 仅单时间深度估计 |

### 关键发现
- **事件引导的非均匀轨迹设计是关键**：相比均匀 B 样条，非均匀 NURBS 在光流 EPE 上提升 4.6%，在深度运动 log-mid 上提升 6.9%
- 依赖空间相关性的方法（Expansion、ScaleFlow 等）在事件数据上严重退化（EPE 增加 1.9-5.3），因为事件数据空间冗余性低
- EMoTive 仅 5.61M 参数（比 ScaleFlow 少 47.6%），推理速度 40ms/100ms 事件数据，比 ScaleFlow 快 52.9%
- 时间梯度正则化对防止螺旋伪影很重要，$\lambda = 10^{-7}$ 是最佳值

## 亮点与洞察
- **优雅的问题分解**：通过轨迹参数曲线统一光流和深度运动估计，物理意义清晰
- **Event Kymograph 是关键创新**：空间解耦+连续高斯时间核，保留了事件相机微秒级时间分辨率的优势
- **密度自适应机制自然匹配事件相机特性**：事件密度反映运动强度，用它引导曲线参数分配是物理一致的
- **自建数据集弥补领域空白**：CarlaEvent3D 提供了多动态场景、多天气条件下的完整 3D 运动标注

## 局限与展望
- CarlaEvent3D 是合成数据集，仿真-真实域差距可能影响模型在真实场景的表现
- NURBS 曲线的阶数和控制点数量是固定的，可能无法适配极端复杂运动
- 刚体恒速假设对非刚体或变速运动场景可能不成立
- 未与最新的帧-事件融合方法（BlinkVision 系列）进行充分对比
- 计算效率虽优于基线，但对实时应用的场景仍需进一步优化

## 相关工作与启发
- **轨迹参数化是运动估计的有力工具**：从 CamLiFlow 的点云轨迹到 EMoTive 的事件轨迹，参数曲线表示越来越受关注
- **事件相机的"时间显微镜"特性**尚未被充分利用，Kymograph 式的时间投影提供了新的表示方式
- **密度作为运动先验**是事件相机独有的信号，可以推广到其他事件驱动任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 事件引导的非均匀轨迹参数化是有意义的创新，Kymograph表示新颖
- 实验充分度: ⭐⭐⭐⭐ 自建数据集+真实基准，但对比方法多为改编而非原生事件方法
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式推导严谨，但部分符号较密
- 价值: ⭐⭐⭐⭐ 对事件驱动3D运动估计领域有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Simultaneous Motion And Noise Estimation with Event Cameras](simultaneous_motion_and_noise_estimation_with_event_cameras.md)
- [\[NeurIPS 2025\] EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes](../../NeurIPS2025/video_understanding/eag3r_event-augmented_3d_geometry_estimation_for_dynamic_and_extreme-lighting_sc.md)
- [\[ECCV 2024\] Motion-prior Contrast Maximization for Dense Continuous-Time Motion Estimation](../../ECCV2024/video_understanding/motion-prior_contrast_maximization_for_dense_continuous-time_motion_estimation.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)

</div>

<!-- RELATED:END -->
