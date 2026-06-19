---
title: >-
  [论文解读] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds
description: >-
  [CVPR 2025][3D视觉][4D reconstruction] 提出4D Motion Scaffold (MoSca)表示，通过稀疏6-DoF轨迹图紧凑编码场景运动，结合2D基础模型先验和物理正则化，从无位姿的随手拍单目视频实现全自动4D场景重建。 领域现状： 动态场景新视角合成是构建AGI数据集、空间计算内容创…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "4D reconstruction"
  - "dynamic scene"
  - "Gaussian splatting"
  - "motion scaffold"
  - "deformation graph"
  - "pose-free"
---

# MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds

**会议**: CVPR 2025  
**arXiv**: [2405.17421](https://arxiv.org/abs/2405.17421)  
**代码**: [https://www.cis.upenn.edu/~leijh/projects/mosca](https://www.cis.upenn.edu/~leijh/projects/mosca)  
**领域**: 3D视觉  
**关键词**: 4D reconstruction, dynamic scene, Gaussian splatting, motion scaffold, deformation graph, pose-free

## 一句话总结

提出4D Motion Scaffold (MoSca)表示，通过稀疏6-DoF轨迹图紧凑编码场景运动，结合2D基础模型先验和物理正则化，从无位姿的随手拍单目视频实现全自动4D场景重建。

## 研究背景与动机

**领域现状**: 动态场景新视角合成是构建AGI数据集、空间计算内容创作和具身智能的关键能力。从单目随手拍视频（最常见的数据格式）进行鲁棒4D重建极具挑战，因为多视角立体线索极为有限。

**现有方案的不足**:
1. **per-frame方法**信息量不足：局部深度warp方法在大偏角测试视角下直接失败，出现大面积缺失区域
2. **局部时序融合方法**（如PGDVS、Gaussian Marbles）仅融合小时间窗口，遮挡区域无法补全
3. **密集Gaussian方法**（如4D-GS）依赖强多视角立体线索，在单目随手拍场景中失效
4. **形变表示过于自由**: 大多数方法用MLP学习形变场，解空间过大，优化容易退化
5. **依赖外部位姿估计**: 多数方法需要COLMAP预估相机位姿，而COLMAP在动态场景中常失败

**核心动机**: 利用2D基础模型强先验 + 物理启发的低秩运动表示，将全时序观测全局融合，构建从无位姿RGB视频到可渲染4D场景的全自动系统。

## 方法详解

### 整体框架

四步系统流水线：
1. **2D基础模型推断**: 获取深度估计、长期2D轨迹、前/背景分离
2. **相机初始化**: 基于静态tracklet的bundle adjustment求解焦距和位姿
3. **MoSca几何优化**: 将2D先验提升到3D，用ARAP正则化优化运动图
4. **光度优化**: 将全时步Gaussian融合到查询时刻，通过Gaussian Splatting渲染优化

### 关键设计

#### 模块一：Motion Scaffold (MoSca) 形变表示

核心创新——用稀疏结构化轨迹图编码运动：
- **图节点** $v^{(m)}$: 每个节点是一条6-DoF轨迹 $[Q_1^{(m)}, ..., Q_T^{(m)}]$，加控制半径 $r^{(m)}$
- **图拓扑**: 使用curve distance（轨迹最大时空距离）构建KNN图，自然处理拓扑变化（如开门不连接门和墙）
- **形变插值**: Dual Quaternion Blending (DQB) 在SE(3)流形上插值多个刚体变换，避免线性蒙皮的伪影
- **权重计算**: RBF核函数 $w_i(x,t) = \exp(-\|x - t_{t}^{(i)}\|^2 / 2r^{(i)})$

MoSca节点数M远少于点数N（如Tab.7），利用了真实运动低秩光滑的物理先验。

#### 模块二：2D基础模型先验融合与相机求解

- **深度**: 使用预训练单目深度估计（如Metric3D、DepthAnything）
- **长期轨迹**: 使用BootsTAPIR/CoTracker获取稠密2D像素轨迹
- **动静分割**: 通过RAFT光流计算的极线误差图分离前/背景
- **相机BA**: 筛选极线误差低的静态轨迹，联合优化相机位姿和焦距，包含反投影误差$\mathcal{L}_{proj}$和scale-invariant深度对齐损失$\mathcal{L}_z$

#### 模块三：全局Gaussian融合与渲染

- **全时步融合**: 从所有时步back-project的depth点初始化Gaussian，通过MoSca形变场变换到查询时刻后融合
- **可学习蒙皮修正**: 每个Gaussian学习额外skinning weight修正 $\Delta w_j$
- **Node Control**: 类似3DGS的densification/pruning策略——高tracking-loss梯度区域增加节点，低贡献节点剪枝

### 损失函数

**Bundle Adjustment**: $\mathcal{L}_{BA} = \lambda_{proj}\mathcal{L}_{proj} + \lambda_z\mathcal{L}_z$

**几何优化**: $\mathcal{L}_{geo} = \lambda_{arap}\mathcal{L}_{arap} + \lambda_{acc}\mathcal{L}_{acc} + \lambda_{vel}\mathcal{L}_{vel}$
- ARAP损失：保持邻居间局部距离和局部坐标不变
- 速度/加速度正则化：确保时序平滑

**光度优化**: $\mathcal{L} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{dep}\mathcal{L}_{dep} + \lambda_{track}\mathcal{L}_{track} + \lambda_{arap}\mathcal{L}_{arap} + \lambda_{acc}\mathcal{L}_{acc} + \lambda_{vel}\mathcal{L}_{vel}$

其中 $\mathcal{L}_{track}$ 通过渲染XYZ坐标图监督2D轨迹一致性。

## 实验关键数据

### 主实验表

**DyCheck数据集（最具挑战，7个场景平均）**:

| 方法 | 位姿 | mPSNR↑ | mSSIM↑ | mLPIPS↓ |
|------|------|--------|--------|---------|
| HyperNeRF | 已知 | 16.81 | 0.569 | 0.332 |
| Shape-of-Motion | 已知 | 17.32 | 0.598 | 0.296 |
| **MoSca** | **已知** | **19.32** | **0.706** | **0.264** |
| RobustDynRF | 未知 | 17.10 | 0.534 | 0.517 |
| **MoSca** | **未知** | **18.84** | **0.676** | **0.289** |
| **MoSca (w/ focal)** | **未知** | **19.02** | **0.683** | **0.279** |

**NVIDIA数据集**: PSNR 26.72, LPIPS 0.070，超越所有对比方法。

**相机位姿精度**: Sintel ATE 0.090（超越DROID-SLAM、MonST3R），TUM-dynamics ATE 0.031（SOTA）。

### 消融表

| 组件 | mPSNR | mSSIM | mLPIPS |
|------|-------|-------|--------|
| Full model | 19.32 | 0.706 | 0.264 |
| No geometric optimization | 18.85 | 0.693 | 0.287 |
| No multi-level topology | 19.14 | 0.701 | 0.270 |
| No dual quaternion blending | 19.18 | 0.701 | 0.276 |
| Only fuse 4 neighboring frames | 16.96 | 0.663 | 0.344 |
| Only fuse 8 neighboring frames | 17.26 | 0.664 | 0.346 |

### 关键发现

1. **全局融合至关重要**: 仅融合4帧邻域 vs 全时序，mPSNR差距2.36 dB，验证全局聚合的核心价值
2. **几何优化阶段**贡献显著（+0.47 dB），ARAP先验有效传播运动信息到遮挡区域
3. **DQB优于线性蒙皮**: 在SE(3)流形上插值避免了线性混合的退化
4. **对应关系追踪**: 重建后的MoSca追踪精度（PCK-T 0.824）超越原始BootsTAPIR（0.779），说明优化改善了初始先验
5. **无位姿设置**仅损失约0.5 dB，系统对未知相机参数具有鲁棒性

## 亮点与洞察

1. **物理先验 + 学习先验的优雅结合**: MoSca的ARAP正则化编码"刚性为主"的运动先验，2D基础模型提供初始化——两者互补
2. **全时序全局融合**: 不同于per-frame或滑窗方法，真正实现跨所有帧的信息聚合，单帧遮挡区域可从其他帧补全
3. **系统完整性**: 从原始RGB视频到可渲染4D场景的全自动流水线，不需COLMAP或任何外部工具
4. **In-the-wild泛化**: 在电影片段、网络视频、SORA生成视频上均可工作

## 局限性

1. MoSca节点初始化依赖2D trackers的质量，严重遮挡或快速运动可能导致tracklet断裂
2. 静态/动态分割基于极线误差阈值，对相机运动极小的场景（如监控视频）可能失效
3. 运行时间未报告，多步pipeline可能较慢
4. 未处理动态光照变化和反射表面等复杂光学现象

## 相关工作与启发

- **Gaussian Marbles**: 类似思路但使用无结构的per-Gaussian运动，只做局部融合→MoSca用结构图实现全局融合
- **Shape-of-Motion**: 同期重要工作，MoSca在DyCheck上显著领先（+2.0 dB）
- **Embedded Deformation Graph**: 经典形变图方法，MoSca的核心创新在于将其与2D基础模型先验和Gaussian Splatting融合
- **启发**: 此方向可进一步探索将语言先验引入动态场景重建（如结合SAM分割实现语义4D重建）

## 评分

⭐⭐⭐⭐⭐ — 方法设计优雅（物理先验+基础模型），系统完备度极高（pose-free全自动），实验结果领先幅度大（mPSNR +2.0 dB），in-the-wild泛化性强，是动态场景重建领域的重要突破。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](../../NeurIPS2025/3d_vision/orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[CVPR 2025\] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)
- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)

</div>

<!-- RELATED:END -->
