---
title: >-
  [论文解读] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation
description: >-
  [ICCV 2025][3D视觉][自中心视角] 首次研究HMD后置相机对自中心3D全身姿态估计的价值，提出基于Transformer的多视图热图细化方法，结合不确定性感知掩码机制，在新建的Ego4View数据集上实现10% MPJPE提升。 自中心3D全身姿态估计通常使用安装在HMD（头戴式设备）前方的相机…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "自中心视角"
  - "3D人体姿态估计"
  - "后置相机"
  - "多视图融合"
  - "头戴式设备"
---

# Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation

**会议**: ICCV 2025  
**arXiv**: [2503.11652](https://arxiv.org/abs/2503.11652)  
**代码**: [https://4dqv.mpi-inf.mpg.de/EgoRear/](https://4dqv.mpi-inf.mpg.de/EgoRear/)  
**领域**: 3D Vision / Human Pose Estimation  
**关键词**: 自中心视角, 3D人体姿态估计, 后置相机, 多视图融合, 头戴式设备

## 一句话总结

首次研究HMD后置相机对自中心3D全身姿态估计的价值，提出基于Transformer的多视图热图细化方法，结合不确定性感知掩码机制，在新建的Ego4View数据集上实现>10% MPJPE提升。

## 研究背景与动机

自中心3D全身姿态估计通常使用安装在HMD（头戴式设备）前方的相机，但这种设计存在根本限制：

**自遮挡严重**：当用户抬头（运动中常见）时，前置相机几乎看不到身体，即使SOTA方法EgoPoseFormer也会失败

**视野有限**：身体后方完全不被捕获，尽管这些区域包含关键的3D重建线索

**现有HMD设计的局限**：Apple Vision Pro有8个前置传感器但不提供全身追踪，可能正是因为仅前置输入的精度不足

一个直观的解决方案是在HMD后方安装相机，但作者发现**简单地将后视图添加到现有方法的输入中并不总是有效，甚至可能降低精度**。根本原因是：现有方法依赖独立的2D关节检测器，没有有效的多视图集成机制——后视图中的自遮挡和缺失身体部位导致不准确的2D关节检测，进而影响3D估计。

## 方法详解

### 整体框架

整体流程：四视图鱼眼图像 → 2D关节热图估计 → 多视图热图细化模块 → 细化后的热图+特征 → 2D到3D提升模块 → 3D姿态。核心贡献是中间的多视图热图细化模块，可即插即用地集成到现有方法（EgoPoseFormer、EgoTAP）中。

### 关键设计

1. **2D关节热图细化模块**: 基于Transformer解码器架构，利用多视图上下文细化初始热图估计。核心假设是**前后视图可互补**——因为人体对称性，不可靠的后视图热图可由可靠的前视图热图改善，反之亦然。

    - 为每个视图定义视图专属关节查询 $\mathbf{Q}_{\text{front\_left}} \in \mathbb{R}^{15 \times 256}$，编码特定视图的2D骨架信息
    - 从初始热图提取2D关节位置作为锚点，使用可变形注意力（Deformable Attention）让关节查询与所有视图的热图特征在锚点附近交互：$\hat{\mathbf{Q}}^k = \text{DeformAttn}(\mathbf{Q}, \mathbf{T}_k, \mathbf{F}_k)$
    - 拼接所有视图的更新查询，经全连接层和自注意力得到多视图感知的关节查询
    - 通过偏移回归网络生成偏移特征，与初始热图特征相加得到细化特征

2. **初始热图状态传播**: 直接将视图查询用于注意力是次优的，因为缺少当前视图初始热图预测的上下文。解决方案：

    - 将初始热图 $\hat{\mathbf{H}}$ 通过MLP投影为热图嵌入 $\mathbf{E}$
    - 将编码器骨干的RGB特征 $\mathbf{B}$ 通过MLP投影为RGB嵌入 $\mathbf{G}$
    - 三者相加后通过查询投影层：$\mathbf{Q'} = \mathcal{P}_Q(\mathbf{Q} + \mathbf{E} + \mathbf{G})$

3. **不确定性感知掩码机制**: 自中心图像中自遮挡频繁导致初始热图可信度参差不齐。通过热图值对锚点的置信度进行评估，构建二值掩码：

    - 若热图值 $\geq 0.5$，掩码为1（可信），否则为0
    - 将掩码作为逐元素乘子应用于更新的查询：$\hat{\mathbf{Q'}}^k = \hat{\mathbf{Q}}^k \times \mathbf{M}^k$
    - 使后续自注意力更关注高置信度的热图特征
    - 细化模块使用MSE损失监督

### 损失函数 / 训练策略

训练分两阶段：
1. 分别训练2D关节热图估计器和细化模块各12个epoch（AdamW，初始lr=10⁻³）
2. 联合训练完整架构（包含3D模块）12个epoch
- 批大小：2D热图估计64，3D姿态估计32
- 学习率在第8和第10个epoch×0.1衰减
- 输入分辨率256×256，热图分辨率64×64

## 实验关键数据

### 主实验

不同相机配置下的3D姿态估计（MPJPE，mm）：

| 设置 | 方法 | Ego4View-Syn | Ego4View-RW |
|------|------|-------------|-------------|
| 2前视图 | EgoPoseFormer | 27.36 | 77.95 |
| 2前视图 | EgoPoseFormer + Ours | 27.04 | 76.35 |
| 2前+2后 | EgoPoseFormer | 20.20 | 63.38 |
| 2前+2后 | **EgoPoseFormer + Ours** | **19.25** | **56.94** |
| 2前视图 | EgoTAP | 32.56 | 91.23 |
| 2前+2后 | EgoTAP | 23.88 | 69.78 |
| 2前+2后 | **EgoTAP + Ours** | **22.57** | **62.11** |

在Ego4View-RW上，完整方法比仅前视图的EgoPoseFormer提升>10%（63.38→56.94 MPJPE）。

### 消融实验

逐关节评估（MPJPE，mm，2前+2后视图，Ego4View-RW）：

| 关节 | head | neck | arms | forearms | hands | legs | feet | toes | 全身 |
|------|------|------|------|----------|-------|------|------|------|------|
| EgoPoseFormer | 11.80 | 16.36 | 21.55 | 34.30 | 60.35 | 85.88 | 115.40 | 129.56 | 63.38 |
| + Ours | **11.49** | **15.89** | **21.27** | **30.90** | **48.17** | **79.67** | **103.46** | **116.04** | **56.94** |

手部改善最大（60.35→48.17，-20.2%），上肢和下肢均有显著提升。

相机数量消融（Ego4View-RW，EgoPoseFormer+Ours）：
- 2前视图：76.35mm
- 2前+1后左：60.96mm（-20.2%）
- 2前+1后右：60.17mm（-21.2%）
- 2前+2后：**56.94mm**（-25.4%）

### 关键发现

- 后置相机对全身追踪价值巨大：仅添加1个后置相机即可获得~20%的MPJPE改善
- 简单拼接后视图到现有方法有时会降低精度，因为后视图中自遮挡导致错误的2D检测会传播到3D估计
- 不确定性感知掩码对于处理后视图不可靠检测至关重要
- 后视图的手部可见率仅8-27%（远低于前视图的47-66%），但通过多视图融合仍能显著改善手部估计
- 前后37cm距离是可见性和外形因素的最佳平衡点

## 亮点与洞察

- **开创性研究方向**：首次质疑"HMD全身追踪只需前置相机"的假设，为HMD硬件设计提供新视角
- **实际问题驱动**：Apple Vision Pro有8个前置传感器但仍无全身追踪，佐证了前置相机的根本局限
- **方法设计简洁有效**：热图细化模块是轻量级即插即用组件，可集成到任何现有框架
- **数据集贡献重大**：Ego4View-Syn/RW是首个包含后置相机的大规模自中心数据集
- **实验设计周到**：包含宽松衣物（长裙、和服等），比现有数据集更具挑战性

## 局限与展望

- HMD原型体积较大（头盔+外置相机），距产品化还有距离
- 后置相机增加了硬件成本和重量，需评估实际部署的可行性
- 鱼眼图像的严重畸变使得极线几何等传统立体方法难以应用
- 未探索时序信息（视频级别）对后视图融合的帮助
- 2D检测器的预训练数据不包含后视图，可能存在域差距
- 可探索IMU等其他传感器与后置相机的融合方案

## 相关工作与启发

- EgoPoseFormer使用可变形注意力直接更新3D姿态，本文则在更早的2D热图阶段进行多视图融合
- 与体穿戴IMU方案（如Meta Quest）互补：后置相机无需额外穿戴设备
- 不确定性感知的设计思路可泛化到其他多视图融合问题
- 后置相机的价值不限于姿态估计——还可用于虚拟分身重建、环境感知、碰撞防护等

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次提出并验证HMD后置相机的价值，开辟全新研究方向
- **实验充分度**: ⭐⭐⭐⭐ 合成+真实数据集，多种配置消融，逐关节分析，但缺少与IMU方案的对比
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，实验详尽，但部分符号较繁琐
- **价值**: ⭐⭐⭐⭐⭐ 对HMD硬件设计和自中心感知社区有重要启发，数据集开源价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation](perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)
- [\[ICCV 2025\] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras](single-scanline_relative_pose_estimation_for_rolling_shutter_cameras.md)
- [\[CVPR 2026\] Egocentric Visibility-Aware Human Pose Estimation](../../CVPR2026/3d_vision/egocentric_visibility-aware_human_pose_estimation.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[CVPR 2025\] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](../../CVPR2025/3d_vision/egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)

</div>

<!-- RELATED:END -->
