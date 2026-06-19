---
title: >-
  [论文解读] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation
description: >-
  [ICCV 2025][人体理解][手物交互] 提出 ViTaM-D，一个视觉-触觉融合框架，通过新提出的分布式力感知接触表示（DF-Field）和两阶段流程（视觉动态跟踪+力感知优化），实现刚性和可变形物体的手物交互动态重建，并引入 HOT 数据集填补可变形物体手物交互的评测空白。 问题定义 手物交互重建旨在从视觉和/或触…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "手物交互"
  - "触觉感知"
  - "接触建模"
  - "可变形物体"
  - "SDF重建"
---

# Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation

**会议**: ICCV 2025  
**arXiv**: [2411.09572](https://arxiv.org/abs/2411.09572)  
**代码**: [sites.google.com/view/vitam-d](https://sites.google.com/view/vitam-d/)  
**领域**: 人体理解  
**关键词**: 手物交互, 触觉感知, 接触建模, 可变形物体, SDF重建

## 一句话总结

提出 ViTaM-D，一个视觉-触觉融合框架，通过新提出的分布式力感知接触表示（DF-Field）和两阶段流程（视觉动态跟踪+力感知优化），实现刚性和可变形物体的手物交互动态重建，并引入 HOT 数据集填补可变形物体手物交互的评测空白。

## 研究背景与动机

### 问题定义

手物交互重建旨在从视觉和/或触觉输入中恢复手和物体的完整几何与姿态，这对 VR/AR、机器人模仿学习等下游任务至关重要。核心挑战在于：接触区域被遮挡、物体在交互中发生形变，仅靠视觉难以获得完整信息。

### 已有方法的不足

**纯视觉方法**（gSDF、HOTrack）：依赖跨帧特征融合来弥补遮挡，但在接触区域仍然信息不足

**触觉融合方法**（ViTaM）：直接用 Transformer 融合触觉和视觉数据，忽略了力信号与点云特征之间的信息不对称

**接触建模方法**（CPF、TOCH）：使用经验性的弹簧-质量系统或优化方法，缺乏物理合理性

**数据集缺陷**：现有手物交互数据集（DexYCB、OakInk）仅覆盖刚性或铰接物体，缺少可变形物体的精确触觉数据

### 核心 idea

将触觉信息通过基于能量的接触表示（DF-Field）集成到优化管线中，而非直接与视觉特征拼接融合。先用视觉获得粗略重建，再用触觉信息进行力感知的手姿态精细优化。

## 方法详解

### 整体框架

ViTaM-D 包含两个阶段：
1. **视觉动态跟踪**：基于点云流预测和特征融合进行手物在线跟踪与重建
2. **力感知优化**：利用分布式触觉阵列读数和 DF-Field 表示精细化手姿态和接触状态

### 关键设计

#### 1. DF-Field（分布式力感知接触表示）

- **功能**：用能量函数建模手物接触状态
- **核心思路**：定义两种能量项：
    - **相对势能**：$E_{ij} = \kappa l_{ij}^2$，其中 $\kappa$ 与触觉读数相关，$l_{ij}$ 为手和物体顶点距离。若 $\kappa > 0$，表示接触，能量趋向 0
    - **屏障能量**：$B_{ij} = -e^{-\kappa}(l_{ij}-\hat{l})^2\log(l_{ij}/\hat{l})$，当距离小于阈值 $\hat{l}$ 时产生排斥力，防止穿透
    - 总能量 $E = \sum_i \sum_j (E_{ij} + B_{ij})$
- **力参数估计**：$\kappa_{ij} \sim \overline{\mathcal{M}^j} / l_{ij}$，其中 $\overline{\mathcal{M}^j}$ 为区域平均触觉读数
- **设计动机**：将手划分为 22 个区域，每个区域的力由对应触觉传感器提供，既保持物理合理性，又降低计算复杂度

#### 2. 视觉动态跟踪网络

- **功能**：从单视角深度图像序列中实时重建手物交互
- **核心思路**：
    - **流预测模块**：用 PointNet++ 提取当前帧和前一帧的逐点特征，预测点云流 $f_{t-1 \to t}$，通过 Transformer 融合静态特征和对应关系特征
    - **物体解码器**：将融合特征散射到体积中，用 3D-UNet 处理后通过 MLP 预测 SDF 值，Marching Cubes 生成网格
    - **手解码器**：基于 MANO 参数模型，通过投票机制预测关节位置，反向运动学估计姿态参数
    - **接触约束**：若采样点在接触区域（$c_x = 1$），强制其 SDF 值趋向 0：$\mathcal{L}_C = \sum_{x \in \mathcal{X}} s_x \cdot \mathbb{1}_{c_x=1}$

#### 3. 力感知手姿态优化

- **功能**：基于 DF-Field 精细化手姿态
- **核心思路**：以 DF-Field 总能量为目标函数，结合姿态合理性约束 $\mathcal{L}_r$ 和偏离惩罚 $\mathcal{L}_o$：
  $$\theta^* = \arg\min_\theta (E + \mathcal{L}_r + \mathcal{L}_o)$$
  使用 Adam 优化器迭代 100 步，每帧约 3.5 秒
- **设计动机**：视觉跟踪获得的手姿态在接触区域精度不足，需要触觉信息补充

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \lambda_f \mathcal{L}_{flow} + \lambda_S \mathcal{L}_{SDF} + \lambda_H \mathcal{L}_{Hand} + \lambda_C \mathcal{L}_C$

- $\lambda_f = 0.01$, $\lambda_S = 0.5$, $\lambda_H = 1$, $\lambda_C = 0.05$
- 流损失用 Chamfer Distance，SDF 用 L1，手关节用 L2
- 训练 100 epochs，学习率 1e-4，Nvidia A40 上约 15 小时

## 实验关键数据

### 主实验

**DexYCB 和 HOT 数据集定量结果**

| 方法 | 数据集 | IoU↑ | CD(mm)↓ | MPJPE(mm)↓ | PD(mm)↓ | CIoU↑ |
|------|--------|------|---------|------------|---------|-------|
| gSDF (RGB) | DexYCB | 86.8 | 13.4 | 14.4 | 8.9 | 31.3 |
| HOTrack | DexYCB | 88.2 | 10.2 | 25.7 | 12.3 | 28.5 |
| **Ours (w/o Force)** | DexYCB | **90.1** | **9.6** | **13.2** | 9.9 | 35.4 |
| ViTaM | HOT | 80.5 | 11.5 | 15.1 | 10.6 | 28.5 |
| **Ours (w/o Force)** | HOT | 81.0 | 10.9 | 13.6 | 10.7 | 29.8 |
| **Ours (w. Force Opt.)** | HOT | - | - | **11.3** | **7.3** | **40.3** |

### 消融实验

**接触约束方式消融（HOT 数据集）**

| 配置 | IoU↑ | CD↓ | CIoU↑ | MPJPE↓ |
|------|------|-----|-------|--------|
| 无接触约束 | 75.9 | 12.8 | 25.3 | 12.0 |
| GT接触 | 81.2 | 11.2 | 29.2 | 11.9 |
| 触觉读数 | 81.0 | 10.9 | 29.8 | 11.9 |
| PointNet预测 | 78.3 | 12.1 | 27.6 | 12.1 |

**力表示消融（HOT 数据集）**

| 配置 | MPJPE↓ | PD↓ | CIoU↑ |
|------|--------|-----|-------|
| 无力优化 | 13.6 | 10.7 | 29.8 |
| 固定力值 | 12.9 | 8.5 | 36.8 |
| **触觉力优化** | **11.3** | **7.3** | **40.3** |

### 关键发现

- 力感知优化将穿透深度从 10.7mm 降到 7.3mm，接触 IoU 从 29.8% 提升到 40.3%，效果显著
- 即使没有触觉数据，使用固定力值的 DF-Field 也能改善结果（DexYCB 上 CIoU: 35.4→39.7），说明能量表示设计本身有效
- 接触约束对物体重建有明显帮助（IoU: 75.9→81.0），触觉读数和 GT 接触状态效果接近
- 在仿真训练的模型可直接迁移到真实场景重建（毛绒玩具实验）

## 亮点与洞察

1. **触觉信息的正确集成方式**：不是简单地将力数据和视觉特征拼接，而是通过基于物理的能量函数在优化阶段使用，避免了信息不对称问题
2. **DF-Field 的灵活性**：有触觉数据时直接使用，无触觉数据时可设经验力值，兼容纯视觉管线
3. **HOT 数据集的价值**：基于 ZeMa 仿真器的 FEM 接触建模提供了无穿透的精确 GT，填补了可变形物体手物交互评测的空白
4. **两阶段设计的实用性**：视觉跟踪部分可实时运行，力优化部分可按需进行离线精化

## 局限与展望

1. **力优化每帧 3.5 秒**：远达不到实时，限制了在线应用
2. **触觉传感器的可获取性**：高密度分布式触觉手套（如 ViTaM）目前仍不普及
3. **HOT 数据集基于仿真**：与真实触觉数据的 domain gap 有待研究
4. **仅支持单手单物体**：未扩展到双手或多物体场景
5. **可变形物体的材料参数需预设**：杨氏模量、泊松比等参数在实际场景中难以获取

## 相关工作与启发

- gSDF 使用 Transformer + SDF 建模复杂手物交互，是 DexYCB 上的强基线
- CPF 采用弹簧-质量系统的经验性接触优化，DF-Field 在物理建模上更完善
- TOCH 在时空一致性优化上有贡献，但力感知优化效果更好（CIoU: 34.1 vs 40.3）
- ZeMa 仿真器提供了高精度 FEM 接触建模，是 HOT 数据集的基础

## 评分

- **新颖性**: ⭐⭐⭐⭐ — DF-Field 的能量建模方式新颖，将触觉融入优化而非特征拼接是正确方向
- **实验充分度**: ⭐⭐⭐⭐ — DexYCB 和 HOT 双数据集验证，消融实验设计合理
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，pipeline 描述完整，图示质量高
- **价值**: ⭐⭐⭐⭐ — HOT 数据集和 DF-Field 表示对手物交互社区有长期价值，但触觉设备门槛限制了实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Contact-Aware Refinement of Human Pose Pseudo-Ground Truth via Bioimpedance Sensing](contact-aware_refinement_of_human_pose_pseudo-ground_truth_via_bioimpedance_sens.md)
- [\[NeurIPS 2025\] Learning Dense Hand Contact Estimation from Imbalanced Data](../../NeurIPS2025/human_understanding/learning_dense_hand_contact_estimation_from_imbalanced_data.md)
- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](../../CVPR2025/human_understanding/unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[CVPR 2025\] WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](../../CVPR2025/human_understanding/wilor_end-to-end_3d_hand_localization_and_reconstruction_in-the-wild.md)
- [\[CVPR 2026\] IMU-HOI: A Symbiotic Framework for Coherent Human-Object Interaction and Motion Capture via Contact-Conscious Inertial Fusion](../../CVPR2026/human_understanding/imu-hoi_a_symbiotic_framework_for_coherent_human-object_interaction_and_motion_c.md)

</div>

<!-- RELATED:END -->
