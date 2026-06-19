---
title: >-
  [论文解读] Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction
description: >-
  [ICCV 2025][3D视觉][动态3D重建] 提出 Dynamic Point Maps (DPM)，将 DUSt3R 的视点不变点图扩展为同时控制视点和时间的时空不变表示，仅通过预测4组点图即可在前馈方式下同时解决深度估计、场景流、运动分割和3D目标跟踪等多种4D任务。 DUSt3R 的突破在于引入了「视点不变点图」…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "动态3D重建"
  - "点图表示"
  - "场景流"
  - "运动分割"
  - "DUSt3R"
---

# Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2503.16318](https://arxiv.org/abs/2503.16318)  
**代码**: [项目页面](https://www.robots.ox.ac.uk/~vgg/research/dynamic-point-maps/)  
**领域**: 3D视觉  
**关键词**: 动态3D重建, 点图表示, 场景流, 运动分割, DUSt3R

## 一句话总结

提出 Dynamic Point Maps (DPM)，将 DUSt3R 的视点不变点图扩展为同时控制视点和时间的时空不变表示，仅通过预测4组点图即可在前馈方式下同时解决深度估计、场景流、运动分割和3D目标跟踪等多种4D任务。

## 研究背景与动机

DUSt3R 的突破在于引入了「视点不变点图」（viewpoint-invariant point maps）概念：给定两张图像，将每个像素映射到统一参考系中的3D点。这一优雅的表示使得相机内外参估计、3D重建、2D匹配等多种任务都可以化约为点图预测。

**但DUSt3R无法处理动态场景。** 当场景中存在运动物体时，即使固定视点参考系，同一物理点在不同时间的3D位置不同，破坏了视点不变性：$P_1(\pi_1)(\boldsymbol{u}_1) \neq P_2(\pi_1)(\boldsymbol{u}_2)$。

**MonST3R 的尝试与局限**：MonST3R 直接将 DUSt3R 应用于动态场景，但由于缺乏时间不变性，它无法直接预测对应的3D点。不得不借助光流网络来建立时间对应，这不仅增加了系统复杂度，还限制在可见像素范围内，对遮挡和去遮挡处理不当。

作者的核心洞察：**动态场景中的不变性需要同时固定视点和时间。** 为每张图像预测两组点图——分别对应两个时间戳的3D位置——就能重建时空不变性，同时保留运动信息。这是能够解决所有4D任务的最小设计（minimal design）。

具体来说：
- $P_1(t_1, \pi_1)$, $P_1(t_2, \pi_1)$：图像1中像素在时间1和时间2的3D位置
- $P_2(t_1, \pi_1)$, $P_2(t_2, \pi_1)$：图像2中像素在时间1和时间2的3D位置
- 同一时间戳下的点图恢复不变性：$P_1(t_1, \pi_1)(\boldsymbol{u}_1) = P_2(t_1, \pi_1)(\boldsymbol{u}_2)$
- 不同时间戳的差直接给出场景流：$P_1(t_2, \pi_1) - P_1(t_1, \pi_1)$

## 方法详解

### 整体框架

在 DUSt3R 的 ViT 编码器-解码器基础上，为每张图像增加一个预测头，总共4个头分别预测4组点图 $P_i(t_j, \pi_1)$，$i,j \in \{1,2\}$。每组点图包含3通道坐标和1通道置信度图。所有点图均在第一张图像的参考系 $\pi_1$ 下表示。

### 关键设计

1. **Dynamic Point Maps 表示**:

    - 功能：为动态场景定义一种最小且完备的点图表示
    - 核心思路：每张图像映射到两组点图（对应两个时间戳），共4组。时间不变的点图对可以直接建立跨视角对应（$P_1(t_1, \pi_1)$ 与 $P_2(t_1, \pi_1)$ 比较得到点匹配），时间差异的点图对可以直接给出场景流（$P_i(t_2, \pi_1) - P_i(t_1, \pi_1)$），静态部分两个时间戳的点图一致即产生运动分割。
    - 设计动机：这是 DUSt3R 到4D场景的自然且最小扩展。MonST3R 实际上只预测了4组中的2组（$P_1(t_1, \pi_1)$ 和 $P_2(t_2, \pi_1)$），缺少跨时间的点图因而无法直接推导场景流和运动匹配。DPM 补全了这个设计空间。

2. **网络架构扩展**:

    - 功能：在 DUSt3R Backbone 上增加2个额外预测头
    - 核心思路：$\{P_i(t_j, \pi_1)\}_{i,j \in \{1,2\}} = \Phi(I_1, I_2)$。4个头共享同一个 Transformer 编码器-解码器 backbone，每个头预测 $(P_i(t_j, \pi_1), C_i(t_j, \pi_1))$（点图+置信度）。新增的2个头用 DUSt3R 原有头的权重初始化，在训练初期近似静态重建。
    - 设计动机：最小化架构修改，充分利用 DUSt3R 的预训练知识。新增2个头但共享backbone，增加的参数量很小。

3. **混合数据训练**:

    - 功能：在合成+真实数据混合上训练 DPM 预测器
    - 核心思路：使用 Kubric 管线生成合成数据集 MOVi-G（含复杂相机轨迹和动态物体），提供完整的4组点图 GT。加入 Waymo 真实数据（利用 LiDAR 生成动态点图 GT）。对于无动态GT的数据集，省略跨时间点图的监督或视为静态场景。总共7个数据集混合训练。
    - 设计动机：完整的动态GT只有合成数据和LiDAR数据能提供，但混合真实视频数据（即使只有静态监督）有助于泛化。

### 损失函数 / 训练策略

使用置信度校准的回归损失：
$$L_{\text{conf}}(\hat{P}, P) = \frac{1}{HW} \sum_{i=1}^{HW} C_i L_{\text{reg}}(\hat{P}, P, i) - \alpha \log C_i$$

其中 $L_{\text{reg}}$ 是尺度归一化的逐像素回归损失，允许预测到任意尺度因子内。4组点图堆叠后统一优化。训练分辨率 $(512, 288)$ 和 $(512, 336)$。

## 实验关键数据

### 主实验

**深度估计（2-View, Abs Rel↓）**：

| 数据集 | DPM | MonST3R | 提升 |
|--------|-----|---------|------|
| Sintel | **0.321** | 0.347 | 7.5% |
| Point Odyssey | **0.059** | 0.065 | 9.2% |
| Kubric | **0.078** | 0.166 | 53% |
| KITTI (crop) | **0.052** | 0.069 | 24.6% |

**动态重建（相对点云误差 $L_{\text{rel}}$↓）**：

| 数据集 | 点图 | DPM | MonST3R |
|--------|------|-----|---------|
| Kubric-G | $P_1(t_1)$ | **0.057** | 0.163 |
| Kubric-G | $P_2(t_1)$ (跨时间) | **0.071** | 0.265 |
| Kubric-G | $P_1(t_2)$ (跨时间) | **0.079** | 0.346 |
| Waymo | $P_1(t_1)$ | **0.068** | 0.197 |

**场景流 3D EPE↓**：

| 数据集 | DPM | MonST3R | RAFT-3D (需深度GT) |
|--------|-----|---------|-------------------|
| Kubric-G Forward | **0.104** | 0.334 | 4.067 |
| Waymo Forward | **0.051** | 0.161 | 0.150 |
| Waymo Backward | **0.053** | 0.135 | 0.145 |

### 消融实验

| 配置 | Kubric $L_{\text{rel}}$ $P_1(t_1)$↓ | Kubric $L_{\text{rel}}$ $P_1(t_2)$↓ | 说明 |
|------|-------------------------------------|-------------------------------------|------|
| MonST3R (仅2组点图) | 0.163 | 0.346 | 跨时间预测退化严重 |
| DPM (4组点图) | **0.057** | **0.079** | 跨时间预测稳定 |
| 目标跟踪 RPE rot↓ | DPM: 33.7° | MonST3R: 56.1° | 旋转误差降低40% |

### 关键发现

1. DPM 在跨时间预测上远超 MonST3R——Kubric-G 上 $P_2(t_1)$ 误差降低 73%（0.265→0.071），证明显式建模时间维度的必要性
2. 仅用 RGB 输入的 DPM 在 Waymo 场景流上超越需要深度GT的 RAFT-3D（0.051 vs 0.150）
3. MonST3R 在 Kubric-G 上场景流 EPE 高达 4.067（RAFT-3D），说明复杂相机运动下2D光流warp方法失效
4. DPM 不需要额外光流模型，比 MonST3R 的流程更简洁高效
5. 在目标跟踪中旋转误差降低 40%，证明显式的时间不变点图对刚体运动估计的价值

## 亮点与洞察

1. **概念层面的贡献**：DPM 是一个通用表示概念，不仅是一个具体方法。它明确了动态场景不变性需要「视点+时间」双重控制，这个设计空间分析为后续工作提供了理论基础
2. **最小设计原则**：4组点图是解决所有4D任务的最小充分集——多种任务（深度、场景流、匹配、分割、跟踪）都化约为简单的点比较或差分运算
3. **架构修改极小**：仅增加2个头，充分复用 DUSt3R 预训练，说明好的表示设计比复杂的架构改进更重要
4. 消除了对光流网络的依赖，使4D重建流程更加统一和简洁

## 局限与展望

1. 合成训练数据的依赖：MOVi-G 的物体运动模式有限，可能影响复杂真实场景的泛化
2. 成对处理模式：DPM 一次只处理2帧，长序列视频需要 bundle adjustment
3. 大物体运动时旋转误差仍然偏高（33.7°），说明跨时间的3D理解仍具挑战
4. 目前仅支持 $T=2$ 个时间戳，扩展到多时间戳将更加通用
5. 遮挡/去遮挡区域的处理仍有改进空间

## 相关工作与启发

- **DUSt3R / MASt3R**：DPM 的直接基础，证明了点图表示的强大通用性 → DPM 将其扩展到4D
- **MonST3R**：将 DUSt3R 应用于动态场景但不完整 → DPM 补全了表示空间
- **CUT3R / Stereo4D**：同期工作，各有侧重但未探索完整的不变性设计空间
- **Shape of Motion**：通过拟合3D高斯轨迹重建动态场景，但需要昂贵的测试时优化 → DPM 是纯前馈方案
- 启发：好的表示设计（而非更大的模型）是3D/4D视觉的关键杠杆

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ DPM 概念优雅，最小设计+多任务解决的思路极具启发性
- **实验充分度**: ⭐⭐⭐⭐ 覆盖深度/场景流/目标跟踪多任务，但真实数据评估可更全面
- **写作质量**: ⭐⭐⭐⭐⭐ 从不变性分析自然推导出设计，逻辑清晰
- **价值**: ⭐⭐⭐⭐⭐ 为动态3D视觉提供了统一的表示基础，影响力可能很大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] V-DPM: 4D Video Reconstruction with Dynamic Point Maps](../../CVPR2026/3d_vision/v-dpm_4d_video_reconstruction_with_dynamic_point_maps.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction](back_on_track_bundle_adjustment_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
