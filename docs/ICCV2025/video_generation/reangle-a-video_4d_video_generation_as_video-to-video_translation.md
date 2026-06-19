---
title: >-
  [论文解读] Reangle-A-Video: 4D Video Generation as Video-to-Video Translation
description: >-
  [ICCV 2025][视频生成][多视角视频] Reangle-A-Video 将多视角视频生成重新定义为视频到视频翻译问题，通过自监督微调视频扩散模型学习视角不变运动，配合 DUSt3R 引导的多视角一致性 inpainting，从单目视频生成同步多视角视频。 从单个输入视频生成多视角同步视频是 4D 内容生成的核心需求…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "多视角视频"
  - "视频翻译"
  - "视角迁移"
  - "相机控制"
  - "扩散模型"
  - "LoRA"
  - "DUSt3R"
---

# Reangle-A-Video: 4D Video Generation as Video-to-Video Translation

**会议**: ICCV 2025  
**arXiv**: [2503.09151](https://arxiv.org/abs/2503.09151)  
**领域**: 视频生成·4D生成  
**关键词**: 多视角视频, 视频翻译, 视角迁移, 相机控制, 扩散模型, LoRA, DUSt3R  

## 一句话总结

Reangle-A-Video 将多视角视频生成重新定义为视频到视频翻译问题，通过自监督微调视频扩散模型学习视角不变运动，配合 DUSt3R 引导的多视角一致性 inpainting，从单目视频生成同步多视角视频。

## 研究背景与动机

从单个输入视频生成多视角同步视频是 4D 内容生成的核心需求。主流方法通过在大规模 4D 数据集上训练多视角视频扩散模型，但存在以下问题：

**数据匮乏**：高质量多视角动态视频数据极其稀缺，合成数据域差距大

**域限制**：训练在合成资产上的模型无法泛化到真实世界场景

**非视频输入**：现有方法多从文本/图像生成，而非从用户输入的视频出发

**封闭代码**：多数方法代码不公开

核心思想：**将视角变化分解为视角相关的外观（起始图像）和视角不变的运动（image-to-video）**，用现成的图像和视频扩散先验分别处理。

## 方法详解

### Stage I: 基于点的视频变形数据增强

给定输入视频 $\mathbf{x}^{1:N}$：
1. 使用 Depth Anything V2 估计每帧深度图 $\mathbf{D}^i$
2. 将 RGBD 图像提升为点云 $\mathcal{P}^i = \phi_{2\to3}([\mathbf{x}^i, \mathbf{D}^i], \mathbf{K}, \mathbf{P}^i_{\text{src}})$
3. 定义 $M$ 条目标相机外参轨迹 $\Phi_j = \{\mathbf{P}^1_j, ..., \mathbf{P}^N_j\}$
4. 重投影得到变形视频和可见性掩码：$(\hat{\mathbf{x}}^i_j, \mathbf{m}^i_j) = \phi_{3\to2}(\mathcal{P}^i, \mathbf{K}, \mathbf{P}^i_j)$

静态视角迁移：目标轨迹在所有帧保持恒定。动态相机控制：目标姿态逐帧递增变化。

### Stage II: 多视角运动学习

在 CogVideoX-5b（MM-DiT 架构）上用 LoRA（rank=128）微调 3D 全注意力层，仅优化约 2% 的参数。

关键设计——**掩码扩散损失**：

$$\mathbb{E}[\|\boldsymbol{\epsilon} \odot \mathbf{m}_{\text{down}}^{1:N} - \epsilon_\theta(\mathbf{z}_t^{1:N}, t, c) \odot \mathbf{m}_{\text{down}}^{1:N}\|_2^2]$$

仅在可见像素上计算损失，避免黑色区域破坏原始模型先验。变形视频和原始视频一起训练，使模型学习视角不变的场景运动。

**动态相机控制**需在文本提示中显式指定相机运动类型（如"horizontal orbit left"），因为所有变形视频共享同一起始帧。

### Stage III: 多视角一致性图像 Inpainting

对静态视角迁移，需要从目标视角的起始图像：

1. 变形第一帧到目标视角
2. 使用 FLUX + inpainting ControlNet 填充不可见区域
3. **随机控制引导**（核心）：每步生成 $S=25$ 个候选，用 DUSt3R 计算多视角一致性得分（DINO 特征相似度），选择最优路径继续去噪

这一推理时计算缩放策略确保了跨视角的一致性。

## 实验

### 定量对比

| 方法 | Subject↑ | Temporal↑ | Dynamic↑ | MEt3R↓ | FID↓ | FVD↓ |
|------|----------|-----------|----------|--------|------|------|
| **静态视角迁移** |
| GCD | 0.885 | 0.873 | 0.761 | 0.124 | 155.2 | 5264.7 |
| Vanilla CogVideoX | **0.945** | **0.974** | 0.729 | 0.054 | 79.6 | 3664.2 |
| **Reangle-A-Video** | 0.952 | 0.976 | **0.766** | **0.041** | **53.4** | **2690.9** |
| **动态相机控制** |
| NVS-Solver | 0.904 | 0.905 | **0.881** | 0.109 | 95.8 | 3516.5 |
| Trajectory Attn | 0.898 | 0.934 | 0.889 | 0.097 | 109.2 | 3624.9 |
| **Reangle-A-Video** | **0.914** | **0.939** | 0.888 | **0.065** | **74.2** | **3019.7** |

在 MEt3R（多视角一致性）、FID、FVD 上全面领先。

### 多视角 Inpainting 消融

| 配置 | MEt3R↓ | SED↓ | TSED↑ |
|------|--------|------|-------|
| 无随机控制引导 | 0.143 | 1.197 | 0.524 |
| **有随机控制引导** | **0.118** | **1.184** | **0.559** |

随机控制引导显著提升了多视角一致性。

### 数据增强消融

仅用原始视频微调 → 无法准确捕获运动（如犀牛在树前移动）；加入变形视频 → 运动忠实度大幅提升。用户研究进一步确认了这一效果。

## 亮点与洞察

1. **视频翻译范式**：将 4D 生成转化为视频到视频翻译，完全避免了昂贵的 4D 数据需求
2. **自监督训练**：仅需单个视频，通过深度变形创建多视角训练数据
3. **掩码扩散损失**的精妙设计：在可见像素上学习运动，在不可见区域保留模型先验
4. **推理时计算缩放**：用 DUSt3R 作为奖励函数，在 inpainting 中强制多视角一致性
5. 统一支持**静态视角迁移**和**动态相机控制**，6DoF 自由度

## 局限性

- 每个视频需要约 1 小时的微调（400 步 LoRA），非即时可用
- 深度估计误差会传播到变形视频中
- 对快速运动或大视角变化可能出现伪影
- 目前仅测试了 28 个视频，大规模效果未知

## 相关工作

- 多视角视频生成：CAT4D、Generative Camera Dolly
- 相机控制视频生成：CameraCtrl、MotionCtrl、Recapture
- 视频扩散模型：CogVideoX、Wan、Sora

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 5 |
| 技术深度 | 5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **综合** | **4.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](../../ICLR2026/video_generation/geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[CVPR 2025\] 4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion](../../CVPR2025/video_generation/4real-video_learning_generalizable_photo-realistic_4d_video_diffusion.md)
- [\[CVPR 2026\] WorldReel: 4D Video Generation with Consistent Geometry and Motion Modeling](../../CVPR2026/video_generation/worldreel_4d_video_generation_with_consistent_geometry_and_motion_modeling.md)
- [\[ICCV 2025\] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](recammaster_camera-controlled_generative_rendering_from_a_single_video.md)

</div>

<!-- RELATED:END -->
